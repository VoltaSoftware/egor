use egor_render::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Extent3d, Renderer, Texture, TextureFormat,
    batch::GeometryBatch,
    target::{OffscreenTarget, RenderTarget},
};
use glam::Vec2;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::primitives::PathBuilder;
use crate::{
    camera::Camera,
    color::Color,
    primitives::{PolygonBuilder, PolylineBuilder, PrimitiveBatch, RectangleBuilder},
    text::{TextBuilder, TextRenderer},
};

// ---------------------------------------------------------------------------
// Render Target Store — persistent across frames, owns OffscreenTargets
// ---------------------------------------------------------------------------

/// Persistent storage for offscreen render targets created by game code.
///
/// Stored in [`App`] and passed to [`Graphics`] each frame by reference.
/// Render targets survive across frames; the game is responsible for
/// recreating them when their size changes.
pub struct RenderTargetStore {
    targets: Vec<OffscreenTarget>,
}

impl RenderTargetStore {
    pub fn new() -> Self {
        Self { targets: Vec::new() }
    }

    /// Create an offscreen render target and return its index.
    pub fn create(&mut self, device: &Device, width: u32, height: u32, format: TextureFormat) -> usize {
        let id = self.targets.len();
        self.targets.push(OffscreenTarget::new(device, width, height, format));
        id
    }

    /// Resize an existing offscreen target. If its dimensions already match, this is a no-op.
    pub fn resize(&mut self, device: &Device, id: usize, width: u32, height: u32) {
        self.targets[id].resize(device, width, height);
    }

    pub fn get(&self, id: usize) -> &OffscreenTarget {
        &self.targets[id]
    }

    pub fn get_mut(&mut self, id: usize) -> &mut OffscreenTarget {
        &mut self.targets[id]
    }

    pub fn len(&self) -> usize {
        self.targets.len()
    }
}

impl Default for RenderTargetStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Screen Capture State — wgpu backend for anti-cheat GPU readback
// ---------------------------------------------------------------------------

const CAP_HEIGHT: u32 = 360;
const SLOT_COUNT: usize = 3;

const MAP_PENDING: u8 = 0;
const MAP_READY: u8 = 1;
const MAP_FAILED: u8 = 2;

/// WGSL shader for fullscreen-triangle blit with bilinear sampling.
const BLIT_SHADER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    // Fullscreen triangle: 3 vertices cover the entire clip space.
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    // Flip Y so texture top-left maps to NDC top-left.
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_src, s_src, in.uv);
}
"#;

/// WGSL shader for fullscreen-triangle blit that outputs BT.601 luminance
/// to a single-channel R8Unorm render target. Eliminates CPU-side grayscale
/// conversion and reduces GPU→CPU readback bandwidth by 4×.
const BLIT_GRAY_SHADER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = textureSample(t_src, s_src, in.uv);
    let lum = dot(c.rgb, vec3<f32>(0.299, 0.587, 0.114));
    return vec4<f32>(lum, 0.0, 0.0, 1.0);
}
"#;

// -- Unsafe pixel-format conversion (matches old OpenGL PBO path perf) ------

/// BGRA → RGB565 with row-pitch padding.
///
/// # Safety
/// `src` must point to at least `h * row_pitch` readable bytes.
/// `dst` must point to at least `w * h * 2` writable bytes.
#[inline]
unsafe fn pack_bgra_to_rgb565(src: *const u8, dst: *mut u8, w: usize, h: usize, row_pitch: usize) {
    let d = dst as *mut u16;
    for y in 0..h {
        let row = unsafe { src.add(y * row_pitch) };
        let dst_off = y * w;
        for x in 0..w {
            let s = unsafe { row.add(x * 4) };
            let b = unsafe { *s } as u16;
            let g = unsafe { *s.add(1) } as u16;
            let r = unsafe { *s.add(2) } as u16;
            unsafe { *d.add(dst_off + x) = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3) };
        }
    }
}

// -- Ring-buffer staging slot -----------------------------------------------

struct StagingSlot {
    buffer: Option<Buffer>,
    buf_size: u64,
    row_pitch: u32,
    cap_w: u32,
    cap_h: u32,
    grayscale: bool,
    map_signal: Arc<AtomicU8>,
    pending: bool,
}

impl StagingSlot {
    fn new() -> Self {
        Self {
            buffer: None,
            buf_size: 0,
            row_pitch: 0,
            cap_w: 0,
            cap_h: 0,
            grayscale: false,
            map_signal: Arc::new(AtomicU8::new(MAP_PENDING)),
            pending: false,
        }
    }
}

/// Asynchronous screen capture with GPU blit-downsample, a ring buffer of
/// staging buffers, and non-blocking readback. Mirrors the old OpenGL
/// PBO + fence-sync pipeline: by the time we read slot N, the GPU has had
/// `SLOT_COUNT − 1` extra frames to finish the copy, so stalls are
/// virtually impossible.
///
/// Zero platform-specific code — works across Vulkan, Metal, DX12, and
/// WebGPU via wgpu.
///
/// Lifecycle:
///   1. Game calls [`ScreenCaptureState::request`] each frame it wants a
///      capture.
///   2. After all render passes, [`App`] calls
///      [`ScreenCaptureState::capture_from_texture`] which blits the
///      backbuffer into a small capture texture and encodes a
///      `copy_texture_to_buffer` into the next ring-buffer slot.
///   3. After `queue.submit()`, [`App`] calls
///      [`ScreenCaptureState::begin_readback_map`] to issue the async map.
///   4. On a subsequent frame the game polls
///      [`ScreenCaptureState::try_complete`]. The oldest ready slot is
///      consumed: raw BGRA → RGB565 (2 B/px) or grayscale (1 B/px).
pub struct ScreenCaptureState {
    // -- GPU blit resources (lazily initialised) --
    blit_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    blit_gray_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    blit_sampler: Option<egor_render::wgpu::Sampler>,
    blit_bind_group_layout: Option<egor_render::wgpu::BindGroupLayout>,
    // Intermediate copy of the backbuffer with TEXTURE_BINDING
    // (surface textures may lack TEXTURE_BINDING on some backends like DX12).
    source_copy: Option<Texture>,
    source_copy_view: Option<egor_render::wgpu::TextureView>,
    source_copy_w: u32,
    source_copy_h: u32,

    capture_texture: Option<Texture>,
    capture_view: Option<egor_render::wgpu::TextureView>,
    capture_tex_w: u32,
    capture_tex_h: u32,
    capture_tex_gray: bool,

    // -- request --
    requested: bool,
    capture_w: u32,
    capture_h: u32,
    pub grayscale: bool,

    // -- ring buffer of staging slots --
    slots: [StagingSlot; SLOT_COUNT],
    write_idx: usize,
    needs_map: Option<usize>,

    // -- completed result --
    result_ready: bool,
    result_w: u16,
    result_h: u16,
    rgb_buf: Vec<u8>,
}

impl ScreenCaptureState {
    pub fn new() -> Self {
        Self {
            blit_pipeline: None,
            blit_gray_pipeline: None,
            blit_sampler: None,
            blit_bind_group_layout: None,
            source_copy: None,
            source_copy_view: None,
            source_copy_w: 0,
            source_copy_h: 0,
            capture_texture: None,
            capture_view: None,
            capture_tex_w: 0,
            capture_tex_h: 0,
            capture_tex_gray: false,
            requested: false,
            capture_w: 0,
            capture_h: 0,
            grayscale: false,
            slots: [StagingSlot::new(), StagingSlot::new(), StagingSlot::new()],
            write_idx: 0,
            needs_map: None,
            result_ready: false,
            result_w: 0,
            result_h: 0,
            rgb_buf: Vec::new(),
        }
    }

    /// Pure math — compute capture dimensions preserving aspect ratio.
    /// Width is quantised to a multiple of 8 so that ±1 px jitter in the
    /// source resolution (common on web with fractional devicePixelRatio)
    /// does not flip the capture size and force every frame to be a keyframe.
    pub fn capture_dims(screen_w: f32, screen_h: f32) -> (u32, u32) {
        let cap_h = CAP_HEIGHT;
        let cap_w = if screen_h > 0.0 {
            let raw = ((screen_w / screen_h) * cap_h as f32) as u32;
            // Round to nearest multiple of 8
            ((raw + 4) / 8) * 8
        } else {
            424 // 8-aligned default for ~16:9
        };
        (cap_w.max(8), cap_h)
    }

    /// Game calls this to request a capture at the given dimensions.
    pub fn request(&mut self, w: u32, h: u32, grayscale: bool) {
        self.requested = true;
        self.capture_w = w;
        self.capture_h = h;
        self.grayscale = grayscale;
    }

    /// Returns `true` if a capture was requested this frame.
    pub fn is_requested(&self) -> bool {
        self.requested
    }

    /// Returns `true` if any ring-buffer slot has a GPU readback in flight.
    pub fn readback_in_flight(&self) -> bool {
        self.slots.iter().any(|s| s.pending)
    }

    // -- pipeline / resource setup (lazy) --------------------------------

    fn ensure_pipeline(&mut self, device: &Device) {
        if self.blit_pipeline.is_some() {
            return;
        }

        let shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: None,
            source: egor_render::wgpu::ShaderSource::Wgsl(BLIT_SHADER_WGSL.into()),
        });

        let gray_shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: None,
            source: egor_render::wgpu::ShaderSource::Wgsl(BLIT_GRAY_SHADER_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&egor_render::wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                egor_render::wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: egor_render::wgpu::ShaderStages::FRAGMENT,
                    ty: egor_render::wgpu::BindingType::Texture {
                        sample_type: egor_render::wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: egor_render::wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                egor_render::wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: egor_render::wgpu::ShaderStages::FRAGMENT,
                    ty: egor_render::wgpu::BindingType::Sampler(egor_render::wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&egor_render::wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let make_pipeline = |label, module: &egor_render::wgpu::ShaderModule, format| {
            device.create_render_pipeline(&egor_render::wgpu::RenderPipelineDescriptor {
                label,
                layout: Some(&pipeline_layout),
                vertex: egor_render::wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(egor_render::wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(egor_render::wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: egor_render::wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: egor_render::wgpu::PrimitiveState {
                    topology: egor_render::wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let pipeline = make_pipeline(None, &shader, TextureFormat::Bgra8Unorm);
        let gray_pipeline = make_pipeline(None, &gray_shader, TextureFormat::R8Unorm);

        let sampler = device.create_sampler(&egor_render::wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: egor_render::wgpu::AddressMode::ClampToEdge,
            address_mode_v: egor_render::wgpu::AddressMode::ClampToEdge,
            mag_filter: egor_render::wgpu::FilterMode::Nearest,
            min_filter: egor_render::wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        self.blit_pipeline = Some(pipeline);
        self.blit_gray_pipeline = Some(gray_pipeline);
        self.blit_sampler = Some(sampler);
        self.blit_bind_group_layout = Some(bind_group_layout);
    }

    fn ensure_capture_texture(&mut self, device: &Device, w: u32, h: u32, grayscale: bool) {
        if self.capture_tex_w == w && self.capture_tex_h == h && self.capture_tex_gray == grayscale && self.capture_texture.is_some() {
            return;
        }

        let format = if grayscale {
            TextureFormat::R8Unorm
        } else {
            TextureFormat::Bgra8Unorm
        };

        let texture = device.create_texture(&egor_render::wgpu::TextureDescriptor {
            label: None,
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: egor_render::wgpu::TextureDimension::D2,
            format,
            usage: egor_render::wgpu::TextureUsages::RENDER_ATTACHMENT | egor_render::wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        self.capture_view = Some(texture.create_view(&Default::default()));
        self.capture_texture = Some(texture);
        self.capture_tex_w = w;
        self.capture_tex_h = h;
        self.capture_tex_gray = grayscale;
    }

    fn ensure_source_copy(&mut self, device: &Device, w: u32, h: u32, format: TextureFormat) {
        if self.source_copy_w == w && self.source_copy_h == h && self.source_copy.is_some() {
            return;
        }
        let tex = device.create_texture(&egor_render::wgpu::TextureDescriptor {
            label: None,
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: egor_render::wgpu::TextureDimension::D2,
            format,
            usage: egor_render::wgpu::TextureUsages::TEXTURE_BINDING | egor_render::wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.source_copy_view = Some(tex.create_view(&Default::default()));
        self.source_copy = Some(tex);
        self.source_copy_w = w;
        self.source_copy_h = h;
    }

    // -- capture entry point (called by app.rs after rendering) -----------

    /// Blit-downsample the backbuffer into a small capture texture, then
    /// encode a `copy_texture_to_buffer` for async readback.
    ///
    /// `source` is the backbuffer `Texture` (must have `COPY_SRC` usage).
    /// The encoder must be the same one that will be submitted this frame.
    pub fn capture_from_texture(&mut self, device: &Device, encoder: &mut CommandEncoder, source: &Texture) {
        self.requested = false;

        let cap_w = self.capture_w.max(1);
        let cap_h = self.capture_h.max(1);
        let grayscale = self.grayscale;

        // -- Ring-buffer slot availability check --------------------------
        // Done BEFORE the blit so we skip ALL GPU work when the ring is full.
        let idx = self.write_idx;
        if self.slots[idx].pending {
            let status = self.slots[idx].map_signal.load(Ordering::Acquire);
            if status == MAP_READY {
                // Harvest the completed readback before reusing this slot.
                self.complete_slot(idx);
            } else {
                // Ring full — GPU hasn't finished this slot yet.
                // Skip capture entirely; no blit, no copy, no allocation.
                return;
            }
        }

        // Ensure GPU resources exist
        self.ensure_pipeline(device);
        self.ensure_capture_texture(device, cap_w, cap_h, grayscale);

        // The surface texture may not support TEXTURE_BINDING (e.g. DX12),
        // so copy it to an intermediate texture that does.
        let src_size = source.size();
        self.ensure_source_copy(device, src_size.width, src_size.height, source.format());
        encoder.copy_texture_to_texture(
            egor_render::wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: egor_render::wgpu::Origin3d::ZERO,
                aspect: egor_render::wgpu::TextureAspect::All,
            },
            egor_render::wgpu::TexelCopyTextureInfo {
                texture: self.source_copy.as_ref().unwrap(),
                mip_level: 0,
                origin: egor_render::wgpu::Origin3d::ZERO,
                aspect: egor_render::wgpu::TextureAspect::All,
            },
            Extent3d {
                width: src_size.width,
                height: src_size.height,
                depth_or_array_layers: 1,
            },
        );

        let bind_group_layout = self.blit_bind_group_layout.as_ref().expect("pipeline init");
        let sampler = self.blit_sampler.as_ref().expect("pipeline init");
        let pipeline = if grayscale {
            self.blit_gray_pipeline.as_ref().expect("pipeline init")
        } else {
            self.blit_pipeline.as_ref().expect("pipeline init")
        };
        let capture_view = self.capture_view.as_ref().expect("capture texture init");
        let source_view = self.source_copy_view.as_ref().expect("source copy init");

        // Create bind group sampling from the intermediate copy
        let bind_group = device.create_bind_group(&egor_render::wgpu::BindGroupDescriptor {
            label: None,
            layout: bind_group_layout,
            entries: &[
                egor_render::wgpu::BindGroupEntry {
                    binding: 0,
                    resource: egor_render::wgpu::BindingResource::TextureView(source_view),
                },
                egor_render::wgpu::BindGroupEntry {
                    binding: 1,
                    resource: egor_render::wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        // Blit render pass: draw fullscreen triangle sampling the backbuffer
        {
            let mut rpass = encoder.begin_render_pass(&egor_render::wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(egor_render::wgpu::RenderPassColorAttachment {
                    view: capture_view,
                    resolve_target: None,
                    ops: egor_render::wgpu::Operations {
                        load: egor_render::wgpu::LoadOp::Clear(egor_render::wgpu::Color::BLACK),
                        store: egor_render::wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        // Copy capture texture → staging buffer for CPU readback
        let slot = &mut self.slots[idx];
        let bytes_per_pixel: u32 = if grayscale { 1 } else { 4 };
        let unpadded_row = cap_w * bytes_per_pixel;
        let align = egor_render::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (unpadded_row + align - 1) / align * align;
        let buffer_size = (padded_row * cap_h) as u64;

        // Reuse the staging buffer if dimensions haven't changed
        let needs_new = slot.buffer.is_none() || slot.buf_size != buffer_size || slot.row_pitch != padded_row;

        if needs_new {
            slot.buffer = Some(device.create_buffer(&BufferDescriptor {
                label: None,
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
            slot.buf_size = buffer_size;
            slot.row_pitch = padded_row;
        }

        let buffer = slot.buffer.as_ref().expect("staging buffer");

        encoder.copy_texture_to_buffer(
            egor_render::wgpu::TexelCopyTextureInfo {
                texture: self.capture_texture.as_ref().expect("capture texture"),
                mip_level: 0,
                origin: egor_render::wgpu::Origin3d::ZERO,
                aspect: egor_render::wgpu::TextureAspect::All,
            },
            egor_render::wgpu::TexelCopyBufferInfo {
                buffer,
                layout: egor_render::wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(cap_h),
                },
            },
            Extent3d {
                width: cap_w,
                height: cap_h,
                depth_or_array_layers: 1,
            },
        );

        // Store capture metadata on the slot so try_complete knows how to
        // decode regardless of whether the request params changed since then.
        let slot = &mut self.slots[idx];
        slot.cap_w = cap_w;
        slot.cap_h = cap_h;
        slot.grayscale = grayscale;
        slot.pending = true;

        self.needs_map = Some(idx);
        self.write_idx = (self.write_idx + 1) % SLOT_COUNT;
    }

    /// Start the async map request for the most recently written slot.
    /// **Must** be called AFTER `queue.submit()` — wgpu rejects submitting
    /// commands that reference a buffer with a pending map operation.
    pub fn begin_readback_map(&mut self) {
        let Some(idx) = self.needs_map.take() else {
            return;
        };
        let slot = &mut self.slots[idx];
        if !slot.pending {
            return;
        }
        let Some(buffer) = slot.buffer.as_ref() else {
            return;
        };
        slot.map_signal.store(MAP_PENDING, Ordering::Release);
        let signal = slot.map_signal.clone();
        buffer.slice(..).map_async(egor_render::wgpu::MapMode::Read, move |result| {
            signal.store(if result.is_ok() { MAP_READY } else { MAP_FAILED }, Ordering::Release);
        });
    }

    // -- readback polling ------------------------------------------------

    /// Consume the oldest completed ring-buffer slot. For grayscale the
    /// staging buffer already holds R8 data from the GPU — just strip row
    /// padding. For RGB565, convert BGRA→RGB565 with unsafe pointer math.
    fn complete_slot(&mut self, idx: usize) {
        let cap_w = self.slots[idx].cap_w;
        let cap_h = self.slots[idx].cap_h;
        let row_pitch = self.slots[idx].row_pitch as usize;
        let grayscale = self.slots[idx].grayscale;
        let w = cap_w as usize;
        let h = cap_h as usize;
        let pixel_count = w * h;

        let buffer = match self.slots[idx].buffer.take() {
            Some(b) => b,
            None => {
                self.slots[idx].pending = false;
                return;
            }
        };

        let data = buffer.slice(..).get_mapped_range();
        let src = data.as_ref().as_ptr();

        if grayscale {
            // R8Unorm capture texture — data is already single-channel.
            // Just strip row padding via a tight copy.
            let out_len = pixel_count;
            self.rgb_buf.reserve(out_len.saturating_sub(self.rgb_buf.len()));
            // SAFETY: we've reserved enough capacity; the buffer is
            // immediately filled by the copy loop below.
            unsafe { self.rgb_buf.set_len(out_len) };
            if row_pitch == w {
                // No padding — single memcpy.
                unsafe {
                    std::ptr::copy_nonoverlapping(src, self.rgb_buf.as_mut_ptr(), out_len);
                }
            } else {
                let dst = self.rgb_buf.as_mut_ptr();
                for y in 0..h {
                    unsafe {
                        std::ptr::copy_nonoverlapping(src.add(y * row_pitch), dst.add(y * w), w);
                    }
                }
            }
        } else {
            let out_len = pixel_count * 2;
            self.rgb_buf.reserve(out_len.saturating_sub(self.rgb_buf.len()));
            // SAFETY: we've reserved enough capacity; pack_bgra_to_rgb565
            // writes exactly pixel_count * 2 bytes.
            unsafe { self.rgb_buf.set_len(out_len) };
            unsafe { pack_bgra_to_rgb565(src, self.rgb_buf.as_mut_ptr(), w, h, row_pitch) };
        }

        drop(data);
        buffer.unmap();

        self.slots[idx].buffer = Some(buffer);
        self.slots[idx].pending = false;
        self.result_ready = true;
        self.result_w = cap_w as u16;
        self.result_h = cap_h as u16;
    }

    /// Poll for a completed readback. Returns `Some((width, height))` when
    /// pixel data is available in [`Self::rgb_buf`].
    ///
    /// Non-blocking: iterates ring-buffer slots oldest-first and consumes
    /// the first whose `map_async` callback has fired. Driven by the game
    /// loop's existing `device.poll(PollType::Poll)`.
    pub fn try_complete(&mut self) -> Option<(u16, u16)> {
        if self.result_ready {
            self.result_ready = false;
            return Some((self.result_w, self.result_h));
        }

        // Iterate oldest → newest (write_idx is the next write position,
        // which wraps to the oldest pending slot).
        for i in 0..SLOT_COUNT {
            let idx = (self.write_idx + i) % SLOT_COUNT;
            if !self.slots[idx].pending {
                continue;
            }
            let status = self.slots[idx].map_signal.load(Ordering::Acquire);
            if status == MAP_PENDING {
                continue;
            }
            if status == MAP_FAILED {
                eprintln!("[ScreenCapture] buffer map failed on slot {idx}");
                self.slots[idx].pending = false;
                continue;
            }
            // MAP_READY
            self.complete_slot(idx);
            if self.result_ready {
                self.result_ready = false;
                return Some((self.result_w, self.result_h));
            }
        }

        None
    }

    /// Access the completed RGB pixel buffer.
    pub fn rgb_buf(&self) -> &[u8] {
        &self.rgb_buf
    }
}

impl Default for ScreenCaptureState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Graphics
// ---------------------------------------------------------------------------

/// High-level 2D drawing interface that simplifies the [`Renderer`]
pub struct Graphics<'a> {
    renderer: &'a mut Renderer,
    batch: &'a mut PrimitiveBatch,
    camera: Camera,
    text_renderer: &'a mut TextRenderer,
    target_format: TextureFormat,
    target_size: (u32, u32),
    current_shader: Option<usize>,
    render_targets: &'a mut RenderTargetStore,
    screen_capture: &'a mut ScreenCaptureState,
    offscreen_batches: &'a mut Vec<PrimitiveBatch>,
}

impl<'a> Graphics<'a> {
    /// Create `Graphics` with [`Renderer`], [`TextRenderer`] & `TextureFormat`
    pub fn new(
        renderer: &'a mut Renderer,
        batch: &'a mut PrimitiveBatch,
        text_renderer: &'a mut TextRenderer,
        render_targets: &'a mut RenderTargetStore,
        screen_capture: &'a mut ScreenCaptureState,
        offscreen_batches: &'a mut Vec<PrimitiveBatch>,
        format: TextureFormat,
        w: u32,
        h: u32,
    ) -> Self {
        Self {
            renderer,
            batch,
            camera: Camera::default(),
            text_renderer,
            target_format: format,
            target_size: (w, h),
            current_shader: None,
            render_targets,
            screen_capture,
            offscreen_batches,
        }
    }

    /// Create a new offscreen render target
    pub fn create_offscreen(&self, width: u32, height: u32) -> OffscreenTarget {
        self.renderer.create_offscreen_target(width, height, self.target_format)
    }

    /// Render to an offscreen target
    pub fn render_offscreen(&mut self, target: &mut OffscreenTarget, render_fn: impl FnMut(&mut Graphics)) {
        self.render_offscreen_with_limits(
            target,
            GeometryBatch::DEFAULT_MAX_VERTICES,
            GeometryBatch::DEFAULT_MAX_INDICES,
            render_fn,
        );
    }

    /// Render to an offscreen target using a temporary batch with custom vertex/index buffer limits.
    /// Use this when the default limits are too large for memory-constrained platforms,
    /// or too small for complex offscreen scenes.
    /// For most cases, prefer [`Self::render_offscreen`] which uses sensible defaults
    pub fn render_offscreen_with_limits(
        &mut self,
        target: &mut OffscreenTarget,
        max_verts: usize,
        max_indices: usize,
        mut render_fn: impl FnMut(&mut Graphics),
    ) {
        let (w, h) = target.size();
        let format = target.format();

        let mut offscreen_batch = self
            .offscreen_batches
            .pop()
            .unwrap_or_else(|| PrimitiveBatch::new(max_verts, max_indices));
        offscreen_batch.reset();
        let mut offscreen_gfx = Graphics {
            renderer: self.renderer,
            batch: &mut offscreen_batch,
            camera: Camera::default(),
            text_renderer: self.text_renderer,
            target_size: (w, h),
            target_format: format,
            current_shader: None,
            render_targets: self.render_targets,
            screen_capture: self.screen_capture,
            offscreen_batches: self.offscreen_batches,
        };

        render_fn(&mut offscreen_gfx);
        offscreen_gfx.upload_camera();
        let mut geometry = offscreen_batch.drain_all();

        let mut encoder = self.renderer.device().create_command_encoder(&Default::default());

        {
            let mut r_pass =
                self.renderer
                    .begin_render_pass_with_depth(&mut encoder, target.render_view(), target.offscreen_depth_view(), true);

            let mut cur_tex: Option<usize> = None;
            let mut cur_shd: Option<usize> = None;
            let mut cur_cam_offset = u32::MAX;
            let mut quad_bound = false;
            let mut cur_scissor = None;

            if let Some(first) = geometry.first() {
                self.renderer.bind_pass_state(&mut r_pass, first.texture_id, first.shader_id);
                cur_tex = first.texture_id;
                cur_shd = first.shader_id;
                quad_bound = true;
            }
            for entry in &mut geometry {
                if entry.scissor != cur_scissor {
                    if let Some(rect) = entry.scissor {
                        let sx = rect.0.min(w.saturating_sub(1));
                        let sy = rect.1.min(h.saturating_sub(1));
                        let sw = rect.2.min(w - sx).max(1);
                        let sh = rect.3.min(h - sy).max(1);
                        r_pass.set_scissor_rect(sx, sy, sw, sh);
                    } else {
                        r_pass.set_scissor_rect(0, 0, w, h);
                    }
                    cur_scissor = entry.scissor;
                }
                self.renderer.draw_batch(
                    &mut r_pass,
                    &mut entry.geometry,
                    entry.texture_id,
                    entry.shader_id,
                    0,
                    &mut cur_tex,
                    &mut cur_shd,
                    &mut cur_cam_offset,
                    &mut quad_bound,
                );
            }
        }

        target.copy_to_sample(&mut encoder);

        let _ = self.renderer.queue().submit(Some(encoder.finish()));

        offscreen_batch.recycle(geometry);
        self.offscreen_batches.push(offscreen_batch);
    }

    /// Use an offscreen target as a texture
    pub fn offscreen_as_texture(&mut self, target: &mut OffscreenTarget) -> usize {
        self.renderer.add_offscreen_texture(target)
    }

    pub(crate) fn set_target_size(&mut self, w: u32, h: u32) {
        self.target_size = (w, h);
    }

    pub fn target_size(&self) -> (u32, u32) {
        self.target_size
    }

    /// Upload camera matrix to the GPU.
    /// Call after user drawing is complete and before the render pass
    pub(crate) fn upload_camera(&mut self) {
        let (w, h) = self.target_size;
        self.renderer
            .upload_camera_matrix(self.camera.view_proj((w as f32, h as f32).into()).to_cols_array_2d());
    }

    /// Clear the screen to a color
    pub fn clear(&mut self, color: Color) {
        self.renderer.set_clear_color(color.into());
    }
    /// Get current surface size in pixels
    pub fn screen_size(&self) -> Vec2 {
        let (w, h) = self.target_size;
        (w as f32, h as f32).into()
    }
    /// Mutable access to [`Camera`]
    pub fn camera(&mut self) -> &mut Camera {
        &mut self.camera
    }

    /// Start building a rectangle primitive
    pub fn rect(&mut self) -> RectangleBuilder<'_> {
        RectangleBuilder::new(self.batch, self.current_shader)
    }

    /// Push an axis-aligned, unrotated, colored sprite instance directly into
    /// the batch — bypasses RectangleBuilder, Mat2::from_angle, and builder
    /// overhead.  Used by the optimised `draw_cmd` fast-path for all
    /// non-rotated sprite draws (entities, UI, health bars, etc.).
    #[inline(always)]
    pub fn push_sprite(&mut self, tex_id: usize, x: f32, y: f32, w: f32, h: f32, uvs: [f32; 4], color: [f32; 4]) {
        self.batch.push_instance(
            egor_render::instance::Instance::new([w, 0.0, 0.0, h], [x + w * 0.5, y + h * 0.5, self.batch.draw_depth()], color, uvs),
            Some(tex_id),
            self.current_shader,
        );
    }

    /// Push an axis-aligned, unrotated, WHITE sprite instance directly into the
    /// batch.  Designed for the tile-map hot loop.
    #[inline(always)]
    pub fn push_sprite_aa(&mut self, tex_id: usize, x: f32, y: f32, w: f32, h: f32, uvs: [f32; 4]) {
        self.push_sprite(tex_id, x, y, w, h, uvs, [1.0, 1.0, 1.0, 1.0]);
    }

    /// Ensure a batch exists for the given tile texture. Call once per atlas
    /// texture change, then use [`push_tile`] for the actual tile instances.
    #[inline(always)]
    pub fn ensure_tile_batch(&mut self, tex_id: usize) {
        self.batch.ensure_batch(Some(tex_id), self.current_shader);
    }

    /// Push a tile instance directly into the current batch, skipping all
    /// batch-key comparisons. The caller MUST call [`ensure_tile_batch`]
    /// first whenever the texture changes.
    #[inline(always)]
    pub fn push_tile(&mut self, x: f32, y: f32, w: f32, h: f32, depth: f32, uvs: [f32; 4]) {
        self.batch.push_instance_unchecked(egor_render::instance::Instance::new(
            [w, 0.0, 0.0, h],
            [x + w * 0.5, y + h * 0.5, depth],
            [1.0, 1.0, 1.0, 1.0],
            uvs,
        ));
    }

    /// Push a colored sprite instance into the current batch WITHOUT any
    /// batch-key comparisons. The caller MUST call [`ensure_tile_batch`]
    /// first whenever the texture changes. Uses the current draw_depth.
    #[inline(always)]
    pub fn push_sprite_unchecked(&mut self, x: f32, y: f32, w: f32, h: f32, uvs: [f32; 4], color: [f32; 4]) {
        self.batch.push_instance_unchecked(egor_render::instance::Instance::new(
            [w, 0.0, 0.0, h],
            [x + w * 0.5, y + h * 0.5, self.batch.draw_depth()],
            color,
            uvs,
        ));
    }
    /// Start building an arbitrary polygon primitive, capable of triangles, circles, n-gons
    pub fn polygon(&mut self) -> PolygonBuilder<'_> {
        PolygonBuilder::new(self.batch, self.current_shader)
    }

    /// Push raw vertex/index geometry directly into the current untextured batch.
    /// Vertices must have positions in world space and colors as RGBA \[0..1\].
    /// Indices reference vertices in the provided slice (0-based); they are
    /// automatically offset to match the batch's vertex base.
    pub fn push_geometry(&mut self, verts: &[egor_render::vertex::Vertex], indices: &[u16]) {
        let vert_count = verts.len();
        let idx_count = indices.len();
        if let Some((v_slice, i_slice, base)) = self.batch.allocate(vert_count, idx_count, None, self.current_shader) {
            v_slice.copy_from_slice(verts);
            for (i, idx) in indices.iter().enumerate() {
                i_slice[i] = *idx + base;
            }
        }
    }

    /// Start building a polyline (stroked path) primitive
    pub fn polyline(&mut self) -> PolylineBuilder<'_> {
        PolylineBuilder::new(self.batch, self.current_shader)
    }
    /// Start building a vector path (lines + curves) to be filled or stroked
    pub fn path(&mut self) -> PathBuilder<'_> {
        PathBuilder::new(self.batch, self.current_shader)
    }
    /// Load a font from disk into the text system.
    pub fn load_font(&mut self, bytes: &[u8]) -> Option<String> {
        self.text_renderer.load_font_bytes(bytes)
    }
    /// Draw a line of text
    pub fn text(&mut self, text: &str) -> TextBuilder<'_> {
        TextBuilder::new(self.text_renderer, text.to_string())
    }

    /// Load a texture from raw image data (e.g., PNG bytes)
    ///
    /// Returns a texture ID that can be used with `.texture(id)` on primitives.
    /// Typically called once during initialization (when `timer.frame == 0`).
    pub fn load_texture(&mut self, data: &[u8]) -> usize {
        self.renderer.add_texture(data)
    }

    /// Load a texture with nearest-neighbor (pixel-perfect) filtering
    pub fn load_texture_nearest(&mut self, data: &[u8]) -> usize {
        self.renderer.add_texture_nearest(data)
    }

    /// Create a texture from raw RGBA8 pixel data.
    pub fn add_texture_raw(&mut self, w: u32, h: u32, data: &[u8]) -> usize {
        self.renderer.add_texture_raw(w, h, data)
    }
    /// Update texture data by index
    pub fn update_texture(&mut self, index: usize, data: &[u8]) {
        self.renderer.update_texture(index, data);
    }
    /// Update texture data by index with raw width/height
    pub fn update_texture_raw(&mut self, index: usize, w: u32, h: u32, data: &[u8]) {
        self.renderer.update_texture_raw(index, w, h, data);
    }

    /// Load a custom shader from WGSL source code
    pub fn load_shader(&mut self, wgsl_source: &str) -> usize {
        self.renderer.add_shader(wgsl_source)
    }

    /// Create a uniform buffer from raw bytes, returns a uniform id
    pub fn create_uniform(&mut self, data: &[u8]) -> usize {
        self.renderer.add_uniform(data)
    }

    /// Update an existing uniform buffer with raw bytes
    pub fn update_uniform(&mut self, id: usize, data: &[u8]) {
        self.renderer.update_uniform(id, data);
    }

    /// Load a custom shader with associated uniform buffers
    pub fn load_shader_with_uniforms(&mut self, wgsl_source: &str, uniform_ids: &[usize]) -> usize {
        self.renderer.add_shader_with_uniforms(wgsl_source, uniform_ids)
    }

    /// Execute drawing commands with a custom shader
    ///
    /// The shader is automatically reset to default after the closure drops
    pub fn with_shader(&mut self, shader_id: usize, mut render_fn: impl FnMut(&mut Self)) {
        let previous_shader = self.current_shader;
        self.current_shader = Some(shader_id);
        render_fn(self);
        self.current_shader = previous_shader;
    }

    /// Directly set (or clear) the active shader for subsequent draw commands.
    pub fn set_current_shader(&mut self, shader_id: Option<usize>) {
        self.current_shader = shader_id;
    }

    /// Set the scissor rect for subsequent draw commands.
    /// `None` disables scissoring (full viewport).
    pub fn set_scissor(&mut self, rect: Option<(u32, u32, u32, u32)>) {
        self.batch.set_scissor(rect);
    }

    /// Override the camera matrix for subsequent draw commands.
    /// Batches tagged with a camera override will trigger a render-pass split
    /// so that each sub-group renders with its own projection.
    pub fn set_camera_matrix(&mut self, mat: [[f32; 4]; 4]) {
        self.batch.set_camera_matrix(mat);
    }

    /// Reset the camera override. Subsequent draws will use the built-in egor camera.
    pub fn reset_camera_matrix(&mut self) {
        self.batch.reset_camera_matrix();
    }

    /// Set the depth value for subsequent draw commands.
    /// Used for GPU depth testing with LessOrEqual comparison.
    pub fn set_draw_depth(&mut self, depth: f32) {
        self.batch.set_draw_depth(depth);
    }

    // -- managed render targets -----------------------------------------

    /// Create a managed offscreen render target and return its store index.
    /// Also registers it as a drawable texture, returning `(store_id, egor_texture_id)`.
    pub fn create_managed_render_target(&mut self, width: u32, height: u32) -> (usize, usize) {
        let store_id = self
            .render_targets
            .create(self.renderer.device(), width, height, self.target_format);
        let tex_id = self.renderer.add_offscreen_texture(self.render_targets.get_mut(store_id));
        (store_id, tex_id)
    }

    /// Resize a managed render target. Re-registers the texture binding.
    /// Returns the (possibly new) egor texture id.
    pub fn resize_managed_render_target(&mut self, store_id: usize, width: u32, height: u32) -> usize {
        self.render_targets.resize(self.renderer.device(), store_id, width, height);
        self.renderer.add_offscreen_texture(self.render_targets.get_mut(store_id))
    }

    /// Direct subsequent draw commands to a managed offscreen render target.
    pub fn set_active_render_target(&mut self, store_id: usize) {
        self.batch.set_render_target(Some(store_id));
    }

    /// Restore drawing to the main backbuffer.
    pub fn clear_active_render_target(&mut self) {
        self.batch.set_render_target(None);
    }

    // -- screen capture -------------------------------------------------

    /// Request a screen capture at the specified dimensions.
    pub fn request_screen_capture(&mut self, w: u32, h: u32, grayscale: bool) {
        self.screen_capture.request(w, h, grayscale);
    }

    /// Poll for a completed screen capture result.
    /// Returns `Some((width, height))` when pixel data is available.
    pub fn poll_screen_capture(&mut self) -> Option<(u16, u16)> {
        self.screen_capture.try_complete()
    }

    /// Access the completed screen capture RGB buffer.
    pub fn screen_capture_rgb_buf(&self) -> &[u8] {
        self.screen_capture.rgb_buf()
    }
}
