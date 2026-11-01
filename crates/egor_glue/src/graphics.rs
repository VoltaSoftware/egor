use egor_render::{
    AdapterInfo, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Extent3d,
    Renderer, Texture, TextureFormat,
    batch::GeometryBatch,
    target::{OffscreenTarget, RenderTarget},
    wgpu::SurfaceTransform,
};
use glam::Vec2;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{self, Receiver, Sender};
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
use web_time::Instant;

use crate::primitives::PathBuilder;
use crate::{
    camera::Camera,
    color::Color,
    primitives::{PolygonBuilder, PolylineBuilder, PrimitiveBatch, RectangleBuilder},
};

// ---------------------------------------------------------------------------
// Render Target Store — persistent across frames, owns OffscreenTargets
// ---------------------------------------------------------------------------

/// A GPU resource the game freed, released once the frame that freed it is drawn.
enum PendingFree {
    Texture(usize),
    RenderTarget(usize),
    Uniform(usize),
    Shader(usize),
}

/// Persistent storage for offscreen render targets created by game code.
///
/// Stored in [`App`] and passed to [`Graphics`] each frame by reference.
/// Render targets survive across frames until the game frees them; resizing keeps their ids.
pub struct RenderTargetStore {
    targets: Vec<Option<OffscreenTarget>>,
    /// Each target's drawable texture, once registered; resizing rebinds it in place.
    textures: Vec<Option<usize>>,
    /// Slots of freed targets, which the next ones reuse.
    free_slots: Vec<usize>,
    /// What the game freed this frame (any resource, not only targets). Draws recorded earlier in the frame may
    /// still use them, so they are released at the start of the next frame.
    pending_frees: Vec<PendingFree>,
}

impl RenderTargetStore {
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
            textures: Vec::new(),
            free_slots: Vec::new(),
            pending_frees: Vec::new(),
        }
    }

    /// Create an offscreen render target and return its index.
    pub fn create(
        &mut self,
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> usize {
        let target = Some(OffscreenTarget::new(device, width, height, format));
        match self.free_slots.pop() {
            Some(id) => {
                self.targets[id] = target;
                self.textures[id] = None;
                id
            }
            None => {
                self.targets.push(target);
                self.textures.push(None);
                self.targets.len() - 1
            }
        }
    }

    /// Resize an existing offscreen target. If its dimensions already match, this is a no-op.
    pub fn resize(&mut self, device: &Device, id: usize, width: u32, height: u32) {
        self.get_mut(id).resize(device, width, height);
    }

    pub fn get(&self, id: usize) -> &OffscreenTarget {
        self.targets[id]
            .as_ref()
            .unwrap_or_else(|| panic!("render target {id} was freed"))
    }

    pub fn get_mut(&mut self, id: usize) -> &mut OffscreenTarget {
        self.targets[id]
            .as_mut()
            .unwrap_or_else(|| panic!("render target {id} was freed"))
    }

    /// Slots in use or free; freed slots are reused before the store grows.
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Releases what was freed last frame. Call at the start of a frame, before the game records draws.
    pub(crate) fn release_pending_frees(&mut self, renderer: &mut Renderer) {
        for free in self.pending_frees.drain(..) {
            match free {
                PendingFree::Texture(id) => renderer.remove_texture(id),
                PendingFree::RenderTarget(id) => {
                    if self.targets.get_mut(id).and_then(Option::take).is_some() {
                        if let Some(texture) = self.textures[id].take() {
                            renderer.remove_texture(texture);
                        }
                        self.free_slots.push(id);
                    }
                }
                PendingFree::Uniform(id) => renderer.remove_uniform(id),
                PendingFree::Shader(id) => renderer.remove_shader(id),
            }
        }
    }
}

impl Default for RenderTargetStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Screen Capture State — RGBA8 screenshots through GPU readback
// ---------------------------------------------------------------------------

const SLOT_COUNT: usize = 3;

const MAP_PENDING: u8 = 0;
const MAP_READY: u8 = 1;
const MAP_FAILED: u8 = 2;
const MAP_CANCELLED: u8 = 3;

struct PreparedCapture {
    slot_idx: usize,
    cap_w: u32,
    cap_h: u32,
}

/// Capture-only counters for profiling. Byte counts exclude driver overhead.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScreenCaptureMetrics {
    pub readback_bytes: u64,
    pub staging_allocations: u64,
    pub staging_allocated_bytes: u64,
    pub skipped_requests: u64,
    pub completed_frames: u64,
    pub decode_us: u64,
    pub conversion_us: u64,
    pub unmap_us: u64,
    pub worker_jobs: u64,
    pub worker_bytes: u64,
    pub gpu_bytes: u64,
    pub cpu_buffer_bytes: u64,
}

/// WGSL shader for fullscreen-triangle blit with bilinear sampling.
const BLIT_SHADER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// The destination's turn from the source, as `wgpu::SurfaceTransform`: 0 none, then 90, 180 and 270
// degrees clockwise. Set when blitting into a pre-rotated backbuffer.
override TURN: u32 = 0u;

fn turn(p: vec2<f32>) -> vec2<f32> {
    switch TURN {
        case 1u: { return vec2<f32>(p.y, -p.x); }
        case 2u: { return -p; }
        case 3u: { return vec2<f32>(-p.y, p.x); }
        default: { return p; }
    }
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    // Fullscreen triangle: 3 vertices cover the entire clip space.
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(turn(uv * 2.0 - 1.0), 0.0, 1.0);
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

/// [`BLIT_SHADER_WGSL`]'s `TURN` for a destination turned `turn` from its source.
fn blit_turn_constants(turn: SurfaceTransform) -> [(&'static str, f64); 1] {
    [("TURN", f64::from(turn as u32))]
}

fn blit_turn_options<'a>(
    constants: &'a [(&'a str, f64)],
) -> egor_render::wgpu::PipelineCompilationOptions<'a> {
    egor_render::wgpu::PipelineCompilationOptions {
        constants,
        ..Default::default()
    }
}

/// Capture shader for sources sampled as linear color where the displayed
/// backbuffer uses sRGB encoding. It writes display-encoded RGB into a
/// non-sRGB readback target so CPU-side packing sees the same byte domain as
/// the direct surface-copy path.
const BLIT_ENCODE_SHADER_WGSL: &str = r#"
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

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = textureSample(t_src, s_src, in.uv);
    return vec4<f32>(linear_to_srgb(c.rgb), c.a);
}
"#;

fn decode_readback_into(
    buffer: &Buffer,
    dst: &mut Vec<u8>,
    cap_w: u32,
    cap_h: u32,
    row_pitch: usize,
) {
    #[cfg(feature = "profiling")]
    profiling::scope!("decode_readback");
    let data = buffer
        .slice(..)
        .get_mapped_range()
        .expect("readback buffer range must remain mapped after map_async succeeds");
    decode_readback_bytes(&data, dst, cap_w, cap_h, row_pitch);
}

/// Copies the RGBA8 rows out of a readback, dropping each row's padding.
fn decode_readback_bytes(data: &[u8], dst: &mut Vec<u8>, cap_w: u32, cap_h: u32, row_pitch: usize) {
    let row_bytes = cap_w as usize * 4;
    dst.clear();
    dst.reserve(row_bytes * cap_h as usize);
    for y in 0..cap_h as usize {
        dst.extend_from_slice(&data[y * row_pitch..y * row_pitch + row_bytes]);
    }
}

// -- Ring-buffer staging slot -----------------------------------------------

struct StagingSlot {
    buffer: Option<Buffer>,
    rgb_buf: Vec<u8>,
    buf_size: u64,
    row_pitch: u32,
    cap_w: u32,
    cap_h: u32,
    map_signal: Arc<AtomicU8>,
    pending: bool,
}

impl StagingSlot {
    fn new() -> Self {
        Self {
            buffer: None,
            rgb_buf: Vec::new(),
            buf_size: 0,
            row_pitch: 0,
            cap_w: 0,
            cap_h: 0,
            map_signal: Arc::new(AtomicU8::new(MAP_PENDING)),
            pending: false,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct ReadbackJob {
    slot_idx: usize,
    map_signal: Arc<AtomicU8>,
    rgb_buf: Vec<u8>,
    buffer: Buffer,
    buf_size: u64,
    row_pitch: usize,
    cap_w: u32,
    cap_h: u32,
}

#[cfg(not(target_arch = "wasm32"))]
struct ReadbackResult {
    slot_idx: usize,
    buffer: Buffer,
    buf_size: u64,
    row_pitch: u32,
    cap_w: u32,
    cap_h: u32,
    rgb_buf: Vec<u8>,
    complete_us: u128,
    conversion_us: u128,
}

#[cfg(not(target_arch = "wasm32"))]
struct ReadbackWorker {
    jobs: Sender<ReadbackJob>,
    results: Receiver<ReadbackResult>,
}

#[cfg(not(target_arch = "wasm32"))]
impl ReadbackWorker {
    fn new() -> Self {
        let (job_tx, job_rx) = mpsc::channel::<ReadbackJob>();
        let (result_tx, result_rx) = mpsc::channel::<ReadbackResult>();
        thread::Builder::new()
            .name("egor-readback".to_owned())
            .spawn(move || {
                while let Ok(job) = job_rx.recv() {
                    let complete_start = Instant::now();
                    let mut rgb_buf = job.rgb_buf;
                    if job.map_signal.load(Ordering::Acquire) != MAP_CANCELLED {
                        decode_readback_into(
                            &job.buffer,
                            &mut rgb_buf,
                            job.cap_w,
                            job.cap_h,
                            job.row_pitch,
                        );
                    }
                    let conversion_us = complete_start.elapsed().as_micros();
                    let complete_us = complete_start.elapsed().as_micros();
                    let result = ReadbackResult {
                        slot_idx: job.slot_idx,
                        buffer: job.buffer,
                        buf_size: job.buf_size,
                        row_pitch: job.row_pitch as u32,
                        cap_w: job.cap_w,
                        cap_h: job.cap_h,
                        rgb_buf,
                        complete_us,
                        conversion_us,
                    };
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }
            })
            .expect("failed to spawn egor readback worker");

        Self {
            jobs: job_tx,
            results: result_rx,
        }
    }
}

/// The downsample blits a capture draws with, and what binds their source.
struct BlitPipelines {
    plain: egor_render::wgpu::RenderPipeline,
    encode: egor_render::wgpu::RenderPipeline,
    sampler: egor_render::wgpu::Sampler,
    bind_group_layout: egor_render::wgpu::BindGroupLayout,
}

/// What a worker thread compiled: the blits if still missing, and the present pipeline for `key`.
#[cfg(not(target_arch = "wasm32"))]
struct BuiltCapturePipelines {
    blits: Option<BlitPipelines>,
    present: egor_render::wgpu::RenderPipeline,
    key: (TextureFormat, SurfaceTransform),
}

/// The capture pipelines compiling for a surface format and turn.
enum PipelineBuild {
    Idle,
    /// A family a frame, on the thread that draws (OpenGL, browsers).
    Frames((TextureFormat, SurfaceTransform)),
    #[cfg(not(target_arch = "wasm32"))]
    Thread(
        (TextureFormat, SurfaceTransform),
        std::sync::mpsc::Receiver<BuiltCapturePipelines>,
    ),
}

impl PipelineBuild {
    fn building(&self) -> Option<(TextureFormat, SurfaceTransform)> {
        match self {
            Self::Idle => None,
            Self::Frames(key) => Some(*key),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Thread(key, _) => Some(*key),
        }
    }
}

/// Asynchronous RGBA8 screen capture with a GPU blit to the requested size, a
/// ring buffer of staging buffers, and asynchronous readback through wgpu.
/// Native row copies run on a worker. An occupied ring skips new requests
/// instead of waiting for the GPU or allowing the worker queue to grow.
///
/// Lifecycle:
///   1. Game calls [`ScreenCaptureState::request_rgba8`] each frame it wants a
///      capture.
///   2. After all render passes, [`App`] calls
///      [`ScreenCaptureState::capture_from_texture`] which blits the
///      backbuffer into a small capture texture and encodes a
///      `copy_texture_to_buffer` into the next ring-buffer slot.
///   3. After `queue.submit()`, [`App`] calls
///      [`ScreenCaptureState::begin_readback_map`] to issue the async map.
///   4. On a subsequent frame the game polls
///      [`ScreenCaptureState::try_complete`]. Native builds hand ready slots
///      to a worker thread; wasm consumes the ready slot synchronously.
pub struct ScreenCaptureState {
    metrics: ScreenCaptureMetrics,
    buffers_released: bool,
    // -- GPU blit resources (lazily initialised) --
    blit_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    blit_encode_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    blit_sampler: Option<egor_render::wgpu::Sampler>,
    blit_bind_group_layout: Option<egor_render::wgpu::BindGroupLayout>,
    present_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    present_pipeline_key: Option<(TextureFormat, SurfaceTransform)>,
    composite_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    composite_pipeline_key: Option<(TextureFormat, SurfaceTransform)>,
    pipeline_build: PipelineBuild,
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

    // -- request --
    requested: bool,
    capture_w: u32,
    capture_h: u32,

    // -- ring buffer of staging slots --
    slots: [StagingSlot; SLOT_COUNT],
    write_idx: usize,
    needs_map: Option<usize>,
    #[cfg(not(target_arch = "wasm32"))]
    readback_worker: Option<ReadbackWorker>,

    // -- completed result --
    result_ready: bool,
    result_w: u16,
    result_h: u16,
    rgb_buf: Vec<u8>,
    composite_render_target: Option<usize>,
}

impl ScreenCaptureState {
    pub fn new() -> Self {
        Self {
            metrics: ScreenCaptureMetrics::default(),
            buffers_released: false,
            blit_pipeline: None,
            blit_encode_pipeline: None,
            blit_sampler: None,
            blit_bind_group_layout: None,
            present_pipeline: None,
            present_pipeline_key: None,
            composite_pipeline: None,
            composite_pipeline_key: None,
            pipeline_build: PipelineBuild::Idle,
            source_copy: None,
            source_copy_view: None,
            source_copy_w: 0,
            source_copy_h: 0,
            capture_texture: None,
            capture_view: None,
            capture_tex_w: 0,
            capture_tex_h: 0,
            requested: false,
            capture_w: 0,
            capture_h: 0,
            slots: [StagingSlot::new(), StagingSlot::new(), StagingSlot::new()],
            write_idx: 0,
            needs_map: None,
            #[cfg(not(target_arch = "wasm32"))]
            readback_worker: None,
            result_ready: false,
            result_w: 0,
            result_h: 0,
            rgb_buf: Vec::new(),
            composite_render_target: None,
        }
    }

    /// Requests the frame's RGBA8 pixels at the given size. Read the completed
    /// buffer through `rgb_buf()` after polling completion.
    pub fn request_rgba8(&mut self, w: u32, h: u32) {
        self.requested = true;
        self.capture_w = w;
        self.capture_h = h;
    }

    pub fn cancel_request(&mut self) {
        self.requested = false;
    }

    pub fn request_composite_render_target(&mut self, source_render_target: usize) {
        self.composite_render_target = Some(source_render_target);
    }

    pub fn requested_composite_render_target(&self) -> Option<usize> {
        self.composite_render_target
    }

    pub fn take_composite_render_target(&mut self) -> Option<usize> {
        self.composite_render_target.take()
    }

    pub fn release_buffers(&mut self) {
        self.requested = false;
        self.composite_render_target = None;
        self.capture_texture = None;
        self.capture_view = None;
        self.capture_tex_w = 0;
        self.capture_tex_h = 0;
        self.source_copy = None;
        self.source_copy_view = None;
        self.source_copy_w = 0;
        self.source_copy_h = 0;
        self.rgb_buf = Vec::new();
        for slot in &mut self.slots {
            if slot.pending && slot.buffer.is_none() {
                // A worker still owns this mapping. Keep the slot occupied
                // until it returns, including across an immediate restart.
                slot.map_signal.store(MAP_CANCELLED, Ordering::Release);
            } else {
                *slot = StagingSlot::new();
            }
        }
        self.write_idx = 0;
        self.needs_map = None;
        self.result_ready = false;
        self.buffers_released = true;
    }

    pub(crate) fn take_buffers_released(&mut self) -> bool {
        std::mem::take(&mut self.buffers_released)
    }

    /// Returns `true` if a capture was requested this frame.
    pub fn is_requested(&self) -> bool {
        self.requested
    }

    pub fn metrics(&self) -> ScreenCaptureMetrics {
        let mut metrics = self.metrics;
        metrics.gpu_bytes = self
            .slots
            .iter()
            .filter_map(|slot| slot.buffer.as_ref())
            .map(Buffer::size)
            .sum::<u64>()
            + metrics.worker_bytes
            + self.capture_tex_w as u64 * self.capture_tex_h as u64 * 4
            + self.source_copy_w as u64 * self.source_copy_h as u64 * 4;
        metrics.cpu_buffer_bytes = self.rgb_buf.capacity() as u64
            + self
                .slots
                .iter()
                .map(|slot| slot.rgb_buf.capacity() as u64)
                .sum::<u64>();
        metrics
    }

    /// Returns `true` while a slot still needs a GPU mapping callback.
    /// Ready mappings and CPU worker jobs do not need device polling.
    pub fn readback_in_flight(&self) -> bool {
        self.slots.iter().any(|s| {
            s.pending && s.buffer.is_some() && s.map_signal.load(Ordering::Acquire) == MAP_PENDING
        })
    }

    // -- pipeline / resource setup (lazy) --------------------------------

    /// Whether every pipeline a capture of a `format` surface turned `turn` draws with is compiled.
    pub(crate) fn pipelines_ready(&self, format: TextureFormat, turn: SurfaceTransform) -> bool {
        self.blit_pipeline.is_some()
            && self.present_pipeline.is_some()
            && self.present_pipeline_key == Some((format, turn))
    }

    /// Starts compiling the pipelines a capture of a `format` surface turned `turn` draws with, unless they are
    /// compiled or compiling: on a worker thread when `on_thread`, otherwise a family a frame through
    /// [`Self::advance_pipeline_build`]. A capture compiling them on the frame that asks would hold that frame
    /// for as long as the driver takes, behind every other compile it has queued.
    pub(crate) fn build_pipelines(
        &mut self,
        device: &Device,
        format: TextureFormat,
        turn: SurfaceTransform,
        on_thread: bool,
    ) {
        let key = (format, turn);
        if self.pipelines_ready(format, turn) || self.pipeline_build.building() == Some(key) {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if on_thread {
            let device = device.clone();
            let blit_layout = self.blit_bind_group_layout.clone();
            let (sender, finished) = std::sync::mpsc::channel();
            std::thread::Builder::new()
                .name("egor capture pipelines".to_owned())
                .spawn(move || {
                    let blits = blit_layout
                        .is_none()
                        .then(|| Self::build_blit_pipelines(&device));
                    let layout = blits
                        .as_ref()
                        .map(|blits| &blits.bind_group_layout)
                        .or(blit_layout.as_ref())
                        .expect("blit layout");
                    let present = Self::build_present_pipeline(&device, layout, format, turn);
                    let _ = sender.send(BuiltCapturePipelines {
                        blits,
                        present,
                        key,
                    });
                })
                .expect("spawn the capture pipeline thread");
            self.pipeline_build = PipelineBuild::Thread(key, finished);
            return;
        }
        let _ = (device, on_thread);
        self.pipeline_build = PipelineBuild::Frames(key);
    }

    /// Takes in the capture pipelines a worker thread finished, or compiles the next family of them on this
    /// thread.
    pub(crate) fn advance_pipeline_build(&mut self, device: &Device) {
        match std::mem::replace(&mut self.pipeline_build, PipelineBuild::Idle) {
            PipelineBuild::Idle => {}
            PipelineBuild::Frames((format, turn)) => {
                if self.blit_pipeline.is_none() {
                    self.ensure_pipeline(device);
                } else {
                    self.ensure_present_pipeline(device, format, turn);
                }
                if !self.pipelines_ready(format, turn) {
                    self.pipeline_build = PipelineBuild::Frames((format, turn));
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            PipelineBuild::Thread(key, finished) => match finished.try_recv() {
                Ok(built) => {
                    // A present pipeline only binds groups made from the layout it was built with.
                    if let Some(blits) = built.blits {
                        self.install_blits(blits);
                    }
                    self.present_pipeline = Some(built.present);
                    self.present_pipeline_key = Some(built.key);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    self.pipeline_build = PipelineBuild::Thread(key, finished);
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    panic!("[egor] compiling the capture pipelines failed")
                }
            },
        }
    }

    fn ensure_pipeline(&mut self, device: &Device) {
        if self.blit_pipeline.is_none() {
            self.install_blits(Self::build_blit_pipelines(device));
        }
    }

    fn install_blits(&mut self, blits: BlitPipelines) {
        self.blit_pipeline = Some(blits.plain);
        self.blit_encode_pipeline = Some(blits.encode);
        self.blit_sampler = Some(blits.sampler);
        self.blit_bind_group_layout = Some(blits.bind_group_layout);
    }

    fn build_blit_pipelines(device: &Device) -> BlitPipelines {
        let shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: None,
            source: egor_render::wgpu::ShaderSource::Wgsl(BLIT_SHADER_WGSL.into()),
        });

        let encode_shader =
            device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
                label: None,
                source: egor_render::wgpu::ShaderSource::Wgsl(BLIT_ENCODE_SHADER_WGSL.into()),
            });

        let bind_group_layout =
            device.create_bind_group_layout(&egor_render::wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    egor_render::wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: egor_render::wgpu::ShaderStages::FRAGMENT,
                        ty: egor_render::wgpu::BindingType::Texture {
                            sample_type: egor_render::wgpu::TextureSampleType::Float {
                                filterable: true,
                            },
                            view_dimension: egor_render::wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    egor_render::wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: egor_render::wgpu::ShaderStages::FRAGMENT,
                        ty: egor_render::wgpu::BindingType::Sampler(
                            egor_render::wgpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                ],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&egor_render::wgpu::PipelineLayoutDescriptor {
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

        let pipeline = make_pipeline(None, &shader, TextureFormat::Rgba8Unorm);
        let encode_pipeline = make_pipeline(None, &encode_shader, TextureFormat::Rgba8Unorm);

        let sampler = device.create_sampler(&egor_render::wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: egor_render::wgpu::AddressMode::ClampToEdge,
            address_mode_v: egor_render::wgpu::AddressMode::ClampToEdge,
            mag_filter: egor_render::wgpu::FilterMode::Nearest,
            min_filter: egor_render::wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        BlitPipelines {
            plain: pipeline,
            encode: encode_pipeline,
            sampler,
            bind_group_layout,
        }
    }

    fn ensure_capture_texture(&mut self, device: &Device, w: u32, h: u32) {
        if self.capture_tex_w == w && self.capture_tex_h == h && self.capture_texture.is_some() {
            return;
        }

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
            format: TextureFormat::Rgba8Unorm,
            usage: egor_render::wgpu::TextureUsages::RENDER_ATTACHMENT
                | egor_render::wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        self.capture_view = Some(texture.create_view(&Default::default()));
        self.capture_texture = Some(texture);
        self.capture_tex_w = w;
        self.capture_tex_h = h;
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
            usage: egor_render::wgpu::TextureUsages::TEXTURE_BINDING
                | egor_render::wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.source_copy_view = Some(tex.create_view(&Default::default()));
        self.source_copy = Some(tex);
        self.source_copy_w = w;
        self.source_copy_h = h;
    }

    fn prepare_capture(&mut self) -> Option<PreparedCapture> {
        self.requested = false;

        let cap_w = self.capture_w.max(1);
        let cap_h = self.capture_h.max(1);

        // -- Ring-buffer slot availability check --------------------------
        // Done BEFORE the blit so we skip ALL GPU work when the ring is full.
        let idx = self.write_idx;
        if self.slots[idx].pending {
            let status = self.slots[idx].map_signal.load(Ordering::Acquire);
            if status == MAP_READY {
                self.metrics.skipped_requests += 1;
                // Capture submission runs on the frame path. Do not consume
                // mapped readbacks here; the poll path/worker owns that work.
                return None;
            } else if status == MAP_FAILED {
                self.slots[idx].pending = false;
            } else {
                // Ring full — GPU hasn't finished this slot yet.
                // Skip capture entirely; no blit, no copy, no allocation.
                self.metrics.skipped_requests += 1;
                return None;
            }
        }

        Some(PreparedCapture {
            slot_idx: idx,
            cap_w,
            cap_h,
        })
    }

    fn source_bind_group(
        &self,
        device: &Device,
        source_view: &egor_render::wgpu::TextureView,
    ) -> egor_render::wgpu::BindGroup {
        let bind_group_layout = self.blit_bind_group_layout.as_ref().expect("pipeline init");
        let sampler = self.blit_sampler.as_ref().expect("pipeline init");

        device.create_bind_group(&egor_render::wgpu::BindGroupDescriptor {
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
        })
    }

    fn capture_prepared_bind_group(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        prepared: PreparedCapture,
        encode_srgb: bool,
        bind_group: &egor_render::wgpu::BindGroup,
    ) {
        let PreparedCapture {
            slot_idx,
            cap_w,
            cap_h,
        } = prepared;
        // Ensure GPU resources exist
        self.ensure_pipeline(device);
        self.ensure_capture_texture(device, cap_w, cap_h);

        let pipeline = if encode_srgb {
            self.blit_encode_pipeline.as_ref()
        } else {
            self.blit_pipeline.as_ref()
        }
        .expect("pipeline init");
        let capture_view = self.capture_view.as_ref().expect("capture texture init");

        // Blit render pass: draw fullscreen triangle sampling the source.
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
            rpass.set_bind_group(0, bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        // Copy capture texture → staging buffer for CPU readback
        let slot = &mut self.slots[slot_idx];
        let unpadded_row = cap_w * 4;
        let align = egor_render::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (unpadded_row + align - 1) / align * align;
        let buffer_size = (padded_row * cap_h) as u64;

        // Reuse the staging buffer if dimensions haven't changed
        let needs_new =
            slot.buffer.is_none() || slot.buf_size != buffer_size || slot.row_pitch != padded_row;

        self.metrics.readback_bytes += buffer_size;
        if needs_new {
            self.metrics.staging_allocations += 1;
            self.metrics.staging_allocated_bytes += buffer_size;
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

        // Store the size on the slot so try_complete knows how to decode
        // regardless of whether the request changed since then.
        let slot = &mut self.slots[slot_idx];
        slot.cap_w = cap_w;
        slot.cap_h = cap_h;
        slot.pending = true;

        self.needs_map = Some(slot_idx);
        self.write_idx = (self.write_idx + 1) % SLOT_COUNT;
    }

    // -- capture entry point (called by app.rs after rendering) -----------

    /// Blit-downsample the backbuffer into a small capture texture, then
    /// encode a `copy_texture_to_buffer` for async readback.
    ///
    /// `source` is the backbuffer `Texture` (must have `COPY_SRC` usage).
    /// The encoder must be the same one that will be submitted this frame.
    pub fn capture_from_texture(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        source: &Texture,
    ) {
        let Some(prepared) = self.prepare_capture() else {
            return;
        };

        self.ensure_pipeline(device);

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

        let source_view = self.source_copy_view.as_ref().expect("source copy init");
        let bind_group = self.source_bind_group(device, source_view);
        self.capture_prepared_bind_group(
            device,
            encoder,
            prepared,
            source.format().is_srgb(),
            &bind_group,
        );
    }

    /// Capture from an already bindable render target. Used on surfaces that
    /// cannot be configured with `COPY_SRC`; avoids a full-resolution texture
    /// copy before the downsample pass.
    pub fn capture_from_sampled_view(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        source_view: &egor_render::wgpu::TextureView,
        encode_srgb: bool,
    ) {
        let Some(prepared) = self.prepare_capture() else {
            return;
        };

        self.ensure_pipeline(device);
        let bind_group = self.source_bind_group(device, source_view);
        self.capture_prepared_bind_group(device, encoder, prepared, encode_srgb, &bind_group);
    }

    fn ensure_present_pipeline(
        &mut self,
        device: &Device,
        format: TextureFormat,
        turn: SurfaceTransform,
    ) {
        if self.present_pipeline.is_some() && self.present_pipeline_key == Some((format, turn)) {
            return;
        }

        self.ensure_pipeline(device);
        let bind_group_layout = self.blit_bind_group_layout.as_ref().expect("pipeline init");
        self.present_pipeline = Some(Self::build_present_pipeline(
            device,
            bind_group_layout,
            format,
            turn,
        ));
        self.present_pipeline_key = Some((format, turn));
    }

    fn build_present_pipeline(
        device: &Device,
        bind_group_layout: &egor_render::wgpu::BindGroupLayout,
        format: TextureFormat,
        turn: SurfaceTransform,
    ) -> egor_render::wgpu::RenderPipeline {
        let shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: None,
            source: egor_render::wgpu::ShaderSource::Wgsl(BLIT_SHADER_WGSL.into()),
        });
        let turn_constants = blit_turn_constants(turn);
        let pipeline_layout =
            device.create_pipeline_layout(&egor_render::wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(bind_group_layout)],
                immediate_size: 0,
            });

        device.create_render_pipeline(
            &egor_render::wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: egor_render::wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: blit_turn_options(&turn_constants),
                },
                fragment: Some(egor_render::wgpu::FragmentState {
                    module: &shader,
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
            },
        )
    }

    fn ensure_composite_pipeline(
        &mut self,
        device: &Device,
        format: TextureFormat,
        turn: SurfaceTransform,
    ) {
        if self.composite_pipeline.is_some() && self.composite_pipeline_key == Some((format, turn))
        {
            return;
        }

        self.ensure_pipeline(device);

        let shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: None,
            source: egor_render::wgpu::ShaderSource::Wgsl(BLIT_SHADER_WGSL.into()),
        });
        let bind_group_layout = self.blit_bind_group_layout.as_ref().expect("pipeline init");
        let turn_constants = blit_turn_constants(turn);
        let pipeline_layout =
            device.create_pipeline_layout(&egor_render::wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(bind_group_layout)],
                immediate_size: 0,
            });

        self.composite_pipeline = Some(device.create_render_pipeline(
            &egor_render::wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: egor_render::wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: blit_turn_options(&turn_constants),
                },
                fragment: Some(egor_render::wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(egor_render::wgpu::ColorTargetState {
                        format,
                        blend: Some(egor_render::wgpu::BlendState {
                            color: egor_render::wgpu::BlendComponent {
                                src_factor: egor_render::wgpu::BlendFactor::One,
                                dst_factor: egor_render::wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: egor_render::wgpu::BlendOperation::Add,
                            },
                            alpha: egor_render::wgpu::BlendComponent {
                                src_factor: egor_render::wgpu::BlendFactor::One,
                                dst_factor: egor_render::wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: egor_render::wgpu::BlendOperation::Add,
                            },
                        }),
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
            },
        ));
        self.composite_pipeline_key = Some((format, turn));
    }

    /// Present an offscreen frame source into the actual surface with a
    /// fullscreen triangle. This is a render pass, not a texture copy, so it
    /// works on GLES/WebGL surfaces that only support color attachment usage.
    pub fn present_sampled_view(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        source_view: &egor_render::wgpu::TextureView,
        dest_view: &egor_render::wgpu::TextureView,
        dest_format: TextureFormat,
        turn: SurfaceTransform,
    ) {
        self.ensure_present_pipeline(device, dest_format, turn);
        let bind_group = self.source_bind_group(device, source_view);
        let pipeline = self.present_pipeline.as_ref().expect("present pipeline");

        let mut rpass = encoder.begin_render_pass(&egor_render::wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(egor_render::wgpu::RenderPassColorAttachment {
                view: dest_view,
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

    pub fn composite_sampled_view(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        source_view: &egor_render::wgpu::TextureView,
        dest_view: &egor_render::wgpu::TextureView,
        dest_format: TextureFormat,
        turn: SurfaceTransform,
    ) {
        self.ensure_composite_pipeline(device, dest_format, turn);
        let bind_group = self.source_bind_group(device, source_view);
        let pipeline = self
            .composite_pipeline
            .as_ref()
            .expect("composite pipeline");

        let mut rpass = encoder.begin_render_pass(&egor_render::wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(egor_render::wgpu::RenderPassColorAttachment {
                view: dest_view,
                resolve_target: None,
                ops: egor_render::wgpu::Operations {
                    load: egor_render::wgpu::LoadOp::Load,
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
        buffer
            .slice(..)
            .map_async(egor_render::wgpu::MapMode::Read, move |result| {
                signal.store(
                    if result.is_ok() {
                        MAP_READY
                    } else {
                        MAP_FAILED
                    },
                    Ordering::Release,
                );
            });
    }

    // -- readback polling ------------------------------------------------

    #[cfg(target_os = "windows")]
    pub(crate) fn dispatch_submitted_readbacks(&mut self) {
        // Queue submission drives callbacks for older maps. Start conversion
        // before present and the frame timer, so it can finish by the next
        // callback without a second device poll or another frame of latency.
        for i in 0..SLOT_COUNT {
            let idx = (self.write_idx + i) % SLOT_COUNT;
            let slot = &self.slots[idx];
            if !slot.pending || slot.buffer.is_none() {
                continue;
            }
            match slot.map_signal.load(Ordering::Acquire) {
                MAP_PENDING => break,
                MAP_READY => self.dispatch_slot_to_worker(idx),
                _ => {}
            }
        }
    }

    /// Consume the oldest completed ring-buffer slot: copy its rows out
    /// without the row padding.
    #[cfg(target_arch = "wasm32")]
    fn complete_slot(&mut self, idx: usize) {
        let complete_start = Instant::now();
        let cap_w = self.slots[idx].cap_w;
        let cap_h = self.slots[idx].cap_h;
        let row_pitch = self.slots[idx].row_pitch as usize;

        let buffer = match self.slots[idx].buffer.take() {
            Some(b) => b,
            None => {
                self.slots[idx].pending = false;
                return;
            }
        };

        decode_readback_into(&buffer, &mut self.rgb_buf, cap_w, cap_h, row_pitch);
        let conversion_us = complete_start.elapsed().as_micros() as u64;
        buffer.unmap();
        let complete_us = complete_start.elapsed().as_micros() as u64;
        self.metrics.completed_frames += 1;
        self.metrics.decode_us += complete_us;
        self.metrics.conversion_us += conversion_us;
        self.metrics.unmap_us += complete_us - conversion_us;

        self.slots[idx].buffer = Some(buffer);
        self.slots[idx].pending = false;
        self.result_ready = true;
        self.result_w = cap_w as u16;
        self.result_h = cap_h as u16;
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn dispatch_slot_to_worker(&mut self, idx: usize) {
        let job = {
            let slot = &mut self.slots[idx];
            let Some(buffer) = slot.buffer.take() else {
                slot.pending = false;
                return;
            };

            ReadbackJob {
                slot_idx: idx,
                map_signal: Arc::clone(&slot.map_signal),
                rgb_buf: std::mem::take(&mut slot.rgb_buf),
                buffer,
                buf_size: slot.buf_size,
                row_pitch: slot.row_pitch as usize,
                cap_w: slot.cap_w,
                cap_h: slot.cap_h,
            }
        };

        let job_bytes = job.buf_size;
        let worker = self.readback_worker.get_or_insert_with(ReadbackWorker::new);
        match worker.jobs.send(job) {
            Ok(()) => {
                self.metrics.worker_jobs += 1;
                self.metrics.worker_bytes += job_bytes;
                // Keep this slot occupied until collection. This bounds jobs
                // and staging buffers to SLOT_COUNT when conversion is slow.
            }
            Err(err) => {
                let job = err.0;
                decode_readback_into(
                    &job.buffer,
                    &mut self.rgb_buf,
                    job.cap_w,
                    job.cap_h,
                    job.row_pitch,
                );
                job.buffer.unmap();
                self.slots[idx].buffer = Some(job.buffer);
                self.slots[idx].pending = false;
                self.result_ready = true;
                self.result_w = job.cap_w as u16;
                self.result_h = job.cap_h as u16;
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn collect_worker_result(&mut self) {
        if self.result_ready {
            return;
        }

        let Some(worker) = self.readback_worker.as_ref() else {
            return;
        };
        let Ok(result) = worker.results.try_recv() else {
            return;
        };

        // Unmapping GL buffers acquires the rendering context. Keep that on
        // the render thread so a CPU conversion worker cannot steal/hold the
        // context while the frame thread is polling, submitting, or presenting.
        let unmap_started = Instant::now();
        {
            #[cfg(feature = "profiling")]
            profiling::scope!("readback_unmap");
            result.buffer.unmap();
        }
        let unmap_us = unmap_started.elapsed().as_micros() as u64;

        self.metrics.worker_jobs -= 1;
        self.metrics.worker_bytes -= result.buf_size;
        let slot = &mut self.slots[result.slot_idx];
        debug_assert!(slot.pending && slot.buffer.is_none());
        debug_assert_eq!(slot.buf_size, result.buf_size);
        debug_assert_eq!(slot.row_pitch, result.row_pitch);
        if slot.map_signal.load(Ordering::Acquire) == MAP_CANCELLED {
            // Never publish a cancelled frame or recycle its buffers after
            // release. Unmapping and destruction stay on the render thread.
            *slot = StagingSlot::new();
            return;
        }
        slot.buffer = Some(result.buffer);
        slot.pending = false;
        slot.rgb_buf = std::mem::replace(&mut self.rgb_buf, result.rgb_buf);
        self.metrics.completed_frames += 1;
        self.metrics.decode_us += result.complete_us as u64 + unmap_us;
        self.metrics.conversion_us += result.conversion_us as u64;
        self.metrics.unmap_us += unmap_us;
        self.result_ready = true;
        self.result_w = result.cap_w as u16;
        self.result_h = result.cap_h as u16;
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn collect_cancelled_readbacks(&mut self) {
        // A stopped client need not poll captures. The single worker returns
        // jobs in order, so cancelled results precede any restarted capture.
        for _ in 0..SLOT_COUNT {
            if !self.slots.iter().any(|slot| {
                slot.pending && slot.map_signal.load(Ordering::Acquire) == MAP_CANCELLED
            }) {
                break;
            }
            let jobs = self.metrics.worker_jobs;
            self.collect_worker_result();
            if self.metrics.worker_jobs == jobs {
                break;
            }
        }
    }

    /// Poll for a completed readback. Returns `Some((width, height))` when
    /// pixel data is available in [`Self::rgb_buf`].
    ///
    /// Non-blocking: iterates ring-buffer slots oldest-first and consumes
    /// the first whose `map_async` callback has fired. Driven by the game
    /// loop's device polling or queue submission.
    pub fn try_complete(&mut self) -> Option<(u16, u16)> {
        #[cfg(not(target_arch = "wasm32"))]
        self.collect_worker_result();

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
            if self.slots[idx].buffer.is_none() {
                continue;
            } // Owned by the worker.
            let status = self.slots[idx].map_signal.load(Ordering::Acquire);
            if status == MAP_PENDING {
                break; // Preserve frame order even if callbacks arrive out of order.
            }
            if status == MAP_FAILED {
                eprintln!("[ScreenCapture] buffer map failed on slot {idx}");
                self.slots[idx].pending = false;
                continue;
            }
            // MAP_READY
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.dispatch_slot_to_worker(idx);
                self.collect_worker_result();
                if self.result_ready {
                    self.result_ready = false;
                    return Some((self.result_w, self.result_h));
                }
                continue;
            }

            #[cfg(target_arch = "wasm32")]
            self.complete_slot(idx);
            #[cfg(target_arch = "wasm32")]
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
    target_format: TextureFormat,
    target_size: (u32, u32),
    current_shader: Option<usize>,
    render_targets: &'a mut RenderTargetStore,
    screen_capture: &'a mut ScreenCaptureState,
    offscreen_batches: &'a mut Vec<PrimitiveBatch>,
}

impl<'a> Graphics<'a> {
    /// Create `Graphics` with [`Renderer`] & `TextureFormat`
    pub fn new(
        renderer: &'a mut Renderer,
        batch: &'a mut PrimitiveBatch,
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
        self.renderer
            .create_offscreen_target(width, height, self.target_format)
    }

    /// Return information about the active GPU adapter/backend.
    pub fn adapter_info(&self) -> AdapterInfo {
        self.renderer.adapter_info()
    }

    /// Flags used when creating this renderer's wgpu instance.
    pub fn instance_flags(&self) -> egor_render::wgpu::InstanceFlags {
        self.renderer.instance_flags()
    }

    /// Render to an offscreen target
    pub fn render_offscreen(
        &mut self,
        target: &mut OffscreenTarget,
        render_fn: impl FnMut(&mut Graphics),
    ) {
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

        let mut encoder = self
            .renderer
            .device()
            .create_command_encoder(&Default::default());

        {
            let mut r_pass = self.renderer.begin_render_pass_with_depth(
                &mut encoder,
                target.view(),
                target.offscreen_depth_view(),
                true,
            );

            let mut cur_tex: Option<usize> = None;
            let mut cur_shd: Option<usize> = None;
            let mut cur_replace_blend = false;
            let mut cur_cam_offset = u32::MAX;
            let mut quad_bound = false;
            let mut cur_scissor = None;

            if let Some(first) = geometry.first() {
                self.renderer.bind_pass_state(
                    &mut r_pass,
                    first.texture_id,
                    first.shader_id,
                    first.replace_blend,
                );
                cur_tex = first.texture_id;
                cur_shd = first.shader_id;
                cur_replace_blend = first.replace_blend;
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
                    entry.replace_blend,
                    0,
                    &mut cur_tex,
                    &mut cur_shd,
                    &mut cur_replace_blend,
                    &mut cur_cam_offset,
                    &mut quad_bound,
                );
            }
        }

        let _ = self.renderer.queue().submit(Some(encoder.finish()));

        offscreen_batch.recycle(geometry);
        self.offscreen_batches.push(offscreen_batch);
    }

    /// Use an offscreen target as a texture. Each call adds a texture; free it with [`Self::free_texture`].
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
        self.renderer.upload_camera_matrix(
            self.camera
                .view_proj((w as f32, h as f32).into())
                .to_cols_array_2d(),
        );
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
    pub fn push_sprite(
        &mut self,
        tex_id: usize,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        uvs: [f32; 4],
        color: [f32; 4],
    ) {
        self.push_sprite_layer(tex_id, x, y, w, h, uvs, color, 0);
    }

    #[inline(always)]
    pub fn push_sprite_layer(
        &mut self,
        tex_id: usize,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        uvs: [f32; 4],
        color: [f32; 4],
        layer: u32,
    ) {
        self.batch.push_instance(
            egor_render::instance::Instance::new(
                [w, 0.0, 0.0, h],
                [x + w * 0.5, y + h * 0.5, self.batch.draw_depth()],
                color,
                uvs,
            )
            .with_texture_layer(layer),
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
        self.push_tile_layer(x, y, w, h, depth, uvs, 0);
    }

    #[inline(always)]
    pub fn push_tile_layer(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        depth: f32,
        uvs: [f32; 4],
        layer: u32,
    ) {
        self.batch.push_instance_unchecked(
            egor_render::instance::Instance::new(
                [w, 0.0, 0.0, h],
                [x + w * 0.5, y + h * 0.5, depth],
                [1.0, 1.0, 1.0, 1.0],
                uvs,
            )
            .with_texture_layer(layer),
        );
    }

    /// Push a colored sprite instance into the current batch WITHOUT any
    /// batch-key comparisons. The caller MUST call [`ensure_tile_batch`]
    /// first whenever the texture changes. Uses the current draw_depth.
    #[inline(always)]
    pub fn push_sprite_unchecked(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        uvs: [f32; 4],
        color: [f32; 4],
    ) {
        self.push_sprite_unchecked_layer(x, y, w, h, uvs, color, 0);
    }

    #[inline(always)]
    pub fn push_sprite_unchecked_layer(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        uvs: [f32; 4],
        color: [f32; 4],
        layer: u32,
    ) {
        self.batch.push_instance_unchecked(
            egor_render::instance::Instance::new(
                [w, 0.0, 0.0, h],
                [x + w * 0.5, y + h * 0.5, self.batch.draw_depth()],
                color,
                uvs,
            )
            .with_texture_layer(layer),
        );
    }

    /// Push an outlined glyph through the default sprite pipeline. The UV rect
    /// must include the one-texel outline border; `outline_color.a > 0` enables
    /// the outline path in the default shader.
    #[inline(always)]
    pub fn push_outlined_sprite_unchecked(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        uvs: [f32; 4],
        color: [f32; 4],
        outline_color: [f32; 4],
    ) {
        self.push_outlined_sprite_unchecked_layer(x, y, w, h, uvs, color, outline_color, 0);
    }

    #[inline(always)]
    pub fn push_outlined_sprite_unchecked_layer(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        uvs: [f32; 4],
        color: [f32; 4],
        outline_color: [f32; 4],
        layer: u32,
    ) {
        self.batch.push_instance_unchecked(
            egor_render::instance::Instance::new_outlined(
                [w, 0.0, 0.0, h],
                [x + w * 0.5, y + h * 0.5, self.batch.draw_depth()],
                color,
                uvs,
                outline_color,
            )
            .with_texture_layer(layer),
        );
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
        if let Some((v_slice, i_slice, base)) =
            self.batch
                .allocate(vert_count, idx_count, None, self.current_shader)
        {
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
    /// Cache a linear sampler binding without changing the source texture's filtering.
    pub fn linear_texture_variant(&mut self, id: usize) -> usize {
        self.renderer.linear_texture_variant(id)
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
    pub fn add_texture_array_raw(
        &mut self,
        w: u32,
        h: u32,
        layers: u32,
        data: &[u8],
    ) -> Result<usize, String> {
        self.renderer.add_texture_array_raw(w, h, layers, data)
    }

    /// Like [`Self::add_texture_array_raw`], followed by `spare_layers` transparent layers
    /// that are filled later with [`Self::write_texture_region`].
    pub fn add_texture_array_raw_with_spare_layers(
        &mut self,
        w: u32,
        h: u32,
        layers: u32,
        spare_layers: u32,
        data: &[u8],
    ) -> Result<usize, String> {
        self.renderer
            .add_texture_array_raw_with_spare_layers(w, h, layers, spare_layers, data)
    }

    /// Starts making an empty `w`×`h` texture array of `layers`, off this thread where the backend allows; filled
    /// later with [`Self::write_texture_region`] once [`Self::take_empty_texture_array`] has its id.
    pub fn start_empty_texture_array(
        &mut self,
        w: u32,
        h: u32,
        layers: u32,
    ) -> Result<egor_render::EmptyTextureArray, String> {
        self.renderer.start_empty_texture_array(w, h, layers)
    }

    /// The texture array's id, once it is made.
    pub fn take_empty_texture_array(
        &mut self,
        array: &mut egor_render::EmptyTextureArray,
    ) -> Option<usize> {
        self.renderer.take_empty_texture_array(array)
    }

    /// Overwrite a `w`×`h` RGBA8 region of one layer of an existing texture.
    #[allow(clippy::too_many_arguments)]
    pub fn write_texture_region(
        &mut self,
        id: usize,
        layer: u32,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        data: &[u8],
    ) -> Result<(), String> {
        self.renderer
            .write_texture_region(id, layer, x, y, w, h, data)
    }

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
        self.renderer
            .add_shader_with_uniforms(wgsl_source, uniform_ids)
    }

    /// Load a custom shader that also samples `texture_count` textures of its own
    /// (see [`Renderer::add_shader_with_textures`]), set with [`Self::set_shader_texture`].
    pub fn load_shader_with_textures(
        &mut self,
        wgsl_source: &str,
        uniform_ids: &[usize],
        texture_count: usize,
    ) -> usize {
        self.renderer
            .add_shader_with_textures(wgsl_source, uniform_ids, texture_count)
    }

    /// Points texture `index` of a shader from [`Self::load_shader_with_textures`] at a texture,
    /// for every draw with the shader this frame.
    pub fn set_shader_texture(&mut self, shader_id: usize, index: usize, texture_id: usize) {
        self.renderer
            .set_shader_texture(shader_id, index, texture_id);
    }

    /// Execute drawing commands with a custom shader
    ///
    /// The shader is automatically reset to default after the closure drops
    pub fn with_shader(&mut self, shader_id: usize, mut render_fn: impl FnMut(&mut Self)) {
        self.renderer.wait_for_shader(shader_id);
        let previous_shader = self.current_shader;
        self.current_shader = Some(shader_id);
        render_fn(self);
        self.current_shader = previous_shader;
    }

    /// Directly set (or clear) the active shader for subsequent draw commands.
    pub fn set_current_shader(&mut self, shader_id: Option<usize>) {
        if let Some(id) = shader_id {
            self.renderer.wait_for_shader(id);
        }
        self.current_shader = shader_id;
    }

    /// Custom shader compiles finished, and started. On Vulkan, Metal and DX12 they run on worker threads, and
    /// drawing with a shader before it is done waits for it.
    pub fn shader_compiles(&self) -> (usize, usize) {
        self.renderer.shader_compiles()
    }

    /// Keeps the pipelines compiled so far for the next run (see `App::pipeline_cache_dir`).
    pub fn save_pipeline_cache(&self) {
        self.renderer.save_pipeline_cache();
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

    pub fn set_replace_blend(&mut self, replace_blend: bool) {
        self.batch.set_replace_blend(replace_blend);
    }

    // -- managed render targets -----------------------------------------

    /// Create a managed offscreen render target and return its store index.
    /// Also registers it as a drawable texture, returning `(store_id, egor_texture_id)`.
    pub fn create_managed_render_target(&mut self, width: u32, height: u32) -> (usize, usize) {
        self.create_managed_render_target_with_filter(
            width,
            height,
            egor_render::wgpu::FilterMode::Linear,
        )
    }

    /// Create a drawable render target with an explicit sampling filter.
    /// Use nearest-neighbor for pixel-art UI and linear for smoothly scaled imagery.
    pub fn create_managed_render_target_with_filter(
        &mut self,
        width: u32,
        height: u32,
        filter: egor_render::wgpu::FilterMode,
    ) -> (usize, usize) {
        let store_id =
            self.render_targets
                .create(self.renderer.device(), width, height, self.target_format);
        let tex_id = self
            .renderer
            .add_offscreen_texture_with_filter(self.render_targets.get_mut(store_id), filter);
        self.render_targets.textures[store_id] = Some(tex_id);
        (store_id, tex_id)
    }

    /// Resize a managed render target and rebind its texture, which keeps its egor texture id (returned).
    pub fn resize_managed_render_target(
        &mut self,
        store_id: usize,
        width: u32,
        height: u32,
    ) -> usize {
        self.resize_managed_render_target_with_filter(
            store_id,
            width,
            height,
            egor_render::wgpu::FilterMode::Linear,
        )
    }

    /// Resize and re-register a render target with an explicit sampling filter.
    pub fn resize_managed_render_target_with_filter(
        &mut self,
        store_id: usize,
        width: u32,
        height: u32,
        filter: egor_render::wgpu::FilterMode,
    ) -> usize {
        self.render_targets
            .resize(self.renderer.device(), store_id, width, height);
        match self.render_targets.textures[store_id] {
            Some(tex_id) => {
                self.renderer.replace_offscreen_texture_with_filter(
                    tex_id,
                    self.render_targets.get(store_id),
                    filter,
                );
                tex_id
            }
            None => {
                let tex_id = self.renderer.add_offscreen_texture_with_filter(
                    self.render_targets.get_mut(store_id),
                    filter,
                );
                self.render_targets.textures[store_id] = Some(tex_id);
                tex_id
            }
        }
    }

    /// Frees a texture once this frame is drawn; a later texture reuses its id. Don't use the id after this.
    pub fn free_texture(&mut self, id: usize) {
        self.render_targets
            .pending_frees
            .push(PendingFree::Texture(id));
    }

    /// Frees a managed render target and its texture once this frame is drawn, like [`Self::free_texture`].
    pub fn free_managed_render_target(&mut self, store_id: usize) {
        self.render_targets
            .pending_frees
            .push(PendingFree::RenderTarget(store_id));
    }

    /// Frees a uniform once this frame is drawn, like [`Self::free_texture`].
    pub fn free_uniform(&mut self, id: usize) {
        self.render_targets
            .pending_frees
            .push(PendingFree::Uniform(id));
    }

    /// Frees a custom shader once this frame is drawn, like [`Self::free_texture`]. Its uniforms are separate.
    pub fn free_shader(&mut self, id: usize) {
        self.render_targets
            .pending_frees
            .push(PendingFree::Shader(id));
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

    /// Capture this frame's lossless RGBA8 pixels at the given size.
    pub fn request_screen_capture_rgba8(&mut self, w: u32, h: u32) {
        self.screen_capture.request_rgba8(w, h);
    }

    pub fn composite_render_target_to_backbuffer(&mut self, source_render_target: usize) {
        self.screen_capture
            .request_composite_render_target(source_render_target);
    }

    pub fn release_screen_capture_resources(&mut self) {
        self.screen_capture.release_buffers();
    }

    /// Poll for a completed screen capture result.
    /// Returns `Some((width, height))` when pixel data is available.
    pub fn poll_screen_capture(&mut self) -> Option<(u16, u16)> {
        self.screen_capture.try_complete()
    }

    /// Access the completed screen capture's RGBA8 pixels.
    pub fn screen_capture_rgb_buf(&self) -> &[u8] {
        self.screen_capture.rgb_buf()
    }

    pub fn screen_capture_metrics(&self) -> ScreenCaptureMetrics {
        self.screen_capture.metrics()
    }
}

#[cfg(test)]
mod capture_format_tests {
    use super::*;

    #[test]
    fn screenshot_readback_preserves_low_bits_alpha_and_strips_row_padding() {
        let data = [
            9, 17, 25, 253, 10, 18, 26, 254, 0xee, 0xee, 0xee, 0xee, 11, 19, 27, 255, 12, 20, 28,
            0, 0xee, 0xee, 0xee, 0xee,
        ];
        let mut output = Vec::new();
        decode_readback_bytes(&data, &mut output, 2, 2, 12);
        assert_eq!(
            output,
            [
                9, 17, 25, 253, 10, 18, 26, 254, 11, 19, 27, 255, 12, 20, 28, 0
            ]
        );
    }
}
