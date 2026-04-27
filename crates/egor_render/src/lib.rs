pub mod batch;
pub mod frame;
pub mod instance;
mod pipeline;
pub mod target;
mod texture;
mod uniforms;
pub mod vertex;

pub use wgpu::{
    self, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Extent3d, MemoryHints, Queue, RenderPass, Texture, TextureFormat,
};

use wgpu::{
    Adapter, BindGroup, BindGroupDescriptor, BindGroupEntry, Color, DepthStencilState, DeviceDescriptor, Instance, InstanceDescriptor,
    LoadOp, Operations, RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor, RequestAdapterOptions, StoreOp,
    SurfaceTarget, TextureView, WindowHandle,
    util::{BufferInitDescriptor, DeviceExt, new_instance_with_webgpu_detection},
};

use crate::{
    batch::GeometryBatch,
    frame::Frame,
    pipeline::Pipelines,
    target::{OffscreenTarget, RenderTarget},
    texture::Textures,
    uniforms::{CameraUniform, Uniforms},
    vertex::{QUAD_INDICES, QUAD_VERTICES},
};

pub(crate) struct Gpu {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

const REQUIRED_MAX_TEXTURE_DIMENSION_2D: u32 = 4096;

/// Low-level GPU renderer built on `wgpu`
///
/// Handles rendering pipelines, surface configuration, resources (textures, buffers), & drawing
pub struct Renderer {
    gpu: Gpu,
    pipelines: Pipelines,
    quad_vertex_buffer: Buffer,
    quad_index_buffer: Buffer,
    dummy_instance_buffer: Buffer,
    camera_bind_group: BindGroup,
    camera_buffer: Buffer,
    camera_slot_stride: u32,
    camera_slot_count: u32,
    surface_format: TextureFormat,
    uniforms: Uniforms,
    textures: Textures,
    clear_color: Color,
    depth_texture: wgpu::Texture,
    depth_view: TextureView,
    depth_size: (u32, u32),
    shared_instance_buffer: Option<Buffer>,
    last_camera: [[[f32; 4]; 4]; 8],
}

impl Renderer {
    /// Creates a renderer & initializes GPU state using the window's surface
    ///
    /// Sets up wgpu, pipelines, default texture & camera resources
    pub async fn new(window: impl Into<SurfaceTarget<'static>> + WindowHandle, memory_hints: &MemoryHints) -> Self {
        let mut desc = InstanceDescriptor::new_without_display_handle_from_env();
        desc.flags.remove(wgpu::InstanceFlags::VALIDATION);
        desc.flags.remove(wgpu::InstanceFlags::DEBUG);
        let instance = new_instance_with_webgpu_detection(desc).await;
        let surface = instance.create_surface(window).unwrap();
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                // Required for WebGL to prevent selecting a non-presentable device
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .unwrap();
        let info = adapter.get_info();
        log::info!(
            "[egor] wgpu backend: {:?} | adapter: {} | driver: {}",
            info.backend,
            info.name,
            info.driver
        );
        let adapter_limits = adapter.limits();
        assert!(
            adapter_limits.max_texture_dimension_2d >= REQUIRED_MAX_TEXTURE_DIMENSION_2D,
            "[egor] adapter max_texture_dimension_2d {} is below required {}",
            adapter_limits.max_texture_dimension_2d,
            REQUIRED_MAX_TEXTURE_DIMENSION_2D
        );
        #[cfg(target_arch = "wasm32")]
        let mut required_limits = wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter_limits.clone());
        #[cfg(not(target_arch = "wasm32"))]
        let mut required_limits = adapter_limits;
        required_limits.max_texture_dimension_2d = required_limits.max_texture_dimension_2d.max(REQUIRED_MAX_TEXTURE_DIMENSION_2D);
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                required_limits,
                memory_hints: memory_hints.clone(),
                ..Default::default()
            })
            .await
            .unwrap();

        let (_surface_config, surface_format, _) = target::surface_config(&surface, &adapter, 1, 1);
        let pipelines = Pipelines::new(&device, surface_format);

        let quad_vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Static Unit Quad VB"),
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: BufferUsages::VERTEX,
        });
        let quad_index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Static Unit Quad IB"),
            contents: bytemuck::cast_slice(&QUAD_INDICES),
            usage: BufferUsages::INDEX,
        });
        let dummy_instance_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Dummy Instance Buffer"),
            contents: bytemuck::bytes_of(&instance::Instance::identity()),
            usage: BufferUsages::VERTEX,
        });
        let min_alignment = device.limits().min_uniform_buffer_offset_alignment as u32;
        let camera_data_size = std::mem::size_of::<CameraUniform>() as u32;
        let camera_slot_stride = ((camera_data_size + min_alignment - 1) / min_alignment) * min_alignment;
        let camera_slot_count: u32 = 8;
        let camera_buffer_size = camera_slot_stride * camera_slot_count;

        let camera_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &vec![0u8; camera_buffer_size as usize],
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let camera_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipelines.camera_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &camera_buffer,
                    offset: 0,
                    size: Some(std::num::NonZeroU64::new(camera_data_size as u64).unwrap()),
                }),
            }],
        });

        let uniforms = Uniforms::new(&device);
        let textures = Textures::new(&device, &queue);

        let (depth_texture, depth_view) = Self::create_depth_texture(&device, 1, 1);

        Renderer {
            gpu: Gpu {
                instance,
                adapter,
                device,
                queue,
            },
            pipelines,
            quad_vertex_buffer,
            quad_index_buffer,
            dummy_instance_buffer,
            camera_bind_group,
            camera_buffer,
            camera_slot_stride,
            camera_slot_count,
            surface_format,
            uniforms,
            textures,
            clear_color: Color::BLACK,
            depth_texture,
            depth_view,
            depth_size: (1, 1),
            shared_instance_buffer: None,
            last_camera: [[[0.0; 4]; 4]; 8],
        }
    }

    pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

    pub fn create_depth_texture(device: &Device, width: u32, height: u32) -> (wgpu::Texture, TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        (texture, view)
    }

    /// Ensure the depth texture matches the given dimensions, recreating if necessary.
    pub fn ensure_depth_size(&mut self, width: u32, height: u32) {
        if self.depth_size == (width, height) {
            return;
        }
        let (tex, view) = Self::create_depth_texture(&self.gpu.device, width, height);
        self.depth_texture = tex;
        self.depth_view = view;
        self.depth_size = (width, height);
    }

    /// Returns a reference to the current depth texture view
    pub fn depth_view(&self) -> &TextureView {
        &self.depth_view
    }

    /// Returns a reference to the underlying wgpu `Instance`
    pub fn instance(&self) -> &Instance {
        &self.gpu.instance
    }
    /// Returns a reference to the underlying wgpu `Adapter`
    pub fn adapter(&self) -> &Adapter {
        &self.gpu.adapter
    }
    /// Returns a reference to the underlying wgpu `Device`
    pub fn device(&self) -> &Device {
        &self.gpu.device
    }
    /// Returns a reference to the underlying wgpu `Queue`
    pub fn queue(&self) -> &Queue {
        &self.gpu.queue
    }

    /// Sets the clear color for future render passes
    pub fn set_clear_color(&mut self, color: [f64; 4]) {
        self.clear_color = Color {
            r: color[0],
            g: color[1],
            b: color[2],
            a: color[3],
        };
    }

    /// Begins a frame with the given render target
    pub fn begin_frame(&mut self, target: &mut dyn RenderTarget) -> Option<Frame> {
        let (view, presentable) = target.acquire(&self.gpu.device)?;
        let encoder = self.gpu.device.create_command_encoder(&Default::default());
        Some(Frame {
            view,
            encoder,
            presentable,
        })
    }

    /// Ends the frame by submitting commands and presenting
    pub fn end_frame(&mut self, frame: Frame) {
        frame.finish(&self.gpu.queue);
    }

    /// Submit the command buffer to the GPU. Returns the presentable for separate present timing.
    pub fn submit_frame(&mut self, frame: Frame) -> Option<crate::frame::Presentable> {
        frame.submit(&self.gpu.queue)
    }

    /// Finish encoder without submitting — for timing encoder.finish() separately.
    pub fn finish_encoder(&mut self, frame: Frame) -> (wgpu::CommandBuffer, Option<crate::frame::Presentable>) {
        frame.finish_encoder()
    }

    /// Submit a pre-finished command buffer
    pub fn submit_commands(&self, commands: wgpu::CommandBuffer) {
        self.gpu.queue.submit(Some(commands));
    }

    /// Begins a render pass with the given encoder and target view.
    /// Clears the view (set by [`Self::set_clear_color`]) and the depth buffer.
    pub fn begin_render_pass<'a>(&'a self, encoder: &'a mut CommandEncoder, view: &'a TextureView) -> RenderPass<'a> {
        self.begin_render_pass_with_depth(encoder, view, &self.depth_view, true)
    }

    /// Begins a render pass that preserves existing content (LoadOp::Load).
    /// Used when splitting a frame into multiple passes (e.g. camera changes).
    pub fn begin_render_pass_load<'a>(&'a self, encoder: &'a mut CommandEncoder, view: &'a TextureView) -> RenderPass<'a> {
        self.begin_render_pass_load_with_depth(encoder, view, &self.depth_view)
    }

    /// Begins a render pass with an explicit depth view, clearing both color and depth.
    pub fn begin_render_pass_with_depth<'a>(
        &'a self,
        encoder: &'a mut CommandEncoder,
        view: &'a TextureView,
        depth_view: &'a TextureView,
        clear_depth: bool,
    ) -> RenderPass<'a> {
        encoder.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(self.clear_color),
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(Operations {
                    load: if clear_depth { LoadOp::Clear(1.0) } else { LoadOp::Load },
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        })
    }

    /// Begins a render pass with an explicit depth view, preserving both color and depth.
    pub fn begin_render_pass_load_with_depth<'a>(
        &'a self,
        encoder: &'a mut CommandEncoder,
        view: &'a TextureView,
        depth_view: &'a TextureView,
    ) -> RenderPass<'a> {
        encoder.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        })
    }

    /// Binds pipeline, texture, and shared quad buffers once for a render pass.
    /// Returns the previously bound (texture_id, shader_id) so the caller can
    /// track state and skip redundant calls between batches.
    pub fn bind_pass_state(&self, r_pass: &mut RenderPass<'_>, texture_id: Option<usize>, shader_id: Option<usize>) {
        let texture = self.textures.get(texture_id);
        texture.bind(r_pass, 0);

        let (pipeline, uniform_ids) = self.pipelines.resolve(shader_id);
        r_pass.set_pipeline(pipeline);

        for (i, &uid) in uniform_ids.iter().enumerate() {
            r_pass.set_bind_group((2 + i) as u32, self.uniforms.bind_group(uid), &[]);
        }

        r_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
        r_pass.set_index_buffer(self.quad_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
    }

    /// Draws a geometry batch. Call [`bind_pass_state`] first to set pipeline/texture.
    /// Only re-binds texture/pipeline/camera if they differ from the tracked state.
    /// `quad_bound` tracks whether the static quad VB/IB are currently bound.
    pub fn draw_batch<'a>(
        &'a self,
        r_pass: &mut RenderPass<'a>,
        batch: &mut GeometryBatch,
        texture_id: Option<usize>,
        shader_id: Option<usize>,
        camera_offset: u32,
        current_texture: &mut Option<usize>,
        current_shader: &mut Option<usize>,
        current_camera_offset: &mut u32,
        quad_bound: &mut bool,
    ) {
        if batch.is_empty() {
            return;
        }

        batch.upload(&self.gpu.device, &self.gpu.queue);

        if *current_texture != texture_id {
            let texture = self.textures.get(texture_id);
            texture.bind(r_pass, 0);
            *current_texture = texture_id;
        }

        if *current_shader != shader_id {
            let (pipeline, uniform_ids) = self.pipelines.resolve(shader_id);
            r_pass.set_pipeline(pipeline);
            for (i, &uid) in uniform_ids.iter().enumerate() {
                r_pass.set_bind_group((2 + i) as u32, self.uniforms.bind_group(uid), &[]);
            }
            *current_shader = shader_id;
        }

        if *current_camera_offset != camera_offset {
            r_pass.set_bind_group(1, &self.camera_bind_group, &[camera_offset]);
            *current_camera_offset = camera_offset;
        }

        batch.draw(
            r_pass,
            &self.quad_vertex_buffer,
            &self.quad_index_buffer,
            &self.dummy_instance_buffer,
            quad_bound,
            None,
        );
        batch.clear();
    }

    /// Like draw_batch but uses a shared instance buffer at the given byte offset.
    /// Skips per-batch instance upload entirely.
    pub fn draw_batch_shared<'a>(
        &'a self,
        r_pass: &mut RenderPass<'a>,
        batch: &mut GeometryBatch,
        texture_id: Option<usize>,
        shader_id: Option<usize>,
        camera_offset: u32,
        current_texture: &mut Option<usize>,
        current_shader: &mut Option<usize>,
        current_camera_offset: &mut u32,
        quad_bound: &mut bool,
        shared_buf: &'a Buffer,
        instance_byte_offset: u64,
    ) {
        if batch.is_empty() {
            return;
        }

        batch.upload_geometry_only(&self.gpu.device, &self.gpu.queue);

        if *current_texture != texture_id {
            let texture = self.textures.get(texture_id);
            texture.bind(r_pass, 0);
            *current_texture = texture_id;
        }

        if *current_shader != shader_id {
            let (pipeline, uniform_ids) = self.pipelines.resolve(shader_id);
            r_pass.set_pipeline(pipeline);
            for (i, &uid) in uniform_ids.iter().enumerate() {
                r_pass.set_bind_group((2 + i) as u32, self.uniforms.bind_group(uid), &[]);
            }
            *current_shader = shader_id;
        }

        if *current_camera_offset != camera_offset {
            r_pass.set_bind_group(1, &self.camera_bind_group, &[camera_offset]);
            *current_camera_offset = camera_offset;
        }

        batch.draw(
            r_pass,
            &self.quad_vertex_buffer,
            &self.quad_index_buffer,
            &self.dummy_instance_buffer,
            quad_bound,
            Some((shared_buf, instance_byte_offset)),
        );
        batch.clear();
    }

    /// Writes a camera matrix into the given slot of the multi-slot camera buffer.
    /// Slot must be < camera_slot_count. Uses queue.write_buffer so ALL writes
    /// are applied before the next queue.submit.
    pub fn write_camera_slot(&mut self, slot: u32, view_proj: [[f32; 4]; 4]) {
        debug_assert!(slot < self.camera_slot_count);
        let s = slot as usize;
        if self.last_camera[s] == view_proj {
            return;
        }
        self.last_camera[s] = view_proj;
        let offset = (slot * self.camera_slot_stride) as u64;
        self.gpu
            .queue
            .write_buffer(&self.camera_buffer, offset, bytemuck::bytes_of(&CameraUniform { view_proj }));
    }

    /// Returns the byte stride between camera slots (aligned to GPU requirements)
    pub fn camera_slot_stride(&self) -> u32 {
        self.camera_slot_stride
    }

    /// Returns the maximum number of camera slots available
    pub fn camera_slot_count(&self) -> u32 {
        self.camera_slot_count
    }

    /// Uploads the given view-projection matrix to slot 0 of the camera buffer
    pub fn upload_camera_matrix(&mut self, view_proj: [[f32; 4]; 4]) {
        self.write_camera_slot(0, view_proj);
    }

    /// Upload all instances merged into a single GPU buffer. Returns a reference
    /// to the shared buffer for use in draw calls.
    /// Accepts per-batch instance slices to avoid an intermediate Vec — writes
    /// directly into the wgpu staging buffer via `write_buffer_with`.
    pub fn upload_shared_instances_batched(&mut self, batch_slices: &[&[instance::Instance]]) {
        let inst_size = std::mem::size_of::<instance::Instance>();
        let total_instances: usize = batch_slices.iter().map(|s| s.len()).sum();
        if total_instances == 0 {
            return;
        }
        let required_bytes = (total_instances * inst_size) as u64;
        let needs_recreate = self.shared_instance_buffer.as_ref().is_none_or(|b| b.size() < required_bytes);
        if needs_recreate {
            let alloc = required_bytes.next_power_of_two().max((1024 * inst_size) as u64);
            self.shared_instance_buffer = Some(self.gpu.device.create_buffer(&BufferDescriptor {
                label: Some("Shared Instance Buffer"),
                size: alloc,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let buf = self.shared_instance_buffer.as_ref().unwrap();
        if let Some(mut view) = self
            .gpu
            .queue
            .write_buffer_with(buf, 0, wgpu::BufferSize::new(required_bytes).unwrap())
        {
            let mut offset = 0usize;
            for slice in batch_slices {
                let bytes = bytemuck::cast_slice::<instance::Instance, u8>(slice);
                let len = bytes.len();
                if len > 0 {
                    view.slice(offset..offset + len).copy_from_slice(bytes);
                    offset += len;
                }
            }
        }
    }

    /// Returns a reference to the shared instance buffer, if any
    pub fn shared_instance_buffer(&self) -> Option<&Buffer> {
        self.shared_instance_buffer.as_ref()
    }

    /// Create an offscreen render target
    pub fn create_offscreen_target(&self, width: u32, height: u32, format: TextureFormat) -> OffscreenTarget {
        OffscreenTarget::new(&self.gpu.device, width, height, format)
    }

    /// Adds an offscreen target texture & returns its id
    pub fn add_offscreen_texture(&mut self, offscreen: &mut OffscreenTarget) -> usize {
        self.textures.insert_offscreen(&self.gpu.device, offscreen)
    }

    /// Adds a new texture from image bytes & returns its id
    pub fn add_texture(&mut self, data: &[u8]) -> usize {
        self.textures.insert(&self.gpu.device, &self.gpu.queue, data)
    }

    /// Adds a new texture from image bytes with nearest-neighbor filtering & returns its id
    pub fn add_texture_nearest(&mut self, data: &[u8]) -> usize {
        self.textures.insert_nearest(&self.gpu.device, &self.gpu.queue, data)
    }

    /// Adds a texture from raw RGBA bytes & returns its id
    pub fn add_texture_raw(&mut self, w: u32, h: u32, data: &[u8]) -> usize {
        self.textures.insert_raw(&self.gpu.device, &self.gpu.queue, w, h, data)
    }

    /// Replaces an existing texture with new image data
    pub fn update_texture(&mut self, index: usize, data: &[u8]) {
        self.textures.replace(&self.gpu.device, &self.gpu.queue, index, data);
    }

    /// Replaces an existing texture with raw RGBA bytes
    pub fn update_texture_raw(&mut self, index: usize, w: u32, h: u32, data: &[u8]) {
        self.textures.replace_raw(&self.gpu.device, &self.gpu.queue, index, w, h, data);
    }

    /// Creates a uniform buffer and returns its id
    pub fn add_uniform(&mut self, data: &[u8]) -> usize {
        self.uniforms.insert(&self.gpu.device, data)
    }

    /// Updates an existing uniform buffer with new data
    pub fn update_uniform(&mut self, id: usize, data: &[u8]) {
        self.uniforms.write(&self.gpu.queue, id, data);
    }

    /// Creates a custom shader pipeline from WGSL source code
    /// Returns the pipeline index for use in draw calls
    pub fn add_shader(&mut self, wgsl_source: &str) -> usize {
        self.pipelines
            .add_custom(&self.gpu.device, self.surface_format, wgsl_source, &[], &[])
    }

    /// Creates a custom shader pipeline with associated uniform buffers
    ///
    /// `uniform_ids` specify which renderer uniform buffers should be bound
    /// after the built-in texture and camera bind groups when this shader is used
    ///
    /// Returns the pipeline index for use in draw calls
    pub fn add_shader_with_uniforms(&mut self, wgsl_source: &str, uniform_ids: &[usize]) -> usize {
        let layouts = vec![self.uniforms.layout(); uniform_ids.len()];
        self.pipelines
            .add_custom(&self.gpu.device, self.surface_format, wgsl_source, &layouts, uniform_ids)
    }
}
