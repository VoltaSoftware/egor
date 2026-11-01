pub mod batch;
#[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
mod display;
pub mod frame;
pub mod instance;
mod instance_upload;
mod pipeline;
#[cfg(feature = "shader-check")]
pub mod shader_check;
pub mod target;
mod texture;
mod uniforms;
pub mod vertex;

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub use instance_upload::InstanceUploadMetrics;
pub use texture::EmptyTextureArray;

pub use wgpu::{
    self, AdapterInfo, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Extent3d,
    MemoryHints, Queue, RenderPass, Texture, TextureFormat,
};

use wgpu::{
    Adapter, BindGroup, BindGroupDescriptor, BindGroupEntry, Color, DeviceDescriptor, Instance,
    InstanceDescriptor, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RequestAdapterOptions, StoreOp,
    SurfaceTarget, TextureView, WindowHandle,
    util::{BufferInitDescriptor, DeviceExt, new_instance_with_webgpu_detection},
};

use crate::{
    batch::GeometryBatch,
    frame::{Frame, Presentable},
    pipeline::{CompileMode, CustomBinding, Pipelines},
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
    pub device_lost: Arc<AtomicBool>,
}

/// Controls which wgpu backend set egor enables when creating its renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RendererBackendPreference {
    /// Use wgpu's default backend set.
    Auto,
    /// Use an explicit backend set.
    Backends(wgpu::Backends),
}

impl Default for RendererBackendPreference {
    fn default() -> Self {
        Self::Auto
    }
}

impl RendererBackendPreference {
    pub const fn auto() -> Self {
        Self::Auto
    }

    pub const fn gl() -> Self {
        Self::Backends(wgpu::Backends::GL)
    }

    pub const fn vulkan() -> Self {
        Self::Backends(wgpu::Backends::VULKAN)
    }

    pub const fn metal() -> Self {
        Self::Backends(wgpu::Backends::METAL)
    }

    pub const fn dx12() -> Self {
        Self::Backends(wgpu::Backends::DX12)
    }

    pub const fn webgpu() -> Self {
        Self::Backends(wgpu::Backends::BROWSER_WEBGPU)
    }
}

fn renderer_instance_descriptor() -> InstanceDescriptor {
    let mut desc = InstanceDescriptor::new_without_display_handle();
    desc.flags = desc.flags.with_env();
    // egor never draws or dispatches indirectly, and validating those calls compiles compute pipelines while the
    // device is created: 60-100 ms of an iPhone's launch with a cold shader cache, and 0.5 s once after an install.
    desc.flags.remove(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL);
    configure_renderer_instance_flags(&mut desc);
    desc
}

#[cfg(target_os = "android")]
fn configure_renderer_instance_flags(desc: &mut InstanceDescriptor) {
    if desc.flags.contains(wgpu::InstanceFlags::DEBUG) {
        log::info!("[egor] renderer init: disabling wgpu DEBUG instance flag on Android");
    }
    // Android emulator Vulkan drivers can crash inside vk_common_SetDebugUtilsObjectNameEXT
    // when wgpu DEBUG labels objects. Keep validation requested, but skip driver debug labels.
    desc.flags.remove(wgpu::InstanceFlags::DEBUG);
}

#[cfg(not(target_os = "android"))]
fn configure_renderer_instance_flags(_desc: &mut InstanceDescriptor) {}

#[derive(Debug)]
pub enum RendererInitError {
    CreateSurface(String),
    RequestAdapter(String),
    RequestDevice(String),
    SurfaceConfig(target::BackbufferError),
}

impl std::fmt::Display for RendererInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateSurface(error) => {
                write!(f, "failed to create adapter selection surface: {error}")
            }
            Self::RequestAdapter(error) => write!(f, "failed to request GPU adapter: {error}"),
            Self::RequestDevice(error) => write!(f, "failed to request GPU device: {error}"),
            Self::SurfaceConfig(error) => write!(f, "failed to determine surface config: {error}"),
        }
    }
}

impl std::error::Error for RendererInitError {}

#[derive(Debug)]
pub enum RendererFrameError {
    BeginFrame(String),
    FinishEncoder(String),
    Submit(String),
    Present(String),
}

impl std::fmt::Display for RendererFrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BeginFrame(error) => write!(f, "failed to begin frame: {error}"),
            Self::FinishEncoder(error) => write!(f, "failed to finish command encoder: {error}"),
            Self::Submit(error) => write!(f, "failed to submit command buffer: {error}"),
            Self::Present(error) => write!(f, "failed to present frame: {error}"),
        }
    }
}

impl std::error::Error for RendererFrameError {}

fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .copied()
        .map(str::to_owned)
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

/// A draw's `current_shader` while the default shader's outline pipeline is bound. No custom shader has this id,
/// so the next plain or custom batch binds its own pipeline again.
const OUTLINE_PIPELINE: usize = usize::MAX;

/// Low-level GPU renderer built on `wgpu`
///
/// Handles rendering pipelines, surface configuration, resources (textures, buffers), & drawing
pub struct Renderer {
    gpu: Gpu,
    instance_flags: wgpu::InstanceFlags,
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
    startup_surface: Option<wgpu::Surface<'static>>,
    depth_texture: wgpu::Texture,
    depth_view: TextureView,
    depth_size: (u32, u32),
    shared_instance_buffer: Option<Buffer>,
    instance_upload_pool: Option<instance_upload::InstanceUploadPool>,
    last_camera: [[[f32; 4]; 4]; 8],
    /// The backbuffer's turn; while it isn't identity, every camera slot has a turned copy
    /// `camera_slot_count` slots on, for draws straight into the backbuffer.
    surface_transform: wgpu::SurfaceTransform,
    /// Where `save_pipeline_cache` keeps the driver's compiled pipelines.
    pipeline_cache_file: Option<std::path::PathBuf>,
    /// Threads compiling custom shaders; 0 when they compile on the thread that adds them.
    #[cfg(not(target_arch = "wasm32"))]
    shader_compile_threads: usize,
    /// The background compiles not all finished yet.
    #[cfg(not(target_arch = "wasm32"))]
    compile_batch: Option<CompileBatch>,
    /// Compiles finished as the last frame began, and as the pipeline cache was last saved.
    #[cfg(not(target_arch = "wasm32"))]
    compiles_last_frame: usize,
    #[cfg(not(target_arch = "wasm32"))]
    compiles_saved: usize,
}

/// Custom shaders compiling on worker threads, from the first to start while none were.
#[cfg(not(target_arch = "wasm32"))]
struct CompileBatch {
    started_at: std::time::Instant,
    first: usize,
}

/// Threads for compiling custom shaders: several at once, without taking every core from the frame.
#[cfg(not(target_arch = "wasm32"))]
fn shader_compile_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(1, |cores| cores.get() / 2)
        .clamp(1, 3)
}

impl Renderer {
    /// Creates a renderer & initializes GPU state using the window's surface
    ///
    /// Sets up wgpu, pipelines, default texture & camera resources
    pub async fn new(
        window: impl Into<SurfaceTarget<'static>> + WindowHandle,
        memory_hints: &MemoryHints,
    ) -> Self {
        Self::try_new(window, memory_hints)
            .await
            .expect("failed to initialize egor renderer")
    }

    pub async fn new_with_backend(
        window: impl Into<SurfaceTarget<'static>> + WindowHandle,
        memory_hints: &MemoryHints,
        backend_preference: RendererBackendPreference,
    ) -> Self {
        Self::try_new_with_backend(window, memory_hints, backend_preference, None)
            .await
            .expect("failed to initialize egor renderer")
    }

    pub async fn try_new(
        window: impl Into<SurfaceTarget<'static>> + WindowHandle,
        memory_hints: &MemoryHints,
    ) -> Result<Self, RendererInitError> {
        Self::try_new_with_backend(window, memory_hints, RendererBackendPreference::Auto, None)
            .await
    }

    /// `pipeline_cache_dir` keeps the driver's compiled pipelines between runs, where the driver can (Vulkan,
    /// and OpenGL's program binaries):
    /// compiling a pipeline can take tens of milliseconds on a phone. Save it with
    /// [`Self::save_pipeline_cache`].
    pub async fn try_new_with_backend(
        window: impl Into<SurfaceTarget<'static>> + WindowHandle,
        memory_hints: &MemoryHints,
        backend_preference: RendererBackendPreference,
        pipeline_cache_dir: Option<&std::path::Path>,
    ) -> Result<Self, RendererInitError> {
        log::info!("[egor] renderer init: creating wgpu instance");
        let mut desc = renderer_instance_descriptor();
        #[cfg(not(any(target_arch = "wasm32", target_os = "android")))]
        let window = display::attach_to_instance(&mut desc, window.into());
        match backend_preference {
            RendererBackendPreference::Auto => {
                log::info!("[egor] renderer init: enabled backend set: auto");
            }
            RendererBackendPreference::Backends(backends) => {
                log::info!("[egor] renderer init: enabled backend set: {backends:?}");
                desc.backends = backends;
            }
        }
        log::info!("[egor] renderer init: instance flags: {:?}", desc.flags);
        let instance_flags = desc.flags;
        let instance = new_instance_with_webgpu_detection(desc).await;
        #[cfg(not(target_os = "android"))]
        log::info!("[egor] renderer init: creating adapter selection surface");
        #[cfg(not(target_os = "android"))]
        let surface = instance
            .create_surface(window)
            .map_err(|error| RendererInitError::CreateSurface(error.to_string()))?;
        #[cfg(target_os = "android")]
        log::info!("[egor] renderer init: requesting adapter before creating an Android surface");
        #[cfg(not(target_os = "android"))]
        let compatible_surface = Some(&surface);
        #[cfg(target_os = "android")]
        let compatible_surface = None;
        log::info!("[egor] renderer init: requesting adapter");
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                // Required for WebGL to prevent selecting a non-presentable device.
                // On Android/GLES, probing through the native window can leave the
                // SurfaceView connected before the real swapchain is configured.
                compatible_surface,
                ..Default::default()
            })
            .await
            .map_err(|error| RendererInitError::RequestAdapter(error.to_string()))?;
        let info = adapter.get_info();
        log::info!(
            "[egor] wgpu backend: {:?} | adapter: {} | driver: {}",
            info.backend,
            info.name,
            info.driver
        );

        #[cfg(target_os = "android")]
        let (instance, adapter, _info) = if backend_preference == RendererBackendPreference::Auto {
            let selected_backends = wgpu::Backends::from(info.backend);
            log::info!(
                "[egor] renderer init: auto selected {:?}; recreating Android instance with {:?}",
                info.backend,
                selected_backends
            );
            drop(adapter);

            let mut narrowed_desc = renderer_instance_descriptor();
            narrowed_desc.backends = selected_backends;
            log::info!(
                "[egor] renderer init: narrowed Android instance flags: {:?}",
                narrowed_desc.flags
            );
            let instance = new_instance_with_webgpu_detection(narrowed_desc).await;
            let adapter = instance
                .request_adapter(&RequestAdapterOptions {
                    compatible_surface: None,
                    ..Default::default()
                })
                .await
                .map_err(|error| RendererInitError::RequestAdapter(error.to_string()))?;
            let info = adapter.get_info();
            log::info!(
                "[egor] wgpu backend after Android auto narrowing: {:?} | adapter: {} | driver: {}",
                info.backend,
                info.name,
                info.driver
            );
            (instance, adapter, info)
        } else {
            (instance, adapter, info)
        };

        let adapter_limits = adapter.limits();
        #[cfg(target_arch = "wasm32")]
        let required_limits =
            wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter_limits);
        #[cfg(not(target_arch = "wasm32"))]
        let required_limits = adapter_limits;
        // Where the driver's compiled pipelines are kept between runs, for drivers that can (Vulkan, and OpenGL's
        // program binaries).
        let pipeline_cache_file = pipeline_cache_dir
            .zip(wgpu::util::pipeline_cache_key(&adapter.get_info()))
            .filter(|_| adapter.features().contains(wgpu::Features::PIPELINE_CACHE))
            .map(|(dir, key)| dir.join(key));
        let required_features = if pipeline_cache_file.is_some() {
            wgpu::Features::PIPELINE_CACHE
        } else {
            wgpu::Features::empty()
        };
        log::info!("[egor] renderer init: requesting device");
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                required_features,
                required_limits,
                memory_hints: memory_hints.clone(),
                ..Default::default()
            })
            .await
            .map_err(|error| RendererInitError::RequestDevice(error.to_string()))?;
        let pipeline_cache = pipeline_cache_file.as_ref().map(|file| {
            let data = std::fs::read(file).ok();
            log::info!(
                "[egor] renderer init: pipeline cache {} ({} bytes)",
                file.display(),
                data.as_ref().map_or(0, Vec::len)
            );
            // SAFETY: the file only ever holds `PipelineCache::get_data` from a device with the same cache key
            // (`save_pipeline_cache`); a driver that can't use it starts an empty cache (`fallback`).
            unsafe {
                device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("egor pipeline cache"),
                    data: data.as_deref(),
                    fallback: true,
                })
            }
        });
        let device_lost = Arc::new(AtomicBool::new(false));
        let device_lost_callback = Arc::clone(&device_lost);
        device.set_device_lost_callback(move |reason, message| {
            device_lost_callback.store(true, Ordering::SeqCst);
            log::error!("[egor] wgpu device lost: {reason:?}: {message:?}");
        });
        device.on_uncaptured_error(Arc::new(|error| {
            log::error!("[egor] wgpu uncaptured error: {error:?}");
        }));

        #[cfg(target_os = "android")]
        let (surface_format, startup_surface) = {
            log::info!(
                "[egor] renderer init: creating Android startup surface for format selection"
            );
            let surface = instance
                .create_surface(window)
                .map_err(|error| RendererInitError::CreateSurface(error.to_string()))?;
            let (_surface_config, surface_format, _) =
                target::surface_config_with_android_gl_fallback(&surface, &adapter, 1, 1)
                    .map_err(RendererInitError::SurfaceConfig)?;
            (surface_format, Some(surface))
        };
        #[cfg(not(target_os = "android"))]
        let (_surface_config, surface_format, _) =
            target::surface_config_with_android_gl_fallback(&surface, &adapter, 1, 1)
                .map_err(RendererInitError::SurfaceConfig)?;
        #[cfg(not(target_os = "android"))]
        let startup_surface = Some(surface);
        log::info!("[egor] renderer init: creating pipelines and core buffers");
        // A pipeline takes tens of milliseconds to compile on a phone, so pipelines compile on worker threads where
        // the backend allows it, and frames draw nothing until egor's own are in rather than wait for them.
        // OpenGL's context belongs to one thread at a time (wgpu gives up waiting for it after a second), so there
        // they compile a family a frame. Browsers compile them away from the page's main thread when asked
        // through `createRenderPipelineAsync`; WebGL has no such call.
        #[cfg(not(target_arch = "wasm32"))]
        let shader_compile_threads = match adapter.get_info().backend {
            wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::Dx12 => {
                shader_compile_threads()
            }
            _ => 0,
        };
        #[cfg(not(target_arch = "wasm32"))]
        let compile_mode = if shader_compile_threads > 0 {
            log::info!(
                "[egor] renderer init: compiling pipelines on {shader_compile_threads} threads"
            );
            CompileMode::Threads(shader_compile_threads)
        } else {
            CompileMode::Frames
        };
        #[cfg(target_arch = "wasm32")]
        let compile_mode = if adapter.get_info().backend == wgpu::Backend::BrowserWebGpu {
            log::info!(
                "[egor] renderer init: compiling pipelines through the browser's async pipelines"
            );
            CompileMode::Browser
        } else {
            CompileMode::Frames
        };
        let pipelines = Pipelines::new_with(&device, surface_format, pipeline_cache, compile_mode);

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
        let min_alignment = device.limits().min_uniform_buffer_offset_alignment.max(256) as u32;
        let camera_data_size = std::mem::size_of::<CameraUniform>() as u32;
        let camera_slot_stride =
            ((camera_data_size + min_alignment - 1) / min_alignment) * min_alignment;
        let camera_slot_count: u32 = 8;
        // Room for the turned copies too.
        let camera_buffer_size = camera_slot_stride * camera_slot_count * 2;

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
        log::info!("[egor] renderer init: complete");

        let instance_upload_pool = (cfg!(target_os = "windows")
            && adapter.get_info().backend == wgpu::Backend::Gl)
            .then(instance_upload::InstanceUploadPool::default);
        Ok(Renderer {
            instance_flags,
            gpu: Gpu {
                instance,
                adapter,
                device,
                queue,
                device_lost,
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
            startup_surface,
            depth_texture,
            depth_view,
            depth_size: (1, 1),
            shared_instance_buffer: None,
            instance_upload_pool,
            last_camera: [[[0.0; 4]; 4]; 8],
            surface_transform: wgpu::SurfaceTransform::Identity,
            pipeline_cache_file,
            #[cfg(not(target_arch = "wasm32"))]
            shader_compile_threads,
            // egor's own pipelines are the first batch.
            #[cfg(not(target_arch = "wasm32"))]
            compile_batch: (shader_compile_threads > 0).then(|| CompileBatch {
                started_at: std::time::Instant::now(),
                first: 0,
            }),
            #[cfg(not(target_arch = "wasm32"))]
            compiles_last_frame: 0,
            #[cfg(not(target_arch = "wasm32"))]
            compiles_saved: 0,
        })
    }

    pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth16Unorm;

    pub fn create_depth_texture(
        device: &Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, TextureView) {
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
    /// Flags used when creating this renderer's wgpu instance.
    pub fn instance_flags(&self) -> wgpu::InstanceFlags {
        self.instance_flags
    }

    /// Returns information about the active GPU adapter/backend.
    pub fn adapter_info(&self) -> AdapterInfo {
        self.gpu.adapter.get_info()
    }
    /// Returns a reference to the underlying wgpu `Device`
    pub fn device(&self) -> &Device {
        &self.gpu.device
    }
    /// Returns a reference to the underlying wgpu `Queue`
    pub fn queue(&self) -> &Queue {
        &self.gpu.queue
    }

    pub fn device_lost(&self) -> bool {
        self.gpu.device_lost.load(Ordering::SeqCst)
    }

    pub fn surface_format(&self) -> TextureFormat {
        self.surface_format
    }

    /// Consume the surface created during renderer startup and configure it as
    /// the first backbuffer. This avoids connecting the same native window to a
    /// second surface on platforms such as Android/EGL.
    pub fn take_startup_backbuffer(
        &mut self,
        w: u32,
        h: u32,
    ) -> Option<Result<target::Backbuffer, target::BackbufferError>> {
        let surface = self.startup_surface.take()?;
        Some(target::Backbuffer::try_from_surface(
            &self.gpu.adapter,
            &self.gpu.device,
            surface,
            w,
            h,
        ))
    }

    pub fn drop_startup_surface(&mut self) {
        self.startup_surface = None;
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
        self.try_begin_frame(target)
            .expect("failed to begin egor frame")
    }

    /// Begins a frame with panic containment around GPU/device calls.
    pub fn try_begin_frame(
        &mut self,
        target: &mut dyn RenderTarget,
    ) -> Result<Option<Frame>, RendererFrameError> {
        self.take_finished_compiles();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (view, presentable) = target.acquire(&self.gpu.device)?;
            let encoder = self.gpu.device.create_command_encoder(&Default::default());
            Some(Frame {
                view,
                encoder,
                presentable,
            })
        }))
        .map_err(|payload| RendererFrameError::BeginFrame(panic_payload_message(payload)))
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
    pub fn finish_encoder(
        &mut self,
        frame: Frame,
    ) -> (wgpu::CommandBuffer, Option<crate::frame::Presentable>) {
        frame.finish_encoder()
    }

    /// Finish encoder without submitting, returning driver/backend panics as errors.
    pub fn try_finish_encoder(
        &mut self,
        frame: Frame,
    ) -> Result<(wgpu::CommandBuffer, Option<Presentable>), RendererFrameError> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| frame.finish_encoder()))
            .map_err(|payload| RendererFrameError::FinishEncoder(panic_payload_message(payload)))
    }

    /// Submit a pre-finished command buffer
    pub fn submit_commands(&self, commands: wgpu::CommandBuffer) {
        self.gpu.queue.submit(Some(commands));
    }

    /// Submit a pre-finished command buffer, returning driver/backend panics as errors.
    pub fn try_submit_commands(
        &self,
        commands: wgpu::CommandBuffer,
    ) -> Result<(), RendererFrameError> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.gpu.queue.submit(Some(commands));
        }))
        .map_err(|payload| RendererFrameError::Submit(panic_payload_message(payload)))
    }

    /// Present a surface texture, returning driver/backend panics as errors.
    pub fn try_present(&self, presentable: Presentable) -> Result<(), RendererFrameError> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            presentable.present(&self.gpu.queue);
        }))
        .map_err(|payload| RendererFrameError::Present(panic_payload_message(payload)))
    }

    /// Begins a render pass with the given encoder and target view.
    /// Clears the view (set by [`Self::set_clear_color`]) and the depth buffer.
    pub fn begin_render_pass<'a>(
        &'a self,
        encoder: &'a mut CommandEncoder,
        view: &'a TextureView,
    ) -> RenderPass<'a> {
        self.begin_render_pass_with_depth(encoder, view, &self.depth_view, true)
    }

    /// Begins a one-pass render pass and discards depth after drawing.
    /// Use this when no later pass needs to load the depth buffer.
    pub fn begin_render_pass_discard_depth<'a>(
        &'a self,
        encoder: &'a mut CommandEncoder,
        view: &'a TextureView,
    ) -> RenderPass<'a> {
        self.begin_render_pass_with_depth_clear_color_and_store(
            encoder,
            view,
            &self.depth_view,
            self.clear_color,
            true,
            StoreOp::Discard,
        )
    }

    /// Begins a render pass that preserves existing content (LoadOp::Load).
    /// Used when splitting a frame into multiple passes (e.g. camera changes).
    pub fn begin_render_pass_load<'a>(
        &'a self,
        encoder: &'a mut CommandEncoder,
        view: &'a TextureView,
    ) -> RenderPass<'a> {
        self.begin_render_pass_load_with_depth(encoder, view, &self.depth_view)
    }

    pub fn clear_color(&self) -> wgpu::Color {
        self.clear_color
    }

    /// Begins a render pass with an explicit depth view, clearing both color and depth.
    pub fn begin_render_pass_with_depth<'a>(
        &'a self,
        encoder: &'a mut CommandEncoder,
        view: &'a TextureView,
        depth_view: &'a TextureView,
        clear_depth: bool,
    ) -> RenderPass<'a> {
        self.begin_render_pass_with_depth_clear_color(
            encoder,
            view,
            depth_view,
            self.clear_color,
            clear_depth,
        )
    }

    /// Begins a render pass with an explicit depth view and explicit clear color.
    pub fn begin_render_pass_with_depth_clear_color<'a>(
        &'a self,
        encoder: &'a mut CommandEncoder,
        view: &'a TextureView,
        depth_view: &'a TextureView,
        clear_color: wgpu::Color,
        clear_depth: bool,
    ) -> RenderPass<'a> {
        self.begin_render_pass_with_depth_clear_color_and_store(
            encoder,
            view,
            depth_view,
            clear_color,
            clear_depth,
            StoreOp::Store,
        )
    }

    fn begin_render_pass_with_depth_clear_color_and_store<'a>(
        &'a self,
        encoder: &'a mut CommandEncoder,
        view: &'a TextureView,
        depth_view: &'a TextureView,
        clear_color: wgpu::Color,
        clear_depth: bool,
        depth_store: StoreOp,
    ) -> RenderPass<'a> {
        encoder.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(clear_color),
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(Operations {
                    load: if clear_depth {
                        LoadOp::Clear(1.0)
                    } else {
                        LoadOp::Load
                    },
                    store: depth_store,
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
    pub fn bind_pass_state(
        &self,
        r_pass: &mut RenderPass<'_>,
        texture_id: Option<usize>,
        shader_id: Option<usize>,
        replace_blend: bool,
    ) {
        let texture = self.textures.get(texture_id);
        texture.bind(r_pass, 0);

        let (pipeline, bindings) = self
            .pipelines
            .resolve_with_replace(shader_id, replace_blend, texture.is_array())
            .expect("pipelines should be compiled before drawing");
        r_pass.set_pipeline(pipeline);
        self.bind_custom(r_pass, bindings);

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
        replace_blend: bool,
        camera_offset: u32,
        current_texture: &mut Option<usize>,
        current_shader: &mut Option<usize>,
        current_replace_blend: &mut bool,
        current_camera_offset: &mut u32,
        quad_bound: &mut bool,
    ) -> u32 {
        if batch.is_empty() {
            return 0;
        }

        batch.upload(&self.gpu.device, &self.gpu.queue);
        self.draw_uploaded_batch(
            r_pass,
            batch,
            texture_id,
            shader_id,
            replace_blend,
            camera_offset,
            current_texture,
            current_shader,
            current_replace_blend,
            current_camera_offset,
            quad_bound,
            None,
        )
    }

    /// Binds what `batch` needs that isn't bound yet and draws it. With the default shader, each run of
    /// outlined text goes through the outline pipeline and everything else through the plain one.
    fn draw_uploaded_batch<'a>(
        &'a self,
        r_pass: &mut RenderPass<'a>,
        batch: &mut GeometryBatch,
        texture_id: Option<usize>,
        shader_id: Option<usize>,
        replace_blend: bool,
        camera_offset: u32,
        current_texture: &mut Option<usize>,
        current_shader: &mut Option<usize>,
        current_replace_blend: &mut bool,
        current_camera_offset: &mut u32,
        quad_bound: &mut bool,
        external_instances: Option<(&'a Buffer, u64)>,
    ) -> u32 {
        let texture = self.textures.get(texture_id);
        let texture_array = texture.is_array();
        let mut texture_kind_changed =
            self.textures.get(*current_texture).is_array() != texture_array;
        if *current_texture != texture_id {
            texture.bind(r_pass, 0);
            *current_texture = texture_id;
        }

        let mut bind = |r_pass: &mut RenderPass<'a>, outlined: bool| {
            let bound = if outlined {
                Some(OUTLINE_PIPELINE)
            } else {
                shader_id
            };
            if !texture_kind_changed
                && *current_shader == bound
                && *current_replace_blend == replace_blend
            {
                return;
            }
            if outlined {
                let pipeline = self
                    .pipelines
                    .resolve_outline(texture_array)
                    .expect("pipelines should be compiled before drawing");
                r_pass.set_pipeline(pipeline);
            } else {
                let (pipeline, bindings) = self
                    .pipelines
                    .resolve_with_replace(shader_id, replace_blend, texture_array)
                    .expect("pipelines should be compiled before drawing");
                r_pass.set_pipeline(pipeline);
                self.bind_custom(r_pass, bindings);
            }
            *current_shader = bound;
            *current_replace_blend = replace_blend;
            texture_kind_changed = false;
        };
        // Custom shaders and flat replacements draw a whole batch with one pipeline.
        let default_shader = shader_id.is_none() && !replace_blend;
        if !default_shader {
            bind(r_pass, false);
        }

        if *current_camera_offset != camera_offset {
            r_pass.set_bind_group(1, &self.camera_bind_group, &[camera_offset]);
            *current_camera_offset = camera_offset;
        }

        let draw_calls = batch.draw(
            r_pass,
            &self.quad_vertex_buffer,
            &self.quad_index_buffer,
            &self.dummy_instance_buffer,
            quad_bound,
            external_instances,
            default_shader.then_some(&mut bind as &mut dyn FnMut(&mut RenderPass<'a>, bool)),
        );
        batch.clear();
        draw_calls
    }

    /// Like draw_batch but uses a shared instance buffer at the given byte offset.
    /// Skips per-batch instance upload entirely.
    pub fn draw_batch_shared<'a>(
        &'a self,
        r_pass: &mut RenderPass<'a>,
        batch: &mut GeometryBatch,
        texture_id: Option<usize>,
        shader_id: Option<usize>,
        replace_blend: bool,
        camera_offset: u32,
        current_texture: &mut Option<usize>,
        current_shader: &mut Option<usize>,
        current_replace_blend: &mut bool,
        current_camera_offset: &mut u32,
        quad_bound: &mut bool,
        shared_buf: &'a Buffer,
        instance_byte_offset: u64,
    ) -> u32 {
        if batch.is_empty() {
            return 0;
        }

        batch.upload_geometry_only(&self.gpu.device, &self.gpu.queue);
        self.draw_uploaded_batch(
            r_pass,
            batch,
            texture_id,
            shader_id,
            replace_blend,
            camera_offset,
            current_texture,
            current_shader,
            current_replace_blend,
            current_camera_offset,
            quad_bound,
            Some((shared_buf, instance_byte_offset)),
        )
    }

    /// Writes a camera matrix into the given slot of the multi-slot camera buffer.
    /// Slot must be < camera_slot_count. Uses queue.write_buffer so ALL writes
    /// are applied before the next queue.submit.
    /// With a turned backbuffer (see [`Self::set_surface_transform`]) it also writes the turned
    /// copy to `slot + camera_slot_count`.
    pub fn write_camera_slot(&mut self, slot: u32, view_proj: [[f32; 4]; 4]) {
        debug_assert!(slot < self.camera_slot_count);
        let s = slot as usize;
        if self.last_camera[s] == view_proj {
            return;
        }
        self.last_camera[s] = view_proj;
        self.write_camera_uniform(slot, view_proj);
        if self.surface_transform != wgpu::SurfaceTransform::Identity {
            self.write_camera_uniform(
                slot + self.camera_slot_count,
                target::turn_view_proj(view_proj, self.surface_transform),
            );
        }
    }

    fn write_camera_uniform(&self, slot: u32, view_proj: [[f32; 4]; 4]) {
        let offset = (slot * self.camera_slot_stride) as u64;
        self.gpu.queue.write_buffer(
            &self.camera_buffer,
            offset,
            bytemuck::bytes_of(&CameraUniform { view_proj }),
        );
    }

    /// Sets the backbuffer's turn ([`target::Backbuffer::transform`]) before this frame's camera
    /// writes, so draws into the backbuffer can use the turned camera slots.
    pub fn set_surface_transform(&mut self, transform: wgpu::SurfaceTransform) {
        if self.surface_transform == transform {
            return;
        }
        self.surface_transform = transform;
        // Write every camera again, turned the new way.
        self.last_camera = [[[f32::NAN; 4]; 4]; 8];
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
        self.upload_instances(batch_slices, None);
    }

    /// Encode reusable instance uploads before any render pass in this encoder.
    /// The encoder must be submitted for the upload slots to become available again.
    pub fn upload_shared_instances_with_encoder(
        &mut self,
        batch_slices: &[&[instance::Instance]],
        encoder: &mut CommandEncoder,
    ) {
        self.upload_instances(batch_slices, Some(encoder));
    }

    pub fn instance_upload_metrics(&self) -> InstanceUploadMetrics {
        self.instance_upload_pool
            .as_ref()
            .map(instance_upload::InstanceUploadPool::metrics)
            .unwrap_or_default()
    }

    fn upload_instances(
        &mut self,
        batch_slices: &[&[instance::Instance]],
        encoder: Option<&mut CommandEncoder>,
    ) {
        let inst_size = std::mem::size_of::<instance::Instance>();
        let total_instances: usize = batch_slices.iter().map(|s| s.len()).sum();
        if total_instances == 0 {
            return;
        }
        let required_bytes = (total_instances * inst_size) as u64;
        let needs_recreate = self
            .shared_instance_buffer
            .as_ref()
            .is_none_or(|b| b.size() < required_bytes);
        if needs_recreate {
            let alloc = required_bytes
                .next_power_of_two()
                .max((1024 * inst_size) as u64);
            self.shared_instance_buffer = Some(self.gpu.device.create_buffer(&BufferDescriptor {
                label: Some("Shared Instance Buffer"),
                size: alloc,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let buf = self.shared_instance_buffer.as_ref().unwrap();
        if let (Some(pool), Some(encoder)) = (&mut self.instance_upload_pool, encoder)
            && pool.upload(&self.gpu.device, encoder, buf, batch_slices, required_bytes)
        {
            return;
        }
        if let Some(mut view) =
            self.gpu
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
    pub fn create_offscreen_target(
        &self,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> OffscreenTarget {
        OffscreenTarget::new(&self.gpu.device, width, height, format)
    }

    /// Adds an offscreen target texture & returns its id
    pub fn add_offscreen_texture(&mut self, offscreen: &mut OffscreenTarget) -> usize {
        self.add_offscreen_texture_with_filter(offscreen, wgpu::FilterMode::Linear)
    }

    /// Adds an offscreen texture with an explicit sampling filter.
    pub fn add_offscreen_texture_with_filter(
        &mut self,
        offscreen: &mut OffscreenTarget,
        filter: wgpu::FilterMode,
    ) -> usize {
        self.textures
            .insert_offscreen(&self.gpu.device, offscreen, filter)
    }

    /// Rebinds offscreen texture `id` to `offscreen` after it was resized, keeping the id.
    pub fn replace_offscreen_texture_with_filter(
        &mut self,
        id: usize,
        offscreen: &OffscreenTarget,
        filter: wgpu::FilterMode,
    ) {
        self.textures
            .replace_offscreen(&self.gpu.device, id, offscreen, filter);
    }

    /// Return a cached linear-filtered binding without modifying the source texture.
    pub fn linear_texture_variant(&mut self, id: usize) -> usize {
        self.textures.linear_variant(&self.gpu.device, id)
    }

    /// Adds a new texture from image bytes & returns its id
    pub fn add_texture(&mut self, data: &[u8]) -> usize {
        self.textures
            .insert(&self.gpu.device, &self.gpu.queue, data)
    }

    /// Adds a new texture from image bytes with nearest-neighbor filtering & returns its id
    pub fn add_texture_nearest(&mut self, data: &[u8]) -> usize {
        self.textures
            .insert_nearest(&self.gpu.device, &self.gpu.queue, data)
    }

    /// Adds a texture from raw RGBA bytes & returns its id
    pub fn add_texture_raw(&mut self, w: u32, h: u32, data: &[u8]) -> usize {
        self.textures
            .insert_raw_nearest(&self.gpu.device, &self.gpu.queue, w, h, data)
    }

    /// Upload all layers in one texture array before issuing draws using its id.
    pub fn add_texture_array_raw(
        &mut self,
        w: u32,
        h: u32,
        layers: u32,
        data: &[u8],
    ) -> Result<usize, String> {
        self.textures
            .insert_array_raw(&self.gpu.device, &self.gpu.queue, w, h, layers, data)
    }

    /// Creates a texture array from `layers` of RGBA8 data followed by
    /// `spare_layers` transparent layers for [`Self::write_texture_region`].
    pub fn add_texture_array_raw_with_spare_layers(
        &mut self,
        w: u32,
        h: u32,
        layers: u32,
        spare_layers: u32,
        data: &[u8],
    ) -> Result<usize, String> {
        self.textures.insert_array_raw_with_spare_layers(
            &self.gpu.device,
            &self.gpu.queue,
            w,
            h,
            layers,
            spare_layers,
            data,
        )
    }

    /// Overwrites a `w`×`h` RGBA8 region of one layer of an existing texture.
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
        self.textures
            .write_region(&self.gpu.queue, id, layer, x, y, w, h, data)
    }

    /// Replaces an existing texture with new image data
    pub fn update_texture(&mut self, index: usize, data: &[u8]) {
        self.textures
            .replace(&self.gpu.device, &self.gpu.queue, index, data);
    }

    /// Replaces an existing texture with raw RGBA bytes
    pub fn update_texture_raw(&mut self, index: usize, w: u32, h: u32, data: &[u8]) {
        self.textures
            .replace_raw(&self.gpu.device, &self.gpu.queue, index, w, h, data);
    }

    /// Frees texture `id` (and its linear variant) right away; the next texture reuses its id. Nothing recorded
    /// but not yet drawn may still use it: `egor_glue`'s `Graphics::free_texture` waits for the frame to end.
    pub fn remove_texture(&mut self, id: usize) {
        self.textures.remove(id);
    }

    /// Creates a uniform buffer and returns its id
    pub fn add_uniform(&mut self, data: &[u8]) -> usize {
        self.uniforms.insert(&self.gpu.device, data)
    }

    /// Updates an existing uniform buffer with new data
    pub fn update_uniform(&mut self, id: usize, data: &[u8]) {
        self.uniforms.write(&self.gpu.queue, id, data);
    }

    /// Frees uniform `id` right away, like [`Self::remove_texture`].
    pub fn remove_uniform(&mut self, id: usize) {
        self.uniforms.remove(id);
    }

    /// Frees custom shader `id` right away, like [`Self::remove_texture`]. Its compiled pipeline stays cached
    /// for the next shader with the same source.
    pub fn remove_shader(&mut self, id: usize) {
        self.pipelines.remove_custom(id);
    }

    /// Creates a custom shader pipeline from WGSL source code
    /// Returns the pipeline index for use in draw calls
    pub fn add_shader(&mut self, wgsl_source: &str) -> usize {
        self.add_shader_with_textures(wgsl_source, &[], 0)
    }

    /// Creates a custom shader pipeline with associated uniform buffers
    ///
    /// `uniform_ids` specify which renderer uniform buffers should be bound
    /// after the built-in texture and camera bind groups when this shader is used
    ///
    /// Returns the pipeline index for use in draw calls
    pub fn add_shader_with_uniforms(&mut self, wgsl_source: &str, uniform_ids: &[usize]) -> usize {
        self.add_shader_with_textures(wgsl_source, uniform_ids, 0)
    }

    /// Like [`Self::add_shader_with_uniforms`], plus `texture_count` textures bound after the
    /// uniforms, so a shader can sample several, each with its own filtering. Texture `i` is
    /// `@group(2 + uniforms + i) @binding(0) var t: texture_2d<f32>;` with its sampler at
    /// `@binding(1)`; set it with [`Self::set_shader_texture`]. WebGL2 has 4 bind groups, so only
    /// two uniforms and textures together fit after the texture and camera ones.
    pub fn add_shader_with_textures(
        &mut self,
        wgsl_source: &str,
        uniform_ids: &[usize],
        texture_count: usize,
    ) -> usize {
        let texture_layout = self.pipelines.texture_layout_2d().clone();
        let mut layouts = vec![self.uniforms.layout(); uniform_ids.len()];
        layouts.extend(std::iter::repeat_n(&texture_layout, texture_count));
        let bindings = uniform_ids
            .iter()
            .map(|&id| CustomBinding::Uniform(id))
            .chain(std::iter::repeat_n(
                CustomBinding::Texture(None),
                texture_count,
            ))
            .collect();
        let (finished, started) = self.pipelines.compiles();
        let id = self.pipelines.add_custom(
            &self.gpu.device,
            self.surface_format,
            wgsl_source,
            &layouts,
            bindings,
        );
        #[cfg(not(target_arch = "wasm32"))]
        if self.shader_compile_threads > 0
            && finished == started
            && self.pipelines.compiles().1 > started
        {
            self.compile_batch = Some(CompileBatch {
                started_at: std::time::Instant::now(),
                first: started,
            });
        }
        #[cfg(target_arch = "wasm32")]
        let _ = (finished, started);
        id
    }

    /// Waits for custom shader `id`'s pipelines, if a compile thread is still on them. A draw needs them, so
    /// `Graphics` waits before a shader can be drawn with.
    pub fn wait_for_shader(&mut self, id: usize) {
        self.pipelines.wait_for_custom(id);
    }

    /// Shader compiles finished, and started; they all finish at once where shaders compile on the thread
    /// that adds them.
    pub fn shader_compiles(&self) -> (usize, usize) {
        self.pipelines.compiles()
    }

    /// Whether other threads can use the GPU: Vulkan, Metal and DX12. OpenGL's context belongs to one thread at a
    /// time, and browsers have no threads to give.
    pub fn allows_worker_threads(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        return self.shader_compile_threads > 0;
        #[cfg(target_arch = "wasm32")]
        false
    }

    /// Starts making an empty `w`×`h` texture array of `layers`, which the GPU clears on first use. Allocating a big
    /// one took a Samsung A16's driver 0.19 s, so where other threads can use the GPU, one makes it.
    pub fn start_empty_texture_array(
        &mut self,
        w: u32,
        h: u32,
        layers: u32,
    ) -> Result<EmptyTextureArray, String> {
        let on_thread = self.allows_worker_threads();
        self.textures
            .start_empty_array(&self.gpu.device, &self.gpu.queue, w, h, layers, on_thread)
    }

    /// The texture array's id, once it is made.
    pub fn take_empty_texture_array(&mut self, array: &mut EmptyTextureArray) -> Option<usize> {
        self.textures.take_empty_array(array)
    }

    /// Takes in finished compiles and says whether egor's own pipelines, which every frame draws with, are
    /// compiled. Until they are, `egor_glue` draws nothing rather than wait.
    pub fn startup_pipelines_ready(&mut self) -> bool {
        if !self.pipelines.builtins_ready() {
            self.take_finished_compiles();
        }
        self.pipelines.builtins_ready()
    }

    /// Takes in the custom shaders the compile threads finished. Once every compile is in and a frame has
    /// passed without another, it saves the driver's compiled pipelines for the next run
    /// ([`Self::save_pipeline_cache`]): after a batch on the threads, or a run of frames compiling on this one
    /// (OpenGL).
    fn take_finished_compiles(&mut self) {
        self.pipelines.take_finished_compiles();
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (finished, started) = self.pipelines.compiles();
            if finished == started
                && let Some(batch) = self.compile_batch.take()
            {
                log::info!(
                    "[egor] compiled {} shaders on {} threads in {} ms",
                    finished - batch.first,
                    self.shader_compile_threads,
                    batch.started_at.elapsed().as_millis()
                );
            }
            if finished == started
                && finished == self.compiles_last_frame
                && finished > self.compiles_saved
            {
                self.compiles_saved = finished;
                self.save_pipeline_cache();
            }
            self.compiles_last_frame = finished;
        }
    }

    /// Writes the driver's compiled pipelines to the pipeline cache file, so the next run starts with them.
    /// Call it after compiling a batch of pipelines, such as while loading. A no-op without a cache.
    ///
    /// Natively the cache is read out and written on another thread: a phone's driver can take a while to
    /// serialize a few MB of it, and the frame shouldn't wait.
    pub fn save_pipeline_cache(&self) {
        let (Some(file), Some(cache)) = (
            self.pipeline_cache_file.clone(),
            self.pipelines.cache().cloned(),
        ) else {
            return;
        };
        let save = move || {
            // One save at a time, so two never write the temporary file at once.
            static SAVING: std::sync::Mutex<()> = std::sync::Mutex::new(());
            let _saving = SAVING.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let Some(data) = cache.get_data() else {
                return;
            };
            // Through a temporary file, so a crash mid-write never leaves a torn cache.
            let temp = file.with_extension("tmp");
            let result = file
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|()| std::fs::write(&temp, &data))
                .and_then(|()| std::fs::rename(&temp, &file));
            match result {
                Ok(()) => log::info!(
                    "[egor] saved pipeline cache {} ({} bytes)",
                    file.display(),
                    data.len()
                ),
                Err(error) => log::warn!(
                    "[egor] saving pipeline cache {} failed: {error}",
                    file.display()
                ),
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        if let Err(error) = std::thread::Builder::new()
            .name("egor pipeline cache".to_owned())
            .spawn(save)
        {
            log::warn!("[egor] couldn't start saving the pipeline cache: {error}");
        }
        #[cfg(target_arch = "wasm32")]
        save();
    }

    /// Points texture `index` of a shader from [`Self::add_shader_with_textures`] at a 2D texture,
    /// such as an offscreen target's. Like a uniform, it applies to every draw with the shader in the
    /// frame being built.
    pub fn set_shader_texture(&mut self, shader_id: usize, index: usize, texture_id: usize) {
        self.pipelines
            .set_custom_texture(shader_id, index, texture_id);
    }

    /// Binds a custom shader's uniforms and textures after the texture and camera groups.
    fn bind_custom(&self, r_pass: &mut RenderPass, bindings: &[CustomBinding]) {
        for (i, binding) in bindings.iter().enumerate() {
            let group = (2 + i) as u32;
            match *binding {
                CustomBinding::Uniform(id) => {
                    r_pass.set_bind_group(group, self.uniforms.bind_group(id), &[])
                }
                CustomBinding::Texture(id) => self.textures.get(id).bind(r_pass, group),
            }
        }
    }
}

/// Expands the optional texture-sampling marker for a 2D or array pipeline.
/// Both variants perform one texture lookup; selection happens per draw batch.
pub fn prepare_texture_shader(source: &str, array: bool) -> String {
    let sampling = if array {
        "@group(0) @binding(0) var egor_texture: texture_2d_array<f32>;\n@group(0) @binding(1) var egor_sampler: sampler;\nfn egor_sample_texture(uv: vec2<f32>, layer: i32) -> vec4<f32> { return textureSampleLevel(egor_texture, egor_sampler, uv, layer, 0.0); }\nfn egor_texture_dimensions() -> vec2<u32> { return textureDimensions(egor_texture, 0); }"
    } else {
        "@group(0) @binding(0) var egor_texture: texture_2d<f32>;\n@group(0) @binding(1) var egor_sampler: sampler;\nfn egor_sample_texture(uv: vec2<f32>, layer: i32) -> vec4<f32> { return textureSampleLevel(egor_texture, egor_sampler, uv, 0.0); }\nfn egor_texture_dimensions() -> vec2<u32> { return textureDimensions(egor_texture, 0); }"
    };
    source.replace("// EGOR_TEXTURE_SAMPLING", sampling)
}
