use wgpu::{
    Adapter, CommandEncoder, CompositeAlphaMode, CurrentSurfaceTexture, Device, DownlevelFlags, Extent3d, Instance, PresentMode, Surface,
    SurfaceConfiguration, SurfaceTarget, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor, WindowHandle,
};

#[cfg(not(target_arch = "wasm32"))]
use wgpu::PollType;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

use crate::frame::Presentable;

#[cfg(target_os = "android")]
const DESIRED_MAXIMUM_FRAME_LATENCY: u32 = 1;
#[cfg(not(target_os = "android"))]
const DESIRED_MAXIMUM_FRAME_LATENCY: u32 = 2;

#[cfg(target_os = "android")]
fn vsync_present_mode() -> PresentMode {
    PresentMode::AutoVsync
}

#[cfg(not(target_os = "android"))]
fn vsync_present_mode() -> PresentMode {
    PresentMode::Fifo
}

#[derive(Debug)]
pub enum BackbufferError {
    ZeroSize { width: u32, height: u32 },
    CreateSurface(String),
    UnsupportedSurface { width: u32, height: u32 },
    Configure(String),
}

impl std::fmt::Display for BackbufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSize { width, height } => write!(f, "surface size is zero ({width}x{height})"),
            Self::CreateSurface(error) => write!(f, "failed to create surface: {error}"),
            Self::UnsupportedSurface { width, height } => {
                write!(f, "surface has no compatible default config for {width}x{height}")
            }
            Self::Configure(error) => write!(f, "failed to configure surface: {error}"),
        }
    }
}

impl std::error::Error for BackbufferError {}

fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .copied()
        .map(str::to_owned)
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceAcquireFailure {
    Timeout,
    Occluded,
    Outdated,
    Lost,
    Validation,
}

/// Trait for render targets (backbuffers, offscreen textures, etc.)
pub trait RenderTarget {
    fn format(&self) -> TextureFormat;
    fn size(&self) -> (u32, u32);
    /// Returns the view and optionally something that must be presented (swapchain)
    fn acquire(&mut self, device: &Device) -> Option<(TextureView, Option<Presentable>)>;
    fn resize(&mut self, device: &Device, w: u32, h: u32);
    /// Only useful for backbuffer targets
    fn set_vsync(&mut self, _device: &Device, _on: bool) {}
}

fn android_gl_fallback_surface_format() -> TextureFormat {
    TextureFormat::Rgba8Unorm
}

fn android_gl_fallback_surface_config(w: u32, h: u32) -> Result<(SurfaceConfiguration, TextureFormat, bool), BackbufferError> {
    if w == 0 || h == 0 {
        return Err(BackbufferError::ZeroSize { width: w, height: h });
    }

    let format = android_gl_fallback_surface_format();
    let config = SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT,
        format,
        width: w,
        height: h,
        present_mode: vsync_present_mode(),
        desired_maximum_frame_latency: DESIRED_MAXIMUM_FRAME_LATENCY,
        alpha_mode: CompositeAlphaMode::Auto,
        view_formats: Vec::new(),
    };
    Ok((config, format, false))
}

#[cfg(target_os = "android")]
fn is_android_gl(adapter: &Adapter) -> bool {
    adapter.get_info().backend == wgpu::Backend::Gl
}

#[cfg(not(target_os = "android"))]
fn is_android_gl(_adapter: &Adapter) -> bool {
    false
}

pub(crate) fn surface_config(
    surface: &Surface<'_>,
    adapter: &Adapter,
    w: u32,
    h: u32,
) -> Result<(SurfaceConfiguration, TextureFormat, bool), BackbufferError> {
    if w == 0 || h == 0 {
        return Err(BackbufferError::ZeroSize { width: w, height: h });
    }

    let caps = surface.get_capabilities(adapter);
    let mut config = surface
        .get_default_config(adapter, w, h)
        .ok_or(BackbufferError::UnsupportedSurface { width: w, height: h })?;
    config.present_mode = vsync_present_mode();
    config.desired_maximum_frame_latency = DESIRED_MAXIMUM_FRAME_LATENCY;
    if cfg!(debug_assertions) {
        log::info!(
            "[egor] surface capabilities: present_modes={:?} usages={:?} formats={:?}",
            caps.present_modes,
            caps.usages,
            caps.formats
        );
    }

    #[cfg(target_os = "android")]
    let surface_copy_src = false;
    #[cfg(not(target_os = "android"))]
    let surface_copy_src = caps.usages.contains(TextureUsages::COPY_SRC);
    if surface_copy_src {
        config.usage |= TextureUsages::COPY_SRC;
    }

    // Prefer sRGB rendering, but do it in the form each backend can present:
    //
    // * WebGPU wants a non-sRGB canvas format plus an sRGB surface view.
    // * GLES/WebGL and Android Vulkan often do not support surface view
    //   formats, so use the sRGB surface format directly when it is exposed.
    // * If neither route exists, fall back to the surface format itself.
    let srgb_format = config.format.add_srgb_suffix();
    let can_surface_view_format = adapter
        .get_downlevel_capabilities()
        .flags
        .contains(DownlevelFlags::SURFACE_VIEW_FORMATS);
    let view_format = if srgb_format == config.format {
        config.format
    } else if !can_surface_view_format && caps.formats.contains(&srgb_format) {
        config.format = srgb_format;
        srgb_format
    } else if can_surface_view_format {
        config.view_formats.push(srgb_format);
        srgb_format
    } else {
        config.format
    };

    Ok((config, view_format, surface_copy_src))
}

pub(crate) fn surface_config_with_android_gl_fallback(
    surface: &Surface<'_>,
    adapter: &Adapter,
    w: u32,
    h: u32,
) -> Result<(SurfaceConfiguration, TextureFormat, bool), BackbufferError> {
    match surface_config(surface, adapter, w, h) {
        Ok(config) => Ok(config),
        Err(error @ BackbufferError::ZeroSize { .. }) => Err(error),
        Err(error) if is_android_gl(adapter) => {
            log::warn!(
                "[egor] Android GL surface capability query failed ({error}); falling back to {:?}",
                android_gl_fallback_surface_format()
            );
            android_gl_fallback_surface_config(w, h)
        }
        Err(error) => Err(error),
    }
}

/// Renders to the window's backbuffer (swapchain)
pub struct Backbuffer {
    surface: Surface<'static>,
    config: SurfaceConfiguration,
    /// The sRGB variant of the surface format, used for pipeline targets and
    /// texture views so that linear framebuffer values get gamma-encoded on
    /// output. On native this usually equals `config.format`; on WebGPU the
    /// canvas context only accepts the non-sRGB format but we create sRGB views
    /// via `view_formats`.
    view_format: TextureFormat,
    surface_copy_src: bool,
    last_acquire_failure: Option<SurfaceAcquireFailure>,
}

impl Backbuffer {
    pub fn new(
        instance: &Instance,
        adapter: &Adapter,
        device: &Device,
        window: impl Into<SurfaceTarget<'static>> + WindowHandle,
        w: u32,
        h: u32,
    ) -> Self {
        Self::try_new(instance, adapter, device, window, w, h).expect("failed to create egor backbuffer")
    }

    pub fn try_new(
        instance: &Instance,
        adapter: &Adapter,
        device: &Device,
        window: impl Into<SurfaceTarget<'static>> + WindowHandle,
        w: u32,
        h: u32,
    ) -> Result<Self, BackbufferError> {
        log::info!("[egor] backbuffer init: creating surface {w}x{h}");
        let surface = instance
            .create_surface(window)
            .map_err(|error| BackbufferError::CreateSurface(error.to_string()))?;
        Self::try_from_surface(adapter, device, surface, w, h)
    }

    pub(crate) fn try_from_surface(
        adapter: &Adapter,
        device: &Device,
        surface: Surface<'static>,
        w: u32,
        h: u32,
    ) -> Result<Self, BackbufferError> {
        log::info!("[egor] backbuffer init: building surface config");
        let (config, view_format, surface_copy_src) = surface_config_with_android_gl_fallback(&surface, adapter, w, h)?;
        if cfg!(debug_assertions) {
            log::info!(
                "[egor] backbuffer init: configuring surface format={:?} view_format={:?} present_mode={:?} frame_latency={} usage={:?} copy_src={}",
                config.format,
                view_format,
                config.present_mode,
                config.desired_maximum_frame_latency,
                config.usage,
                surface_copy_src
            );
        }
        Self::configure_surface(&surface, device, &config)?;
        log::info!("[egor] backbuffer init: complete");
        Ok(Self {
            surface,
            config,
            view_format,
            surface_copy_src,
            last_acquire_failure: None,
        })
    }

    pub fn supports_copy_src(&self) -> bool {
        self.surface_copy_src
    }

    pub fn last_acquire_failure(&self) -> Option<SurfaceAcquireFailure> {
        self.last_acquire_failure
    }

    fn configure_surface(surface: &Surface<'_>, device: &Device, config: &SurfaceConfiguration) -> Result<(), BackbufferError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = device.poll(PollType::wait_indefinitely());
            let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let configure_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                surface.configure(device, config);
            }));
            if let Err(payload) = configure_result {
                drop(error_scope);
                return Err(BackbufferError::Configure(panic_payload_message(payload)));
            }
            if let Some(error) = pollster::block_on(error_scope.pop()) {
                return Err(BackbufferError::Configure(error.to_string()));
            }
            Ok(())
        }

        #[cfg(target_arch = "wasm32")]
        {
            surface.configure(device, config);
            Ok(())
        }
    }

    fn reconfigure(&self, device: &Device) -> Result<(), BackbufferError> {
        Self::configure_surface(&self.surface, device, &self.config)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn get_current_texture(&self) -> CurrentSurfaceTexture {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.surface.get_current_texture())) {
            Ok(texture) => texture,
            Err(payload) => {
                let panic_message = panic_payload_message(payload);
                log::error!("[egor] surface acquire panicked: {panic_message:?}");
                CurrentSurfaceTexture::Validation
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn get_current_texture(&self) -> CurrentSurfaceTexture {
        self.surface.get_current_texture()
    }

    fn record_acquire_success(&mut self) {
        if let Some(failure) = self.last_acquire_failure.take() {
            log::info!("[egor] surface acquire recovered after {failure:?}");
        }
    }

    fn record_acquire_failure(&mut self, failure: SurfaceAcquireFailure) {
        if self.last_acquire_failure != Some(failure) {
            log::warn!("[egor] surface acquire failed: {failure:?}");
            self.last_acquire_failure = Some(failure);
        }
    }
}

impl RenderTarget for Backbuffer {
    fn format(&self) -> TextureFormat {
        self.view_format
    }

    fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    fn acquire(&mut self, device: &Device) -> Option<(TextureView, Option<Presentable>)> {
        // On WASM the canvas can be resized externally (e.g. via a JS
        // ResizeObserver) without going through winit's resize event.
        // Detect the mismatch and reconfigure the surface before acquiring.
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(canvas) = Self::get_canvas_size() {
                let (cw, ch) = canvas;
                if cw > 0 && ch > 0 && (cw != self.config.width || ch != self.config.height) {
                    self.resize(device, cw, ch);
                }
            }
        }

        match self.get_current_texture() {
            CurrentSurfaceTexture::Success(surface_texture) | CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                self.record_acquire_success();
                let view = surface_texture.texture.create_view(&TextureViewDescriptor {
                    format: Some(self.view_format),
                    ..Default::default()
                });
                Some((view, Some(Presentable::Surface(surface_texture))))
            }
            CurrentSurfaceTexture::Outdated => {
                self.record_acquire_failure(SurfaceAcquireFailure::Outdated);
                self.resize(device, self.config.width, self.config.height);
                None
            }
            CurrentSurfaceTexture::Lost => {
                self.record_acquire_failure(SurfaceAcquireFailure::Lost);
                if let Err(error) = self.reconfigure(device) {
                    log::warn!("[egor] surface reconfigure after Lost failed: {error:?}");
                }
                None
            }
            CurrentSurfaceTexture::Timeout => {
                self.record_acquire_failure(SurfaceAcquireFailure::Timeout);
                None
            }
            CurrentSurfaceTexture::Occluded => {
                self.record_acquire_failure(SurfaceAcquireFailure::Occluded);
                None
            }
            CurrentSurfaceTexture::Validation => {
                self.record_acquire_failure(SurfaceAcquireFailure::Validation);
                if let Err(error) = self.reconfigure(device) {
                    log::warn!("[egor] surface reconfigure after Validation failed: {error:?}");
                }
                None
            }
        }
    }

    fn resize(&mut self, device: &Device, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        if self.config.width == w && self.config.height == h {
            return;
        }
        (self.config.width, self.config.height) = (w, h);
        if let Err(error) = self.reconfigure(device) {
            log::warn!("[egor] surface resize configure failed: {error:?}");
        }
    }

    fn set_vsync(&mut self, device: &Device, on: bool) {
        let present_mode = if on { vsync_present_mode() } else { PresentMode::AutoNoVsync };
        if self.config.present_mode == present_mode {
            return;
        }
        self.config.present_mode = present_mode;
        if cfg!(debug_assertions) {
            log::info!(
                "[egor] surface vsync change: enabled={} present_mode={:?}",
                on,
                self.config.present_mode
            );
        }
        if let Err(error) = self.reconfigure(device) {
            log::warn!("[egor] surface vsync configure failed: {error:?}");
        }
    }
}

impl Backbuffer {
    /// Read the canvas element's physical pixel dimensions directly from the DOM.
    #[cfg(target_arch = "wasm32")]
    fn get_canvas_size() -> Option<(u32, u32)> {
        let document = web_sys::window()?.document()?;
        let canvas = document.query_selector("canvas").ok()??;
        let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into().ok()?;
        let w = canvas.width();
        let h = canvas.height();
        Some((w, h))
    }
}

/// Renders to an offscreen texture that can be read back or used as a texture
pub struct OffscreenTarget {
    render_texture: Texture,
    render_view: TextureView,
    sample_texture: Texture,
    sample_view: TextureView,
    _depth_texture: Texture,
    depth_view: TextureView,
    format: TextureFormat,
    width: u32,
    height: u32,
}

impl OffscreenTarget {
    pub fn new(device: &Device, width: u32, height: u32, format: TextureFormat) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let render_texture = device.create_texture(&TextureDescriptor {
            label: Some("Offscreen Render Texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let sample_texture = device.create_texture(&TextureDescriptor {
            label: Some("Offscreen Sample Texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let (depth_texture, depth_view) = crate::Renderer::create_depth_texture(device, width, height);

        let render_view = render_texture.create_view(&Default::default());
        let sample_view = sample_texture.create_view(&Default::default());

        Self {
            render_texture,
            render_view,
            sample_texture,
            sample_view,
            _depth_texture: depth_texture,
            depth_view,
            format,
            width,
            height,
        }
    }

    pub fn texture(&self) -> &Texture {
        &self.sample_texture
    }

    pub fn view(&self) -> &TextureView {
        &self.sample_view
    }

    pub fn render_view(&self) -> &TextureView {
        &self.render_view
    }

    pub fn offscreen_depth_view(&self) -> &TextureView {
        &self.depth_view
    }

    /// Copy render texture into sample texture so it can be sampled
    pub fn copy_to_sample(&self, encoder: &mut CommandEncoder) {
        encoder.copy_texture_to_texture(
            self.render_texture.as_image_copy(),
            self.sample_texture.as_image_copy(),
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }
}

impl RenderTarget for OffscreenTarget {
    fn format(&self) -> TextureFormat {
        self.format
    }

    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn acquire(&mut self, _: &Device) -> Option<(TextureView, Option<Presentable>)> {
        // no presentation needed for offscreen targets
        Some((self.render_view.clone(), None))
    }

    fn resize(&mut self, device: &Device, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);
        if self.width == w && self.height == h {
            return;
        }
        // recreate the texture with new dimensions
        *self = Self::new(device, w, h, self.format);
    }
}
