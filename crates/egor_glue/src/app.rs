use std::sync::Arc;

use web_time::{Duration, Instant};

use crate::{
    graphics::{Graphics, RenderTargetStore, ScreenCaptureState},
    primitives::PrimitiveBatch,
    profile_new_frame,
    surface_recovery::{
        DeviceLossAction, SurfaceFailure, SurfaceRecoveryAction, SurfaceRecoveryState, max_frame_interval, retry_interval_for_refresh,
    },
    text::TextRenderer,
};

#[cfg(not(target_os = "ios"))]
use crate::surface_recovery::frame_interval_for_fps;
#[cfg(target_os = "android")]
use egor_app::AndroidLifecycle;
#[cfg(not(any(target_arch = "wasm32", target_os = "android", target_os = "ios")))]
use egor_app::PhysicalPosition;
#[cfg(target_os = "ios")]
use egor_app::WindowExtIOS;
use egor_app::{
    AppConfig, AppHandler, AppRunner, ControlFlow, Fullscreen, PhysicalSize, StartCause, Window, WindowEvent, input::Input,
    time::FrameTimer,
};
use egor_render::{
    MemoryHints, Renderer, RendererBackendPreference, TextureFormat,
    instance::Instance,
    target::{Backbuffer, RenderTarget},
};

type UpdateFn = dyn FnMut(&mut FrameContext);
type UrgentEventsFn = dyn FnMut();
#[cfg(target_os = "android")]
type AndroidLifecycleFn = dyn FnMut(AndroidLifecycle);

#[cfg(target_os = "ios")]
fn software_frame_interval_for_fps_limit(
    _window: &Window,
    _native_refresh_rate_fps: Option<u16>,
    _fps: u16,
    _vsync: bool,
) -> Option<Duration> {
    None
}

#[cfg(not(target_os = "ios"))]
fn software_frame_interval_for_fps_limit(
    _window: &Window,
    _native_refresh_rate_fps: Option<u16>,
    fps: u16,
    _vsync: bool,
) -> Option<Duration> {
    Some(frame_interval_for_fps(fps))
}

#[cfg(target_os = "ios")]
fn hardware_vsync_enabled(vsync: bool, _fps_limit: Option<u16>) -> bool {
    vsync
}

#[cfg(not(target_os = "ios"))]
fn hardware_vsync_enabled(vsync: bool, fps_limit: Option<u16>) -> bool {
    vsync && fps_limit.is_none()
}

#[cfg(target_os = "ios")]
fn set_native_preferred_fps(window: &Window, fps: u16) {
    let fps = i32::from(fps.max(1));
    window.set_preferred_frames_per_second(fps);
    window.set_native_display_link_enabled(true);
}

#[cfg(not(target_os = "ios"))]
fn set_native_preferred_fps(_window: &Window, _fps: u16) {}

#[cfg(target_os = "ios")]
fn clear_native_preferred_fps(window: &Window, native_refresh_rate_fps: Option<u16>) {
    // MTKView treats preferredFramesPerSecond as an actual rate, not as an
    // optional limit. Restore the display's maximum rate explicitly: passing
    // zero through winit is clamped to 1 FPS and leaves the app stuck there.
    let native_fps = refresh_rate_fps(window, native_refresh_rate_fps).unwrap_or(60);
    set_native_preferred_fps(window, native_fps);
}

#[cfg(not(target_os = "ios"))]
fn clear_native_preferred_fps(_window: &Window, _native_refresh_rate_fps: Option<u16>) {}

#[cfg(target_os = "ios")]
fn set_native_redraw_enabled(window: &Window, enabled: bool) {
    window.set_native_display_link_enabled(enabled);
}

#[cfg(not(target_os = "ios"))]
fn set_native_redraw_enabled(_window: &Window, _enabled: bool) {}

fn refresh_rate_fps(window: &Window, native_refresh_rate_fps: Option<u16>) -> Option<u16> {
    native_refresh_rate_fps.or_else(|| {
        window
            .current_monitor()
            .and_then(|monitor| monitor.refresh_rate_millihertz())
            .filter(|refresh_rate_millihertz| *refresh_rate_millihertz > 0)
            .map(|refresh_rate_millihertz| {
                let fps = (refresh_rate_millihertz + 500) / 1000;
                fps.clamp(1, u32::from(u16::MAX)) as u16
            })
    })
}

fn surface_acquire_retry_interval(window: &Window, native_refresh_rate_fps: Option<u16>, consecutive_failures: u32) -> Duration {
    retry_interval_for_refresh(refresh_rate_fps(window, native_refresh_rate_fps), consecutive_failures)
}

fn surface_wait_retry_interval() -> Duration {
    Duration::from_millis(100)
}

fn should_wait_for_surface_restore(is_minimized: bool, size: PhysicalSize<u32>) -> bool {
    is_minimized || size.width == 0 || size.height == 0
}

fn backbuffer_format_matches_renderer(backbuffer_format: TextureFormat, renderer_format: TextureFormat) -> bool {
    backbuffer_format == renderer_format
}

fn window_surface_size(window: &Window) -> PhysicalSize<u32> {
    #[cfg(target_os = "ios")]
    {
        window.outer_size()
    }

    #[cfg(not(target_os = "ios"))]
    {
        window.inner_size()
    }
}

struct CaptureFrameTarget {
    _texture: egor_render::Texture,
    view: egor_render::wgpu::TextureView,
    width: u32,
    height: u32,
    format: TextureFormat,
}

impl CaptureFrameTarget {
    fn new(device: &egor_render::Device, width: u32, height: u32, format: TextureFormat) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let texture = device.create_texture(&egor_render::wgpu::TextureDescriptor {
            label: Some("Screen Capture Frame Target"),
            size: egor_render::wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: egor_render::wgpu::TextureDimension::D2,
            format,
            usage: egor_render::wgpu::TextureUsages::RENDER_ATTACHMENT | egor_render::wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        Self {
            _texture: texture,
            view,
            width,
            height,
            format,
        }
    }

    fn ensure(slot: &mut Option<Self>, device: &egor_render::Device, width: u32, height: u32, format: TextureFormat) {
        let width = width.max(1);
        let height = height.max(1);
        let needs_new = slot
            .as_ref()
            .is_none_or(|target| target.width != width || target.height != height || target.format != format);
        if needs_new {
            *slot = Some(Self::new(device, width, height, format));
        }
    }

    fn view(&self) -> &egor_render::wgpu::TextureView {
        &self.view
    }
}

struct WatchFrameTarget {
    _color_texture: egor_render::Texture,
    color_view: egor_render::wgpu::TextureView,
    _overlay_texture: egor_render::Texture,
    overlay_view: egor_render::wgpu::TextureView,
    width: u32,
    height: u32,
    format: TextureFormat,
}

impl WatchFrameTarget {
    fn new(device: &egor_render::Device, width: u32, height: u32, format: TextureFormat) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let color_texture = device.create_texture(&egor_render::wgpu::TextureDescriptor {
            label: Some("Watch Frame Color Target"),
            size: egor_render::wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: egor_render::wgpu::TextureDimension::D2,
            format,
            usage: egor_render::wgpu::TextureUsages::RENDER_ATTACHMENT | egor_render::wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&Default::default());

        let overlay_texture = device.create_texture(&egor_render::wgpu::TextureDescriptor {
            label: Some("Watch Frame Dynamic Overlay Target"),
            size: egor_render::wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: egor_render::wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: egor_render::wgpu::TextureUsages::RENDER_ATTACHMENT | egor_render::wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let overlay_view = overlay_texture.create_view(&Default::default());

        Self {
            _color_texture: color_texture,
            color_view,
            _overlay_texture: overlay_texture,
            overlay_view,
            width,
            height,
            format,
        }
    }

    fn ensure(slot: &mut Option<Self>, device: &egor_render::Device, width: u32, height: u32, format: TextureFormat) {
        let width = width.max(1);
        let height = height.max(1);
        let needs_new = slot
            .as_ref()
            .is_none_or(|target| target.width != width || target.height != height || target.format != format);
        if needs_new {
            *slot = Some(Self::new(device, width, height, format));
        }
    }
}

pub struct AppControl<'a> {
    window: &'a Window,
    requested_size: Option<(u32, u32)>,
    requested_vsync: Option<bool>,
    requested_fps_limit: Option<Option<u16>>,
    native_refresh_rate_fps: Option<u16>,
    requested_native_refresh_rate_fps: Option<Option<u16>>,
    requested_renderer_backend: Option<RendererBackendPreference>,
    gpu_device_recreated: bool,
}

impl<'a> AppControl<'a> {
    /// Request the window to redraw its contents on the next frame
    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    /// Set the inner size of the window in physical pixels
    /// Returns the new size depending on platform
    pub fn set_size(&mut self, w: u32, h: u32) {
        let _ = self.window.request_inner_size(PhysicalSize::new(w, h));
        self.requested_size = Some((w, h));
    }

    /// Set the outer window position in physical desktop coordinates.
    #[cfg(not(any(target_arch = "wasm32", target_os = "android", target_os = "ios")))]
    pub fn set_position(&self, x: i32, y: i32) {
        self.window.set_outer_position(PhysicalPosition::new(x, y));
    }

    /// Return the outer window position in physical desktop coordinates.
    #[cfg(not(any(target_arch = "wasm32", target_os = "android", target_os = "ios")))]
    pub fn outer_position(&self) -> Option<(i32, i32)> {
        self.window.outer_position().ok().map(|pos| (pos.x, pos.y))
    }

    /// Enable or disable borderless fullscreen mode
    pub fn set_fullscreen(&self, enabled: bool) {
        self.window.set_fullscreen(enabled.then_some(Fullscreen::Borderless(None)));
    }

    /// Enable or disable vertical sync
    /// When enabled, frame presentation is synchronized to the display's refresh
    /// rate, preventing screen tearing
    pub fn set_vsync(&mut self, on: bool) {
        self.requested_vsync = Some(on);
    }

    /// Limit continuous redraws to the requested frames per second.
    pub fn set_fps_limit(&mut self, fps: u16) {
        self.requested_fps_limit = Some(Some(fps.max(1)));
    }

    /// Clear the explicit FPS limit and leave redraw pacing to the platform.
    pub fn clear_fps_limit(&mut self) {
        self.requested_fps_limit = Some(None);
    }

    /// Recreate the GPU renderer with the requested backend after this frame is presented.
    pub fn set_renderer_backend(&mut self, backend: RendererBackendPreference) {
        self.requested_renderer_backend = Some(backend);
    }

    /// Override the detected native refresh rate for platforms whose winit
    /// monitor backend cannot report it yet, such as Android in our current
    /// winit fork. This lets the app report a value from its native shell
    /// without adding JNI/platform code to Egor or winit.
    pub fn set_native_refresh_rate_fps(&mut self, fps: Option<u16>) {
        let fps = fps.filter(|fps| *fps > 0);
        self.native_refresh_rate_fps = fps;
        self.requested_native_refresh_rate_fps = Some(fps);
    }

    /// Returns the current display refresh rate rounded to frames per second, if known.
    pub fn refresh_rate_fps(&self) -> Option<u16> {
        refresh_rate_fps(self.window, self.native_refresh_rate_fps)
    }

    /// Returns true on the first frame after the GPU device and renderer were recreated.
    pub fn gpu_device_recreated(&self) -> bool {
        self.gpu_device_recreated
    }

    /// Returns the window's DPI scale factor
    pub fn scale_factor(&self) -> f64 {
        self.window.scale_factor()
    }
}

pub struct FrameContext<'a> {
    pub events: Vec<WindowEvent>,
    pub app: AppControl<'a>,
    pub gfx: Graphics<'a>,
    pub input: &'a mut Input,
    pub timer: &'a FrameTimer,
    pub last_frame_stats: FrameStats,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameStats {
    /// GPU draw calls emitted by egor's main renderer in the previous frame.
    pub draw_calls: u32,
    /// Full egor frame time, including the user callback and egor render/present work.
    pub egor_frame_time: Duration,
    /// Time spent in the user update callback.
    pub user_callback_time: Duration,
    /// Egor work between the user callback and surface acquisition.
    pub prep_time: Duration,
    /// Time spent acquiring the current surface texture.
    pub surface_acquire_time: Duration,
    /// Time spent encoding draw calls into render passes.
    pub render_pass_time: Duration,
    /// Time spent encoding optional screen capture work.
    pub screen_capture_time: Duration,
    /// Time spent finishing the command encoder.
    pub finish_encoder_time: Duration,
    /// Time spent submitting command buffers to the GPU queue.
    pub queue_submit_time: Duration,
    /// Time spent presenting the backbuffer.
    pub present_time: Duration,
    /// Time spent after present handling readback/resizes/pacing changes.
    pub post_present_time: Duration,
}

impl FrameStats {
    pub fn egor_outside_callback_time(self) -> Duration {
        self.egor_frame_time.saturating_sub(self.user_callback_time)
    }
}

pub struct App {
    events: Vec<WindowEvent>,
    events_drained: Vec<WindowEvent>,
    update: Option<Box<UpdateFn>>,
    urgent_events: Option<Box<UrgentEventsFn>>,
    #[cfg(target_os = "android")]
    android_lifecycle: Option<Box<AndroidLifecycleFn>>,
    config: Option<AppConfig>,
    vsync: bool,
    text_renderer: Option<TextRenderer>,
    backbuffer: Option<Backbuffer>,
    window: Option<Arc<Window>>,
    primitive_batch: PrimitiveBatch,
    memory_hints: MemoryHints,
    renderer_backend: RendererBackendPreference,
    render_targets: RenderTargetStore,
    screen_capture: ScreenCaptureState,
    prewarm_watch_capture: bool,
    fps_limit: Option<u16>,
    native_refresh_rate_fps: Option<u16>,
    capture_frame_target: Option<CaptureFrameTarget>,
    watch_frame_target: Option<WatchFrameTarget>,
    watch_overlay_capture_unsupported_logged: bool,
    offscreen_batches: Vec<PrimitiveBatch>,
    instance_byte_offsets: Vec<u64>,
    surface_acquire_retry_interval: Option<Duration>,
    surface_recovery: SurfaceRecoveryState,
    waiting_for_surface_change: bool,
    window_focused: bool,
    hidden_window: bool,
    surface_occluded: bool,
    app_suspended: bool,
    frame_timer_reset_requested: bool,
    renderer_recreate_requested: bool,
    renderer_recreate_window_requested: bool,
    renderer_recreate_in_progress: bool,
    gpu_device_recreated_pending_frame: bool,
    last_frame_stats: FrameStats,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Create a new [`App`]
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            events_drained: Vec::new(),
            update: None,
            urgent_events: None,
            #[cfg(target_os = "android")]
            android_lifecycle: None,
            config: Some(AppConfig::default()),
            vsync: true,
            text_renderer: None,
            backbuffer: None,
            window: None,
            memory_hints: MemoryHints::Performance,
            renderer_backend: RendererBackendPreference::Auto,
            primitive_batch: PrimitiveBatch::default(),
            render_targets: RenderTargetStore::new(),
            screen_capture: ScreenCaptureState::new(),
            prewarm_watch_capture: false,
            fps_limit: None,
            native_refresh_rate_fps: None,
            capture_frame_target: None,
            watch_frame_target: None,
            watch_overlay_capture_unsupported_logged: false,
            offscreen_batches: Vec::new(),
            instance_byte_offsets: Vec::new(),
            surface_acquire_retry_interval: None,
            surface_recovery: SurfaceRecoveryState::new(),
            waiting_for_surface_change: false,
            window_focused: true,
            hidden_window: false,
            surface_occluded: false,
            app_suspended: false,
            frame_timer_reset_requested: true,
            renderer_recreate_requested: false,
            renderer_recreate_window_requested: false,
            renderer_recreate_in_progress: false,
            gpu_device_recreated_pending_frame: false,
            last_frame_stats: FrameStats::default(),
        }
    }

    /// Set application title
    pub fn title(mut self, title: &str) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.title = title.into();
        }
        self
    }

    /// Render in an invisible, inactive desktop window for GPU tests.
    #[cfg(not(any(target_arch = "wasm32", target_os = "android", target_os = "ios")))]
    pub fn hidden(mut self) -> Self {
        if let Some(config) = self.config.as_mut() {
            config.visible = false;
        }
        self.hidden_window = true;
        self
    }

    /// Prepare watch shaders during renderer initialization, before gameplay
    /// frames begin. Capture textures and staging buffers remain on demand.
    pub fn prewarm_watch_capture(mut self) -> Self {
        self.prewarm_watch_capture = true;
        self
    }

    /// Set window icon
    pub fn icon(mut self, icon: egor_app::Icon) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.icon = Some(icon);
        }
        self
    }

    /// Set window size (width, height in pixels)
    pub fn window_size(mut self, width: u32, height: u32) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.width = Some(width);
            c.height = Some(height);
        }
        self
    }

    /// Set initial outer window position in physical desktop coordinates.
    #[cfg(not(any(target_arch = "wasm32", target_os = "android", target_os = "ios")))]
    pub fn window_position(mut self, x: i32, y: i32) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.position = Some((x, y));
        }
        self
    }

    /// Set the minimum allowed window size (width, height in pixels).
    /// Outside of mobile platforms, the window will not resize below these constraints
    pub fn min_size(mut self, w: u32, h: u32) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.min_size = Some((w, h));
        }
        self
    }

    /// Set the maximum allowed window size (width, height in pixels).
    /// Outside of mobile platforms, the window will not resize above these constraints
    pub fn max_size(mut self, w: u32, h: u32) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.max_size = Some((w, h));
        }
        self
    }

    /// Enable or disable window resizing (defaults to true)
    pub fn resizable(mut self, resizable: bool) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.resizable = resizable;
        }
        self
    }

    /// Enable or disable window maximized (defaults to false)
    pub fn maximized(mut self, maximized: bool) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.maximized = maximized;
        }
        self
    }

    /// Enable or disable fullscreen (defaults to false)
    pub fn fullscreen(mut self, fullscreen: bool) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.fullscreen = fullscreen;
        }
        self
    }

    /// Enable or disable window decorations (defaults to true)
    pub fn decorations(mut self, decorations: bool) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.decorations = decorations;
        }
        self
    }

    /// Enable or disable vsync
    pub fn vsync(mut self, enabled: bool) -> Self {
        self.vsync = enabled;
        self
    }

    /// Set the maximum redraw rate while using `ControlFlow::Poll`.
    /// Desktop platforms use software pacing for explicit FPS limits. Platforms
    /// with native dynamic frame-rate support may apply the limit in the driver.
    pub fn fps_limit(mut self, fps: u16) -> Self {
        self.fps_limit = Some(fps.max(1));
        self
    }

    /// Set the event loop control flow (defaults to [`ControlFlow::Poll`])
    ///
    /// - `ControlFlow::Poll`: continuously redraws (game-style loop)
    /// - `ControlFlow::Wait`: no frames are produced unless
    ///   [`AppControl::request_redraw()`] is called
    ///
    /// When using `Wait`, you are responsible for requesting redraws
    pub fn control_flow(mut self, control_flow: ControlFlow) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.control_flow = control_flow;
        }
        self
    }

    /// Configure wgpu device memory allocation strategy.
    /// Affects GPU sub-allocation block sizes, useful for mobile or low end devices.
    /// See [`MemoryHints`] for more
    pub fn memory_hints(mut self, hints: MemoryHints) -> Self {
        self.memory_hints = hints;
        self
    }

    /// Select which wgpu backend set the renderer should enable.
    pub fn renderer_backend(mut self, backend: RendererBackendPreference) -> Self {
        self.renderer_backend = backend;
        self
    }

    /// Set the vertex and index buffer limits for the main frame batch.
    /// Defaults to [`egor_render::batch::GeometryBatch::DEFAULT_MAX_VERTICES`] and [`egor_render::batch::GeometryBatch::DEFAULT_MAX_INDICES`].
    /// Reduce these on memory-constrained platforms, or increase for scenes with dense geometry.
    pub fn batch_limits(mut self, max_verts: usize, max_indices: usize) -> Self {
        self.primitive_batch = PrimitiveBatch::new(max_verts, max_indices);
        self
    }

    /// When enabled, left mouse button presses/moves/releases generate touch events with id 0.
    /// Useful for testing touch logic on desktop.
    pub fn simulate_touch_with_mouse(mut self, enabled: bool) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.simulate_touch_with_mouse = enabled;
        }
        self
    }

    /// When enabled, the first active touch generates mouse position, delta, and left-button events.
    /// Useful on mobile to make existing mouse-based code work with touch.
    pub fn simulate_mouse_with_touch(mut self, enabled: bool) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.simulate_mouse_with_touch = enabled;
        }
        self
    }

    /// Run urgent platform work on the app/event-loop thread outside the frame callback.
    pub fn urgent_events(mut self, callback: impl FnMut() + 'static) -> Self {
        self.urgent_events = Some(Box::new(callback));
        self
    }

    /// Run Android activity lifecycle work on the app/event-loop thread.
    #[cfg(target_os = "android")]
    pub fn android_lifecycle(mut self, callback: impl FnMut(AndroidLifecycle) + 'static) -> Self {
        self.android_lifecycle = Some(Box::new(callback));
        self
    }

    /// Run the app with a per-frame update closure
    pub fn run(mut self, #[allow(unused_mut)] mut update: impl FnMut(&mut FrameContext) + 'static) {
        #[cfg(all(feature = "hot_reload", not(target_arch = "wasm32")))]
        let update = {
            dioxus_devtools::connect_subsecond();

            move |ctx: &mut FrameContext| {
                dioxus_devtools::subsecond::call(|| update(ctx));
            }
        };
        self.update = Some(Box::new(update));

        let config = self.config.take().unwrap();
        AppRunner::new(self, config).run();
    }

    fn run_urgent_events(&mut self) {
        if let Some(urgent_events) = self.urgent_events.as_mut() {
            urgent_events();
        }
    }

    fn finish_frame_stats(&mut self, mut frame_stats: FrameStats, frame_started_at: Instant) {
        frame_stats.egor_frame_time = frame_started_at.elapsed();
        self.last_frame_stats = frame_stats;
    }

    fn recreate_backbuffer(&mut self, renderer: &mut Renderer) -> bool {
        if self.waiting_for_surface_change {
            return false;
        }
        if self.app_suspended {
            return false;
        }

        let Some(window) = self.window.as_ref().cloned() else {
            log::warn!("[egor] cannot recreate backbuffer because no window handle is available");
            return false;
        };

        let size = window_surface_size(&window);
        if size.width == 0 || size.height == 0 {
            self.surface_recovery.record_surface_failure(SurfaceFailure::ZeroSizedSurface);
            return false;
        }

        self.backbuffer = None;
        let mut backbuffer = match renderer.take_startup_backbuffer(size.width, size.height) {
            Some(Ok(backbuffer)) => backbuffer,
            Some(Err(error)) => {
                log::warn!("[egor] backbuffer recreation failed: {error:?}");
                return false;
            }
            None => match Backbuffer::try_new(
                renderer.instance(),
                renderer.adapter(),
                renderer.device(),
                window,
                size.width,
                size.height,
            ) {
                Ok(backbuffer) => backbuffer,
                Err(error) => {
                    log::warn!("[egor] backbuffer recreation failed: {error:?}");
                    return false;
                }
            },
        };
        let backbuffer_format = backbuffer.format();
        let renderer_format = renderer.surface_format();
        if !backbuffer_format_matches_renderer(backbuffer_format, renderer_format) {
            log::warn!(
                "[egor] backbuffer format changed during surface recovery: renderer={renderer_format:?} backbuffer={backbuffer_format:?}; recreating renderer"
            );
            self.request_renderer_recreation("backbuffer format changed during surface recovery");
            return false;
        }
        backbuffer.set_vsync(renderer.device(), hardware_vsync_enabled(self.vsync, self.fps_limit));

        self.backbuffer = Some(backbuffer);
        self.waiting_for_surface_change = false;
        renderer.ensure_depth_size(size.width, size.height);
        if let Some(text_renderer) = self.text_renderer.as_mut() {
            text_renderer.resize(size.width, size.height, renderer.queue());
        }

        log::warn!(
            "[egor] recreated backbuffer after surface acquire failure at {}x{}",
            size.width,
            size.height
        );

        true
    }

    fn drop_renderer_owned_resources(&mut self) {
        self.backbuffer = None;
        self.text_renderer = None;
        self.render_targets = RenderTargetStore::new();
        self.screen_capture = ScreenCaptureState::new();
        self.capture_frame_target = None;
        self.watch_frame_target = None;
        self.watch_overlay_capture_unsupported_logged = false;
        self.primitive_batch.drop_gpu_resources();
        self.offscreen_batches.clear();
        self.instance_byte_offsets.clear();
        self.surface_acquire_retry_interval = Some(Duration::from_millis(1000));
    }

    fn request_renderer_recreation(&mut self, reason: &str) {
        if !self.renderer_recreate_requested {
            log::error!("[egor] requesting GPU renderer recreation: {reason:?}");
        }
        self.renderer_recreate_requested = true;
        self.drop_renderer_owned_resources();
    }

    fn request_renderer_backend_change(&mut self, backend: RendererBackendPreference) {
        if self.renderer_backend == backend {
            return;
        }

        log::info!("[egor] changing renderer backend from {:?} to {:?}", self.renderer_backend, backend);
        self.renderer_backend = backend;
        self.renderer_recreate_requested = true;
        self.frame_timer_reset_requested = true;
        // WGL permanently sets a Win32 window's pixel format. A fresh HWND is
        // required before another API (notably DX12) can create its swapchain.
        self.renderer_recreate_window_requested = cfg!(any(target_arch = "wasm32", target_os = "windows"));
        self.drop_renderer_owned_resources();
    }

    async fn create_renderer_with_retry(&self, window: Arc<Window>) -> Renderer {
        loop {
            match Renderer::try_new_with_backend(window.clone(), &self.memory_hints, self.renderer_backend).await {
                Ok(renderer) => return renderer,
                Err(error) => {
                    log::error!("[egor] renderer init failed: {error:?}; retrying");
                    #[cfg(not(target_arch = "wasm32"))]
                    std::thread::sleep(std::time::Duration::from_millis(1000));
                    #[cfg(target_arch = "wasm32")]
                    panic!("failed to initialize egor renderer after GPU recovery: {error:?}");
                }
            }
        }
    }
}

impl AppHandler<Renderer> for App {
    fn new_events(&mut self, _cause: StartCause) {
        self.run_urgent_events();
    }

    fn about_to_wait(&mut self) {
        self.run_urgent_events();
    }

    #[cfg(target_os = "android")]
    fn android_lifecycle(&mut self, lifecycle: AndroidLifecycle) {
        log::info!("[egor] android lifecycle: {lifecycle:?}");
        if let Some(android_lifecycle) = self.android_lifecycle.as_mut() {
            android_lifecycle(lifecycle);
        }
    }

    fn on_window_event(&mut self, _window: &Window, event: &WindowEvent) {
        match event {
            WindowEvent::Focused(focused) => {
                self.window_focused = *focused;
                if *focused {
                    #[cfg(not(target_os = "android"))]
                    {
                        self.waiting_for_surface_change = false;
                    }
                }
            }
            WindowEvent::Occluded(occluded) => {
                self.surface_occluded = *occluded;
                if *occluded {
                    self.backbuffer = None;
                    self.surface_recovery
                        .record_surface_failure(SurfaceFailure::Acquire(egor_render::target::SurfaceAcquireFailure::Occluded));
                    self.surface_acquire_retry_interval = Some(Duration::from_millis(100));
                } else {
                    self.waiting_for_surface_change = false;
                }
            }
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                self.waiting_for_surface_change = false;
            }
            _ => {}
        }
        self.events.push(event.clone());
    }

    fn resource_recreate_requested(&self) -> bool {
        self.renderer_recreate_requested
    }

    fn window_recreate_requested(&self) -> bool {
        self.renderer_recreate_window_requested
    }

    fn frame_timer_reset_requested(&self) -> bool {
        self.frame_timer_reset_requested
    }

    fn before_resource_recreate(&mut self) {
        self.renderer_recreate_requested = false;
        if self.renderer_recreate_window_requested {
            self.window = None;
        }
        self.renderer_recreate_window_requested = false;
        self.renderer_recreate_in_progress = true;
        self.drop_renderer_owned_resources();
    }

    async fn with_resource(&mut self, window: Arc<Window>) -> Renderer {
        // WebGPU throws error 'size is zero' if not set
        let size = window_surface_size(&window);
        let (w, h) = (
            if size.width == 0 { 800 } else { size.width },
            if size.height == 0 { 600 } else { size.height },
        );
        log::info!("[egor] app resource init: start {w}x{h}");
        let mut renderer = self.create_renderer_with_retry(window.clone()).await;
        log::info!("[egor] app resource init: renderer ready");
        self.window = Some(window.clone());
        #[cfg(target_os = "android")]
        {
            let _ = (w, h);
            log::info!("[egor] app resource init: deferring first backbuffer until redraw");
            self.backbuffer = None;
        }
        #[cfg(not(target_os = "android"))]
        {
            log::info!("[egor] app resource init: creating first backbuffer");
            self.backbuffer = match renderer.take_startup_backbuffer(w, h) {
                Some(Ok(backbuffer)) => Some(backbuffer),
                Some(Err(error)) => {
                    log::warn!("[egor] initial backbuffer creation failed: {error:?}");
                    None
                }
                None => match Backbuffer::try_new(renderer.instance(), renderer.adapter(), renderer.device(), window, w, h) {
                    Ok(backbuffer) => Some(backbuffer),
                    Err(error) => {
                        log::warn!("[egor] initial backbuffer creation failed: {error:?}");
                        None
                    }
                },
            };
        }
        log::info!("[egor] app resource init: complete");
        renderer
    }

    fn on_ready(&mut self, window: &Window, renderer: &mut Renderer) {
        log::info!("[egor] app ready: configuring render resources");
        let renderer_was_recreated = self.renderer_recreate_in_progress;
        self.renderer_recreate_in_progress = false;
        if renderer_was_recreated {
            self.gpu_device_recreated_pending_frame = true;
            log::warn!("[egor] GPU renderer recreation complete");
        }
        let (device, format) = (
            renderer.device(),
            self.backbuffer
                .as_ref()
                .map(RenderTarget::format)
                .unwrap_or_else(|| renderer.surface_format()),
        );
        if let Some(backbuffer) = self.backbuffer.as_mut() {
            backbuffer.set_vsync(device, hardware_vsync_enabled(self.vsync, self.fps_limit));
        }
        self.text_renderer = Some(TextRenderer::new(device, renderer.queue(), format));
        if self.prewarm_watch_capture && renderer.supports_watch_overlay_capture() {
            self.screen_capture.prewarm_watch_pipelines(device, format);
        }
        if let Some(fps_limit) = self.fps_limit {
            set_native_preferred_fps(window, fps_limit);
        }

        let size = window_surface_size(window);
        if size.width > 0 && size.height > 0 && self.backbuffer.is_some() {
            self.resize(size.width, size.height, renderer);
        }
        log::info!("[egor] app ready: complete");
    }

    fn poll_frame_interval(&self, window: &Window) -> Option<Duration> {
        let fps_limit_interval = self
            .fps_limit
            .and_then(|fps_limit| software_frame_interval_for_fps_limit(window, self.native_refresh_rate_fps, fps_limit, self.vsync));
        let background_interval =
            (!self.hidden_window && (!self.window_focused || self.surface_occluded)).then_some(Duration::from_millis(100));

        max_frame_interval(
            max_frame_interval(self.surface_acquire_retry_interval, fps_limit_interval),
            background_interval,
        )
    }

    fn frame(&mut self, _window: &Window, renderer: &mut Renderer, input: &mut Input, timer: &FrameTimer) {
        let egor_frame_started_at = Instant::now();
        let mut frame_stats = FrameStats::default();

        profile_new_frame!();
        #[cfg(feature = "profiling")]
        profiling::scope!("frame");

        self.frame_timer_reset_requested = false;

        if self.update.is_none() {
            self.finish_frame_stats(frame_stats, egor_frame_started_at);
            return;
        }

        if self.app_suspended {
            self.frame_timer_reset_requested = true;
            self.backbuffer = None;
            self.surface_acquire_retry_interval = None;
            self.finish_frame_stats(frame_stats, egor_frame_started_at);
            return;
        }

        if self.surface_occluded && !self.hidden_window {
            self.frame_timer_reset_requested = true;
            self.surface_acquire_retry_interval = Some(surface_wait_retry_interval());
            self.finish_frame_stats(frame_stats, egor_frame_started_at);
            return;
        }

        if should_wait_for_surface_restore(_window.is_minimized().unwrap_or(false), window_surface_size(_window)) {
            self.frame_timer_reset_requested = true;
            self.backbuffer = None;
            self.surface_recovery.record_surface_failure(SurfaceFailure::ZeroSizedSurface);
            self.surface_acquire_retry_interval = Some(surface_wait_retry_interval());
            self.finish_frame_stats(frame_stats, egor_frame_started_at);
            return;
        }

        if renderer.device_lost() {
            let (action, should_log) = self.surface_recovery.record_device_lost();
            if should_log {
                log::error!("[egor] GPU device lost; recovery action: {action:?}");
            }
            match action {
                DeviceLossAction::PauseRendering => {
                    self.frame_timer_reset_requested = true;
                    self.backbuffer = None;
                    self.surface_acquire_retry_interval = Some(Duration::from_millis(1000));
                    self.finish_frame_stats(frame_stats, egor_frame_started_at);
                    return;
                }
                DeviceLossAction::RecreateRenderer => {
                    self.frame_timer_reset_requested = true;
                    self.request_renderer_recreation("wgpu device lost");
                    self.finish_frame_stats(frame_stats, egor_frame_started_at);
                    return;
                }
            }
        }

        // Drive wgpu map_async callbacks at the START of the frame.
        // By polling here (not at end-of-frame), the GPU has had a full
        // frame since begin_readback_map — virtually guaranteeing the
        // oldest ring-buffer slot is complete, eliminating stalls.
        if self.screen_capture.readback_in_flight() {
            let _ = renderer.device().poll(egor_render::wgpu::PollType::Poll);
        }

        if self.backbuffer.is_none() && should_wait_for_surface_restore(false, window_surface_size(_window)) {
            self.frame_timer_reset_requested = true;
            self.surface_recovery.record_surface_failure(SurfaceFailure::ZeroSizedSurface);
            self.surface_acquire_retry_interval = Some(surface_wait_retry_interval());
            self.finish_frame_stats(frame_stats, egor_frame_started_at);
            return;
        }

        if self.backbuffer.is_none() && self.waiting_for_surface_change {
            self.frame_timer_reset_requested = true;
            self.surface_acquire_retry_interval = Some(surface_wait_retry_interval());
            self.finish_frame_stats(frame_stats, egor_frame_started_at);
            return;
        }

        if self.backbuffer.is_none() && !self.recreate_backbuffer(renderer) {
            self.frame_timer_reset_requested = true;
            let action = self.surface_recovery.record_surface_failure(SurfaceFailure::BackbufferCreateFailed);
            match action {
                SurfaceRecoveryAction::WaitForResize => {
                    self.surface_acquire_retry_interval = Some(surface_wait_retry_interval());
                }
                SurfaceRecoveryAction::WaitForSurfaceChange => {
                    if !self.waiting_for_surface_change {
                        log::warn!("[egor] pausing backbuffer creation until Android reports a surface change");
                    }
                    self.waiting_for_surface_change = true;
                    self.surface_acquire_retry_interval = Some(surface_wait_retry_interval());
                }
                SurfaceRecoveryAction::SkipFrame | SurfaceRecoveryAction::RecreateBackbuffer => {
                    log::debug!("[egor] missing backbuffer recovery action: {action:?}");
                    self.surface_acquire_retry_interval = Some(surface_acquire_retry_interval(
                        _window,
                        self.native_refresh_rate_fps,
                        self.surface_recovery.consecutive_acquire_failures(),
                    ));
                }
            }
            self.finish_frame_stats(frame_stats, egor_frame_started_at);
            return;
        }

        let Some(backbuffer) = &mut self.backbuffer else {
            self.frame_timer_reset_requested = true;
            self.finish_frame_stats(frame_stats, egor_frame_started_at);
            return;
        };

        let (w, h) = backbuffer.size();
        renderer.ensure_depth_size(w, h);
        let (device, queue) = (renderer.device().clone(), renderer.queue().clone());
        let format = backbuffer.format();
        let text_renderer = self.text_renderer.as_mut().unwrap();
        let update = self.update.as_mut().unwrap();
        let gpu_device_recreated = self.gpu_device_recreated_pending_frame;

        self.events_drained.clear();
        std::mem::swap(&mut self.events, &mut self.events_drained);

        let (requested_size, requested_vsync, requested_fps_limit, requested_native_refresh_rate_fps, requested_renderer_backend) = {
            let mut ctx = FrameContext {
                events: std::mem::take(&mut self.events_drained),
                app: AppControl {
                    window: _window,
                    requested_size: None,
                    requested_vsync: None,
                    requested_fps_limit: None,
                    native_refresh_rate_fps: self.native_refresh_rate_fps,
                    requested_native_refresh_rate_fps: None,
                    requested_renderer_backend: None,
                    gpu_device_recreated,
                },
                gfx: Graphics::new(
                    renderer,
                    &mut self.primitive_batch,
                    text_renderer,
                    &mut self.render_targets,
                    &mut self.screen_capture,
                    &mut self.offscreen_batches,
                    format,
                    w,
                    h,
                ),
                input,
                timer,
                last_frame_stats: self.last_frame_stats,
            };

            {
                #[cfg(feature = "profiling")]
                profiling::scope!("user_callback");
                let user_callback_started_at = Instant::now();
                let update_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    update(&mut ctx);
                }));
                frame_stats.user_callback_time = user_callback_started_at.elapsed();
                if let Err(payload) = update_result {
                    let panic_message = payload
                        .downcast_ref::<&str>()
                        .copied()
                        .map(str::to_owned)
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "unknown panic".to_string());
                    drop(ctx);
                    if renderer.device_lost() {
                        log::error!("[egor] user frame callback panicked after GPU device loss: {panic_message:?}");
                        self.gpu_device_recreated_pending_frame = false;
                        self.request_renderer_recreation("user callback panicked after device loss");
                        self.finish_frame_stats(frame_stats, egor_frame_started_at);
                        return;
                    }
                    std::panic::resume_unwind(payload);
                }
            }
            self.gpu_device_recreated_pending_frame = false;

            ctx.events.clear();
            self.events_drained = ctx.events;

            let requested_size = ctx.app.requested_size;
            let requested_vsync = ctx.app.requested_vsync;
            let requested_fps_limit = ctx.app.requested_fps_limit;
            let requested_native_refresh_rate_fps = ctx.app.requested_native_refresh_rate_fps;
            let requested_renderer_backend = ctx.app.requested_renderer_backend;
            if let Some((pw, ph)) = requested_size {
                ctx.gfx.set_target_size(pw, ph);
            }

            ctx.gfx.upload_camera();

            (
                requested_size,
                requested_vsync,
                requested_fps_limit,
                requested_native_refresh_rate_fps,
                requested_renderer_backend,
            )
        };

        let prep_started_at = Instant::now();
        if self.screen_capture.take_buffers_released() {
            self.watch_frame_target = None;
            self.capture_frame_target = None;
        }

        if let Some(native_refresh_rate_fps) = requested_native_refresh_rate_fps {
            self.native_refresh_rate_fps = native_refresh_rate_fps;
        }

        let mut fps_limit_changed = false;
        if let Some(fps_limit) = requested_fps_limit {
            match fps_limit {
                Some(fps_limit) => {
                    if self.fps_limit != Some(fps_limit) {
                        set_native_preferred_fps(_window, fps_limit);
                        fps_limit_changed = true;
                    }
                    self.fps_limit = Some(fps_limit);
                }
                None => {
                    if self.fps_limit.is_some() {
                        clear_native_preferred_fps(_window, self.native_refresh_rate_fps);
                        fps_limit_changed = true;
                    }
                    self.fps_limit = None;
                }
            }
        }

        let has_text = text_renderer.has_entries();

        // Use the flag tracked during batch building instead of scanning all batches.
        let has_rt_overrides = self.primitive_batch.has_rt_overrides();

        let mut batches = self.primitive_batch.drain_all();
        let _batch_count = batches.len();

        // Write custom camera matrices to GPU slots 1..N.
        // Slot 0 is the default egor camera (already written by upload_camera()).
        let stride = renderer.camera_slot_stride();
        {
            let custom_cameras = self.primitive_batch.drain_camera_matrices();
            for (i, cam) in custom_cameras.iter().enumerate() {
                renderer.write_camera_slot((i as u32) + 1, *cam);
            }
        }

        {
            #[cfg(feature = "profiling")]
            profiling::scope!("batch_upload");
            // Pre-upload all batch geometry to GPU before starting any render pass.
            // This batches the write_buffer calls together for better cache/driver behavior
            // and makes the upload() calls inside draw_batch a no-op (dirty flags already cleared).
            let mut _total_verts: usize = 0;
            let mut _total_indices: usize = 0;
            let mut _total_instances: usize = 0;
            let mut _dirty_batches: usize = 0;
            let inst_size = std::mem::size_of::<Instance>();
            self.instance_byte_offsets.clear();
            let mut running_offset: usize = 0;
            for batch in &mut batches {
                _total_verts += batch.geometry.vertex_count();
                _total_indices += batch.geometry.index_count();
                _total_instances += batch.geometry.instance_count();
                if batch.geometry.is_dirty() {
                    _dirty_batches += 1;
                }
                self.instance_byte_offsets.push((running_offset * inst_size) as u64);
                running_offset += batch.geometry.instance_count();
                batch.geometry.upload_geometry_only(&device, &queue);
            }
            {
                let batch_instance_slices: Vec<&[Instance]> = batches.iter().map(|b| b.geometry.instances()).collect();
                renderer.upload_shared_instances_batched(&batch_instance_slices);
            }
        } // profile_scope batch_upload

        let mut capture_active = self.screen_capture.is_requested();
        let mut capture_source_render_target = self.screen_capture.requested_source_render_target();
        let mut use_watch_frame_target = self.screen_capture.is_watch_overlay_capture_requested();
        if use_watch_frame_target {
            let unsupported_capture = !renderer.supports_watch_overlay_capture();
            let unsupported_shader = batches
                .iter()
                .filter(|batch| batch.render_target.is_none())
                .find_map(|batch| (!renderer.supports_watch_overlay_pipeline(batch.shader_id)).then_some(batch.shader_id));
            if unsupported_capture || has_text || unsupported_shader.is_some() {
                if unsupported_capture {
                    if !self.watch_overlay_capture_unsupported_logged {
                        log::warn!("[egor] watch overlay capture skipped: backend does not support the dynamic overlay MRT path");
                        self.watch_overlay_capture_unsupported_logged = true;
                    }
                } else {
                    log::warn!(
                        "[egor] watch overlay capture skipped: unsupported main-pass source (has_text={}, shader={:?})",
                        has_text,
                        unsupported_shader.flatten()
                    );
                }
                self.screen_capture.cancel_request();
                capture_active = false;
                capture_source_render_target = None;
                use_watch_frame_target = false;
            } else {
                WatchFrameTarget::ensure(&mut self.watch_frame_target, &device, w, h, format);
            }
        }
        let use_capture_frame_target =
            capture_active && !use_watch_frame_target && capture_source_render_target.is_none() && !backbuffer.supports_copy_src();
        if use_capture_frame_target {
            CaptureFrameTarget::ensure(&mut self.capture_frame_target, &device, w, h, format);
        }
        let capture_frame_view = if use_capture_frame_target {
            self.capture_frame_target.as_ref().map(CaptureFrameTarget::view)
        } else {
            None
        };

        frame_stats.prep_time = prep_started_at.elapsed();

        let surface_acquire_started_at = Instant::now();
        let frame_result = match renderer.try_begin_frame(backbuffer) {
            Ok(frame_result) => frame_result,
            Err(error) => {
                frame_stats.surface_acquire_time = surface_acquire_started_at.elapsed();
                log::error!("[egor] begin frame failed: {error:?}");
                self.primitive_batch.recycle(batches);
                self.request_renderer_recreation("begin frame failed");
                self.finish_frame_stats(frame_stats, egor_frame_started_at);
                return;
            }
        };
        frame_stats.surface_acquire_time = surface_acquire_started_at.elapsed();

        let Some(mut frame) = frame_result else {
            let acquire_failure = backbuffer.last_acquire_failure();
            let action = self.surface_recovery.record_surface_failure(match acquire_failure {
                Some(failure) => SurfaceFailure::Acquire(failure),
                None => SurfaceFailure::ConfigureFailed,
            });
            self.surface_acquire_retry_interval = Some(surface_acquire_retry_interval(
                _window,
                self.native_refresh_rate_fps,
                self.surface_recovery.consecutive_acquire_failures(),
            ));
            self.primitive_batch.recycle(batches);
            match action {
                SurfaceRecoveryAction::RecreateBackbuffer => {
                    if self.recreate_backbuffer(renderer) {
                        self.surface_recovery.record_frame_acquired();
                        self.surface_acquire_retry_interval =
                            Some(surface_acquire_retry_interval(_window, self.native_refresh_rate_fps, 0));
                    }
                }
                SurfaceRecoveryAction::WaitForSurfaceChange => {
                    if !self.waiting_for_surface_change {
                        log::warn!("[egor] pausing surface recovery until Android reports a surface change");
                    }
                    self.waiting_for_surface_change = true;
                    self.surface_acquire_retry_interval = Some(surface_wait_retry_interval());
                }
                SurfaceRecoveryAction::SkipFrame | SurfaceRecoveryAction::WaitForResize => {}
            }
            self.finish_frame_stats(frame_stats, egor_frame_started_at);
            return;
        };
        self.surface_recovery.record_frame_acquired();
        self.waiting_for_surface_change = false;
        self.surface_acquire_retry_interval = None;

        let watch_frame_target = if use_watch_frame_target {
            self.watch_frame_target.as_ref()
        } else {
            None
        };
        let main_view = watch_frame_target
            .map(|target| &target.color_view)
            .or(capture_frame_view)
            .unwrap_or(&frame.view);
        // The watch color attachments have the backbuffer's dimensions and
        // replace its main pass, so they can share the existing depth target.
        let main_depth_view = renderer.depth_view();
        let watch_overlay_view = watch_frame_target.map(|target| &target.overlay_view);

        let render_pass_started_at = Instant::now();
        {
            #[cfg(feature = "profiling")]
            profiling::scope!("render_pass");
            if has_rt_overrides {
                // Multi-pass rendering: split ONLY on render_target changes.
                let mut current_rt: Option<usize> = None;
                let mut first_pass_on_backbuffer = true;
                let mut initialized_render_targets: Vec<usize> = Vec::new();

                let mut batch_start = 0;
                while batch_start < batches.len() {
                    let group_rt = batches[batch_start].render_target;
                    let mut batch_end = batch_start + 1;
                    while batch_end < batches.len() && batches[batch_end].render_target == group_rt {
                        batch_end += 1;
                    }

                    // Handle render-target transition
                    if current_rt != group_rt {
                        if let Some(prev_rt) = current_rt {
                            self.render_targets.get(prev_rt).copy_to_sample(&mut frame.encoder);
                        }
                        current_rt = group_rt;
                    }

                    let (view, depth_view, is_first) = if let Some(rt_id) = group_rt {
                        let rt = self.render_targets.get(rt_id);
                        let view = rt.render_view();
                        let dv = rt.offscreen_depth_view();
                        let is_first = !initialized_render_targets.contains(&rt_id);
                        if is_first {
                            initialized_render_targets.push(rt_id);
                        }
                        (view, dv, is_first)
                    } else {
                        let is_first = first_pass_on_backbuffer;
                        if is_first {
                            first_pass_on_backbuffer = false;
                        }
                        (main_view, main_depth_view, is_first)
                    };

                    let (rt_w, rt_h) = if let Some(rt_id) = group_rt {
                        self.render_targets.get(rt_id).size()
                    } else {
                        (w, h)
                    };

                    {
                        let watch_pass = use_watch_frame_target && group_rt.is_none();
                        let mut r_pass = if is_first {
                            if group_rt.is_some() {
                                renderer.begin_render_pass_with_depth_clear_color(
                                    &mut frame.encoder,
                                    view,
                                    depth_view,
                                    egor_render::wgpu::Color::TRANSPARENT,
                                    true,
                                )
                            } else if watch_pass {
                                renderer.begin_render_pass_with_watch_overlay_depth_clear_color(
                                    &mut frame.encoder,
                                    view,
                                    watch_overlay_view.expect("watch overlay view"),
                                    depth_view,
                                    renderer.clear_color(),
                                    true,
                                )
                            } else {
                                renderer.begin_render_pass_with_depth(&mut frame.encoder, view, depth_view, true)
                            }
                        } else if watch_pass {
                            renderer.begin_render_pass_load_with_watch_overlay_depth(
                                &mut frame.encoder,
                                view,
                                watch_overlay_view.expect("watch overlay view"),
                                depth_view,
                            )
                        } else {
                            renderer.begin_render_pass_load_with_depth(&mut frame.encoder, view, depth_view)
                        };

                        let first_batch = &batches[batch_start];
                        renderer.bind_pass_state_with_watch_overlay(
                            &mut r_pass,
                            first_batch.texture_id,
                            first_batch.shader_id,
                            first_batch.replace_blend,
                            watch_pass,
                        );
                        let mut cur_tex = first_batch.texture_id;
                        let mut cur_shd = first_batch.shader_id;
                        let mut cur_replace_blend = first_batch.replace_blend;
                        let mut cur_cam_offset = u32::MAX;
                        let mut quad_bound = true;
                        let full_scissor = (0u32, 0u32, rt_w.max(1), rt_h.max(1));
                        let mut cur_scissor = (u32::MAX, u32::MAX, u32::MAX, u32::MAX);

                        for idx in batch_start..batch_end {
                            let batch = &mut batches[idx];
                            let target_scissor = match batch.scissor {
                                Some((sx, sy, sw, sh)) => {
                                    let sx = sx.min(rt_w.saturating_sub(1));
                                    let sy = sy.min(rt_h.saturating_sub(1));
                                    let sw = sw.min(rt_w - sx).max(1);
                                    let sh = sh.min(rt_h - sy).max(1);
                                    (sx, sy, sw, sh)
                                }
                                None => full_scissor,
                            };
                            if cur_scissor != target_scissor {
                                r_pass.set_scissor_rect(target_scissor.0, target_scissor.1, target_scissor.2, target_scissor.3);
                                cur_scissor = target_scissor;
                            }
                            let offset = batch.camera_slot * stride;
                            if let Some(shared_buf) = renderer.shared_instance_buffer() {
                                frame_stats.draw_calls += renderer.draw_batch_shared_with_watch_overlay(
                                    &mut r_pass,
                                    &mut batch.geometry,
                                    batch.texture_id,
                                    batch.shader_id,
                                    batch.replace_blend,
                                    offset,
                                    &mut cur_tex,
                                    &mut cur_shd,
                                    &mut cur_replace_blend,
                                    &mut cur_cam_offset,
                                    &mut quad_bound,
                                    shared_buf,
                                    self.instance_byte_offsets[idx],
                                    watch_pass,
                                );
                            } else {
                                frame_stats.draw_calls += renderer.draw_batch_with_watch_overlay(
                                    &mut r_pass,
                                    &mut batch.geometry,
                                    batch.texture_id,
                                    batch.shader_id,
                                    batch.replace_blend,
                                    offset,
                                    &mut cur_tex,
                                    &mut cur_shd,
                                    &mut cur_replace_blend,
                                    &mut cur_cam_offset,
                                    &mut quad_bound,
                                    watch_pass,
                                );
                            }
                        }

                        let is_last_group_for_target = batches[batch_end..].iter().all(|batch| batch.render_target != group_rt);
                        if has_text && is_last_group_for_target {
                            text_renderer.prepare(&device, &queue, rt_w, rt_h, group_rt);
                            text_renderer.render(&mut r_pass);
                            frame_stats.draw_calls += 1;
                        }
                    }

                    batch_start = batch_end;
                }

                // Copy the last offscreen target if it was active
                if let Some(prev_rt) = current_rt {
                    self.render_targets.get(prev_rt).copy_to_sample(&mut frame.encoder);
                }

                if batches.is_empty() {
                    let mut r_pass = if use_watch_frame_target {
                        renderer.begin_render_pass_with_watch_overlay_depth_clear_color(
                            &mut frame.encoder,
                            main_view,
                            watch_overlay_view.expect("watch overlay view"),
                            main_depth_view,
                            renderer.clear_color(),
                            true,
                        )
                    } else {
                        renderer.begin_render_pass_discard_depth(&mut frame.encoder, main_view)
                    };
                    if has_text {
                        text_renderer.prepare(&device, &queue, w, h, None);
                        text_renderer.render(&mut r_pass);
                        frame_stats.draw_calls += 1;
                    }
                }
            } else {
                // Single render pass (no render target overrides)
                {
                    let watch_pass = use_watch_frame_target;
                    let mut r_pass = if watch_pass {
                        renderer.begin_render_pass_with_watch_overlay_depth_clear_color(
                            &mut frame.encoder,
                            main_view,
                            watch_overlay_view.expect("watch overlay view"),
                            main_depth_view,
                            renderer.clear_color(),
                            true,
                        )
                    } else {
                        renderer.begin_render_pass_discard_depth(&mut frame.encoder, main_view)
                    };

                    if let Some(first) = batches.first() {
                        renderer.bind_pass_state_with_watch_overlay(
                            &mut r_pass,
                            first.texture_id,
                            first.shader_id,
                            first.replace_blend,
                            watch_pass,
                        );
                        let mut cur_tex = first.texture_id;
                        let mut cur_shd = first.shader_id;
                        let mut cur_replace_blend = first.replace_blend;
                        let mut cur_cam_offset = u32::MAX;
                        let mut quad_bound = true;
                        let full_scissor = (0u32, 0u32, w.max(1), h.max(1));
                        let mut cur_scissor = (u32::MAX, u32::MAX, u32::MAX, u32::MAX);

                        for (idx, batch) in batches.iter_mut().enumerate() {
                            let target_scissor = match batch.scissor {
                                Some((sx, sy, sw, sh)) => {
                                    let sx = sx.min(w.saturating_sub(1));
                                    let sy = sy.min(h.saturating_sub(1));
                                    let sw = sw.min(w - sx).max(1);
                                    let sh = sh.min(h - sy).max(1);
                                    (sx, sy, sw, sh)
                                }
                                None => full_scissor,
                            };
                            if cur_scissor != target_scissor {
                                r_pass.set_scissor_rect(target_scissor.0, target_scissor.1, target_scissor.2, target_scissor.3);
                                cur_scissor = target_scissor;
                            }
                            let offset = batch.camera_slot * stride;
                            if let Some(shared_buf) = renderer.shared_instance_buffer() {
                                frame_stats.draw_calls += renderer.draw_batch_shared_with_watch_overlay(
                                    &mut r_pass,
                                    &mut batch.geometry,
                                    batch.texture_id,
                                    batch.shader_id,
                                    batch.replace_blend,
                                    offset,
                                    &mut cur_tex,
                                    &mut cur_shd,
                                    &mut cur_replace_blend,
                                    &mut cur_cam_offset,
                                    &mut quad_bound,
                                    shared_buf,
                                    self.instance_byte_offsets[idx],
                                    watch_pass,
                                );
                            } else {
                                frame_stats.draw_calls += renderer.draw_batch_with_watch_overlay(
                                    &mut r_pass,
                                    &mut batch.geometry,
                                    batch.texture_id,
                                    batch.shader_id,
                                    batch.replace_blend,
                                    offset,
                                    &mut cur_tex,
                                    &mut cur_shd,
                                    &mut cur_replace_blend,
                                    &mut cur_cam_offset,
                                    &mut quad_bound,
                                    watch_pass,
                                );
                            }
                        }
                    }

                    if has_text {
                        text_renderer.prepare(&device, &queue, w, h, None);
                        text_renderer.render(&mut r_pass);
                        frame_stats.draw_calls += 1;
                    }
                }
            }
            if has_text {
                text_renderer.finish_frame();
            }

            // Recycle batch GPU buffers for reuse next frame.
            self.primitive_batch.recycle(batches);
        } // profile_scope render_pass
        frame_stats.render_pass_time = render_pass_started_at.elapsed();

        let screen_capture_started_at = Instant::now();
        if let Some(source_rt_id) = self.screen_capture.take_composite_render_target() {
            #[cfg(feature = "profiling")]
            profiling::scope!("watch_composite");
            let source_view = self.render_targets.get(source_rt_id).view();
            self.screen_capture
                .composite_sampled_view(&device, &mut frame.encoder, source_view, &frame.view, format);
        }

        // Screen capture: blit-downsample the final frame into a small capture
        // texture and encode a copy_texture_to_buffer for async readback.
        if capture_active {
            #[cfg(feature = "profiling")]
            profiling::scope!("screen_capture");
            if let Some(watch_target) = watch_frame_target {
                self.screen_capture.capture_from_watch_overlay(
                    &device,
                    &queue,
                    &mut frame.encoder,
                    &watch_target.overlay_view,
                    w,
                    h,
                    format.is_srgb(),
                );
                self.screen_capture
                    .present_sampled_view(&device, &mut frame.encoder, &watch_target.color_view, &frame.view, format);
            } else if let Some(source_rt_id) = capture_source_render_target {
                let source_view = self.render_targets.get(source_rt_id).view();
                self.screen_capture
                    .capture_from_sampled_view(&device, &mut frame.encoder, source_view, format.is_srgb());
            } else if let Some(source_view) = capture_frame_view {
                self.screen_capture
                    .capture_from_sampled_view(&device, &mut frame.encoder, source_view, format.is_srgb());
                self.screen_capture
                    .present_sampled_view(&device, &mut frame.encoder, source_view, &frame.view, format);
            } else {
                let bb_ptr = frame.backbuffer_texture().map(|t| t as *const egor_render::Texture);
                if let Some(ptr) = bb_ptr {
                    // SAFETY: the texture is owned by Frame.presentable which is
                    // not dropped until after this block, and we only read it.
                    let bb_tex = unsafe { &*ptr };
                    self.screen_capture.capture_from_texture(&device, &mut frame.encoder, bb_tex);
                } else {
                    eprintln!("[egor] Screen capture requested but no backbuffer texture available");
                    self.screen_capture.request(0, 0, false);
                }
            }
        }
        frame_stats.screen_capture_time = screen_capture_started_at.elapsed();

        {
            #[cfg(feature = "profiling")]
            profiling::scope!("submit_present");
            let finish_encoder_started_at = Instant::now();
            let (commands, presentable) = match renderer.try_finish_encoder(frame) {
                Ok(result) => result,
                Err(error) => {
                    frame_stats.finish_encoder_time = finish_encoder_started_at.elapsed();
                    log::error!("[egor] finish encoder failed: {error:?}");
                    self.request_renderer_recreation("finish encoder failed");
                    self.finish_frame_stats(frame_stats, egor_frame_started_at);
                    return;
                }
            };
            frame_stats.finish_encoder_time = finish_encoder_started_at.elapsed();

            let queue_submit_started_at = Instant::now();
            if let Err(error) = renderer.try_submit_commands(commands) {
                frame_stats.queue_submit_time = queue_submit_started_at.elapsed();
                log::error!("[egor] queue submit failed: {error:?}");
                self.request_renderer_recreation("queue submit failed");
                self.finish_frame_stats(frame_stats, egor_frame_started_at);
                return;
            }
            frame_stats.queue_submit_time = queue_submit_started_at.elapsed();
            if let Some(p) = presentable {
                let present_started_at = Instant::now();
                if let Err(error) = renderer.try_present(p) {
                    frame_stats.present_time = present_started_at.elapsed();
                    log::error!("[egor] present failed: {error:?}");
                    self.request_renderer_recreation("present failed");
                    self.finish_frame_stats(frame_stats, egor_frame_started_at);
                    return;
                }
                frame_stats.present_time = present_started_at.elapsed();
            }
        } // profile_scope submit_present

        let post_present_started_at = Instant::now();

        // Start the async map AFTER submit so the staging buffer isn't
        // in a pending-map state when the command buffer is submitted.
        if capture_active {
            self.screen_capture.begin_readback_map();
        }

        if let Some((rw, rh)) = requested_size {
            if let Some(backbuffer) = self.backbuffer.as_mut() {
                backbuffer.resize(&device, rw, rh);
            }
        }
        if let Some(vsync) = requested_vsync {
            self.vsync = vsync;
        }
        if requested_vsync.is_some() || fps_limit_changed {
            if let Some(backbuffer) = self.backbuffer.as_mut() {
                backbuffer.set_vsync(&device, hardware_vsync_enabled(self.vsync, self.fps_limit));
            }
        }
        if let Some(renderer_backend) = requested_renderer_backend {
            self.request_renderer_backend_change(renderer_backend);
        }
        frame_stats.post_present_time = post_present_started_at.elapsed();
        self.finish_frame_stats(frame_stats, egor_frame_started_at);
    }

    fn resize(&mut self, w: u32, h: u32, renderer: &mut Renderer) {
        if w == 0 || h == 0 {
            self.surface_recovery.record_surface_failure(SurfaceFailure::ZeroSizedSurface);
            self.backbuffer = None;
            self.surface_acquire_retry_interval = Some(Duration::from_millis(100));
            return;
        }

        self.surface_occluded = false;
        self.waiting_for_surface_change = false;
        if self.backbuffer.is_none() && !self.recreate_backbuffer(renderer) {
            return;
        }

        if let Some(backbuffer) = self.backbuffer.as_mut() {
            backbuffer.resize(renderer.device(), w, h);
        }
        renderer.ensure_depth_size(w, h);
        if let Some(text_renderer) = self.text_renderer.as_mut() {
            text_renderer.resize(w, h, renderer.queue());
        }
    }

    fn suspended(&mut self) {
        log::info!("[egor] app suspended: dropping backbuffer");
        self.app_suspended = true;
        self.frame_timer_reset_requested = true;
        self.surface_acquire_retry_interval = None;
        self.waiting_for_surface_change = false;
        self.backbuffer = None;
        if let Some(window) = self.window.as_deref() {
            set_native_redraw_enabled(window, false);
        }
    }

    fn resumed(&mut self, window: Arc<Window>, renderer: &mut Renderer) {
        let size = window_surface_size(&window);
        self.window = Some(window.clone());
        self.app_suspended = false;
        self.frame_timer_reset_requested = true;
        self.surface_acquire_retry_interval = None;
        self.waiting_for_surface_change = false;
        self.surface_occluded = false;
        set_native_redraw_enabled(&window, true);

        #[cfg(target_os = "android")]
        {
            renderer.drop_startup_surface();
            self.backbuffer = None;
            if size.width == 0 || size.height == 0 {
                self.surface_recovery.record_surface_failure(SurfaceFailure::ZeroSizedSurface);
                log::info!("[egor] app resumed with zero-sized surface; waiting for resize");
                self.surface_acquire_retry_interval = Some(Duration::from_millis(100));
                return;
            }

            log::info!(
                "[egor] app resumed: deferring backbuffer recreation until redraw {}x{}",
                size.width,
                size.height
            );
            return;
        }

        let device = renderer.device();
        if size.width == 0 || size.height == 0 {
            self.surface_recovery.record_surface_failure(SurfaceFailure::ZeroSizedSurface);
            log::info!("[egor] app resumed with zero-sized surface; waiting for resize");
            self.backbuffer = None;
            self.surface_acquire_retry_interval = Some(Duration::from_millis(100));
            return;
        }

        log::info!("[egor] app resumed: recreating backbuffer {}x{}", size.width, size.height);
        let mut backbuffer = match Backbuffer::try_new(renderer.instance(), renderer.adapter(), device, window, size.width, size.height) {
            Ok(backbuffer) => backbuffer,
            Err(error) => {
                log::warn!("[egor] app resume backbuffer creation failed: {error:?}");
                self.backbuffer = None;
                return;
            }
        };
        let backbuffer_format = backbuffer.format();
        let renderer_format = renderer.surface_format();
        if !backbuffer_format_matches_renderer(backbuffer_format, renderer_format) {
            log::warn!(
                "[egor] backbuffer format changed during app resume: renderer={renderer_format:?} backbuffer={backbuffer_format:?}; recreating renderer"
            );
            self.request_renderer_recreation("backbuffer format changed during app resume");
            return;
        }
        backbuffer.set_vsync(device, hardware_vsync_enabled(self.vsync, self.fps_limit));
        self.backbuffer = Some(backbuffer);
        log::info!("[egor] app resumed: complete");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_recreation_request_drops_owned_gpu_state_and_backs_off() {
        let mut app = App::new();
        app.surface_acquire_retry_interval = None;
        app.gpu_device_recreated_pending_frame = true;

        app.request_renderer_recreation("injected queue submit failure");

        assert!(app.renderer_recreate_requested);
        assert!(app.backbuffer.is_none());
        assert!(app.text_renderer.is_none());
        assert_eq!(app.render_targets.len(), 0);
        assert!(app.capture_frame_target.is_none());
        assert!(app.offscreen_batches.is_empty());
        assert!(app.instance_byte_offsets.is_empty());
        assert_eq!(app.surface_acquire_retry_interval, Some(Duration::from_millis(1000)));
    }

    #[test]
    fn renderer_backend_change_requests_recreation() {
        let mut app = App::new();

        app.request_renderer_backend_change(RendererBackendPreference::vulkan());

        assert_eq!(app.renderer_backend, RendererBackendPreference::vulkan());
        assert!(app.renderer_recreate_requested);
        assert_eq!(
            app.renderer_recreate_window_requested,
            cfg!(any(target_arch = "wasm32", target_os = "windows"))
        );
        assert!(app.frame_timer_reset_requested);
    }

    #[test]
    fn unchanged_renderer_backend_does_not_request_recreation() {
        let mut app = App::new();

        app.request_renderer_backend_change(RendererBackendPreference::Auto);

        assert!(!app.renderer_recreate_requested);
        assert!(!app.renderer_recreate_window_requested);
    }

    #[test]
    fn minimized_or_zero_sized_surface_waits_for_restore() {
        assert!(should_wait_for_surface_restore(true, PhysicalSize::new(1280, 720)));
        assert!(should_wait_for_surface_restore(false, PhysicalSize::new(0, 720)));
        assert!(should_wait_for_surface_restore(false, PhysicalSize::new(1280, 0)));
        assert!(!should_wait_for_surface_restore(false, PhysicalSize::new(1280, 720)));
    }

    #[test]
    fn backbuffer_format_change_requires_renderer_recreation() {
        assert!(backbuffer_format_matches_renderer(
            TextureFormat::Bgra8UnormSrgb,
            TextureFormat::Bgra8UnormSrgb
        ));
        assert!(!backbuffer_format_matches_renderer(
            TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Bgra8UnormSrgb
        ));
    }
}
