pub mod input;
pub mod time;

#[cfg(all(
    target_os = "android",
    not(any(feature = "android-native-activity", feature = "android-game-activity"))
))]
compile_error!("On Android, enable either the `android-native-activity` or `android-game-activity` feature for egor_app");

use crate::{input::Input, time::FrameTimer};
use std::sync::Arc;
use web_time::{Duration, Instant};
pub use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{DeviceEvent, DeviceId, StartCause, WindowEvent},
    event_loop::ControlFlow,
    window::{Fullscreen, Icon, Window},
};

#[cfg(target_os = "ios")]
pub use winit::platform::ios::WindowExtIOS;

#[cfg(target_os = "android")]
use std::sync::OnceLock;
#[cfg(target_os = "android")]
pub use winit::platform::android::activity::AndroidApp;
#[cfg(target_os = "android")]
pub static ANDROID_APP: OnceLock<AndroidApp> = OnceLock::new();

use winit::{
    application::ApplicationHandler,
    event::MouseScrollDelta,
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
    window::WindowId,
};

#[cfg(target_os = "ios")]
fn platform_control_flow(control_flow: ControlFlow, poll_frame_interval: Option<Duration>) -> ControlFlow {
    match control_flow {
        ControlFlow::Poll => poll_frame_interval
            .map(|interval| ControlFlow::WaitUntil(Instant::now() + interval))
            .unwrap_or(ControlFlow::Wait),
        control_flow => control_flow,
    }
}

#[cfg(not(target_os = "ios"))]
fn platform_control_flow(control_flow: ControlFlow, poll_frame_interval: Option<Duration>) -> ControlFlow {
    match control_flow {
        ControlFlow::Poll => poll_frame_interval
            .map(|interval| ControlFlow::WaitUntil(Instant::now() + interval))
            .unwrap_or(ControlFlow::Poll),
        control_flow => control_flow,
    }
}

#[cfg(target_os = "ios")]
fn should_request_poll_redraw_manually() -> bool {
    false
}

#[cfg(not(target_os = "ios"))]
fn should_request_poll_redraw_manually() -> bool {
    true
}

pub struct AppConfig {
    pub control_flow: ControlFlow,
    pub title: String,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub position: Option<(i32, i32)>,
    pub resizable: bool,
    pub maximized: bool,
    pub fullscreen: bool,
    pub decorations: bool,
    pub min_size: Option<(u32, u32)>,
    pub max_size: Option<(u32, u32)>,
    pub simulate_touch_with_mouse: bool,
    pub simulate_mouse_with_touch: bool,
    pub icon: Option<winit::window::Icon>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            control_flow: ControlFlow::Poll,
            title: "Egor App".to_string(),
            width: None,
            height: None,
            position: None,
            resizable: true,
            maximized: false,
            fullscreen: false,
            decorations: true,
            min_size: None,
            max_size: None,
            simulate_touch_with_mouse: false,
            simulate_mouse_with_touch: false,
            icon: None,
        }
    }
}

/// Trait defining application behavior
///
/// Implement this for your app logic. Hooks are called during window creation,
/// every frame, on resize, & before quitting
#[allow(async_fn_in_trait)]
pub trait AppHandler<R> {
    /// Called when app is resumed
    fn resumed(&mut self, _window: Arc<Window>, _resource: &mut R) {}
    /// Called when app is suspended (happens for Android in background)
    fn suspended(&mut self) {}
    /// Called for every WindowEvent before default input handling
    fn on_window_event(&mut self, _window: &Window, _event: &WindowEvent) {}
    /// Called once the window exists; should create & return the resource
    async fn with_resource(&mut self, _window: Arc<Window>) -> R;
    /// Called after the resource is initialized & window is ready
    fn on_ready(&mut self, _window: &Window, _resource: &mut R) {}
    /// Called every frame
    fn frame(&mut self, _window: &Window, _resource: &mut R, _input: &Input, _timer: &FrameTimer) {}
    /// Called on window resize
    fn resize(&mut self, _w: u32, _h: u32, _resource: &mut R) {}
    /// Called when new events arrive, before they are dispatched
    fn new_events(&mut self, _cause: StartCause) {}
    /// Called when all queued events have been processed
    fn about_to_wait(&mut self) {}
    /// Return a frame interval when `ControlFlow::Poll` should be paced instead of busy-polled.
    fn poll_frame_interval(&self, _window: &Window) -> Option<Duration> {
        None
    }
    /// Called when the event loop is shutting down
    fn exiting(&mut self) {}
    /// Called when the OS signals memory pressure (mobile platforms)
    fn memory_warning(&mut self) {}
    /// Called for raw device events (e.g. gamepad, unprocessed input)
    fn device_event(&mut self, _device_id: DeviceId, _event: &DeviceEvent) {}
}

/// Generic application entry point
///
/// Manages window creation, input, event loop, & delegating hooks
/// to your `AppHandler`
/// Use `AppRunner::new()` to construct it, then call `.run(...)` to start the loop
pub struct AppRunner<R: 'static, H: AppHandler<R> + 'static> {
    handler: Option<H>,
    resource: Option<R>,
    window: Option<Arc<Window>>,
    proxy: Option<EventLoopProxy<(R, H)>>,
    queued_poll_redraw: bool,
    queued_poll_redraw_at: Option<Instant>,
    input: Input,
    timer: FrameTimer,
    config: AppConfig,
}

#[doc(hidden)]
impl<R, H: AppHandler<R> + 'static> ApplicationHandler<(R, H)> for AppRunner<R, H> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let (Some(window), Some(resource), Some(handler)) = (self.window.clone(), self.resource.as_mut(), self.handler.as_mut()) {
            handler.resumed(window, resource);
        }

        // Called when window is ready; initializes the resource async (wasm) or sync (native)
        let Some(proxy) = self.proxy.take() else {
            return;
        };

        let fullscreen = match self.config.fullscreen {
            true => Some(Fullscreen::Borderless(None)),
            false => None,
        };

        let mut win_attrs = Window::default_attributes()
            .with_visible(false)
            .with_title(&self.config.title)
            .with_resizable(self.config.resizable)
            .with_maximized(self.config.maximized)
            .with_fullscreen(fullscreen)
            .with_window_icon(self.config.icon.take())
            .with_decorations(self.config.decorations);

        if let (Some(w), Some(h)) = (self.config.width, self.config.height) {
            win_attrs = win_attrs.with_inner_size(PhysicalSize::new(w, h));
        }
        #[cfg(not(any(target_arch = "wasm32", target_os = "android", target_os = "ios")))]
        if let Some((x, y)) = self.config.position {
            win_attrs = win_attrs.with_position(PhysicalPosition::new(x, y));
        }
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowAttributesExtWebSys;
            win_attrs = win_attrs.with_append(true);
        }
        #[cfg(target_os = "ios")]
        {
            use winit::platform::ios::{ScreenEdge, WindowAttributesExtIOS};
            win_attrs = win_attrs
                .with_prefers_home_indicator_hidden(true)
                .with_preferred_screen_edges_deferring_system_gestures(ScreenEdge::ALL);
        }

        let window = Arc::new(event_loop.create_window(win_attrs).unwrap());
        #[cfg(not(any(target_arch = "wasm32", target_os = "android", target_os = "ios")))]
        if let Some((x, y)) = self.config.position {
            window.set_outer_position(PhysicalPosition::new(x, y));
        }
        self.window = Some(window.clone());
        self.input.set_scale_factor(window.scale_factor());

        if let Some((w, h)) = self.config.min_size {
            window.set_min_inner_size(Some(PhysicalSize::new(w, h)));
        }
        if let Some((w, h)) = self.config.max_size {
            window.set_max_inner_size(Some(PhysicalSize::new(w, h)));
        }

        let mut handler = self.handler.take().unwrap();
        #[cfg(target_arch = "wasm32")]
        {
            wasm_bindgen_futures::spawn_local(async move {
                let resource = handler.with_resource(window).await;
                _ = proxy.send_event((resource, handler));
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let resource = pollster::block_on(handler.with_resource(window));
            _ = proxy.send_event((resource, handler));
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(handler) = self.handler.as_mut() {
            handler.suspended();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if let Some(handler) = &mut self.handler {
            handler.on_window_event(self.window.as_ref().unwrap(), &event);
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let Some(window) = &self.window else { return };
                let (Some(resource), Some(handler)) = (&mut self.resource, &mut self.handler) else {
                    return;
                };

                if self.config.control_flow == ControlFlow::Poll
                    && self.queued_poll_redraw
                    && let Some(redraw_at) = self.queued_poll_redraw_at
                {
                    let now = Instant::now();
                    if now < redraw_at {
                        event_loop.set_control_flow(ControlFlow::WaitUntil(redraw_at));
                        return;
                    }

                    self.queued_poll_redraw = false;
                    self.queued_poll_redraw_at = None;
                }

                let frame_started_at = Instant::now();
                self.timer.update();
                handler.frame(window, resource, &self.input, &self.timer);
                self.input.end_frame();

                if self.config.control_flow == ControlFlow::Poll {
                    if let Some(interval) = Self::poll_frame_interval_for_handler(handler, window) {
                        self.queued_poll_redraw = true;
                        self.queued_poll_redraw_at = Some(frame_started_at + interval);
                    } else if should_request_poll_redraw_manually() {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    return;
                }

                if let (Some(resource), Some(handler)) = (self.resource.as_mut(), self.handler.as_mut()) {
                    handler.resize(size.width, size.height, resource);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => self.input.update_key(event),
            WindowEvent::MouseInput { button, state, .. } => {
                self.input.update_mouse_button(button, state);
                self.input.simulate_touch_from_mouse(button, state);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.update_cursor(position);
                self.input.simulate_touch_move_from_mouse();
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let wheel_delta = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
                };
                self.input.update_scroll(wheel_delta);
            }
            WindowEvent::Touch(touch) => {
                self.input.update_touch(touch);
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.input.set_scale_factor(scale_factor);
            }
            WindowEvent::Ime(ime) => {
                self.input.handle_ime(ime);
            }
            _ => {}
        }
    }

    fn user_event(&mut self, _: &ActiveEventLoop, (mut resource, mut handler): (R, H)) {
        let Some(window) = &self.window else { return };

        let frame_started_at = Instant::now();
        handler.on_ready(window, &mut resource);
        handler.frame(window, &mut resource, &self.input, &self.timer);

        window.set_visible(true);
        if self.config.control_flow == ControlFlow::Poll {
            if let Some(interval) = Self::poll_frame_interval_for_handler(&handler, window) {
                self.queued_poll_redraw = true;
                self.queued_poll_redraw_at = Some(frame_started_at + interval);
            } else if should_request_poll_redraw_manually() {
                window.request_redraw();
            }
        } else {
            window.request_redraw();
        }

        self.resource = Some(resource);
        self.handler = Some(handler);
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, cause: StartCause) {
        if let Some(handler) = self.handler.as_mut() {
            handler.new_events(cause);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.config.control_flow == ControlFlow::Poll
            && self.queued_poll_redraw
            && let Some(window) = &self.window
        {
            if let Some(redraw_at) = self.queued_poll_redraw_at {
                if Instant::now() >= redraw_at {
                    self.queued_poll_redraw = false;
                    self.queued_poll_redraw_at = None;
                    window.request_redraw();
                    event_loop.set_control_flow(ControlFlow::Wait);
                } else {
                    event_loop.set_control_flow(ControlFlow::WaitUntil(redraw_at));
                }
            } else {
                self.queued_poll_redraw = false;
                window.request_redraw();
                event_loop.set_control_flow(ControlFlow::Wait);
            }
        } else if self.config.control_flow == ControlFlow::Poll {
            event_loop.set_control_flow(platform_control_flow(self.config.control_flow, self.poll_frame_interval()));
        }

        if let Some(handler) = self.handler.as_mut() {
            handler.about_to_wait();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(handler) = self.handler.as_mut() {
            handler.exiting();
        }
    }

    fn memory_warning(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(handler) = self.handler.as_mut() {
            handler.memory_warning();
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, device_id: DeviceId, event: DeviceEvent) {
        if let Some(handler) = self.handler.as_mut() {
            handler.device_event(device_id, &event);
        }
    }
}

impl<R, H: AppHandler<R> + 'static> AppRunner<R, H> {
    /// Creates a new runner with the given handler & configuration
    pub fn new(handler: H, config: AppConfig) -> Self {
        let mut input = Input::default();
        input.set_simulate_touch_with_mouse(config.simulate_touch_with_mouse);
        input.set_simulate_mouse_with_touch(config.simulate_mouse_with_touch);

        Self {
            handler: Some(handler),
            resource: None,
            window: None,
            proxy: None,
            queued_poll_redraw: false,
            queued_poll_redraw_at: None,
            input,
            timer: FrameTimer::default(),
            config,
        }
    }

    fn poll_frame_interval(&self) -> Option<Duration> {
        let window = self.window.as_deref()?;
        self.handler.as_ref().and_then(|handler| handler.poll_frame_interval(window))
    }

    fn poll_frame_interval_for_handler(handler: &H, window: &Window) -> Option<Duration> {
        handler.poll_frame_interval(window)
    }

    /// Runs the app’s event loop on the current platform
    ///
    /// Handles Android, WASM and native setups, plus logging and user events
    pub fn run(mut self) {
        let mut event_loop_builder = EventLoop::<(R, H)>::with_user_event();
        #[cfg(target_os = "android")]
        {
            #[cfg(feature = "log")]
            android_logger::init_once(android_logger::Config::default().with_max_level(log::LevelFilter::Info));

            use winit::platform::android::EventLoopBuilderExtAndroid;
            let android_app = ANDROID_APP.get().unwrap().clone();
            event_loop_builder.with_android_app(android_app);
        }

        let event_loop = event_loop_builder.build().unwrap();
        event_loop.set_control_flow(platform_control_flow(self.config.control_flow, None));
        self.proxy = Some(event_loop.create_proxy());

        #[cfg(target_arch = "wasm32")]
        {
            #[cfg(feature = "log")]
            {
                std::panic::set_hook(Box::new(console_error_panic_hook::hook));
                // Use .ok() instead of .unwrap() — a logger may already be installed
                // by the host application (e.g. the web client sets up its own logger
                // before calling App::run()).
                let _ = console_log::init_with_level(log::Level::Error);
            }

            use winit::platform::web::EventLoopExtWebSys;
            wasm_bindgen_futures::spawn_local(async move {
                event_loop.spawn_app(self);
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            #[cfg(all(feature = "log", not(target_os = "android")))]
            let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("error")).try_init();

            event_loop.run_app(&mut self).unwrap();
        }
    }
}
