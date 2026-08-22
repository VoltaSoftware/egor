#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WEB_PAGE_LIFECYCLE_RESET_REQUESTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static WEB_PAGE_LIFECYCLE_HOOK_INSTALLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(target_arch = "wasm32")]
fn now() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now() / 1000.0
}

#[cfg(target_arch = "wasm32")]
fn request_web_page_lifecycle_reset() {
    WEB_PAGE_LIFECYCLE_RESET_REQUESTED.with(|requested| requested.set(true));
}

#[cfg(target_arch = "wasm32")]
fn take_web_page_lifecycle_reset_requested() -> bool {
    WEB_PAGE_LIFECYCLE_RESET_REQUESTED.with(|requested| requested.replace(false))
}

#[cfg(not(target_arch = "wasm32"))]
fn take_web_page_lifecycle_reset_requested() -> bool {
    false
}

#[cfg(target_arch = "wasm32")]
fn install_web_page_lifecycle_reset_hook() {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;

    WEB_PAGE_LIFECYCLE_HOOK_INSTALLED.with(|installed| {
        if installed.replace(true) {
            return;
        }

        let Some(window) = web_sys::window() else {
            return;
        };

        if let Some(document) = window.document() {
            let visibility_callback = Closure::<dyn FnMut(web_sys::Event)>::wrap(Box::new(|_| {
                request_web_page_lifecycle_reset();
            }));
            let _ = document.add_event_listener_with_callback("visibilitychange", visibility_callback.as_ref().unchecked_ref());
            visibility_callback.forget();
        }

        for event_name in ["pagehide", "pageshow"] {
            let callback = Closure::<dyn FnMut(web_sys::Event)>::wrap(Box::new(|_| {
                request_web_page_lifecycle_reset();
            }));
            let _ = window.add_event_listener_with_callback(event_name, callback.as_ref().unchecked_ref());
            callback.forget();
        }
    });
}

pub struct FrameTimer {
    #[cfg(not(target_arch = "wasm32"))]
    last_frame_at: Instant,
    #[cfg(target_arch = "wasm32")]
    last_frame_at: f64,
    reset_next_update: bool,
    accumulator: f32,
    frame_count: u32,
    /// Time in seconds since the last frame
    pub delta: f32,
    /// Frames per second, updated once per second
    pub fps: u32,
    /// Total number of frames rendered since start
    pub frame: u64,
}

impl Default for FrameTimer {
    fn default() -> Self {
        #[cfg(target_arch = "wasm32")]
        install_web_page_lifecycle_reset_hook();

        Self {
            #[cfg(not(target_arch = "wasm32"))]
            last_frame_at: Instant::now(),
            #[cfg(target_arch = "wasm32")]
            last_frame_at: now(),
            reset_next_update: true,
            accumulator: 0.0,
            frame_count: 0,
            delta: 0.0,
            fps: 0,
            frame: 0,
        }
    }
}

impl FrameTimer {
    /// Discards elapsed wall-clock time on the next update.
    ///
    /// Use this when the app is resuming from a period where no visual
    /// simulation should have occurred, such as browser tab suspension,
    /// app backgrounding, occlusion, minimization, or surface recovery.
    pub fn reset_next_update(&mut self) {
        self.reset_next_update = true;
    }

    fn reset_stats_after_discontinuity(&mut self) {
        self.delta = 0.0;
        self.accumulator = 0.0;
        self.frame_count = 0;
    }

    /// Updates delta time & calculates FPS
    pub(crate) fn update(&mut self) {
        let reset = self.reset_next_update || take_web_page_lifecycle_reset_requested();
        self.reset_next_update = false;

        #[cfg(not(target_arch = "wasm32"))]
        {
            let now = Instant::now();
            self.delta = if reset {
                0.0
            } else {
                now.saturating_duration_since(self.last_frame_at).as_secs_f32()
            };
            self.last_frame_at = now;
        }

        #[cfg(target_arch = "wasm32")]
        {
            let now = now();
            self.delta = if reset { 0.0 } else { (now - self.last_frame_at).max(0.0) as f32 };
            self.last_frame_at = now;
        };

        if reset {
            self.reset_stats_after_discontinuity();
        }

        self.accumulator += self.delta;
        self.frame_count += 1;
        self.frame += 1;

        if self.accumulator >= 1.0 {
            self.fps = self.frame_count;
            self.frame_count = 0;
            self.accumulator = 0.0;
        }
    }
}
