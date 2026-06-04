#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
fn now() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now() / 1000.0
}

pub struct FrameTimer {
    #[cfg(not(target_arch = "wasm32"))]
    last_frame_at: Instant,
    #[cfg(target_arch = "wasm32")]
    last_frame_at: f64,
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
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            last_frame_at: Instant::now(),
            #[cfg(target_arch = "wasm32")]
            last_frame_at: now(),
            accumulator: 0.0,
            frame_count: 0,
            delta: 0.0,
            fps: 0,
            frame: 0,
        }
    }
}

impl FrameTimer {
    /// Updates delta time & calculates FPS
    pub(crate) fn update(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let now = Instant::now();
            self.delta = now.saturating_duration_since(self.last_frame_at).as_secs_f32();
            self.last_frame_at = now;
        }

        #[cfg(target_arch = "wasm32")]
        {
            let now = now();
            self.delta = (now - self.last_frame_at).max(0.0) as f32;
            self.last_frame_at = now;
        };

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
