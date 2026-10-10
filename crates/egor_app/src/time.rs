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
            let _ = document.add_event_listener_with_callback(
                "visibilitychange",
                visibility_callback.as_ref().unchecked_ref(),
            );
            visibility_callback.forget();
        }

        for event_name in ["pagehide", "pageshow"] {
            let callback = Closure::<dyn FnMut(web_sys::Event)>::wrap(Box::new(|_| {
                request_web_page_lifecycle_reset();
            }));
            let _ = window
                .add_event_listener_with_callback(event_name, callback.as_ref().unchecked_ref());
            callback.forget();
        }
    });
}

/// How presented frames reach the display, which decides what `FrameTimer::delta` measures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FramePacing {
    /// Frames are shown as soon as they are ready (software FPS cap, mailbox or no vsync).
    /// The delta is the measured time between frame starts.
    Unlocked,
    /// Every frame stays on screen for whole display refreshes (FIFO vsync, browser animation
    /// frames, display links). `refresh_hz` is the display rate when the platform reports it.
    Vsync { refresh_hz: Option<f64> },
}

const COMMON_REFRESH_HZ: [f64; 8] = [60.0, 75.0, 90.0, 100.0, 120.0, 144.0, 165.0, 240.0];
/// Frames averaged to measure the exact refresh period (about two seconds at 120 Hz).
const PERIOD_AVERAGE_FRAMES: f64 = 256.0;
const SETTLE_FRAMES: u32 = 30;
/// The measured frame rate must be this close to the refresh rate before deltas are locked to it.
const REFRESH_MATCH_TOLERANCE: f64 = 0.02;
/// Longer gaps are stalls, not missed refreshes; they pass through unchanged.
const MAX_LOCKED_DELTA_SECS: f64 = 0.1;

/// Turns measured frame intervals into whole display refreshes while vsync paces the frames.
///
/// With vsync the CPU blocks until the swapchain returns an image, and that wait moves with the
/// compositor's timing: one frame can measure two refreshes and the next almost none while the
/// display still shows one new frame per refresh. Animating with those intervals makes motion
/// step unevenly on a steady display. This clock advances by whole refreshes instead and carries
/// the difference, so game time stays within a few refreshes of wall time and a frame the
/// display really repeated still catches up.
#[derive(Default)]
struct VsyncClock {
    average_interval: f64,
    frames: u32,
    /// Wall time not yet given to the game (negative when the game is ahead).
    carry: f64,
}

impl VsyncClock {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn delta(&mut self, measured: f64, refresh_hz: Option<f64>) -> f64 {
        if measured > MAX_LOCKED_DELTA_SECS {
            self.reset();
            return measured;
        }
        self.frames = self.frames.saturating_add(1);
        self.average_interval +=
            (measured - self.average_interval) / f64::from(self.frames).min(PERIOD_AVERAGE_FRAMES);

        let refresh_hz = refresh_hz
            .filter(|hz| *hz > 0.0)
            .unwrap_or_else(|| nearest_common_refresh_hz(self.average_interval));
        // Only lock while frames really arrive once per refresh. A game that cannot keep up,
        // a variable-refresh display below its maximum, or a wrong refresh hint all fail this
        // check and keep the measured interval.
        let keeps_refresh =
            (self.average_interval * refresh_hz - 1.0).abs() <= REFRESH_MATCH_TOLERANCE;
        if self.frames < SETTLE_FRAMES || !keeps_refresh {
            self.carry = 0.0;
            return measured;
        }

        let period = self.average_interval;
        self.carry += measured;
        let mut refreshes = (self.carry / period).floor().max(1.0);
        if self.carry - refreshes * period < -2.0 * period {
            // A burst of quick frames (an empty swapchain after a resize) must not run the game
            // ahead of wall time without bound, so hold it back instead of advancing a refresh.
            refreshes = (self.carry / period + 2.0).floor().max(0.0);
        }
        self.carry -= refreshes * period;
        refreshes * period
    }
}

fn nearest_common_refresh_hz(interval: f64) -> f64 {
    let measured_hz = 1.0 / interval;
    COMMON_REFRESH_HZ
        .into_iter()
        .min_by(|a, b| (a - measured_hz).abs().total_cmp(&(b - measured_hz).abs()))
        .unwrap_or(60.0)
}

pub struct FrameTimer {
    #[cfg(not(target_arch = "wasm32"))]
    last_frame_at: Instant,
    #[cfg(target_arch = "wasm32")]
    last_frame_at: f64,
    reset_next_update: bool,
    vsync_clock: VsyncClock,
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
            vsync_clock: VsyncClock::default(),
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
    pub(crate) fn update(&mut self, pacing: FramePacing) {
        let reset = self.reset_next_update || take_web_page_lifecycle_reset_requested();
        self.reset_next_update = false;

        #[cfg(not(target_arch = "wasm32"))]
        let measured = {
            let now = Instant::now();
            let measured = now
                .saturating_duration_since(self.last_frame_at)
                .as_secs_f64();
            self.last_frame_at = now;
            measured
        };

        #[cfg(target_arch = "wasm32")]
        let measured = {
            let now = now();
            let measured = (now - self.last_frame_at).max(0.0);
            self.last_frame_at = now;
            measured
        };

        self.delta = match pacing {
            FramePacing::Vsync { refresh_hz } if !reset => {
                self.vsync_clock.delta(measured, refresh_hz) as f32
            }
            _ => {
                self.vsync_clock.reset();
                if reset { 0.0 } else { measured as f32 }
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    const PERIOD: f64 = 1.0 / 119.88;
    const HINT: Option<f64> = Some(120.0);

    fn settled_clock() -> VsyncClock {
        let mut clock = VsyncClock::default();
        for _ in 0..300 {
            clock.delta(PERIOD, HINT);
        }
        clock
    }

    #[test]
    fn acquire_jitter_on_a_steady_display_advances_one_refresh_per_frame() {
        let mut clock = settled_clock();
        let mut uneven = 0;
        for _ in 0..200 {
            // A late acquire followed by an immediate one, as FIFO does when the compositor
            // releases buffers unevenly while still showing a new frame every refresh.
            for measured in [
                2.0 * PERIOD,
                0.02 * PERIOD,
                PERIOD,
                1.4 * PERIOD,
                0.6 * PERIOD,
            ] {
                if (clock.delta(measured, HINT) - PERIOD).abs() > PERIOD * 0.01 {
                    uneven += 1;
                }
            }
        }
        // Only the first late frame can be taken as a repeat before the carry absorbs it.
        assert!(uneven <= 2, "{uneven} uneven steps");
    }

    #[test]
    fn repeated_refreshes_and_bursts_keep_game_time_near_wall_time() {
        let mut clock = settled_clock();
        let (mut wall, mut game) = (0.0, 0.0);
        let mut pattern = Vec::new();
        for i in 0..2000 {
            // The display drops a refresh now and then, and a resize lets frames through at once.
            let measured = match i % 500 {
                100 | 300 => 2.0 * PERIOD,
                400..=405 => 0.05 * PERIOD,
                _ => PERIOD,
            };
            pattern.push(measured);
        }
        for measured in pattern {
            wall += measured;
            game += clock.delta(measured, HINT);
            assert!((game - wall).abs() <= 3.0 * PERIOD, "drift {}", game - wall);
        }
    }

    #[test]
    fn frames_that_do_not_match_the_refresh_keep_measured_time() {
        // A wrong hint (or a game that cannot keep up) must not stretch or squash time.
        let mut clock = VsyncClock::default();
        for _ in 0..300 {
            assert_eq!(clock.delta(PERIOD, Some(60.0)), PERIOD);
        }
        let mut clock = settled_clock();
        assert_eq!(clock.delta(0.5, HINT), 0.5);
    }
}
