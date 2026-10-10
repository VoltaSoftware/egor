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
    /// Frames are shown as soon as they are ready (software FPS cap, mailbox or no vsync), or the
    /// platform does not report the display's refresh rate. The delta is the measured time between
    /// frame starts.
    Unlocked,
    /// Vsync shows one frame per refresh of a display running at `refresh_hz`, as the OS reports it.
    Vsync { refresh_hz: f64 },
}

/// The most a frame may differ from one refresh while game time catches up with wall time.
const CATCH_UP_RATE: f64 = 0.05;
/// Game time is never allowed further than this from wall time.
const MAX_CARRY_REFRESHES: f64 = 3.0;
/// Longer gaps are stalls, not missed refreshes; they pass through with any time still owed.
const MAX_LOCKED_DELTA_SECS: f64 = 0.1;

/// Advances game time by about one display refresh per frame while vsync paces the frames.
///
/// With vsync the CPU blocks until the swapchain returns an image, and that wait moves with the
/// compositor's timing: one frame can measure two refreshes and the next almost none while the
/// display still shows one new frame per refresh. Animating with those intervals makes motion
/// step unevenly on a steady display. This clock lets each frame differ from one refresh by at
/// most a few percent and carries the rest, paying it back over the following frames. Game time
/// stays within a few refreshes of wall time, a refresh the display really repeated is caught up
/// smoothly instead of as a jump, and when frames do not match the refresh (a game that cannot
/// keep up, a wrong rate) the carry fills up and the measured intervals pass through.
#[derive(Default)]
struct VsyncClock {
    /// Wall time not yet given to the game (negative when the game is ahead).
    carry: f64,
}

impl VsyncClock {
    fn restart(&mut self) {
        self.carry = 0.0;
    }

    fn delta(&mut self, measured: f64, refresh_hz: f64) -> f64 {
        self.carry += measured;
        if measured > MAX_LOCKED_DELTA_SECS {
            let step = self.carry.max(0.0);
            self.restart();
            return step;
        }
        let refresh = 1.0 / refresh_hz;
        let mut step = self.carry.clamp(
            refresh * (1.0 - CATCH_UP_RATE),
            refresh * (1.0 + CATCH_UP_RATE),
        );
        let remaining = self.carry - step;
        let max_carry = MAX_CARRY_REFRESHES * refresh;
        if remaining.abs() > max_carry {
            step += remaining - max_carry.copysign(remaining);
        }
        self.carry -= step;
        step
    }
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
    /// How long this frame stayed on screen as far as the CPU can tell, for FPS statistics:
    /// whole display refreshes under vsync, the measured interval otherwise. Advance the game
    /// with `delta`, which also carries vsync catch-up.
    pub frame_time: f32,
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
            frame_time: 0.0,
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
        self.frame_time = 0.0;
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

        (self.delta, self.frame_time) = match pacing {
            FramePacing::Vsync { refresh_hz } if !reset && refresh_hz > 0.0 => {
                let step = self.vsync_clock.delta(measured, refresh_hz);
                // Vsync keeps every frame on screen for whole refreshes; the clock's catch-up is
                // not screen time.
                let shown = (step * refresh_hz).round().max(1.0) / refresh_hz;
                (step as f32, shown as f32)
            }
            _ => {
                self.vsync_clock.restart();
                let delta = if reset { 0.0 } else { measured as f32 };
                (delta, delta)
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

    /// The display's real period; the OS reports it rounded to 120 Hz, as Windows does.
    const PERIOD: f64 = 1.0 / 119.88;
    const REPORTED_HZ: f64 = 120.0;

    #[test]
    fn acquire_jitter_on_a_steady_display_advances_one_refresh_per_frame() {
        let mut clock = VsyncClock::default();
        let (mut wall, mut game) = (0.0, 0.0);
        for _ in 0..400 {
            // A late acquire followed by an immediate one, as FIFO does when the compositor
            // releases buffers unevenly while still showing a new frame every refresh.
            for measured in [
                2.0 * PERIOD,
                0.02 * PERIOD,
                PERIOD,
                1.4 * PERIOD,
                0.6 * PERIOD,
            ] {
                let step = clock.delta(measured, REPORTED_HZ);
                wall += measured;
                game += step;
                assert!(
                    (step / PERIOD - 1.0).abs() <= 0.06,
                    "step of {:.2} refreshes",
                    step / PERIOD
                );
                assert!((game - wall).abs() <= 1.1 * PERIOD, "drift {}", game - wall);
            }
        }
    }

    #[test]
    fn repeated_refreshes_and_bursts_keep_game_time_near_wall_time() {
        let mut clock = VsyncClock::default();
        let (mut wall, mut game) = (0.0, 0.0);
        for i in 0..2000 {
            // The display drops a refresh now and then, and a resize lets frames through at once.
            let measured = match i % 500 {
                100 | 300 => 2.0 * PERIOD,
                400..=405 => 0.05 * PERIOD,
                _ => PERIOD,
            };
            wall += measured;
            game += clock.delta(measured, REPORTED_HZ);
            assert!(
                (game - wall).abs() <= MAX_CARRY_REFRESHES / REPORTED_HZ + 1e-9,
                "drift {}",
                game - wall
            );
        }
        // Once the display is steady again, the carry is paid back.
        for _ in 0..200 {
            wall += PERIOD;
            game += clock.delta(PERIOD, REPORTED_HZ);
        }
        assert!((game - wall).abs() < 0.01 * PERIOD, "drift {}", game - wall);
    }

    #[test]
    fn frames_that_do_not_match_the_refresh_keep_measured_time() {
        // A wrong rate, or a game that cannot keep up, ends with the carry full and the measured
        // intervals passing straight through.
        let mut clock = VsyncClock::default();
        for i in 0..300 {
            let step = clock.delta(PERIOD, 60.0);
            if i > 20 {
                assert!((step - PERIOD).abs() < 1e-9, "frame {i}: {step}");
            }
        }
        let mut clock = VsyncClock::default();
        assert!((clock.delta(0.5, REPORTED_HZ) - 0.5).abs() < 1e-9);
    }
}
