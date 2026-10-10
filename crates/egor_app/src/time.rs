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
const PERIOD_AVERAGE_FRAMES: u32 = 256;
const SETTLE_FRAMES: u32 = 30;
/// The measured frame rate must be this close to the refresh rate before deltas are locked to it.
const REFRESH_MATCH_TOLERANCE: f64 = 0.02;
/// The most a locked frame may differ from one refresh while game time catches up with wall time.
const CATCH_UP_RATE: f64 = 0.05;
/// Game time is never allowed further than this from wall time.
const MAX_CARRY_REFRESHES: f64 = 3.0;
/// Longer gaps are stalls, not missed refreshes; they pass through with any time still owed.
const MAX_LOCKED_DELTA_SECS: f64 = 0.1;

/// Advances game time by one display refresh per frame while vsync paces the frames.
///
/// With vsync the CPU blocks until the swapchain returns an image, and that wait moves with the
/// compositor's timing: one frame can measure two refreshes and the next almost none while the
/// display still shows one new frame per refresh. Animating with those intervals makes motion
/// step unevenly on a steady display. This clock gives every frame one refresh and carries the
/// difference to wall time, paying it back a few percent per frame. A refresh the display really
/// repeated is then caught up smoothly instead of as a jump, and game time stays within a few
/// refreshes of wall time.
#[derive(Default)]
struct VsyncClock {
    /// Measured refresh period. It survives restarts, since the display does not change when the
    /// window loses focus or the app resumes.
    period: f64,
    period_samples: u32,
    refresh_hint: Option<f64>,
    /// Frames since the last restart.
    frames: u32,
    /// Wall time not yet given to the game (negative when the game is ahead).
    carry: f64,
}

impl VsyncClock {
    /// Drops the carry after a discontinuity, keeping the measured period.
    fn restart(&mut self) {
        self.frames = 0;
        self.carry = 0.0;
    }

    fn delta(&mut self, measured: f64, refresh_hz: Option<f64>) -> f64 {
        if measured > MAX_LOCKED_DELTA_SECS {
            let step = (measured + self.carry).max(0.0);
            self.restart();
            return step;
        }
        let refresh_hz = refresh_hz.filter(|hz| *hz > 0.0);
        if refresh_hz != self.refresh_hint {
            // A newly reported display: start from its rate and refine it from there.
            self.refresh_hint = refresh_hz;
            if let Some(hz) = refresh_hz {
                self.period = 1.0 / hz;
                self.period_samples = PERIOD_AVERAGE_FRAMES;
            }
        }
        // Once the period is known, a late acquire, the quick one after it, or a burst of frames
        // must not drag the estimate, so samples are limited to half to one and a half refreshes.
        let sample = if self.period_samples >= PERIOD_AVERAGE_FRAMES {
            measured.clamp(0.5 * self.period, 1.5 * self.period)
        } else {
            measured
        };
        self.period_samples = (self.period_samples + 1).min(PERIOD_AVERAGE_FRAMES);
        self.period += (sample - self.period) / f64::from(self.period_samples);
        self.frames = self.frames.saturating_add(1);
        self.carry += measured;

        // Only lock while frames really arrive once per refresh. A game that cannot keep up,
        // a variable-refresh display below its maximum, or a wrong refresh hint all fail this
        // check and keep the measured interval, plus any time still owed to the game.
        let nominal_hz = refresh_hz.unwrap_or_else(|| nearest_common_refresh_hz(self.period));
        let keeps_refresh = (self.period * nominal_hz - 1.0).abs() <= REFRESH_MATCH_TOLERANCE;
        if self.frames < SETTLE_FRAMES || !keeps_refresh {
            let step = self.carry.max(0.0);
            self.carry -= step;
            return step;
        }

        let mut step = self.carry.clamp(
            self.period * (1.0 - CATCH_UP_RATE),
            self.period * (1.0 + CATCH_UP_RATE),
        );
        let remaining = self.carry - step;
        let max_carry = MAX_CARRY_REFRESHES * self.period;
        if remaining.abs() > max_carry {
            step += remaining - max_carry.copysign(remaining);
        }
        self.carry -= step;
        step
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
                self.vsync_clock.restart();
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
        for round in 0..400 {
            if round == 200 {
                // Focus changes restart the clock; the measured period must carry over so the
                // clock relocks without catching up.
                clock.restart();
            }
            // A late acquire followed by an immediate one, as FIFO does when the compositor
            // releases buffers unevenly while still showing a new frame every refresh.
            for measured in [
                2.0 * PERIOD,
                0.02 * PERIOD,
                PERIOD,
                1.4 * PERIOD,
                0.6 * PERIOD,
            ] {
                let step = clock.delta(measured, HINT);
                if clock.frames >= SETTLE_FRAMES {
                    assert!(
                        (step / PERIOD - 1.0).abs() <= CATCH_UP_RATE + 0.01,
                        "round {round}: step of {:.2} refreshes",
                        step / PERIOD
                    );
                }
            }
        }
    }

    #[test]
    fn repeated_refreshes_and_bursts_keep_game_time_near_wall_time() {
        let mut clock = settled_clock();
        let (mut wall, mut game) = (0.0, 0.0);
        for i in 0..2000 {
            // The display drops a refresh now and then, and a resize lets frames through at once.
            let measured = match i % 500 {
                100 | 300 => 2.0 * PERIOD,
                400..=405 => 0.05 * PERIOD,
                _ => PERIOD,
            };
            wall += measured;
            game += clock.delta(measured, HINT);
            assert!(
                (game - wall).abs() <= (MAX_CARRY_REFRESHES + 0.1) * PERIOD,
                "drift {}",
                game - wall
            );
        }
        // Once the display is steady again, the carry is paid back.
        for _ in 0..200 {
            wall += PERIOD;
            game += clock.delta(PERIOD, HINT);
        }
        assert!((game - wall).abs() < 0.01 * PERIOD, "drift {}", game - wall);
    }

    #[test]
    fn frames_that_do_not_match_the_refresh_keep_measured_time() {
        // A wrong hint (or a game that cannot keep up) must not stretch or squash time.
        let mut clock = VsyncClock::default();
        for _ in 0..300 {
            assert_eq!(clock.delta(PERIOD, Some(60.0)), PERIOD);
        }
        let mut clock = settled_clock();
        assert!((clock.delta(0.5, HINT) - 0.5).abs() < 1e-9);
    }
}
