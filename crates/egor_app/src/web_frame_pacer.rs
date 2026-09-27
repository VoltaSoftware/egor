use web_time::{Duration, Instant};

/// Select browser animation frames without adding a timer before each redraw.
/// The deadline advances from its previous phase, never from a late callback.
#[derive(Default)]
pub(crate) struct WebFramePacer {
    interval: Option<Duration>,
    next_frame: Option<Instant>,
}

impl WebFramePacer {
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn should_render(&mut self, now: Instant, interval: Option<Duration>) -> bool {
        let interval = interval.filter(|value| !value.is_zero());
        if self.interval != interval {
            self.interval = interval;
            self.next_frame = None;
        }
        let Some(interval) = interval else {
            self.next_frame = None;
            return true;
        };
        let deadline = self.next_frame.unwrap_or(now);
        // winit exposes callback arrival time, not the browser's rAF timestamp.
        // Allow small dispatch jitter so an aligned refresh is not skipped.
        let tolerance = Duration::from_micros(500).min(interval / 4);
        if now + tolerance < deadline {
            return false;
        }
        // Consume missed slots once after a stall; never issue catch-up bursts.
        let slots = (now.saturating_duration_since(deadline).as_secs_f64()
            / interval.as_secs_f64())
        .floor()
            + 1.0;
        self.next_frame = Some(deadline + interval.mul_f64(slots));
        true
    }
}
