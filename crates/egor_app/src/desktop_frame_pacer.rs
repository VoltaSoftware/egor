use web_time::{Duration, Instant};

/// Keep the requested rate independent of event-loop wakeup latency.
#[derive(Default)]
pub(crate) struct DesktopFramePacer {
    interval: Option<Duration>,
    deadline: Option<Instant>,
}

impl DesktopFramePacer {
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn next_deadline(
        &mut self,
        frame_start: Instant,
        interval: Option<Duration>,
    ) -> Option<Instant> {
        let interval = interval.filter(|interval| !interval.is_zero());
        if self.interval != interval {
            self.reset();
            self.interval = interval;
        }
        let interval = interval?;
        let deadline = match self.deadline {
            None => frame_start + interval,
            Some(previous) => {
                // Advance from the scheduled frame, not its late actual start.
                // After a stall skip missed slots instead of creating a burst.
                let slots = frame_start.saturating_duration_since(previous).as_nanos()
                    / interval.as_nanos()
                    + 1;
                previous + Duration::from_nanos((interval.as_nanos() * slots) as u64)
            }
        };
        self.deadline = Some(deadline);
        Some(deadline)
    }
}
