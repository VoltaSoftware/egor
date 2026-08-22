use egor_render::target::SurfaceAcquireFailure;
use web_time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SurfaceFailure {
    Acquire(SurfaceAcquireFailure),
    #[allow(dead_code)]
    AcquirePanic,
    ConfigureFailed,
    BackbufferCreateFailed,
    ZeroSizedSurface,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SurfaceRecoveryAction {
    SkipFrame,
    WaitForResize,
    WaitForSurfaceChange,
    RecreateBackbuffer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeviceLossAction {
    #[allow(dead_code)]
    PauseRendering,
    RecreateRenderer,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SurfaceRecoveryState {
    consecutive_acquire_failures: u32,
    consecutive_surface_create_failures: u32,
    device_lost_reported: bool,
}

impl SurfaceRecoveryState {
    pub(crate) fn new() -> Self {
        Self {
            consecutive_acquire_failures: 0,
            consecutive_surface_create_failures: 0,
            device_lost_reported: false,
        }
    }

    pub(crate) fn consecutive_acquire_failures(&self) -> u32 {
        self.consecutive_acquire_failures
    }

    pub(crate) fn record_frame_acquired(&mut self) {
        self.consecutive_acquire_failures = 0;
        self.consecutive_surface_create_failures = 0;
        self.device_lost_reported = false;
    }

    pub(crate) fn record_surface_failure(&mut self, failure: SurfaceFailure) -> SurfaceRecoveryAction {
        match failure {
            SurfaceFailure::ZeroSizedSurface => {
                self.reset_surface_create_failures();
                SurfaceRecoveryAction::WaitForResize
            }
            SurfaceFailure::BackbufferCreateFailed => {
                self.bump_surface_create_failures();
                self.surface_create_failure_action()
            }
            SurfaceFailure::ConfigureFailed => {
                self.bump_surface_create_failures();
                self.surface_create_failure_action()
            }
            SurfaceFailure::Acquire(SurfaceAcquireFailure::Lost) => {
                self.reset_surface_create_failures();
                self.bump_acquire_failures();
                SurfaceRecoveryAction::RecreateBackbuffer
            }
            SurfaceFailure::Acquire(SurfaceAcquireFailure::Validation) | SurfaceFailure::AcquirePanic => {
                self.reset_surface_create_failures();
                self.bump_acquire_failures();
                if self.consecutive_acquire_failures > 1 {
                    SurfaceRecoveryAction::RecreateBackbuffer
                } else {
                    SurfaceRecoveryAction::SkipFrame
                }
            }
            SurfaceFailure::Acquire(SurfaceAcquireFailure::Outdated) => {
                self.reset_surface_create_failures();
                self.bump_acquire_failures();
                SurfaceRecoveryAction::RecreateBackbuffer
            }
            SurfaceFailure::Acquire(SurfaceAcquireFailure::Timeout | SurfaceAcquireFailure::Occluded) => {
                self.reset_surface_create_failures();
                self.bump_acquire_failures();
                SurfaceRecoveryAction::SkipFrame
            }
        }
    }

    pub(crate) fn record_device_lost(&mut self) -> (DeviceLossAction, bool) {
        let should_log = !self.device_lost_reported;
        self.device_lost_reported = true;
        (DeviceLossAction::RecreateRenderer, should_log)
    }

    fn bump_acquire_failures(&mut self) {
        self.consecutive_acquire_failures = self.consecutive_acquire_failures.saturating_add(1);
    }

    fn bump_surface_create_failures(&mut self) {
        self.bump_acquire_failures();
        self.consecutive_surface_create_failures = self.consecutive_surface_create_failures.saturating_add(1);
    }

    fn reset_surface_create_failures(&mut self) {
        self.consecutive_surface_create_failures = 0;
    }

    fn surface_create_failure_action(&self) -> SurfaceRecoveryAction {
        if self.consecutive_surface_create_failures > 3 {
            SurfaceRecoveryAction::WaitForSurfaceChange
        } else {
            SurfaceRecoveryAction::SkipFrame
        }
    }
}

impl Default for SurfaceRecoveryState {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn frame_interval_for_fps(fps: u16) -> Duration {
    let fps = u64::from(fps.max(1));
    Duration::from_nanos((1_000_000_000u64 + fps / 2) / fps)
}

pub(crate) fn retry_interval_for_refresh(native_refresh_rate_fps: Option<u16>, consecutive_failures: u32) -> Duration {
    if consecutive_failures > 3 {
        return Duration::from_millis(100);
    }

    native_refresh_rate_fps
        .map(frame_interval_for_fps)
        .unwrap_or_else(|| Duration::from_millis(16))
}

pub(crate) fn max_frame_interval(first: Option<Duration>, second: Option<Duration>) -> Option<Duration> {
    match (first, second) {
        (Some(first), Some(second)) => Some(first.max(second)),
        (Some(interval), None) | (None, Some(interval)) => Some(interval),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    enum InjectedFault {
        SurfaceAcquire(SurfaceAcquireFailure),
        AcquirePanic,
        ConfigureFailed,
        BackbufferCreateFailed,
        ZeroSizedSurface,
        DeviceLost,
        BeginFrameFailed,
        FinishEncoderFailed,
        QueueSubmitFailed,
        PresentFailed,
        FrameAcquired,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum InjectedRecoveryOutcome {
        SkipFrame,
        WaitForResize,
        WaitForSurfaceChange,
        RecreateBackbuffer,
        RecreateRenderer,
        FrameAcquired,
    }

    #[derive(Debug, Default)]
    struct FaultRecoveryHarness {
        state: SurfaceRecoveryState,
        renderer_recreate_pending: bool,
        renderer_recreate_count: u32,
        backbuffer_recreate_count: u32,
        skipped_frame_count: u32,
        wait_resize_count: u32,
        acquired_frame_count: u32,
    }

    impl FaultRecoveryHarness {
        fn inject(&mut self, fault: InjectedFault) -> InjectedRecoveryOutcome {
            match fault {
                InjectedFault::SurfaceAcquire(failure) => {
                    let action = self.state.record_surface_failure(SurfaceFailure::Acquire(failure));
                    self.record_surface_action(action)
                }
                InjectedFault::AcquirePanic => {
                    let action = self.state.record_surface_failure(SurfaceFailure::AcquirePanic);
                    self.record_surface_action(action)
                }
                InjectedFault::ConfigureFailed => {
                    let action = self.state.record_surface_failure(SurfaceFailure::ConfigureFailed);
                    self.record_surface_action(action)
                }
                InjectedFault::BackbufferCreateFailed => {
                    let action = self.state.record_surface_failure(SurfaceFailure::BackbufferCreateFailed);
                    self.record_surface_action(action)
                }
                InjectedFault::ZeroSizedSurface => {
                    let action = self.state.record_surface_failure(SurfaceFailure::ZeroSizedSurface);
                    self.record_surface_action(action)
                }
                InjectedFault::DeviceLost
                | InjectedFault::BeginFrameFailed
                | InjectedFault::FinishEncoderFailed
                | InjectedFault::QueueSubmitFailed
                | InjectedFault::PresentFailed => {
                    if matches!(fault, InjectedFault::DeviceLost) {
                        let (action, _should_log) = self.state.record_device_lost();
                        assert_eq!(action, DeviceLossAction::RecreateRenderer);
                    }
                    self.request_renderer_recreate();
                    InjectedRecoveryOutcome::RecreateRenderer
                }
                InjectedFault::FrameAcquired => {
                    self.state.record_frame_acquired();
                    self.renderer_recreate_pending = false;
                    self.acquired_frame_count += 1;
                    InjectedRecoveryOutcome::FrameAcquired
                }
            }
        }

        fn record_surface_action(&mut self, action: SurfaceRecoveryAction) -> InjectedRecoveryOutcome {
            match action {
                SurfaceRecoveryAction::SkipFrame => {
                    self.skipped_frame_count += 1;
                    InjectedRecoveryOutcome::SkipFrame
                }
                SurfaceRecoveryAction::WaitForResize => {
                    self.wait_resize_count += 1;
                    InjectedRecoveryOutcome::WaitForResize
                }
                SurfaceRecoveryAction::WaitForSurfaceChange => InjectedRecoveryOutcome::WaitForSurfaceChange,
                SurfaceRecoveryAction::RecreateBackbuffer => {
                    self.backbuffer_recreate_count += 1;
                    InjectedRecoveryOutcome::RecreateBackbuffer
                }
            }
        }

        fn request_renderer_recreate(&mut self) {
            if !self.renderer_recreate_pending {
                self.renderer_recreate_count += 1;
                self.renderer_recreate_pending = true;
            }
        }
    }

    #[test]
    fn timeout_and_occluded_skip_without_recreating() {
        for failure in [SurfaceAcquireFailure::Timeout, SurfaceAcquireFailure::Occluded] {
            let mut state = SurfaceRecoveryState::new();

            let action = state.record_surface_failure(SurfaceFailure::Acquire(failure));

            assert_eq!(action, SurfaceRecoveryAction::SkipFrame);
            assert_eq!(state.consecutive_acquire_failures(), 1);
        }
    }

    #[test]
    fn outdated_surface_recreates_backbuffer_immediately() {
        let mut state = SurfaceRecoveryState::new();

        let action = state.record_surface_failure(SurfaceFailure::Acquire(SurfaceAcquireFailure::Outdated));

        assert_eq!(action, SurfaceRecoveryAction::RecreateBackbuffer);
        assert_eq!(state.consecutive_acquire_failures(), 1);
    }

    #[test]
    fn lost_surface_recreates_backbuffer_immediately() {
        let mut state = SurfaceRecoveryState::new();

        let action = state.record_surface_failure(SurfaceFailure::Acquire(SurfaceAcquireFailure::Lost));

        assert_eq!(action, SurfaceRecoveryAction::RecreateBackbuffer);
        assert_eq!(state.consecutive_acquire_failures(), 1);
    }

    #[test]
    fn validation_gets_one_retry_before_backbuffer_recreate() {
        let mut state = SurfaceRecoveryState::new();

        let first = state.record_surface_failure(SurfaceFailure::Acquire(SurfaceAcquireFailure::Validation));
        let second = state.record_surface_failure(SurfaceFailure::Acquire(SurfaceAcquireFailure::Validation));

        assert_eq!(first, SurfaceRecoveryAction::SkipFrame);
        assert_eq!(second, SurfaceRecoveryAction::RecreateBackbuffer);
    }

    #[test]
    fn acquire_panic_is_treated_like_validation() {
        let mut state = SurfaceRecoveryState::new();

        let first = state.record_surface_failure(SurfaceFailure::AcquirePanic);
        let second = state.record_surface_failure(SurfaceFailure::AcquirePanic);

        assert_eq!(first, SurfaceRecoveryAction::SkipFrame);
        assert_eq!(second, SurfaceRecoveryAction::RecreateBackbuffer);
    }

    #[test]
    fn configure_failure_skips_and_backs_off_without_recreating_immediately() {
        let mut state = SurfaceRecoveryState::new();

        let action = state.record_surface_failure(SurfaceFailure::ConfigureFailed);

        assert_eq!(action, SurfaceRecoveryAction::SkipFrame);
        assert_eq!(state.consecutive_acquire_failures(), 1);
    }

    #[test]
    fn backbuffer_create_failure_skips_for_later_retry() {
        let mut state = SurfaceRecoveryState::new();

        let action = state.record_surface_failure(SurfaceFailure::BackbufferCreateFailed);

        assert_eq!(action, SurfaceRecoveryAction::SkipFrame);
        assert_eq!(state.consecutive_acquire_failures(), 1);
    }

    #[test]
    fn repeated_backbuffer_create_failures_back_off() {
        let mut state = SurfaceRecoveryState::new();

        for expected_count in 1..=3 {
            let action = state.record_surface_failure(SurfaceFailure::BackbufferCreateFailed);
            assert_eq!(action, SurfaceRecoveryAction::SkipFrame);
            assert_eq!(state.consecutive_acquire_failures(), expected_count);
        }
        let action = state.record_surface_failure(SurfaceFailure::BackbufferCreateFailed);
        assert_eq!(action, SurfaceRecoveryAction::WaitForSurfaceChange);
        assert_eq!(state.consecutive_acquire_failures(), 4);

        assert_eq!(
            retry_interval_for_refresh(Some(120), state.consecutive_acquire_failures()),
            Duration::from_millis(100)
        );
    }

    #[test]
    fn repeated_configure_failures_wait_for_surface_change() {
        let mut state = SurfaceRecoveryState::new();

        for _ in 0..3 {
            assert_eq!(
                state.record_surface_failure(SurfaceFailure::ConfigureFailed),
                SurfaceRecoveryAction::SkipFrame
            );
        }

        assert_eq!(
            state.record_surface_failure(SurfaceFailure::ConfigureFailed),
            SurfaceRecoveryAction::WaitForSurfaceChange
        );
    }

    #[test]
    fn zero_sized_surface_waits_for_resize() {
        let mut state = SurfaceRecoveryState::new();

        let action = state.record_surface_failure(SurfaceFailure::ZeroSizedSurface);

        assert_eq!(action, SurfaceRecoveryAction::WaitForResize);
        assert_eq!(state.consecutive_acquire_failures(), 0);
    }

    #[test]
    fn successful_frame_resets_surface_failure_state_and_device_loss_logging() {
        let mut state = SurfaceRecoveryState::new();
        state.record_surface_failure(SurfaceFailure::Acquire(SurfaceAcquireFailure::Timeout));
        let (_action, should_log) = state.record_device_lost();
        assert!(should_log);

        state.record_frame_acquired();

        assert_eq!(state.consecutive_acquire_failures(), 0);
        let (_action, should_log_again) = state.record_device_lost();
        assert!(should_log_again);
    }

    #[test]
    fn retry_interval_matches_refresh_rate_then_backs_off() {
        assert_eq!(retry_interval_for_refresh(Some(120), 1), Duration::from_nanos(8_333_333));
        assert_eq!(retry_interval_for_refresh(Some(60), 3), Duration::from_nanos(16_666_667));
        assert_eq!(retry_interval_for_refresh(Some(120), 4), Duration::from_millis(100));
        assert_eq!(retry_interval_for_refresh(None, 1), Duration::from_millis(16));
    }

    #[test]
    fn repeated_device_loss_logs_once_until_recovery() {
        let mut state = SurfaceRecoveryState::new();

        let (first_action, first_log) = state.record_device_lost();
        let (second_action, second_log) = state.record_device_lost();

        assert_eq!(first_action, DeviceLossAction::RecreateRenderer);
        assert!(first_log);
        assert_eq!(second_action, DeviceLossAction::RecreateRenderer);
        assert!(!second_log);
    }

    #[test]
    fn device_loss_should_request_renderer_recreation() {
        let mut state = SurfaceRecoveryState::new();

        let (action, _should_log) = state.record_device_lost();

        assert_eq!(action, DeviceLossAction::RecreateRenderer);
    }

    #[test]
    fn validation_recreate_then_success_resets_for_next_incident() {
        let mut state = SurfaceRecoveryState::new();

        assert_eq!(
            state.record_surface_failure(SurfaceFailure::Acquire(SurfaceAcquireFailure::Validation)),
            SurfaceRecoveryAction::SkipFrame
        );
        assert_eq!(
            state.record_surface_failure(SurfaceFailure::Acquire(SurfaceAcquireFailure::Validation)),
            SurfaceRecoveryAction::RecreateBackbuffer
        );

        state.record_frame_acquired();

        assert_eq!(
            state.record_surface_failure(SurfaceFailure::Acquire(SurfaceAcquireFailure::Validation)),
            SurfaceRecoveryAction::SkipFrame
        );
        assert_eq!(state.consecutive_acquire_failures(), 1);
    }

    #[test]
    fn zero_size_does_not_inflate_retry_backoff() {
        let mut state = SurfaceRecoveryState::new();

        for _ in 0..8 {
            assert_eq!(
                state.record_surface_failure(SurfaceFailure::ZeroSizedSurface),
                SurfaceRecoveryAction::WaitForResize
            );
        }

        assert_eq!(state.consecutive_acquire_failures(), 0);
        assert_eq!(
            retry_interval_for_refresh(Some(120), state.consecutive_acquire_failures()),
            Duration::from_nanos(8_333_333)
        );
    }

    #[test]
    fn sustained_transient_failures_back_off_without_recreate_storm() {
        let mut state = SurfaceRecoveryState::new();

        for expected_count in 1..=8 {
            let action = state.record_surface_failure(SurfaceFailure::Acquire(SurfaceAcquireFailure::Occluded));
            assert_eq!(action, SurfaceRecoveryAction::SkipFrame);
            assert_eq!(state.consecutive_acquire_failures(), expected_count);
        }

        assert_eq!(
            retry_interval_for_refresh(Some(120), state.consecutive_acquire_failures()),
            Duration::from_millis(100)
        );
    }

    #[test]
    fn max_frame_interval_prefers_slower_interval() {
        assert_eq!(
            max_frame_interval(Some(Duration::from_millis(8)), Some(Duration::from_millis(100))),
            Some(Duration::from_millis(100))
        );
        assert_eq!(
            max_frame_interval(Some(Duration::from_millis(16)), None),
            Some(Duration::from_millis(16))
        );
        assert_eq!(
            max_frame_interval(None, Some(Duration::from_millis(100))),
            Some(Duration::from_millis(100))
        );
        assert_eq!(max_frame_interval(None, None), None);
    }

    #[test]
    fn injected_background_device_loss_sequence_requests_one_renderer_recreate() {
        let mut harness = FaultRecoveryHarness::default();

        assert_eq!(
            harness.inject(InjectedFault::SurfaceAcquire(SurfaceAcquireFailure::Occluded)),
            InjectedRecoveryOutcome::SkipFrame
        );
        assert_eq!(
            harness.inject(InjectedFault::SurfaceAcquire(SurfaceAcquireFailure::Validation)),
            InjectedRecoveryOutcome::RecreateBackbuffer
        );
        assert_eq!(harness.inject(InjectedFault::DeviceLost), InjectedRecoveryOutcome::RecreateRenderer);
        assert_eq!(harness.inject(InjectedFault::DeviceLost), InjectedRecoveryOutcome::RecreateRenderer);

        assert_eq!(harness.renderer_recreate_count, 1);
        assert_eq!(harness.backbuffer_recreate_count, 1);
        assert_eq!(harness.skipped_frame_count, 1);

        assert_eq!(harness.inject(InjectedFault::FrameAcquired), InjectedRecoveryOutcome::FrameAcquired);
        assert_eq!(harness.inject(InjectedFault::DeviceLost), InjectedRecoveryOutcome::RecreateRenderer);
        assert_eq!(harness.renderer_recreate_count, 2);
    }

    #[test]
    fn injected_pipeline_failures_all_request_renderer_recreate() {
        for fault in [
            InjectedFault::BeginFrameFailed,
            InjectedFault::FinishEncoderFailed,
            InjectedFault::QueueSubmitFailed,
            InjectedFault::PresentFailed,
        ] {
            let mut harness = FaultRecoveryHarness::default();

            assert_eq!(harness.inject(fault), InjectedRecoveryOutcome::RecreateRenderer);
            assert_eq!(harness.renderer_recreate_count, 1);
            assert!(harness.renderer_recreate_pending);
        }
    }

    #[test]
    fn injected_surface_fault_matrix_matches_recovery_policy() {
        let mut harness = FaultRecoveryHarness::default();

        let sequence = [
            (InjectedFault::ZeroSizedSurface, InjectedRecoveryOutcome::WaitForResize),
            (InjectedFault::BackbufferCreateFailed, InjectedRecoveryOutcome::SkipFrame),
            (
                InjectedFault::SurfaceAcquire(SurfaceAcquireFailure::Timeout),
                InjectedRecoveryOutcome::SkipFrame,
            ),
            (
                InjectedFault::SurfaceAcquire(SurfaceAcquireFailure::Outdated),
                InjectedRecoveryOutcome::RecreateBackbuffer,
            ),
            (
                InjectedFault::SurfaceAcquire(SurfaceAcquireFailure::Lost),
                InjectedRecoveryOutcome::RecreateBackbuffer,
            ),
            (InjectedFault::AcquirePanic, InjectedRecoveryOutcome::RecreateBackbuffer),
            (InjectedFault::ConfigureFailed, InjectedRecoveryOutcome::SkipFrame),
        ];

        for (fault, expected) in sequence {
            assert_eq!(harness.inject(fault), expected);
        }

        assert_eq!(harness.wait_resize_count, 1);
        assert_eq!(harness.skipped_frame_count, 3);
        assert_eq!(harness.backbuffer_recreate_count, 3);
        assert_eq!(harness.renderer_recreate_count, 0);
    }
}
