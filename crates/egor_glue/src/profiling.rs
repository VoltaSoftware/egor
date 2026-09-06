/// When the `profiling` feature is enabled, these re-export the `profiling`
/// crate macros which delegate to puffin (or any other backend).
/// When disabled, they expand to nothing — zero cost, not even a branch.

/// Call once at the very start of each frame.
#[cfg(feature = "profiling")]
#[macro_export]
macro_rules! profile_new_frame {
    () => {
        ::puffin::GlobalProfiler::lock().new_frame();
    };
}

#[cfg(not(feature = "profiling"))]
#[macro_export]
macro_rules! profile_new_frame {
    () => {};
}

/// Start the puffin HTTP server on the given address (e.g. `"0.0.0.0:8585"`).
/// Does nothing when profiling is disabled.
#[cfg(feature = "profiling")]
pub fn start_puffin_server(bind: &str) {
    puffin::set_scopes_on(true);
    let _server = puffin_http::Server::new(bind).expect("puffin HTTP server start");
    std::mem::forget(_server);
    eprintln!("[puffin] Profiling server listening on {bind}");
}

#[cfg(not(feature = "profiling"))]
pub fn start_puffin_server(_bind: &str) {}

#[cfg(feature = "profiling")]
pub(crate) fn describe_wakeup(cause: egor_app::StartCause) -> String {
    use egor_app::StartCause;
    let now = web_time::Instant::now();
    let (kind, start, deadline) = match cause {
        StartCause::ResumeTimeReached { start, requested_resume } => ("timer", Some(start), Some(requested_resume)),
        StartCause::WaitCancelled { start, requested_resume } => ("event", Some(start), requested_resume),
        StartCause::Poll => ("poll", None, None),
        StartCause::Init => ("init", None, None),
    };
    let waited_us = start.map(|start| now.saturating_duration_since(start).as_micros()).unwrap_or(0);
    let late_us = deadline
        .map(|deadline| now.saturating_duration_since(deadline).as_micros())
        .unwrap_or(0);
    format!("{kind} waited_us={waited_us} late_us={late_us}")
}

#[cfg(all(feature = "profiling", target_os = "windows"))]
pub(crate) struct FrameCpuCycles(u64);

#[cfg(all(feature = "profiling", target_os = "windows"))]
impl FrameCpuCycles {
    pub(crate) fn start() -> Self {
        Self(Self::current())
    }

    fn current() -> u64 {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentThread() -> *mut std::ffi::c_void;
            fn QueryThreadCycleTime(thread: *mut std::ffi::c_void, cycles: *mut u64) -> i32;
        }
        let mut cycles = 0;
        // The pseudo-handle is valid for this thread and must not be closed.
        unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut cycles) };
        cycles
    }
}

#[cfg(all(feature = "profiling", target_os = "windows"))]
impl Drop for FrameCpuCycles {
    fn drop(&mut self) {
        // Keep cycles as cycles: CPU frequency changes prevent conversion to time.
        let cycles = Self::current().saturating_sub(self.0);
        profiling::scope!("render_thread_cycles", &cycles.to_string());
    }
}
