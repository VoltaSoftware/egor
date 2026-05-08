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
