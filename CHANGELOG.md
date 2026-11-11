## Unreleased

- Enable triple buffering (2 frames maximum latency) on Android swapchains to avoid single-frame VSync deadline misses and frame rate halving.

- Discard screenshot readbacks invalidated by WebGPU device loss instead of panicking during recovery.
- Fixed OpenGL initialization, resizing, and renderer switching on Linux Wayland and X11 by keeping the window's display connection alive and selecting a compatible EGL configuration through the matching wgpu fork.
