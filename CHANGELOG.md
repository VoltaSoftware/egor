## Unreleased

- Fixed OpenGL initialization, resizing, and renderer switching on Linux Wayland and X11 by keeping the window's display connection alive and selecting a compatible EGL configuration through the matching wgpu fork.
