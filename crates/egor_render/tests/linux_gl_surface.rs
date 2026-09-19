#![cfg(target_os = "linux")]

use egor_render::{
    Renderer, RendererBackendPreference,
    target::{Backbuffer, RenderTarget},
    wgpu,
};
use std::{sync::Arc, time::Duration};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    platform::{wayland::EventLoopBuilderExtWayland, x11::EventLoopBuilderExtX11},
    window::{Window, WindowId},
};

const BACKENDS: [wgpu::Backend; 4] = [
    wgpu::Backend::Gl,
    wgpu::Backend::Vulkan,
    wgpu::Backend::Gl,
    wgpu::Backend::Gl,
];
const FRAMES_PER_BACKEND: u32 = 3;

struct TestLogger;

impl log::Log for TestLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
            || (metadata.level() <= log::Level::Debug && metadata.target() == "wgpu_hal::gles::egl")
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("{} {}: {}", record.level(), record.target(), record.args());
        }
    }

    fn flush(&self) {}
}

#[test]
#[ignore = "requires a GPU and Wayland display; set EGOR_TEST_WINDOW_SYSTEM=x11 to test X11"]
fn opengl_presents_after_backend_switch_and_resize() {
    log::set_logger(&TestLogger).unwrap();
    log::set_max_level(log::LevelFilter::Debug);
    let mut builder = EventLoop::builder();
    EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
    match std::env::var("EGOR_TEST_WINDOW_SYSTEM").as_deref() {
        Ok("x11") => {
            builder.with_x11();
        }
        Ok("wayland") | Err(_) => {
            builder.with_wayland();
        }
        Ok(other) => panic!("unsupported test window system: {other}"),
    }
    let event_loop = builder.build().unwrap();
    let mut app = SurfaceTest::default();
    event_loop.run_app(&mut app).unwrap();
    assert_eq!(
        app.backend_index,
        BACKENDS.len(),
        "all backend transitions must present frames"
    );
    assert_eq!(
        app.resize_count,
        BACKENDS.len(),
        "each backend must present after resizing"
    );
}

#[derive(Default)]
struct SurfaceTest {
    window: Option<Arc<Window>>,
    backbuffer: Option<Backbuffer>,
    renderer: Option<Renderer>,
    backend_index: usize,
    frames: u32,
    resize_count: usize,
    resized: bool,
}

impl SurfaceTest {
    fn resized_size(&self) -> PhysicalSize<u32> {
        PhysicalSize::new(352 + self.backend_index as u32 * 16, 256)
    }

    fn create_renderer(&mut self) {
        self.backbuffer = None;
        self.renderer = None;
        let window = self.window.as_ref().unwrap();
        let backend = BACKENDS[self.backend_index];
        let preference = RendererBackendPreference::Backends(wgpu::Backends::from(backend));
        let mut renderer = pollster::block_on(Renderer::try_new_with_backend(
            window.clone(),
            &wgpu::MemoryHints::default(),
            preference,
        ))
        .unwrap_or_else(|error| panic!("{backend:?} initialization failed: {error}"));
        assert_eq!(renderer.adapter_info().backend, backend);
        println!(
            "stage {}: {:?}",
            self.backend_index,
            renderer.adapter_info()
        );
        renderer
            .device()
            .on_uncaptured_error(Arc::new(|error| panic!("GPU validation failed: {error}")));
        let size = window.inner_size();
        renderer.ensure_depth_size(size.width, size.height);
        self.backbuffer = Some(
            renderer
                .take_startup_backbuffer(size.width, size.height)
                .unwrap()
                .unwrap(),
        );
        self.renderer = Some(renderer);
        self.frames = 0;
        self.resized = false;
        window.request_redraw();
    }
}

impl ApplicationHandler for SurfaceTest {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.window = Some(Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("egor Linux OpenGL regression")
                        .with_inner_size(PhysicalSize::new(320, 240)),
                )
                .unwrap(),
        ));
        self.create_renderer();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let resized_size = self.resized_size();
        let window = self.window.as_ref().unwrap();
        let (Some(renderer), Some(backbuffer)) = (self.renderer.as_mut(), self.backbuffer.as_mut())
        else {
            return;
        };
        match event {
            WindowEvent::RedrawRequested => {
                let mut frame = renderer
                    .begin_frame(backbuffer)
                    .expect("surface must acquire a frame");
                drop(renderer.begin_render_pass(&mut frame.encoder, &frame.view));
                window.pre_present_notify();
                renderer.end_frame(frame);
                renderer
                    .device()
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: Some(Duration::from_secs(10)),
                    })
                    .unwrap();
                self.frames += 1;
                if self.frames == 1
                    && let Some(size) = window.request_inner_size(resized_size)
                {
                    backbuffer.resize(renderer.device(), size.width, size.height);
                    renderer.ensure_depth_size(size.width, size.height);
                    self.resized = size == resized_size;
                }
                if self.frames >= FRAMES_PER_BACKEND && self.resized {
                    println!(
                        "stage {}: presented {} frames, including after resize",
                        self.backend_index, self.frames
                    );
                    self.resize_count += 1;
                    self.backend_index += 1;
                    if self.backend_index == BACKENDS.len() {
                        self.backbuffer = None;
                        self.renderer = None;
                        event_loop.exit();
                    } else {
                        self.create_renderer();
                    }
                } else {
                    window.request_redraw();
                }
            }
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                backbuffer.resize(renderer.device(), size.width, size.height);
                renderer.ensure_depth_size(size.width, size.height);
                if self.frames > 0 && size == resized_size {
                    self.resized = true;
                }
                window.request_redraw();
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            _ => {}
        }
    }
}
