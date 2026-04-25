use std::sync::Arc;

use crate::{
    graphics::{Graphics, RenderTargetStore, ScreenCaptureState},
    primitives::PrimitiveBatch,
    profile_new_frame,
    text::TextRenderer,
};

use egor_app::{
    AppConfig, AppHandler, AppRunner, ControlFlow, Fullscreen, PhysicalSize, Window, WindowEvent, input::Input, time::FrameTimer,
};
use egor_render::{
    MemoryHints, Renderer,
    instance::Instance,
    target::{Backbuffer, RenderTarget},
};

type UpdateFn = dyn FnMut(&mut FrameContext);

fn window_surface_size(window: &Window) -> PhysicalSize<u32> {
    #[cfg(target_os = "ios")]
    {
        window.outer_size()
    }

    #[cfg(not(target_os = "ios"))]
    {
        window.inner_size()
    }
}

pub struct AppControl<'a> {
    window: &'a Window,
    requested_size: Option<(u32, u32)>,
    requested_vsync: Option<bool>,
}

impl<'a> AppControl<'a> {
    /// Request the window to redraw its contents on the next frame
    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    /// Set the inner size of the window in physical pixels
    /// Returns the new size depending on platform
    pub fn set_size(&mut self, w: u32, h: u32) {
        let _ = self.window.request_inner_size(PhysicalSize::new(w, h));
        self.requested_size = Some((w, h));
    }

    /// Enable or disable borderless fullscreen mode
    pub fn set_fullscreen(&self, enabled: bool) {
        self.window.set_fullscreen(enabled.then_some(Fullscreen::Borderless(None)));
    }

    /// Enable or disable vertical sync
    /// When enabled, frame presentation is synchronized to the display's refresh
    /// rate, preventing screen tearing
    pub fn set_vsync(&mut self, on: bool) {
        self.requested_vsync = Some(on);
    }

    /// Returns the window's DPI scale factor
    pub fn scale_factor(&self) -> f64 {
        self.window.scale_factor()
    }
}

pub struct FrameContext<'a> {
    pub events: Vec<WindowEvent>,
    pub app: AppControl<'a>,
    pub gfx: Graphics<'a>,
    pub input: &'a Input,
    pub timer: &'a FrameTimer,
}

pub struct App {
    events: Vec<WindowEvent>,
    events_drained: Vec<WindowEvent>,
    update: Option<Box<UpdateFn>>,
    config: Option<AppConfig>,
    vsync: bool,
    text_renderer: Option<TextRenderer>,
    backbuffer: Option<Backbuffer>,
    primitive_batch: PrimitiveBatch,
    memory_hints: MemoryHints,
    render_targets: RenderTargetStore,
    screen_capture: ScreenCaptureState,
    offscreen_batches: Vec<PrimitiveBatch>,
    instance_byte_offsets: Vec<u64>,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Create a new [`App`]
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            events_drained: Vec::new(),
            update: None,
            config: Some(AppConfig::default()),
            vsync: true,
            text_renderer: None,
            backbuffer: None,
            memory_hints: MemoryHints::Performance,
            primitive_batch: PrimitiveBatch::default(),
            render_targets: RenderTargetStore::new(),
            screen_capture: ScreenCaptureState::new(),
            offscreen_batches: Vec::new(),
            instance_byte_offsets: Vec::new(),
        }
    }

    /// Set application title
    pub fn title(mut self, title: &str) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.title = title.into();
        }
        self
    }

    /// Set window icon
    pub fn icon(mut self, icon: egor_app::Icon) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.icon = Some(icon);
        }
        self
    }

    /// Set window size (width, height in pixels)
    pub fn window_size(mut self, width: u32, height: u32) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.width = Some(width);
            c.height = Some(height);
        }
        self
    }

    /// Set the minimum allowed window size (width, height in pixels).
    /// Outside of mobile platforms, the window will not resize below these constraints
    pub fn min_size(mut self, w: u32, h: u32) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.min_size = Some((w, h));
        }
        self
    }

    /// Set the maximum allowed window size (width, height in pixels).
    /// Outside of mobile platforms, the window will not resize above these constraints
    pub fn max_size(mut self, w: u32, h: u32) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.max_size = Some((w, h));
        }
        self
    }

    /// Enable or disable window resizing (defaults to true)
    pub fn resizable(mut self, resizable: bool) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.resizable = resizable;
        }
        self
    }

    /// Enable or disable window maximized (defaults to false)
    pub fn maximized(mut self, maximized: bool) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.maximized = maximized;
        }
        self
    }

    /// Enable or disable fullscreen (defaults to false)
    pub fn fullscreen(mut self, fullscreen: bool) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.fullscreen = fullscreen;
        }
        self
    }

    /// Enable or disable window decorations (defaults to true)
    pub fn decorations(mut self, decorations: bool) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.decorations = decorations;
        }
        self
    }

    /// Enable or disable vsync
    pub fn vsync(mut self, enabled: bool) -> Self {
        self.vsync = enabled;
        self
    }

    /// Set the event loop control flow (defaults to [`ControlFlow::Poll`])
    ///
    /// - `ControlFlow::Poll`: continuously redraws (game-style loop)
    /// - `ControlFlow::Wait`: no frames are produced unless
    ///   [`AppControl::request_redraw()`] is called
    ///
    /// When using `Wait`, you are responsible for requesting redraws
    pub fn control_flow(mut self, control_flow: ControlFlow) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.control_flow = control_flow;
        }
        self
    }

    /// Configure wgpu device memory allocation strategy.
    /// Affects GPU sub-allocation block sizes, useful for mobile or low end devices.
    /// See [`MemoryHints`] for more
    pub fn memory_hints(mut self, hints: MemoryHints) -> Self {
        self.memory_hints = hints;
        self
    }

    /// Set the vertex and index buffer limits for the main frame batch.
    /// Defaults to [`egor_render::batch::GeometryBatch::DEFAULT_MAX_VERTICES`] and [`egor_render::batch::GeometryBatch::DEFAULT_MAX_INDICES`].
    /// Reduce these on memory-constrained platforms, or increase for scenes with dense geometry.
    pub fn batch_limits(mut self, max_verts: usize, max_indices: usize) -> Self {
        self.primitive_batch = PrimitiveBatch::new(max_verts, max_indices);
        self
    }

    /// When enabled, left mouse button presses/moves/releases generate touch events with id 0.
    /// Useful for testing touch logic on desktop.
    pub fn simulate_touch_with_mouse(mut self, enabled: bool) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.simulate_touch_with_mouse = enabled;
        }
        self
    }

    /// When enabled, the first active touch generates mouse position, delta, and left-button events.
    /// Useful on mobile to make existing mouse-based code work with touch.
    pub fn simulate_mouse_with_touch(mut self, enabled: bool) -> Self {
        if let Some(c) = self.config.as_mut() {
            c.simulate_mouse_with_touch = enabled;
        }
        self
    }

    /// Run the app with a per-frame update closure
    pub fn run(mut self, #[allow(unused_mut)] mut update: impl FnMut(&mut FrameContext) + 'static) {
        #[cfg(all(feature = "hot_reload", not(target_arch = "wasm32")))]
        let update = {
            dioxus_devtools::connect_subsecond();

            move |ctx: &mut FrameContext| {
                dioxus_devtools::subsecond::call(|| update(ctx));
            }
        };
        self.update = Some(Box::new(update));

        let config = self.config.take().unwrap();
        AppRunner::new(self, config).run();
    }
}

impl AppHandler<Renderer> for App {
    fn on_window_event(&mut self, _window: &Window, event: &WindowEvent) {
        self.events.push(event.clone());
    }

    async fn with_resource(&mut self, window: Arc<Window>) -> Renderer {
        // WebGPU throws error 'size is zero' if not set
        let size = window_surface_size(&window);
        let (w, h) = (
            if size.width == 0 { 800 } else { size.width },
            if size.height == 0 { 600 } else { size.height },
        );
        let renderer = Renderer::new(window.clone(), &self.memory_hints).await;
        self.backbuffer = Some(Backbuffer::new(
            renderer.instance(),
            renderer.adapter(),
            renderer.device(),
            window,
            w,
            h,
        ));
        renderer
    }

    fn on_ready(&mut self, window: &Window, renderer: &mut Renderer) {
        let (device, format) = (renderer.device(), self.backbuffer.as_ref().unwrap().format());
        self.backbuffer.as_mut().unwrap().set_vsync(device, self.vsync);
        self.text_renderer = Some(TextRenderer::new(device, renderer.queue(), format));

        let size = window_surface_size(window);
        self.resize(size.width, size.height, renderer);
    }

    fn frame(&mut self, _window: &Window, renderer: &mut Renderer, input: &Input, timer: &FrameTimer) {
        profile_new_frame!();
        #[cfg(feature = "profiling")]
        profiling::scope!("frame");

        let Some(update) = &mut self.update else {
            return;
        };

        // Drive wgpu map_async callbacks at the START of the frame.
        // By polling here (not at end-of-frame), the GPU has had a full
        // frame since begin_readback_map — virtually guaranteeing the
        // oldest ring-buffer slot is complete, eliminating stalls.
        if self.screen_capture.readback_in_flight() {
            let _ = renderer.device().poll(egor_render::wgpu::PollType::Poll);
        }

        let Some(backbuffer) = &mut self.backbuffer else {
            return;
        };

        let _ta = web_time::Instant::now();

        let Some(mut frame) = renderer.begin_frame(backbuffer) else {
            return;
        };

        let _t0 = web_time::Instant::now();

        let (w, h) = backbuffer.size();
        renderer.ensure_depth_size(w, h);
        let (device, queue) = (renderer.device().clone(), renderer.queue().clone());
        let format = backbuffer.format();
        let text_renderer = self.text_renderer.as_mut().unwrap();

        self.events_drained.clear();
        std::mem::swap(&mut self.events, &mut self.events_drained);

        let _t_ctx = web_time::Instant::now();

        let mut ctx = FrameContext {
            events: std::mem::take(&mut self.events_drained),
            app: AppControl {
                window: _window,
                requested_size: None,
                requested_vsync: None,
            },
            gfx: Graphics::new(
                renderer,
                &mut self.primitive_batch,
                text_renderer,
                &mut self.render_targets,
                &mut self.screen_capture,
                &mut self.offscreen_batches,
                format,
                w,
                h,
            ),
            input,
            timer,
        };

        let _t_update_start = web_time::Instant::now();
        {
            #[cfg(feature = "profiling")]
            profiling::scope!("user_callback");
            update(&mut ctx);
        }
        let _t_update_end = web_time::Instant::now();

        ctx.events.clear();
        self.events_drained = ctx.events;

        let requested_size = ctx.app.requested_size;
        let requested_vsync = ctx.app.requested_vsync;
        if let Some((pw, ph)) = requested_size {
            ctx.gfx.set_target_size(pw, ph);
        }

        ctx.gfx.upload_camera();

        let _t1 = web_time::Instant::now();

        let has_text = text_renderer.has_entries();
        if has_text {
            text_renderer.prepare(&device, &queue, w, h);
        }

        let _t2 = web_time::Instant::now();

        // Use the flag tracked during batch building instead of scanning all batches.
        let has_rt_overrides = self.primitive_batch.has_rt_overrides();

        let mut batches = self.primitive_batch.drain_all();
        let _batch_count = batches.len();

        // Write custom camera matrices to GPU slots 1..N.
        // Slot 0 is the default egor camera (already written by upload_camera()).
        let stride = renderer.camera_slot_stride();
        {
            let custom_cameras = self.primitive_batch.drain_camera_matrices();
            for (i, cam) in custom_cameras.iter().enumerate() {
                renderer.write_camera_slot((i as u32) + 1, *cam);
            }
        }

        let _t_drain = web_time::Instant::now();

        {
            #[cfg(feature = "profiling")]
            profiling::scope!("batch_upload");
            // Pre-upload all batch geometry to GPU before starting any render pass.
            // This batches the write_buffer calls together for better cache/driver behavior
            // and makes the upload() calls inside draw_batch a no-op (dirty flags already cleared).
            let mut _total_verts: usize = 0;
            let mut _total_indices: usize = 0;
            let mut _total_instances: usize = 0;
            let mut _dirty_batches: usize = 0;
            let inst_size = std::mem::size_of::<Instance>();
            self.instance_byte_offsets.clear();
            let mut running_offset: usize = 0;
            for batch in &mut batches {
                _total_verts += batch.geometry.vertex_count();
                _total_indices += batch.geometry.index_count();
                _total_instances += batch.geometry.instance_count();
                if batch.geometry.is_dirty() {
                    _dirty_batches += 1;
                }
                self.instance_byte_offsets.push((running_offset * inst_size) as u64);
                running_offset += batch.geometry.instance_count();
                batch.geometry.upload_geometry_only(&device, &queue);
            }
            {
                let batch_instance_slices: Vec<&[Instance]> = batches.iter().map(|b| b.geometry.instances()).collect();
                renderer.upload_shared_instances_batched(&batch_instance_slices);
            }
        } // profile_scope batch_upload

        let _t_upload = web_time::Instant::now();

        {
            #[cfg(feature = "profiling")]
            profiling::scope!("render_pass");
            if has_rt_overrides {
                // Multi-pass rendering: split ONLY on render_target changes.
                let mut current_rt: Option<usize> = None;
                let mut first_pass_on_backbuffer = true;
                let mut first_pass_on_rt: Option<usize> = None;

                let mut batch_start = 0;
                while batch_start < batches.len() {
                    let group_rt = batches[batch_start].render_target;
                    let mut batch_end = batch_start + 1;
                    while batch_end < batches.len() && batches[batch_end].render_target == group_rt {
                        batch_end += 1;
                    }

                    // Handle render-target transition
                    if current_rt != group_rt {
                        if let Some(prev_rt) = current_rt {
                            self.render_targets.get(prev_rt).copy_to_sample(&mut frame.encoder);
                        }
                        current_rt = group_rt;
                    }

                    let (view, depth_view, is_first) = if let Some(rt_id) = group_rt {
                        let rt = self.render_targets.get(rt_id);
                        let view = rt.render_view();
                        let dv = rt.offscreen_depth_view();
                        let is_first = first_pass_on_rt != Some(rt_id);
                        if is_first {
                            first_pass_on_rt = Some(rt_id);
                        }
                        (view, dv, is_first)
                    } else {
                        let is_first = first_pass_on_backbuffer;
                        if is_first {
                            first_pass_on_backbuffer = false;
                        }
                        (&frame.view, renderer.depth_view(), is_first)
                    };

                    let (rt_w, rt_h) = if let Some(rt_id) = group_rt {
                        self.render_targets.get(rt_id).size()
                    } else {
                        (w, h)
                    };

                    {
                        let mut r_pass = if is_first {
                            renderer.begin_render_pass_with_depth(&mut frame.encoder, view, depth_view, true)
                        } else {
                            renderer.begin_render_pass_load_with_depth(&mut frame.encoder, view, depth_view)
                        };

                        let first_batch = &batches[batch_start];
                        renderer.bind_pass_state(&mut r_pass, first_batch.texture_id, first_batch.shader_id);
                        let mut cur_tex = first_batch.texture_id;
                        let mut cur_shd = first_batch.shader_id;
                        let mut cur_cam_offset = u32::MAX;
                        let mut quad_bound = true;
                        let full_scissor = (0u32, 0u32, rt_w.max(1), rt_h.max(1));
                        let mut cur_scissor = (u32::MAX, u32::MAX, u32::MAX, u32::MAX);

                        for idx in batch_start..batch_end {
                            let batch = &mut batches[idx];
                            let target_scissor = match batch.scissor {
                                Some((sx, sy, sw, sh)) => {
                                    let sx = sx.min(rt_w.saturating_sub(1));
                                    let sy = sy.min(rt_h.saturating_sub(1));
                                    let sw = sw.min(rt_w - sx).max(1);
                                    let sh = sh.min(rt_h - sy).max(1);
                                    (sx, sy, sw, sh)
                                }
                                None => full_scissor,
                            };
                            if cur_scissor != target_scissor {
                                r_pass.set_scissor_rect(target_scissor.0, target_scissor.1, target_scissor.2, target_scissor.3);
                                cur_scissor = target_scissor;
                            }
                            let offset = batch.camera_slot * stride;
                            if let Some(shared_buf) = renderer.shared_instance_buffer() {
                                renderer.draw_batch_shared(
                                    &mut r_pass,
                                    &mut batch.geometry,
                                    batch.texture_id,
                                    batch.shader_id,
                                    offset,
                                    &mut cur_tex,
                                    &mut cur_shd,
                                    &mut cur_cam_offset,
                                    &mut quad_bound,
                                    shared_buf,
                                    self.instance_byte_offsets[idx],
                                );
                            } else {
                                renderer.draw_batch(
                                    &mut r_pass,
                                    &mut batch.geometry,
                                    batch.texture_id,
                                    batch.shader_id,
                                    offset,
                                    &mut cur_tex,
                                    &mut cur_shd,
                                    &mut cur_cam_offset,
                                    &mut quad_bound,
                                );
                            }
                        }

                        if batch_end >= batches.len() && group_rt.is_none() && has_text {
                            text_renderer.render(&mut r_pass);
                        }
                    }

                    batch_start = batch_end;
                }

                // Copy the last offscreen target if it was active
                if let Some(prev_rt) = current_rt {
                    self.render_targets.get(prev_rt).copy_to_sample(&mut frame.encoder);
                }

                if batches.is_empty() {
                    let mut r_pass = renderer.begin_render_pass(&mut frame.encoder, &frame.view);
                    if has_text {
                        text_renderer.render(&mut r_pass);
                    }
                }
            } else {
                // Single render pass (no render target overrides)
                {
                    let mut r_pass = renderer.begin_render_pass(&mut frame.encoder, &frame.view);

                    if let Some(first) = batches.first() {
                        renderer.bind_pass_state(&mut r_pass, first.texture_id, first.shader_id);
                        let mut cur_tex = first.texture_id;
                        let mut cur_shd = first.shader_id;
                        let mut cur_cam_offset = u32::MAX;
                        let mut quad_bound = true;
                        let full_scissor = (0u32, 0u32, w.max(1), h.max(1));
                        let mut cur_scissor = (u32::MAX, u32::MAX, u32::MAX, u32::MAX);

                        for (idx, batch) in batches.iter_mut().enumerate() {
                            let target_scissor = match batch.scissor {
                                Some((sx, sy, sw, sh)) => {
                                    let sx = sx.min(w.saturating_sub(1));
                                    let sy = sy.min(h.saturating_sub(1));
                                    let sw = sw.min(w - sx).max(1);
                                    let sh = sh.min(h - sy).max(1);
                                    (sx, sy, sw, sh)
                                }
                                None => full_scissor,
                            };
                            if cur_scissor != target_scissor {
                                r_pass.set_scissor_rect(target_scissor.0, target_scissor.1, target_scissor.2, target_scissor.3);
                                cur_scissor = target_scissor;
                            }
                            let offset = batch.camera_slot * stride;
                            if let Some(shared_buf) = renderer.shared_instance_buffer() {
                                renderer.draw_batch_shared(
                                    &mut r_pass,
                                    &mut batch.geometry,
                                    batch.texture_id,
                                    batch.shader_id,
                                    offset,
                                    &mut cur_tex,
                                    &mut cur_shd,
                                    &mut cur_cam_offset,
                                    &mut quad_bound,
                                    shared_buf,
                                    self.instance_byte_offsets[idx],
                                );
                            } else {
                                renderer.draw_batch(
                                    &mut r_pass,
                                    &mut batch.geometry,
                                    batch.texture_id,
                                    batch.shader_id,
                                    offset,
                                    &mut cur_tex,
                                    &mut cur_shd,
                                    &mut cur_cam_offset,
                                    &mut quad_bound,
                                );
                            }
                        }
                    }

                    if has_text {
                        text_renderer.render(&mut r_pass);
                    }
                }
            }

            // Recycle batch GPU buffers for reuse next frame.
            self.primitive_batch.recycle(batches);
        } // profile_scope render_pass

        let _t_pass = web_time::Instant::now();

        // Screen capture: blit-downsample the backbuffer into a small capture
        // texture and encode a copy_texture_to_buffer for async readback.
        let capture_active = self.screen_capture.is_requested();
        if capture_active {
            #[cfg(feature = "profiling")]
            profiling::scope!("screen_capture");
            let bb_ptr = frame.backbuffer_texture().map(|t| t as *const egor_render::Texture);
            if let Some(ptr) = bb_ptr {
                // SAFETY: the texture is owned by Frame.presentable which is
                // not dropped until after this block, and we only read it.
                let bb_tex = unsafe { &*ptr };
                self.screen_capture.capture_from_texture(&device, &mut frame.encoder, bb_tex);
            } else {
                eprintln!("[egor] Screen capture requested but no backbuffer texture available");
                self.screen_capture.request(0, 0, false);
            }
        }

        let _t_submit0 = web_time::Instant::now();
        {
            #[cfg(feature = "profiling")]
            profiling::scope!("submit_present");
            let (commands, presentable) = renderer.finish_encoder(frame);
            let _t_encode = web_time::Instant::now();
            renderer.submit_commands(commands);
            let _t_submit1 = web_time::Instant::now();
            if let Some(p) = presentable {
                p.present();
            }
        } // profile_scope submit_present
        let _t_present = web_time::Instant::now();

        // Start the async map AFTER submit so the staging buffer isn't
        // in a pending-map state when the command buffer is submitted.
        if capture_active {
            self.screen_capture.begin_readback_map();
        }

        let _t_end = web_time::Instant::now();

        if let Some((rw, rh)) = requested_size {
            self.backbuffer.as_mut().unwrap().resize(&device, rw, rh);
        }
        if let Some(vsync) = requested_vsync {
            self.backbuffer.as_mut().unwrap().set_vsync(&device, vsync);
            self.vsync = vsync;
        }
    }

    fn resize(&mut self, w: u32, h: u32, renderer: &mut Renderer) {
        self.backbuffer.as_mut().unwrap().resize(renderer.device(), w, h);
        renderer.ensure_depth_size(w, h);
        self.text_renderer.as_mut().unwrap().resize(w, h, renderer.queue());
    }

    fn suspended(&mut self) {
        self.backbuffer = None;
    }

    fn resumed(&mut self, window: Arc<Window>, renderer: &mut Renderer) {
        let size = window_surface_size(&window);
        let device = renderer.device();
        let mut backbuffer = Backbuffer::new(renderer.instance(), renderer.adapter(), device, window, size.width, size.height);
        backbuffer.set_vsync(device, self.vsync);
        self.backbuffer = Some(backbuffer);
    }
}
