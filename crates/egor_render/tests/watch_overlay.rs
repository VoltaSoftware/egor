#![cfg(any(target_os = "windows", target_os = "linux"))]

use egor_render::{Renderer, batch::GeometryBatch, instance::Instance, wgpu};
use std::{sync::Arc, time::Duration};
#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopBuilderExtWindows;
#[cfg(target_os = "linux")]
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::{event_loop::EventLoop, window::Window};

#[test]
#[ignore = "requires a native GPU and display; run with --ignored"]
fn watch_overlay_preserves_transparency_behind_foreground() {
    let event_loop = EventLoop::builder().with_any_thread(true).build().unwrap();
    #[allow(deprecated)]
    let window = Arc::new(
        event_loop
            .create_window(Window::default_attributes().with_visible(false))
            .unwrap(),
    );
    let mut renderer = pollster::block_on(Renderer::new(window, &wgpu::MemoryHints::default()));
    assert!(renderer.supports_watch_overlay_capture());
    renderer.upload_camera_matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    let device = renderer.device();
    let size = wgpu::Extent3d {
        width: 4,
        height: 1,
        depth_or_array_layers: 1,
    };
    let make_texture = |format| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("watch overlay regression"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let color = make_texture(renderer.surface_format());
    let overlay = make_texture(wgpu::TextureFormat::Rgba8Unorm);
    let color_view = color.create_view(&Default::default());
    let overlay_view = overlay.create_view(&Default::default());
    let (_depth, depth_view) = Renderer::create_depth_texture(device, size.width, size.height);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("watch overlay pixels"),
        size: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Leave pixel 0 undrawn. Put server-owned terrain behind pixels 1..3,
    // opaque black foreground at pixel 2, and translucent red at pixel 3.
    let mut batch = GeometryBatch::default();
    batch.push_instance(
        Instance::new(
            [1.5, 0.0, 0.0, 2.0],
            [0.25, 0.0, -0.5],
            [0.2, 0.7, 0.3, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        )
        .with_watch_overlay(0.0),
    );
    for (x, color) in [(0.25, [0.0, 0.0, 0.0, 1.0]), (0.75, [1.0, 0.0, 0.0, 0.5])] {
        batch.push_instance(Instance::new(
            [0.5, 0.0, 0.0, 2.0],
            [x, 0.0, -0.5],
            color,
            [0.0, 0.0, 1.0, 1.0],
        ));
    }

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = renderer.begin_render_pass_with_watch_overlay_depth_clear_color(
            &mut encoder,
            &color_view,
            &overlay_view,
            &depth_view,
            wgpu::Color::BLACK,
            true,
        );
        renderer.bind_pass_state_with_watch_overlay(&mut pass, None, None, false, true);
        let mut camera_offset = u32::MAX;
        renderer.draw_batch_with_watch_overlay(
            &mut pass,
            &mut batch,
            None,
            None,
            false,
            0,
            &mut None,
            &mut None,
            &mut false,
            &mut camera_offset,
            &mut false,
            true,
        );
    }
    encoder.copy_texture_to_buffer(
        overlay.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                rows_per_image: Some(1),
            },
        },
        size,
    );
    let submission = renderer.queue().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(10)),
        })
        .unwrap();
    rx.recv_timeout(Duration::from_secs(10)).unwrap().unwrap();
    let pixels = readback.slice(..).get_mapped_range().unwrap();
    assert_eq!(
        &pixels[..4],
        &[0, 0, 0, 0],
        "undrawn background must reveal the server map"
    );
    assert_eq!(
        &pixels[4..8],
        &[0, 0, 0, 0],
        "server-owned terrain must be omitted from the overlay"
    );
    assert_eq!(
        &pixels[8..12],
        &[0, 0, 0, 255],
        "black foreground must remain opaque"
    );
    assert!(
        pixels[12].abs_diff(128) <= 1 && pixels[15].abs_diff(128) <= 1,
        "translucent foreground must retain its alpha: {:?}",
        &pixels[12..16]
    );
    assert_eq!(&pixels[13..15], &[0, 0]);
}
