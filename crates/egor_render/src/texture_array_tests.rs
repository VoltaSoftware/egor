use super::*;
use crate::instance::Instance;
use crate::vertex::{QUAD_INDICES, QUAD_VERTICES, Vertex};
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires a real GPU; run with WGPU_BACKEND=vulkan or gl"]
fn array_pages_render_in_one_draw() {
    pollster::block_on(async {
        let gpu =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = gpu.request_adapter(&Default::default()).await.unwrap();
        println!("{:?}", adapter.get_info());
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        // Exercise the actual pipeline layout, default/watch shaders and upload path.
        let _pipelines =
            crate::pipeline::Pipelines::new(&device, TextureFormat::Rgba8UnormSrgb, true);
        let mut textures = Textures::new(&device, &queue);
        let colors = [
            [255u8, 0, 0, 255],
            [0, 255, 0, 255],
            [0, 0, 255, 255],
            [255, 255, 0, 255],
        ];
        let pixels = colors[..3]
            .iter()
            .flat_map(|c| c.repeat(64))
            .collect::<Vec<_>>();
        let id = textures
            .insert_array_raw(&device, &queue, 8, 8, 3, &pixels)
            .unwrap();
        assert!(
            textures
                .insert_array_raw(&device, &queue, 8, 8, 4, &pixels)
                .is_err()
        );
        let normal_id = textures.insert_raw_nearest(&device, &queue, 8, 8, &colors[3].repeat(64));
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[
                1f32, 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
            ]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let camera_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera.as_entire_binding(),
            }],
        });
        let make_pipeline = |array| {
            let source = crate::prepare_texture_shader(include_str!("../shader.wgsl"), array);
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&textures.layout), Some(&camera_layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(Vertex::desc()), Some(Instance::desc())],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: TextureFormat::Rgba8UnormSrgb,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
            pipeline
        };
        let pipelines = [make_pipeline(false), make_pipeline(true)];
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&QUAD_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let instances = (0..4)
            .map(|page| {
                Instance::new(
                    [0.5, 0., 0., 2.],
                    [-0.75 + page as f32 * 0.5, 0., 0.],
                    [1.; 4],
                    [0., 0., 1., 1.],
                )
                .with_texture_layer(page % 3)
            })
            .collect::<Vec<_>>();
        let instances = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let target = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: 128,
                height: 32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 512 * 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipelines[1]);
            textures.get(Some(id)).bind(&mut pass, 0);
            pass.set_bind_group(1, &camera_bind, &[]);
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.set_vertex_buffer(1, instances.slice(..));
            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..6, 0, 0..3);
            pass.set_pipeline(&pipelines[0]);
            textures.get(Some(normal_id)).bind(&mut pass, 0);
            pass.draw_indexed(0..6, 0, 3..4);
        }
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(512),
                    rows_per_image: Some(32),
                },
            },
            target.size(),
        );
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let data = readback.slice(..).get_mapped_range().unwrap();
        for (page, color) in colors.iter().enumerate() {
            for y in 0..32 {
                for x in page * 32..page * 32 + 32 {
                    assert_eq!(&data[y * 512 + x * 4..y * 512 + x * 4 + 4], color);
                }
            }
        }
    });
}
