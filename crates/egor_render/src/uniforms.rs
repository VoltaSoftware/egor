use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferUsages, Device, Queue,
    ShaderStages,
    util::{BufferInitDescriptor, DeviceExt},
};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

struct UniformEntry {
    buffer: Buffer,
    bind_group: BindGroup,
}

pub(crate) struct Uniforms {
    layout: BindGroupLayout,
    store: Vec<Option<UniformEntry>>,
    /// Slots of removed uniforms, which the next inserts reuse.
    free_slots: Vec<usize>,
}

impl Uniforms {
    pub fn new(device: &Device) -> Self {
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Shared Uniform Layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        Self {
            layout,
            store: Vec::new(),
            free_slots: Vec::new(),
        }
    }

    pub fn bind_group(&self, uniform_id: usize) -> &BindGroup {
        &self.store[uniform_id]
            .as_ref()
            .expect("drew with a removed uniform")
            .bind_group
    }

    pub fn layout(&self) -> &BindGroupLayout {
        &self.layout
    }

    pub fn insert(&mut self, device: &Device, data: &[u8]) -> usize {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("User Uniform Buffer"),
            contents: data,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &self.layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        let entry = Some(UniformEntry { buffer, bind_group });
        match self.free_slots.pop() {
            Some(id) => {
                self.store[id] = entry;
                id
            }
            None => {
                self.store.push(entry);
                self.store.len() - 1
            }
        }
    }

    /// Frees uniform `id`; its slot goes to the next insert.
    pub fn remove(&mut self, id: usize) {
        if self.store.get_mut(id).and_then(Option::take).is_some() {
            self.free_slots.push(id);
        }
    }

    pub fn write(&mut self, queue: &Queue, id: usize, data: &[u8]) {
        match self.store.get(id) {
            Some(Some(entry)) => queue.write_buffer(&entry.buffer, 0, data),
            _ => log::error!("[egor] can't write uniform {id}: it doesn't exist"),
        }
    }
}
