use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};

use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, MapMode};

use crate::instance::Instance;

const PENDING: u8 = 0;
const READY: u8 = 1;
const FAILED: u8 = 2;
const MAX_SLOTS: usize = 3;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InstanceUploadMetrics {
    pub pool_buffers: u64,
    pub pool_bytes: u64,
    pub allocations: u64,
    pub fallback_uploads: u64,
}

struct Slot {
    buffer: Buffer,
    state: Arc<AtomicU8>,
}

impl Slot {
    fn new(device: &Device, size: u64) -> Self {
        Self {
            buffer: device.create_buffer(&BufferDescriptor {
                label: Some("Reusable Instance Upload"),
                size,
                usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            }),
            state: Arc::new(AtomicU8::new(READY)),
        }
    }
}

/// Reuses completed uploads without waiting for a GPU fence. Saturation uses
/// the renderer's normal queue-write path instead of growing the pool.
#[derive(Default)]
pub(crate) struct InstanceUploadPool {
    slots: Vec<Slot>,
    allocations: u64,
    fallback_uploads: u64,
}

impl InstanceUploadPool {
    pub(crate) fn metrics(&self) -> InstanceUploadMetrics {
        InstanceUploadMetrics {
            pool_buffers: self.slots.len() as u64,
            pool_bytes: self.slots.iter().map(|s| s.buffer.size()).sum(),
            allocations: self.allocations,
            fallback_uploads: self.fallback_uploads,
        }
    }

    pub(crate) fn upload(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        target: &Buffer,
        batches: &[&[Instance]],
        size: u64,
    ) -> bool {
        let available = self
            .slots
            .iter()
            .position(|slot| slot.state.load(Ordering::Acquire) != PENDING);
        let index = if let Some(index) = available {
            index
        } else if self.slots.len() < MAX_SLOTS {
            self.slots.push(Slot::new(device, target.size()));
            self.allocations += 1;
            self.slots.len() - 1
        } else {
            self.fallback_uploads += 1;
            return false;
        };
        let slot = &mut self.slots[index];
        if slot.buffer.size() < size || slot.state.load(Ordering::Acquire) == FAILED {
            *slot = Slot::new(device, target.size());
            self.allocations += 1;
        }
        {
            let mut view = slot
                .buffer
                .slice(0..size)
                .get_mapped_range_mut()
                .expect("completed instance upload map");
            let mut offset = 0;
            for batch in batches {
                let bytes = bytemuck::cast_slice::<Instance, u8>(batch);
                if !bytes.is_empty() {
                    view.slice(offset..offset + bytes.len())
                        .copy_from_slice(bytes);
                    offset += bytes.len();
                }
            }
        }
        slot.buffer.unmap();
        encoder.copy_buffer_to_buffer(&slot.buffer, 0, target, 0, size);
        slot.state.store(PENDING, Ordering::Release);
        let state = Arc::clone(&slot.state);
        encoder.map_buffer_on_submit(&slot.buffer, MapMode::Write, .., move |result| {
            state.store(
                if result.is_ok() { READY } else { FAILED },
                Ordering::Release,
            );
        });
        true
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a native OpenGL adapter"]
    fn upload_pool_bounds_reuses_and_grows_after_gpu_completion() {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::GL;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        assert_eq!(adapter.get_info().backend, wgpu::Backend::Gl);
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).unwrap();
        let mut pool = InstanceUploadPool::default();
        for round in 0..8 {
            let multiplier = if round < 4 { 1 } else { 5 };
            let target = device.create_buffer(&BufferDescriptor {
                label: Some("instance upload target"),
                size: (MAX_SLOTS * multiplier * size_of::<Instance>()) as u64,
                usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut commands = Vec::new();
            let mut outputs = Vec::new();
            for slot in 0..MAX_SLOTS {
                let mut values = vec![Instance::identity(); (slot + 1) * multiplier];
                for (index, value) in values.iter_mut().enumerate() {
                    value.translate[0] = (round * 100 + slot * 10 + index) as f32;
                }
                let bytes = bytemuck::cast_slice::<Instance, u8>(&values);
                let size = bytes.len() as u64;
                let output = device.create_buffer(&BufferDescriptor {
                    label: Some("instance upload verification"),
                    size,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                let split = values.len() / 2;
                assert!(pool.upload(
                    &device,
                    &mut encoder,
                    &target,
                    &[&values[..split], &values[split..]],
                    size
                ));
                encoder.copy_buffer_to_buffer(&target, 0, &output, 0, size);
                encoder.map_buffer_on_submit(&output, MapMode::Read, .., Result::unwrap);
                commands.push(encoder.finish());
                outputs.push((output, bytes.to_vec()));
            }
            let mut extra = device.create_command_encoder(&Default::default());
            assert!(!pool.upload(
                &device,
                &mut extra,
                &target,
                &[&[Instance::identity()]],
                size_of::<Instance>() as u64
            ));
            assert_eq!(pool.metrics().pool_buffers, MAX_SLOTS as u64);
            queue.submit(commands);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            for (output, expected) in outputs {
                {
                    let view = output.slice(..).get_mapped_range().unwrap();
                    assert_eq!(&*view, &expected);
                }
                output.unmap();
            }
            assert_eq!(pool.metrics().allocations, if round < 4 { 3 } else { 6 });
            assert_eq!(pool.metrics().fallback_uploads, (round + 1) as u64);
        }
    }
}
