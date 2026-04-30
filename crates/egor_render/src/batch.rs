use wgpu::{Buffer, BufferDescriptor, BufferUsages, COPY_BUFFER_ALIGNMENT, Device, IndexFormat, Queue, RenderPass};

use crate::{instance::Instance, vertex::Vertex};

/// A batch of geometry (vertices + indices) that can be drawn in a single GPU call
///
/// Tracks CPU vertex/index data, lazily uploads GPU buffers and prevents overflowing `u16` indices.
/// Supports two draw paths:
/// - Baked geometry (vertices + indices) for paths, polygons, arbitrary meshes
/// - Instanced drawing (instance buffer) for quads/rects/sprites via a static unit quad
pub struct GeometryBatch {
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
    vertex_buffer: Option<Buffer>,
    index_buffer: Option<Buffer>,
    vertices_dirty: bool,
    indices_dirty: bool,
    instances: Vec<Instance>,
    instance_buffer: Option<Buffer>,
    instances_dirty: bool,
    max_verticies: usize,
    max_indices: usize,
}

impl Default for GeometryBatch {
    fn default() -> Self {
        Self::new(Self::DEFAULT_MAX_VERTICES, Self::DEFAULT_MAX_INDICES)
    }
}

impl GeometryBatch {
    pub const DEFAULT_INDICES_PER_VERT: usize = 6;
    pub const DEFAULT_MAX_VERTICES: usize = 128 as usize;
    pub const DEFAULT_MAX_INDICES: usize = Self::DEFAULT_MAX_VERTICES * Self::DEFAULT_INDICES_PER_VERT;

    // Creates a new batch with specified max vert/idx counts
    pub fn new(max_verticies: usize, max_indices: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(max_verticies),
            indices: Vec::with_capacity(max_indices),
            vertex_buffer: None,
            index_buffer: None,
            vertices_dirty: false,
            indices_dirty: false,
            instances: Vec::with_capacity(Self::INITIAL_INSTANCE_CAPACITY),
            instance_buffer: None,
            instances_dirty: false,
            max_verticies,
            max_indices,
        }
    }

    const INITIAL_INSTANCE_CAPACITY: usize = 1_024;

    // Returns true if adding verts/indices would exceed max allowed
    pub fn would_overflow(&self, vert_count: usize, idx_count: usize) -> bool {
        self.vertices.len() + vert_count > self.max_verticies || self.indices.len() + idx_count > self.max_indices
    }

    /// Reserves space for `vert_count` + `idx_count`
    ///
    /// Returns mutable slices to the new ranges and the base vertex offset.
    /// Returns `None` if this would exceed `u16` limits.
    /// Marks buffers dirty
    pub fn try_allocate(&mut self, vert_count: usize, idx_count: usize) -> Option<(&mut [Vertex], &mut [u16], u16)> {
        if self.would_overflow(vert_count, idx_count) {
            return None;
        }

        let v_start = self.vertices.len();
        let i_start = self.indices.len();

        self.vertices.resize(v_start + vert_count, Vertex::zeroed());
        self.indices.resize(i_start + idx_count, 0);

        self.vertices_dirty = true;
        self.indices_dirty = true;

        Some((&mut self.vertices[v_start..], &mut self.indices[i_start..], v_start as u16))
    }

    /// Adds vertices/indices, returns false if it would overflow
    pub fn push(&mut self, verts: &[Vertex], indices: &[u16]) -> bool {
        if self.would_overflow(verts.len(), indices.len()) {
            return false;
        }

        let idx_offset = self.vertices.len() as u16;
        self.vertices.extend_from_slice(verts);
        self.indices.extend(indices.iter().map(|i| *i + idx_offset));

        self.vertices_dirty = true;
        self.indices_dirty = true;

        true
    }

    /// Pushes an instance for instanced drawing
    pub fn push_instance(&mut self, instance: Instance) {
        self.instances.push(instance);
        self.instances_dirty = true;
    }

    /// Push an instance without setting the dirty flag.
    /// Caller MUST ensure `mark_instances_dirty()` is called once before upload.
    #[inline(always)]
    pub fn push_instance_no_dirty(&mut self, instance: Instance) {
        self.instances.push(instance);
    }

    /// Mark the instance buffer as needing re-upload.
    #[inline(always)]
    pub fn mark_instances_dirty(&mut self) {
        self.instances_dirty = true;
    }

    /// Returns true if there is nothing to draw in either path
    pub(crate) fn is_empty(&self) -> bool {
        self.indices.is_empty() && self.instances.is_empty()
    }

    /// Returns true if any buffer needs uploading
    pub fn is_dirty(&self) -> bool {
        self.vertices_dirty || self.indices_dirty || self.instances_dirty
    }

    /// Returns the raw instance slice for merged upload
    pub fn instances(&self) -> &[Instance] {
        &self.instances
    }

    /// Number of instances queued for instanced drawing
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Number of vertices queued for baked geometry
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of indices queued for baked geometry
    pub fn index_count(&self) -> usize {
        self.indices.len()
    }

    /// Clears CPU-side geometry and instances, keeps buffer allocations for reuse.
    /// Only marks non-empty buffers dirty to avoid redundant GPU uploads.
    pub fn clear(&mut self) {
        if !self.vertices.is_empty() {
            self.vertices.clear();
            self.vertices_dirty = true;
        }
        if !self.indices.is_empty() {
            self.indices.clear();
            self.indices_dirty = true;
        }
        if !self.instances.is_empty() {
            self.instances.clear();
            self.instances_dirty = true;
        }
    }

    // Uploads buffers to GPU only if needed
    pub fn upload(&mut self, device: &Device, queue: &Queue) {
        self.upload_geometry_only(device, queue);

        if self.instances_dirty && !self.instances.is_empty() {
            let required_bytes = (self.instances.len() * std::mem::size_of::<Instance>()) as u64;
            let needs_recreate = self.instance_buffer.as_ref().is_none_or(|b| b.size() < required_bytes);
            if needs_recreate {
                let alloc = required_bytes
                    .next_power_of_two()
                    .max((Self::INITIAL_INSTANCE_CAPACITY * std::mem::size_of::<Instance>()) as u64);
                self.instance_buffer = Some(device.create_buffer(&BufferDescriptor {
                    label: Some("GeometryBatch Instance Buffer"),
                    size: alloc,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            queue.write_buffer(self.instance_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&self.instances));
            self.instances_dirty = false;
        }
    }

    /// Upload only vertex/index data, skip instances (used when instances go through shared buffer)
    pub fn upload_geometry_only(&mut self, device: &Device, queue: &Queue) {
        if self.vertices_dirty && !self.vertices.is_empty() {
            if self.vertex_buffer.is_none() {
                self.vertex_buffer = Some(device.create_buffer(&BufferDescriptor {
                    label: Some("GeometryBatch Vertex Buffer"),
                    size: (self.max_verticies * std::mem::size_of::<Vertex>()) as u64,
                    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
            queue.write_buffer(self.vertex_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&self.vertices));
            self.vertices_dirty = false;
        }

        if self.indices_dirty && !self.indices.is_empty() {
            if self.index_buffer.is_none() {
                self.index_buffer = Some(device.create_buffer(&BufferDescriptor {
                    label: Some("GeometryBatch Index Buffer"),
                    size: (self.max_indices * std::mem::size_of::<u16>()) as u64,
                    usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }

            let byte_len = self.indices.len() * std::mem::size_of::<u16>();
            let needs_padding = !byte_len.is_multiple_of(COPY_BUFFER_ALIGNMENT as usize);
            if needs_padding {
                self.indices.push(0);
            }
            queue.write_buffer(self.index_buffer.as_ref().unwrap(), 0, bytemuck::cast_slice(&self.indices));
            if needs_padding {
                self.indices.pop();
            }
            self.indices_dirty = false;
        }
    }

    /// Draws baked geometry and/or instanced quads as separate draw calls.
    /// `quad_bound` tracks whether the static quad VB/IB are currently bound
    /// to avoid redundant `set_vertex_buffer` / `set_index_buffer` calls.
    pub(crate) fn draw<'a>(
        &self,
        r_pass: &mut RenderPass<'a>,
        quad_vb: &'a Buffer,
        quad_ib: &'a Buffer,
        dummy_instance: &'a Buffer,
        quad_bound: &mut bool,
        external_instances: Option<(&'a Buffer, u64)>,
    ) {
        if !self.instances.is_empty() {
            if !*quad_bound {
                r_pass.set_vertex_buffer(0, quad_vb.slice(..));
                r_pass.set_index_buffer(quad_ib.slice(..), IndexFormat::Uint16);
            }
            if let Some((buf, byte_offset)) = external_instances {
                r_pass.set_vertex_buffer(1, buf.slice(byte_offset..));
            } else if let Some(instance_buf) = &self.instance_buffer {
                r_pass.set_vertex_buffer(1, instance_buf.slice(..));
            }
            r_pass.draw_indexed(0..6, 0, 0..self.instances.len() as u32);
            *quad_bound = true;
        }
        if !self.indices.is_empty()
            && let (Some(vb), Some(ib)) = (&self.vertex_buffer, &self.index_buffer)
        {
            r_pass.set_vertex_buffer(0, vb.slice(..));
            r_pass.set_vertex_buffer(1, dummy_instance.slice(..));
            r_pass.set_index_buffer(ib.slice(..), IndexFormat::Uint16);
            r_pass.draw_indexed(0..self.indices.len() as u32, 0, 0..1);
            *quad_bound = false;
        }
    }
}
