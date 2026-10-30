use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource, Device,
    Extent3d, FilterMode, Origin3d, Queue, RenderPass, Sampler, SamplerDescriptor,
    TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
    TextureViewDimension,
};

use crate::target::OffscreenTarget;

const INVALID_TEXTURE_FALLBACK_RGBA: [u8; 4] = [255, 0, 255, 255];

fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .copied()
        .map(str::to_owned)
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

/// A GPU texture that can be bound in shaders for rendering
///
/// Wraps a `wgpu::Texture`, its view, sampler, & bind group
pub(crate) struct Texture {
    bind_group: BindGroup,
    view: TextureView,
    linear_variant: Option<usize>,
    /// The texture this is the linear variant of.
    variant_of: Option<usize>,
    is_array: bool,
}

impl Texture {
    fn create_bind_group(
        device: &Device,
        layout: &BindGroupLayout,
        view: &TextureView,
        sampler: &Sampler,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    /// Creates a new texture from raw RGBA image data,
    /// uploads the data, & builds the bind group using the layout and shared sampler
    ///
    /// - `data`: Must be in tightly packed 8-bit RGBA format
    /// - `width`, `height`: Dimensions of the image in pixels
    fn from_bytes(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        sampler: &Sampler,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Self {
        if let Err(reason) = Self::validate_upload(device, data, width, height) {
            log::warn!("[egor] invalid texture upload ({reason}); using 1x1 fallback");
            return Self::from_trusted_bytes(
                device,
                queue,
                layout,
                sampler,
                &INVALID_TEXTURE_FALLBACK_RGBA,
                1,
                1,
                Some("Egor Invalid Texture Fallback"),
            );
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let texture_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                Self::from_trusted_bytes(
                    device,
                    queue,
                    layout,
                    sampler,
                    data,
                    width,
                    height,
                    Some("Egor Texture"),
                )
            }));

            let texture = match texture_result {
                Ok(texture) => texture,
                Err(payload) => {
                    drop(error_scope);
                    log::error!(
                        "[egor] texture upload panicked ({}); using 1x1 fallback",
                        panic_payload_message(payload)
                    );
                    return Self::from_trusted_bytes(
                        device,
                        queue,
                        layout,
                        sampler,
                        &INVALID_TEXTURE_FALLBACK_RGBA,
                        1,
                        1,
                        Some("Egor Invalid Texture Fallback"),
                    );
                }
            };

            if let Some(error) = pollster::block_on(error_scope.pop()) {
                log::error!("[egor] texture upload validation failed: {error}; using 1x1 fallback");
                return Self::from_trusted_bytes(
                    device,
                    queue,
                    layout,
                    sampler,
                    &INVALID_TEXTURE_FALLBACK_RGBA,
                    1,
                    1,
                    Some("Egor Invalid Texture Fallback"),
                );
            }

            texture
        }

        #[cfg(target_arch = "wasm32")]
        {
            Self::from_trusted_bytes(
                device,
                queue,
                layout,
                sampler,
                data,
                width,
                height,
                Some("Egor Texture"),
            )
        }
    }

    fn validate_size(device: &Device, width: u32, height: u32) -> Result<(), String> {
        if width == 0 || height == 0 {
            return Err(format!("zero-sized texture {width}x{height}"));
        }

        let max_dimension = device.limits().max_texture_dimension_2d;
        if width > max_dimension || height > max_dimension {
            return Err(format!(
                "texture {width}x{height} exceeds max dimension {max_dimension}"
            ));
        }
        Ok(())
    }

    fn validate_upload(
        device: &Device,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        Self::validate_size(device, width, height)?;

        let expected_len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| format!("texture size overflow for {width}x{height}"))?;

        if data.len() < expected_len {
            return Err(format!(
                "RGBA buffer too small for {width}x{height}: got {} bytes, need {expected_len}",
                data.len()
            ));
        }

        Ok(())
    }

    fn from_trusted_bytes(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        sampler: &Sampler,
        data: &[u8],
        width: u32,
        height: u32,
        label: Option<&str>,
    ) -> Self {
        Self::from_array_bytes(
            device, queue, layout, sampler, data, width, height, 1, 0, label,
        )
    }

    /// Uploads `layers` layers from `data`; `spare_layers` more follow them and
    /// start transparent.
    #[allow(clippy::too_many_arguments)]
    fn from_array_bytes(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        sampler: &Sampler,
        data: &[u8],
        width: u32,
        height: u32,
        layers: u32,
        spare_layers: u32,
        label: Option<&str>,
    ) -> Self {
        let total_layers = layers + spare_layers;
        let texture = device.create_texture(&TextureDescriptor {
            label,
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: Self::storage_layer_count(
                    device,
                    width,
                    height,
                    total_layers,
                ),
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        if layers > 0 {
            queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                data,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * width),
                    rows_per_image: Some(height),
                },
                Extent3d {
                    width,
                    height,
                    depth_or_array_layers: layers,
                },
            );
        }
        // GLES cannot reinterpret ordinary 2D storage as array storage. Create
        // only the view that matches the texture's actual storage target; the
        // pipeline variant uses the corresponding bind-group layout.
        let view_dimension = if total_layers > 1 {
            TextureViewDimension::D2Array
        } else {
            TextureViewDimension::D2
        };
        let view = texture.create_view(&TextureViewDescriptor {
            dimension: Some(view_dimension),
            array_layer_count: Some(total_layers),
            ..Default::default()
        });

        Self {
            bind_group: Self::create_bind_group(device, layout, &view, sampler),
            view,
            linear_variant: None,
            variant_of: None,
            is_array: total_layers > 1,
        }
    }

    fn storage_layer_count(device: &Device, width: u32, height: u32, layers: u32) -> u32 {
        // wgpu's GLES backend infers cube storage for square textures with a
        // multiple of six layers. Reserve an unused layer to keep array storage
        // on desktop GL, Android GLES and WebGL2. Uploads and views expose only
        // the requested layers; other backends keep the original allocation.
        let needs_padding = layers > 1
            && layers.is_multiple_of(6)
            && width == height
            && device.adapter_info().backend == wgpu::Backend::Gl;
        layers + u32::from(needs_padding)
    }

    /// Creates a bindable texture from an existing GPU texture view.
    ///
    /// This does not allocate or upload image data.
    /// It wraps a view produced elsewhere (an offscreen render target)
    /// and builds the bind group required for sampling in shaders
    fn from_view(
        texture: &wgpu::Texture,
        device: &Device,
        layout: &BindGroupLayout,
        sampler: &Sampler,
    ) -> Self {
        let view = texture.create_view(&Default::default());
        Self {
            bind_group: Self::create_bind_group(device, layout, &view, sampler),
            view,
            linear_variant: None,
            variant_of: None,
            is_array: false,
        }
    }

    /// Creates a 1×1 white fallback texture
    ///
    /// Used when no valid texture is provided for a draw call
    fn create_default(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        sampler: &Sampler,
    ) -> Self {
        Self::from_bytes(
            device,
            queue,
            layout,
            sampler,
            &[255u8, 255, 255, 255],
            1,
            1,
        )
    }

    /// Binds this texture at the given index in the render pass
    ///
    /// - `index` must match the bind group index used in the pipeline layout
    pub fn is_array(&self) -> bool {
        self.is_array
    }

    pub fn bind(&self, pass: &mut RenderPass, index: u32) {
        pass.set_bind_group(index, &self.bind_group, &[]);
    }
}

pub(crate) struct Textures {
    layouts: [BindGroupLayout; 2],
    default_sampler: Sampler,
    nearest_sampler: Sampler,
    default_texture: Texture,
    store: Vec<Option<Texture>>,
    /// Slots of removed textures, which the next inserts reuse.
    free_slots: Vec<usize>,
}

impl Textures {
    pub fn new(device: &Device, queue: &Queue) -> Self {
        let layouts = std::array::from_fn(|kind| {
            crate::pipeline::create_texture_bind_group_layout(device, kind == 1)
        });

        let default_sampler = device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        let nearest_sampler = device.create_sampler(&SamplerDescriptor {
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            ..Default::default()
        });

        let default_texture = Texture::create_default(device, queue, &layouts[0], &default_sampler);

        Self {
            layouts,
            default_sampler,
            nearest_sampler,
            default_texture,
            store: Vec::new(),
            free_slots: Vec::new(),
        }
    }

    fn push(&mut self, texture: Texture) -> usize {
        match self.free_slots.pop() {
            Some(id) => {
                self.store[id] = Some(texture);
                id
            }
            None => {
                self.store.push(Some(texture));
                self.store.len() - 1
            }
        }
    }

    /// Frees texture `id` and its linear variant; their slots go to the next inserts. Removing a linear variant
    /// alone leaves its source, which makes a new variant when next asked.
    pub fn remove(&mut self, id: usize) {
        let Some(texture) = self.store.get_mut(id).and_then(Option::take) else {
            return;
        };
        self.free_slots.push(id);
        match texture.variant_of {
            Some(source) => {
                if let Some(Some(source)) = self.store.get_mut(source) {
                    source.linear_variant = None;
                }
            }
            None => {
                if let Some(variant) = texture.linear_variant {
                    self.remove(variant);
                }
            }
        }
    }

    /// Cache another binding to the same GPU image, without changing the original
    /// sampler or duplicating pixel storage. A different ID also splits draw batches.
    pub fn linear_variant(&mut self, device: &Device, id: usize) -> usize {
        let Some(source) = self.store.get(id).and_then(Option::as_ref) else {
            return id;
        };
        if let Some(variant) = source.linear_variant {
            return variant;
        }
        let variant = self.linear_binding(device, source, id);
        let variant_id = self.push(variant);
        let variant = self.store[variant_id].as_mut().unwrap();
        variant.linear_variant = Some(variant_id);
        self.store[id].as_mut().unwrap().linear_variant = Some(variant_id);
        variant_id
    }

    /// A linear-sampled binding of `source` (texture `source_id`)'s image.
    fn linear_binding(&self, device: &Device, source: &Texture, source_id: usize) -> Texture {
        Texture {
            bind_group: Texture::create_bind_group(
                device,
                &self.layouts[usize::from(source.is_array)],
                &source.view,
                &self.default_sampler,
            ),
            view: source.view.clone(),
            linear_variant: None,
            variant_of: Some(source_id),
            is_array: source.is_array,
        }
    }

    /// Puts `texture` in slot `id`, rebinding the slot's linear variant to its image so the
    /// old image isn't kept alive.
    fn replace_slot(&mut self, device: &Device, id: usize, mut texture: Texture) {
        let Some(Some(old)) = self.store.get(id) else {
            log::error!("[egor] can't replace texture {id}: it doesn't exist");
            return;
        };
        if let Some(variant) = old.linear_variant {
            let mut binding = self.linear_binding(device, &texture, id);
            binding.linear_variant = Some(variant);
            self.store[variant] = Some(binding);
            texture.linear_variant = Some(variant);
        }
        self.store[id] = Some(texture);
    }

    fn decode_rgba(data: &[u8]) -> (u32, u32, image::RgbaImage) {
        let img = image::load_from_memory(data).unwrap().to_rgba8();
        let (w, h) = img.dimensions();
        (w, h, img)
    }

    pub fn get(&self, id: Option<usize>) -> &Texture {
        id.and_then(|i| self.store.get(i))
            .and_then(Option::as_ref)
            .unwrap_or(&self.default_texture)
    }

    pub fn insert(&mut self, device: &Device, queue: &Queue, data: &[u8]) -> usize {
        let (w, h, img) = Self::decode_rgba(data);
        self.insert_raw(device, queue, w, h, &img)
    }

    pub fn insert_raw(
        &mut self,
        device: &Device,
        queue: &Queue,
        w: u32,
        h: u32,
        data: &[u8],
    ) -> usize {
        let texture = Texture::from_bytes(
            device,
            queue,
            &self.layouts[0],
            &self.default_sampler,
            data,
            w,
            h,
        );
        self.push(texture)
    }

    pub fn insert_raw_nearest(
        &mut self,
        device: &Device,
        queue: &Queue,
        w: u32,
        h: u32,
        data: &[u8],
    ) -> usize {
        let texture = Texture::from_bytes(
            device,
            queue,
            &self.layouts[0],
            &self.nearest_sampler,
            data,
            w,
            h,
        );
        self.push(texture)
    }

    pub fn insert_array_raw(
        &mut self,
        device: &Device,
        queue: &Queue,
        w: u32,
        h: u32,
        layers: u32,
        data: &[u8],
    ) -> Result<usize, String> {
        self.insert_array_raw_with_spare_layers(device, queue, w, h, layers, 0, data)
    }

    /// Like [`Self::insert_array_raw`], followed by `spare_layers` transparent layers.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_array_raw_with_spare_layers(
        &mut self,
        device: &Device,
        queue: &Queue,
        w: u32,
        h: u32,
        layers: u32,
        spare_layers: u32,
        data: &[u8],
    ) -> Result<usize, String> {
        Texture::validate_size(device, w, h)?;
        let total_layers = layers
            .checked_add(spare_layers)
            .ok_or("Texture array layer count overflow")?;
        let storage_layers = Texture::storage_layer_count(device, w, h, total_layers);
        let max_layers = device.limits().max_texture_array_layers;
        if total_layers == 0 || storage_layers > max_layers {
            return Err(format!(
                "Texture array with {total_layers} layers requires {storage_layers} storage layers; device limit is {max_layers}"
            ));
        }
        let expected = (w as usize)
            .checked_mul(h as usize)
            .and_then(|n| n.checked_mul(layers as usize))
            .and_then(|n| n.checked_mul(4))
            .ok_or("Texture array size overflow")?;
        if data.len() != expected {
            return Err(format!(
                "Texture array needs {expected} bytes, got {}",
                data.len()
            ));
        }
        let texture = Texture::from_array_bytes(
            device,
            queue,
            &self.layouts[usize::from(total_layers > 1)],
            &self.nearest_sampler,
            data,
            w,
            h,
            layers,
            spare_layers,
            Some("Egor Texture Array"),
        );
        Ok(self.push(texture))
    }

    /// Overwrites a `w`×`h` RGBA8 region of one layer of texture `id`.
    #[allow(clippy::too_many_arguments)]
    pub fn write_region(
        &self,
        queue: &Queue,
        id: usize,
        layer: u32,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        data: &[u8],
    ) -> Result<(), String> {
        let texture = self
            .store
            .get(id)
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("no texture {id}"))?
            .view
            .texture();
        let size = texture.size();
        let fits = x.checked_add(w).is_some_and(|right| right <= size.width)
            && y.checked_add(h).is_some_and(|bottom| bottom <= size.height)
            && layer < size.depth_or_array_layers;
        if !fits {
            return Err(format!(
                "region {w}x{h} at ({x}, {y}) layer {layer} is outside texture {id} ({}x{}x{})",
                size.width, size.height, size.depth_or_array_layers
            ));
        }
        if data.len() != w as usize * h as usize * 4 {
            return Err(format!(
                "region {w}x{h} needs {} bytes, got {}",
                w as usize * h as usize * 4,
                data.len()
            ));
        }
        if w == 0 || h == 0 {
            return Ok(());
        }
        queue.write_texture(
            TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: Origin3d { x, y, z: layer },
                aspect: TextureAspect::All,
            },
            data,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    pub fn replace(&mut self, device: &Device, queue: &Queue, id: usize, data: &[u8]) {
        let (w, h, img) = Self::decode_rgba(data);
        self.replace_raw(device, queue, id, w, h, &img);
    }

    pub fn replace_raw(
        &mut self,
        device: &Device,
        queue: &Queue,
        id: usize,
        w: u32,
        h: u32,
        data: &[u8],
    ) {
        let texture = Texture::from_bytes(
            device,
            queue,
            &self.layouts[0],
            &self.default_sampler,
            data,
            w,
            h,
        );
        self.replace_slot(device, id, texture);
    }

    pub fn insert_nearest(&mut self, device: &Device, queue: &Queue, data: &[u8]) -> usize {
        let (w, h, img) = Self::decode_rgba(data);
        let texture = Texture::from_bytes(
            device,
            queue,
            &self.layouts[0],
            &self.nearest_sampler,
            &img,
            w,
            h,
        );
        self.push(texture)
    }

    pub fn insert_offscreen(
        &mut self,
        device: &Device,
        offscreen: &OffscreenTarget,
        filter: FilterMode,
    ) -> usize {
        let texture = self.offscreen_texture(device, offscreen, filter);
        self.push(texture)
    }

    /// Rebinds texture `id` to `offscreen`'s image after the target was resized. The id stays valid, and the
    /// old image is freed instead of kept alive by a binding nothing draws with. A linear variant follows it.
    pub fn replace_offscreen(
        &mut self,
        device: &Device,
        id: usize,
        offscreen: &OffscreenTarget,
        filter: FilterMode,
    ) {
        let texture = self.offscreen_texture(device, offscreen, filter);
        self.replace_slot(device, id, texture);
    }

    fn offscreen_texture(
        &self,
        device: &Device,
        offscreen: &OffscreenTarget,
        filter: FilterMode,
    ) -> Texture {
        Texture::from_view(
            offscreen.texture(),
            device,
            &self.layouts[0],
            match filter {
                FilterMode::Nearest => &self.nearest_sampler,
                FilterMode::Linear => &self.default_sampler,
            },
        )
    }
}

#[cfg(test)]
#[path = "texture_array_tests.rs"]
mod texture_array_tests;
