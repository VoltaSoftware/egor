use wgpu::{
    Adapter, CommandEncoder, CurrentSurfaceTexture, Device, DownlevelFlags, Extent3d, Instance, PresentMode, Surface, SurfaceConfiguration,
    SurfaceTarget, Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
    WindowHandle,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

use crate::frame::Presentable;

/// Trait for render targets (backbuffers, offscreen textures, etc.)
pub trait RenderTarget {
    fn format(&self) -> TextureFormat;
    fn size(&self) -> (u32, u32);
    /// Returns the view and optionally something that must be presented (swapchain)
    fn acquire(&mut self, device: &Device) -> Option<(TextureView, Option<Presentable>)>;
    fn resize(&mut self, device: &Device, w: u32, h: u32);
    /// Only useful for backbuffer targets
    fn set_vsync(&mut self, _device: &Device, _on: bool) {}
}

pub(crate) fn surface_config(surface: &Surface<'_>, adapter: &Adapter, w: u32, h: u32) -> (SurfaceConfiguration, TextureFormat, bool) {
    let caps = surface.get_capabilities(adapter);
    let mut config = surface.get_default_config(adapter, w, h).unwrap();
    config.present_mode = PresentMode::AutoVsync;

    let surface_copy_src = caps.usages.contains(TextureUsages::COPY_SRC);
    if surface_copy_src {
        config.usage |= TextureUsages::COPY_SRC;
    }

    // Prefer sRGB rendering, but do it in the form each backend can present:
    //
    // * WebGPU wants a non-sRGB canvas format plus an sRGB surface view.
    // * GLES/WebGL and Android Vulkan often do not support surface view
    //   formats, so use the sRGB surface format directly when it is exposed.
    // * If neither route exists, fall back to the surface format itself.
    let srgb_format = config.format.add_srgb_suffix();
    let can_surface_view_format = adapter
        .get_downlevel_capabilities()
        .flags
        .contains(DownlevelFlags::SURFACE_VIEW_FORMATS);
    let view_format = if srgb_format == config.format {
        config.format
    } else if !can_surface_view_format && caps.formats.contains(&srgb_format) {
        config.format = srgb_format;
        srgb_format
    } else if can_surface_view_format {
        config.view_formats.push(srgb_format);
        srgb_format
    } else {
        config.format
    };

    (config, view_format, surface_copy_src)
}

/// Renders to the window's backbuffer (swapchain)
pub struct Backbuffer {
    surface: Surface<'static>,
    config: SurfaceConfiguration,
    /// The sRGB variant of the surface format, used for pipeline targets and
    /// texture views so that linear framebuffer values get gamma-encoded on
    /// output. On native this usually equals `config.format`; on WebGPU the
    /// canvas context only accepts the non-sRGB format but we create sRGB views
    /// via `view_formats`.
    view_format: TextureFormat,
    surface_copy_src: bool,
}

impl Backbuffer {
    pub fn new(
        instance: &Instance,
        adapter: &Adapter,
        device: &Device,
        window: impl Into<SurfaceTarget<'static>> + WindowHandle,
        w: u32,
        h: u32,
    ) -> Self {
        let surface = instance.create_surface(window).unwrap();
        let (config, view_format, surface_copy_src) = surface_config(&surface, adapter, w, h);
        surface.configure(device, &config);
        Self {
            surface,
            config,
            view_format,
            surface_copy_src,
        }
    }

    pub fn supports_copy_src(&self) -> bool {
        self.surface_copy_src
    }
}

impl RenderTarget for Backbuffer {
    fn format(&self) -> TextureFormat {
        self.view_format
    }

    fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    fn acquire(&mut self, device: &Device) -> Option<(TextureView, Option<Presentable>)> {
        // On WASM the canvas can be resized externally (e.g. via a JS
        // ResizeObserver) without going through winit's resize event.
        // Detect the mismatch and reconfigure the surface before acquiring.
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(canvas) = Self::get_canvas_size() {
                let (cw, ch) = canvas;
                if cw > 0 && ch > 0 && (cw != self.config.width || ch != self.config.height) {
                    self.resize(device, cw, ch);
                }
            }
        }

        match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(surface_texture) | CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                let view = surface_texture.texture.create_view(&TextureViewDescriptor {
                    format: Some(self.view_format),
                    ..Default::default()
                });
                Some((view, Some(Presentable::Surface(surface_texture))))
            }
            CurrentSurfaceTexture::Outdated => {
                self.resize(device, self.config.width, self.config.height);
                None
            }
            other => {
                //eprintln!("Surface error: {:?}", other);
                None
            }
        }
    }

    fn resize(&mut self, device: &Device, w: u32, h: u32) {
        (self.config.width, self.config.height) = (w, h);
        self.surface.configure(device, &self.config);
    }

    fn set_vsync(&mut self, device: &Device, on: bool) {
        self.config.present_mode = if on { PresentMode::Fifo } else { PresentMode::AutoNoVsync };
        self.surface.configure(device, &self.config);
    }
}

impl Backbuffer {
    /// Read the canvas element's physical pixel dimensions directly from the DOM.
    #[cfg(target_arch = "wasm32")]
    fn get_canvas_size() -> Option<(u32, u32)> {
        let document = web_sys::window()?.document()?;
        let canvas = document.query_selector("canvas").ok()??;
        let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into().ok()?;
        let w = canvas.width();
        let h = canvas.height();
        Some((w, h))
    }
}

/// Renders to an offscreen texture that can be read back or used as a texture
pub struct OffscreenTarget {
    render_texture: Texture,
    render_view: TextureView,
    sample_texture: Texture,
    sample_view: TextureView,
    depth_texture: Texture,
    depth_view: TextureView,
    format: TextureFormat,
    width: u32,
    height: u32,
}

impl OffscreenTarget {
    pub fn new(device: &Device, width: u32, height: u32, format: TextureFormat) -> Self {
        let render_texture = device.create_texture(&TextureDescriptor {
            label: Some("Offscreen Render Texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let sample_texture = device.create_texture(&TextureDescriptor {
            label: Some("Offscreen Sample Texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let (depth_texture, depth_view) = crate::Renderer::create_depth_texture(device, width, height);

        let render_view = render_texture.create_view(&Default::default());
        let sample_view = sample_texture.create_view(&Default::default());

        Self {
            render_texture,
            render_view,
            sample_texture,
            sample_view,
            depth_texture,
            depth_view,
            format,
            width,
            height,
        }
    }

    pub fn texture(&self) -> &Texture {
        &self.sample_texture
    }

    pub fn view(&self) -> &TextureView {
        &self.sample_view
    }

    pub fn render_view(&self) -> &TextureView {
        &self.render_view
    }

    pub fn offscreen_depth_view(&self) -> &TextureView {
        &self.depth_view
    }

    /// Copy render texture into sample texture so it can be sampled
    pub fn copy_to_sample(&self, encoder: &mut CommandEncoder) {
        encoder.copy_texture_to_texture(
            self.render_texture.as_image_copy(),
            self.sample_texture.as_image_copy(),
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }
}

impl RenderTarget for OffscreenTarget {
    fn format(&self) -> TextureFormat {
        self.format
    }

    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn acquire(&mut self, _: &Device) -> Option<(TextureView, Option<Presentable>)> {
        // no presentation needed for offscreen targets
        Some((self.render_view.clone(), None))
    }

    fn resize(&mut self, device: &Device, w: u32, h: u32) {
        if self.width == w && self.height == h {
            return;
        }
        // recreate the texture with new dimensions
        *self = Self::new(device, w, h, self.format);
    }
}
