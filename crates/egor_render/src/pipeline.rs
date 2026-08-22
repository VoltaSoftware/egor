use std::borrow::Cow;

use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BlendState, BufferBindingType, ColorTargetState,
    ColorWrites, CompareFunction, DepthStencilState, Device, FragmentState, PipelineLayoutDescriptor, RenderPipeline,
    RenderPipelineDescriptor, SamplerBindingType, ShaderModuleDescriptor, ShaderSource, ShaderStages, StencilState, TextureFormat,
    TextureSampleType, TextureViewDimension, VertexState, include_wgsl,
};

use crate::{instance::Instance, vertex::Vertex};

const WATCH_OVERLAY_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

const SRGB_OUTPUT_HELPERS: &str = r#"
fn egor_linear_to_srgb_channel(value: f32) -> f32 {
    let clamped = clamp(value, 0.0, 1.0);
    if clamped <= 0.0031308 {
        return clamped * 12.92;
    }
    return 1.055 * pow(clamped, 1.0 / 2.4) - 0.055;
}

fn egor_linear_to_srgb(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        egor_linear_to_srgb_channel(color.r),
        egor_linear_to_srgb_channel(color.g),
        egor_linear_to_srgb_channel(color.b),
        color.a,
    );
}
"#;

fn surface_needs_srgb_encode(surface_format: TextureFormat) -> bool {
    surface_format.add_srgb_suffix() != surface_format
}

fn skip_ascii_whitespace(source: &str, mut index: usize) -> usize {
    while let Some(byte) = source.as_bytes().get(index) {
        if !byte.is_ascii_whitespace() {
            break;
        }
        index += 1;
    }
    index
}

fn parse_fragment_argument_names(params: &str) -> Option<String> {
    let mut names = Vec::new();
    for param in params.split(',').map(str::trim).filter(|param| !param.is_empty()) {
        let (name, _) = param.split_once(':')?;
        names.push(name.trim());
    }
    Some(names.join(", "))
}

fn wrap_custom_shader_for_srgb_output(wgsl_source: &str) -> Cow<'_, str> {
    const FRAGMENT_ATTR: &str = "@fragment";
    const USER_FN: &str = "fn fs_main";
    const RETURN_LOCATION: &str = "@location(0)";
    const RETURN_TYPE: &str = "vec4<f32>";

    let Some(fn_pos) = wgsl_source.find(USER_FN) else {
        log::warn!("[egor] custom shader has no fs_main entrypoint; sRGB output encoding was not applied");
        return Cow::Borrowed(wgsl_source);
    };
    let Some(attr_pos) = wgsl_source[..fn_pos].rfind(FRAGMENT_ATTR) else {
        log::warn!("[egor] custom shader fs_main has no @fragment attribute; sRGB output encoding was not applied");
        return Cow::Borrowed(wgsl_source);
    };
    if !wgsl_source[attr_pos + FRAGMENT_ATTR.len()..fn_pos].trim().is_empty() {
        log::warn!("[egor] custom shader @fragment attribute could not be matched to fs_main; sRGB output encoding was not applied");
        return Cow::Borrowed(wgsl_source);
    }

    let Some(params_start) = wgsl_source[fn_pos..].find('(').map(|offset| fn_pos + offset) else {
        log::warn!("[egor] custom shader fs_main params could not be parsed; sRGB output encoding was not applied");
        return Cow::Borrowed(wgsl_source);
    };
    let Some(params_end) = wgsl_source[params_start..].find(')').map(|offset| params_start + offset) else {
        log::warn!("[egor] custom shader fs_main params could not be parsed; sRGB output encoding was not applied");
        return Cow::Borrowed(wgsl_source);
    };
    let params = &wgsl_source[params_start + 1..params_end];
    let Some(argument_names) = parse_fragment_argument_names(params) else {
        log::warn!("[egor] custom shader fs_main arguments could not be parsed; sRGB output encoding was not applied");
        return Cow::Borrowed(wgsl_source);
    };

    let mut cursor = skip_ascii_whitespace(wgsl_source, params_end + 1);
    if !wgsl_source[cursor..].starts_with("->") {
        log::warn!("[egor] custom shader fs_main return type could not be parsed; sRGB output encoding was not applied");
        return Cow::Borrowed(wgsl_source);
    }
    cursor = skip_ascii_whitespace(wgsl_source, cursor + 2);
    if !wgsl_source[cursor..].starts_with(RETURN_LOCATION) {
        log::warn!("[egor] custom shader fs_main return location could not be parsed; sRGB output encoding was not applied");
        return Cow::Borrowed(wgsl_source);
    }
    cursor = skip_ascii_whitespace(wgsl_source, cursor + RETURN_LOCATION.len());
    if !wgsl_source[cursor..].starts_with(RETURN_TYPE) {
        log::warn!("[egor] custom shader fs_main return type is not vec4<f32>; sRGB output encoding was not applied");
        return Cow::Borrowed(wgsl_source);
    }
    let return_type_end = cursor + RETURN_TYPE.len();

    let mut wrapped = String::with_capacity(wgsl_source.len() + SRGB_OUTPUT_HELPERS.len() + 180);
    wrapped.push_str(&wgsl_source[..attr_pos]);
    wrapped.push_str("fn egor_user_fs_main");
    wrapped.push_str(&wgsl_source[params_start..params_end + 1]);
    wrapped.push_str(" -> vec4<f32>");
    wrapped.push_str(&wgsl_source[return_type_end..]);
    wrapped.push_str(SRGB_OUTPUT_HELPERS);
    wrapped.push_str("\n@fragment\nfn fs_main(");
    wrapped.push_str(params);
    wrapped.push_str(") -> @location(0) vec4<f32> {\n    return egor_linear_to_srgb(egor_user_fs_main(");
    wrapped.push_str(&argument_names);
    wrapped.push_str("));\n}\n");

    Cow::Owned(wrapped)
}

fn first_fragment_argument_name(params: &str) -> Option<String> {
    let first = params.split(',').map(str::trim).find(|param| !param.is_empty())?;
    let (name, _) = first.split_once(':')?;
    Some(name.trim().to_owned())
}

fn wrap_custom_shader_for_watch_output(wgsl_source: &str, encode_srgb: bool) -> Option<Cow<'_, str>> {
    const FRAGMENT_ATTR: &str = "@fragment";
    const USER_FN: &str = "fn fs_main";
    const RETURN_LOCATION: &str = "@location(0)";
    const RETURN_TYPE: &str = "vec4<f32>";

    let Some(fn_pos) = wgsl_source.find(USER_FN) else {
        log::warn!("[egor] custom shader has no fs_main entrypoint; watch overlay output was not applied");
        return None;
    };
    let Some(attr_pos) = wgsl_source[..fn_pos].rfind(FRAGMENT_ATTR) else {
        log::warn!("[egor] custom shader fs_main has no @fragment attribute; watch overlay output was not applied");
        return None;
    };
    if !wgsl_source[attr_pos + FRAGMENT_ATTR.len()..fn_pos].trim().is_empty() {
        log::warn!("[egor] custom shader @fragment attribute could not be matched to fs_main; watch overlay output was not applied");
        return None;
    }

    let Some(params_start) = wgsl_source[fn_pos..].find('(').map(|offset| fn_pos + offset) else {
        log::warn!("[egor] custom shader fs_main params could not be parsed; watch overlay output was not applied");
        return None;
    };
    let Some(params_end) = wgsl_source[params_start..].find(')').map(|offset| params_start + offset) else {
        log::warn!("[egor] custom shader fs_main params could not be parsed; watch overlay output was not applied");
        return None;
    };
    let params = &wgsl_source[params_start + 1..params_end];
    let Some(argument_names) = parse_fragment_argument_names(params) else {
        log::warn!("[egor] custom shader fs_main arguments could not be parsed; watch overlay output was not applied");
        return None;
    };
    let Some(first_argument_name) = first_fragment_argument_name(params) else {
        log::warn!("[egor] custom shader fs_main first argument could not be parsed; watch overlay output was not applied");
        return None;
    };
    if !wgsl_source.contains("watch_overlay") {
        log::warn!("[egor] custom shader has no watch_overlay fragment input; watch overlay output was not applied");
        return None;
    };

    let mut cursor = skip_ascii_whitespace(wgsl_source, params_end + 1);
    if !wgsl_source[cursor..].starts_with("->") {
        log::warn!("[egor] custom shader fs_main return type could not be parsed; watch overlay output was not applied");
        return None;
    }
    cursor = skip_ascii_whitespace(wgsl_source, cursor + 2);
    if !wgsl_source[cursor..].starts_with(RETURN_LOCATION) {
        log::warn!("[egor] custom shader fs_main return location could not be parsed; watch overlay output was not applied");
        return None;
    }
    cursor = skip_ascii_whitespace(wgsl_source, cursor + RETURN_LOCATION.len());
    if !wgsl_source[cursor..].starts_with(RETURN_TYPE) {
        log::warn!("[egor] custom shader fs_main return type is not vec4<f32>; watch overlay output was not applied");
        return None;
    }
    let return_type_end = cursor + RETURN_TYPE.len();

    let mut wrapped = String::with_capacity(wgsl_source.len() + SRGB_OUTPUT_HELPERS.len() + 360);
    wrapped.push_str(&wgsl_source[..attr_pos]);
    wrapped.push_str("fn egor_user_fs_main");
    wrapped.push_str(&wgsl_source[params_start..params_end + 1]);
    wrapped.push_str(" -> vec4<f32>");
    wrapped.push_str(&wgsl_source[return_type_end..]);
    if encode_srgb {
        wrapped.push_str(SRGB_OUTPUT_HELPERS);
    }
    wrapped.push_str(
        r#"
struct EgorWatchFragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) overlay: vec4<f32>,
};

@fragment
fn fs_main("#,
    );
    wrapped.push_str(params);
    wrapped.push_str(") -> EgorWatchFragmentOutput {\n    let user_color = egor_user_fs_main(");
    wrapped.push_str(&argument_names);
    wrapped.push_str(");\n    var out: EgorWatchFragmentOutput;\n");
    if encode_srgb {
        wrapped.push_str("    out.color = egor_linear_to_srgb(user_color);\n");
        wrapped.push_str("    out.overlay = vec4<f32>(out.color.rgb, user_color.a * ");
        wrapped.push_str(&first_argument_name);
        wrapped.push_str(".watch_overlay);\n");
    } else {
        wrapped.push_str("    out.color = user_color;\n");
        wrapped.push_str("    out.overlay = vec4<f32>(user_color.rgb, user_color.a * ");
        wrapped.push_str(&first_argument_name);
        wrapped.push_str(".watch_overlay);\n");
    }
    wrapped.push_str("    return out;\n}\n");

    Some(Cow::Owned(wrapped))
}

pub(crate) struct CustomPipeline {
    pipeline: RenderPipeline,
    watch_pipeline: Option<RenderPipeline>,
    uniform_ids: Vec<usize>,
}

/// Contains all render pipelines and bind group layouts for [`crate::Renderer`]
///
/// Centralizes GPU pipeline configuration, including:
/// - The main primitive rendering pipeline (textured quads, sprites, shapes)
/// - Texture bind group layout (for sampling textures in shaders)
/// - Camera bind group layout (for view/projection transforms)
pub(crate) struct Pipelines {
    primitive: RenderPipeline,
    primitive_replace: RenderPipeline,
    primitive_watch: Option<RenderPipeline>,
    primitive_replace_watch: Option<RenderPipeline>,
    custom: Vec<CustomPipeline>,
    texture_layout: BindGroupLayout,
    pub camera_layout: BindGroupLayout,
    watch_overlay_supported: bool,
}

impl Pipelines {
    /// Creates all pipelines and bind group layouts for the given device and surface format
    pub fn new(device: &Device, surface_format: TextureFormat, watch_overlay_supported: bool) -> Self {
        let texture_layout = create_texture_bind_group_layout(device);
        let camera_layout = create_camera_bind_group_layout(device);

        let primitive = create_primitive_pipeline(
            device,
            surface_format,
            &texture_layout,
            &camera_layout,
            Some(BlendState::ALPHA_BLENDING),
            true,
            false,
            false,
        );
        let primitive_replace =
            create_primitive_pipeline(device, surface_format, &texture_layout, &camera_layout, None, false, true, false);
        let primitive_watch = watch_overlay_supported.then(|| {
            create_primitive_pipeline(
                device,
                surface_format,
                &texture_layout,
                &camera_layout,
                Some(BlendState::ALPHA_BLENDING),
                true,
                false,
                true,
            )
        });
        let primitive_replace_watch = watch_overlay_supported
            .then(|| create_primitive_pipeline(device, surface_format, &texture_layout, &camera_layout, None, false, true, true));

        Self {
            primitive,
            primitive_replace,
            primitive_watch,
            primitive_replace_watch,
            custom: Vec::new(),
            texture_layout,
            camera_layout,
            watch_overlay_supported,
        }
    }

    /// Creates a custom shader pipeline from WGSL source
    pub fn add_custom(
        &mut self,
        device: &Device,
        surface_format: TextureFormat,
        wgsl_source: &str,
        uniform_layouts: &[&BindGroupLayout],
        uniform_ids: &[usize],
    ) -> usize {
        let pipeline = create_custom_pipeline(
            device,
            surface_format,
            &self.texture_layout,
            &self.camera_layout,
            uniform_layouts,
            wgsl_source,
        );
        let watch_pipeline = if self.watch_overlay_supported {
            create_custom_watch_pipeline(
                device,
                surface_format,
                &self.texture_layout,
                &self.camera_layout,
                uniform_layouts,
                wgsl_source,
            )
        } else {
            None
        };

        self.custom.push(CustomPipeline {
            pipeline,
            watch_pipeline,
            uniform_ids: uniform_ids.to_vec(),
        });
        self.custom.len() - 1
    }

    pub fn resolve_with_replace(
        &self,
        shader_id: Option<usize>,
        replace_blend: bool,
        watch_overlay: bool,
    ) -> Option<(&RenderPipeline, &[usize])> {
        if replace_blend && shader_id.is_none() {
            return Some((
                if watch_overlay {
                    self.primitive_replace_watch.as_ref()?
                } else {
                    &self.primitive_replace
                },
                &[],
            ));
        }
        if let Some(custom) = shader_id.and_then(|id| self.custom.get(id)) {
            if watch_overlay {
                custom
                    .watch_pipeline
                    .as_ref()
                    .map(|pipeline| (pipeline, custom.uniform_ids.as_slice()))
            } else {
                Some((&custom.pipeline, &custom.uniform_ids))
            }
        } else if watch_overlay {
            Some((self.primitive_watch.as_ref()?, &[]))
        } else {
            Some((&self.primitive, &[]))
        }
    }

    pub fn supports_watch_overlay(&self, shader_id: Option<usize>) -> bool {
        self.watch_overlay_supported
            && match shader_id {
                Some(id) => self.custom.get(id).is_some_and(|custom| custom.watch_pipeline.is_some()),
                None => self.primitive_watch.is_some() && self.primitive_replace_watch.is_some(),
            }
    }
}

/// Creates the bind group layout for texture sampling
///
/// Defines two bindings:
/// - Binding 0: 2D texture (fragment shader)
/// - Binding 1: Sampler (fragment shader)
fn create_texture_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Texture Bind Group Layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Creates the bind group layout for camera uniforms
///
/// Defines a single binding:
/// - Binding 0: Uniform buffer containing view-projection matrix (vertex shader)
///
/// Uses `has_dynamic_offset: true` so multiple camera matrices can be stored
/// in a single buffer and selected per-draw-call via dynamic offset, avoiding
/// render pass splits on camera changes.
fn create_camera_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Camera Bind Group Layout"),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// Creates the main rendering pipeline for 2D primitives
///
/// Configured with:
/// - Alpha blending for transparency
/// - Vertex shader transforms using camera uniform
/// - Fragment shader samples from texture
/// - `Vertex` buffer layout from vertex module
fn create_primitive_pipeline(
    device: &Device,
    surface_format: TextureFormat,
    texture_layout: &BindGroupLayout,
    camera_layout: &BindGroupLayout,
    blend: Option<BlendState>,
    depth_enabled: bool,
    replace: bool,
    watch_overlay: bool,
) -> RenderPipeline {
    let shader = device.create_shader_module(include_wgsl!("../shader.wgsl"));
    let fragment_entry_point = match (replace, surface_needs_srgb_encode(surface_format), watch_overlay) {
        (true, true, true) => "fs_replace_srgb_encoded_watch",
        (true, false, true) => "fs_replace_watch",
        (false, true, true) => "fs_main_srgb_encoded_watch",
        (false, false, true) => "fs_main_watch",
        (true, true, false) => "fs_replace_srgb_encoded",
        (true, false, false) => "fs_replace",
        (false, true, false) => "fs_main_srgb_encoded",
        (false, false, false) => "fs_main",
    };
    let mut targets = vec![Some(ColorTargetState {
        format: surface_format,
        blend,
        write_mask: ColorWrites::ALL,
    })];
    if watch_overlay {
        targets.push(Some(ColorTargetState {
            format: WATCH_OVERLAY_FORMAT,
            blend,
            write_mask: ColorWrites::ALL,
        }));
    }
    let pipeline_label = match (replace, watch_overlay) {
        (true, true) => "Primitive Replace Watch Overlay Pipeline",
        (false, true) => "Primitive Watch Overlay Pipeline",
        (true, false) => "Primitive Replace Pipeline",
        (false, false) => "Primitive Pipeline",
    };
    let pipeline_layout_label = match (replace, watch_overlay) {
        (true, true) => "Primitive Replace Watch Overlay Pipeline Layout",
        (false, true) => "Primitive Watch Overlay Pipeline Layout",
        (true, false) => "Primitive Replace Pipeline Layout",
        (false, false) => "Primitive Pipeline Layout",
    };

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(pipeline_layout_label),
        bind_group_layouts: &[Some(texture_layout), Some(camera_layout)],
        immediate_size: 0,
    });

    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some(pipeline_label),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(Vertex::desc()), Some(Instance::desc())],
            compilation_options: Default::default(),
        },
        primitive: Default::default(),
        depth_stencil: Some(DepthStencilState {
            format: crate::Renderer::DEPTH_FORMAT,
            depth_write_enabled: Some(depth_enabled),
            depth_compare: Some(if depth_enabled {
                CompareFunction::LessEqual
            } else {
                CompareFunction::Always
            }),
            stencil: StencilState::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some(fragment_entry_point),
            targets: &targets,
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Creates a custom rendering pipeline from user-provided WGSL source
///
/// Configured with the same layout as the primitive pipeline:
/// - Alpha blending for transparency
/// - Vertex shader transforms using camera uniform
/// - Fragment shader samples from texture
/// - `Vertex` buffer layout from vertex module
fn create_custom_pipeline(
    device: &Device,
    surface_format: TextureFormat,
    texture_layout: &BindGroupLayout,
    camera_layout: &BindGroupLayout,
    extra_layouts: &[&BindGroupLayout],
    wgsl_source: &str,
) -> RenderPipeline {
    let wgsl_source = if surface_needs_srgb_encode(surface_format) {
        wrap_custom_shader_for_srgb_output(wgsl_source)
    } else {
        Cow::Borrowed(wgsl_source)
    };
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Custom Shader"),
        source: ShaderSource::Wgsl(wgsl_source),
    });

    let mut layouts: Vec<Option<&BindGroupLayout>> = vec![Some(texture_layout), Some(camera_layout)];
    layouts.extend(extra_layouts.iter().map(|l| Some(*l)));

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Custom Pipeline Layout"),
        bind_group_layouts: &layouts,
        immediate_size: 0,
    });

    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("Custom Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(Vertex::desc()), Some(Instance::desc())],
            compilation_options: Default::default(),
        },
        primitive: Default::default(),
        depth_stencil: Some(DepthStencilState {
            format: crate::Renderer::DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::LessEqual),
            stencil: StencilState::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(ColorTargetState {
                format: surface_format,
                blend: Some(BlendState::ALPHA_BLENDING),
                write_mask: ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn create_custom_watch_pipeline(
    device: &Device,
    surface_format: TextureFormat,
    texture_layout: &BindGroupLayout,
    camera_layout: &BindGroupLayout,
    extra_layouts: &[&BindGroupLayout],
    wgsl_source: &str,
) -> Option<RenderPipeline> {
    let wgsl_source = wrap_custom_shader_for_watch_output(wgsl_source, surface_needs_srgb_encode(surface_format))?;
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Custom Watch Shader"),
        source: ShaderSource::Wgsl(wgsl_source),
    });

    let mut layouts: Vec<Option<&BindGroupLayout>> = vec![Some(texture_layout), Some(camera_layout)];
    layouts.extend(extra_layouts.iter().map(|l| Some(*l)));

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Custom Watch Pipeline Layout"),
        bind_group_layouts: &layouts,
        immediate_size: 0,
    });

    Some(device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("Custom Watch Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(Vertex::desc()), Some(Instance::desc())],
            compilation_options: Default::default(),
        },
        primitive: Default::default(),
        depth_stencil: Some(DepthStencilState {
            format: crate::Renderer::DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::LessEqual),
            stencil: StencilState::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[
                Some(ColorTargetState {
                    format: surface_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                }),
                Some(ColorTargetState {
                    format: WATCH_OVERLAY_FORMAT,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                }),
            ],
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUSTOM_SHADER_WITH_OVERLAY: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(1) @interpolate(flat) watch_overlay: f32,
};

@vertex
fn vs_main() -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(0.0);
    out.watch_overlay = 1.0;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.watch_overlay, 0.0, 0.0, 1.0);
}
"#;

    const CUSTOM_SHADER_WITHOUT_OVERLAY: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main() -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(0.0);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.position;
}
"#;

    #[test]
    fn custom_watch_wrapper_requires_overlay_factor() {
        assert!(wrap_custom_shader_for_watch_output(CUSTOM_SHADER_WITHOUT_OVERLAY, false).is_none());
    }

    #[test]
    fn custom_watch_wrapper_emits_overlay_attachment() {
        let wrapped = wrap_custom_shader_for_watch_output(CUSTOM_SHADER_WITH_OVERLAY, false).expect("wrapper should succeed");

        assert!(wrapped.contains("@location(1) overlay: vec4<f32>"));
        assert!(wrapped.contains("input.watch_overlay"));
    }
}
