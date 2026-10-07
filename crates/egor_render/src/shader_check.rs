//! Checks a custom shader for every problem that does not depend on the GPU or driver, so tools can
//! reject it before it ships instead of players hitting wgpu validation errors.
//!
//! The check builds every pipeline [`crate::Renderer::add_shader_with_uniforms`] would build, on wgpu's
//! validation-only device with WebGL2 limits, the weakest target egor supports. It then translates every
//! WGSL module those pipelines compile through each backend's shader writer: SPIR-V for Vulkan, GLSL ES
//! 3.00 for OpenGL ES and WebGL2, MSL for Metal and HLSL for DX12. Only the platform shader compiler
//! (the GL driver, Metal, FXC/DXC) runs after this.

use std::sync::{Arc, Mutex};

use naga::back::{glsl, hlsl, msl, spv};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use wgpu::{Device, DeviceDescriptor, Limits, TextureFormat};

use crate::REQUIRED_MAX_TEXTURE_DIMENSION_2D;
use crate::pipeline::{
    Pipelines, custom_shader_source, surface_needs_srgb_encode, wrap_custom_shader_for_watch_output,
};
use crate::uniforms::Uniforms;

/// Surface formats that select each output path: one egor sRGB-encodes in the shader, one it does not.
const SURFACE_FORMATS: [TextureFormat; 2] =
    [TextureFormat::Rgba8Unorm, TextureFormat::Rgba8UnormSrgb];

/// Returns every reason `wgsl_source` would fail to compile, link or draw on some backend when loaded with
/// uniform buffers of `uniform_sizes` bytes, bound from `@group(2)` up. An empty list means the shader is
/// valid everywhere.
pub fn check_custom_shader(wgsl_source: &str, uniform_sizes: &[u64]) -> Vec<String> {
    let mut required_limits = Limits::downlevel_webgl2_defaults();
    required_limits.max_texture_dimension_2d = REQUIRED_MAX_TEXTURE_DIMENSION_2D;
    let (device, _queue) = Device::noop(&DeviceDescriptor {
        required_limits,
        ..Default::default()
    });
    let wgpu_errors = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&wgpu_errors);
    device.on_uncaptured_error(Arc::new(move |error| {
        sink.lock().unwrap().push(error.to_string())
    }));

    let mut errors = Vec::new();
    let uniforms = Uniforms::new(&device);
    let uniform_layouts = vec![uniforms.layout(); uniform_sizes.len()];
    let uniform_ids: Vec<usize> = (0..uniform_sizes.len()).collect();
    for surface_format in SURFACE_FORMATS {
        let mut pipelines = Pipelines::new(&device, surface_format, true);
        let shader_id = pipelines.add_custom(
            &device,
            surface_format,
            wgsl_source,
            &uniform_layouts,
            &uniform_ids,
        );
        errors.extend(
            wgpu_errors
                .lock()
                .unwrap()
                .drain(..)
                .map(|error| format!("wgpu ({surface_format:?} surface): {error}")),
        );
        if !pipelines.supports_watch_overlay(Some(shader_id)) {
            errors.push(format!(
                "no watch overlay pipeline ({surface_format:?} surface): fs_main must take a struct carrying \
                 `watch_overlay` and return `@location(0) vec4<f32>`"
            ));
        }
    }

    errors.extend(check_uniform_sizes(wgsl_source, uniform_sizes));
    for (label, source) in compiled_modules(wgsl_source) {
        errors.extend(
            translate(&source)
                .into_iter()
                .map(|error| format!("{label}: {error}")),
        );
    }
    errors
}

/// egor's uniform layouts leave the binding size open, so wgpu only rejects a uniform the shader reads past
/// the end of when it is drawn.
fn check_uniform_sizes(wgsl_source: &str, uniform_sizes: &[u64]) -> Vec<String> {
    // wgpu already reported parse errors.
    let Ok(module) =
        naga::front::wgsl::parse_str(&crate::prepare_texture_shader(wgsl_source, false))
    else {
        return Vec::new();
    };
    let mut errors = Vec::new();
    for (index, &bound) in uniform_sizes.iter().enumerate() {
        let group = index as u32 + 2;
        let needed = module
            .global_variables
            .iter()
            .filter(|(_, var)| {
                var.binding
                    .as_ref()
                    .is_some_and(|binding| binding.group == group)
            })
            .map(|(_, var)| u64::from(module.types[var.ty].inner.size(module.to_ctx())))
            .max();
        if let Some(needed) = needed.filter(|&needed| needed > bound) {
            errors.push(format!(
                "@group({group}) uniform needs {needed} bytes but is bound with {bound}"
            ));
        }
    }
    errors
}

/// Every WGSL module egor compiles for a custom shader, labelled by the pipeline it belongs to.
fn compiled_modules(wgsl_source: &str) -> Vec<(String, String)> {
    let texture_kinds: &[bool] = if wgsl_source.contains("// EGOR_TEXTURE_SAMPLING") {
        &[false, true]
    } else {
        &[false]
    };
    let mut modules = Vec::new();
    for &array in texture_kinds {
        let prepared = crate::prepare_texture_shader(wgsl_source, array);
        let texture = if array { "texture array" } else { "2D texture" };
        for surface_format in SURFACE_FORMATS {
            modules.push((
                format!("{texture}, {surface_format:?} surface"),
                custom_shader_source(&prepared, surface_format).into_owned(),
            ));
            if let Some(watch) = wrap_custom_shader_for_watch_output(
                &prepared,
                surface_needs_srgb_encode(surface_format),
            ) {
                modules.push((
                    format!("{texture}, {surface_format:?} surface, watch overlay"),
                    watch.into_owned(),
                ));
            }
        }
    }
    modules
}

/// Runs each backend's shader writer over both entry points, with the options the matching wgpu backend
/// uses on its least capable devices.
fn translate(source: &str) -> Vec<String> {
    // wgpu already reported parse and validation errors for this module.
    let Ok(module) = naga::front::wgsl::parse_str(source) else {
        return Vec::new();
    };
    // A device with no optional features and WebGL2 downlevel flags gives naga no extra capabilities.
    let info = match Validator::new(ValidationFlags::all(), Capabilities::empty()).validate(&module)
    {
        Ok(info) => info,
        Err(error) => {
            return vec![format!(
                "needs a capability WebGL2 and baseline devices lack:\n{}",
                error.emit_to_string(source)
            )];
        }
    };

    let mut errors = Vec::new();
    for entry_point in &module.entry_points {
        let stage = entry_point.stage;
        let name = entry_point.name.as_str();
        let (module, info) = match naga::back::pipeline_constants::process_overrides(
            &module,
            &info,
            Some((stage, name)),
            &Default::default(),
        ) {
            Ok(processed) => processed,
            Err(error) => {
                errors.push(format!("{name}: pipeline constants: {error}"));
                continue;
            }
        };
        let mut check = |backend: &str, result: Result<(), String>| {
            if let Err(error) = result {
                errors.push(format!("{backend} {name}: {error}"));
            }
        };

        check("SPIR-V (Vulkan)", write_spirv(&module, &info, stage, name));
        for is_webgl in [false, true] {
            let backend = if is_webgl {
                "GLSL ES 3.00 (WebGL2)"
            } else {
                "GLSL ES 3.00 (OpenGL ES)"
            };
            check(backend, write_glsl(&module, &info, stage, name, is_webgl));
        }
        check("MSL (Metal)", write_msl(&module, &info, stage, name));
        check("HLSL (DX12)", write_hlsl(&module, &info, stage, name));
    }
    errors
}

fn write_spirv(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    stage: naga::ShaderStage,
    name: &str,
) -> Result<(), String> {
    // SPIR-V 1.0 with the capabilities wgpu's Vulkan backend always enables.
    let options = spv::Options {
        lang_version: (1, 0),
        capabilities: Some(
            [
                spv::Capability::Shader,
                spv::Capability::Matrix,
                spv::Capability::Sampled1D,
                spv::Capability::Image1D,
                spv::Capability::ImageQuery,
                spv::Capability::DerivativeControl,
                spv::Capability::StorageImageExtendedFormats,
            ]
            .into_iter()
            .collect(),
        ),
        flags: spv::WriterFlags::FORCE_POINT_SIZE,
        ..Default::default()
    };
    let pipeline = spv::PipelineOptions {
        shader_stage: stage,
        entry_point: name.to_owned(),
    };
    spv::write_vec(module, info, &options, Some(&pipeline))
        .map(drop)
        .map_err(|error| error.to_string())
}

fn write_glsl(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    stage: naga::ShaderStage,
    name: &str,
    is_webgl: bool,
) -> Result<(), String> {
    let options = glsl::Options {
        version: glsl::Version::Embedded {
            version: 300,
            is_webgl,
        },
        writer_flags: glsl::WriterFlags::ADJUST_COORDINATE_SPACE
            | glsl::WriterFlags::FORCE_POINT_SIZE,
        binding_map: Default::default(),
        zero_initialize_workgroup_memory: true,
    };
    let pipeline = glsl::PipelineOptions {
        shader_stage: stage,
        entry_point: name.to_owned(),
        multiview: None,
    };
    let mut output = String::new();
    glsl::Writer::new(
        &mut output,
        module,
        info,
        &options,
        &pipeline,
        naga::proc::BoundsCheckPolicies::default(),
    )
    .and_then(|mut writer| writer.write().map(drop))
    .map_err(|error| error.to_string())
}

fn write_msl(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    stage: naga::ShaderStage,
    name: &str,
) -> Result<(), String> {
    // MSL 1.2 is the oldest version wgpu's Metal backend targets.
    let options = msl::Options {
        lang_version: (1, 2),
        fake_missing_bindings: true,
        ..Default::default()
    };
    let pipeline = msl::PipelineOptions {
        entry_point: Some((stage, name.to_owned())),
        allow_and_force_point_size: stage == naga::ShaderStage::Vertex,
        ..Default::default()
    };
    msl::write_string(module, info, &options, &pipeline)
        .map(drop)
        .map_err(|error| error.to_string())
}

fn write_hlsl(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    stage: naga::ShaderStage,
    name: &str,
) -> Result<(), String> {
    // Shader model 5.1 is the oldest wgpu's DX12 backend targets.
    let options = hlsl::Options {
        shader_model: hlsl::ShaderModel::V5_1,
        fake_missing_bindings: true,
        ..Default::default()
    };
    let pipeline = hlsl::PipelineOptions {
        entry_point: Some((stage, name.to_owned())),
    };
    // The vertex stage is written against the fragment input it feeds, as wgpu does.
    let fragment = (stage == naga::ShaderStage::Vertex)
        .then(|| hlsl::FragmentEntryPoint::new(module, "fs_main"))
        .flatten();
    let mut output = String::new();
    hlsl::Writer::new(&mut output, &options, &pipeline)
        .write(module, info, fragment.as_ref())
        .map(drop)
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::check_custom_shader;

    const VALID: &str = r#"
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

    #[test]
    fn valid_shader_passes_on_every_backend() {
        assert_eq!(check_custom_shader(VALID, &[]), Vec::<String>::new());
    }

    #[test]
    fn rejects_watch_overlay_missing_from_the_fragment_input() {
        let source = r#"
struct InstanceInput {
    @location(8) watch_overlay: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(inst: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(inst.watch_overlay);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.position;
}
"#;
        let errors = check_custom_shader(source, &[]);
        assert!(
            errors
                .iter()
                .any(|error| error.contains("invalid field accessor `watch_overlay`")),
            "{errors:#?}"
        );
    }

    #[test]
    fn rejects_uniforms_the_client_does_not_bind_in_full() {
        let source = VALID
            .replace(
                "@vertex",
                "@group(2) @binding(0) var<uniform> params: vec4<f32>;\n\n@vertex",
            )
            .replace("input.watch_overlay, 0.0", "input.watch_overlay, params.x");
        assert!(check_custom_shader(&source, &[16]).is_empty());
        assert!(!check_custom_shader(&source, &[]).is_empty());
        assert_eq!(
            check_custom_shader(&source, &[8]),
            ["@group(2) uniform needs 16 bytes but is bound with 8"]
        );
    }

    #[test]
    fn rejects_capabilities_webgl2_lacks() {
        let source = VALID.replace(
            "@fragment\nfn fs_main(input: VertexOutput)",
            "struct FragmentInput {\n    @builtin(position) position: vec4<f32>,\n    @location(1) @interpolate(flat) watch_overlay: f32,\n    @builtin(sample_index) sample: u32,\n};\n\n@fragment\nfn fs_main(input: FragmentInput)",
        );
        let errors = check_custom_shader(&source, &[]);
        assert!(
            !errors.is_empty()
                && errors
                    .iter()
                    .all(|error| error.contains("needs a capability WebGL2")),
            "{errors:#?}"
        );
    }
}
