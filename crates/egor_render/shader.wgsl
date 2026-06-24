@group(0) @binding(0)
var texture_binding: texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler: sampler;

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) tex_coords: vec2<f32>,
};

struct InstanceInput {
    @location(3) affine: vec4<f32>,
    @location(4) translate: vec3<f32>,
    @location(5) color: vec4<f32>,
    @location(6) uv: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(vert: VertexInput, inst: InstanceInput) -> VertexOutput {
    let rotscale = mat2x2<f32>(inst.affine.xy, inst.affine.zw);
    let world_pos = rotscale * vert.position + inst.translate.xy;
    let uv = vec2<f32>(
        mix(inst.uv.x, inst.uv.z, vert.tex_coords.x),
        mix(inst.uv.y, inst.uv.w, vert.tex_coords.y),
    );

    var out: VertexOutput;
    out.position = camera.view_proj * vec4<f32>(world_pos, -inst.translate.z, 1.0);
    out.color = vert.color * inst.color;
    out.tex_coords = uv;
    return out;
}

fn fs_main_linear(input: VertexOutput) -> vec4<f32> {
    let color = textureSample(texture_binding, texture_sampler, input.tex_coords) * input.color;
    if color.a < 0.004 {
        discard;
    }
    return color;
}

fn linear_to_srgb_channel(value: f32) -> f32 {
    let clamped = clamp(value, 0.0, 1.0);
    if clamped <= 0.0031308 {
        return clamped * 12.92;
    }
    return 1.055 * pow(clamped, 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        linear_to_srgb_channel(color.r),
        linear_to_srgb_channel(color.g),
        linear_to_srgb_channel(color.b),
        color.a,
    );
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return fs_main_linear(input);
}

@fragment
fn fs_main_srgb_encoded(input: VertexOutput) -> @location(0) vec4<f32> {
    return linear_to_srgb(fs_main_linear(input));
}
