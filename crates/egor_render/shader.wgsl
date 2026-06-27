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
    @location(7) outline_color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) @interpolate(flat) uv_rect: vec4<f32>,
    @location(3) @interpolate(flat) outline_color: vec4<f32>,
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
    out.uv_rect = inst.uv;
    out.outline_color = inst.outline_color;
    return out;
}

fn inside(pos: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>) -> f32 {
    return step(lo.x, pos.x) * step(pos.x, hi.x)
         * step(lo.y, pos.y) * step(pos.y, hi.y);
}

fn fs_main_linear(input: VertexOutput) -> vec4<f32> {
    if input.outline_color.a > 0.0 {
        let dims = vec2<f32>(textureDimensions(texture_binding, 0));
        let texel = 1.0 / dims;
        let glyph_min = input.uv_rect.xy + texel;
        let glyph_max = input.uv_rect.zw - texel;
        let uv = input.tex_coords;
        let center = textureSample(texture_binding, texture_sampler, uv);
        let c_in = inside(uv, glyph_min, glyph_max);
        if c_in > 0.5 && center.a > 0.004 {
            return center * input.color;
        }

        var nb: f32 = 0.0;
        let n0 = uv + vec2<f32>(-texel.x, -texel.y);
        nb = max(nb, textureSampleLevel(texture_binding, texture_sampler, n0, 0.0).a * inside(n0, glyph_min, glyph_max));
        let n1 = uv + vec2<f32>( 0.0,     -texel.y);
        nb = max(nb, textureSampleLevel(texture_binding, texture_sampler, n1, 0.0).a * inside(n1, glyph_min, glyph_max));
        let n2 = uv + vec2<f32>( texel.x, -texel.y);
        nb = max(nb, textureSampleLevel(texture_binding, texture_sampler, n2, 0.0).a * inside(n2, glyph_min, glyph_max));
        let n3 = uv + vec2<f32>(-texel.x,  0.0);
        nb = max(nb, textureSampleLevel(texture_binding, texture_sampler, n3, 0.0).a * inside(n3, glyph_min, glyph_max));
        let n4 = uv + vec2<f32>( texel.x,  0.0);
        nb = max(nb, textureSampleLevel(texture_binding, texture_sampler, n4, 0.0).a * inside(n4, glyph_min, glyph_max));
        let n5 = uv + vec2<f32>(-texel.x,  texel.y);
        nb = max(nb, textureSampleLevel(texture_binding, texture_sampler, n5, 0.0).a * inside(n5, glyph_min, glyph_max));
        let n6 = uv + vec2<f32>( 0.0,      texel.y);
        nb = max(nb, textureSampleLevel(texture_binding, texture_sampler, n6, 0.0).a * inside(n6, glyph_min, glyph_max));
        let n7 = uv + vec2<f32>( texel.x,  texel.y);
        nb = max(nb, textureSampleLevel(texture_binding, texture_sampler, n7, 0.0).a * inside(n7, glyph_min, glyph_max));

        let outline_alpha = input.outline_color.a * nb;
        if outline_alpha > 0.004 {
            return vec4<f32>(input.outline_color.rgb, outline_alpha);
        }
        discard;
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

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
