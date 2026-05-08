// ---------------------------------------------------------------------------
// Single-pass outlined text shader.
//
// Instead of drawing each glyph 9 times (8 outline offsets + 1 center),
// this shader samples the 8 neighboring texels in the font atlas and
// composites the outline in a single fragment pass.
//
// The instance `color` carries the CENTER (foreground) text colour.
// The outline colour is provided via a uniform at group(2).
// ---------------------------------------------------------------------------

@group(0) @binding(0)
var texture_binding: texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler: sampler;

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct OutlineParams {
    color: vec4<f32>,
};
@group(2) @binding(0)
var<uniform> outline: OutlineParams;

// Same layouts as the base shader — custom pipelines reuse Vertex + Instance.
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
    // The expanded UV rect from the instance (glyph + 1 texel border).
    // Used to derive the original glyph UV bounds in the fragment shader.
    @location(2) @interpolate(flat) uv_rect: vec4<f32>,
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
    return out;
}

// Branchless inside-rect test: returns 1.0 if pos is within [lo, hi], else 0.0.
fn inside(pos: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>) -> f32 {
    return step(lo.x, pos.x) * step(pos.x, hi.x)
         * step(lo.y, pos.y) * step(pos.y, hi.y);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(texture_binding, 0));
    let texel = 1.0 / dims;

    // The instance UV covers the EXPANDED region (glyph + 1 texel border).
    // Shrink by 1 texel on each side to recover the original glyph bounds.
    let glyph_min = input.uv_rect.xy + texel;
    let glyph_max = input.uv_rect.zw - texel;

    let uv = input.tex_coords;
    let center = textureSample(texture_binding, texture_sampler, uv);

    // Only treat center as foreground if it falls within the original glyph.
    let c_in = inside(uv, glyph_min, glyph_max);
    if c_in > 0.5 && center.a > 0.004 {
        return center * input.color;
    }

    // 8 neighbours — only count those whose position falls inside the glyph.
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

    if nb > 0.004 {
        return vec4<f32>(outline.color.rgb, outline.color.a * nb);
    }

    // Fully transparent — FXC requires a return on every control path.
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
