use egor_render::{
    AdapterInfo, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Extent3d, Renderer, Texture, TextureFormat,
    batch::GeometryBatch,
    target::{OffscreenTarget, RenderTarget},
};
use glam::Vec2;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::{self, Receiver, Sender};
#[cfg(not(target_arch = "wasm32"))]
use std::thread;
use web_time::Instant;

use crate::primitives::PathBuilder;
use crate::{
    camera::Camera,
    color::Color,
    math::Rect,
    primitives::{PolygonBuilder, PolylineBuilder, PrimitiveBatch, RectangleBuilder},
    text::{TextBuilder, TextRenderer},
};

// ---------------------------------------------------------------------------
// Render Target Store — persistent across frames, owns OffscreenTargets
// ---------------------------------------------------------------------------

/// Persistent storage for offscreen render targets created by game code.
///
/// Stored in [`App`] and passed to [`Graphics`] each frame by reference.
/// Render targets survive across frames; the game is responsible for
/// recreating them when their size changes.
pub struct RenderTargetStore {
    targets: Vec<OffscreenTarget>,
}

impl RenderTargetStore {
    pub fn new() -> Self {
        Self { targets: Vec::new() }
    }

    /// Create an offscreen render target and return its index.
    pub fn create(&mut self, device: &Device, width: u32, height: u32, format: TextureFormat) -> usize {
        let id = self.targets.len();
        self.targets.push(OffscreenTarget::new(device, width, height, format));
        id
    }

    /// Resize an existing offscreen target. If its dimensions already match, this is a no-op.
    pub fn resize(&mut self, device: &Device, id: usize, width: u32, height: u32) {
        self.targets[id].resize(device, width, height);
    }

    pub fn get(&self, id: usize) -> &OffscreenTarget {
        &self.targets[id]
    }

    pub fn get_mut(&mut self, id: usize) -> &mut OffscreenTarget {
        &mut self.targets[id]
    }

    pub fn len(&self) -> usize {
        self.targets.len()
    }
}

impl Default for RenderTargetStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Screen Capture State — wgpu backend for anti-cheat GPU readback
// ---------------------------------------------------------------------------

const CAP_HEIGHT: u32 = 360;
const SLOT_COUNT: usize = 3;

const MAP_PENDING: u8 = 0;
const MAP_READY: u8 = 1;
const MAP_FAILED: u8 = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct WatchCaptureUniform {
    source_w: u32,
    source_h: u32,
    logical_w: u32,
    logical_h: u32,
    factor: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct PreparedCapture {
    slot_idx: usize,
    cap_w: u32,
    cap_h: u32,
    grayscale: bool,
    alpha_mask: bool,
    metadata: Option<[f32; 10]>,
    dirty_rects: Option<Vec<[u16; 4]>>,
    frame_tag: Option<[u64; 4]>,
}

pub type WatchCaptureDirtyRects = Vec<[u16; 4]>;
pub type WatchCaptureFrameTag = [u64; 4];

/// Capture-only counters for profiling. Byte counts exclude driver overhead.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScreenCaptureMetrics {
    pub readback_bytes: u64,
    pub staging_allocations: u64,
    pub staging_allocated_bytes: u64,
    pub skipped_requests: u64,
    pub completed_frames: u64,
    pub decode_us: u64,
    pub conversion_us: u64,
    pub unmap_us: u64,
    pub worker_jobs: u64,
    pub worker_bytes: u64,
    pub gpu_bytes: u64,
    pub cpu_buffer_bytes: u64,
}

impl WatchCaptureUniform {
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts((self as *const Self).cast::<u8>(), std::mem::size_of::<Self>()) }
    }
}

/// WGSL shader for fullscreen-triangle blit with bilinear sampling.
const BLIT_SHADER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    // Fullscreen triangle: 3 vertices cover the entire clip space.
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    // Flip Y so texture top-left maps to NDC top-left.
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_src, s_src, in.uv);
}
"#;

/// WGSL shader for fullscreen-triangle blit that outputs BT.601 luminance
/// to a single-channel R8Unorm render target. Eliminates CPU-side grayscale
/// conversion and reduces GPU→CPU readback bandwidth by 4×.
const BLIT_GRAY_SHADER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = textureSample(t_src, s_src, in.uv);
    let lum = dot(c.rgb, vec3<f32>(0.299, 0.587, 0.114));
    return vec4<f32>(lum, 0.0, 0.0, 1.0);
}
"#;

/// Grayscale+alpha capture variant. Writes luminance to R and alpha to G in
/// an Rg8Unorm target so CPU readback is 2 B/px instead of RGBA8 4 B/px.
const BLIT_GRAY_ALPHA_SHADER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = textureSample(t_src, s_src, in.uv);
    let lum = dot(c.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let gray = select(0.0, lum, c.a > 0.0);
    return vec4<f32>(gray, c.a, 0.0, 1.0);
}
"#;

/// Capture shader for sources sampled as linear color where the displayed
/// backbuffer uses sRGB encoding. It writes display-encoded RGB into a
/// non-sRGB readback target so CPU-side packing sees the same byte domain as
/// the direct surface-copy path.
const BLIT_ENCODE_SHADER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = textureSample(t_src, s_src, in.uv);
    return vec4<f32>(linear_to_srgb(c.rgb), c.a);
}
"#;

/// Grayscale capture variant for linear sources shown through an sRGB target.
const BLIT_GRAY_ENCODE_SHADER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = linear_to_srgb(textureSample(t_src, s_src, in.uv).rgb);
    let lum = dot(c, vec3<f32>(0.299, 0.587, 0.114));
    return vec4<f32>(lum, 0.0, 0.0, 1.0);
}
"#;

/// Grayscale+alpha capture variant for linear sources shown through an sRGB target.
const BLIT_GRAY_ALPHA_ENCODE_SHADER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sampled = textureSample(t_src, s_src, in.uv);
    let c = linear_to_srgb(sampled.rgb);
    let lum = dot(c, vec3<f32>(0.299, 0.587, 0.114));
    let gray = select(0.0, lum, sampled.a > 0.0);
    return vec4<f32>(gray, sampled.a, 0.0, 1.0);
}
"#;

const WATCH_CAPTURE_SHADER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

struct CaptureUniform {
    source_w: u32,
    source_h: u32,
    logical_w: u32,
    logical_h: u32,
    factor: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var t_overlay: texture_2d<f32>;
@group(0) @binding(1) var<uniform> capture: CaptureUniform;

fn logical_to_source(logical: u32, logical_size: u32, source_size: u32) -> u32 {
    if logical_size <= 1u || source_size <= 1u {
        return 0u;
    }
    let mapped = u32(((f32(logical) + 0.5) * f32(source_size)) / f32(logical_size));
    return min(mapped, source_size - 1u);
}

fn unpremultiply(rgb: vec3<f32>, alpha: f32) -> vec3<f32> {
    if alpha <= 0.0001 {
        return vec3<f32>(0.0);
    }
    return clamp(rgb / alpha, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn selected_overlay(pixel: vec2<u32>) -> vec4<f32> {
    let factor = max(capture.factor, 1u);
    let start_x = pixel.x * factor;
    let start_y = pixel.y * factor;
    var best_sample = vec4<f32>(0.0);
    var best_alpha = -1.0;
    var max_alpha = 0.0;

    for (var by = 0u; by < 8u; by = by + 1u) {
        if by >= factor || start_y + by >= capture.logical_h {
            break;
        }
        let sy = logical_to_source(start_y + by, capture.logical_h, capture.source_h);
        for (var bx = 0u; bx < 8u; bx = bx + 1u) {
            if bx >= factor || start_x + bx >= capture.logical_w {
                break;
            }
            let sx = logical_to_source(start_x + bx, capture.logical_w, capture.source_w);
            let sample = textureLoad(t_overlay, vec2<i32>(i32(sx), i32(sy)), 0);
            max_alpha = max(max_alpha, sample.a);
            if sample.a > best_alpha {
                best_alpha = sample.a;
                best_sample = sample;
            }
        }
    }

    if max_alpha <= 0.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(unpremultiply(best_sample.rgb, best_sample.a), max_alpha);
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(u32(in.position.x), u32(in.position.y));
    let c = selected_overlay(pixel);
    return vec4<f32>(c.rgb * c.a, c.a);
}
"#;

const WATCH_CAPTURE_GRAY_SHADER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

struct CaptureUniform {
    source_w: u32,
    source_h: u32,
    logical_w: u32,
    logical_h: u32,
    factor: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var t_overlay: texture_2d<f32>;
@group(0) @binding(1) var<uniform> capture: CaptureUniform;

fn logical_to_source(logical: u32, logical_size: u32, source_size: u32) -> u32 {
    if logical_size <= 1u || source_size <= 1u {
        return 0u;
    }
    let mapped = u32(((f32(logical) + 0.5) * f32(source_size)) / f32(logical_size));
    return min(mapped, source_size - 1u);
}

fn unpremultiply(rgb: vec3<f32>, alpha: f32) -> vec3<f32> {
    if alpha <= 0.0001 {
        return vec3<f32>(0.0);
    }
    return clamp(rgb / alpha, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn selected_overlay(pixel: vec2<u32>) -> vec4<f32> {
    let factor = max(capture.factor, 1u);
    let start_x = pixel.x * factor;
    let start_y = pixel.y * factor;
    var best_sample = vec4<f32>(0.0);
    var best_alpha = -1.0;
    var max_alpha = 0.0;

    for (var by = 0u; by < 8u; by = by + 1u) {
        if by >= factor || start_y + by >= capture.logical_h {
            break;
        }
        let sy = logical_to_source(start_y + by, capture.logical_h, capture.source_h);
        for (var bx = 0u; bx < 8u; bx = bx + 1u) {
            if bx >= factor || start_x + bx >= capture.logical_w {
                break;
            }
            let sx = logical_to_source(start_x + bx, capture.logical_w, capture.source_w);
            let sample = textureLoad(t_overlay, vec2<i32>(i32(sx), i32(sy)), 0);
            max_alpha = max(max_alpha, sample.a);
            if sample.a > best_alpha {
                best_alpha = sample.a;
                best_sample = sample;
            }
        }
    }

    if max_alpha <= 0.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(unpremultiply(best_sample.rgb, best_sample.a), max_alpha);
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(u32(in.position.x), u32(in.position.y));
    let c = selected_overlay(pixel);
    let lum = dot(c.rgb, vec3<f32>(0.299, 0.587, 0.114));
    return vec4<f32>(select(0.0, lum, c.a > 0.0), c.a, 0.0, 1.0);
}
"#;

const WATCH_CAPTURE_ENCODE_SHADER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

struct CaptureUniform {
    source_w: u32,
    source_h: u32,
    logical_w: u32,
    logical_h: u32,
    factor: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var t_overlay: texture_2d<f32>;
@group(0) @binding(1) var<uniform> capture: CaptureUniform;

fn logical_to_source(logical: u32, logical_size: u32, source_size: u32) -> u32 {
    if logical_size <= 1u || source_size <= 1u {
        return 0u;
    }
    let mapped = u32(((f32(logical) + 0.5) * f32(source_size)) / f32(logical_size));
    return min(mapped, source_size - 1u);
}

fn unpremultiply(rgb: vec3<f32>, alpha: f32) -> vec3<f32> {
    if alpha <= 0.0001 {
        return vec3<f32>(0.0);
    }
    return clamp(rgb / alpha, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn selected_overlay(pixel: vec2<u32>) -> vec4<f32> {
    let factor = max(capture.factor, 1u);
    let start_x = pixel.x * factor;
    let start_y = pixel.y * factor;
    var best_sample = vec4<f32>(0.0);
    var best_alpha = -1.0;
    var max_alpha = 0.0;

    for (var by = 0u; by < 8u; by = by + 1u) {
        if by >= factor || start_y + by >= capture.logical_h {
            break;
        }
        let sy = logical_to_source(start_y + by, capture.logical_h, capture.source_h);
        for (var bx = 0u; bx < 8u; bx = bx + 1u) {
            if bx >= factor || start_x + bx >= capture.logical_w {
                break;
            }
            let sx = logical_to_source(start_x + bx, capture.logical_w, capture.source_w);
            let sample = textureLoad(t_overlay, vec2<i32>(i32(sx), i32(sy)), 0);
            max_alpha = max(max_alpha, sample.a);
            if sample.a > best_alpha {
                best_alpha = sample.a;
                best_sample = sample;
            }
        }
    }

    if max_alpha <= 0.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(unpremultiply(best_sample.rgb, best_sample.a), max_alpha);
}
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(u32(in.position.x), u32(in.position.y));
    let c = selected_overlay(pixel);
    let rgb = linear_to_srgb(c.rgb);
    return vec4<f32>(rgb * c.a, c.a);
}
"#;

const WATCH_CAPTURE_GRAY_ENCODE_SHADER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

struct CaptureUniform {
    source_w: u32,
    source_h: u32,
    logical_w: u32,
    logical_h: u32,
    factor: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var t_overlay: texture_2d<f32>;
@group(0) @binding(1) var<uniform> capture: CaptureUniform;

fn logical_to_source(logical: u32, logical_size: u32, source_size: u32) -> u32 {
    if logical_size <= 1u || source_size <= 1u {
        return 0u;
    }
    let mapped = u32(((f32(logical) + 0.5) * f32(source_size)) / f32(logical_size));
    return min(mapped, source_size - 1u);
}

fn unpremultiply(rgb: vec3<f32>, alpha: f32) -> vec3<f32> {
    if alpha <= 0.0001 {
        return vec3<f32>(0.0);
    }
    return clamp(rgb / alpha, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn selected_overlay(pixel: vec2<u32>) -> vec4<f32> {
    let factor = max(capture.factor, 1u);
    let start_x = pixel.x * factor;
    let start_y = pixel.y * factor;
    var best_sample = vec4<f32>(0.0);
    var best_alpha = -1.0;
    var max_alpha = 0.0;

    for (var by = 0u; by < 8u; by = by + 1u) {
        if by >= factor || start_y + by >= capture.logical_h {
            break;
        }
        let sy = logical_to_source(start_y + by, capture.logical_h, capture.source_h);
        for (var bx = 0u; bx < 8u; bx = bx + 1u) {
            if bx >= factor || start_x + bx >= capture.logical_w {
                break;
            }
            let sx = logical_to_source(start_x + bx, capture.logical_w, capture.source_w);
            let sample = textureLoad(t_overlay, vec2<i32>(i32(sx), i32(sy)), 0);
            max_alpha = max(max_alpha, sample.a);
            if sample.a > best_alpha {
                best_alpha = sample.a;
                best_sample = sample;
            }
        }
    }

    if max_alpha <= 0.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(unpremultiply(best_sample.rgb, best_sample.a), max_alpha);
}
fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(u32(in.position.x), u32(in.position.y));
    let c = selected_overlay(pixel);
    let rgb = linear_to_srgb(c.rgb);
    let lum = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
    return vec4<f32>(select(0.0, lum, c.a > 0.0), c.a, 0.0, 1.0);
}
"#;

// -- Unsafe pixel-format conversion (matches old OpenGL PBO path perf) ------

/// RGBA → RGB565 with row-pitch padding.
///
/// # Safety
/// `src` must point to at least `h * row_pitch` readable bytes.
/// `dst` must point to at least `w * h * 2` writable bytes.
#[inline]
unsafe fn pack_rgba_to_rgb565(src: *const u8, dst: *mut u8, w: usize, h: usize, row_pitch: usize) {
    let d = dst as *mut u16;
    for y in 0..h {
        let row = unsafe { src.add(y * row_pitch) };
        let dst_off = y * w;
        for x in 0..w {
            let s = unsafe { row.add(x * 4) };
            let r = unsafe { *s } as u16;
            let g = unsafe { *s.add(1) } as u16;
            let b = unsafe { *s.add(2) } as u16;
            unsafe { *d.add(dst_off + x) = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3) };
        }
    }
}

#[inline]
fn unpremultiply_channel(channel: u8, alpha: u8) -> u8 {
    if alpha == 0 {
        0
    } else if alpha == 255 {
        channel
    } else {
        (((channel as u32 * 255) + (alpha as u32 / 2)) / alpha as u32).min(255) as u8
    }
}

fn readback_output_len(cap_w: u32, cap_h: u32, grayscale: bool, alpha_mask: bool) -> usize {
    let pixel_count = cap_w as usize * cap_h as usize;
    match (grayscale, alpha_mask) {
        (true, false) => pixel_count,
        (true, true) => pixel_count * 2,
        (false, false) => pixel_count * 2,
        (false, true) => pixel_count * 3,
    }
}

fn decode_readback_into(
    buffer: &Buffer,
    dst: &mut Vec<u8>,
    cap_w: u32,
    cap_h: u32,
    row_pitch: usize,
    grayscale: bool,
    alpha_mask: bool,
    _source_cache: &mut Vec<u8>,
) {
    let w = cap_w as usize;
    let h = cap_h as usize;
    let pixel_count = w * h;
    let out_len = readback_output_len(cap_w, cap_h, grayscale, alpha_mask);

    dst.reserve(out_len.saturating_sub(dst.len()));
    unsafe { dst.set_len(out_len) };

    let data = buffer
        .slice(..)
        .get_mapped_range()
        .expect("readback buffer range must remain mapped after map_async succeeds");
    // Small loads from host-visible Adreno memory are very expensive.
    // One sequential copy lets color conversion read ordinary cached RAM.
    // Native GLES readback must avoid wgpu's emulated main-thread copy too.
    // Grayscale already uses a bulk copy and needs no intermediate storage.
    #[cfg(target_os = "android")]
    let src = if !grayscale {
        _source_cache.clear();
        _source_cache.extend_from_slice(&data);
        _source_cache.as_ptr()
    } else {
        data.as_ptr()
    };
    #[cfg(not(target_os = "android"))]
    let src = data.as_ptr();

    if alpha_mask {
        if grayscale {
            let unpadded_row = w * 2;
            if row_pitch == unpadded_row {
                unsafe {
                    std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), out_len);
                }
            } else {
                let out = dst.as_mut_ptr();
                for y in 0..h {
                    unsafe {
                        std::ptr::copy_nonoverlapping(src.add(y * row_pitch), out.add(y * unpadded_row), unpadded_row);
                    }
                }
            }
        } else {
            let color_len = pixel_count * 2;
            let (color_out, alpha_out) = dst.split_at_mut(color_len);
            let d = color_out.as_mut_ptr() as *mut u16;
            for y in 0..h {
                let row = unsafe { src.add(y * row_pitch) };
                let dst_off = y * w;
                for x in 0..w {
                    let s = unsafe { row.add(x * 4) };
                    let a = unsafe { *s.add(3) };
                    let idx = dst_off + x;
                    // Map pixels are transparent and HUD/sprites are mostly
                    // opaque. Handle alpha once per pixel, and only read RGB
                    // when it contributes to the captured foreground.
                    let packed = match a {
                        0 => 0,
                        255 => {
                            let r = unsafe { *s } as u16;
                            let g = unsafe { *s.add(1) } as u16;
                            let b = unsafe { *s.add(2) } as u16;
                            ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
                        }
                        _ => {
                            let r = unpremultiply_channel(unsafe { *s }, a) as u16;
                            let g = unpremultiply_channel(unsafe { *s.add(1) }, a) as u16;
                            let b = unpremultiply_channel(unsafe { *s.add(2) }, a) as u16;
                            ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
                        }
                    };
                    unsafe { *d.add(idx) = packed };
                    alpha_out[idx] = a;
                }
            }
        }
    } else if grayscale {
        if row_pitch == w {
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), out_len);
            }
        } else {
            let out = dst.as_mut_ptr();
            for y in 0..h {
                unsafe {
                    std::ptr::copy_nonoverlapping(src.add(y * row_pitch), out.add(y * w), w);
                }
            }
        }
    } else {
        unsafe { pack_rgba_to_rgb565(src, dst.as_mut_ptr(), w, h, row_pitch) };
    }
}

// -- Ring-buffer staging slot -----------------------------------------------

struct StagingSlot {
    buffer: Option<Buffer>,
    rgb_buf: Vec<u8>,
    buf_size: u64,
    row_pitch: u32,
    cap_w: u32,
    cap_h: u32,
    grayscale: bool,
    alpha_mask: bool,
    metadata: Option<[f32; 10]>,
    dirty_rects: Option<WatchCaptureDirtyRects>,
    frame_tag: Option<WatchCaptureFrameTag>,
    map_signal: Arc<AtomicU8>,
    pending: bool,
}

impl StagingSlot {
    fn new() -> Self {
        Self {
            buffer: None,
            rgb_buf: Vec::new(),
            buf_size: 0,
            row_pitch: 0,
            cap_w: 0,
            cap_h: 0,
            grayscale: false,
            alpha_mask: false,
            metadata: None,
            dirty_rects: None,
            frame_tag: None,
            map_signal: Arc::new(AtomicU8::new(MAP_PENDING)),
            pending: false,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct ReadbackJob {
    slot_idx: usize,
    rgb_buf: Vec<u8>,
    buffer: Buffer,
    buf_size: u64,
    row_pitch: usize,
    cap_w: u32,
    cap_h: u32,
    grayscale: bool,
    alpha_mask: bool,
    metadata: Option<[f32; 10]>,
    dirty_rects: Option<WatchCaptureDirtyRects>,
    frame_tag: Option<WatchCaptureFrameTag>,
}

#[cfg(not(target_arch = "wasm32"))]
struct ReadbackResult {
    slot_idx: usize,
    buffer: Buffer,
    buf_size: u64,
    row_pitch: u32,
    cap_w: u32,
    cap_h: u32,
    grayscale: bool,
    alpha_mask: bool,
    metadata: Option<[f32; 10]>,
    dirty_rects: Option<WatchCaptureDirtyRects>,
    frame_tag: Option<WatchCaptureFrameTag>,
    rgb_buf: Vec<u8>,
    complete_us: u128,
    conversion_us: u128,
}

#[cfg(not(target_arch = "wasm32"))]
struct ReadbackWorker {
    jobs: Sender<ReadbackJob>,
    results: Receiver<ReadbackResult>,
}

#[cfg(not(target_arch = "wasm32"))]
impl ReadbackWorker {
    fn new() -> Self {
        let (job_tx, job_rx) = mpsc::channel::<ReadbackJob>();
        let (result_tx, result_rx) = mpsc::channel::<ReadbackResult>();
        thread::Builder::new()
            .name("egor-readback".to_owned())
            .spawn(move || {
                let mut source_cache = Vec::new();
                while let Ok(job) = job_rx.recv() {
                    let complete_start = Instant::now();
                    let mut rgb_buf = job.rgb_buf;
                    decode_readback_into(
                        &job.buffer,
                        &mut rgb_buf,
                        job.cap_w,
                        job.cap_h,
                        job.row_pitch,
                        job.grayscale,
                        job.alpha_mask,
                        &mut source_cache,
                    );
                    let conversion_us = complete_start.elapsed().as_micros();
                    let complete_us = complete_start.elapsed().as_micros();
                    let result = ReadbackResult {
                        slot_idx: job.slot_idx,
                        buffer: job.buffer,
                        buf_size: job.buf_size,
                        row_pitch: job.row_pitch as u32,
                        cap_w: job.cap_w,
                        cap_h: job.cap_h,
                        grayscale: job.grayscale,
                        alpha_mask: job.alpha_mask,
                        metadata: job.metadata,
                        dirty_rects: job.dirty_rects,
                        frame_tag: job.frame_tag,
                        rgb_buf,
                        complete_us,
                        conversion_us,
                    };
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }
            })
            .expect("failed to spawn egor readback worker");

        Self {
            jobs: job_tx,
            results: result_rx,
        }
    }
}

/// Asynchronous screen capture with GPU blit-downsample, a ring buffer of
/// staging buffers, and asynchronous readback through wgpu. Native conversion
/// runs on a worker. An occupied ring skips new requests instead of waiting
/// for the GPU or allowing the worker queue to grow.
///
/// Lifecycle:
///   1. Game calls [`ScreenCaptureState::request`] each frame it wants a
///      capture.
///   2. After all render passes, [`App`] calls
///      [`ScreenCaptureState::capture_from_texture`] which blits the
///      backbuffer into a small capture texture and encodes a
///      `copy_texture_to_buffer` into the next ring-buffer slot.
///   3. After `queue.submit()`, [`App`] calls
///      [`ScreenCaptureState::begin_readback_map`] to issue the async map.
///   4. On a subsequent frame the game polls
///      [`ScreenCaptureState::try_complete`]. Native builds hand ready slots
///      to a worker thread; wasm consumes the ready slot synchronously.
pub struct ScreenCaptureState {
    metrics: ScreenCaptureMetrics,
    buffers_released: bool,
    full_dirty_after_skip: bool,
    // -- GPU blit resources (lazily initialised) --
    blit_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    blit_gray_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    blit_gray_alpha_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    blit_encode_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    blit_gray_encode_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    blit_gray_alpha_encode_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    watch_capture_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    watch_capture_gray_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    watch_capture_encode_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    watch_capture_gray_encode_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    blit_sampler: Option<egor_render::wgpu::Sampler>,
    blit_bind_group_layout: Option<egor_render::wgpu::BindGroupLayout>,
    watch_capture_bind_group_layout: Option<egor_render::wgpu::BindGroupLayout>,
    watch_capture_uniform: Option<Buffer>,
    present_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    present_pipeline_format: Option<TextureFormat>,
    composite_pipeline: Option<egor_render::wgpu::RenderPipeline>,
    composite_pipeline_format: Option<TextureFormat>,
    // Intermediate copy of the backbuffer with TEXTURE_BINDING
    // (surface textures may lack TEXTURE_BINDING on some backends like DX12).
    source_copy: Option<Texture>,
    source_copy_view: Option<egor_render::wgpu::TextureView>,
    source_copy_w: u32,
    source_copy_h: u32,

    capture_texture: Option<Texture>,
    capture_view: Option<egor_render::wgpu::TextureView>,
    capture_tex_w: u32,
    capture_tex_h: u32,
    capture_tex_gray: bool,
    capture_tex_alpha_mask: bool,

    // -- request --
    requested: bool,
    capture_w: u32,
    capture_h: u32,
    pub grayscale: bool,
    pub alpha_mask: bool,
    source_render_target: Option<usize>,
    watch_overlay_capture: bool,
    final_frame_logical_w: u32,
    final_frame_logical_h: u32,
    final_frame_scale_factor: u32,
    request_metadata: Option<[f32; 10]>,
    request_dirty_rects: Option<WatchCaptureDirtyRects>,
    request_frame_tag: Option<WatchCaptureFrameTag>,

    // -- ring buffer of staging slots --
    slots: [StagingSlot; SLOT_COUNT],
    write_idx: usize,
    needs_map: Option<usize>,
    #[cfg(not(target_arch = "wasm32"))]
    readback_worker: Option<ReadbackWorker>,

    // -- completed result --
    result_ready: bool,
    result_w: u16,
    result_h: u16,
    result_metadata: Option<[f32; 10]>,
    result_dirty_rects: Option<WatchCaptureDirtyRects>,
    result_frame_tag: Option<WatchCaptureFrameTag>,
    rgb_buf: Vec<u8>,
    composite_render_target: Option<usize>,
}

impl ScreenCaptureState {
    pub fn new() -> Self {
        Self {
            metrics: ScreenCaptureMetrics::default(),
            buffers_released: false,
            full_dirty_after_skip: false,
            blit_pipeline: None,
            blit_gray_pipeline: None,
            blit_gray_alpha_pipeline: None,
            blit_encode_pipeline: None,
            blit_gray_encode_pipeline: None,
            blit_gray_alpha_encode_pipeline: None,
            watch_capture_pipeline: None,
            watch_capture_gray_pipeline: None,
            watch_capture_encode_pipeline: None,
            watch_capture_gray_encode_pipeline: None,
            blit_sampler: None,
            blit_bind_group_layout: None,
            watch_capture_bind_group_layout: None,
            watch_capture_uniform: None,
            present_pipeline: None,
            present_pipeline_format: None,
            composite_pipeline: None,
            composite_pipeline_format: None,
            source_copy: None,
            source_copy_view: None,
            source_copy_w: 0,
            source_copy_h: 0,
            capture_texture: None,
            capture_view: None,
            capture_tex_w: 0,
            capture_tex_h: 0,
            capture_tex_gray: false,
            capture_tex_alpha_mask: false,
            requested: false,
            capture_w: 0,
            capture_h: 0,
            grayscale: false,
            alpha_mask: false,
            source_render_target: None,
            watch_overlay_capture: false,
            final_frame_logical_w: 0,
            final_frame_logical_h: 0,
            final_frame_scale_factor: 1,
            request_metadata: None,
            request_dirty_rects: None,
            request_frame_tag: None,
            slots: [StagingSlot::new(), StagingSlot::new(), StagingSlot::new()],
            write_idx: 0,
            needs_map: None,
            #[cfg(not(target_arch = "wasm32"))]
            readback_worker: None,
            result_ready: false,
            result_w: 0,
            result_h: 0,
            result_metadata: None,
            result_dirty_rects: None,
            result_frame_tag: None,
            rgb_buf: Vec::new(),
            composite_render_target: None,
        }
    }

    /// Pure math — compute capture dimensions preserving aspect ratio.
    /// Width is quantised to a multiple of 8 so that ±1 px jitter in the
    /// source resolution (common on web with fractional devicePixelRatio)
    /// does not flip the capture size and force every frame to be a keyframe.
    pub fn capture_dims(screen_w: f32, screen_h: f32) -> (u32, u32) {
        let cap_h = CAP_HEIGHT;
        let cap_w = if screen_h > 0.0 {
            let raw = ((screen_w / screen_h) * cap_h as f32) as u32;
            // Round to nearest multiple of 8
            ((raw + 4) / 8) * 8
        } else {
            424 // 8-aligned default for ~16:9
        };
        (cap_w.max(8), cap_h)
    }

    /// Game calls this to request a capture at the given dimensions.
    pub fn request(&mut self, w: u32, h: u32, grayscale: bool) {
        self.requested = true;
        self.capture_w = w;
        self.capture_h = h;
        self.grayscale = grayscale;
        self.alpha_mask = false;
        self.source_render_target = None;
        self.watch_overlay_capture = false;
        self.request_metadata = None;
        self.request_dirty_rects = None;
        self.request_frame_tag = None;
    }

    pub fn request_with_alpha_mask(&mut self, w: u32, h: u32, grayscale: bool, source_render_target: usize, metadata: Option<[f32; 10]>) {
        self.requested = true;
        self.capture_w = w;
        self.capture_h = h;
        self.grayscale = grayscale;
        self.alpha_mask = true;
        self.source_render_target = Some(source_render_target);
        self.watch_overlay_capture = false;
        self.request_metadata = metadata;
        self.request_dirty_rects = None;
        self.request_frame_tag = None;
    }

    pub fn request_watch_overlay_capture(
        &mut self,
        w: u32,
        h: u32,
        logical_w: u32,
        logical_h: u32,
        scale_factor: u32,
        grayscale: bool,
        metadata: Option<[f32; 10]>,
        dirty_rects: Option<WatchCaptureDirtyRects>,
        frame_tag: Option<WatchCaptureFrameTag>,
    ) {
        self.requested = true;
        self.capture_w = w;
        self.capture_h = h;
        self.grayscale = grayscale;
        self.alpha_mask = true;
        self.source_render_target = None;
        self.watch_overlay_capture = true;
        self.final_frame_logical_w = logical_w.max(1);
        self.final_frame_logical_h = logical_h.max(1);
        self.final_frame_scale_factor = scale_factor.clamp(1, 8);
        self.request_metadata = metadata;
        self.request_dirty_rects = dirty_rects;
        self.request_frame_tag = frame_tag;
    }

    pub fn requested_source_render_target(&self) -> Option<usize> {
        if self.requested { self.source_render_target } else { None }
    }

    pub fn is_watch_overlay_capture_requested(&self) -> bool {
        self.requested && self.watch_overlay_capture
    }

    pub fn cancel_request(&mut self) {
        self.requested = false;
        self.source_render_target = None;
        self.watch_overlay_capture = false;
        self.request_metadata = None;
        self.request_dirty_rects = None;
        self.request_frame_tag = None;
    }

    pub fn request_composite_render_target(&mut self, source_render_target: usize) {
        self.composite_render_target = Some(source_render_target);
    }

    pub fn requested_composite_render_target(&self) -> Option<usize> {
        self.composite_render_target
    }

    pub fn take_composite_render_target(&mut self) -> Option<usize> {
        self.composite_render_target.take()
    }

    pub fn release_buffers(&mut self) {
        self.requested = false;
        self.source_render_target = None;
        self.watch_overlay_capture = false;
        self.composite_render_target = None;
        self.request_metadata = None;
        self.request_dirty_rects = None;
        self.request_frame_tag = None;
        self.capture_texture = None;
        self.capture_view = None;
        self.capture_tex_w = 0;
        self.capture_tex_h = 0;
        self.capture_tex_gray = false;
        self.capture_tex_alpha_mask = false;
        self.source_copy = None;
        self.source_copy_view = None;
        self.source_copy_w = 0;
        self.source_copy_h = 0;
        self.watch_capture_uniform = None;
        self.rgb_buf = Vec::new();
        self.slots = std::array::from_fn(|_| StagingSlot::new());
        self.write_idx = 0;
        self.needs_map = None;
        self.result_ready = false;
        self.result_metadata = None;
        self.result_dirty_rects = None;
        self.result_frame_tag = None;
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.readback_worker = None;
        }
        self.metrics.worker_jobs = 0;
        self.metrics.worker_bytes = 0;
        self.buffers_released = true;
        self.full_dirty_after_skip = false;
    }

    pub(crate) fn finish_private_warmup(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        let worker = self.readback_worker.take();
        self.release_buffers();
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.readback_worker = worker;
        }
    }

    pub(crate) fn take_buffers_released(&mut self) -> bool {
        std::mem::take(&mut self.buffers_released)
    }

    /// Returns `true` if a capture was requested this frame.
    pub fn is_requested(&self) -> bool {
        self.requested
    }

    pub fn metrics(&self) -> ScreenCaptureMetrics {
        let mut metrics = self.metrics;
        metrics.gpu_bytes = self
            .slots
            .iter()
            .filter_map(|slot| slot.buffer.as_ref())
            .map(Buffer::size)
            .sum::<u64>()
            + metrics.worker_bytes
            + self.capture_tex_w as u64
                * self.capture_tex_h as u64
                * if self.capture_tex_gray {
                    if self.capture_tex_alpha_mask { 2 } else { 1 }
                } else {
                    4
                }
            + self.source_copy_w as u64 * self.source_copy_h as u64 * 4;
        metrics.cpu_buffer_bytes =
            self.rgb_buf.capacity() as u64 + self.slots.iter().map(|slot| slot.rgb_buf.capacity() as u64).sum::<u64>();
        metrics
    }

    /// Returns `true` while a slot still needs a GPU mapping callback.
    /// Ready mappings and CPU worker jobs do not need device polling.
    pub fn readback_in_flight(&self) -> bool {
        self.slots
            .iter()
            .any(|s| s.pending && s.buffer.is_some() && s.map_signal.load(Ordering::Acquire) == MAP_PENDING)
    }

    // -- pipeline / resource setup (lazy) --------------------------------

    pub(crate) fn prewarm_watch_pipelines(&mut self, device: &Device, format: TextureFormat) {
        self.ensure_watch_capture_pipeline(device);
        self.ensure_present_pipeline(device, format);
    }

    fn ensure_pipeline(&mut self, device: &Device) {
        if self.blit_pipeline.is_some() {
            return;
        }

        let shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: None,
            source: egor_render::wgpu::ShaderSource::Wgsl(BLIT_SHADER_WGSL.into()),
        });

        let gray_shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: None,
            source: egor_render::wgpu::ShaderSource::Wgsl(BLIT_GRAY_SHADER_WGSL.into()),
        });

        let gray_alpha_shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: None,
            source: egor_render::wgpu::ShaderSource::Wgsl(BLIT_GRAY_ALPHA_SHADER_WGSL.into()),
        });

        let encode_shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: None,
            source: egor_render::wgpu::ShaderSource::Wgsl(BLIT_ENCODE_SHADER_WGSL.into()),
        });

        let gray_encode_shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: None,
            source: egor_render::wgpu::ShaderSource::Wgsl(BLIT_GRAY_ENCODE_SHADER_WGSL.into()),
        });

        let gray_alpha_encode_shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: None,
            source: egor_render::wgpu::ShaderSource::Wgsl(BLIT_GRAY_ALPHA_ENCODE_SHADER_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&egor_render::wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                egor_render::wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: egor_render::wgpu::ShaderStages::FRAGMENT,
                    ty: egor_render::wgpu::BindingType::Texture {
                        sample_type: egor_render::wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: egor_render::wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                egor_render::wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: egor_render::wgpu::ShaderStages::FRAGMENT,
                    ty: egor_render::wgpu::BindingType::Sampler(egor_render::wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&egor_render::wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let make_pipeline = |label, module: &egor_render::wgpu::ShaderModule, format| {
            device.create_render_pipeline(&egor_render::wgpu::RenderPipelineDescriptor {
                label,
                layout: Some(&pipeline_layout),
                vertex: egor_render::wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(egor_render::wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(egor_render::wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: egor_render::wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: egor_render::wgpu::PrimitiveState {
                    topology: egor_render::wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let pipeline = make_pipeline(None, &shader, TextureFormat::Rgba8Unorm);
        let gray_pipeline = make_pipeline(None, &gray_shader, TextureFormat::R8Unorm);
        let gray_alpha_pipeline = make_pipeline(None, &gray_alpha_shader, TextureFormat::Rg8Unorm);
        let encode_pipeline = make_pipeline(None, &encode_shader, TextureFormat::Rgba8Unorm);
        let gray_encode_pipeline = make_pipeline(None, &gray_encode_shader, TextureFormat::R8Unorm);
        let gray_alpha_encode_pipeline = make_pipeline(None, &gray_alpha_encode_shader, TextureFormat::Rg8Unorm);

        let sampler = device.create_sampler(&egor_render::wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: egor_render::wgpu::AddressMode::ClampToEdge,
            address_mode_v: egor_render::wgpu::AddressMode::ClampToEdge,
            mag_filter: egor_render::wgpu::FilterMode::Nearest,
            min_filter: egor_render::wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        self.blit_pipeline = Some(pipeline);
        self.blit_gray_pipeline = Some(gray_pipeline);
        self.blit_gray_alpha_pipeline = Some(gray_alpha_pipeline);
        self.blit_encode_pipeline = Some(encode_pipeline);
        self.blit_gray_encode_pipeline = Some(gray_encode_pipeline);
        self.blit_gray_alpha_encode_pipeline = Some(gray_alpha_encode_pipeline);
        self.blit_sampler = Some(sampler);
        self.blit_bind_group_layout = Some(bind_group_layout);
    }

    fn ensure_capture_texture(&mut self, device: &Device, w: u32, h: u32, grayscale: bool, alpha_mask: bool) {
        if self.capture_tex_w == w
            && self.capture_tex_h == h
            && self.capture_tex_gray == grayscale
            && self.capture_tex_alpha_mask == alpha_mask
            && self.capture_texture.is_some()
        {
            return;
        }

        let format = match (grayscale, alpha_mask) {
            (true, false) => TextureFormat::R8Unorm,
            (true, true) => TextureFormat::Rg8Unorm,
            (false, _) => TextureFormat::Rgba8Unorm,
        };

        let texture = device.create_texture(&egor_render::wgpu::TextureDescriptor {
            label: None,
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: egor_render::wgpu::TextureDimension::D2,
            format,
            usage: egor_render::wgpu::TextureUsages::RENDER_ATTACHMENT | egor_render::wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        self.capture_view = Some(texture.create_view(&Default::default()));
        self.capture_texture = Some(texture);
        self.capture_tex_w = w;
        self.capture_tex_h = h;
        self.capture_tex_gray = grayscale;
        self.capture_tex_alpha_mask = alpha_mask;
    }

    fn ensure_source_copy(&mut self, device: &Device, w: u32, h: u32, format: TextureFormat) {
        if self.source_copy_w == w && self.source_copy_h == h && self.source_copy.is_some() {
            return;
        }
        let tex = device.create_texture(&egor_render::wgpu::TextureDescriptor {
            label: None,
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: egor_render::wgpu::TextureDimension::D2,
            format,
            usage: egor_render::wgpu::TextureUsages::TEXTURE_BINDING | egor_render::wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.source_copy_view = Some(tex.create_view(&Default::default()));
        self.source_copy = Some(tex);
        self.source_copy_w = w;
        self.source_copy_h = h;
    }

    fn prepare_capture(&mut self) -> Option<PreparedCapture> {
        self.requested = false;

        let cap_w = self.capture_w.max(1);
        let cap_h = self.capture_h.max(1);
        let grayscale = self.grayscale;
        let alpha_mask = self.alpha_mask;
        self.source_render_target = None;

        // -- Ring-buffer slot availability check --------------------------
        // Done BEFORE the blit so we skip ALL GPU work when the ring is full.
        let idx = self.write_idx;
        if self.slots[idx].pending {
            let status = self.slots[idx].map_signal.load(Ordering::Acquire);
            if status == MAP_READY {
                self.metrics.skipped_requests += 1;
                self.full_dirty_after_skip = true;
                // Capture submission runs on the frame path. Do not consume
                // mapped readbacks here; the poll path/worker owns that work.
                return None;
            } else if status == MAP_FAILED {
                self.slots[idx].pending = false;
            } else {
                // Ring full — GPU hasn't finished this slot yet.
                // Skip capture entirely; no blit, no copy, no allocation.
                self.metrics.skipped_requests += 1;
                self.full_dirty_after_skip = true;
                return None;
            }
        }

        let dirty_rects = self.request_dirty_rects.take();
        // Hints may describe only the last requested frame. After a skipped
        // capture, scan the next complete image to include every missed change.
        let dirty_rects = if std::mem::take(&mut self.full_dirty_after_skip) {
            None
        } else {
            dirty_rects
        };
        Some(PreparedCapture {
            slot_idx: idx,
            cap_w,
            cap_h,
            grayscale,
            alpha_mask,
            metadata: self.request_metadata.take(),
            dirty_rects,
            frame_tag: self.request_frame_tag.take(),
        })
    }

    fn source_bind_group(&self, device: &Device, source_view: &egor_render::wgpu::TextureView) -> egor_render::wgpu::BindGroup {
        let bind_group_layout = self.blit_bind_group_layout.as_ref().expect("pipeline init");
        let sampler = self.blit_sampler.as_ref().expect("pipeline init");

        device.create_bind_group(&egor_render::wgpu::BindGroupDescriptor {
            label: None,
            layout: bind_group_layout,
            entries: &[
                egor_render::wgpu::BindGroupEntry {
                    binding: 0,
                    resource: egor_render::wgpu::BindingResource::TextureView(source_view),
                },
                egor_render::wgpu::BindGroupEntry {
                    binding: 1,
                    resource: egor_render::wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    fn ensure_watch_capture_pipeline(&mut self, device: &Device) {
        if self.watch_capture_pipeline.is_some() {
            return;
        }

        let shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: Some("Watch Capture Shader"),
            source: egor_render::wgpu::ShaderSource::Wgsl(WATCH_CAPTURE_SHADER_WGSL.into()),
        });
        let gray_shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: Some("Watch Capture Gray Shader"),
            source: egor_render::wgpu::ShaderSource::Wgsl(WATCH_CAPTURE_GRAY_SHADER_WGSL.into()),
        });
        let encode_shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: Some("Watch Capture Encode Shader"),
            source: egor_render::wgpu::ShaderSource::Wgsl(WATCH_CAPTURE_ENCODE_SHADER_WGSL.into()),
        });
        let gray_encode_shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: Some("Watch Capture Gray Encode Shader"),
            source: egor_render::wgpu::ShaderSource::Wgsl(WATCH_CAPTURE_GRAY_ENCODE_SHADER_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&egor_render::wgpu::BindGroupLayoutDescriptor {
            label: Some("Watch Capture Bind Group Layout"),
            entries: &[
                egor_render::wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: egor_render::wgpu::ShaderStages::FRAGMENT,
                    ty: egor_render::wgpu::BindingType::Texture {
                        sample_type: egor_render::wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: egor_render::wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                egor_render::wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: egor_render::wgpu::ShaderStages::FRAGMENT,
                    ty: egor_render::wgpu::BindingType::Buffer {
                        ty: egor_render::wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&egor_render::wgpu::PipelineLayoutDescriptor {
            label: Some("Watch Capture Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let make_pipeline = |label, module: &egor_render::wgpu::ShaderModule, format| {
            device.create_render_pipeline(&egor_render::wgpu::RenderPipelineDescriptor {
                label,
                layout: Some(&pipeline_layout),
                vertex: egor_render::wgpu::VertexState {
                    module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(egor_render::wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(egor_render::wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: egor_render::wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: egor_render::wgpu::PrimitiveState {
                    topology: egor_render::wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        self.watch_capture_pipeline = Some(make_pipeline(Some("Watch Capture"), &shader, TextureFormat::Rgba8Unorm));
        self.watch_capture_gray_pipeline = Some(make_pipeline(Some("Watch Capture Gray"), &gray_shader, TextureFormat::Rg8Unorm));
        self.watch_capture_encode_pipeline = Some(make_pipeline(
            Some("Watch Capture Encode"),
            &encode_shader,
            TextureFormat::Rgba8Unorm,
        ));
        self.watch_capture_gray_encode_pipeline = Some(make_pipeline(
            Some("Watch Capture Gray Encode"),
            &gray_encode_shader,
            TextureFormat::Rg8Unorm,
        ));
        self.watch_capture_bind_group_layout = Some(bind_group_layout);
    }

    fn watch_capture_bind_group(
        &mut self,
        device: &Device,
        queue: &egor_render::Queue,
        overlay_view: &egor_render::wgpu::TextureView,
        uniform: WatchCaptureUniform,
    ) -> egor_render::wgpu::BindGroup {
        self.ensure_watch_capture_pipeline(device);
        if self.watch_capture_uniform.is_none() {
            self.watch_capture_uniform = Some(device.create_buffer(&BufferDescriptor {
                label: Some("Watch Capture Uniform"),
                size: std::mem::size_of::<WatchCaptureUniform>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let uniform_buffer = self.watch_capture_uniform.as_ref().expect("watch capture uniform");
        queue.write_buffer(uniform_buffer, 0, uniform.bytes());
        let layout = self.watch_capture_bind_group_layout.as_ref().expect("watch capture layout");

        device.create_bind_group(&egor_render::wgpu::BindGroupDescriptor {
            label: Some("Watch Capture Bind Group"),
            layout,
            entries: &[
                egor_render::wgpu::BindGroupEntry {
                    binding: 0,
                    resource: egor_render::wgpu::BindingResource::TextureView(overlay_view),
                },
                egor_render::wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }

    fn capture_prepared_bind_group(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        prepared: PreparedCapture,
        encode_srgb: bool,
        bind_group: &egor_render::wgpu::BindGroup,
    ) {
        let PreparedCapture {
            slot_idx,
            cap_w,
            cap_h,
            grayscale,
            alpha_mask,
            metadata,
            dirty_rects,
            frame_tag,
        } = prepared;
        // Ensure GPU resources exist
        self.ensure_pipeline(device);
        self.ensure_capture_texture(device, cap_w, cap_h, grayscale, alpha_mask);

        let pipeline = match (grayscale, alpha_mask, encode_srgb) {
            (true, false, false) => self.blit_gray_pipeline.as_ref().expect("pipeline init"),
            (true, false, true) => self.blit_gray_encode_pipeline.as_ref().expect("pipeline init"),
            (true, true, false) => self.blit_gray_alpha_pipeline.as_ref().expect("pipeline init"),
            (true, true, true) => self.blit_gray_alpha_encode_pipeline.as_ref().expect("pipeline init"),
            (false, _, false) => self.blit_pipeline.as_ref().expect("pipeline init"),
            (false, _, true) => self.blit_encode_pipeline.as_ref().expect("pipeline init"),
        };
        let capture_view = self.capture_view.as_ref().expect("capture texture init");

        // Blit render pass: draw fullscreen triangle sampling the source.
        {
            let mut rpass = encoder.begin_render_pass(&egor_render::wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(egor_render::wgpu::RenderPassColorAttachment {
                    view: capture_view,
                    resolve_target: None,
                    ops: egor_render::wgpu::Operations {
                        load: egor_render::wgpu::LoadOp::Clear(egor_render::wgpu::Color::BLACK),
                        store: egor_render::wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(pipeline);
            rpass.set_bind_group(0, bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        // Copy capture texture → staging buffer for CPU readback
        let slot = &mut self.slots[slot_idx];
        let bytes_per_pixel: u32 = match (grayscale, alpha_mask) {
            (true, false) => 1,
            (true, true) => 2,
            (false, _) => 4,
        };
        let unpadded_row = cap_w * bytes_per_pixel;
        let align = egor_render::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (unpadded_row + align - 1) / align * align;
        let buffer_size = (padded_row * cap_h) as u64;

        // Reuse the staging buffer if dimensions haven't changed
        let needs_new = slot.buffer.is_none() || slot.buf_size != buffer_size || slot.row_pitch != padded_row;

        self.metrics.readback_bytes += buffer_size;
        if needs_new {
            self.metrics.staging_allocations += 1;
            self.metrics.staging_allocated_bytes += buffer_size;
            slot.buffer = Some(device.create_buffer(&BufferDescriptor {
                label: None,
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
            slot.buf_size = buffer_size;
            slot.row_pitch = padded_row;
        }

        let buffer = slot.buffer.as_ref().expect("staging buffer");

        encoder.copy_texture_to_buffer(
            egor_render::wgpu::TexelCopyTextureInfo {
                texture: self.capture_texture.as_ref().expect("capture texture"),
                mip_level: 0,
                origin: egor_render::wgpu::Origin3d::ZERO,
                aspect: egor_render::wgpu::TextureAspect::All,
            },
            egor_render::wgpu::TexelCopyBufferInfo {
                buffer,
                layout: egor_render::wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(cap_h),
                },
            },
            Extent3d {
                width: cap_w,
                height: cap_h,
                depth_or_array_layers: 1,
            },
        );

        // Store capture metadata on the slot so try_complete knows how to
        // decode regardless of whether the request params changed since then.
        let slot = &mut self.slots[slot_idx];
        slot.cap_w = cap_w;
        slot.cap_h = cap_h;
        slot.grayscale = grayscale;
        slot.alpha_mask = alpha_mask;
        slot.metadata = metadata;
        slot.dirty_rects = dirty_rects;
        slot.frame_tag = frame_tag;
        slot.pending = true;

        self.needs_map = Some(slot_idx);
        self.write_idx = (self.write_idx + 1) % SLOT_COUNT;
    }

    // -- capture entry point (called by app.rs after rendering) -----------

    /// Blit-downsample the backbuffer into a small capture texture, then
    /// encode a `copy_texture_to_buffer` for async readback.
    ///
    /// `source` is the backbuffer `Texture` (must have `COPY_SRC` usage).
    /// The encoder must be the same one that will be submitted this frame.
    pub fn capture_from_texture(&mut self, device: &Device, encoder: &mut CommandEncoder, source: &Texture) {
        let Some(prepared) = self.prepare_capture() else {
            return;
        };

        self.ensure_pipeline(device);

        // The surface texture may not support TEXTURE_BINDING (e.g. DX12),
        // so copy it to an intermediate texture that does.
        let src_size = source.size();
        self.ensure_source_copy(device, src_size.width, src_size.height, source.format());
        encoder.copy_texture_to_texture(
            egor_render::wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: egor_render::wgpu::Origin3d::ZERO,
                aspect: egor_render::wgpu::TextureAspect::All,
            },
            egor_render::wgpu::TexelCopyTextureInfo {
                texture: self.source_copy.as_ref().unwrap(),
                mip_level: 0,
                origin: egor_render::wgpu::Origin3d::ZERO,
                aspect: egor_render::wgpu::TextureAspect::All,
            },
            Extent3d {
                width: src_size.width,
                height: src_size.height,
                depth_or_array_layers: 1,
            },
        );

        let source_view = self.source_copy_view.as_ref().expect("source copy init");
        let bind_group = self.source_bind_group(device, source_view);
        self.capture_prepared_bind_group(device, encoder, prepared, source.format().is_srgb(), &bind_group);
    }

    /// Capture from an already bindable render target. Used on surfaces that
    /// cannot be configured with `COPY_SRC`; avoids a full-resolution texture
    /// copy before the downsample pass.
    pub fn capture_from_sampled_view(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        source_view: &egor_render::wgpu::TextureView,
        encode_srgb: bool,
    ) {
        let Some(prepared) = self.prepare_capture() else {
            return;
        };

        self.ensure_pipeline(device);
        let bind_group = self.source_bind_group(device, source_view);
        self.capture_prepared_bind_group(device, encoder, prepared, encode_srgb, &bind_group);
    }

    pub fn capture_from_watch_overlay(
        &mut self,
        device: &Device,
        queue: &egor_render::Queue,
        encoder: &mut CommandEncoder,
        overlay_view: &egor_render::wgpu::TextureView,
        source_w: u32,
        source_h: u32,
        encode_srgb: bool,
    ) {
        let logical_w = self.final_frame_logical_w.max(1);
        let logical_h = self.final_frame_logical_h.max(1);
        let factor = self.final_frame_scale_factor.clamp(1, 8);
        let Some(prepared) = self.prepare_capture() else {
            return;
        };
        let PreparedCapture {
            slot_idx,
            cap_w,
            cap_h,
            grayscale,
            alpha_mask,
            metadata,
            dirty_rects,
            frame_tag,
        } = prepared;

        self.ensure_watch_capture_pipeline(device);
        self.ensure_capture_texture(device, cap_w, cap_h, grayscale, alpha_mask);

        let uniform = WatchCaptureUniform {
            source_w: source_w.max(1),
            source_h: source_h.max(1),
            logical_w,
            logical_h,
            factor,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let bind_group = self.watch_capture_bind_group(device, queue, overlay_view, uniform);
        let pipeline = match (grayscale, encode_srgb) {
            (true, false) => self.watch_capture_gray_pipeline.as_ref().expect("watch capture pipeline"),
            (true, true) => self.watch_capture_gray_encode_pipeline.as_ref().expect("watch capture pipeline"),
            (false, false) => self.watch_capture_pipeline.as_ref().expect("watch capture pipeline"),
            (false, true) => self.watch_capture_encode_pipeline.as_ref().expect("watch capture pipeline"),
        };
        let capture_view = self.capture_view.as_ref().expect("capture texture init");

        {
            let mut rpass = encoder.begin_render_pass(&egor_render::wgpu::RenderPassDescriptor {
                label: Some("Watch Capture Downsample Pass"),
                color_attachments: &[Some(egor_render::wgpu::RenderPassColorAttachment {
                    view: capture_view,
                    resolve_target: None,
                    ops: egor_render::wgpu::Operations {
                        load: egor_render::wgpu::LoadOp::Clear(egor_render::wgpu::Color::BLACK),
                        store: egor_render::wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        let slot = &mut self.slots[slot_idx];
        let bytes_per_pixel: u32 = match (grayscale, alpha_mask) {
            (true, false) => 1,
            (true, true) => 2,
            (false, _) => 4,
        };
        let unpadded_row = cap_w * bytes_per_pixel;
        let align = egor_render::wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (unpadded_row + align - 1) / align * align;
        let buffer_size = (padded_row * cap_h) as u64;

        let needs_new = slot.buffer.is_none() || slot.buf_size != buffer_size || slot.row_pitch != padded_row;
        self.metrics.readback_bytes += buffer_size;
        if needs_new {
            self.metrics.staging_allocations += 1;
            self.metrics.staging_allocated_bytes += buffer_size;
            slot.buffer = Some(device.create_buffer(&BufferDescriptor {
                label: None,
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }));
            slot.buf_size = buffer_size;
            slot.row_pitch = padded_row;
        }

        let buffer = slot.buffer.as_ref().expect("staging buffer");
        encoder.copy_texture_to_buffer(
            egor_render::wgpu::TexelCopyTextureInfo {
                texture: self.capture_texture.as_ref().expect("capture texture"),
                mip_level: 0,
                origin: egor_render::wgpu::Origin3d::ZERO,
                aspect: egor_render::wgpu::TextureAspect::All,
            },
            egor_render::wgpu::TexelCopyBufferInfo {
                buffer,
                layout: egor_render::wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(cap_h),
                },
            },
            Extent3d {
                width: cap_w,
                height: cap_h,
                depth_or_array_layers: 1,
            },
        );

        let slot = &mut self.slots[slot_idx];
        slot.cap_w = cap_w;
        slot.cap_h = cap_h;
        slot.grayscale = grayscale;
        slot.alpha_mask = alpha_mask;
        slot.metadata = metadata;
        slot.dirty_rects = dirty_rects;
        slot.frame_tag = frame_tag;
        slot.pending = true;

        self.needs_map = Some(slot_idx);
        self.write_idx = (self.write_idx + 1) % SLOT_COUNT;
    }

    fn ensure_present_pipeline(&mut self, device: &Device, format: TextureFormat) {
        if self.present_pipeline.is_some() && self.present_pipeline_format == Some(format) {
            return;
        }

        self.ensure_pipeline(device);

        let shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: None,
            source: egor_render::wgpu::ShaderSource::Wgsl(BLIT_SHADER_WGSL.into()),
        });
        let bind_group_layout = self.blit_bind_group_layout.as_ref().expect("pipeline init");
        let pipeline_layout = device.create_pipeline_layout(&egor_render::wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(bind_group_layout)],
            immediate_size: 0,
        });

        self.present_pipeline = Some(device.create_render_pipeline(&egor_render::wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: egor_render::wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(egor_render::wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(egor_render::wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: egor_render::wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: egor_render::wgpu::PrimitiveState {
                topology: egor_render::wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        }));
        self.present_pipeline_format = Some(format);
    }

    fn ensure_composite_pipeline(&mut self, device: &Device, format: TextureFormat) {
        if self.composite_pipeline.is_some() && self.composite_pipeline_format == Some(format) {
            return;
        }

        self.ensure_pipeline(device);

        let shader = device.create_shader_module(egor_render::wgpu::ShaderModuleDescriptor {
            label: None,
            source: egor_render::wgpu::ShaderSource::Wgsl(BLIT_SHADER_WGSL.into()),
        });
        let bind_group_layout = self.blit_bind_group_layout.as_ref().expect("pipeline init");
        let pipeline_layout = device.create_pipeline_layout(&egor_render::wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(bind_group_layout)],
            immediate_size: 0,
        });

        self.composite_pipeline = Some(device.create_render_pipeline(&egor_render::wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: egor_render::wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(egor_render::wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(egor_render::wgpu::ColorTargetState {
                    format,
                    blend: Some(egor_render::wgpu::BlendState {
                        color: egor_render::wgpu::BlendComponent {
                            src_factor: egor_render::wgpu::BlendFactor::One,
                            dst_factor: egor_render::wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: egor_render::wgpu::BlendOperation::Add,
                        },
                        alpha: egor_render::wgpu::BlendComponent {
                            src_factor: egor_render::wgpu::BlendFactor::One,
                            dst_factor: egor_render::wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: egor_render::wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: egor_render::wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: egor_render::wgpu::PrimitiveState {
                topology: egor_render::wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        }));
        self.composite_pipeline_format = Some(format);
    }

    /// Present an offscreen frame source into the actual surface with a
    /// fullscreen triangle. This is a render pass, not a texture copy, so it
    /// works on GLES/WebGL surfaces that only support color attachment usage.
    pub fn present_sampled_view(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        source_view: &egor_render::wgpu::TextureView,
        dest_view: &egor_render::wgpu::TextureView,
        dest_format: TextureFormat,
    ) {
        self.ensure_present_pipeline(device, dest_format);
        let bind_group = self.source_bind_group(device, source_view);
        let pipeline = self.present_pipeline.as_ref().expect("present pipeline");

        let mut rpass = encoder.begin_render_pass(&egor_render::wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(egor_render::wgpu::RenderPassColorAttachment {
                view: dest_view,
                resolve_target: None,
                ops: egor_render::wgpu::Operations {
                    load: egor_render::wgpu::LoadOp::Clear(egor_render::wgpu::Color::BLACK),
                    store: egor_render::wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }

    pub fn composite_sampled_view(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        source_view: &egor_render::wgpu::TextureView,
        dest_view: &egor_render::wgpu::TextureView,
        dest_format: TextureFormat,
    ) {
        self.ensure_composite_pipeline(device, dest_format);
        let bind_group = self.source_bind_group(device, source_view);
        let pipeline = self.composite_pipeline.as_ref().expect("composite pipeline");

        let mut rpass = encoder.begin_render_pass(&egor_render::wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(egor_render::wgpu::RenderPassColorAttachment {
                view: dest_view,
                resolve_target: None,
                ops: egor_render::wgpu::Operations {
                    load: egor_render::wgpu::LoadOp::Load,
                    store: egor_render::wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }

    /// Start the async map request for the most recently written slot.
    /// **Must** be called AFTER `queue.submit()` — wgpu rejects submitting
    /// commands that reference a buffer with a pending map operation.
    pub fn begin_readback_map(&mut self) {
        let Some(idx) = self.needs_map.take() else {
            return;
        };
        let slot = &mut self.slots[idx];
        if !slot.pending {
            return;
        }
        let Some(buffer) = slot.buffer.as_ref() else {
            return;
        };
        slot.map_signal.store(MAP_PENDING, Ordering::Release);
        let signal = slot.map_signal.clone();
        buffer.slice(..).map_async(egor_render::wgpu::MapMode::Read, move |result| {
            signal.store(if result.is_ok() { MAP_READY } else { MAP_FAILED }, Ordering::Release);
        });
    }

    // -- readback polling ------------------------------------------------

    /// Consume the oldest completed ring-buffer slot. For grayscale the
    /// staging buffer already holds R8 data from the GPU — just strip row
    /// padding. For RGB565, convert BGRA→RGB565 with unsafe pointer math.
    #[cfg(target_arch = "wasm32")]
    fn complete_slot(&mut self, idx: usize) {
        let complete_start = Instant::now();
        let cap_w = self.slots[idx].cap_w;
        let cap_h = self.slots[idx].cap_h;
        let row_pitch = self.slots[idx].row_pitch as usize;
        let grayscale = self.slots[idx].grayscale;
        let alpha_mask = self.slots[idx].alpha_mask;

        let buffer = match self.slots[idx].buffer.take() {
            Some(b) => b,
            None => {
                self.slots[idx].pending = false;
                return;
            }
        };

        decode_readback_into(
            &buffer,
            &mut self.rgb_buf,
            cap_w,
            cap_h,
            row_pitch,
            grayscale,
            alpha_mask,
            &mut Vec::new(),
        );
        let conversion_us = complete_start.elapsed().as_micros() as u64;
        buffer.unmap();
        let complete_us = complete_start.elapsed().as_micros() as u64;
        self.metrics.completed_frames += 1;
        self.metrics.decode_us += complete_us;
        self.metrics.conversion_us += conversion_us;
        self.metrics.unmap_us += complete_us - conversion_us;

        self.slots[idx].buffer = Some(buffer);
        self.slots[idx].pending = false;
        self.result_ready = true;
        self.result_w = cap_w as u16;
        self.result_h = cap_h as u16;
        self.result_metadata = self.slots[idx].metadata.take();
        self.result_dirty_rects = self.slots[idx].dirty_rects.take();
        self.result_frame_tag = self.slots[idx].frame_tag.take();
        // log::info!(
        //     target: "watchperf",
        //     "[watchperf] egor_complete slot={} size={}x{} grayscale={} alpha={} row_pitch={} output_bytes={} complete_us={}",
        //     idx,
        //     cap_w,
        //     cap_h,
        //     grayscale,
        //     alpha_mask,
        //     row_pitch,
        //     self.rgb_buf.len(),
        //     complete_start.elapsed().as_micros(),
        // );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn dispatch_slot_to_worker(&mut self, idx: usize) {
        let job = {
            let slot = &mut self.slots[idx];
            let Some(buffer) = slot.buffer.take() else {
                slot.pending = false;
                return;
            };

            ReadbackJob {
                slot_idx: idx,
                rgb_buf: std::mem::take(&mut slot.rgb_buf),
                buffer,
                buf_size: slot.buf_size,
                row_pitch: slot.row_pitch as usize,
                cap_w: slot.cap_w,
                cap_h: slot.cap_h,
                grayscale: slot.grayscale,
                alpha_mask: slot.alpha_mask,
                metadata: slot.metadata.take(),
                dirty_rects: slot.dirty_rects.take(),
                frame_tag: slot.frame_tag.take(),
            }
        };

        let job_bytes = job.buf_size;
        let worker = self.readback_worker.get_or_insert_with(ReadbackWorker::new);
        match worker.jobs.send(job) {
            Ok(()) => {
                self.metrics.worker_jobs += 1;
                self.metrics.worker_bytes += job_bytes;
                // Keep this slot occupied until collection. This bounds jobs
                // and staging buffers to SLOT_COUNT when conversion is slow.
            }
            Err(err) => {
                let job = err.0;
                let complete_start = Instant::now();
                let cap_w = job.cap_w;
                let cap_h = job.cap_h;
                let grayscale = job.grayscale;
                let alpha_mask = job.alpha_mask;
                let row_pitch = job.row_pitch;
                let metadata = job.metadata;
                decode_readback_into(
                    &job.buffer,
                    &mut self.rgb_buf,
                    cap_w,
                    cap_h,
                    row_pitch,
                    grayscale,
                    alpha_mask,
                    &mut Vec::new(),
                );
                job.buffer.unmap();
                self.slots[idx].buffer = Some(job.buffer);
                self.slots[idx].pending = false;
                self.result_ready = true;
                self.result_w = cap_w as u16;
                self.result_h = cap_h as u16;
                self.result_metadata = metadata;
                self.result_dirty_rects = job.dirty_rects;
                self.result_frame_tag = job.frame_tag;
                // log::warn!(
                //     target: "watchperf",
                //     "[watchperf] egor_complete_worker_fallback slot={} size={}x{} grayscale={} alpha={} row_pitch={} output_bytes={} complete_us={}",
                //     idx,
                //     self.result_w,
                //     self.result_h,
                //     grayscale,
                //     alpha_mask,
                //     row_pitch,
                //     self.rgb_buf.len(),
                //     complete_start.elapsed().as_micros(),
                // );
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn collect_worker_result(&mut self) {
        if self.result_ready {
            return;
        }

        let Some(worker) = self.readback_worker.as_ref() else {
            return;
        };
        let Ok(result) = worker.results.try_recv() else {
            return;
        };

        // Unmapping GL buffers acquires the rendering context. Keep that on
        // the render thread so a CPU conversion worker cannot steal/hold the
        // context while the frame thread is polling, submitting, or presenting.
        let unmap_started = Instant::now();
        result.buffer.unmap();
        let unmap_us = unmap_started.elapsed().as_micros() as u64;

        if let Some(slot) = self.slots.get_mut(result.slot_idx)
            && slot.pending
            && slot.buffer.is_none()
            && slot.buf_size == result.buf_size
            && slot.row_pitch == result.row_pitch
        {
            slot.buffer = Some(result.buffer);
            slot.pending = false;
            slot.rgb_buf = std::mem::replace(&mut self.rgb_buf, result.rgb_buf);
        }

        self.metrics.worker_jobs -= 1;
        self.metrics.worker_bytes -= result.buf_size;
        self.metrics.completed_frames += 1;
        self.metrics.decode_us += result.complete_us as u64 + unmap_us;
        self.metrics.conversion_us += result.conversion_us as u64;
        self.metrics.unmap_us += unmap_us;
        self.result_ready = true;
        self.result_w = result.cap_w as u16;
        self.result_h = result.cap_h as u16;
        self.result_metadata = result.metadata;
        self.result_dirty_rects = result.dirty_rects;
        self.result_frame_tag = result.frame_tag;
        // log::info!(
        //     target: "watchperf",
        //     "[watchperf] egor_complete_worker slot={} size={}x{} grayscale={} alpha={} row_pitch={} output_bytes={} worker_complete_us={}",
        //     result.slot_idx,
        //     result.cap_w,
        //     result.cap_h,
        //     result.grayscale,
        //     result.alpha_mask,
        //     result.row_pitch,
        //     self.rgb_buf.len(),
        //     result.complete_us,
        // );
    }

    /// Poll for a completed readback. Returns `Some((width, height))` when
    /// pixel data is available in [`Self::rgb_buf`].
    ///
    /// Non-blocking: iterates ring-buffer slots oldest-first and consumes
    /// the first whose `map_async` callback has fired. Driven by the game
    /// loop's existing `device.poll(PollType::Poll)`.
    pub fn try_complete(
        &mut self,
    ) -> Option<(
        u16,
        u16,
        Option<[f32; 10]>,
        Option<WatchCaptureDirtyRects>,
        Option<WatchCaptureFrameTag>,
    )> {
        #[cfg(not(target_arch = "wasm32"))]
        self.collect_worker_result();

        if self.result_ready {
            self.result_ready = false;
            return Some((
                self.result_w,
                self.result_h,
                self.result_metadata.take(),
                self.result_dirty_rects.take(),
                self.result_frame_tag.take(),
            ));
        }

        // Iterate oldest → newest (write_idx is the next write position,
        // which wraps to the oldest pending slot).
        for i in 0..SLOT_COUNT {
            let idx = (self.write_idx + i) % SLOT_COUNT;
            if !self.slots[idx].pending {
                continue;
            }
            if self.slots[idx].buffer.is_none() {
                continue;
            } // Owned by the worker.
            let status = self.slots[idx].map_signal.load(Ordering::Acquire);
            if status == MAP_PENDING {
                break; // Preserve frame order even if callbacks arrive out of order.
            }
            if status == MAP_FAILED {
                eprintln!("[ScreenCapture] buffer map failed on slot {idx}");
                self.slots[idx].pending = false;
                continue;
            }
            // MAP_READY
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.dispatch_slot_to_worker(idx);
                self.collect_worker_result();
                if self.result_ready {
                    self.result_ready = false;
                    return Some((
                        self.result_w,
                        self.result_h,
                        self.result_metadata.take(),
                        self.result_dirty_rects.take(),
                        self.result_frame_tag.take(),
                    ));
                }
                continue;
            }

            #[cfg(target_arch = "wasm32")]
            self.complete_slot(idx);
            #[cfg(target_arch = "wasm32")]
            if self.result_ready {
                self.result_ready = false;
                return Some((
                    self.result_w,
                    self.result_h,
                    self.result_metadata.take(),
                    self.result_dirty_rects.take(),
                    self.result_frame_tag.take(),
                ));
            }
        }

        None
    }

    /// Access the completed RGB pixel buffer.
    pub fn rgb_buf(&self) -> &[u8] {
        &self.rgb_buf
    }
}

impl Default for ScreenCaptureState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Graphics
// ---------------------------------------------------------------------------

/// High-level 2D drawing interface that simplifies the [`Renderer`]
pub struct Graphics<'a> {
    renderer: &'a mut Renderer,
    batch: &'a mut PrimitiveBatch,
    camera: Camera,
    text_renderer: &'a mut TextRenderer,
    target_format: TextureFormat,
    target_size: (u32, u32),
    current_shader: Option<usize>,
    render_targets: &'a mut RenderTargetStore,
    screen_capture: &'a mut ScreenCaptureState,
    offscreen_batches: &'a mut Vec<PrimitiveBatch>,
}

impl<'a> Graphics<'a> {
    /// Create `Graphics` with [`Renderer`], [`TextRenderer`] & `TextureFormat`
    pub fn new(
        renderer: &'a mut Renderer,
        batch: &'a mut PrimitiveBatch,
        text_renderer: &'a mut TextRenderer,
        render_targets: &'a mut RenderTargetStore,
        screen_capture: &'a mut ScreenCaptureState,
        offscreen_batches: &'a mut Vec<PrimitiveBatch>,
        format: TextureFormat,
        w: u32,
        h: u32,
    ) -> Self {
        Self {
            renderer,
            batch,
            camera: Camera::default(),
            text_renderer,
            target_format: format,
            target_size: (w, h),
            current_shader: None,
            render_targets,
            screen_capture,
            offscreen_batches,
        }
    }

    /// Create a new offscreen render target
    pub fn create_offscreen(&self, width: u32, height: u32) -> OffscreenTarget {
        self.renderer.create_offscreen_target(width, height, self.target_format)
    }

    /// Return information about the active GPU adapter/backend.
    pub fn adapter_info(&self) -> AdapterInfo {
        self.renderer.adapter_info()
    }

    /// Render to an offscreen target
    pub fn render_offscreen(&mut self, target: &mut OffscreenTarget, render_fn: impl FnMut(&mut Graphics)) {
        self.render_offscreen_with_limits(
            target,
            GeometryBatch::DEFAULT_MAX_VERTICES,
            GeometryBatch::DEFAULT_MAX_INDICES,
            render_fn,
        );
    }

    /// Render to an offscreen target using a temporary batch with custom vertex/index buffer limits.
    /// Use this when the default limits are too large for memory-constrained platforms,
    /// or too small for complex offscreen scenes.
    /// For most cases, prefer [`Self::render_offscreen`] which uses sensible defaults
    pub fn render_offscreen_with_limits(
        &mut self,
        target: &mut OffscreenTarget,
        max_verts: usize,
        max_indices: usize,
        mut render_fn: impl FnMut(&mut Graphics),
    ) {
        let (w, h) = target.size();
        let format = target.format();

        let mut offscreen_batch = self
            .offscreen_batches
            .pop()
            .unwrap_or_else(|| PrimitiveBatch::new(max_verts, max_indices));
        offscreen_batch.reset();
        let mut offscreen_gfx = Graphics {
            renderer: self.renderer,
            batch: &mut offscreen_batch,
            camera: Camera::default(),
            text_renderer: self.text_renderer,
            target_size: (w, h),
            target_format: format,
            current_shader: None,
            render_targets: self.render_targets,
            screen_capture: self.screen_capture,
            offscreen_batches: self.offscreen_batches,
        };

        render_fn(&mut offscreen_gfx);
        offscreen_gfx.upload_camera();
        let mut geometry = offscreen_batch.drain_all();

        let mut encoder = self.renderer.device().create_command_encoder(&Default::default());

        {
            let mut r_pass =
                self.renderer
                    .begin_render_pass_with_depth(&mut encoder, target.render_view(), target.offscreen_depth_view(), true);

            let mut cur_tex: Option<usize> = None;
            let mut cur_shd: Option<usize> = None;
            let mut cur_replace_blend = false;
            let mut cur_cam_offset = u32::MAX;
            let mut quad_bound = false;
            let mut cur_scissor = None;

            if let Some(first) = geometry.first() {
                self.renderer
                    .bind_pass_state(&mut r_pass, first.texture_id, first.shader_id, first.replace_blend);
                cur_tex = first.texture_id;
                cur_shd = first.shader_id;
                cur_replace_blend = first.replace_blend;
                quad_bound = true;
            }
            for entry in &mut geometry {
                if entry.scissor != cur_scissor {
                    if let Some(rect) = entry.scissor {
                        let sx = rect.0.min(w.saturating_sub(1));
                        let sy = rect.1.min(h.saturating_sub(1));
                        let sw = rect.2.min(w - sx).max(1);
                        let sh = rect.3.min(h - sy).max(1);
                        r_pass.set_scissor_rect(sx, sy, sw, sh);
                    } else {
                        r_pass.set_scissor_rect(0, 0, w, h);
                    }
                    cur_scissor = entry.scissor;
                }
                self.renderer.draw_batch(
                    &mut r_pass,
                    &mut entry.geometry,
                    entry.texture_id,
                    entry.shader_id,
                    entry.replace_blend,
                    0,
                    &mut cur_tex,
                    &mut cur_shd,
                    &mut cur_replace_blend,
                    &mut cur_cam_offset,
                    &mut quad_bound,
                );
            }
        }

        target.copy_to_sample(&mut encoder);

        let _ = self.renderer.queue().submit(Some(encoder.finish()));

        offscreen_batch.recycle(geometry);
        self.offscreen_batches.push(offscreen_batch);
    }

    /// Use an offscreen target as a texture
    pub fn offscreen_as_texture(&mut self, target: &mut OffscreenTarget) -> usize {
        self.renderer.add_offscreen_texture(target)
    }

    pub(crate) fn set_target_size(&mut self, w: u32, h: u32) {
        self.target_size = (w, h);
    }

    pub fn target_size(&self) -> (u32, u32) {
        self.target_size
    }

    /// Upload camera matrix to the GPU.
    /// Call after user drawing is complete and before the render pass
    pub(crate) fn upload_camera(&mut self) {
        let (w, h) = self.target_size;
        self.renderer
            .upload_camera_matrix(self.camera.view_proj((w as f32, h as f32).into()).to_cols_array_2d());
    }

    /// Clear the screen to a color
    pub fn clear(&mut self, color: Color) {
        self.renderer.set_clear_color(color.into());
    }
    /// Get current surface size in pixels
    pub fn screen_size(&self) -> Vec2 {
        let (w, h) = self.target_size;
        (w as f32, h as f32).into()
    }
    /// Mutable access to [`Camera`]
    pub fn camera(&mut self) -> &mut Camera {
        &mut self.camera
    }

    /// Start building a rectangle primitive
    pub fn rect(&mut self) -> RectangleBuilder<'_> {
        RectangleBuilder::new(self.batch, self.current_shader)
    }

    /// Push an axis-aligned, unrotated, colored sprite instance directly into
    /// the batch — bypasses RectangleBuilder, Mat2::from_angle, and builder
    /// overhead.  Used by the optimised `draw_cmd` fast-path for all
    /// non-rotated sprite draws (entities, UI, health bars, etc.).
    #[inline(always)]
    pub fn push_sprite(&mut self, tex_id: usize, x: f32, y: f32, w: f32, h: f32, uvs: [f32; 4], color: [f32; 4]) {
        self.batch.push_instance(
            egor_render::instance::Instance::new([w, 0.0, 0.0, h], [x + w * 0.5, y + h * 0.5, self.batch.draw_depth()], color, uvs)
                .with_watch_overlay(self.batch.watch_overlay()),
            Some(tex_id),
            self.current_shader,
        );
    }

    /// Push an axis-aligned, unrotated, WHITE sprite instance directly into the
    /// batch.  Designed for the tile-map hot loop.
    #[inline(always)]
    pub fn push_sprite_aa(&mut self, tex_id: usize, x: f32, y: f32, w: f32, h: f32, uvs: [f32; 4]) {
        self.push_sprite(tex_id, x, y, w, h, uvs, [1.0, 1.0, 1.0, 1.0]);
    }

    /// Ensure a batch exists for the given tile texture. Call once per atlas
    /// texture change, then use [`push_tile`] for the actual tile instances.
    #[inline(always)]
    pub fn ensure_tile_batch(&mut self, tex_id: usize) {
        self.batch.ensure_batch(Some(tex_id), self.current_shader);
    }

    /// Push a tile instance directly into the current batch, skipping all
    /// batch-key comparisons. The caller MUST call [`ensure_tile_batch`]
    /// first whenever the texture changes.
    #[inline(always)]
    pub fn push_tile(&mut self, x: f32, y: f32, w: f32, h: f32, depth: f32, uvs: [f32; 4]) {
        self.batch.push_instance_unchecked(
            egor_render::instance::Instance::new([w, 0.0, 0.0, h], [x + w * 0.5, y + h * 0.5, depth], [1.0, 1.0, 1.0, 1.0], uvs)
                .with_watch_overlay(self.batch.watch_overlay()),
        );
    }

    /// Push a colored sprite instance into the current batch WITHOUT any
    /// batch-key comparisons. The caller MUST call [`ensure_tile_batch`]
    /// first whenever the texture changes. Uses the current draw_depth.
    #[inline(always)]
    pub fn push_sprite_unchecked(&mut self, x: f32, y: f32, w: f32, h: f32, uvs: [f32; 4], color: [f32; 4]) {
        self.batch.push_instance_unchecked(
            egor_render::instance::Instance::new([w, 0.0, 0.0, h], [x + w * 0.5, y + h * 0.5, self.batch.draw_depth()], color, uvs)
                .with_watch_overlay(self.batch.watch_overlay()),
        );
    }

    /// Push an outlined glyph through the default sprite pipeline. The UV rect
    /// must include the one-texel outline border; `outline_color.a > 0` enables
    /// the outline path in the default shader.
    #[inline(always)]
    pub fn push_outlined_sprite_unchecked(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        uvs: [f32; 4],
        color: [f32; 4],
        outline_color: [f32; 4],
    ) {
        self.batch.push_instance_unchecked(
            egor_render::instance::Instance::new_outlined(
                [w, 0.0, 0.0, h],
                [x + w * 0.5, y + h * 0.5, self.batch.draw_depth()],
                color,
                uvs,
                outline_color,
            )
            .with_watch_overlay(self.batch.watch_overlay()),
        );
    }
    /// Start building an arbitrary polygon primitive, capable of triangles, circles, n-gons
    pub fn polygon(&mut self) -> PolygonBuilder<'_> {
        PolygonBuilder::new(self.batch, self.current_shader)
    }

    /// Push raw vertex/index geometry directly into the current untextured batch.
    /// Vertices must have positions in world space and colors as RGBA \[0..1\].
    /// Indices reference vertices in the provided slice (0-based); they are
    /// automatically offset to match the batch's vertex base.
    pub fn push_geometry(&mut self, verts: &[egor_render::vertex::Vertex], indices: &[u16]) {
        let vert_count = verts.len();
        let idx_count = indices.len();
        if let Some((v_slice, i_slice, base)) = self.batch.allocate(vert_count, idx_count, None, self.current_shader) {
            v_slice.copy_from_slice(verts);
            for (i, idx) in indices.iter().enumerate() {
                i_slice[i] = *idx + base;
            }
        }
    }

    /// Start building a polyline (stroked path) primitive
    pub fn polyline(&mut self) -> PolylineBuilder<'_> {
        PolylineBuilder::new(self.batch, self.current_shader)
    }
    /// Start building a vector path (lines + curves) to be filled or stroked
    pub fn path(&mut self) -> PathBuilder<'_> {
        PathBuilder::new(self.batch, self.current_shader)
    }
    /// Load a font from disk into the text system.
    pub fn load_font(&mut self, bytes: &[u8]) -> Option<String> {
        self.text_renderer.load_font_bytes(bytes)
    }
    /// Set the BCP 47 locale used for script-aware font fallback and shaping.
    pub fn set_text_locale(&mut self, locale: &str) {
        self.text_renderer.set_locale(locale);
    }
    /// Draw a line of text
    pub fn text(&mut self, text: &str) -> TextBuilder<'_> {
        let mut builder = TextBuilder::new(self.text_renderer, text.to_string(), self.batch.render_target());
        if let Some((x, y, width, height)) = self.batch.scissor() {
            builder = builder.clip(Rect::new(
                glam::Vec2::new(x as f32, y as f32),
                glam::Vec2::new(width as f32, height as f32),
            ));
        }
        builder
    }

    /// Load a texture from raw image data (e.g., PNG bytes)
    ///
    /// Returns a texture ID that can be used with `.texture(id)` on primitives.
    /// Typically called once during initialization (when `timer.frame == 0`).
    pub fn load_texture(&mut self, data: &[u8]) -> usize {
        self.renderer.add_texture(data)
    }

    /// Load a texture with nearest-neighbor (pixel-perfect) filtering
    pub fn load_texture_nearest(&mut self, data: &[u8]) -> usize {
        self.renderer.add_texture_nearest(data)
    }

    /// Create a texture from raw RGBA8 pixel data.
    pub fn add_texture_raw(&mut self, w: u32, h: u32, data: &[u8]) -> usize {
        self.renderer.add_texture_raw(w, h, data)
    }
    /// Update texture data by index
    pub fn update_texture(&mut self, index: usize, data: &[u8]) {
        self.renderer.update_texture(index, data);
    }
    /// Update texture data by index with raw width/height
    pub fn update_texture_raw(&mut self, index: usize, w: u32, h: u32, data: &[u8]) {
        self.renderer.update_texture_raw(index, w, h, data);
    }

    /// Load a custom shader from WGSL source code
    pub fn load_shader(&mut self, wgsl_source: &str) -> usize {
        self.renderer.add_shader(wgsl_source)
    }

    /// Create a uniform buffer from raw bytes, returns a uniform id
    pub fn create_uniform(&mut self, data: &[u8]) -> usize {
        self.renderer.add_uniform(data)
    }

    /// Update an existing uniform buffer with raw bytes
    pub fn update_uniform(&mut self, id: usize, data: &[u8]) {
        self.renderer.update_uniform(id, data);
    }

    /// Load a custom shader with associated uniform buffers
    pub fn load_shader_with_uniforms(&mut self, wgsl_source: &str, uniform_ids: &[usize]) -> usize {
        self.renderer.add_shader_with_uniforms(wgsl_source, uniform_ids)
    }

    /// Execute drawing commands with a custom shader
    ///
    /// The shader is automatically reset to default after the closure drops
    pub fn with_shader(&mut self, shader_id: usize, mut render_fn: impl FnMut(&mut Self)) {
        let previous_shader = self.current_shader;
        self.current_shader = Some(shader_id);
        render_fn(self);
        self.current_shader = previous_shader;
    }

    /// Directly set (or clear) the active shader for subsequent draw commands.
    pub fn set_current_shader(&mut self, shader_id: Option<usize>) {
        self.current_shader = shader_id;
    }

    /// Set the scissor rect for subsequent draw commands.
    /// `None` disables scissoring (full viewport).
    pub fn set_scissor(&mut self, rect: Option<(u32, u32, u32, u32)>) {
        self.batch.set_scissor(rect);
    }

    /// Override the camera matrix for subsequent draw commands.
    /// Batches tagged with a camera override will trigger a render-pass split
    /// so that each sub-group renders with its own projection.
    pub fn set_camera_matrix(&mut self, mat: [[f32; 4]; 4]) {
        self.batch.set_camera_matrix(mat);
    }

    /// Reset the camera override. Subsequent draws will use the built-in egor camera.
    pub fn reset_camera_matrix(&mut self) {
        self.batch.reset_camera_matrix();
    }

    /// Set the depth value for subsequent draw commands.
    /// Used for GPU depth testing with LessOrEqual comparison.
    pub fn set_draw_depth(&mut self, depth: f32) {
        self.batch.set_draw_depth(depth);
    }

    pub fn set_replace_blend(&mut self, replace_blend: bool) {
        self.batch.set_replace_blend(replace_blend);
    }

    pub fn set_watch_overlay(&mut self, watch_overlay: f32) {
        self.batch.set_watch_overlay(watch_overlay);
    }

    // -- managed render targets -----------------------------------------

    /// Create a managed offscreen render target and return its store index.
    /// Also registers it as a drawable texture, returning `(store_id, egor_texture_id)`.
    pub fn create_managed_render_target(&mut self, width: u32, height: u32) -> (usize, usize) {
        let store_id = self
            .render_targets
            .create(self.renderer.device(), width, height, self.target_format);
        let tex_id = self.renderer.add_offscreen_texture(self.render_targets.get_mut(store_id));
        (store_id, tex_id)
    }

    /// Resize a managed render target. Re-registers the texture binding.
    /// Returns the (possibly new) egor texture id.
    pub fn resize_managed_render_target(&mut self, store_id: usize, width: u32, height: u32) -> usize {
        self.render_targets.resize(self.renderer.device(), store_id, width, height);
        self.renderer.add_offscreen_texture(self.render_targets.get_mut(store_id))
    }

    /// Direct subsequent draw commands to a managed offscreen render target.
    pub fn set_active_render_target(&mut self, store_id: usize) {
        self.batch.set_render_target(Some(store_id));
    }

    /// Restore drawing to the main backbuffer.
    pub fn clear_active_render_target(&mut self) {
        self.batch.set_render_target(None);
    }

    // -- screen capture -------------------------------------------------

    /// Request a screen capture at the specified dimensions.
    pub fn request_screen_capture(&mut self, w: u32, h: u32, grayscale: bool) {
        self.screen_capture.request(w, h, grayscale);
    }

    pub fn request_screen_capture_with_alpha_mask(
        &mut self,
        w: u32,
        h: u32,
        grayscale: bool,
        source_render_target: usize,
        metadata: Option<[f32; 10]>,
    ) {
        self.screen_capture
            .request_with_alpha_mask(w, h, grayscale, source_render_target, metadata);
    }

    pub fn request_watch_frame_capture(
        &mut self,
        w: u32,
        h: u32,
        logical_w: u32,
        logical_h: u32,
        scale_factor: u32,
        grayscale: bool,
        metadata: Option<[f32; 10]>,
        dirty_rects: Option<WatchCaptureDirtyRects>,
        frame_tag: Option<WatchCaptureFrameTag>,
    ) {
        self.screen_capture.request_watch_overlay_capture(
            w,
            h,
            logical_w,
            logical_h,
            scale_factor,
            grayscale,
            metadata,
            dirty_rects,
            frame_tag,
        );
    }

    pub fn composite_render_target_to_backbuffer(&mut self, source_render_target: usize) {
        self.screen_capture.request_composite_render_target(source_render_target);
    }

    pub fn release_screen_capture_resources(&mut self) {
        self.screen_capture.release_buffers();
    }

    /// Poll for a completed screen capture result.
    /// Returns `Some((width, height, metadata, dirty_rects, frame_tag))` when pixel data is available.
    pub fn poll_screen_capture(
        &mut self,
    ) -> Option<(
        u16,
        u16,
        Option<[f32; 10]>,
        Option<WatchCaptureDirtyRects>,
        Option<WatchCaptureFrameTag>,
    )> {
        self.screen_capture.try_complete()
    }

    /// Access the completed screen capture RGB buffer.
    pub fn screen_capture_rgb_buf(&self) -> &[u8] {
        self.screen_capture.rgb_buf()
    }

    pub fn screen_capture_metrics(&self) -> ScreenCaptureMetrics {
        self.screen_capture.metrics()
    }
}
