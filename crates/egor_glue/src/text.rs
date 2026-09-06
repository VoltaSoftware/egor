use egor_render::{
    Device, Queue, RenderPass, Renderer, TextureFormat,
    wgpu::{CompareFunction, DepthStencilState},
};
use glam::Vec2;
use glyphon::{
    Attrs, Buffer, Cache, Color as GlyphonColor, Family, FontSystem, Metrics, Resolution, Shaping,
    Style, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer as GlyphonRenderer, Viewport,
    Weight, Wrap, fontdb,
};

use crate::{color::Color, math::Rect};

use std::collections::HashMap;
use std::ops::Range;

struct TextEntry {
    buffer: Buffer,
    effect_buffer: Option<Buffer>,
    position: Vec2,
    bounds: Option<Rect>,
    color: GlyphonColor,
    effect: TextEffect,
    render_target: Option<usize>,
}

#[derive(Clone, Copy)]
enum TextEffect {
    None,
    Shadow { offset: Vec2, color: GlyphonColor },
    Outline { radius: f32, color: GlyphonColor },
}

pub struct TextRenderer {
    font_system: FontSystem,
    swash_cache: SwashCache,
    atlas: TextAtlas,
    renderer: GlyphonRenderer,
    viewport: Viewport,
    entries: Vec<TextEntry>,
    buffer_pool: Vec<Buffer>,
    loaded_fonts: HashMap<(u128, usize), String>,
}

const MAX_POOLED_BUFFERS: usize = 64;

impl TextRenderer {
    pub(crate) fn new(device: &Device, queue: &Queue, format: TextureFormat) -> Self {
        let font_system = new_font_system("en");
        let swash_cache = SwashCache::new();
        let cache = Cache::new(device);
        let viewport = Viewport::new(device, &cache);
        let mut atlas = TextAtlas::new(device, queue, &cache, format);
        let renderer = GlyphonRenderer::new(
            &mut atlas,
            device,
            Default::default(),
            Some(DepthStencilState {
                format: Renderer::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
        );

        Self {
            font_system,
            swash_cache,
            atlas,
            renderer,
            viewport,
            entries: Vec::new(),
            buffer_pool: Vec::new(),
            loaded_fonts: HashMap::new(),
        }
    }

    pub fn load_font_bytes(&mut self, bytes: &[u8]) -> Option<String> {
        let fingerprint = (fnv1a_128(bytes), bytes.len());
        if let Some(family) = self.loaded_fonts.get(&fingerprint) {
            return Some(family.clone());
        }

        let previous_face_count = self.font_system.db().faces().count();
        self.font_system.db_mut().load_font_data(bytes.to_vec());
        let face = self.font_system.db().faces().nth(previous_face_count)?;
        let family = face.families.first()?.0.clone();
        self.loaded_fonts.insert(fingerprint, family.clone());
        Some(family)
    }

    /// Select the locale used by cosmic-text's script-aware font fallback.
    /// Loaded font data is retained; only shaping and fallback caches are rebuilt.
    pub fn set_locale(&mut self, locale: &str) {
        if self.font_system.locale() == locale {
            return;
        }

        let mut database = fontdb::Database::new();
        std::mem::swap(self.font_system.db_mut(), &mut database);
        self.font_system = FontSystem::new_with_locale_and_db(locale.to_owned(), database);
        self.buffer_pool.clear();
    }

    /// Returns true if any text was queued this frame.
    pub(crate) fn has_entries(&self) -> bool {
        !self.entries.is_empty()
    }

    /// Prepare the text renderer for drawing.
    /// Skipping this when `has_entries()` is false avoids glyphon overhead.
    pub(crate) fn prepare(
        &mut self,
        device: &Device,
        queue: &Queue,
        width: u32,
        height: u32,
        render_target: Option<usize>,
    ) {
        self.viewport.update(queue, Resolution { width, height });
        let mut text_areas = Vec::with_capacity(self.entries.len());
        for entry in self
            .entries
            .iter()
            .filter(|entry| entry.render_target == render_target)
        {
            let bounds = entry.bounds.map_or(
                TextBounds {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                },
                |bounds| TextBounds {
                    left: bounds.position.x.floor() as i32,
                    top: bounds.position.y.floor() as i32,
                    right: (bounds.position.x + bounds.size.x).ceil() as i32,
                    bottom: (bounds.position.y + bounds.size.y).ceil() as i32,
                },
            );

            macro_rules! push_area {
                ($buffer:expr, $position:expr, $color:expr) => {{
                    let position = $position;
                    text_areas.push(TextArea {
                        buffer: $buffer,
                        left: position.x,
                        top: position.y,
                        bounds,
                        scale: 1.0,
                        default_color: $color,
                        custom_glyphs: &[],
                    });
                }};
            }

            match entry.effect {
                TextEffect::None => {}
                TextEffect::Shadow { offset, color } => {
                    push_area!(
                        entry.effect_buffer.as_ref().unwrap_or(&entry.buffer),
                        entry.position + offset,
                        color
                    )
                }
                TextEffect::Outline { radius, color } => {
                    for offset in [
                        Vec2::new(-radius, -radius),
                        Vec2::new(0.0, -radius),
                        Vec2::new(radius, -radius),
                        Vec2::new(-radius, 0.0),
                        Vec2::new(radius, 0.0),
                        Vec2::new(-radius, radius),
                        Vec2::new(0.0, radius),
                        Vec2::new(radius, radius),
                    ] {
                        push_area!(
                            entry.effect_buffer.as_ref().unwrap_or(&entry.buffer),
                            entry.position + offset,
                            color
                        );
                    }
                }
            }
            push_area!(&entry.buffer, entry.position, entry.color);
        }
        self.renderer
            .prepare(
                device,
                queue,
                &mut self.font_system,
                &mut self.atlas,
                &self.viewport,
                text_areas,
                &mut self.swash_cache,
            )
            .unwrap();
    }

    pub(crate) fn finish_frame(&mut self) {
        // Return buffers to the pool for reuse next frame
        for entry in self.entries.drain(..) {
            if self.buffer_pool.len() < MAX_POOLED_BUFFERS {
                self.buffer_pool.push(entry.buffer);
            }
            if let Some(buffer) = entry.effect_buffer
                && self.buffer_pool.len() < MAX_POOLED_BUFFERS
            {
                self.buffer_pool.push(buffer);
            }
        }
    }

    pub(crate) fn render<'a>(&'a self, pass: &mut RenderPass<'a>) {
        self.renderer
            .render(&self.atlas, &self.viewport, pass)
            .unwrap();
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32, queue: &Queue) {
        self.viewport.update(queue, Resolution { width, height });
    }

    /// Takes a buffer from the pool, or creates a new one with the given metrics
    fn take_buffer(&mut self, metrics: Metrics) -> Buffer {
        if let Some(mut buf) = self.buffer_pool.pop() {
            buf.set_metrics(metrics);
            buf
        } else {
            Buffer::new(&mut self.font_system, metrics)
        }
    }
}

fn new_font_system(locale: &str) -> FontSystem {
    // Do not consult platform fonts: desktop, mobile, and WebAssembly must
    // shape the same text with the same explicitly loaded faces.
    let mut database = fontdb::Database::new();
    database.load_font_data(include_bytes!("../inter-v19-latin-regular.ttf").to_vec());
    database.set_sans_serif_family("Inter");
    FontSystem::new_with_locale_and_db(locale.to_owned(), database)
}

fn fnv1a_128(bytes: &[u8]) -> u128 {
    let mut hash = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d_u128;
    for byte in bytes {
        hash ^= u128::from(*byte);
        hash = hash.wrapping_mul(0x0000_0000_0100_0000_0000_0000_0000_013b_u128);
    }
    hash
}

/// Alignment of text (for use with and) relative to a rectangle
pub enum Align {
    TopLeft,
    TopCenter,
    TopRight,
    MiddleLeft,
    MiddleCenter,
    MiddleRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

/// A builder for queuing a single line of text to the [`TextRenderer`].
/// The text is uploaded and rendered on the next frame
///
/// # Example
/// ```ignore
/// gfx.text("Hello World").at((100.0, 50.0)).size(24.0).color(Color::WHITE);
/// ```
pub struct TextBuilder<'a> {
    /// Reference to the renderer that will draw this text
    renderer: &'a mut TextRenderer,
    render_target: Option<usize>,
    /// The string content to render
    text: String,
    /// Top-left anchor position; may be offset by alignment
    position: Vec2,
    position_is_baseline: bool,
    /// Optional bounding rectangle for alignment (origin, size)
    rect: Option<Rect>,
    /// Line height in pixels; defaults to `size * 1.2`
    line_height: Option<f32>,
    max_width: Option<f32>,
    wrap: Wrap,
    clip: Option<Rect>,
    size: f32,
    color: Color,
    color_ranges: Vec<(Range<usize>, Color)>,
    effect: TextEffect,
    /// Font family name used for matching
    family: String,
    weight: Weight,
    style: Style,
    align: Align,
}

impl<'a> TextBuilder<'a> {
    /// Create a new text builder that will push text to the renderer
    ///
    /// A default font family is selected automatically. Use [`Self::font`] to override it
    pub fn new(renderer: &'a mut TextRenderer, text: String, render_target: Option<usize>) -> Self {
        Self {
            renderer,
            render_target,
            text,
            position: Vec2::new(10.0, 10.0),
            position_is_baseline: false,
            rect: None,
            size: 16.0,
            line_height: None,
            max_width: None,
            wrap: Wrap::None,
            clip: None,
            color: Color::BLACK,
            color_ranges: Vec::new(),
            effect: TextEffect::None,
            family: "Inter".into(),
            weight: Weight::NORMAL,
            style: Style::Normal,
            align: Align::TopLeft,
        }
    }

    /// Set the font family used to render the text
    ///
    /// The family must match a font that has been loaded into the renderer.
    /// If the family cannot be found, a fallback font will be used (Inter)
    pub fn font(mut self, family: String) -> Self {
        self.family = family;
        self
    }

    /// Set the screen-space position of the text (top-left corner)
    pub fn at(mut self, position: impl Into<Vec2>) -> Self {
        self.position = position.into();
        self.position_is_baseline = false;
        self
    }

    /// Set a baseline position, matching APIs which place bitmap-font glyphs
    /// relative to their baseline rather than their top edge.
    pub fn baseline_at(mut self, position: impl Into<Vec2>) -> Self {
        self.position = position.into();
        self.position_is_baseline = true;
        self
    }

    /// Sets a bounding rectangle for the text
    ///
    /// The text will be positioned inside `rect` according to the given
    /// [`Align`] value instead of using a raw point
    ///
    /// `rect.position` is the top-left corner and `rect.size` is its width/height
    pub fn in_rect(mut self, rect: Rect, align: Align) -> Self {
        self.rect = Some(rect);
        self.align = align;
        self
    }

    /// Set the font size in points
    pub fn size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    /// Set the line height in pixels.
    ///
    /// Defaults to `size * 1.2` if not set.
    pub fn line_height(mut self, line_height: f32) -> Self {
        self.line_height = Some(line_height);
        self
    }

    /// Wrap text within `max_width`, falling back to grapheme boundaries for
    /// scripts which do not separate every word with spaces.
    pub fn wrap(mut self, max_width: f32) -> Self {
        self.max_width = Some(max_width.max(0.0));
        self.wrap = Wrap::WordOrGlyph;
        self
    }

    /// Clip this text to a screen-space rectangle.
    pub fn clip(mut self, rect: Rect) -> Self {
        self.clip = Some(rect);
        self
    }

    /// Set the text color
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Apply a color to a UTF-8 byte range. Ranges must be ordered,
    /// non-overlapping, and end on character boundaries.
    pub fn color_range(mut self, range: Range<usize>, color: Color) -> Self {
        self.color_ranges.push((range, color));
        self
    }

    /// Draw a shadow using the same shaped glyph run.
    pub fn shadow(mut self, offset: impl Into<Vec2>, color: Color) -> Self {
        self.effect = TextEffect::Shadow {
            offset: offset.into(),
            color: color.into(),
        };
        self
    }

    /// Draw an eight-direction outline using the same shaped glyph run. The
    /// copies are still emitted by Glyphon in one instanced draw call.
    pub fn outline(mut self, radius: f32, color: Color) -> Self {
        self.effect = TextEffect::Outline {
            radius: radius.max(0.0),
            color: color.into(),
        };
        self
    }

    /// Render the text in bold
    pub fn bold(mut self) -> Self {
        self.weight = Weight::BOLD;
        self
    }

    /// Render the text in italic
    pub fn italic(mut self) -> Self {
        self.style = Style::Italic;
        self
    }

    /// Set a specific font weight (100–900).
    ///
    /// Overrides [`Self::bold`]. Common values: 400 = normal, 700 = bold.
    pub fn weight(mut self, weight: u16) -> Self {
        self.weight = Weight(weight);
        self
    }
}

impl Drop for TextBuilder<'_> {
    fn drop(&mut self) {
        let line_height = self.line_height.unwrap_or(self.size * 1.2);
        let mut buffer = self
            .renderer
            .take_buffer(Metrics::new(self.size, line_height));
        buffer.set_size(self.max_width, None);
        buffer.set_wrap(self.wrap);
        let default_attrs = Attrs::new()
            .family(Family::Name(&self.family))
            .weight(self.weight)
            .style(self.style);
        if self.color_ranges.is_empty() {
            buffer.set_text(&self.text, &default_attrs, Shaping::Advanced, None);
        } else {
            self.color_ranges.sort_by_key(|(range, _)| range.start);
            let mut spans = Vec::with_capacity(self.color_ranges.len() * 2 + 1);
            let mut cursor = 0;
            for (range, color) in &self.color_ranges {
                assert!(
                    range.start >= cursor
                        && range.end >= range.start
                        && range.end <= self.text.len()
                        && self.text.is_char_boundary(range.start)
                        && self.text.is_char_boundary(range.end),
                    "text color ranges must be ordered, non-overlapping UTF-8 byte ranges"
                );
                if cursor < range.start {
                    spans.push((&self.text[cursor..range.start], default_attrs.clone()));
                }
                spans.push((
                    &self.text[range.clone()],
                    default_attrs.clone().color((*color).into()),
                ));
                cursor = range.end;
            }
            if cursor < self.text.len() {
                spans.push((&self.text[cursor..], default_attrs.clone()));
            }
            buffer.set_rich_text(spans, &default_attrs, Shaping::Advanced, None);
        }

        // Rich colors override TextArea::default_color. Use a second, plain
        // buffer for shadows/outlines so their requested color stays uniform.
        let effect_buffer =
            if !self.color_ranges.is_empty() && !matches!(self.effect, TextEffect::None) {
                let mut effect_buffer = self
                    .renderer
                    .take_buffer(Metrics::new(self.size, line_height));
                effect_buffer.set_size(self.max_width, None);
                effect_buffer.set_wrap(self.wrap);
                effect_buffer.set_text(&self.text, &default_attrs, Shaping::Advanced, None);
                Some(effect_buffer)
            } else {
                None
            };

        let needs_layout = self.rect.is_some() || self.position_is_baseline;
        if needs_layout {
            buffer.shape_until_scroll(&mut self.renderer.font_system, false);
        }

        // Compute final position, applying alignment within rect if set.
        let mut position = if let Some(rect) = self.rect {
            let text_w = buffer
                .layout_runs()
                .map(|r| r.line_w)
                .fold(0.0_f32, f32::max);
            let text_h = buffer
                .layout_runs()
                .map(|run| run.line_top + run.line_height)
                .fold(0.0_f32, f32::max);

            let x = match self.align {
                Align::TopLeft | Align::MiddleLeft | Align::BottomLeft => rect.position.x,
                Align::TopCenter | Align::MiddleCenter | Align::BottomCenter => {
                    rect.position.x + (rect.size.x - text_w) * 0.5
                }
                Align::TopRight | Align::MiddleRight | Align::BottomRight => {
                    rect.position.x + rect.size.x - text_w
                }
            };
            let y = match self.align {
                Align::TopLeft | Align::TopCenter | Align::TopRight => rect.position.y,
                Align::MiddleLeft | Align::MiddleCenter | Align::MiddleRight => {
                    rect.position.y + (rect.size.y - text_h) * 0.5
                }
                Align::BottomLeft | Align::BottomCenter | Align::BottomRight => {
                    rect.position.y + rect.size.y - text_h
                }
            };

            Vec2::new(x, y)
        } else {
            self.position
        };

        if self.position_is_baseline {
            let baseline_offset = buffer
                .layout_runs()
                .next()
                .map(|run| run.line_y)
                .unwrap_or(line_height);
            position.y -= baseline_offset;
        }

        self.renderer.entries.push(TextEntry {
            buffer,
            effect_buffer,
            position,
            bounds: self.clip,
            color: self.color.into(),
            effect: self.effect,
            render_target: self.render_target,
        });
    }
}
