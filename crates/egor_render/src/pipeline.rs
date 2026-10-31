use std::{borrow::Cow, collections::HashMap};

use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BlendState,
    BufferBindingType, ColorTargetState, ColorWrites, CompareFunction, DepthStencilState, Device,
    FragmentState, PipelineCache, PipelineLayoutDescriptor, RenderPipeline,
    RenderPipelineDescriptor, SamplerBindingType, ShaderModuleDescriptor, ShaderSource,
    ShaderStages, StencilState, TextureFormat, TextureSampleType, TextureViewDimension,
    VertexState,
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

pub(crate) fn surface_needs_srgb_encode(surface_format: TextureFormat) -> bool {
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
    for param in params
        .split(',')
        .map(str::trim)
        .filter(|param| !param.is_empty())
    {
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
        log::warn!(
            "[egor] custom shader has no fs_main entrypoint; sRGB output encoding was not applied"
        );
        return Cow::Borrowed(wgsl_source);
    };
    let Some(attr_pos) = wgsl_source[..fn_pos].rfind(FRAGMENT_ATTR) else {
        log::warn!(
            "[egor] custom shader fs_main has no @fragment attribute; sRGB output encoding was not applied"
        );
        return Cow::Borrowed(wgsl_source);
    };
    if !wgsl_source[attr_pos + FRAGMENT_ATTR.len()..fn_pos]
        .trim()
        .is_empty()
    {
        log::warn!(
            "[egor] custom shader @fragment attribute could not be matched to fs_main; sRGB output encoding was not applied"
        );
        return Cow::Borrowed(wgsl_source);
    }

    let Some(params_start) = wgsl_source[fn_pos..]
        .find('(')
        .map(|offset| fn_pos + offset)
    else {
        log::warn!(
            "[egor] custom shader fs_main params could not be parsed; sRGB output encoding was not applied"
        );
        return Cow::Borrowed(wgsl_source);
    };
    let Some(params_end) = wgsl_source[params_start..]
        .find(')')
        .map(|offset| params_start + offset)
    else {
        log::warn!(
            "[egor] custom shader fs_main params could not be parsed; sRGB output encoding was not applied"
        );
        return Cow::Borrowed(wgsl_source);
    };
    let params = &wgsl_source[params_start + 1..params_end];
    let Some(argument_names) = parse_fragment_argument_names(params) else {
        log::warn!(
            "[egor] custom shader fs_main arguments could not be parsed; sRGB output encoding was not applied"
        );
        return Cow::Borrowed(wgsl_source);
    };

    let mut cursor = skip_ascii_whitespace(wgsl_source, params_end + 1);
    if !wgsl_source[cursor..].starts_with("->") {
        log::warn!(
            "[egor] custom shader fs_main return type could not be parsed; sRGB output encoding was not applied"
        );
        return Cow::Borrowed(wgsl_source);
    }
    cursor = skip_ascii_whitespace(wgsl_source, cursor + 2);
    if !wgsl_source[cursor..].starts_with(RETURN_LOCATION) {
        log::warn!(
            "[egor] custom shader fs_main return location could not be parsed; sRGB output encoding was not applied"
        );
        return Cow::Borrowed(wgsl_source);
    }
    cursor = skip_ascii_whitespace(wgsl_source, cursor + RETURN_LOCATION.len());
    if !wgsl_source[cursor..].starts_with(RETURN_TYPE) {
        log::warn!(
            "[egor] custom shader fs_main return type is not vec4<f32>; sRGB output encoding was not applied"
        );
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
    wrapped.push_str(
        ") -> @location(0) vec4<f32> {\n    return egor_linear_to_srgb(egor_user_fs_main(",
    );
    wrapped.push_str(&argument_names);
    wrapped.push_str("));\n}\n");

    Cow::Owned(wrapped)
}

/// The WGSL egor compiles for a custom shader's normal pipeline on `surface_format`.
pub(crate) fn custom_shader_source(
    wgsl_source: &str,
    surface_format: TextureFormat,
) -> Cow<'_, str> {
    if surface_needs_srgb_encode(surface_format) {
        wrap_custom_shader_for_srgb_output(wgsl_source)
    } else {
        Cow::Borrowed(wgsl_source)
    }
}

fn first_fragment_argument_name(params: &str) -> Option<String> {
    let first = params
        .split(',')
        .map(str::trim)
        .find(|param| !param.is_empty())?;
    let (name, _) = first.split_once(':')?;
    Some(name.trim().to_owned())
}

pub(crate) fn wrap_custom_shader_for_watch_output(
    wgsl_source: &str,
    encode_srgb: bool,
) -> Option<Cow<'_, str>> {
    const FRAGMENT_ATTR: &str = "@fragment";
    const USER_FN: &str = "fn fs_main";
    const RETURN_LOCATION: &str = "@location(0)";
    const RETURN_TYPE: &str = "vec4<f32>";

    let Some(fn_pos) = wgsl_source.find(USER_FN) else {
        log::warn!(
            "[egor] custom shader has no fs_main entrypoint; watch overlay output was not applied"
        );
        return None;
    };
    let Some(attr_pos) = wgsl_source[..fn_pos].rfind(FRAGMENT_ATTR) else {
        log::warn!(
            "[egor] custom shader fs_main has no @fragment attribute; watch overlay output was not applied"
        );
        return None;
    };
    if !wgsl_source[attr_pos + FRAGMENT_ATTR.len()..fn_pos]
        .trim()
        .is_empty()
    {
        log::warn!(
            "[egor] custom shader @fragment attribute could not be matched to fs_main; watch overlay output was not applied"
        );
        return None;
    }

    let Some(params_start) = wgsl_source[fn_pos..]
        .find('(')
        .map(|offset| fn_pos + offset)
    else {
        log::warn!(
            "[egor] custom shader fs_main params could not be parsed; watch overlay output was not applied"
        );
        return None;
    };
    let Some(params_end) = wgsl_source[params_start..]
        .find(')')
        .map(|offset| params_start + offset)
    else {
        log::warn!(
            "[egor] custom shader fs_main params could not be parsed; watch overlay output was not applied"
        );
        return None;
    };
    let params = &wgsl_source[params_start + 1..params_end];
    let Some(argument_names) = parse_fragment_argument_names(params) else {
        log::warn!(
            "[egor] custom shader fs_main arguments could not be parsed; watch overlay output was not applied"
        );
        return None;
    };
    let Some(first_argument_name) = first_fragment_argument_name(params) else {
        log::warn!(
            "[egor] custom shader fs_main first argument could not be parsed; watch overlay output was not applied"
        );
        return None;
    };
    if !wgsl_source.contains("watch_overlay") {
        log::warn!(
            "[egor] custom shader has no watch_overlay fragment input; watch overlay output was not applied"
        );
        return None;
    };

    let mut cursor = skip_ascii_whitespace(wgsl_source, params_end + 1);
    if !wgsl_source[cursor..].starts_with("->") {
        log::warn!(
            "[egor] custom shader fs_main return type could not be parsed; watch overlay output was not applied"
        );
        return None;
    }
    cursor = skip_ascii_whitespace(wgsl_source, cursor + 2);
    if !wgsl_source[cursor..].starts_with(RETURN_LOCATION) {
        log::warn!(
            "[egor] custom shader fs_main return location could not be parsed; watch overlay output was not applied"
        );
        return None;
    }
    cursor = skip_ascii_whitespace(wgsl_source, cursor + RETURN_LOCATION.len());
    if !wgsl_source[cursor..].starts_with(RETURN_TYPE) {
        log::warn!(
            "[egor] custom shader fs_main return type is not vec4<f32>; watch overlay output was not applied"
        );
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

struct CompiledCustomPipeline {
    pipeline: [RenderPipeline; 2],
    watch_pipeline: [Option<RenderPipeline>; 2],
}

/// A custom shader's pipelines, or a worker thread still compiling them.
enum CompiledSlot {
    Ready(CompiledCustomPipeline),
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    Compiling,
}

/// Everything compiling one custom shader takes, so it can run on a worker thread.
struct CustomCompile {
    device: Device,
    surface_format: TextureFormat,
    texture_layout: [BindGroupLayout; 2],
    camera_layout: BindGroupLayout,
    uniform_layouts: Vec<BindGroupLayout>,
    wgsl_source: String,
    cache: Option<PipelineCache>,
    watch_overlay_supported: bool,
}

impl CustomCompile {
    fn run(&self) -> CompiledCustomPipeline {
        let uniform_layouts: Vec<&BindGroupLayout> = self.uniform_layouts.iter().collect();
        let build = |array| {
            let source = crate::prepare_texture_shader(&self.wgsl_source, array);
            let texture_layout = &self.texture_layout[usize::from(array)];
            let pipeline = create_custom_pipeline(
                &self.device,
                self.surface_format,
                texture_layout,
                &self.camera_layout,
                &uniform_layouts,
                &source,
                self.cache.as_ref(),
            );
            let watch = if self.watch_overlay_supported {
                create_custom_watch_pipeline(
                    &self.device,
                    self.surface_format,
                    texture_layout,
                    &self.camera_layout,
                    &uniform_layouts,
                    &source,
                    self.cache.as_ref(),
                )
            } else {
                None
            };
            (pipeline, watch)
        };
        let (normal, normal_watch) = build(false);
        let (array, array_watch) = if self.wgsl_source.contains("// EGOR_TEXTURE_SAMPLING") {
            build(true)
        } else {
            (normal.clone(), normal_watch.clone())
        };
        CompiledCustomPipeline {
            pipeline: [normal, array],
            watch_pipeline: [normal_watch, array_watch],
        }
    }
}

/// What a compile thread hands back: the compiled slot's index, and its pipelines or the panic that stopped them.
#[cfg(not(target_arch = "wasm32"))]
type FinishedCompile = (usize, std::thread::Result<CompiledCustomPipeline>);

/// A compile waiting for a worker thread. The thread that wants its shader takes it over if no worker has
/// started it, rather than waiting behind every compile queued before it.
#[cfg(not(target_arch = "wasm32"))]
struct QueuedCompile {
    compile: CustomCompile,
    claimed: std::sync::atomic::AtomicBool,
}

#[cfg(not(target_arch = "wasm32"))]
impl QueuedCompile {
    /// True for the one thread that gets to run it.
    fn claim(&self) -> bool {
        !self.claimed.swap(true, std::sync::atomic::Ordering::AcqRel)
    }
}

/// Worker threads compiling custom shaders. Dropping it lets them finish the compile they are on and stop,
/// without waiting for them.
#[cfg(not(target_arch = "wasm32"))]
struct BackgroundCompiles {
    jobs: std::sync::mpsc::Sender<(usize, std::sync::Arc<QueuedCompile>)>,
    finished: std::sync::mpsc::Receiver<FinishedCompile>,
    /// The compiles still waiting for a worker or running on one, by compiled slot.
    queued: HashMap<usize, std::sync::Arc<QueuedCompile>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl BackgroundCompiles {
    fn start(threads: usize) -> Self {
        let (jobs, job_queue) =
            std::sync::mpsc::channel::<(usize, std::sync::Arc<QueuedCompile>)>();
        let (done, finished) = std::sync::mpsc::channel();
        let job_queue = std::sync::Arc::new(std::sync::Mutex::new(job_queue));
        for index in 0..threads {
            let job_queue = job_queue.clone();
            let done = done.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("egor-shaders-{index}"))
                .spawn(move || {
                    loop {
                        let job = job_queue
                            .lock()
                            .unwrap_or_else(|poison| poison.into_inner())
                            .recv();
                        let Ok((id, job)) = job else { break };
                        if !job.claim() {
                            continue;
                        }
                        // A panic (a wgpu validation error) goes back to the thread that wants the shader, where
                        // it would have happened without the worker.
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            job.compile.run()
                        }));
                        if done.send((id, result)).is_err() {
                            break;
                        }
                    }
                });
            if let Err(error) = spawned {
                log::warn!("[egor] shader compile thread {index} did not start: {error}");
            }
        }
        Self {
            jobs,
            finished,
            queued: HashMap::new(),
        }
    }
}

/// One logical custom material. Its compiled pipeline may be shared with other
/// materials while its bindings remain independent.
struct CustomPipelineBinding {
    compiled_pipeline_id: usize,
    bindings: Vec<CustomBinding>,
}

/// What a custom shader's bind groups after the texture and camera ones hold, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CustomBinding {
    /// A renderer uniform buffer.
    Uniform(usize),
    /// A 2D texture and its sampler, set with `Renderer::set_shader_texture`; egor's default
    /// texture until then.
    Texture(Option<usize>),
}

/// Every input that can affect custom pipeline compilation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CustomPipelineKey {
    wgsl_source: String,
    surface_format: TextureFormat,
    uniform_layouts: Vec<BindGroupLayout>,
}

/// Contains all render pipelines and bind group layouts for [`crate::Renderer`]
///
/// Centralizes GPU pipeline configuration, including:
/// - The main primitive rendering pipeline (textured quads, sprites, shapes)
/// - Texture bind group layout (for sampling textures in shaders)
/// - Camera bind group layout (for view/projection transforms)
pub(crate) struct Pipelines {
    primitive: [RenderPipeline; 2],
    /// The default shader with its text outline path, only for instances with an outline.
    primitive_outline: [RenderPipeline; 2],
    primitive_replace: [RenderPipeline; 2],
    primitive_watch: [Option<RenderPipeline>; 2],
    primitive_outline_watch: [Option<RenderPipeline>; 2],
    primitive_replace_watch: [Option<RenderPipeline>; 2],
    custom: Vec<Option<CustomPipelineBinding>>,
    /// Slots of removed custom shaders, which the next ones reuse. Their compiled pipelines stay cached.
    free_custom_slots: Vec<usize>,
    compiled_custom: Vec<CompiledSlot>,
    custom_cache: HashMap<CustomPipelineKey, usize>,
    texture_layout: [BindGroupLayout; 2],
    pub camera_layout: BindGroupLayout,
    watch_overlay_supported: bool,
    /// The driver's compiled pipelines, kept between runs (`Renderer::save_pipeline_cache`).
    cache: Option<PipelineCache>,
    #[cfg(not(target_arch = "wasm32"))]
    background: Option<BackgroundCompiles>,
    /// Custom shader compiles started and finished.
    compiles_started: usize,
    compiles_finished: usize,
}

impl Pipelines {
    /// Creates all pipelines and bind group layouts for the given device and surface format
    pub fn new(
        device: &Device,
        surface_format: TextureFormat,
        watch_overlay_supported: bool,
        cache: Option<PipelineCache>,
    ) -> Self {
        let texture_layout =
            std::array::from_fn(|kind| create_texture_bind_group_layout(device, kind == 1));
        let camera_layout = create_camera_bind_group_layout(device);

        let pipelines = |fragment: PrimitiveFragment, watch_overlay: bool| {
            let (blend, depth_enabled) = match fragment {
                PrimitiveFragment::Replace => (None, false),
                PrimitiveFragment::Plain | PrimitiveFragment::Outline => {
                    (Some(BlendState::ALPHA_BLENDING), true)
                }
            };
            std::array::from_fn(|kind| {
                create_primitive_pipeline(
                    device,
                    surface_format,
                    &texture_layout[kind],
                    &camera_layout,
                    blend,
                    depth_enabled,
                    fragment,
                    watch_overlay,
                    kind == 1,
                    cache.as_ref(),
                )
            })
        };
        let watch_pipelines = |fragment: PrimitiveFragment| {
            if watch_overlay_supported {
                pipelines(fragment, true).map(Some)
            } else {
                [None, None]
            }
        };
        let primitive = pipelines(PrimitiveFragment::Plain, false);
        let primitive_outline = pipelines(PrimitiveFragment::Outline, false);
        let primitive_replace = pipelines(PrimitiveFragment::Replace, false);
        let primitive_watch = watch_pipelines(PrimitiveFragment::Plain);
        let primitive_outline_watch = watch_pipelines(PrimitiveFragment::Outline);
        let primitive_replace_watch = watch_pipelines(PrimitiveFragment::Replace);

        Self {
            primitive,
            primitive_outline,
            primitive_outline_watch,
            primitive_replace,
            primitive_watch,
            primitive_replace_watch,
            custom: Vec::new(),
            free_custom_slots: Vec::new(),
            compiled_custom: Vec::new(),
            custom_cache: HashMap::new(),
            texture_layout,
            camera_layout,
            watch_overlay_supported,
            cache,
            #[cfg(not(target_arch = "wasm32"))]
            background: None,
            compiles_started: 0,
            compiles_finished: 0,
        }
    }

    /// From now on custom shaders compile on `threads` worker threads, and [`Self::add_custom`] returns before
    /// they are ready; [`Self::wait_for_custom`] waits for one.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn compile_in_background(&mut self, threads: usize) {
        self.background = Some(BackgroundCompiles::start(threads));
    }

    /// Custom shader compiles finished, and started.
    pub fn compiles(&self) -> (usize, usize) {
        (self.compiles_finished, self.compiles_started)
    }

    /// Takes in the compiles the worker threads have finished.
    pub fn take_finished_compiles(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        while let Some(Ok((id, result))) = self
            .background
            .as_ref()
            .map(|background| background.finished.try_recv())
        {
            self.finish_compile(id, result);
        }
    }

    /// Waits until custom shader `id`'s pipelines are compiled, if a worker thread is still on them.
    #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
    pub fn wait_for_custom(&mut self, id: usize) {
        if self.compiles_finished == self.compiles_started {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(Some(custom)) = self.custom.get(id) {
            let compiled_id = custom.compiled_pipeline_id;
            while matches!(self.compiled_custom[compiled_id], CompiledSlot::Compiling) {
                let Some(background) = self.background.as_mut() else {
                    break;
                };
                // No worker has started it: compiling it here beats waiting for every compile queued first.
                if let Some(job) = background
                    .queued
                    .get(&compiled_id)
                    .filter(|job| job.claim())
                    .cloned()
                {
                    background.queued.remove(&compiled_id);
                    self.compiled_custom[compiled_id] = CompiledSlot::Ready(job.compile.run());
                    self.compiles_finished += 1;
                    break;
                }
                let Ok((id, result)) = background.finished.recv() else {
                    break;
                };
                self.finish_compile(id, result);
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn finish_compile(&mut self, id: usize, result: std::thread::Result<CompiledCustomPipeline>) {
        if let Some(background) = self.background.as_mut() {
            background.queued.remove(&id);
        }
        match result {
            Ok(compiled) => self.compiled_custom[id] = CompiledSlot::Ready(compiled),
            Err(panic) => std::panic::resume_unwind(panic),
        }
        self.compiles_finished += 1;
    }

    /// Compiles `compile` on a worker thread when there are any, else here; the index of its slot.
    fn start_compile(&mut self, compile: CustomCompile) -> usize {
        let id = self.compiled_custom.len();
        self.compiles_started += 1;
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(background) = self.background.as_mut() {
            let job = std::sync::Arc::new(QueuedCompile {
                compile,
                claimed: false.into(),
            });
            if background.jobs.send((id, job.clone())).is_ok() {
                background.queued.insert(id, job);
                self.compiled_custom.push(CompiledSlot::Compiling);
            } else {
                // No worker thread started.
                self.compiled_custom
                    .push(CompiledSlot::Ready(job.compile.run()));
                self.compiles_finished += 1;
            }
            return id;
        }
        self.compiled_custom
            .push(CompiledSlot::Ready(compile.run()));
        self.compiles_finished += 1;
        id
    }

    /// The driver's compiled pipelines so far, to keep for the next run.
    pub fn cache_data(&self) -> Option<Vec<u8>> {
        self.cache.as_ref()?.get_data()
    }

    /// Creates a custom shader pipeline from WGSL source
    pub fn add_custom(
        &mut self,
        device: &Device,
        surface_format: TextureFormat,
        wgsl_source: &str,
        uniform_layouts: &[&BindGroupLayout],
        bindings: Vec<CustomBinding>,
    ) -> usize {
        assert_eq!(
            uniform_layouts.len(),
            bindings.len(),
            "custom shader layouts and bindings must have the same length"
        );

        let cache_key = CustomPipelineKey {
            wgsl_source: wgsl_source.to_owned(),
            surface_format,
            uniform_layouts: uniform_layouts
                .iter()
                .map(|layout| (*layout).clone())
                .collect(),
        };
        let compiled_pipeline_id = if let Some(&id) = self.custom_cache.get(&cache_key) {
            id
        } else {
            let id = self.start_compile(CustomCompile {
                device: device.clone(),
                surface_format,
                texture_layout: self.texture_layout.clone(),
                camera_layout: self.camera_layout.clone(),
                uniform_layouts: cache_key.uniform_layouts.clone(),
                wgsl_source: cache_key.wgsl_source.clone(),
                cache: self.cache.clone(),
                watch_overlay_supported: self.watch_overlay_supported,
            });
            self.custom_cache.insert(cache_key, id);
            id
        };

        let custom = Some(CustomPipelineBinding {
            compiled_pipeline_id,
            bindings,
        });
        match self.free_custom_slots.pop() {
            Some(id) => {
                self.custom[id] = custom;
                id
            }
            None => {
                self.custom.push(custom);
                self.custom.len() - 1
            }
        }
    }

    /// Frees custom shader `id`; its slot goes to the next one. Draws with it fall back to the default pipeline.
    pub fn remove_custom(&mut self, id: usize) {
        if self.custom.get_mut(id).and_then(Option::take).is_some() {
            self.free_custom_slots.push(id);
        }
    }

    /// The layout of a [`CustomBinding::Texture`] group.
    pub fn texture_layout_2d(&self) -> &BindGroupLayout {
        &self.texture_layout[0]
    }

    /// Points the `index`th texture binding of a custom shader at `texture_id`.
    pub fn set_custom_texture(&mut self, shader_id: usize, index: usize, texture_id: usize) {
        let Some(Some(custom)) = self.custom.get_mut(shader_id) else {
            return;
        };
        if let Some(CustomBinding::Texture(texture)) = custom
            .bindings
            .iter_mut()
            .filter(|binding| matches!(binding, CustomBinding::Texture(_)))
            .nth(index)
        {
            *texture = Some(texture_id);
        }
    }

    pub fn resolve_with_replace(
        &self,
        shader_id: Option<usize>,
        replace_blend: bool,
        watch_overlay: bool,
        texture_array: bool,
    ) -> Option<(&RenderPipeline, &[CustomBinding])> {
        let kind = usize::from(texture_array);
        if replace_blend && shader_id.is_none() {
            return Some((
                if watch_overlay {
                    self.primitive_replace_watch[kind].as_ref()?
                } else {
                    &self.primitive_replace[kind]
                },
                &[],
            ));
        }
        if let Some(custom) = shader_id.and_then(|id| self.custom.get(id)?.as_ref()) {
            let CompiledSlot::Ready(compiled) =
                self.compiled_custom.get(custom.compiled_pipeline_id)?
            else {
                return None;
            };
            if watch_overlay {
                compiled.watch_pipeline[kind]
                    .as_ref()
                    .map(|pipeline| (pipeline, custom.bindings.as_slice()))
            } else {
                Some((&compiled.pipeline[kind], &custom.bindings))
            }
        } else if watch_overlay {
            Some((self.primitive_watch[kind].as_ref()?, &[]))
        } else {
            Some((&self.primitive[kind], &[]))
        }
    }

    /// The default shader's outlined-text pipeline, for instances with an outline; `resolve_with_replace` gives
    /// the plain one, which every other default draw uses.
    pub fn resolve_outline(
        &self,
        watch_overlay: bool,
        texture_array: bool,
    ) -> Option<&RenderPipeline> {
        let kind = usize::from(texture_array);
        if watch_overlay {
            self.primitive_outline_watch[kind].as_ref()
        } else {
            Some(&self.primitive_outline[kind])
        }
    }

    pub fn supports_watch_overlay(&self, shader_id: Option<usize>) -> bool {
        self.watch_overlay_supported
            && match shader_id {
                Some(id) => self
                    .custom
                    .get(id)
                    .and_then(Option::as_ref)
                    .and_then(|custom| self.compiled_custom.get(custom.compiled_pipeline_id))
                    .is_some_and(|compiled| match compiled {
                        CompiledSlot::Ready(compiled) => {
                            compiled.watch_pipeline.iter().all(Option::is_some)
                        }
                        CompiledSlot::Compiling => false,
                    }),
                None => {
                    self.primitive_watch.iter().all(Option::is_some)
                        && self.primitive_replace_watch.iter().all(Option::is_some)
                }
            }
    }
}

/// Creates the bind group layout for texture sampling
///
/// Defines the texture view and its sampler:
/// - Binding 0: 2D or 2D-array texture (fragment shader)
/// - Binding 1: Sampler (fragment shader)
pub(crate) fn create_texture_bind_group_layout(device: &Device, array: bool) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("Texture Bind Group Layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: if array {
                        TextureViewDimension::D2Array
                    } else {
                        TextureViewDimension::D2
                    },
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
/// Which fragment body a default pipeline runs (`shader.wgsl`).
#[derive(Clone, Copy, Debug)]
enum PrimitiveFragment {
    /// Sprites, shapes and text without an outline.
    Plain,
    /// Text with an outline.
    Outline,
    /// A flat color written over what is there.
    Replace,
}

impl PrimitiveFragment {
    fn entry_name(self) -> &'static str {
        match self {
            Self::Plain => "main",
            Self::Outline => "outline",
            Self::Replace => "replace",
        }
    }
}

fn create_primitive_pipeline(
    device: &Device,
    surface_format: TextureFormat,
    texture_layout: &BindGroupLayout,
    camera_layout: &BindGroupLayout,
    blend: Option<BlendState>,
    depth_enabled: bool,
    fragment: PrimitiveFragment,
    watch_overlay: bool,
    texture_array: bool,
    cache: Option<&PipelineCache>,
) -> RenderPipeline {
    let source = crate::prepare_texture_shader(include_str!("../shader.wgsl"), texture_array);
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(source.into()),
    });
    let fragment_entry_point = format!(
        "fs_{}{}{}",
        fragment.entry_name(),
        if surface_needs_srgb_encode(surface_format) {
            "_srgb_encoded"
        } else {
            ""
        },
        if watch_overlay { "_watch" } else { "" },
    );
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
    let pipeline_label = format!(
        "Primitive {fragment:?}{} Pipeline",
        if watch_overlay { " Watch Overlay" } else { "" }
    );
    let pipeline_layout_label = format!("{pipeline_label} Layout");

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(&pipeline_layout_label),
        bind_group_layouts: &[Some(texture_layout), Some(camera_layout)],
        immediate_size: 0,
    });

    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some(&pipeline_label),
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
            entry_point: Some(&fragment_entry_point),
            targets: &targets,
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache,
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
    cache: Option<&PipelineCache>,
) -> RenderPipeline {
    let wgsl_source = custom_shader_source(wgsl_source, surface_format);
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Custom Shader"),
        source: ShaderSource::Wgsl(wgsl_source),
    });

    let mut layouts: Vec<Option<&BindGroupLayout>> =
        vec![Some(texture_layout), Some(camera_layout)];
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
        cache,
    })
}

fn create_custom_watch_pipeline(
    device: &Device,
    surface_format: TextureFormat,
    texture_layout: &BindGroupLayout,
    camera_layout: &BindGroupLayout,
    extra_layouts: &[&BindGroupLayout],
    wgsl_source: &str,
    cache: Option<&PipelineCache>,
) -> Option<RenderPipeline> {
    let wgsl_source = wrap_custom_shader_for_watch_output(
        wgsl_source,
        surface_needs_srgb_encode(surface_format),
    )?;
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Custom Watch Shader"),
        source: ShaderSource::Wgsl(wgsl_source),
    });

    let mut layouts: Vec<Option<&BindGroupLayout>> =
        vec![Some(texture_layout), Some(camera_layout)];
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
        cache,
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
        assert!(
            wrap_custom_shader_for_watch_output(CUSTOM_SHADER_WITHOUT_OVERLAY, false).is_none()
        );
    }

    #[test]
    fn custom_watch_wrapper_emits_overlay_attachment() {
        let wrapped = wrap_custom_shader_for_watch_output(CUSTOM_SHADER_WITH_OVERLAY, false)
            .expect("wrapper should succeed");

        assert!(wrapped.contains("@location(1) overlay: vec4<f32>"));
        assert!(wrapped.contains("input.watch_overlay"));
    }

    #[test]
    fn background_compiles_hand_their_pipelines_to_the_shaders_waiting_for_them() {
        let (device, _queue) = Device::noop(&wgpu::DeviceDescriptor::default());
        let surface_format = TextureFormat::Rgba8Unorm;
        let mut pipelines = Pipelines::new(&device, surface_format, true, None);
        pipelines.compile_in_background(2);
        let uniform_layout = create_camera_bind_group_layout(&device);
        let mut add = |source, uniform| {
            pipelines.add_custom(
                &device,
                surface_format,
                source,
                &[&uniform_layout],
                vec![CustomBinding::Uniform(uniform)],
            )
        };
        let first = add(CUSTOM_SHADER_WITH_OVERLAY, 1);
        let same_program = add(CUSTOM_SHADER_WITH_OVERLAY, 2);
        let other = add(CUSTOM_SHADER_WITHOUT_OVERLAY, 3);

        // One compile per program, however many shaders share it.
        assert_eq!(pipelines.compiles().1, 2);
        for id in [first, same_program, other] {
            pipelines.wait_for_custom(id);
            assert!(
                pipelines
                    .resolve_with_replace(Some(id), false, false, false)
                    .is_some()
            );
        }
        assert_eq!(pipelines.compiles(), (2, 2));
        assert!(pipelines.supports_watch_overlay(Some(first)));
        assert!(!pipelines.supports_watch_overlay(Some(other)));
    }

    #[test]
    fn custom_pipeline_programs_are_reused_across_uniform_bindings() {
        let (device, _queue) = Device::noop(&wgpu::DeviceDescriptor::default());
        let surface_format = TextureFormat::Rgba8Unorm;
        let mut pipelines = Pipelines::new(&device, surface_format, true, None);
        let uniform_layout = create_camera_bind_group_layout(&device);

        let first = pipelines.add_custom(
            &device,
            surface_format,
            CUSTOM_SHADER_WITH_OVERLAY,
            &[&uniform_layout],
            vec![CustomBinding::Uniform(4)],
        );
        let second = pipelines.add_custom(
            &device,
            surface_format,
            CUSTOM_SHADER_WITH_OVERLAY,
            &[&uniform_layout],
            vec![CustomBinding::Uniform(9)],
        );

        assert_eq!(pipelines.compiled_custom.len(), 1);
        assert_eq!(pipelines.custom.len(), 2);

        let (first_pipeline, first_uniforms) = pipelines
            .resolve_with_replace(Some(first), false, false, false)
            .expect("first custom pipeline should resolve");
        let (second_pipeline, second_uniforms) = pipelines
            .resolve_with_replace(Some(second), false, false, false)
            .expect("second custom pipeline should resolve");
        assert_eq!(first_pipeline, second_pipeline);
        assert_eq!(first_uniforms, &[CustomBinding::Uniform(4)]);
        assert_eq!(second_uniforms, &[CustomBinding::Uniform(9)]);

        let (first_watch_pipeline, _) = pipelines
            .resolve_with_replace(Some(first), false, true, false)
            .expect("first custom watch pipeline should resolve");
        let (second_watch_pipeline, _) = pipelines
            .resolve_with_replace(Some(second), false, true, false)
            .expect("second custom watch pipeline should resolve");
        assert_eq!(first_watch_pipeline, second_watch_pipeline);
    }

    #[test]
    fn custom_pipeline_cache_keeps_compilation_inputs_separate() {
        let (device, _queue) = Device::noop(&wgpu::DeviceDescriptor::default());
        let surface_format = TextureFormat::Rgba8Unorm;
        let mut pipelines = Pipelines::new(&device, surface_format, false, None);
        let first_layout = create_camera_bind_group_layout(&device);
        let second_layout = create_camera_bind_group_layout(&device);

        pipelines.add_custom(
            &device,
            surface_format,
            CUSTOM_SHADER_WITH_OVERLAY,
            &[&first_layout],
            vec![CustomBinding::Uniform(4)],
        );
        pipelines.add_custom(
            &device,
            surface_format,
            CUSTOM_SHADER_WITH_OVERLAY,
            &[&second_layout],
            vec![CustomBinding::Uniform(9)],
        );
        pipelines.add_custom(
            &device,
            TextureFormat::Rgba8UnormSrgb,
            CUSTOM_SHADER_WITH_OVERLAY,
            &[&first_layout],
            vec![CustomBinding::Uniform(12)],
        );

        assert_eq!(pipelines.compiled_custom.len(), 3);
    }
}
