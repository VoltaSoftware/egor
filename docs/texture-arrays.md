# Packed texture arrays

`Graphics::add_texture_array_raw(width, height, layers, rgba)` uploads consecutive, equally sized RGBA8 pages with nearest filtering. It returns an error for invalid dimensions, layer limits, or byte counts. Upload every page before issuing gameplay draws. A single-page upload uses ordinary 2D storage.

Use `push_sprite_layer`, `push_sprite_unchecked_layer`, `push_outlined_sprite_unchecked_layer`, `push_tile_layer`, or `rect().texture_layer(page)` to select the page. Existing methods select page zero. Pages of one array share a texture id and batch; page changes never split batches.

The default shader has separately compiled 2D and array variants. The renderer selects the variant with the texture binding. This avoids a per-fragment branch or second texture fetch and works around GLES drivers that cannot sample ordinary 2D storage through an array view. Ordinary textures, offscreen targets, and legacy custom shaders retain the existing 2D path.

Custom shaders can include `// EGOR_TEXTURE_SAMPLING` and call `egor_sample_texture(uv, layer)` and `egor_texture_dimensions()`. The pipeline compiler expands this marker into the correct sampler declarations and compiles both variants once per unique program. Custom shaders without the marker keep their original source and use ordinary 2D textures. Each variant has its own bind-group layout: binding 0 is its matching 2D or array view, and binding 1 is its sampler. No incompatible alternate view is created or bound. The layer is instance attribute 9 (`f32`), increasing the instance stride from 80 to 84 bytes.

On desktop OpenGL, Android GLES and WebGL2, wgpu infers cube storage for square textures whose layer count is a multiple of six. Egor reserves one unused storage layer for these arrays, validates the padded count against the device limit, and uploads and exposes only the requested pages. Page indices and shader dimensions stay unchanged. Vulkan, Direct3D 12, Metal and WebGPU retain the original allocation. Ordinary textures, single-page arrays and offscreen targets use only a 2D view on every backend.

The ignored `array_pages_render_in_one_draw` GPU test checks single-page, three-page, six-page and twelve-page uploads, sampling the first, second and last available pages in one instanced draw followed by an ordinary texture. It uses the real upload path and default shader. Run with `WGPU_BACKEND=vulkan` and `WGPU_BACKEND=gl` (enable `gles` on Windows). Existing watch-overlay and custom pipeline-cache tests cover those paths too.
