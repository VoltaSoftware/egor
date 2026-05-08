use wgpu::{CommandEncoder, Queue, SurfaceTexture, Texture, TextureView};

/// Stack-allocated presentable surface — avoids a heap allocation every frame.
pub enum Presentable {
    Surface(SurfaceTexture),
}

impl Presentable {
    pub fn present(self) {
        match self {
            Presentable::Surface(st) => st.present(),
        }
    }

    fn texture(&self) -> Option<&Texture> {
        match self {
            Presentable::Surface(st) => Some(&st.texture),
        }
    }
}

pub struct Frame {
    pub view: TextureView,
    pub encoder: CommandEncoder,
    pub(crate) presentable: Option<Presentable>,
}

impl Frame {
    pub(crate) fn finish(self, queue: &Queue) {
        queue.submit(Some(self.encoder.finish()));
        if let Some(p) = self.presentable {
            p.present();
        }
    }

    /// Submit the command buffer to the GPU without presenting.
    /// Returns the presentable surface (if any) for separate timing.
    pub(crate) fn submit(self, queue: &Queue) -> Option<Presentable> {
        let commands = self.encoder.finish();
        queue.submit(Some(commands));
        self.presentable
    }

    /// Finish the command encoder and submit separately for timing.
    /// Returns (CommandBuffer, Option<Presentable>).
    pub fn finish_encoder(self) -> (wgpu::CommandBuffer, Option<Presentable>) {
        let commands = self.encoder.finish();
        (commands, self.presentable)
    }

    /// Access the backbuffer texture for copy operations (screen capture).
    /// Returns `None` for offscreen targets without a presentable surface.
    pub fn backbuffer_texture(&self) -> Option<&Texture> {
        self.presentable.as_ref().and_then(|p| p.texture())
    }
}
