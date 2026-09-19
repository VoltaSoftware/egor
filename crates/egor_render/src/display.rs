use std::{fmt, sync::Arc};
use wgpu::{
    DisplayAndWindowHandle, InstanceDescriptor, SurfaceTarget,
    rwh::{DisplayHandle, HandleError, HasDisplayHandle, RawWindowHandle},
};

struct WindowDisplay(Arc<dyn DisplayAndWindowHandle>);

impl fmt::Debug for WindowDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WindowDisplay").finish_non_exhaustive()
    }
}

impl HasDisplayHandle for WindowDisplay {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        self.0.display_handle()
    }
}

pub(super) fn attach_to_instance(
    desc: &mut InstanceDescriptor,
    target: SurfaceTarget<'static>,
) -> SurfaceTarget<'static> {
    let SurfaceTarget::DisplayAndWindow(window) = target else {
        return target;
    };
    // EGL selects its presentation platform before surface creation. Keep the
    // same display alive through both the instance and the surface.
    let window: Arc<dyn DisplayAndWindowHandle> = Arc::from(window);
    desc.backend_options.gl.egl_native_visual_id =
        window
            .window_handle()
            .ok()
            .and_then(|handle| match handle.as_raw() {
                RawWindowHandle::Xlib(handle) => {
                    u32::try_from(handle.visual_id).ok().filter(|&id| id != 0)
                }
                RawWindowHandle::Xcb(handle) => handle.visual_id.map(|id| id.get()),
                _ => None,
            });
    desc.display = Some(Box::new(WindowDisplay(Arc::clone(&window))));
    SurfaceTarget::DisplayAndWindow(Box::new(window))
}
