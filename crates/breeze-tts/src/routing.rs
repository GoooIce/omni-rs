//! Keep TTS phases on Metal when using the measured latency policy. Both short
//! and long prefills, token decoding and streaming codec were faster there on
//! the target M5. Explicit ANE-first / strict sessions retain ANE execution.
use burn::tensor::backend::Backend;

#[cfg(all(feature = "ane", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn metal_phase<B: Backend>(device: &B::Device) -> Option<burn_ane::FallbackScope> {
    use std::any::Any;
    let device =
        (device as &dyn Any).downcast_ref::<burn_ane::AneDevice<burn::backend::wgpu::Metal>>()?;
    if !device.prefers_fallback_phases() {
        return None;
    }
    Some(device.fallback_scope().expect("Metal phase handoff failed"))
}

#[cfg(not(all(feature = "ane", target_os = "macos", target_arch = "aarch64")))]
pub(crate) fn metal_phase<B: Backend>(_device: &B::Device) {}
