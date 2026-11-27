//! Single backend/device selection seam for all desktop inference workers.
//!
//! Three backend roles are exposed so every engine reads its backend, device,
//! default precision and display labels from one place:
//! - [`Backend`] / [`Device`] / [`device`]: primary speech backend (ANE + Metal
//!   on Apple Silicon defaults, Metal/WGPU on GPU-only builds, NdArray on CPU).
//! - [`GpuBackend`] / [`GpuDevice`] / [`gpu_device`]: fused GPU backend used by
//!   translation, image and music workers (or NdArray on CPU builds).
//! - [`EagerBackend`] / [`EagerDevice`] / [`eager_device`]: non-fusion cube
//!   backend used by LFM3 Omni and the MiniCPM-V speech enhancer (or NdArray on
//!   CPU builds), avoiding burn-fusion dtype-mix panics on int4 / FP16+FP32 ops.

#[cfg(not(any(feature = "wgpu", feature = "metal", feature = "cpu")))]
compile_error!("r2t2-makepad requires at least one compute backend feature: metal, wgpu, or cpu");

#[cfg(any(feature = "wgpu", feature = "metal"))]
mod wgpu_common {
    use burn::backend::wgpu::{graphics, init_setup, WgpuDevice};

    pub type EagerBackend = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;
    pub type GpuDevice = WgpuDevice;
    pub type EagerDevice = WgpuDevice;

    pub const fn default_precision() -> omni_r2t2::Precision {
        omni_r2t2::Precision::F16
    }

    pub fn initialize<G: graphics::GraphicsApi>() {
        static INITIALIZED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        INITIALIZED.get_or_init(|| {
            let device = WgpuDevice::default();
            let setup = init_setup::<G>(&device, Default::default());
            eprintln!("[engine] {:?}", setup.adapter.get_info());
        });
    }
}

#[cfg(all(
    feature = "ane",
    feature = "metal",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod imp {
    use super::wgpu_common;
    use burn::backend::wgpu::{graphics, WgpuDevice};

    pub type Backend = burn::backend::Ane<burn::backend::wgpu::Metal>;
    pub type Device = burn_ane::AneDevice<burn::backend::wgpu::Metal>;
    pub type GpuBackend = burn::backend::wgpu::Metal;
    pub use wgpu_common::{default_precision, EagerBackend, EagerDevice, GpuDevice};

    /// Both speech workers use one synchronized ANE session and Metal device on
    /// Apple Silicon defaults, sharing the bounded program-slot pool.
    pub fn device() -> Device {
        static DEVICE: std::sync::OnceLock<Device> = std::sync::OnceLock::new();
        DEVICE
            .get_or_init(|| {
                burn_ane::AneDevice::new(
                    WgpuDevice::default(),
                    burn_ane::AneOptions {
                        runtime: burn_ane::AneRuntime::Direct,
                        fallback: burn_ane::FallbackPolicy::Allow,
                        // Explicit diagnostic override retains the old policy;
                        // normal UI operation uses the measured latency policy.
                        ane_first: std::env::var("OMNI_ANE_FIRST").as_deref() == Ok("1"),
                        prefer_fallback_phases: true,
                        min_compute_rows: std::env::var("OMNI_ANE_MIN_ROWS")
                            .ok()
                            .and_then(|v| v.parse::<usize>().ok())
                            .filter(|v| (1..=2048).contains(v))
                            .unwrap_or(8),
                        // One process-wide limit covers ASR, TTS and reference encoding.
                        // LRU lets active decoding displace an idle model's programs.
                        max_loaded_direct_graphs: 112,
                        max_cached_graphs: 2048,
                        program_cache_policy: burn_ane::ProgramCachePolicy::Lru,
                        max_cached_bytes: native_cache_budget(),
                        max_cached_weight_bytes: native_cache_budget(),
                        ..Default::default()
                    },
                )
            })
            .clone()
    }

    fn native_cache_budget() -> usize {
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const std::ffi::c_char,
                old: *mut std::ffi::c_void,
                old_len: *mut usize,
                new: *mut std::ffi::c_void,
                new_len: usize,
            ) -> i32;
        }
        let mut memory = 0u64;
        let mut len = std::mem::size_of_val(&memory);
        // The descriptor and packed-weight budgets together use at most one quarter
        // of physical RAM. GPU model tensors/native runtime allocations
        // are additional; this is a cache bound, not a total-process memory limit.
        let ok = unsafe {
            sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&mut memory as *mut u64).cast(),
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        } == 0;
        let gib = 1024 * 1024 * 1024;
        if ok && len == 8 {
            (memory as usize / 8).clamp(gib, 16 * gib)
        } else {
            2 * gib
        }
    }

    pub const fn backend_label() -> &'static str {
        "ANE + Metal · FP16"
    }

    pub const fn initial_backend_label() -> &'static str {
        "Metal · 本机推理"
    }

    pub fn log_stats(device: &Device, engine: &str) {
        eprintln!(
            "[accelerator:{engine}] {}",
            serde_json::to_string(&device.stats()).unwrap()
        );
    }

    pub fn initialize() {
        wgpu_common::initialize::<graphics::Metal>();
    }
}

#[cfg(all(
    feature = "metal",
    target_os = "macos",
    not(all(feature = "ane", target_arch = "aarch64"))
))]
mod imp {
    use super::wgpu_common;
    use burn::backend::wgpu::{graphics, WgpuDevice};

    pub type Backend = burn::backend::wgpu::Metal;
    pub type Device = WgpuDevice;
    pub type GpuBackend = burn::backend::wgpu::Metal;
    pub use wgpu_common::{default_precision, EagerBackend, EagerDevice, GpuDevice};

    pub fn device() -> Device {
        Device::default()
    }

    pub const fn backend_label() -> &'static str {
        "Metal · FP16"
    }

    pub const fn initial_backend_label() -> &'static str {
        "Metal · 本机推理"
    }

    pub fn log_stats(_device: &Device, _engine: &str) {}

    pub fn initialize() {
        wgpu_common::initialize::<graphics::Metal>();
    }
}

#[cfg(all(
    any(feature = "wgpu", feature = "metal"),
    not(all(feature = "metal", target_os = "macos"))
))]
mod imp {
    use super::wgpu_common;
    use burn::backend::wgpu::{graphics, WgpuDevice};

    pub type Backend = burn::backend::wgpu::Wgpu;
    pub type Device = WgpuDevice;
    pub type GpuBackend = burn::backend::wgpu::Wgpu;
    pub use wgpu_common::{default_precision, EagerBackend, EagerDevice, GpuDevice};

    pub fn device() -> Device {
        Device::default()
    }

    pub const fn backend_label() -> &'static str {
        "WGPU · FP16"
    }

    pub const fn initial_backend_label() -> &'static str {
        "WGPU · 本机推理"
    }

    pub fn log_stats(_device: &Device, _engine: &str) {}

    pub fn initialize() {
        wgpu_common::initialize::<graphics::AutoGraphicsApi>();
    }
}

#[cfg(all(not(any(feature = "wgpu", feature = "metal")), feature = "cpu"))]
mod imp {
    pub type Backend = burn::backend::NdArray;
    pub type Device = burn::backend::ndarray::NdArrayDevice;
    pub type GpuBackend = burn::backend::NdArray;
    pub type GpuDevice = burn::backend::ndarray::NdArrayDevice;
    pub type EagerBackend = burn::backend::NdArray;
    pub type EagerDevice = burn::backend::ndarray::NdArrayDevice;

    pub fn device() -> Device {
        Device::default()
    }

    pub const fn default_precision() -> omni_r2t2::Precision {
        omni_r2t2::Precision::F32
    }

    pub const fn backend_label() -> &'static str {
        "CPU · FP32"
    }

    pub const fn initial_backend_label() -> &'static str {
        "CPU · 本机推理"
    }

    pub fn log_stats(_device: &Device, _engine: &str) {}

    pub fn initialize() {}
}

#[allow(unused_imports)]
pub use imp::{
    backend_label, default_precision, device, initial_backend_label, initialize, log_stats,
    Backend, Device, EagerBackend, EagerDevice, GpuBackend, GpuDevice,
};

pub fn gpu_device() -> GpuDevice {
    GpuDevice::default()
}

pub fn eager_device() -> EagerDevice {
    EagerDevice::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_selection_labels_and_precision_match_compiled_features() {
        let _ = (device(), gpu_device(), eager_device());
        #[cfg(all(
            feature = "ane",
            feature = "metal",
            target_os = "macos",
            target_arch = "aarch64"
        ))]
        {
            assert_eq!(backend_label(), "ANE + Metal · FP16");
            assert_eq!(initial_backend_label(), "Metal · 本机推理");
            assert_eq!(default_precision(), omni_r2t2::Precision::F16);
        }
        #[cfg(all(
            any(feature = "wgpu", feature = "metal"),
            not(all(feature = "metal", target_os = "macos"))
        ))]
        {
            assert_eq!(backend_label(), "WGPU · FP16");
            assert_eq!(initial_backend_label(), "WGPU · 本机推理");
            assert_eq!(default_precision(), omni_r2t2::Precision::F16);
        }
        #[cfg(all(not(any(feature = "wgpu", feature = "metal")), feature = "cpu"))]
        {
            assert_eq!(backend_label(), "CPU · FP32");
            assert_eq!(initial_backend_label(), "CPU · 本机推理");
            assert_eq!(default_precision(), omni_r2t2::Precision::F32);
        }
    }
}
