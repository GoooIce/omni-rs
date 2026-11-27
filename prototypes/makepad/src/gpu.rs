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

#[cfg(any(feature = "wgpu", feature = "metal"))]
use burn::backend::wgpu::{graphics, init_setup, WgpuDevice};

#[cfg(not(any(feature = "wgpu", feature = "metal", feature = "cpu")))]
compile_error!("r2t2-makepad requires at least one compute backend feature: metal, wgpu, or cpu");

// --- Role 1: Primary speech backend (ASR, TTS) ---

#[cfg(all(
    feature = "ane",
    feature = "metal",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub type Backend = burn::backend::Ane<burn::backend::wgpu::Metal>;

#[cfg(all(
    feature = "ane",
    feature = "metal",
    target_os = "macos",
    target_arch = "aarch64"
))]
pub type Device = burn_ane::AneDevice<burn::backend::wgpu::Metal>;

#[cfg(all(
    feature = "metal",
    target_os = "macos",
    not(all(feature = "ane", target_arch = "aarch64"))
))]
pub type Backend = burn::backend::wgpu::Metal;

#[cfg(all(
    feature = "metal",
    target_os = "macos",
    not(all(feature = "ane", target_arch = "aarch64"))
))]
pub type Device = WgpuDevice;

#[cfg(all(
    any(feature = "wgpu", feature = "metal"),
    not(all(feature = "metal", target_os = "macos"))
))]
pub type Backend = burn::backend::wgpu::Wgpu;

#[cfg(all(
    any(feature = "wgpu", feature = "metal"),
    not(all(feature = "metal", target_os = "macos"))
))]
pub type Device = WgpuDevice;

#[cfg(all(not(any(feature = "wgpu", feature = "metal")), feature = "cpu"))]
pub type Backend = burn::backend::NdArray;

#[cfg(all(not(any(feature = "wgpu", feature = "metal")), feature = "cpu"))]
pub type Device = burn::backend::ndarray::NdArrayDevice;

// --- Role 2: Non-ANE fused GPU backend (translation, image, music) ---

#[cfg(all(feature = "metal", target_os = "macos"))]
pub type GpuBackend = burn::backend::wgpu::Metal;

#[cfg(all(
    any(feature = "wgpu", feature = "metal"),
    not(all(feature = "metal", target_os = "macos"))
))]
pub type GpuBackend = burn::backend::wgpu::Wgpu;

#[cfg(all(not(any(feature = "wgpu", feature = "metal")), feature = "cpu"))]
pub type GpuBackend = burn::backend::NdArray;

#[cfg(any(feature = "wgpu", feature = "metal"))]
pub type GpuDevice = WgpuDevice;

#[cfg(all(not(any(feature = "wgpu", feature = "metal")), feature = "cpu"))]
pub type GpuDevice = burn::backend::ndarray::NdArrayDevice;

// --- Role 3: Eager non-fusion backend (LFM3 Omni, MiniCPM-V speech enhancer) ---

#[cfg(any(feature = "wgpu", feature = "metal"))]
pub type EagerBackend = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

#[cfg(all(not(any(feature = "wgpu", feature = "metal")), feature = "cpu"))]
pub type EagerBackend = burn::backend::NdArray;

#[cfg(any(feature = "wgpu", feature = "metal"))]
pub type EagerDevice = WgpuDevice;

#[cfg(all(not(any(feature = "wgpu", feature = "metal")), feature = "cpu"))]
pub type EagerDevice = burn::backend::ndarray::NdArrayDevice;

/// Both speech workers use one synchronized ANE session and Metal device on
/// Apple Silicon defaults, sharing the bounded program-slot pool. Portable
/// builds return the configured WGPU or CPU device.
pub fn device() -> Device {
    #[cfg(all(
        feature = "ane",
        feature = "metal",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    {
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
    #[cfg(not(all(
        feature = "ane",
        feature = "metal",
        target_os = "macos",
        target_arch = "aarch64"
    )))]
    {
        Device::default()
    }
}

pub fn gpu_device() -> GpuDevice {
    GpuDevice::default()
}

pub fn eager_device() -> EagerDevice {
    EagerDevice::default()
}

pub const fn default_precision() -> omni_r2t2::Precision {
    #[cfg(any(feature = "wgpu", feature = "metal"))]
    {
        omni_r2t2::Precision::F16
    }
    #[cfg(all(not(any(feature = "wgpu", feature = "metal")), feature = "cpu"))]
    {
        omni_r2t2::Precision::F32
    }
}

#[cfg(all(
    feature = "ane",
    feature = "metal",
    target_os = "macos",
    target_arch = "aarch64"
))]
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
    #[cfg(all(
        feature = "ane",
        feature = "metal",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    {
        "ANE + Metal · FP16"
    }
    #[cfg(all(
        feature = "metal",
        target_os = "macos",
        not(all(feature = "ane", target_arch = "aarch64"))
    ))]
    {
        "Metal · FP16"
    }
    #[cfg(all(
        any(feature = "wgpu", feature = "metal"),
        not(all(feature = "metal", target_os = "macos"))
    ))]
    {
        "WGPU · FP16"
    }
    #[cfg(all(not(any(feature = "wgpu", feature = "metal")), feature = "cpu"))]
    {
        "CPU · FP32"
    }
}

pub const fn initial_backend_label() -> &'static str {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        "Metal · 本机推理"
    }
    #[cfg(all(
        any(feature = "wgpu", feature = "metal"),
        not(all(feature = "metal", target_os = "macos"))
    ))]
    {
        "WGPU · 本机推理"
    }
    #[cfg(all(not(any(feature = "wgpu", feature = "metal")), feature = "cpu"))]
    {
        "CPU · 本机推理"
    }
}

pub fn log_stats(device: &Device, engine: &str) {
    #[cfg(all(
        feature = "ane",
        feature = "metal",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    eprintln!(
        "[accelerator:{engine}] {}",
        serde_json::to_string(&device.stats()).unwrap()
    );
    #[cfg(not(all(
        feature = "ane",
        feature = "metal",
        target_os = "macos",
        target_arch = "aarch64"
    )))]
    let _ = (device, engine);
}

/// Workers share the same Burn device registration.
pub fn initialize() {
    #[cfg(any(feature = "wgpu", feature = "metal"))]
    {
        static INITIALIZED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        INITIALIZED.get_or_init(|| {
            let device = WgpuDevice::default();
            #[cfg(all(feature = "metal", target_os = "macos"))]
            let setup = init_setup::<graphics::Metal>(&device, Default::default());
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            let setup = init_setup::<graphics::AutoGraphicsApi>(&device, Default::default());
            eprintln!("[engine] {:?}", setup.adapter.get_info());
        });
    }
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
