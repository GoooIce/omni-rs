use burn::backend::wgpu::{graphics, init_setup, WgpuDevice};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub type Backend = burn::backend::Ane<burn::backend::wgpu::Metal>;
#[cfg(all(target_os = "macos", not(target_arch = "aarch64")))]
pub type Backend = burn::backend::wgpu::Metal;
#[cfg(not(target_os = "macos"))]
pub type Backend = burn::backend::wgpu::WebGpu;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub type Device = burn_ane::AneDevice<burn::backend::wgpu::Metal>;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub type Device = WgpuDevice;

/// Both workers use one synchronized ANE session and Metal device. They share
/// the bounded program-slot pool, so an idle engine cannot reserve half the ANE.
pub fn device() -> Device {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
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
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        WgpuDevice::default()
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
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

pub fn backend_label() -> &'static str {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        "ANE + Metal · FP16"
    } else if cfg!(target_os = "macos") {
        "Metal · 本机推理"
    } else {
        "WGPU · 本机推理"
    }
}

pub fn log_stats(device: &Device, engine: &str) {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    eprintln!(
        "[accelerator:{engine}] {}",
        serde_json::to_string(&device.stats()).unwrap()
    );
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let _ = (device, engine);
}

/// ASR and TTS share the same Burn device registration.
pub fn initialize() {
    static INITIALIZED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    INITIALIZED.get_or_init(|| {
        let device = WgpuDevice::default();
        #[cfg(target_os = "macos")]
        let setup = init_setup::<graphics::Metal>(&device, Default::default());
        #[cfg(not(target_os = "macos"))]
        let setup = init_setup::<graphics::AutoGraphicsApi>(&device, Default::default());
        eprintln!("[engine] {:?}", setup.adapter.get_info());
    });
}
