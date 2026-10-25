//! Standalone verify-forward kernel profile for the 35B DSpark path.
//!
//! Runs only verify forwards (no propose/accept phases) so the cubecl
//! per-kernel log attributes cleanly. The loop mirrors one dspark cycle at
//! acceptance 1: verify a 16-token block, then roll the caches back to the
//! bonus so the context stays flat across iterations.
//!
//! Run twice and aggregate the second run's log — the first run pays autotune
//! for the m=16 shapes, the second replays the cache:
//!
//! ```text
//! CUBECL_DEBUG_OPTION=profile CUBECL_DEBUG_LOG=1 cargo test --release \
//!   --features metal --test verify_profile -- --ignored --nocapture
//! ```
//!
//! `VERIFY_PROFILE_ITERS` (default 24) bounds the measured iterations. The
//! per-launch lines land in `/tmp/cubecl.log` as `| duration | Kernel | ...`.

use burn::backend::wgpu::{Metal, WgpuDevice};
use burn::tensor::backend::Backend as _;
use omni_minicpm_v::config::ModelDirectory;
use omni_minicpm_v::model::{TextModel, VerifyNumerics};
use omni_minicpm_v::multimodal::Prompt;

/// Mirrors `dspark::TARGET_LAYERS` (private there): the layers the draft
/// reads. Capturing them keeps the verify forward on the dspark code path.
const TARGET_LAYERS: [usize; 5] = [3, 11, 19, 27, 35];

#[test]
#[ignore = "diagnostic harness: loads the full 35B checkpoint"]
fn verify_forward_kernel_profile() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = root.join("models/MiniCPM-V-4.7-35B-A3B-int4");
    if !model_dir.is_dir() {
        eprintln!("skipping: checkpoint not present");
        return;
    }
    type B = Metal;
    let device = WgpuDevice::default();
    let directory = ModelDirectory::new(&model_dir);
    let config = directory.config().unwrap();
    let tokenizer = directory.tokenizer().unwrap();
    let weights = omni_r2t2::weights::Weights::open(&model_dir).unwrap();
    let model =
        TextModel::<B>::load(&weights, &config, omni_r2t2::Precision::F16, &device).unwrap();
    let prompt = Prompt::<B>::text(tokenizer.encode("用几句话说明分页 KV 缓存是什么。"));
    let mut state = model.new_state(&device);
    let (_, _, captured) = model
        .forward_capturing(&prompt.ids, None, None, &mut state, &TARGET_LAYERS, false)
        .unwrap();
    drop(captured);
    let block: Vec<u32> = (0..16)
        .map(|index| prompt.ids[index % prompt.ids.len()])
        .collect();
    let iterations: usize = std::env::var("VERIFY_PROFILE_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(24);
    // Two warmup forwards pay shape compilation so the measured window is
    // steady even without a warm autotune cache.
    for _ in 0..2 {
        let length_before = state.length;
        let (_, _, linear_after) = model
            .forward_verifying_with(&block, &mut state, &TARGET_LAYERS, VerifyNumerics::Batched)
            .unwrap();
        state.retain_accepted(length_before, 1, &linear_after);
    }
    B::sync(&device).unwrap();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let length_before = state.length;
        let (_, _, linear_after) = model
            .forward_verifying_with(&block, &mut state, &TARGET_LAYERS, VerifyNumerics::Batched)
            .unwrap();
        state.retain_accepted(length_before, 1, &linear_after);
    }
    B::sync(&device).unwrap();
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "verify profile: {iterations} forwards in {elapsed:.0} ms ({:.1} ms each, includes profiling overhead)",
        elapsed / iterations as f64
    );
}
