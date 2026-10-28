//! Quota-constrained conversation turn gate (#76) on the 1B CPU checkpoint.
//!
//! Protocol (#66): a released turn checkpoint is the idle KV the quota
//! releases. With `state_cache` on it must be spilled before being dropped,
//! and the next turn must still resume it (from SSD) while the request
//! completes under the budget. Live slots are not evicted, so the over-budget
//! remainder is allowed to be reported. Requires the local 1B checkpoint;
//! skips otherwise. Diagnostic gate, not a benchmark.
//!
//! ```text
//! cargo test --manifest-path crates/resin-metal/Cargo.toml \
//!   --no-default-features --features cpu --test minicpm_quota -- \
//!   --ignored --nocapture
//! ```

#![cfg(feature = "cpu")]

use std::fs;

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use omni_minicpm_v::config::ModelDirectory;
use omni_minicpm_v::model::TextModel;
use omni_minicpm_v::tokenizer::MiniCpmTokenizer;
use resin_core::engine::{
    EngineCore, ExecutionPlan, OwnedTokenRows, Runner, RunnerConfig, SchedulerConfig, Submission,
};
use resin_metal::config::ResinConfig;
use resin_metal::minicpm::MinicpmRunner;
use resin_metal::session::DecodeRequest;
use resin_metal::state_cache::{STATE_LAYOUT_VERSION, StateCache, StateCacheConfig, namespace_for};

type B = NdArray;
const MAX_NEW_TOKENS: usize = 4;
const SLOT: u64 = 1;

fn prompt_ids(tokenizer: &MiniCpmTokenizer, tokens: usize) -> Vec<u32> {
    let seed = "分页缓存把键值对按页保存，引擎在请求之间复用已经算好的前缀。受限合批在\
                内存顶到上限时按次序释放空闲检查点，先落盘再丢弃。";
    let mut ids = tokenizer.encode(seed);
    assert!(!ids.is_empty(), "tokenizer produced no ids");
    while ids.len() < tokens {
        ids.extend_from_slice(&tokenizer.encode(seed));
    }
    ids.truncate(tokens);
    ids
}

fn run_request(
    engine: &mut EngineCore<DecodeRequest>,
    runner: &mut MinicpmRunner<B>,
    prompt: Vec<u32>,
    slot: Option<u64>,
    new_tokens: usize,
    eos: &[u32],
) -> Vec<u32> {
    let mut input = Submission::new(
        prompt,
        DecodeRequest {
            max_new_tokens: new_tokens,
            slot,
        },
    );
    input.max_output_tokens = Some(new_tokens);
    input.stop_token_ids = eos.to_vec();
    let handle = engine.submit(input).unwrap();
    while engine.has_work() {
        let plan: ExecutionPlan<DecodeRequest> =
            engine.prepare_step(runner, None).unwrap().unwrap();
        let rows: OwnedTokenRows = runner.forward(plan).unwrap();
        engine
            .complete_step_with(runner, Ok(rows.view()), Ok)
            .unwrap();
    }
    handle.take_result().unwrap()
}

#[test]
#[ignore = "1B CPU quota gate: loads the full checkpoint and exercises KV eviction"]
fn quota_bounded_turn_completes_and_reloads_the_evicted_prefix() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = root.join("models/MiniCPM-V-4.7-1B");
    if !model_dir.is_dir() {
        eprintln!("skipping: checkpoint not present");
        return;
    }
    let cache_dir = std::env::temp_dir().join(format!(
        "resin-quota-gate-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _ = fs::remove_dir_all(&cache_dir);

    let device = NdArrayDevice::default();
    let directory = ModelDirectory::new(&model_dir);
    let config = directory.config().unwrap();
    let tokenizer = MiniCpmTokenizer::from_directory(&model_dir).unwrap();
    let eos = directory.eos_ids();
    let weights = omni_r2t2::weights::Weights::open(&model_dir).unwrap();
    let model =
        TextModel::<B>::load(&weights, &config, omni_r2t2::Precision::F32, &device).unwrap();
    let namespace = namespace_for(&model_dir, None, STATE_LAYOUT_VERSION, 0).unwrap();
    let cache = StateCache::open(StateCacheConfig::new(&cache_dir, namespace).with_quota(u64::MAX))
        .unwrap();
    let mut runner = MinicpmRunner::with_state_cache(
        model,
        device,
        ResinConfig {
            conversation_slots: true,
            state_cache: true,
            ..ResinConfig::default()
        },
        Some(cache),
    );
    let mut engine = EngineCore::new(
        RunnerConfig::default(),
        4,
        8192,
        SchedulerConfig::default(),
        false,
    )
    .unwrap();

    // Turn one seeds the memory checkpoint and its SSD copy.
    let prompt = prompt_ids(&tokenizer, 48);
    let first = run_request(
        &mut engine,
        &mut runner,
        prompt.clone(),
        Some(SLOT),
        1,
        &eos,
    );
    assert!(!first.is_empty(), "seed turn emitted nothing");
    runner.flush_state_cache();
    let seeded = runner.memory_stats();
    assert!(seeded.checkpoint_bytes > 0, "no checkpoint to evict");
    if seeded.checkpoint_bytes <= 1 << 20 {
        eprintln!(
            "skipping: checkpoint ({} bytes) is below the 1 MiB switch granularity",
            seeded.checkpoint_bytes
        );
        let _ = fs::remove_dir_all(&cache_dir);
        return;
    }
    let budget_mb = usize::try_from(seeded.checkpoint_bytes / 2 / (1 << 20))
        .unwrap()
        .max(1);

    // Lowering the budget releases the idle checkpoint; with the cache on it
    // is spilled, not dropped.
    runner.set_config(ResinConfig {
        conversation_slots: true,
        state_cache: true,
        kv_budget_mb: Some(budget_mb),
        ..ResinConfig::default()
    });
    let evicted = runner.memory_stats();
    assert_eq!(evicted.live_bytes, 0);
    assert_eq!(
        evicted.over_budget_bytes, 0,
        "checkpoints exist, so the quota can be met without live slots"
    );
    assert!(
        evicted.spilled_checkpoints > 0 || evicted.dropped_idle_checkpoints > 0,
        "the quota did not release any checkpoint"
    );

    // The next turn resumes the spilled prefix from SSD and completes.
    let mut prompt2 = prompt.clone();
    prompt2.extend_from_slice(&tokenizer.encode("再举一个例子。"));
    let before = runner.conversation_stats();
    let second = run_request(
        &mut engine,
        &mut runner,
        prompt2.clone(),
        Some(SLOT),
        MAX_NEW_TOKENS,
        &eos,
    );
    assert!(!second.is_empty(), "quota-bounded turn emitted nothing");
    let stats = runner.conversation_stats();
    assert!(
        stats.checkpoint_hits > before.checkpoint_hits,
        "the second turn did not resume a checkpoint"
    );
    assert!(
        stats.disk_hits > before.disk_hits,
        "the evicted checkpoint did not reload from SSD"
    );
    assert!(
        (stats.prefill_tokens - before.prefill_tokens) < prompt2.len() as u64,
        "the resumed turn prefilled the whole prompt"
    );
    eprintln!(
        "quota gate: seeded_checkpoint_bytes={} spilled={} dropped_idle={} \
         over_budget_bytes={} hits={} disk_hits={} second_prefill={} prompt={}",
        seeded.checkpoint_bytes,
        evicted.spilled_checkpoints,
        evicted.dropped_idle_checkpoints,
        runner.memory_stats().over_budget_bytes,
        stats.checkpoint_hits - before.checkpoint_hits,
        stats.disk_hits - before.disk_hits,
        stats.prefill_tokens - before.prefill_tokens,
        prompt2.len()
    );
    let _ = fs::remove_dir_all(&cache_dir);
}
