//! EngineCore integration against the vendored resin-core, using a stub
//! runner. These tests pin the control-plane behavior every real runner
//! depends on: continuous batching, prefix-cache reuse and cancellation.
//! The stub mirrors the runner contract from the upstream engine tests —
//! deterministic `last token + 1` outputs — so assertions are exact.

use resin_core::Phase;
use resin_core::engine::{
    EngineCore, Error, ExecutionPlan, OwnedTokenRows, Runner, RunnerConfig, SchedulerConfig,
    Submission,
};
use resin_core::resources::kv::{KvGroupLayout, KvLayout, KvRetention};
use std::collections::VecDeque;
use std::num::NonZeroU32;

const BLOCK_TOKENS: u32 = 2;
// The pool must hold the retained published prefix of the seed request
// alongside the follow-up's new pages: 16 retained blocks plus headroom.
const PARENT_CAPACITY: u32 = 64;

#[derive(Default)]
struct StubRunner {
    plans: Vec<ExecutionPlan<&'static str>>,
    outputs: VecDeque<Result<Vec<Vec<u32>>, Error>>,
    released: Vec<u64>,
}

impl Runner<&'static str> for StubRunner {
    fn forward(&mut self, plan: ExecutionPlan<&'static str>) -> Result<OwnedTokenRows, Error> {
        let fallback: Vec<Vec<u32>> = plan
            .token_indptr
            .windows(2)
            .map(|range| vec![plan.token_ids[range[1] as usize - 1] + 1])
            .collect();
        self.plans.push(plan);
        self.outputs
            .pop_front()
            .unwrap_or(Ok(fallback))
            .map(owned_rows)
    }

    fn release(&mut self, request_ids: &[u64]) -> Result<(), Error> {
        self.released.extend_from_slice(request_ids);
        Ok(())
    }
}

fn owned_rows(rows: Vec<Vec<u32>>) -> OwnedTokenRows {
    let mut token_ids = Vec::new();
    let mut indptr = vec![0];
    for row in rows {
        token_ids.extend(row);
        indptr.push(u32::try_from(token_ids.len()).unwrap());
    }
    OwnedTokenRows::new(token_ids, indptr).unwrap()
}

fn engine(prefix_cache: bool) -> EngineCore<&'static str> {
    EngineCore::new(
        RunnerConfig {
            cache_layout: Some(
                KvLayout::new(
                    PARENT_CAPACITY,
                    vec![KvGroupLayout::new(BLOCK_TOKENS, 1, KvRetention::FullContext).unwrap()],
                )
                .unwrap(),
            ),
            continuation_prefill: true,
            ..Default::default()
        },
        4,
        128,
        SchedulerConfig {
            prefill_chunk_size: NonZeroU32::new(64).unwrap(),
            ..Default::default()
        },
        prefix_cache,
    )
    .unwrap()
}

fn input(tokens: Vec<u32>, limit: usize) -> Submission<&'static str> {
    let mut input = Submission::new(tokens, "opaque model metadata");
    input.max_output_tokens = Some(limit);
    input
}

fn drain(engine: &mut EngineCore<&'static str>, runner: &mut StubRunner) {
    while engine.has_work() {
        engine.step(runner).unwrap();
    }
}

#[test]
fn concurrent_prefills_share_one_forward_batch() {
    let mut engine = engine(false);
    let mut runner = StubRunner::default();
    let first = engine.submit(input((0..8).collect(), 2)).unwrap();
    let second = engine.submit(input((100..108).collect(), 2)).unwrap();

    assert!(engine.has_work());
    drain(&mut engine, &mut runner);
    assert!(!engine.has_work());

    // Both prompts fit in one prefill chunk, so the first forward plan must
    // carry both sequences in a single batch.
    let first_plan = runner.plans.first().unwrap();
    assert_eq!(first_plan.phase, Phase::Prefill);
    assert_eq!(first_plan.sequence_ids.len(), 2);
    assert_eq!(first_plan.token_indptr.len(), 3);

    // Stub outputs are deterministic: prompt (0..8] yields [8, 9].
    assert_eq!(first.take_result().unwrap(), vec![8, 9]);
    assert_eq!(second.take_result().unwrap(), vec![108, 109]);
}

#[test]
fn shared_prefix_restores_cached_tokens() {
    let mut engine = engine(true);
    let mut runner = StubRunner::default();

    // A 32-token prompt fills exactly 16 whole blocks; the single generated
    // token lands in a partial block that cannot be published. The follow-up
    // prompt reuses the 32 cached tokens and prefills only its 3-token suffix.
    let prefix: Vec<u32> = (0..32).collect();
    let seed = engine.submit(input(prefix.clone(), 1)).unwrap();
    drain(&mut engine, &mut runner);
    seed.take_result().unwrap();

    let mut prompt = prefix;
    prompt.extend([200, 201, 202]);
    let handle = engine.submit(input(prompt, 1)).unwrap();
    drain(&mut engine, &mut runner);

    assert_eq!(handle.cached_tokens(), 32);
    let mut followup_plans = runner
        .plans
        .iter()
        .filter(|plan| plan.sequence_ids.contains(&handle.id()));
    // The first follow-up forward resumes at context position 32: the cached
    // prefix is already resident, so only the suffix is prefilled. The final
    // prompt token gets its own forward because sampling happens at prefill
    // end, so the suffix totals three fed tokens across its plans.
    let first = followup_plans.next().unwrap();
    assert_eq!(first.phase, Phase::Prefill);
    assert_eq!(first.context_lengths, vec![32]);
    let prefill_tokens: usize = std::iter::once(first)
        .chain(followup_plans)
        .filter(|plan| plan.phase == Phase::Prefill)
        .map(|plan| plan.token_ids.len())
        .sum();
    assert_eq!(prefill_tokens, 3);
    assert_eq!(handle.take_result().unwrap(), vec![203]);
}

#[test]
fn cancel_mid_decode_releases_pages_and_reports_cancelled() {
    let mut engine = engine(true);
    let mut runner = StubRunner::default();
    let handle = engine.submit(input((0..8).collect(), 100)).unwrap();

    // Finish the prefill forward so the request owns resident pages, then
    // cancel before the decode drains to the token limit.
    engine.step(&mut runner).unwrap();
    assert!(!handle.done());
    assert!(engine.cancel(&mut runner, &handle).unwrap());

    drain(&mut engine, &mut runner);
    assert!(!engine.has_work());
    assert!(handle.done());
    assert!(handle.cancelled());
    assert!(runner.released.contains(&handle.id()));
}

#[test]
fn async_step_path_drives_the_same_results() {
    // The overlap seam: freeze a plan, execute outside the engine call, then
    // harvest through the atomic commit. Real runners enqueue device work in
    // the gap between prepare and complete.
    let mut engine = engine(false);
    let mut runner = StubRunner::default();
    let handle = engine.submit(input((0..8).collect(), 2)).unwrap();
    assert!(!engine.has_in_flight_step());
    while engine.has_work() {
        let plan = engine.prepare_step(&mut runner, None).unwrap().unwrap();
        assert!(engine.has_in_flight_step());
        let rows = runner.forward(plan).unwrap();
        engine
            .complete_step_with(&mut runner, Ok(rows.view()), Ok)
            .unwrap();
        assert!(!engine.has_in_flight_step());
    }
    assert!(!engine.has_work());
    assert_eq!(handle.take_result().unwrap(), vec![8, 9]);
}
