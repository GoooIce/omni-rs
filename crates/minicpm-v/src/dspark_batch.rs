//! Batched speculative cycle: several slots share one target verify (#72).
//!
//! [`crate::dspark::run_cycle`] owns the single-sequence cycle (propose →
//! verify → accept → retain). This module splits the same phases around
//! [`TextModel::forward_verifying_multi`](crate::model::TextModel::forward_verifying_multi)
//! so every decode row of one resin plan is verified by one target forward.
//! Draft work stays per sequence — each member proposes its own chain, exactly
//! like the per-row path — and acceptance, trimming and cache retention run
//! once per member in plan order.
//!
//! Greedy only: acceptance reads each verify row's argmax, matching
//! `run_cycle`'s greedy branch. Callers with a temperature or a penalty stay
//! on the sequential path. Chain layout only: tree blocks (#70/#73) need the
//! ancestor-masked verify and stay out of this seam.

use burn::tensor::backend::Backend;
use std::time::Instant;

use crate::dspark::{
    AcceptedPath, CycleOutcome, Draft, DraftState, SpecStats, accept_path, greedy_argmax_block,
    select_captured, trim_to_room,
};
use crate::model::{DecodeState, TextModel, VerifyBlock};

/// One sequence's slice of a batched speculative cycle.
pub struct BatchCycleMember<'a, B: Backend> {
    /// This sequence's draft cross-attention context.
    pub draft_state: &'a mut DraftState<B>,
    /// This sequence's target KV / linear-attention history.
    pub state: &'a mut DecodeState<B>,
    /// The last committed target token the cycle proposes from.
    pub bonus: u32,
    /// Remaining emission room for this row.
    pub room: usize,
}

/// Propose per member, verify all members in one forward, then accept, trim
/// and retain per member in plan order. Returns one outcome per member.
///
/// The target states are left exactly where a per-row `run_cycle` would leave
/// them: each block's state ends at its accepted path (`state.length +=
/// path.slots.len()`), and the draft context absorbs the accepted captures.
/// Phase timings accumulate into `stats.trace` when enabled; per-member cycle
/// counters stay with the caller so both paths account alike.
pub fn run_cycle_batch<B: Backend>(
    model: &TextModel<B>,
    draft: &Draft<B>,
    block_len: usize,
    eos_ids: &[u32],
    members: &mut [BatchCycleMember<'_, B>],
    stats: &mut SpecStats,
) -> Result<Vec<CycleOutcome>, String> {
    if members.is_empty() {
        return Err("batched cycle needs at least one member".into());
    }
    let phase = Instant::now();
    let mut blocks: Vec<Vec<u32>> = Vec::with_capacity(members.len());
    for member in members.iter_mut() {
        let proposals = draft
            .propose(model, member.draft_state, member.bonus, block_len)
            .map_err(|error| error.to_string())?;
        let mut block = Vec::with_capacity(1 + proposals.len());
        block.push(member.bonus);
        block.extend_from_slice(&proposals);
        blocks.push(block);
    }
    if stats.trace.enabled {
        stats.trace.propose_ms += phase.elapsed().as_secs_f64() * 1000.0;
    }

    let length_before: Vec<usize> = members.iter().map(|member| member.state.length).collect();
    let rows: Vec<usize> = blocks.iter().map(Vec::len).collect();
    let phase = Instant::now();
    let outputs = {
        let mut verify: Vec<VerifyBlock<'_, B>> = members
            .iter_mut()
            .zip(blocks.iter())
            .map(|(member, ids)| VerifyBlock::new(ids, member.state, crate::knobs::dspark_layers()))
            .collect();
        model
            .forward_verifying_multi(&mut verify)
            .map_err(|error| error.to_string())?
    };
    let verify_ms = phase.elapsed().as_secs_f64() * 1000.0;
    if stats.trace.enabled {
        stats.trace.verify_ms += verify_ms;
    }

    let phase = Instant::now();
    let mut paths: Vec<(AcceptedPath, bool)> = Vec::with_capacity(outputs.len());
    for ((member, block), output) in members.iter_mut().zip(blocks.iter()).zip(outputs.iter()) {
        let prediction = greedy_argmax_block(&output.logits)?;
        let mut path = accept_path(None, &block[1..], eos_ids, &mut |index: usize| {
            Ok(prediction[index])
        })?;
        let trimmed = trim_to_room(&mut path, member.room);
        paths.push((path, trimmed));
    }
    if stats.trace.enabled {
        stats.trace.accept_ms += phase.elapsed().as_secs_f64() * 1000.0;
    }

    let phase = Instant::now();
    let mut outcomes = Vec::with_capacity(outputs.len());
    for (index, ((member, (path, trimmed)), output)) in
        members.iter_mut().zip(paths).zip(outputs).enumerate()
    {
        member
            .state
            .retain_path(length_before[index], &path.slots, &output.linear_after);
        draft
            .extend(
                member.draft_state,
                select_captured(output.captured, &path.slots),
            )
            .map_err(|error| error.to_string())?;
        outcomes.push(CycleOutcome {
            tokens: path.tokens,
            correction: path.correction,
            draft_eos: path.draft_eos,
            trimmed,
            verify_ms,
            rows: rows[index],
        });
    }
    if stats.trace.enabled {
        stats.trace.retain_ms += phase.elapsed().as_secs_f64() * 1000.0;
    }
    Ok(outcomes)
}
