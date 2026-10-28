//! Batched speculative cycle: several slots share one target verify (#72).
//!
//! [`crate::dspark::run_cycle`] owns the single-sequence cycle (propose →
//! verify → accept → retain). This module splits the same phases around
//! [`TextModel::forward_verifying_multi`](crate::model::TextModel::forward_verifying_multi)
//! so every decode row of one resin plan is verified by one target forward.
//! Draft work stays per sequence — each member proposes its own block, exactly
//! like the per-row path — and acceptance, trimming and cache retention run
//! once per member in plan order.
//!
//! Every member carries its own [`BlockLayout`], so one target forward can
//! verify mixed chain and tree blocks: tree members take the same
//! [`VerifyBlock::with_tree`] contract as the single-sequence tree path —
//! per-sequence positions, per-sequence path states, `retain_path` with the
//! accepted tree slots. The resin runner passes a chain layout for every
//! member (the production path); the per-member seam stays so the
//! multi-sequence tree support covered by the model tests keeps its
//! one-forward contract. The composite tree+batch mode was measured against
//! #67's gate and removed (#73): per accepted token it was 1.64x slower than
//! chain batching on the 35B int4 + DSpark load.
//!
//! Greedy only: acceptance reads each verify row's argmax, matching
//! `run_cycle`'s greedy branch. Callers with a temperature or a penalty stay
//! on the sequential path.

use burn::tensor::backend::Backend;
use std::time::Instant;

use crate::dspark::{
    AcceptedPath, BlockLayout, CycleOutcome, DraftState, Drafter, SpecStats, TreeSpec, accept_path,
    assemble_block, greedy_argmax_block, select_captured, trim_to_room,
};
use crate::model::{DecodeState, TextModel, VerifyBlock};

/// The row ceiling for one multi-sequence verify (#59). The runner falls
/// back to the per-row path when the members' blocks do not fit; a block-16
/// chain pair (34 rows) is outside the mechanism's contract.
pub const MAX_PLAN_ROWS: usize = 8;

/// Whether one plan's members fit the multi-sequence verify row budget.
pub fn batch_fits(members: usize, rows_per_member: usize) -> bool {
    members
        .checked_mul(rows_per_member)
        .is_some_and(|rows| rows <= MAX_PLAN_ROWS)
}

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
    /// This sequence's proposal layout: a chain, or a tree whose verify block
    /// is `[bonus, tree tokens…]` in breadth-first order.
    pub layout: &'a BlockLayout,
}

/// Propose per member, verify all members in one forward, then accept, trim
/// and retain per member in plan order. Returns one outcome per member.
///
/// The target states are left exactly where a per-row `run_cycle` would leave
/// them: each block's state ends at its accepted path (`state.length +=
/// path.slots.len()`), and the draft context absorbs the accepted captures.
/// A tree member is proposed and accepted with the same `TreeSpec` contracts
/// as [`crate::dspark::run_cycle`]'s tree path. Phase timings accumulate into
/// `stats.trace` when enabled; per-member cycle counters stay with the caller
/// so both paths account alike.
pub fn run_cycle_batch<B: Backend, D: Drafter<B, State = DraftState<B>>>(
    model: &TextModel<B>,
    draft: &D,
    eos_ids: &[u32],
    members: &mut [BatchCycleMember<'_, B>],
    stats: &mut SpecStats,
) -> Result<Vec<CycleOutcome>, String> {
    if members.is_empty() {
        return Err("batched cycle needs at least one member".into());
    }
    let phase = Instant::now();
    let mut blocks: Vec<Vec<u32>> = Vec::with_capacity(members.len());
    let mut trees: Vec<Option<TreeSpec>> = Vec::with_capacity(members.len());
    for member in members.iter_mut() {
        let (tokens, tree) = draft
            .propose_layout(model, member.draft_state, member.bonus, member.layout)
            .map_err(|error| error.to_string())?;
        let block = assemble_block(member.bonus, &tokens);
        debug_assert!(
            tree.as_ref().is_none_or(|tree| tree.rows() == block.len()),
            "tree rows must match the verify block"
        );
        blocks.push(block);
        trees.push(tree);
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
            .zip(trees.iter())
            .map(|((member, ids), tree)| match tree {
                Some(tree) => {
                    VerifyBlock::with_tree(ids, member.state, crate::knobs::dspark_layers(), tree)
                }
                None => VerifyBlock::new(ids, member.state, crate::knobs::dspark_layers()),
            })
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
    for (index, ((member, block), output)) in members
        .iter_mut()
        .zip(blocks.iter())
        .zip(outputs.iter())
        .enumerate()
    {
        let prediction = greedy_argmax_block(&output.logits)?;
        let mut path = accept_path(
            trees[index].as_ref(),
            &block[1..],
            eos_ids,
            &mut |slot: usize| Ok(prediction[slot]),
        )?;
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

#[cfg(test)]
mod tests {
    use super::{MAX_PLAN_ROWS, batch_fits};

    #[test]
    fn chain_pairs_outside_the_eight_row_contract_do_not_batch() {
        assert!(batch_fits(1, MAX_PLAN_ROWS), "one chain fits the ceiling");
        assert!(batch_fits(2, 4), "two block-3 chains are eight rows");
        assert!(!batch_fits(2, 17), "two block-16 chains are 34 rows");
        assert!(!batch_fits(3, 4), "a third four-row member exceeds eight");
        assert!(!batch_fits(usize::MAX, 2), "overflow is not a fit");
    }
}
