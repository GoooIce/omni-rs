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
//! Every member carries its own [`BlockLayout`] (#73): a chain, or a
//! best-first tree whose verify rows sit at their tree depth under the
//! ancestor mask. Tree members take the same [`VerifyBlock::with_tree`]
//! contract as the single-sequence tree path — per-sequence positions,
//! per-sequence path states, `retain_path` with the accepted tree slots — so
//! one target forward can verify mixed chain and tree blocks. A best-first
//! verify block reserves its bonus plus `live` rows; [`co_batch_layouts`]
//! applies the first stacking budget (2 sequences x 4 rows, total ≤ 8).
//!
//! Greedy only: acceptance reads each verify row's argmax, matching
//! `run_cycle`'s greedy branch. Callers with a temperature or a penalty stay
//! on the sequential path.

use burn::tensor::backend::Backend;
use std::time::Instant;

use crate::dspark::{
    AcceptedPath, BlockLayout, CycleOutcome, Draft, DraftState, SpecStats, TreeSpec, accept_path,
    assemble_block, greedy_argmax_block, select_captured, trim_to_room,
};
use crate::model::{DecodeState, TextModel, VerifyBlock};

/// The live-node cap per sequence in a stacked plan: the bonus plus these
/// rows is #73's 4-row budget (`live <= 3`).
pub const STACKED_TREE_LIVE: usize = 3;

/// The per-sequence row budget in a stacked plan, bonus included.
pub const STACKED_TREE_ROWS: usize = STACKED_TREE_LIVE + 1;

/// The total row budget for one stacked plan: two sequences of
/// [`STACKED_TREE_ROWS`] rows.
pub const STACKED_MAX_ROWS: usize = 2 * STACKED_TREE_ROWS;

/// Draft rows one propose occupies past the draft context: the chain's block
/// length, or at least the best-first candidate grid's depth (its draft
/// forward runs one masked row per level). Callers reserve draft K/V capacity
/// with this so a stacked plan fits even when `block_len` is smaller than the
/// grid.
pub fn propose_rows(tree_batch: bool, block_len: usize) -> usize {
    if tree_batch {
        block_len.max(crate::dspark::TREE_GRID_LEVELS)
    } else {
        block_len
    }
}

/// The per-sequence layout of a stacked plan: a best-first tree over the
/// candidate grid, capped at [`STACKED_TREE_LIVE`] live nodes (4 rows with
/// the bonus).
///
/// `ledger_live` is the calibrated size rule's choice when the ledger is
/// enabled; it is clamped here so no sizing rule can exceed the first
/// version's budget. The resin runner has no ledger yet, so it passes
/// `None` and every stacked sequence uses the fixed 3-node confidence
/// prefix (ticket #73).
pub fn stacked_tree_layout(ledger_live: Option<usize>) -> BlockLayout {
    BlockLayout::BestTree {
        levels: crate::dspark::TREE_GRID_LEVELS,
        live: ledger_live
            .unwrap_or(STACKED_TREE_LIVE)
            .min(STACKED_TREE_LIVE),
    }
}

/// Check a co-batched plan against the stacking row budget: no member over
/// [`STACKED_TREE_ROWS`] rows and no plan over [`STACKED_MAX_ROWS`]. Returns
/// the plan's total row count.
///
/// This is the fixed even split of #67 — there is no survival × marginal-cost
/// allocator yet, so an over-budget plan is rejected instead of reshaped.
pub fn validate_stacked_plan(layouts: &[BlockLayout]) -> Result<usize, String> {
    if layouts.is_empty() {
        return Err("stacked plan needs at least one member".into());
    }
    let mut total = 0usize;
    for (index, layout) in layouts.iter().enumerate() {
        let rows = layout.rows();
        if rows > STACKED_TREE_ROWS {
            return Err(format!(
                "stacked member {index} needs {rows} rows; the per-sequence budget is \
                 {STACKED_TREE_ROWS} including the bonus ({layout:?})"
            ));
        }
        total += rows;
    }
    if total > STACKED_MAX_ROWS {
        return Err(format!(
            "stacked plan needs {total} rows; the budget is {STACKED_MAX_ROWS} \
             (2 sequences x {STACKED_TREE_ROWS} rows)"
        ));
    }
    Ok(total)
}

/// The layouts for one co-batched cycle.
///
/// With `tree_batch` on, every member of a genuine co-batch (`>= 2` rows)
/// takes a capped best-first tree and the plan must fit
/// [`validate_stacked_plan`]. A single member keeps the chain layout: solo
/// rows never change behavior, and the stacking switch only shapes plans the
/// per-row path never sees. With `tree_batch` off every member keeps the
/// chain layout #72 ships (uncapped, exactly like the per-row cycle).
pub fn co_batch_layouts(
    members: usize,
    tree_batch: bool,
    block_len: usize,
) -> Result<Vec<BlockLayout>, String> {
    if members == 0 {
        return Err("batched cycle needs at least one member".into());
    }
    if tree_batch && members >= 2 {
        let layouts = vec![stacked_tree_layout(None); members];
        validate_stacked_plan(&layouts)?;
        Ok(layouts)
    } else {
        Ok(vec![BlockLayout::Chain { len: block_len }; members])
    }
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
pub fn run_cycle_batch<B: Backend>(
    model: &TextModel<B>,
    draft: &Draft<B>,
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
    use super::{
        STACKED_MAX_ROWS, STACKED_TREE_LIVE, co_batch_layouts, propose_rows, stacked_tree_layout,
        validate_stacked_plan,
    };
    use crate::dspark::{BlockLayout, TREE_GRID_LEVELS};

    fn tree() -> BlockLayout {
        BlockLayout::BestTree {
            levels: TREE_GRID_LEVELS,
            live: STACKED_TREE_LIVE,
        }
    }

    #[test]
    fn layout_rows_include_the_bonus() {
        assert_eq!(BlockLayout::Chain { len: 3 }.rows(), 4);
        assert_eq!(BlockLayout::Tree { levels: 1 }.rows(), 3);
        assert_eq!(BlockLayout::BestTree { levels: 3, live: 3 }.rows(), 4);
        assert_eq!(BlockLayout::BestTree { levels: 3, live: 6 }.rows(), 7);
    }

    #[test]
    fn two_sequences_get_two_capped_trees_totalling_eight_rows() {
        let layouts = co_batch_layouts(2, true, 16).unwrap();
        assert_eq!(layouts, vec![tree(), tree()]);
        assert_eq!(validate_stacked_plan(&layouts), Ok(STACKED_MAX_ROWS));
    }

    #[test]
    fn stacking_rejects_plans_over_eight_rows() {
        // A third sequence would add a fourth row past the total budget.
        let error = co_batch_layouts(3, true, 16).unwrap_err();
        assert!(error.contains("8"), "{error}");
        // Two chains of five rows each are already over the total.
        let plan = [BlockLayout::Chain { len: 4 }, BlockLayout::Chain { len: 4 }];
        assert!(validate_stacked_plan(&plan).is_err());
    }

    #[test]
    fn stacking_rejects_a_member_over_the_four_row_budget() {
        let plan = [tree(), BlockLayout::BestTree { levels: 3, live: 4 }];
        let error = validate_stacked_plan(&plan).unwrap_err();
        assert!(error.contains("4"), "{error}");
        assert!(validate_stacked_plan(&[]).is_err());
    }

    #[test]
    fn mixed_chain_and_tree_members_fit_the_budget() {
        let plan = [BlockLayout::Chain { len: 2 }, tree()];
        assert_eq!(validate_stacked_plan(&plan), Ok(7));
    }

    #[test]
    fn stacking_off_keeps_the_chain_layouts_unchanged() {
        let layouts = co_batch_layouts(2, false, 16).unwrap();
        assert_eq!(
            layouts,
            vec![
                BlockLayout::Chain { len: 16 },
                BlockLayout::Chain { len: 16 }
            ]
        );
    }

    #[test]
    fn a_solo_member_keeps_the_chain_even_with_stacking_on() {
        assert_eq!(
            co_batch_layouts(1, true, 8).unwrap(),
            vec![BlockLayout::Chain { len: 8 }]
        );
    }

    #[test]
    fn stacked_propose_reserves_at_least_the_candidate_grid() {
        assert_eq!(propose_rows(true, 1), TREE_GRID_LEVELS);
        assert_eq!(propose_rows(true, 16), 16);
        assert_eq!(propose_rows(false, 1), 1);
    }

    #[test]
    fn a_ledger_tree_choice_is_capped_at_the_four_row_budget() {
        let capped = stacked_tree_layout(Some(6));
        assert_eq!(capped, tree());
        assert_eq!(capped.rows(), 4);
        assert_eq!(
            stacked_tree_layout(None),
            BlockLayout::BestTree {
                levels: TREE_GRID_LEVELS,
                live: STACKED_TREE_LIVE,
            }
        );
        assert_eq!(
            stacked_tree_layout(Some(2)),
            BlockLayout::BestTree {
                levels: TREE_GRID_LEVELS,
                live: 2,
            }
        );
    }
}
