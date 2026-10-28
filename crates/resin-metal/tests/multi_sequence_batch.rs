//! Weight-free routing for the multi-sequence batch switch (#72).
//!
//! `ResinConfig::multi_sequence_batch` turns on the co-batched path for plans
//! with two or more decode rows and changes nothing for solo rows. The
//! scripted adapter records which session path ran, so these assertions are
//! about observable routing, not internals. The row budget lives in
//! `omni_minicpm_v::dspark_batch` and is covered there.

use resin_core::Phase;
use resin_core::engine::{Error, ExecutionPlan, OwnedTokenRows, Runner};
use resin_metal::config::ResinConfig;
use resin_metal::session::{DecodeAdapter, DecodeRequest, DecodeSession};

/// Records whether a decode row took the per-row `decode` path or the shared
/// `decode_batch` path, and the row count of every batch.
#[derive(Default)]
struct RoutingAdapter {
    decode_calls: usize,
    batch_row_counts: Vec<usize>,
}

impl DecodeAdapter for RoutingAdapter {
    type Slot = u32;
    type Checkpoint = ();

    fn prefill(&mut self, input: &[u32], _budget: usize) -> Result<(Self::Slot, Vec<u32>), Error> {
        Ok((u32::try_from(input.len()).unwrap(), vec![input[0]]))
    }

    fn decode(
        &mut self,
        _slot: &mut Self::Slot,
        input: &[u32],
        _room: usize,
        _budget: usize,
    ) -> Result<Vec<u32>, Error> {
        self.decode_calls += 1;
        Ok(vec![input[0] + 1])
    }

    fn decode_batch(
        &mut self,
        rows: &mut [resin_metal::session::DecodeRow<'_, Self::Slot>],
    ) -> Result<Vec<Vec<u32>>, Error> {
        self.batch_row_counts.push(rows.len());
        Ok(rows.iter().map(|row| vec![row.input[0] + 1]).collect())
    }
}

fn rows(output: OwnedTokenRows) -> Vec<Vec<u32>> {
    output.view().iter().map(|row| row.to_vec()).collect()
}

fn plan(
    phase: Phase,
    sequences: &[(u64, Vec<u32>)],
    context: u32,
    prefill_length: u32,
) -> ExecutionPlan<DecodeRequest> {
    let mut token_ids = Vec::new();
    let mut indptr = vec![0u32];
    for (_, tokens) in sequences {
        token_ids.extend_from_slice(tokens);
        indptr.push(u32::try_from(token_ids.len()).unwrap());
    }
    let rows = sequences.len();
    ExecutionPlan {
        prompt_score_indices: Vec::new(),
        prompt_score_token_ids: Vec::new(),
        prompt_score_indptr: Vec::new(),
        prompt_score_top_counts: Vec::new(),
        prompt_target_indptr: Vec::new(),
        prompt_target_token_ids: Vec::new(),
        batch_id: 1,
        batch_key: None,
        phase,
        sequence_ids: sequences.iter().map(|(sequence, _)| *sequence).collect(),
        context_lengths: vec![context; rows],
        prefill_lengths: vec![prefill_length; rows],
        token_ids,
        token_indptr: indptr,
        sampling_mask: vec![true; rows],
        cache_groups: Vec::new(),
        payloads: vec![
            DecodeRequest {
                max_new_tokens: 16,
                slot: None,
            };
            rows
        ],
    }
}

fn prefill(sequence: u64, tokens: Vec<u32>) -> ExecutionPlan<DecodeRequest> {
    let length = u32::try_from(tokens.len()).unwrap();
    plan(Phase::Prefill, &[(sequence, tokens)], 0, length)
}

fn decode(sequences: &[(u64, Vec<u32>)]) -> ExecutionPlan<DecodeRequest> {
    plan(Phase::Decode, sequences, 0, 0)
}

fn session(config: ResinConfig) -> DecodeSession<RoutingAdapter> {
    let mut session = DecodeSession::with_config(RoutingAdapter::default(), config);
    for sequence in [11u64, 12] {
        let output = session
            .forward(prefill(sequence, vec![sequence as u32, 1, 2]))
            .unwrap();
        assert_eq!(rows(output), vec![vec![sequence as u32]]);
    }
    session
}

fn two_decode_rows() -> Vec<(u64, Vec<u32>)> {
    vec![(11, vec![101]), (12, vec![102])]
}

#[test]
fn multi_sequence_batch_routes_co_batched_rows_to_one_batch() {
    let mut session = session(ResinConfig {
        multi_sequence_batch: true,
        ..ResinConfig::default()
    });
    let output = session.forward(decode(&two_decode_rows())).unwrap();
    assert_eq!(session.adapter().batch_row_counts, vec![2]);
    assert_eq!(session.adapter().decode_calls, 0);
    assert_eq!(rows(output), vec![vec![102], vec![103]]);
}

#[test]
fn solo_decode_rows_stay_on_the_per_row_path_with_batching_on() {
    let mut session = session(ResinConfig {
        multi_sequence_batch: true,
        ..ResinConfig::default()
    });
    let output = session.forward(decode(&[(11, vec![201])])).unwrap();
    assert_eq!(rows(output), vec![vec![202]]);
    assert_eq!(session.adapter().decode_calls, 1);
    assert!(
        session.adapter().batch_row_counts.is_empty(),
        "a solo row must never take the batched path"
    );

    let output = session.forward(decode(&two_decode_rows())).unwrap();
    assert_eq!(rows(output), vec![vec![102], vec![103]]);
    assert_eq!(session.adapter().batch_row_counts, vec![2]);
}

#[test]
fn every_switch_off_keeps_decoding_row_by_row() {
    let mut session = session(ResinConfig::default());
    let output = session.forward(decode(&two_decode_rows())).unwrap();
    assert_eq!(rows(output), vec![vec![102], vec![103]]);
    assert_eq!(session.adapter().decode_calls, 2);
    assert!(session.adapter().batch_row_counts.is_empty());
}
