//! DSpark block mode on the resin engine (#55, after M0-R greedy).
//!
//! One decode call is exactly one speculative cycle: the shared
//! `dspark::run_cycle` (propose → verify → greedy accept → rollback) — the
//! same acceptance path the CLI loop drives — returns the accepted block
//! plus its correction as one engine row. The engine treats k−1 tokens as
//! accepted context and feeds the correction back as the next row's input,
//! which is the cycle's next bonus, so state advances with zero extra glue.
//! Engine-generic bookkeeping lives in [`crate::session::DecodeSession`];
//! the model-side knowledge (draft, captured layers, capacity, numerics)
//! stays here, next to the model types it names.
//!
//! Co-batched plans (#72) verify every decode row in one target forward
//! (`dspark_batch::run_cycle_batch`), each member a chain block; solo rows
//! keep the per-row path. The composite tree+batch mode failed #67's gate and
//! was removed (#73); the model-side single-stream tree mode is independent
//! and unchanged.
//!
//! Conversation checkpoints (#74) carry the target decode state, the draft
//! context and the KV-aligned token history (`history[..state.length]`). The
//! draft context grows to the new request's prompt plus budget on restore,
//! because the checkpoint was sized for the request that produced it. SSD
//! persistence (#75) encodes the target state, the filled draft K/V rows and
//! the pending token as named tensors.

use burn::tensor::{Tensor, TensorData, backend::Backend};
use omni_minicpm_v::dspark::{self, BlockLayout, Draft, DraftState, SpecStats};
use omni_minicpm_v::dspark_batch::{self, BatchCycleMember};
use omni_minicpm_v::generate::{Options, Sampler};
use omni_minicpm_v::ledger::{CycleObservation, Ledger, LedgerKey};
use omni_minicpm_v::model::{DecodeState, TextModel, VerifyNumerics};
use omni_minicpm_v::text::LinearState;
use omni_r2t2::tensor::KvCache;
use resin_core::engine::{Error, Runner};
use std::time::Instant;

use crate::config::ResinConfig;
use crate::conversation::ConversationStats;
use crate::session::{DecodeAdapter, DecodeRequest, DecodeRow, DecodeSession};
use crate::state_cache::{CheckpointPayload, tensor_from};

struct DsparkAdapter<B: Backend> {
    model: TextModel<B>,
    draft: Draft<B>,
    device: B::Device,
    options: Options,
    block_len: usize,
    numerics: VerifyNumerics,
    eos_ids: Vec<u32>,
    stats: SpecStats,
    /// Calibrated verify-size ledger (#69/#65), `Some` only when
    /// `MINICPM_DSPARK_LEDGER=1`. Solo rows choose their chain length from it;
    /// the co-batched path keeps its fixed-size chain (per-member sizing is
    /// out of scope for #65).
    ledger: Option<Ledger>,
    /// Number of `decode_batch` calls that ran one shared verify forward.
    batched_cycles: u64,
}

impl<B: Backend> DsparkAdapter<B> {
    /// The same greedy gate `run_cycle` uses: the batched verify reads one
    /// argmax per row, so any sampling option falls back to the per-row path.
    fn greedy(&self) -> bool {
        self.options.temperature == 0.0
            && self.options.repetition_penalty == 1.0
            && self.options.presence_penalty == 0.0
    }

    /// The largest chain the drafter's K/V buffers must hold: the fixed block
    /// or the ledger's largest allowed size, whichever is larger.
    fn max_block(&self) -> usize {
        self.ledger
            .as_ref()
            .and_then(|ledger| ledger.allowed().iter().copied().max())
            .map_or(self.block_len, |size| size.max(self.block_len))
    }
}

/// One sequence's private decode state: target state, draft state and the
/// committed-token history the accept walk predicts against.
struct DsparkSlot<B: Backend> {
    state: DecodeState<B>,
    draft_state: DraftState<B>,
    history: Vec<u32>,
}

/// Record a decode row's bonus in the slot history exactly once.
///
/// The slot invariant is KV alignment: after prefill (or a resumed prefill)
/// `history.len() == state.length + 1`, because the prefill sample is the
/// pending bonus and is already in the history; after a cycle
/// `history.len() == state.length`, because the pending correction is in
/// neither. The engine feeds that pending token back as the next decode
/// row's input, so a bonus belongs in the history only when the state
/// already covers it. Recording by KV alignment rather than token identity
/// keeps a legitimate repeat (a correction equal to the last accepted token)
/// recorded; the prefill sample itself is never pushed twice.
fn record_bonus(history: &mut Vec<u32>, state_length: usize, bonus: u32) {
    debug_assert!(
        history.len() <= state_length + 1,
        "history ran ahead of the KV state"
    );
    if history.len() <= state_length {
        history.push(bonus);
    }
}

/// One DSpark slot's conversation checkpoint (#74).
#[derive(Clone)]
struct DsparkCheckpoint<B: Backend> {
    state: DecodeState<B>,
    draft_state: DraftState<B>,
    /// `history[..state.length]`; the pending bonus is excluded.
    tokens: Vec<u32>,
    /// The sampled token that had not entered the KV state (#75).
    pending: Option<u32>,
}

/// The filled prefix of a draft K/V buffer (#75). The saved entry only keeps
/// `length` rows; `DraftState::ensure_capacity` grows it on restore.
fn prefix_rows<B: Backend>(tensor: &Tensor<B, 4>, rows: usize) -> Tensor<B, 4> {
    let [batch, heads, capacity, width] = tensor.dims();
    assert!(rows <= capacity, "draft context length exceeds its buffer");
    tensor
        .clone()
        .slice([0..batch, 0..heads, 0..rows, 0..width])
}

impl<B: Backend> DecodeAdapter for DsparkAdapter<B> {
    type Slot = DsparkSlot<B>;
    type Checkpoint = DsparkCheckpoint<B>;

    fn prefill(&mut self, input: &[u32], budget: usize) -> Result<(Self::Slot, Vec<u32>), Error> {
        // Single-chunk prefill: prompt fits the default chunk size; the
        // capture feeds the draft exactly like the CLI prefill.
        let mut state = self.model.new_state(&self.device);
        let (logits, _, captured) = self
            .model
            .forward_capturing(
                input,
                None,
                None,
                &mut state,
                omni_minicpm_v::knobs::dspark_layers(),
                false,
            )
            .map_err(|error| Error::execution(error.to_string()))?;
        let mut history = input.to_vec();
        let first = {
            let mut sampler = Sampler::new(self.options.clone());
            let values = logits
                .into_data()
                .convert::<f32>()
                .as_slice::<f32>()
                .map_err(|error| Error::execution(error.to_string()))?
                .to_vec();
            sampler.sample(values, &history).map_err(Error::execution)?
        };
        history.push(first);
        let capacity = dspark::draft_capacity(input.len(), budget, self.max_block());
        let draft_state = self
            .draft
            .prepare(captured, capacity)
            .map_err(|error| Error::execution(error.to_string()))?;
        self.stats.cycles += 1;
        Ok((
            DsparkSlot {
                state,
                draft_state,
                history,
            },
            vec![first],
        ))
    }

    fn decode(
        &mut self,
        slot: &mut Self::Slot,
        input: &[u32],
        room: usize,
        _budget: usize,
    ) -> Result<Vec<u32>, Error> {
        // Decode rows carry exactly the last sampled token: the bonus the
        // cycle proposes from. After prefill that token is already the last
        // history entry (the prefill sample), so record it only when the KV
        // state has caught up to the history.
        let bonus = input[0];
        record_bonus(&mut slot.history, slot.state.length, bonus);
        // `MINICPM_DSPARK_LEDGER=1` replaces the fixed block with the
        // calibrated size rule; the observed cycle feeds the estimates back.
        let chosen = self.ledger.as_mut().map(Ledger::choose);
        let layout = dspark::BlockLayout::Chain {
            len: chosen.unwrap_or(self.block_len),
        };
        let cycle = Instant::now();
        let outcome = dspark::run_cycle(
            &self.model,
            &self.draft,
            &mut slot.draft_state,
            &mut slot.state,
            &layout,
            bonus,
            self.numerics,
            &self.eos_ids,
            &self.options,
            &mut Sampler::new(self.options.clone()),
            slot.history.as_slice(),
            room,
            &mut self.stats,
        )
        .map_err(Error::execution)?;
        let cycle_ms = cycle.elapsed().as_secs_f64() * 1000.0;
        if let (Some(ledger), Some(size)) = (self.ledger.as_mut(), chosen) {
            ledger.observe(CycleObservation {
                size,
                rows: outcome.rows,
                accepted: outcome.tokens.len(),
                verify_ms: outcome.verify_ms,
                cycle_ms,
                trimmed: outcome.trimmed,
            });
        }
        self.stats.cycles += 1;
        self.stats.draft_tokens_accepted += outcome.tokens.len() as u64;
        self.stats.output_tokens += outcome.tokens.len() as u64 + 1;
        slot.history.extend_from_slice(&outcome.tokens);
        let mut row = outcome.tokens;
        if let Some(correction) = outcome.correction {
            row.push(correction);
        }
        Ok(row)
    }

    /// One plan's decode rows through one multi-sequence verify (#72).
    ///
    /// Propose stays per sequence (one draft forward per member), the target
    /// forward runs once over every member's block, and accept/trim/retain
    /// run per member in plan order. The greedy gate is the same one
    /// `dspark::run_cycle` applies; anything else falls back to the default
    /// per-row loop so numerics never change silently. Every member keeps the
    /// chain layout #72 ships; solo rows never reach this method.
    fn decode_batch(
        &mut self,
        rows: &mut [DecodeRow<'_, Self::Slot>],
    ) -> Result<Vec<Vec<u32>>, Error> {
        if !self.greedy() {
            let mut outputs = Vec::with_capacity(rows.len());
            for row in rows {
                outputs.push(self.decode(&mut *row.slot, row.input, row.room, row.budget)?);
            }
            return Ok(outputs);
        }
        // #59 caps one multi-sequence verify at 8 rows. A plan outside that
        // contract (two block-16 chains are 34 rows) takes the per-row path:
        // the >32-row MoE fallback is the wrong shape for this batch and the
        // multi-row mechanism was never specified for it.
        if !dspark_batch::batch_fits(rows.len(), self.block_len + 1) {
            let mut outputs = Vec::with_capacity(rows.len());
            for row in rows {
                outputs.push(self.decode(&mut *row.slot, row.input, row.room, row.budget)?);
            }
            return Ok(outputs);
        }
        let layouts = vec![
            BlockLayout::Chain {
                len: self.block_len
            };
            rows.len()
        ];
        let mut members = Vec::with_capacity(rows.len());
        for (row, layout) in rows.iter_mut().zip(layouts.iter()) {
            let slot = &mut *row.slot;
            // Every decode row carries exactly the bonus; the history the
            // accept walk predicts against includes it, like the per-row path,
            // and the same KV-alignment guard keeps the prefill sample from
            // being recorded twice after the first decode.
            let bonus = row.input[0];
            record_bonus(&mut slot.history, slot.state.length, bonus);
            members.push(BatchCycleMember {
                draft_state: &mut slot.draft_state,
                state: &mut slot.state,
                bonus,
                room: row.room,
                layout,
            });
        }
        let outcomes = dspark_batch::run_cycle_batch(
            &self.model,
            &self.draft,
            &self.eos_ids,
            &mut members,
            &mut self.stats,
        )
        .map_err(Error::execution)?;
        drop(members);
        self.batched_cycles += 1;
        let mut outputs = Vec::with_capacity(rows.len());
        for (row, outcome) in rows.iter_mut().zip(outcomes) {
            let slot = &mut *row.slot;
            self.stats.cycles += 1;
            self.stats.draft_tokens_accepted += outcome.tokens.len() as u64;
            self.stats.output_tokens += outcome.tokens.len() as u64 + 1;
            slot.history.extend_from_slice(&outcome.tokens);
            let mut tokens = outcome.tokens;
            if let Some(correction) = outcome.correction {
                tokens.push(correction);
            }
            outputs.push(tokens);
        }
        Ok(outputs)
    }

    fn checkpoint(&self, slot: &Self::Slot) -> Option<Self::Checkpoint> {
        let aligned = slot.state.length.min(slot.history.len());
        if aligned == 0 {
            return None;
        }
        Some(DsparkCheckpoint {
            state: slot.state.clone(),
            draft_state: slot.draft_state.clone(),
            tokens: slot.history[..aligned].to_vec(),
            pending: slot.history.get(aligned).copied(),
        })
    }

    fn restore(&mut self, checkpoint: Self::Checkpoint) -> Result<Self::Slot, Error> {
        Ok(DsparkSlot {
            state: checkpoint.state,
            draft_state: checkpoint.draft_state,
            history: checkpoint.tokens,
        })
    }

    fn checkpoint_tokens<'a>(&self, checkpoint: &'a Self::Checkpoint) -> &'a [u32] {
        &checkpoint.tokens
    }

    /// Restore the target and draft state, extend both by the prompt suffix
    /// and sample at prefill end. The draft buffers grow first when the new
    /// request's prompt plus budget no longer fits the checkpoint's capacity.
    /// An empty suffix is the exact-length disk restore (#75): the target and
    /// draft already cover the prompt, and the stored pending token is the
    /// prefill-end sample.
    fn prefill_resume(
        &mut self,
        checkpoint: Self::Checkpoint,
        input: &[u32],
        budget: usize,
    ) -> Result<Option<(Self::Slot, Vec<u32>)>, Error> {
        let DsparkCheckpoint {
            mut state,
            mut draft_state,
            tokens,
            pending,
        } = checkpoint;
        if input.is_empty() {
            let Some(pending) = pending else {
                return Ok(None);
            };
            draft_state.ensure_capacity(tokens.len() + budget + self.max_block() + 1);
            let mut history = tokens;
            history.push(pending);
            return Ok(Some((
                DsparkSlot {
                    state,
                    draft_state,
                    history,
                },
                vec![pending],
            )));
        }
        draft_state.ensure_capacity(tokens.len() + input.len() + budget + self.max_block() + 1);
        let (logits, _, captured) = self
            .model
            .forward_capturing(
                input,
                None,
                None,
                &mut state,
                omni_minicpm_v::knobs::dspark_layers(),
                false,
            )
            .map_err(|error| Error::execution(error.to_string()))?;
        let mut history = tokens;
        history.extend_from_slice(input);
        let first = {
            let mut sampler = Sampler::new(self.options.clone());
            let values = logits
                .into_data()
                .convert::<f32>()
                .as_slice::<f32>()
                .map_err(|error| Error::execution(error.to_string()))?
                .to_vec();
            sampler.sample(values, &history).map_err(Error::execution)?
        };
        history.push(first);
        self.draft
            .extend(&mut draft_state, captured)
            .map_err(|error| Error::execution(error.to_string()))?;
        self.stats.cycles += 1;
        Ok(Some((
            DsparkSlot {
                state,
                draft_state,
                history,
            },
            vec![first],
        )))
    }

    /// Resident target state plus the draft K/V buffers of one slot (#76).
    fn slot_bytes(&self, slot: &Self::Slot) -> usize {
        usize::try_from(
            crate::memory::state_bytes(&slot.state)
                .saturating_add(crate::memory::draft_state_bytes(&slot.draft_state)),
        )
        .unwrap_or(usize::MAX)
    }

    fn checkpoint_bytes(&self, checkpoint: &Self::Checkpoint) -> usize {
        usize::try_from(
            crate::memory::state_bytes(&checkpoint.state)
                .saturating_add(crate::memory::draft_state_bytes(&checkpoint.draft_state)),
        )
        .unwrap_or(usize::MAX)
    }

    /// DSpark checkpoint as named tensors (#75): target KV/linear state, the
    /// filled draft K/V rows and the pending token plus draft length.
    fn encode_checkpoint(&self, checkpoint: &Self::Checkpoint) -> Option<CheckpointPayload> {
        let state = &checkpoint.state;
        let draft_state = &checkpoint.draft_state;
        let draft_length = draft_state.len();
        let draft_layers = draft_state.keys().len();
        if draft_layers != draft_state.values().len() {
            return None;
        }
        let mut payload = CheckpointPayload::new(
            state.length,
            state.rope_delta,
            checkpoint.pending,
            Some(serde_json::json!({
                "kv_layers": state.kv_caches().len(),
                "linear_layers": state.linear_states().len(),
                "draft_layers": draft_layers,
                "draft_length": draft_length,
            })),
        );
        for (index, cache) in state.kv_caches().iter().enumerate() {
            if let Some((key, value)) = cache.snapshot() {
                payload.push(format!("kv.{index}.k"), key);
                payload.push(format!("kv.{index}.v"), value);
            }
        }
        for (index, linear) in state.linear_states().iter().enumerate() {
            payload.push(format!("linear.{index}.conv"), &linear.conv);
            payload.push(format!("linear.{index}.recurrent"), &linear.recurrent);
        }
        for (index, (key, value)) in draft_state
            .keys()
            .iter()
            .zip(draft_state.values())
            .enumerate()
        {
            payload.push(format!("draft.{index}.k"), &prefix_rows(key, draft_length));
            payload.push(
                format!("draft.{index}.v"),
                &prefix_rows(value, draft_length),
            );
        }
        Some(payload)
    }

    fn decode_checkpoint(
        &self,
        tokens: &[u32],
        payload: CheckpointPayload,
    ) -> Option<Self::Checkpoint> {
        if tokens.len() != payload.length {
            return None;
        }
        let (length, rope_delta, pending, meta, mut tensors) = payload.into_parts();
        let meta = meta?;
        let kv_layers = usize::try_from(meta.get("kv_layers")?.as_u64()?).ok()?;
        let linear_layers = usize::try_from(meta.get("linear_layers")?.as_u64()?).ok()?;
        let draft_layers = usize::try_from(meta.get("draft_layers")?.as_u64()?).ok()?;
        let draft_length = usize::try_from(meta.get("draft_length")?.as_u64()?).ok()?;
        // Hybrid towers have a different number of full-attention and linear
        // layers (1B and 35B both do); only the total has to be non-empty and
        // each count bounded.
        if kv_layers > 1024
            || linear_layers > 1024
            || kv_layers + linear_layers == 0
            || draft_layers == 0
            || draft_layers > 1024
            || draft_length > length
        {
            return None;
        }
        let mut kv = Vec::with_capacity(kv_layers);
        for index in 0..kv_layers {
            let key = tensors.remove(&format!("kv.{index}.k"));
            let value = tensors.remove(&format!("kv.{index}.v"));
            match (key, value) {
                (Some(key), Some(value)) => kv.push(KvCache::from_tensors(
                    tensor_from(key, &self.device)?,
                    tensor_from(value, &self.device)?,
                )),
                (None, None) => kv.push(KvCache::default()),
                _ => return None,
            }
        }
        let mut linear = Vec::with_capacity(linear_layers);
        for index in 0..linear_layers {
            let conv: TensorData = tensors.remove(&format!("linear.{index}.conv"))?;
            let recurrent: TensorData = tensors.remove(&format!("linear.{index}.recurrent"))?;
            linear.push(LinearState {
                conv: tensor_from(conv, &self.device)?,
                recurrent: tensor_from(recurrent, &self.device)?,
            });
        }
        let mut keys = Vec::with_capacity(draft_layers);
        let mut values = Vec::with_capacity(draft_layers);
        for index in 0..draft_layers {
            let key = tensor_from(tensors.remove(&format!("draft.{index}.k"))?, &self.device)?;
            let value = tensor_from(tensors.remove(&format!("draft.{index}.v"))?, &self.device)?;
            let [batch, heads, capacity, width] = key.dims();
            if key.dims() != value.dims()
                || batch != 1
                || heads == 0
                || capacity < draft_length
                || width == 0
            {
                return None;
            }
            keys.push(key);
            values.push(value);
        }
        if !tensors.is_empty() {
            return None;
        }
        Some(DsparkCheckpoint {
            state: DecodeState::from_parts(kv, linear, length, rope_delta),
            draft_state: DraftState::from_buffers(keys, values, draft_length),
            tokens: tokens.to_vec(),
            pending,
        })
    }
}

pub struct MinicpmDsparkRunner<B: Backend> {
    session: DecodeSession<DsparkAdapter<B>>,
}

impl<B: Backend> MinicpmDsparkRunner<B> {
    /// Shared model and draft references for in-process A/B harnesses (the
    /// #61 baseline example) driving the direct dspark loop on the same
    /// weights the engine runner holds.
    pub fn model(&self) -> &TextModel<B> {
        &self.session.adapter().model
    }

    pub fn draft(&self) -> &Draft<B> {
        &self.session.adapter().draft
    }

    /// Default engine config; `RESIN_MULTI_SEQUENCE_BATCH=1` still applies as
    /// the experimental same-binary override.
    pub fn new(
        model: TextModel<B>,
        draft: Draft<B>,
        device: B::Device,
        block_len: usize,
        eos_ids: Vec<u32>,
    ) -> Self {
        Self::with_config(
            model,
            draft,
            device,
            block_len,
            eos_ids,
            ResinConfig::default(),
        )
    }

    /// Explicit engine config. `multi_sequence_batch` hands every decode row
    /// of a plan to [`DsparkAdapter::decode_batch`].
    pub fn with_config(
        model: TextModel<B>,
        draft: Draft<B>,
        device: B::Device,
        block_len: usize,
        eos_ids: Vec<u32>,
        config: ResinConfig,
    ) -> Self {
        Self::with_state_cache(model, draft, device, block_len, eos_ids, config, None)
    }

    /// Explicit config plus the SSD state cache (#75). The caller derives the
    /// cache namespace from the model and drafter directories plus the block
    /// row count; the session never guesses a path. This is also the combined
    /// builder: `config` and `cache` are both preserved.
    #[allow(clippy::too_many_arguments)]
    pub fn with_state_cache(
        model: TextModel<B>,
        draft: Draft<B>,
        device: B::Device,
        block_len: usize,
        eos_ids: Vec<u32>,
        config: ResinConfig,
        cache: Option<crate::state_cache::StateCache>,
    ) -> Self {
        // Resolve the environment once; the session folds the overrides
        // again, which is idempotent.
        let config = config.with_env_overrides();
        // `MINICPM_DSPARK_LEDGER=1` turns the calibrated ledger on for solo
        // rows. The default identity is the engine's, not the weight file's
        // (the runner does not see the precision); callers with a real model
        // pair override it through `with_ledger_key`.
        let ledger = omni_minicpm_v::knobs::dspark_ledger().then(|| {
            Ledger::load(
                omni_minicpm_v::ledger::ALLOWED_SIZES.to_vec(),
                LedgerKey::new("resin", "dspark", "engine", "adaptive"),
            )
        });
        Self {
            session: DecodeSession::with_config_and_state_cache(
                DsparkAdapter {
                    model,
                    draft,
                    device,
                    options: Options {
                        temperature: 0.0,
                        ..Default::default()
                    },
                    block_len,
                    numerics: VerifyNumerics::Batched,
                    eos_ids,
                    stats: {
                        let mut stats = SpecStats::default();
                        stats.trace.enabled = omni_minicpm_v::knobs::dspark_trace();
                        stats
                    },
                    ledger,
                    batched_cycles: 0,
                },
                config,
                cache,
            ),
        }
    }

    /// Override the ledger checkpoint identity (`MINICPM_DSPARK_LEDGER=1`).
    /// The default identity is model `resin`, drafter `dspark`, precision
    /// `engine` and block `adaptive`; pass the real model/drafter/precision
    /// labels so estimates are keyed like the CLI's. Ignored while the
    /// switch is off.
    pub fn with_ledger_key(mut self, key: LedgerKey) -> Self {
        if let Some(ledger) = self.session.adapter_mut().ledger.as_mut() {
            let allowed = ledger.allowed().to_vec();
            *ledger = Ledger::load(allowed, key);
        }
        self
    }

    /// Replace the experimental switches at runtime (same-process A/B arms).
    pub fn set_config(&mut self, config: ResinConfig) {
        self.session.set_config(config);
    }

    /// The resolved engine switches (environment overrides folded in).
    pub fn config(&self) -> &ResinConfig {
        self.session.config()
    }

    /// Batched verify forwards run so far (0 when the switches are off).
    pub fn batch_cycles(&self) -> u64 {
        self.session.adapter().batched_cycles
    }

    /// The loaded target model and drafter. The co-batch A/B example drives
    /// the model-side single-stream tree mode on the same loaded weights;
    /// production callers go through [`Runner::forward`].
    pub fn model_and_draft(&self) -> (&TextModel<B>, &Draft<B>) {
        let adapter = self.session.adapter();
        (&adapter.model, &adapter.draft)
    }

    /// Prefill tokens fed and checkpoint hit/miss counters (#74, #75).
    pub fn conversation_stats(&self) -> ConversationStats {
        self.session.conversation_stats()
    }

    /// Disk cache counters, when a cache is attached (#75).
    pub fn state_cache_stats(&self) -> Option<crate::state_cache::StateCacheStats> {
        self.session.state_cache_stats()
    }

    /// Resident KV bytes, budget and release-order counters (#76).
    pub fn memory_stats(&self) -> crate::memory::MemoryStats {
        self.session.memory_stats()
    }

    /// Encode and write queued checkpoints now (#75).
    pub fn flush_state_cache(&mut self) -> usize {
        self.session.flush_state_cache()
    }

    /// Aggregate cycle stats (τ via `acceptance_length`).
    pub fn stats(&self) -> &SpecStats {
        &self.session.adapter().stats
    }
}

impl<B: Backend> Runner<DecodeRequest> for MinicpmDsparkRunner<B> {
    fn forward(
        &mut self,
        plan: resin_core::engine::ExecutionPlan<DecodeRequest>,
    ) -> Result<resin_core::engine::OwnedTokenRows, Error> {
        self.session.forward(plan)
    }

    fn release(&mut self, request_ids: &[u64]) -> Result<(), Error> {
        self.session.release(request_ids)?;
        // The release is the idle boundary: persist the ledger estimates the
        // solo cycles produced (the CLI saves at the end of `generate`).
        if let Some(ledger) = self.session.adapter().ledger.as_ref()
            && ledger.cycles() > 0
        {
            ledger.save();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::record_bonus;

    /// The prefill sample is the first decode's bonus and is already in the
    /// history; a second push would shift `history[..state.length]` off the
    /// KV-aligned checkpoint prefix.
    #[test]
    fn prefill_sample_is_not_recorded_twice() {
        // Prompt [1, 2, 3], prefill state covers it, sample 10 pending.
        let mut history = vec![1, 2, 3, 10];
        record_bonus(&mut history, 3, 10);
        assert_eq!(history, vec![1, 2, 3, 10], "prefill sample pushed twice");
        // The checkpoint token prefix is the prompt, and the pending sample
        // is what the checkpoint stores as `pending`.
        assert_eq!(&history[..3], &[1, 2, 3]);
        assert_eq!(history.get(3), Some(&10));
    }

    /// After a cycle the history has caught up to the KV state and the
    /// correction is pending; it must be recorded, even when it repeats the
    /// last accepted token.
    #[test]
    fn cycle_correction_is_recorded_even_when_it_repeats() {
        // State covers prompt + bonus + accepted [10, 11]; history matches.
        let mut history = vec![1, 2, 3, 10, 11];
        record_bonus(&mut history, 5, 11);
        assert_eq!(history, vec![1, 2, 3, 10, 11, 11]);
        assert_eq!(history.len(), 6);
    }

    /// End-to-end alignment over the engine's row sequence: prefill, one
    /// cycle that accepts two drafts and emits a correction, then that
    /// correction's cycle.
    #[test]
    fn checkpoint_history_stays_kv_aligned_across_cycles() {
        let mut state_length = 3usize; // prompt
        let mut history = vec![1, 2, 3, 10]; // prefill sample 10
        assert_eq!(history.len(), state_length + 1);

        // Cycle 1: the engine feeds the prefill sample back as the bonus.
        record_bonus(&mut history, state_length, 10);
        assert_eq!(history, vec![1, 2, 3, 10], "first row is not a duplicate");
        // Two drafts accepted, correction 13 pending.
        history.extend_from_slice(&[11, 12]);
        state_length += 3; // bonus + two accepted
        assert_eq!(history.len(), state_length, "correction stays pending");
        assert_eq!(&history[..state_length], &[1, 2, 3, 10, 11, 12]);

        // Cycle 2: the correction is the next row's bonus.
        record_bonus(&mut history, state_length, 13);
        assert_eq!(history.len(), state_length + 1);
        history.extend_from_slice(&[14]); // one draft accepted, correction 15
        state_length += 2; // bonus + one accepted
        assert_eq!(history.len(), state_length);
        assert_eq!(&history[..state_length], &[1, 2, 3, 10, 11, 12, 13, 14]);
    }
}
