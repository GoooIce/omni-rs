//! MiniCPM-V-4.7 greedy text decode as a resin adapter (#55 M0-R).
//!
//! The engine-generic bookkeeping (slot lifecycle, emission budget, row
//! assembly) lives in [`crate::session::DecodeSession`]; this adapter holds
//! the model and the history discipline: prefill chunks and decode steps
//! feed the same `TextModel::forward` the CLI generate loop uses, and
//! sampling goes through the same `generate::Sampler` with greedy options,
//! so engine-driven streams stay token-for-token identical to the direct
//! path by construction.
//!
//! Co-batched decode (#63/#72): with `multi_sequence_batch` on, every decode
//! row of a plan shares one [`TextModel::forward_verifying_multi`] forward;
//! each row is one pending token and is sampled through its own sampler and
//! history, so the batched stream is the per-row stream by construction.
//!
//! Conversation checkpoints (#74) snapshot the decode state plus the
//! KV-aligned history (`history[..state.length]`): the pending sampled token
//! is not in the KV state, so it is left out and replayed by the next prompt
//! (see the alignment note in [`crate::session`]). SSD persistence (#75)
//! encodes the same checkpoint as named tensors, keeping every dtype as-is.

use burn::tensor::{TensorData, backend::Backend};
use omni_minicpm_v::dspark_batch;
use omni_minicpm_v::generate::{Options, Sampler};
use omni_minicpm_v::model::{DecodeState, TextModel, VerifyBlock};
use omni_minicpm_v::text::LinearState;
use omni_r2t2::tensor::KvCache;
use resin_core::engine::{Error, Runner};

use crate::config::ResinConfig;
use crate::conversation::ConversationStats;
use crate::session::{DecodeAdapter, DecodeRequest, DecodeRow, DecodeSession};
use crate::state_cache::{CheckpointPayload, tensor_from};

struct GreedyAdapter<B: Backend> {
    model: TextModel<B>,
    device: B::Device,
    options: Options,
    /// Number of `decode_batch` calls that shared one multi-sequence verify.
    batched_cycles: u64,
}

/// One greedy slot's conversation checkpoint.
#[derive(Clone)]
struct GreedyCheckpoint<B: Backend> {
    state: DecodeState<B>,
    /// `history[..state.length]`; the pending bonus is excluded.
    tokens: Vec<u32>,
    /// The sampled token that had not entered the KV state (#75): a disk
    /// entry matching the whole prompt emits it without a forward.
    pending: Option<u32>,
}

/// Record a decode row's input unless it is already the tail of the history.
///
/// The engine feeds the previous row's sampled token back, and that token is
/// already the last history entry; appending it again would desynchronise the
/// history from the KV state and move checkpoint tokens off the prompt
/// prefix. A row carrying anything else (a caller-built suffix) is recorded.
fn record_decode_input(history: &mut Vec<u32>, input: &[u32]) {
    if input.len() != 1 || history.last() != input.first() {
        history.extend_from_slice(input);
    }
}

/// Greedy sampler over one logits row, shared with the direct loop.
fn sample_greedy<B: Backend>(
    options: &Options,
    logits: burn::tensor::Tensor<B, 2>,
    history: &[u32],
) -> Result<u32, Error> {
    let mut sampler = Sampler::new(options.clone());
    let values = logits
        .into_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .map_err(|error| Error::execution(error.to_string()))?
        .to_vec();
    sampler.sample(values, history).map_err(Error::execution)
}

impl<B: Backend> DecodeAdapter for GreedyAdapter<B> {
    type Slot = (DecodeState<B>, Vec<u32>);
    type Checkpoint = GreedyCheckpoint<B>;

    fn prefill(&mut self, input: &[u32], _budget: usize) -> Result<(Self::Slot, Vec<u32>), Error> {
        let mut state = self.model.new_state(&self.device);
        let logits = self
            .model
            .forward(input, None, None, &mut state)
            .map_err(|error| Error::execution(error.to_string()))?;
        let mut history = input.to_vec();
        let token = sample_greedy(&self.options, logits, &history)?;
        history.push(token);
        Ok(((state, history), vec![token]))
    }

    fn decode(
        &mut self,
        slot: &mut Self::Slot,
        input: &[u32],
        _room: usize,
        _budget: usize,
    ) -> Result<Vec<u32>, Error> {
        let (state, history) = slot;
        record_decode_input(history, input);
        let logits = self
            .model
            .forward(input, None, None, state)
            .map_err(|error| Error::execution(error.to_string()))?;
        let token = sample_greedy(&self.options, logits, history)?;
        history.push(token);
        Ok(vec![token])
    }

    /// One plan's decode rows through one multi-sequence verify (#63/#72).
    ///
    /// Each row is a single pending token; the shared forward returns that
    /// row's logits, and every sequence samples through its own [`Sampler`]
    /// and history exactly like [`Self::decode`]. The state advances by the
    /// one row the block appends, so history and KV stay aligned. Rows that
    /// are not single tokens, a solo row, or a plan above the #59 eight-row
    /// budget keep the per-row path (the batching contract is one token per
    /// sequence).
    fn decode_batch(
        &mut self,
        rows: &mut [DecodeRow<'_, Self::Slot>],
    ) -> Result<Vec<Vec<u32>>, Error> {
        let fits = rows.len() >= 2
            && rows.iter().all(|row| row.input.len() == 1)
            && dspark_batch::batch_fits(rows.len(), 1);
        if !fits {
            let mut outputs = Vec::with_capacity(rows.len());
            for row in rows {
                outputs.push(self.decode(&mut *row.slot, row.input, row.room, row.budget)?);
            }
            return Ok(outputs);
        }
        let capture: [usize; 0] = [];
        let mut histories = Vec::with_capacity(rows.len());
        let mut blocks = Vec::with_capacity(rows.len());
        for row in rows.iter_mut() {
            let (state, history) = &mut *row.slot;
            record_decode_input(history, row.input);
            histories.push(history);
            blocks.push(VerifyBlock::new(row.input, state, &capture));
        }
        let outputs = self
            .model
            .forward_verifying_multi(&mut blocks)
            .map_err(|error| Error::execution(error.to_string()))?;
        self.batched_cycles += 1;
        let mut result = Vec::with_capacity(outputs.len());
        for (output, history) in outputs.into_iter().zip(histories) {
            debug_assert_eq!(output.logits.dims()[0], 1, "one decode row per block");
            let token = sample_greedy(&self.options, output.logits, history)?;
            history.push(token);
            result.push(vec![token]);
        }
        Ok(result)
    }

    fn checkpoint(&self, slot: &Self::Slot) -> Option<Self::Checkpoint> {
        let (state, history) = slot;
        let aligned = state.length.min(history.len());
        if aligned == 0 {
            return None;
        }
        Some(GreedyCheckpoint {
            state: state.clone(),
            tokens: history[..aligned].to_vec(),
            pending: history.get(aligned).copied(),
        })
    }

    fn restore(&mut self, checkpoint: Self::Checkpoint) -> Result<Self::Slot, Error> {
        Ok((checkpoint.state, checkpoint.tokens))
    }

    fn checkpoint_tokens<'a>(&self, checkpoint: &'a Self::Checkpoint) -> &'a [u32] {
        &checkpoint.tokens
    }

    /// Replay the prompt suffix over the restored state. The final prompt
    /// token is always part of `input` (the session's prefix rule), so the
    /// forward produces the same prefill-end logits a full prefill would. An
    /// empty suffix is the exact-length disk restore (#75): the state already
    /// covers the prompt, so the stored pending token is the prefill-end
    /// sample and is emitted without a forward.
    fn prefill_resume(
        &mut self,
        checkpoint: Self::Checkpoint,
        input: &[u32],
        _budget: usize,
    ) -> Result<Option<(Self::Slot, Vec<u32>)>, Error> {
        let GreedyCheckpoint {
            mut state,
            tokens,
            pending,
        } = checkpoint;
        if input.is_empty() {
            let Some(pending) = pending else {
                return Ok(None);
            };
            let mut history = tokens;
            history.push(pending);
            return Ok(Some(((state, history), vec![pending])));
        }
        let logits = self
            .model
            .forward(input, None, None, &mut state)
            .map_err(|error| Error::execution(error.to_string()))?;
        let mut history = tokens;
        history.extend_from_slice(input);
        let token = sample_greedy(&self.options, logits, &history)?;
        history.push(token);
        Ok(Some(((state, history), vec![token])))
    }

    /// Resident target-state bytes of one slot (#76): KV pairs plus linear
    /// conv/recurrence tensors.
    fn slot_bytes(&self, slot: &Self::Slot) -> usize {
        usize::try_from(crate::memory::state_bytes(&slot.0)).unwrap_or(usize::MAX)
    }

    fn checkpoint_bytes(&self, checkpoint: &Self::Checkpoint) -> usize {
        usize::try_from(crate::memory::state_bytes(&checkpoint.state)).unwrap_or(usize::MAX)
    }

    /// Greedy checkpoint as named tensors (#75): one `kv.<layer>.k/v` pair per
    /// full-attention layer and `linear.<layer>.conv/recurrent` per linear
    /// layer, in the state's own dtype.
    fn encode_checkpoint(&self, checkpoint: &Self::Checkpoint) -> Option<CheckpointPayload> {
        let state = &checkpoint.state;
        let mut payload = CheckpointPayload::new(
            state.length,
            state.rope_delta,
            checkpoint.pending,
            Some(serde_json::json!({
                "kv_layers": state.kv_caches().len(),
                "linear_layers": state.linear_states().len(),
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
        let layers = usize::try_from(meta.get("kv_layers")?.as_u64()?).ok()?;
        let linear_layers = usize::try_from(meta.get("linear_layers")?.as_u64()?).ok()?;
        // Hybrid towers have a different number of full-attention and linear
        // layers (1B and 35B both do); only the total has to be non-empty and
        // each count bounded.
        if layers > 1024 || linear_layers > 1024 || layers + linear_layers == 0 {
            return None;
        }
        let mut kv = Vec::with_capacity(layers);
        for index in 0..layers {
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
        if !tensors.is_empty() {
            return None;
        }
        Some(GreedyCheckpoint {
            state: DecodeState::from_parts(kv, linear, length, rope_delta),
            tokens: tokens.to_vec(),
            pending,
        })
    }
}

/// Greedy text-decode runner over the shared burn model.
pub struct MinicpmRunner<B: Backend> {
    session: DecodeSession<GreedyAdapter<B>>,
}

impl<B: Backend> MinicpmRunner<B> {
    /// Default engine config; `RESIN_CONVERSATION_SLOTS=1` still applies as
    /// the experimental override.
    pub fn new(model: TextModel<B>, device: B::Device) -> Self {
        Self::with_config(model, device, ResinConfig::default())
    }

    /// Explicit engine config (`conversation_slots` enables #74 checkpoints).
    pub fn with_config(model: TextModel<B>, device: B::Device, config: ResinConfig) -> Self {
        Self::with_state_cache(model, device, config, None)
    }

    /// Explicit config plus the SSD state cache (#75). The caller derives the
    /// cache namespace from the model directory; the session never guesses a
    /// path.
    pub fn with_state_cache(
        model: TextModel<B>,
        device: B::Device,
        config: ResinConfig,
        cache: Option<crate::state_cache::StateCache>,
    ) -> Self {
        Self {
            session: DecodeSession::with_config_and_state_cache(
                GreedyAdapter {
                    model,
                    device,
                    options: Options {
                        temperature: 0.0,
                        ..Default::default()
                    },
                    batched_cycles: 0,
                },
                config,
                cache,
            ),
        }
    }

    /// Replace the experimental switches at runtime (same-process A/B arms,
    /// #76 quota measurements).
    pub fn set_config(&mut self, config: ResinConfig) {
        self.session.set_config(config);
    }

    /// The resolved engine switches (environment overrides folded in).
    pub fn config(&self) -> &ResinConfig {
        self.session.config()
    }

    /// Batched verify forwards run so far (0 when `multi_sequence_batch` is
    /// off or no plan had two rows).
    pub fn batch_cycles(&self) -> u64 {
        self.session.adapter().batched_cycles
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
}

impl<B: Backend> Runner<DecodeRequest> for MinicpmRunner<B> {
    fn forward(
        &mut self,
        plan: resin_core::engine::ExecutionPlan<DecodeRequest>,
    ) -> Result<resin_core::engine::OwnedTokenRows, Error> {
        self.session.forward(plan)
    }

    fn release(&mut self, request_ids: &[u64]) -> Result<(), Error> {
        self.session.release(request_ids)
    }
}
