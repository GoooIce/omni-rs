//! KV memory accounting for the resin session budget (#76).
//!
//! The session owns two kinds of resident KV state: live slots (one per
//! running request) and in-memory conversation checkpoints (#74/#75). Both are
//! measured by the adapter from tensor shapes and dtypes — never by device
//! allocator counters, which the runner does not see. [`MemoryStats`] is the
//! accounting the budget check and the #66 scripts read.
//!
//! The model-side helpers measure the same tensors the checkpoint encoder
//! writes: target KV per full-attention layer, the linear-attention conv and
//! recurrence state, and the DSpark draft K/V buffers (preallocated capacity,
//! not just the filled rows, because that is the resident allocation). Token
//! id vectors are not counted; the switch is a KV/state budget, not a total
//! process-memory cap.

use burn::tensor::{Tensor, backend::Backend};
use omni_minicpm_v::dspark::DraftState;
use omni_minicpm_v::model::DecodeState;

/// Resident KV bytes plus the release-order counters of one session.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryStats {
    /// Configured budget; `None` means the unbounded pre-#76 behavior.
    pub budget_bytes: Option<u64>,
    /// Slot KV/linear/draft state of every running request.
    pub live_bytes: u64,
    /// In-memory conversation checkpoints (turn + prefix). Adapter clones of
    /// the same state share tensor storage, so this is a conservative upper
    /// bound, which is the safe direction for eviction.
    pub checkpoint_bytes: u64,
    /// Tier 3 releases whose checkpoint was written to the SSD cache before
    /// being dropped from memory.
    pub spilled_checkpoints: u64,
    pub spilled_bytes: u64,
    /// Tier 1 self-drop: checkpoints of a key whose new request had no usable
    /// prefix checkpoint.
    pub dropped_self_checkpoints: u64,
    pub dropped_self_bytes: u64,
    /// Tier 3 drops that were not spilled (no SSD cache, encoding refused or
    /// the write failed).
    pub dropped_idle_checkpoints: u64,
    pub dropped_idle_bytes: u64,
    /// Bytes still above the budget after every permitted release ran. Only
    /// live slots can produce a non-zero value; the plan still completes.
    pub over_budget_bytes: u64,
}

impl MemoryStats {
    pub fn total_bytes(&self) -> u64 {
        self.live_bytes.saturating_add(self.checkpoint_bytes)
    }
}

/// Allocated bytes of one tensor: shape product times dtype width.
pub fn tensor_bytes<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> u64 {
    let elements = tensor
        .dims()
        .iter()
        .fold(1u64, |total, &dim| total.saturating_mul(dim as u64));
    elements.saturating_mul(tensor.dtype().size() as u64)
}

/// Resident bytes of one target decode state: KV pairs of every full-attention
/// layer with a written cache plus every linear layer's conv and recurrence
/// tensors.
pub fn state_bytes<B: Backend>(state: &DecodeState<B>) -> u64 {
    let kv: u64 = state
        .kv_caches()
        .iter()
        .filter_map(|cache| cache.snapshot())
        .map(|(key, value)| tensor_bytes(key).saturating_add(tensor_bytes(value)))
        .sum();
    let linear: u64 = state
        .linear_states()
        .iter()
        .map(|linear| tensor_bytes(&linear.conv).saturating_add(tensor_bytes(&linear.recurrent)))
        .sum();
    kv.saturating_add(linear)
}

/// Resident bytes of the DSpark draft context: the preallocated K/V buffers
/// (all layers, full capacity) because the checkpoint clones them whole.
pub fn draft_state_bytes<B: Backend>(draft: &DraftState<B>) -> u64 {
    draft
        .keys()
        .iter()
        .chain(draft.values().iter())
        .map(tensor_bytes)
        .sum()
}

/// Accumulated release-order counters (the live/checkpoint parts of
/// [`MemoryStats`] are measured when the stats are read).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MemoryCounters {
    pub spilled_checkpoints: u64,
    pub spilled_bytes: u64,
    pub dropped_self_checkpoints: u64,
    pub dropped_self_bytes: u64,
    pub dropped_idle_checkpoints: u64,
    pub dropped_idle_bytes: u64,
    pub over_budget_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::tensor_bytes;
    use burn::tensor::{Tensor, TensorData};

    type B = burn::backend::NdArray;

    #[test]
    fn tensor_bytes_multiply_shape_by_dtype_width() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let tensor = Tensor::<B, 2>::from_data(TensorData::new(vec![0.0f32; 12], [3, 4]), &device);
        assert_eq!(tensor_bytes(&tensor), 12 * 4);
        let rank_3 =
            Tensor::<B, 3>::from_data(TensorData::new(vec![0.0f32; 24], [2, 3, 4]), &device);
        assert_eq!(tensor_bytes(&rank_3), 24 * 4);
    }
}
