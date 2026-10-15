//! Hy-MT owns page lifetimes; the shared kernel owns GPU addressing and softmax.
use crate::paged::Batch;
use burn::tensor::{Tensor, backend::Backend};
pub(crate) use omni_kernel::paged::supported;
pub(crate) fn write<B: Backend>(
    atlas: Tensor<B, 5>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    batch: &Batch,
) -> Tensor<B, 5> {
    // SAFETY: the scheduler pins copy sources and destinations and completes
    // logits readback before the next mutation of this private atlas.
    unsafe {
        omni_kernel::paged::write(
            atlas,
            k,
            v,
            batch.rows.clone(),
            batch
                .copies
                .iter()
                .flat_map(|c| [*c.source as u32, *c.destination as u32])
                .collect(),
            None,
        )
    }
}
pub(crate) fn attend<B: Backend>(
    q: Tensor<B, 3>,
    atlas: &Tensor<B, 5>,
    batch: &Batch,
) -> Tensor<B, 3> {
    // SAFETY: the scheduler initializes each causal position before reading it.
    unsafe { omni_kernel::paged::attend(q, atlas, batch.rows.clone(), batch.table.clone()) }
}
