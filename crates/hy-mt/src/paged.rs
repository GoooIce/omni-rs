//! Backend-neutral paging interface. Metal uses direct block-table reads;
//! other backends retain a Burn gather/attention reference implementation.
use crate::pages::{CopyPage, PAGE_SIZE};
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use omni_r2t2::tensor::{AttentionOptions, attention};

pub(crate) struct Segment {
    pub start: usize,
    pub count: usize,
    pub offset: usize,
    pub table: usize,
}
#[derive(Default)]
pub(crate) struct Batch {
    pub ids: Vec<u32>,
    pub positions: Vec<usize>,
    // Three integers per query: absolute position, block table offset, write slot.
    pub rows: Vec<u32>,
    pub table: Vec<u32>,
    pub copies: Vec<CopyPage>,
    pub segments: Vec<Segment>,
}
impl Batch {
    pub fn last_rows<B: Backend>(&self, device: &B::Device) -> Tensor<B, 1, Int> {
        Tensor::from_data(
            TensorData::new(
                self.segments
                    .iter()
                    .map(|s| (s.start + s.count - 1) as i32)
                    .collect(),
                [self.segments.len()],
            ),
            device,
        )
    }
}

pub(crate) fn write<B: Backend>(
    atlas: Tensor<B, 5>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    batch: &Batch,
) -> Tensor<B, 5> {
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    if crate::paged_kernel::supported::<B>(&atlas.device(), atlas.dtype()) {
        return crate::paged_kernel::write(atlas, k, v, batch);
    }
    write_reference(atlas, k, v, batch)
}
fn write_reference<B: Backend>(
    mut atlas: Tensor<B, 5>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    batch: &Batch,
) -> Tensor<B, 5> {
    let shape = atlas.dims();
    let heads = shape[3];
    for copy in &batch.copies {
        let source = *copy.source;
        let destination = *copy.destination;
        let values = atlas.clone().slice([
            source..source + 1,
            0..2,
            0..PAGE_SIZE,
            0..heads,
            0..shape[4],
        ]);
        atlas = atlas.slice_assign(
            [
                destination..destination + 1,
                0..2,
                0..PAGE_SIZE,
                0..heads,
                0..shape[4],
            ],
            values,
        );
    }
    let mut flat = atlas.reshape([shape[0] * 2 * PAGE_SIZE * heads, shape[4]]);
    for (which, values) in [k, v].into_iter().enumerate() {
        let values = values.reshape([batch.ids.len() * heads, shape[4]]);
        for (i, row) in batch.rows.as_chunks::<3>().0.iter().enumerate() {
            let slot = row[2] as usize;
            let base =
                (slot / PAGE_SIZE * 2 * PAGE_SIZE + which * PAGE_SIZE + slot % PAGE_SIZE) * heads;
            flat = flat.slice_assign(
                [base..base + heads, 0..shape[4]],
                values
                    .clone()
                    .slice([i * heads..(i + 1) * heads, 0..shape[4]]),
            );
        }
    }
    flat.reshape(shape)
}
pub(crate) fn attend<B: Backend>(
    q: Tensor<B, 3>,
    atlas: &Tensor<B, 5>,
    batch: &Batch,
) -> Tensor<B, 3> {
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    if crate::paged_kernel::supported::<B>(&q.device(), q.dtype()) {
        return crate::paged_kernel::attend(q, atlas, batch);
    }
    attend_reference(q, atlas, batch)
}
fn attend_reference<B: Backend>(
    q: Tensor<B, 3>,
    atlas: &Tensor<B, 5>,
    batch: &Batch,
) -> Tensor<B, 3> {
    let [_, heads, dim] = q.dims();
    let kv_heads = atlas.dims()[3];
    let device = q.device();
    let mut outputs = Vec::new();
    for s in &batch.segments {
        let len = s.offset + s.count;
        let pages = len.div_ceil(PAGE_SIZE);
        let indices = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                batch.table[s.table..s.table + pages]
                    .iter()
                    .map(|&id| id as i32)
                    .collect(),
                [pages],
            ),
            &device,
        );
        let selected = atlas.clone().select(0, indices);
        let kv = |which: usize| {
            selected
                .clone()
                .slice([
                    0..pages,
                    which..which + 1,
                    0..PAGE_SIZE,
                    0..kv_heads,
                    0..dim,
                ])
                .reshape([1, pages * PAGE_SIZE, kv_heads, dim])
                .slice([0..1, 0..len, 0..kv_heads, 0..dim])
                .swap_dims(1, 2)
        };
        let queries = q
            .clone()
            .slice([s.start..s.start + s.count, 0..heads, 0..dim])
            .reshape([1, s.count, heads, dim])
            .swap_dims(1, 2);
        outputs.push(
            attention(
                queries,
                kv(0),
                kv(1),
                AttentionOptions {
                    causal: true,
                    offset: s.offset,
                    ..Default::default()
                },
            )
            .swap_dims(1, 2)
            .reshape([s.count, heads, dim]),
        );
    }
    Tensor::cat(outputs, 0)
}
