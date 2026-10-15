//! PagedAttention in Rust/CubeCL: noncontiguous block-table addressing, GQA,
//! causal ragged queries and split-K online softmax with FP32 accumulation.
//! Callers own page lifetimes and serialize atlas writes after earlier readers,
//! either by completing a scheduler step or supplying its reader dependency.
const PAGE_SIZE: usize = 16;
use crate::RawTensor;
use burn::tensor::{DType, Shape, Tensor as BurnTensor, TensorPrimitive, backend::Backend, f16};
use cubecl::{prelude::*, wgpu::WgpuRuntime};

const PARTITION_SIZE: usize = 256;

#[cube(launch_unchecked)]
fn copy_pages<F: Float>(atlas: &mut Tensor<F>, copies: &Tensor<u32>, #[comptime] width: usize) {
    let i = ABSOLUTE_POS;
    let pair = i / width;
    if pair < copies.len() / 2 {
        let column = i % width;
        let source = copies[pair * 2] as usize * width + column;
        let destination = copies[pair * 2 + 1] as usize * width + column;
        atlas[destination] = atlas[source];
    }
}
#[cube(launch_unchecked)]
fn store_kv<F: Float>(
    atlas: &mut Tensor<F>,
    k: &Tensor<F>,
    v: &Tensor<F>,
    rows: &Tensor<u32>,
    #[comptime] heads: usize,
    #[comptime] dim: usize,
) {
    let i = ABSOLUTE_POS;
    if i < k.shape(0) * heads * dim {
        let row = i / (heads * dim);
        let h = i / dim % heads;
        let d = i % dim;
        let slot = rows[row * 3 + 2] as usize;
        let base = ((slot / PAGE_SIZE * 2 * PAGE_SIZE + slot % PAGE_SIZE) * heads + h) * dim + d;
        atlas[base] = k[row * k.stride(0) + h * k.stride(1) + d * k.stride(2)];
        atlas[base + PAGE_SIZE * heads * dim] =
            v[row * v.stride(0) + h * v.stride(1) + d * v.stride(2)];
    }
}

#[cube(launch_unchecked)]
fn attention_parts<F: Float>(
    q: &Tensor<F>,
    atlas: &Tensor<F>,
    rows: &Tensor<u32>,
    table: &Tensor<u32>,
    partial: &mut Tensor<f32>,
    #[comptime] heads: usize,
    #[comptime] kv_heads: usize,
    #[comptime] dim: usize,
    #[comptime] parts: usize,
) {
    let row = CUBE_POS_X as usize;
    let h = CUBE_POS_Y as usize;
    let part = CUBE_POS_Z as usize;
    let lane = UNIT_POS_PLANE as usize;
    let kv_head = h / (heads / kv_heads);
    let position = rows[row * 3] as usize;
    let table_start = rows[row * 3 + 1] as usize;
    let columns = comptime!(dim.div_ceil(32));
    let mut query = Array::<f32>::new(columns);
    let mut acc = Array::<f32>::new(columns);
    #[unroll]
    for j in 0..columns {
        let d = lane + j * 32;
        query[j] = 0.0;
        if d < dim {
            query[j] = f32::cast_from(q[row * q.stride(0) + h * q.stride(1) + d * q.stride(2)]);
        }
        acc[j] = 0.0;
    }
    let mut maximum = f32::new(f32::MIN);
    let mut denominator = 0.0f32;
    let end = usize::min(position + 1, (part + 1) * PARTITION_SIZE);
    for t in part * PARTITION_SIZE..end {
        let physical = table[table_start + t / PAGE_SIZE] as usize;
        let base = ((physical * 2 * PAGE_SIZE + t % PAGE_SIZE) * kv_heads + kv_head) * dim;
        let mut dot = 0.0f32;
        #[unroll]
        for j in 0..columns {
            let d = lane + j * 32;
            if d < dim {
                dot += query[j] * f32::cast_from(atlas[base + d]);
            }
        }
        let score = plane_sum(dot) / f32::sqrt(dim as f32);
        let next = f32::max(maximum, score);
        let correction = f32::exp(maximum - next);
        let weight = f32::exp(score - next);
        denominator = denominator * correction + weight;
        #[unroll]
        for j in 0..columns {
            let d = lane + j * 32;
            if d < dim {
                acc[j] = acc[j] * correction
                    + weight * f32::cast_from(atlas[base + PAGE_SIZE * kv_heads * dim + d]);
            }
        }
        maximum = next;
    }
    let out = ((row * heads + h) * parts + part) * (dim + 2);
    #[unroll]
    for j in 0..columns {
        let d = lane + j * 32;
        if d < dim {
            partial[out + d] = acc[j];
        }
    }
    if lane == 0 {
        partial[out + dim] = maximum;
        partial[out + dim + 1] = denominator;
    }
}
#[cube(launch_unchecked)]
fn reduce_parts<F: Float>(
    partial: &Tensor<f32>,
    out: &mut Tensor<F>,
    #[comptime] heads: usize,
    #[comptime] dim: usize,
    #[comptime] parts: usize,
) {
    let row = CUBE_POS_X as usize;
    let h = CUBE_POS_Y as usize;
    let lane = UNIT_POS_PLANE as usize;
    let base = (row * heads + h) * parts * (dim + 2);
    let mut maximum = f32::new(f32::MIN);
    for p in 0..parts {
        maximum = f32::max(maximum, partial[base + p * (dim + 2) + dim]);
    }
    let mut denominator = 0.0f32;
    let columns = comptime!(dim.div_ceil(32));
    let mut acc = Array::<f32>::new(columns);
    #[unroll]
    for j in 0..columns {
        acc[j] = 0.0;
    }
    for p in 0..parts {
        let src = base + p * (dim + 2);
        let weight = f32::exp(partial[src + dim] - maximum);
        denominator += weight * partial[src + dim + 1];
        #[unroll]
        for j in 0..columns {
            let d = lane + j * 32;
            if d < dim {
                acc[j] += weight * partial[src + d];
            }
        }
    }
    #[unroll]
    for j in 0..columns {
        let d = lane + j * 32;
        if d < dim {
            out[(row * heads + h) * dim + d] = F::cast_from(acc[j] / denominator);
        }
    }
}

fn metadata(reference: &RawTensor, values: &[u32]) -> RawTensor {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    RawTensor::new_contiguous(
        reference.client.clone(),
        reference.device.clone(),
        Shape::new([values.len()]),
        reference.client.create_from_slice(&bytes),
        DType::U32,
    )
}
fn raw_write<F: Float + CubeElement>(
    atlas: RawTensor,
    k: RawTensor,
    v: RawTensor,
    rows: &[u32],
    copies: &[u32],
) -> RawTensor {
    let shape = atlas.meta.shape();
    let (heads, dim) = (shape[3], shape[4]);
    let row_data = metadata(&atlas, rows);
    // SAFETY: the caller pins and bounds every page; the dispatch dependency
    // or the caller's completed readback serializes earlier readers.
    unsafe {
        if !copies.is_empty() {
            let copy_data = metadata(&atlas, copies);
            let width = 2 * PAGE_SIZE * heads * dim;
            copy_pages::launch_unchecked::<F, WgpuRuntime>(
                &atlas.client,
                CubeCount::Static((copies.len() / 2 * width).div_ceil(256) as u32, 1, 1),
                CubeDim::new_1d(256),
                atlas.clone().into_tensor_arg(),
                copy_data.into_tensor_arg(),
                width,
            );
        }
        store_kv::launch_unchecked::<F, WgpuRuntime>(
            &atlas.client,
            CubeCount::Static((k.meta.shape()[0] * heads * dim).div_ceil(256) as u32, 1, 1),
            CubeDim::new_1d(256),
            atlas.clone().into_tensor_arg(),
            k.into_tensor_arg(),
            v.into_tensor_arg(),
            row_data.into_tensor_arg(),
            heads,
            dim,
        );
    }
    atlas
}
fn raw_attention<F: Float + CubeElement>(
    q: RawTensor,
    atlas: RawTensor,
    rows: &[u32],
    table: &[u32],
    parts: usize,
) -> RawTensor {
    let (n, heads, dim) = (q.meta.shape()[0], q.meta.shape()[1], q.meta.shape()[2]);
    let kv_heads = atlas.meta.shape()[3];
    let partial = RawTensor::new_contiguous(
        q.client.clone(),
        q.device.clone(),
        Shape::new([n, heads, parts, dim + 2]),
        q.client.empty(n * heads * parts * (dim + 2) * 4),
        DType::F32,
    );
    let out = RawTensor::new_contiguous(
        q.client.clone(),
        q.device.clone(),
        Shape::new([n, heads, dim]),
        q.client.empty(n * heads * dim * q.dtype.size()),
        q.dtype,
    );
    let row_data = metadata(&q, rows);
    let table_data = metadata(&q, table);
    // SAFETY: the caller supplies valid nonempty causal ranges, block tables
    // and GQA shapes. One 32-lane SIMD group owns each output partition.
    unsafe {
        attention_parts::launch_unchecked::<F, WgpuRuntime>(
            &q.client,
            CubeCount::Static(n as u32, heads as u32, parts as u32),
            CubeDim::new_1d(32),
            q.clone().into_tensor_arg(),
            atlas.into_tensor_arg(),
            row_data.into_tensor_arg(),
            table_data.into_tensor_arg(),
            partial.clone().into_tensor_arg(),
            heads,
            kv_heads,
            dim,
            parts,
        );
        reduce_parts::launch_unchecked::<F, WgpuRuntime>(
            &q.client,
            CubeCount::Static(n as u32, heads as u32, 1),
            CubeDim::new_1d(32),
            partial.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            heads,
            dim,
            parts,
        );
    }
    out
}
pub fn supported<B: Backend>(device: &B::Device, dtype: DType) -> bool {
    let Some(device) = crate::device::<B>(device) else {
        return false;
    };
    let client = <WgpuRuntime as cubecl::Runtime>::client(device);
    let hardware = &client.properties().hardware;
    hardware.plane_size_min == 32
        && hardware.plane_size_max == 32
        && matches!(dtype, DType::F16 | DType::F32)
}
/// Store rows in a private page atlas, optionally depending on its last reader.
///
/// # Safety
/// `supported` must be true. The atlas is contiguous [pages,2,16,kv_heads,dim];
/// K/V are [rows,kv_heads,dim] on the same device and dtype. Metadata has one
/// (position,table offset,slot) triple per row and pairs of source/destination
/// page indices. Copy sources must not overlap any copy destination.
/// Every slot/copy must lie inside the atlas, destinations must be disjoint,
/// and all earlier atlas readers must finish before this operation. Retain the
/// previous attention tensor as `reader` when building successive lazy graphs.
pub unsafe fn write<B: Backend>(
    atlas: BurnTensor<B, 5>,
    k: BurnTensor<B, 3>,
    v: BurnTensor<B, 3>,
    rows: Vec<u32>,
    copies: Vec<u32>,
    reader: Option<BurnTensor<B, 3>>,
) -> BurnTensor<B, 5> {
    let shape = Shape::new(atlas.dims());
    let run = move |atlas: RawTensor, k: RawTensor, v: RawTensor| match atlas.dtype {
        DType::F16 => raw_write::<f16>(atlas, k, v, &rows, &copies),
        DType::F32 => raw_write::<f32>(atlas, k, v, &rows, &copies),
        _ => unreachable!(),
    };
    let output = if let Some(reader) = reader {
        crate::dispatch::<B, 4>(
            "paged_kv_store_after_reader",
            [
                atlas.into_primitive().tensor(),
                k.into_primitive().tensor(),
                v.into_primitive().tensor(),
                reader.into_primitive().tensor(),
            ],
            shape,
            move |[atlas, k, v, _]| run(atlas, k, v),
        )
    } else {
        crate::dispatch::<B, 3>(
            "paged_kv_store",
            [
                atlas.into_primitive().tensor(),
                k.into_primitive().tensor(),
                v.into_primitive().tensor(),
            ],
            shape,
            move |[atlas, k, v]| run(atlas, k, v),
        )
    }
    .expect("supported paging backend");
    BurnTensor::from_primitive(TensorPrimitive::Float(output))
}
/// Read causal attention directly from the physical page table.
///
/// # Safety
/// `supported` must be true; atlas is contiguous [pages,2,16,kv_heads,dim] and
/// Q is [rows,heads,dim], on the same device and dtype, with heads divisible
/// by kv_heads. Metadata has exactly one triple per nonempty query row.
/// Rows are triples (position, table offset, slot); all positions and referenced
/// pages must be initialized and in bounds. Shapes/dtypes must implement GQA.
/// No writer may mutate this atlas until this operation completes.
pub unsafe fn attend<B: Backend>(
    q: BurnTensor<B, 3>,
    atlas: &BurnTensor<B, 5>,
    rows: Vec<u32>,
    table: Vec<u32>,
) -> BurnTensor<B, 3> {
    let shape = Shape::new(q.dims());
    let parts =
        (rows.chunks_exact(3).map(|r| r[0] as usize).max().unwrap() + 1).div_ceil(PARTITION_SIZE);
    let output = crate::dispatch::<B, 2>(
        "paged_attention",
        [
            q.into_primitive().tensor(),
            atlas.clone().into_primitive().tensor(),
        ],
        shape,
        move |[q, atlas]| match q.dtype {
            DType::F16 => raw_attention::<f16>(q, atlas, &rows, &table, parts),
            DType::F32 => raw_attention::<f32>(q, atlas, &rows, &table, parts),
            _ => unreachable!(),
        },
    )
    .expect("supported paging backend");
    BurnTensor::from_primitive(TensorPrimitive::Float(output))
}
