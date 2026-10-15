use crate::{
    paged::{self, Batch, Segment},
    pages::{CopyPage, PAGE_SIZE},
};
use burn::tensor::{FloatDType, Tensor, TensorData, backend::Backend, f16};
use std::sync::Arc;

fn value(index: usize) -> f32 {
    ((index * 73 % 997) as f32 / 997.0 - 0.5) * 1.4
}
fn verify<B: Backend>(dim: usize, dtype: FloatDType, split_offset: usize) {
    let device = Default::default();
    let long_pages = (split_offset + 2).div_ceil(PAGE_SIZE);
    let (pages, heads, kv_heads) = (long_pages + 40, 4, 2);
    let destination = long_pages + 30;
    let quantize = |x: f32| {
        if dtype == FloatDType::F16 {
            f16::from_f32(x).to_f32()
        } else {
            x
        }
    };
    let mut expected: Vec<f32> = (0..pages * 2 * PAGE_SIZE * kv_heads * dim)
        .map(|i| quantize(value(i)))
        .collect();
    let original = expected.clone();
    let mut batch = Batch::default();
    for (offset, count, page_ids) in [
        (15, 3, vec![destination as u32, 7]),
        (
            split_offset,
            2,
            (10..10 + long_pages as u32).rev().collect(),
        ),
        (0, 1, vec![1]),
    ] {
        let start = batch.ids.len();
        let table = batch.table.len();
        batch.table.extend(&page_ids);
        for pos in offset..offset + count {
            batch.ids.push(0);
            batch.positions.push(pos);
            batch.rows.extend([
                pos as u32,
                table as u32,
                page_ids[pos / PAGE_SIZE] * 16 + (pos % 16) as u32,
            ]);
        }
        batch.segments.push(Segment {
            start,
            count,
            offset,
            table,
        });
    }
    batch.copies.push(CopyPage {
        source: Arc::new(3),
        destination: Arc::new(destination),
    });
    let page_width = 2 * PAGE_SIZE * kv_heads * dim;
    expected[destination * page_width..(destination + 1) * page_width]
        .copy_from_slice(&original[3 * page_width..4 * page_width]);
    let n = batch.ids.len();
    let qdata: Vec<_> = (0..heads * n * dim)
        .map(|i| quantize(value(i + 91)))
        .collect();
    let kdata: Vec<_> = (0..kv_heads * n * dim)
        .map(|i| quantize(value(i + 19)))
        .collect();
    let vdata: Vec<_> = (0..kv_heads * n * dim)
        .map(|i| quantize(value(i + 371)))
        .collect();
    for row in 0..n {
        let slot = batch.rows[row * 3 + 2] as usize;
        for h in 0..kv_heads {
            for d in 0..dim {
                let base = ((slot / 16 * 2 * 16 + slot % 16) * kv_heads + h) * dim + d;
                expected[base] = kdata[(h * n + row) * dim + d];
                expected[base + 16 * kv_heads * dim] = vdata[(h * n + row) * dim + d];
            }
        }
    }
    let mut oracle = vec![0.0; n * heads * dim];
    for row in 0..n {
        for h in 0..heads {
            let length = batch.rows[row * 3] as usize + 1;
            let table = batch.rows[row * 3 + 1] as usize;
            let kv_head = h / (heads / kv_heads);
            let mut scores = Vec::new();
            for t in 0..length {
                let page = batch.table[table + t / 16] as usize;
                let base = ((page * 2 * 16 + t % 16) * kv_heads + kv_head) * dim;
                scores.push(
                    (0..dim)
                        .map(|d| qdata[(h * n + row) * dim + d] as f64 * expected[base + d] as f64)
                        .sum::<f64>()
                        / (dim as f64).sqrt(),
                );
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let sum = scores.iter().map(|s| (s - max).exp()).sum::<f64>();
            for d in 0..dim {
                oracle[(row * heads + h) * dim + d] = scores
                    .iter()
                    .enumerate()
                    .map(|(t, s)| {
                        let page = batch.table[table + t / 16] as usize;
                        let base = ((page * 2 * 16 + 16 + t % 16) * kv_heads + kv_head) * dim;
                        (s - max).exp() / sum * expected[base + d] as f64
                    })
                    .sum::<f64>() as f32;
            }
        }
    }
    let atlas = Tensor::<B, 5>::from_data(
        TensorData::new(original, [pages, 2, 16, kv_heads, dim]),
        &device,
    )
    .cast(dtype);
    let k = Tensor::<B, 3>::from_data(TensorData::new(kdata, [kv_heads, n, dim]), &device)
        .cast(dtype)
        .swap_dims(0, 1);
    let v = Tensor::<B, 3>::from_data(TensorData::new(vdata, [kv_heads, n, dim]), &device)
        .cast(dtype)
        .swap_dims(0, 1);
    let q = Tensor::<B, 3>::from_data(TensorData::new(qdata, [heads, n, dim]), &device)
        .cast(dtype)
        .swap_dims(0, 1);
    let atlas = paged::write(atlas, k, v, &batch);
    let actual = paged::attend(q, &atlas, &batch)
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap();
    let error = actual
        .iter()
        .zip(&oracle)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(actual.iter().all(|x| x.is_finite()));
    assert!(
        error
            < if dtype == FloatDType::F16 {
                0.001
            } else {
                0.000003
            },
        "dim={dim}, {dtype:?}: {error}"
    );
    assert_eq!(
        atlas.into_data().convert::<f32>().to_vec::<f32>().unwrap(),
        expected,
        "only designated KV slots may change"
    );
}
#[test]
fn cpu_paged_attention_matches_scalar_oracle() {
    verify::<burn::backend::NdArray>(8, FloatDType::F32, 257);
}
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_paged_attention_handles_gqa_strides_tails_cow_and_split_context() {
    crate::test_model::init_metal();
    for dim in [8, 64, 128] {
        for dtype in [FloatDType::F32, FloatDType::F16] {
            verify::<burn::backend::wgpu::Metal>(dim, dtype, 257);
        }
    }
}
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_paged_attention_handles_8192_token_contexts() {
    crate::test_model::init_metal();
    for dtype in [FloatDType::F32, FloatDType::F16] {
        verify::<burn::backend::wgpu::Metal>(128, dtype, 8190);
    }
}
