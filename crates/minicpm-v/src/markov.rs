//! Quantized markov-head scoring for the DSpark draft.
//!
//! The markov chain's correction `w2 · w1[prev]` re-reads the full
//! `[VOCAB, RANK]` f16 `markov_w2` (~127 MB) once per chain row on the
//! device. An int8 copy (four biased bytes per i32 word, one f32 scale per
//! output row, ~64 MB) halves that traffic; one kernel replaces the
//! gather + matmul + add launches per row and folds the `w1` lookup into
//! the index math (`prev` stays on the device as an id tensor). Scores keep
//! the f16 layout the argmax/top-k already consumes. Draft-only numerics:
//! quantization may flip near-tie proposals, which the greedy verify
//! corrects, so committed text is unaffected while τ can drift slightly.

use crate::metal_bridge::take_as;
use burn::tensor::{DType, Shape, Tensor as BurnTensor, TensorPrimitive, backend::Backend};
use burn_cubecl::tensor::CubeTensor;
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use half::f16;

const LANES: u32 = 64;

/// The quantized `markov_w2` payload: packed words and per-row scales.
pub(crate) struct MarkovQuant<B: Backend> {
    /// `[VOCAB, RANK / 4]` i32 words; byte `b` of word `w` holds
    /// `round(weight / scale) + 128`.
    pub words: BurnTensor<B, 2, burn::tensor::Int>,
    /// `[VOCAB]` f32 scales (`max |row| / 127`).
    pub scales: BurnTensor<B, 1>,
}

/// Pack `weights` (checkpoint layout `[vocab, rank]`) into [`MarkovQuant`].
/// Symmetric int8 with one scale per output row.
pub(crate) fn quantize_w2<B: Backend>(
    weights: &BurnTensor<B, 2>,
    vocab: usize,
    rank: usize,
    device: &B::Device,
) -> MarkovQuant<B> {
    assert_eq!(rank % 4, 0, "rank must pack into whole bytes");
    let values = weights
        .clone()
        .into_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .expect("contiguous w2 readback")
        .to_vec();
    let mut scales = Vec::with_capacity(vocab);
    let mut words = Vec::with_capacity(vocab * rank / 4);
    for row in 0..vocab {
        let slice = &values[row * rank..(row + 1) * rank];
        let max = slice.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        let scale = (max / 127.0).max(1e-12);
        scales.push(scale);
        for word in 0..rank / 4 {
            let mut packed = 0u32;
            for byte in 0..4 {
                let quantized = (slice[word * 4 + byte] / scale).round() + 128.0;
                let quantized = quantized.clamp(0.0, 255.0) as u32;
                packed |= quantized << (8 * byte);
            }
            words.push(packed as i32);
        }
    }
    MarkovQuant {
        words: BurnTensor::<B, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(words, [vocab, rank / 4]),
            burn::tensor::TensorCreationOptions::<B> {
                device: device.clone(),
                dtype: Some(DType::I32),
            },
        ),
        scales: BurnTensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(scales, [vocab]),
            burn::tensor::TensorCreationOptions::<B> {
                device: device.clone(),
                dtype: Some(DType::F32),
            },
        ),
    }
}

/// scores[node, j] = base[j] + scale[j] * Σ_r q[node? no — q[j, r]·w1[ids[node], r]
#[cube(launch_unchecked)]
fn markov_scores(
    base: &Tensor<f16>,    // [vocab]
    ids: &Tensor<i32>,     // [n] previous picks
    w1: &Tensor<f16>,      // [vocab, rank]
    words: &Tensor<i32>,   // [vocab, rank / 4]
    scales: &Tensor<f32>,  // [vocab]
    out: &mut Tensor<f16>, // [n, vocab]
    vocab: u32,
    #[comptime] rank: u32,
) {
    let node = CUBE_POS_X;
    let j = CUBE_POS_Y * CUBE_DIM_X + UNIT_POS_X;
    if j < vocab {
        let id = ids[node as usize];
        let words_row = j * (rank / 4);
        let mut acc = 0.0f32;
        for word in 0..rank / 4 {
            let packed = words[(words_row + word) as usize];
            let w1_base = id * rank as i32 + (word * 4) as i32;
            for byte in 0..4 {
                let quantized = f32::cast_from((packed >> (8 * byte)) & 0xFF) - 128.0;
                acc += quantized * f32::cast_from(w1[(w1_base + byte) as usize]);
            }
        }
        let corrected = f32::cast_from(base[j as usize]) + acc * scales[j as usize];
        out[(node * vocab + j) as usize] = f16::cast_from(corrected);
    }
}

/// One launch producing the chain's corrected scores `[n, vocab]` in f16.
/// `None` off Metal or on shapes the kernel does not cover; the caller
/// falls back to the burn gather + matmul + add chain.
pub(crate) fn try_scores<B: Backend>(
    base: BurnTensor<B, 2>,                   // [1, vocab] row of the block logits
    ids: BurnTensor<B, 2, burn::tensor::Int>, // [n, 1] previous picks
    w1: BurnTensor<B, 2>,                     // [vocab, rank]
    quant: &MarkovQuant<B>,
) -> Option<BurnTensor<B, 2>> {
    use burn::backend::wgpu::Metal;
    if !crate::metal_bridge::borrowable::<B, 2>() {
        return None;
    }
    let [_, vocab] = base.dims();
    let rank = w1.dims()[1];
    let n = ids.dims()[0];
    if vocab == 0 || n == 0 || rank % 4 != 0 || base.dtype() != DType::F16 {
        return None;
    }
    let base: BurnTensor<Metal, 2> = unsafe { take_as(base) };
    let ids: BurnTensor<Metal, 2, burn::tensor::Int> = unsafe { take_as(ids) };
    let w1: BurnTensor<Metal, 2> = unsafe { take_as(w1) };
    let words: BurnTensor<Metal, 2, burn::tensor::Int> = unsafe { take_as(quant.words.clone()) };
    let scales: BurnTensor<Metal, 1> = unsafe { take_as(quant.scales.clone()) };
    let base = crate::metal_bridge::cube_float(base)?;
    let ids = crate::metal_bridge::cube_int(ids);
    let w1 = crate::metal_bridge::cube_float(w1)?;
    let words = crate::metal_bridge::cube_int(words);
    let scales = crate::metal_bridge::cube_float(scales)?;
    let client = base.client.clone();
    let device = base.device.clone();
    let out = CubeTensor::new_contiguous(
        client.clone(),
        device,
        Shape::from([n, vocab]),
        client.empty(n * vocab * size_of::<f16>()),
        DType::F16,
    );
    unsafe {
        markov_scores::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(n as u32, (vocab as u32).div_ceil(LANES), 1),
            CubeDim::new_1d(LANES),
            base.into_tensor_arg(),
            ids.into_tensor_arg(),
            w1.into_tensor_arg(),
            words.into_tensor_arg(),
            scales.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            vocab as u32,
            rank as u32,
        );
    }
    let out = BurnTensor::<Metal, 2>::from_primitive(TensorPrimitive::Float(out));
    Some(unsafe { take_as(out) })
}

#[cfg(test)]
mod tests {
    use super::{MarkovQuant, quantize_w2, try_scores};
    use burn::backend::wgpu::{Metal, WgpuDevice};
    use burn::tensor::{DType, Int, Tensor, TensorData};

    /// The kernel's scores against a CPU mirror of the same quantized
    /// arithmetic (base + scale * Σ q·w1), on small shapes and on the real
    /// head's (vocab, rank) strided down to keep the test fast.
    #[test]
    fn markov_scores_match_quantized_reference() {
        type B = Metal;
        let device = WgpuDevice::default();
        let (vocab, rank, n) = (513usize, 256usize, 3usize);
        let value = |index: usize| ((index % 37) as f32 - 18.0) * 0.031;
        let w2_values: Vec<f32> = (0..vocab * rank).map(value).collect();
        let w1_values: Vec<f32> = (0..vocab * rank)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.05)
            .collect();
        let base_values: Vec<f32> = (0..vocab).map(|j| ((j % 17) as f32 - 8.0) * 0.5).collect();
        let ids_values: Vec<i64> = vec![7, 512, 0];
        let opts = burn::tensor::TensorCreationOptions::<B> {
            device: device.clone(),
            dtype: Some(DType::F16),
        };
        let w2 = Tensor::<B, 2>::from_data(
            TensorData::new(w2_values.clone(), [vocab, rank]),
            opts.clone(),
        );
        let w1 = Tensor::<B, 2>::from_data(
            TensorData::new(w1_values.clone(), [vocab, rank]),
            opts.clone(),
        );
        let base =
            Tensor::<B, 2>::from_data(TensorData::new(base_values.clone(), [1, vocab]), opts);
        let ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(ids_values.clone(), [n, 1]),
            burn::tensor::TensorCreationOptions::<B> {
                device: device.clone(),
                dtype: Some(DType::I32),
            },
        );
        let quant: MarkovQuant<B> = quantize_w2(&w2, vocab, rank, &device);
        let got = try_scores(base.clone(), ids.clone(), w1.clone(), &quant).expect("metal scores");

        // CPU mirror: unpack the same words/scales arithmetic in f32.
        let words = quant.words.into_data().as_slice::<i32>().unwrap().to_vec();
        let scales = quant.scales.into_data().as_slice::<f32>().unwrap().to_vec();
        let got_values = got
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let mut worst = 0.0f32;
        for node in 0..n {
            let id = ids_values[node] as usize;
            for j in 0..vocab {
                let mut acc = 0.0f32;
                for word in 0..rank / 4 {
                    let packed = words[j * rank / 4 + word] as u32;
                    for byte in 0..4 {
                        let q = (packed >> (8 * byte) & 0xFF) as f32 - 128.0;
                        acc += q * w1_values[id * rank + word * 4 + byte];
                    }
                }
                let expected = base_values[j] + acc * scales[j];
                let index = node * vocab + j;
                worst = worst.max((got_values[index] - expected).abs());
            }
        }
        assert!(worst < 5e-2, "worst {worst}");
    }
}
