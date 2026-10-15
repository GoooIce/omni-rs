//! Pack-quantized int4 loading (compressed-tensors pack-quantized semantics:
//! symmetric, group 128, two 4-bit values per byte, low nibble first).
//!
//! The checkpoint stays packed int4 on disk and in the read-only mmap; weights
//! dequantize once at the load point and compute in FP16/FP32 (ADR-0001's
//! W4A16 route: no re-quantization, no hand-written kernels). Stock
//! burn-cubecl 0.21's fused sub-byte matmul mis-indexes per-block scales
//! beyond the first block (verified by probe, 2026-10-15), so the in-kernel
//! path is not used; load-point dequantization gives identical numerics.
use burn::tensor::{FloatDType, Tensor, TensorData, backend::Backend};
use burn_tensor::quantization::{
    QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};
use omni_r2t2::{Precision, Result, weights::WeightSource};

/// Default compressed-tensors group size for this model family.
pub const GROUP_SIZE: usize = 128;

/// Unpack `packed` (two 4-bit values per byte, low nibble first, row-major)
/// into sign-extended i8 values.
pub fn unpack_int4(packed: &[u8]) -> Vec<i8> {
    let mut out = Vec::with_capacity(packed.len() * 2);
    for byte in packed {
        out.push(((byte & 0xF) << 4) as i8 >> 4);
        out.push(((byte >> 4) << 4) as i8 >> 4);
    }
    out
}

/// Scheme for the checkpoint layout `[out, in]` with row-major group scales.
pub fn scheme(group: usize) -> QuantScheme {
    QuantScheme {
        value: QuantValue::Q4S,
        param: QuantParam::F32,
        store: QuantStore::PackedU32(0),
        level: QuantLevel::block([1, group as u8]),
        mode: QuantMode::Symmetric,
    }
}

/// A dequantized int4 weight, `[out, in]` like the checkpoint.
pub struct PackedWeight<B: Backend>(Tensor<B, 2>);

impl<B: Backend> PackedWeight<B> {
    /// `x @ W^T` with `W` stored `[out, in]`.
    pub fn matmul_transpose(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.matmul(self.0.clone().transpose())
    }

    pub fn weight(&self) -> &Tensor<B, 2> {
        &self.0
    }
}

#[allow(clippy::chunks_exact_to_as_chunks)] // as_chunks needs a newer MSRV than the workspace
fn bf16_scale_bytes(bytes: &[u8], count: usize) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .take(count)
        .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect()
}

/// Load one `name`/`name_scale` pair as a packed int4 weight `[out, in]`.
///
/// `quantized` selects the GPU packed representation; the CPU reference path
/// passes false and receives exact dequantized values.
pub fn load_packed<B: Backend>(
    w: &impl WeightSource<B>,
    name: &str,
    shape: [usize; 2],
    group: usize,
    precision: Precision,
    device: &B::Device,
) -> Result<PackedWeight<B>> {
    let [out_dim, in_dim] = shape;
    let raw = w.raw_bytes(name)?;
    let values = unpack_int4(&raw);
    if values.len() != out_dim * in_dim {
        return Err(omni_r2t2::Error::Checkpoint(format!(
            "{name}: packed {} values for shape {shape:?}",
            values.len()
        )));
    }
    let scale_bytes = w.raw_bytes(&format!("{name}_scale"))?;
    let scales = bf16_scale_bytes(&scale_bytes, out_dim * in_dim.div_ceil(group));

    // One-time load-point dequantization; compute proceeds at `precision`.
    let dequantized: Vec<f32> = values
        .chunks(group)
        .zip(scales.iter().cycle())
        .flat_map(|(chunk, scale)| chunk.iter().map(move |v| *v as f32 * scale))
        .collect();
    let dtype = match precision {
        Precision::F16 => FloatDType::F16,
        Precision::F32 => FloatDType::F32,
    };
    Ok(PackedWeight(
        Tensor::<B, 2>::from_data(TensorData::new(dequantized, shape), device).cast(dtype),
    ))
}

/// Packed int4 payload size in bytes (two values per byte) plus bf16 scales.
pub fn packed_footprint(out_dim: usize, in_dim: usize, group: usize) -> usize {
    out_dim * in_dim / 2 + out_dim * in_dim.div_ceil(group) * 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use safetensors::{Dtype, SafeTensors};

    fn fixture() -> Vec<u8> {
        std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/pack-quantized-int4.safetensors"),
        )
        .unwrap()
    }

    // as_chunks would need a newer MSRV than the workspace
    #[allow(clippy::chunks_exact_to_as_chunks)]
    fn tensor_f32(tensors: &SafeTensors<'_>, name: &str) -> Vec<f32> {
        let view = tensors.tensor(name).unwrap();
        match view.dtype() {
            Dtype::F32 => view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            Dtype::U16 | Dtype::BF16 => view
                .data()
                .chunks_exact(2)
                .map(|b| half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect(),
            other => panic!("unexpected fixture dtype {other:?}"),
        }
    }

    #[test]
    fn unpacked_values_match_llm_compressor_semantics() {
        let bytes = fixture();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let packed = tensors.tensor("quant.weight").unwrap();
        let values = unpack_int4(packed.data());
        let original = tensor_f32(&tensors, "quant.weight_original");
        let scales = tensor_f32(&tensors, "quant.weight_scale");
        let group = GROUP_SIZE;
        let mut worst = 0f32;
        for (index, value) in values.iter().enumerate() {
            let scale = scales[index / group];
            worst = worst.max(((*value as f32 * scale) - original[index]).abs());
        }
        let step = scales.iter().cloned().fold(0f32, f32::max);
        assert!(
            worst <= step / 2.0 + 1e-6,
            "worst {worst} vs half step {}",
            step / 2.0
        );
    }

    #[test]
    fn load_point_dequant_matmul_matches_fixture_reference() {
        type B = NdArray;
        let bytes = fixture();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let weights = omni_r2t2::weights::Weights::from_bytes(vec![bytes.clone()]).unwrap();
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let packed = load_packed::<B>(
            &weights,
            "quant.weight",
            [64, 256],
            GROUP_SIZE,
            Precision::F32,
            &device,
        )
        .unwrap();
        let x_values = tensor_f32(&tensors, "input");
        let x = Tensor::<B, 2>::from_data(TensorData::new(x_values, [8, 256]), &device);
        let output = packed.matmul_transpose(x);
        let actual = output
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let expected = tensor_f32(&tensors, "output_quant_matmul");
        let mut relative = 0f64;
        for (a, b) in actual.iter().zip(&expected) {
            relative = relative.max((a - b).abs() as f64 / b.abs().max(1e-3) as f64);
        }
        assert!(relative < 1e-5, "worst relative {relative}");
    }

    #[test]
    fn packed_payload_is_about_a_quarter_of_bf16() {
        let bytes = fixture();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let packed = tensors.tensor("quant.weight").unwrap().data().len();
        let scales = tensors.tensor("quant.weight_scale").unwrap().data().len();
        let bf16 = 64 * 256 * 2;
        assert_eq!(packed + scales, packed_footprint(64, 256, GROUP_SIZE));
        // int4 payload + bf16 scales ≈ 1/4 of the bf16 tensor.
        assert!((packed + scales) as f64 / (bf16 as f64) < 0.34);
    }
}

#[cfg(all(test, feature = "metal"))]
mod gpu_tests {
    use super::*;
    use burn::backend::wgpu::{Metal, WgpuDevice};
    use safetensors::SafeTensors;

    // as_chunks would need a newer MSRV than the workspace
    #[allow(clippy::chunks_exact_to_as_chunks)]
    fn tensor_f32(tensors: &SafeTensors<'_>, name: &str) -> Vec<f32> {
        tensors
            .tensor(name)
            .unwrap()
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    #[test]
    fn fp16_compute_matches_reference() {
        let bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/pack-quantized-int4.safetensors"),
        )
        .unwrap();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        let weights = omni_r2t2::weights::Weights::from_bytes(vec![bytes.clone()]).unwrap();
        let device = WgpuDevice::default();
        let packed = load_packed::<Metal>(
            &weights,
            "quant.weight",
            [64, 256],
            GROUP_SIZE,
            Precision::F16,
            &device,
        )
        .unwrap();
        // Packed checkpoint footprint stays ≈1/4 of bf16 storage.
        assert!((packed_footprint(64, 256, GROUP_SIZE) as f64) / ((64 * 256 * 2) as f64) < 0.34);
        let x_values = tensor_f32(&tensors, "input");
        let x = Tensor::<Metal, 2>::from_data(TensorData::new(x_values, [8, 256]), &device)
            .cast(FloatDType::F16);
        let output = packed.matmul_transpose(x);
        let actual = output
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let expected = tensor_f32(&tensors, "output_quant_matmul");
        let mut worst = 0f64;
        for (a, b) in actual.iter().zip(&expected) {
            worst = worst.max((a - b).abs() as f64 / b.abs().max(0.1) as f64);
        }
        // FP16 storage and accumulation over 256-element dots vs the f32
        // reference: bounded by a few percent at the smallest magnitudes.
        assert!(worst < 5e-2, "worst relative {worst}");
    }
}
