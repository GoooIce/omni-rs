//! Pack-quantized int4 loading (compressed-tensors pack-quantized semantics:
//! symmetric, group 128, two 4-bit values per byte, low nibble first).
//!
//! On cubecl backends the weight becomes a Burn Q4S quantized tensor — packed
//! int4 stays resident and matmul dequantizes in kernel (W4A16, ADR-0001; no
//! re-quantization, no hand-written kernels). The NdArray reference backend
//! cannot hold sub-byte tensors, so it dequantizes once at the load point —
//! identical values, reference-path memory only.
use burn::tensor::{FloatDType, Tensor, TensorData, backend::Backend};
use burn_tensor::quantization::{
    QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};
use omni_r2t2::{Precision, Result, weights::WeightSource};

/// Default compressed-tensors group size for this model family.
pub const GROUP_SIZE: usize = 128;

/// Unpack `packed` (two 4-bit values per byte, low nibble first, row-major)
/// into i8 values. compressed-tensors `pack_to_int32` stores unsigned nibbles
/// offset by `1 << (bits-1)` — a value of 8 means zero, 0 means -8 — not
/// two's-complement sign-extended nibbles.
pub fn unpack_int4(packed: &[u8]) -> Vec<i8> {
    let mut out = Vec::with_capacity(packed.len() * 2);
    for byte in packed {
        out.push((byte & 0xF) as i8 - 8);
        out.push((byte >> 4) as i8 - 8);
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

/// How the loaded weight participates in matmul.
pub enum PackedWeight<B: Backend> {
    /// GPU: packed int4 stays resident; matmul dequantizes in kernel.
    Quantized(Tensor<B, 2>),
    /// CPU reference: dequantized values.
    Dequantized(Tensor<B, 2>),
}

impl<B: Backend> PackedWeight<B> {
    /// `x @ W^T` with `W` stored `[out, in]` like the checkpoint.
    pub fn matmul_transpose(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            PackedWeight::Quantized(weight) => x.matmul(weight.clone().transpose()),
            PackedWeight::Dequantized(weight) => x.matmul(weight.clone().transpose()),
        }
    }

    /// Resident value bytes (packed payload without scales).
    pub fn resident_bytes(&self) -> usize {
        match self {
            // Packed u32 storage: half a byte per value.
            PackedWeight::Quantized(weight) => {
                let [out, in_dim] = weight.dims();
                out * in_dim / 2
            }
            PackedWeight::Dequantized(weight) => {
                let [out, in_dim] = weight.dims();
                out * in_dim * std::mem::size_of::<f32>()
            }
        }
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
/// How packed int4 weights should be represented after loading.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LoadMode {
    /// Packed quantized tensor (cubecl backends): int4 resident, fused
    /// dequant matmul.
    Packed,
    /// One-time load-point dequantization (NdArray reference path).
    Dequantize,
}

pub fn load_packed<B: Backend>(
    w: &impl WeightSource<B>,
    name: &str,
    scale_name: &str,
    shape: [usize; 2],
    mode: LoadMode,
    precision: Precision,
    device: &B::Device,
) -> Result<PackedWeight<B>> {
    let [out_dim, in_dim] = shape;
    let group = GROUP_SIZE;
    let raw = w.raw_bytes(name)?;
    let values = unpack_int4(&raw);
    if values.len() != out_dim * in_dim {
        return Err(omni_r2t2::Error::Checkpoint(format!(
            "{name}: packed {} values for shape {shape:?}",
            values.len()
        )));
    }
    let scale_bytes = w.raw_bytes(scale_name)?;
    let scales = bf16_scale_bytes(&scale_bytes, out_dim * in_dim.div_ceil(group));

    if mode == LoadMode::Packed {
        // Packed u32 payload (little endian, first value in the lowest
        // nibble) with f32 scales appended — cubecl's quantized layout.
        let scheme = scheme(group);
        let mut bytes = Vec::with_capacity(values.len() / 2 + scales.len() * 4);
        for chunk in values.chunks(8) {
            let mut packed = 0u32;
            for (nibble, value) in chunk.iter().enumerate() {
                packed |= (*value as u32 & 0xF) << (nibble * 4);
            }
            bytes.extend_from_slice(&packed.to_le_bytes());
        }
        for scale in &scales {
            bytes.extend_from_slice(&scale.to_le_bytes());
        }
        let data = TensorData::from_bytes_vec(bytes, shape, burn::tensor::DType::QFloat(scheme));
        Ok(PackedWeight::Quantized(Tensor::<B, 2>::from_data(
            data, device,
        )))
    } else {
        // Reference backend: one-time load-point dequantization.
        let dequantized: Vec<f32> = values
            .chunks(group)
            .zip(scales.iter().cycle())
            .flat_map(|(chunk, scale)| chunk.iter().map(move |v| *v as f32 * scale))
            .collect();
        let dtype = match precision {
            Precision::F16 => FloatDType::F16,
            Precision::F32 => FloatDType::F32,
        };
        Ok(PackedWeight::Dequantized(
            Tensor::<B, 2>::from_data(TensorData::new(dequantized, shape), device).cast(dtype),
        ))
    }
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
            "quant.weight_scale",
            [64, 256],
            LoadMode::Dequantize,
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
    fn fused_int4_matmul_matches_reference() {
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
            "quant.weight_scale",
            [64, 256],
            LoadMode::Packed,
            Precision::F16,
            &device,
        )
        .unwrap();
        assert!(
            matches!(packed, PackedWeight::Quantized(_)),
            "expected packed representation on GPU"
        );
        // Resident int4 payload (without scales) is 1/4 of bf16 values.
        assert_eq!(packed.resident_bytes(), 64 * 256 / 2);
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
