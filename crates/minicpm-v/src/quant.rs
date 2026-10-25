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

/// How a backend holds MiniCPM-V int4 expert weights after checkpoint load.
///
/// A raw Cube backend keeps the packed tensor and dequantizes inside matmul.
/// NdArray and a Fusion wrapper dequantize once at load. Callers of
/// `TextModel::load` do not choose.
pub trait ExpertResidence {
    fn load_mode() -> LoadMode;
}

#[cfg(any(feature = "cpu", feature = "metal", feature = "webgpu"))]
impl ExpertResidence for burn::backend::NdArray {
    fn load_mode() -> LoadMode {
        LoadMode::Dequantize
    }
}

#[cfg(any(feature = "metal", feature = "webgpu"))]
mod cube_residence {
    use super::{ExpertResidence, LoadMode};
    use burn::backend::wgpu::{CubeBackend, WgpuRuntime};
    use burn_fusion::Fusion;

    impl ExpertResidence for CubeBackend<WgpuRuntime, f32, i32, u8> {
        fn load_mode() -> LoadMode {
            LoadMode::Packed
        }
    }

    impl ExpertResidence for CubeBackend<WgpuRuntime, f32, i32, u32> {
        fn load_mode() -> LoadMode {
            LoadMode::Packed
        }
    }

    impl ExpertResidence for Fusion<CubeBackend<WgpuRuntime, f32, i32, u8>> {
        fn load_mode() -> LoadMode {
            LoadMode::Dequantize
        }
    }

    impl ExpertResidence for Fusion<CubeBackend<WgpuRuntime, f32, i32, u32>> {
        fn load_mode() -> LoadMode {
            LoadMode::Dequantize
        }
    }
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

/// Raw payload of a concatenated expert tensor: u32 words (eight
/// two's-complement nibbles each, low nibble first) plus one f32 scale per
/// 128-value group, laid out in the order the concatenation produces. Kept
/// raw because a big concatenated QFloat tensor does not survive the quantized
/// matmul path with the same fidelity as the per-expert tensors; the grouped
/// kernels read these buffers linearly.
pub struct PackedParts {
    pub words: Vec<u32>,
    pub scales: Vec<f32>,
}

/// Pack `count` same-shaped checkpoint tensors `[out_e, in]` as one logical
/// `[count * out_e, in]` payload, **k-major**: word `(k_word, row)` at index
/// `k_word * rows + row`. The grouped kernels assign consecutive threads to
/// consecutive rows, so this order makes their weight reads contiguous
/// (coalesced); a row-major layout puts them a full row apart. Scales stay
/// row-major `[row, in / group]`.
pub fn packed_vconcat_kmajor_parts<B: Backend>(
    w: &impl WeightSource<B>,
    names: &[String],
    scale_names: &[String],
    per_expert: [usize; 2],
) -> Result<PackedParts> {
    let [out_e, in_dim] = per_expert;
    let group = GROUP_SIZE;
    let experts = names.len();
    let rows = experts * out_e;
    let mut per_expert_values = Vec::with_capacity(experts);
    let mut scales = Vec::with_capacity(rows * in_dim.div_ceil(group));
    for (name, scale_name) in names.iter().zip(scale_names) {
        per_expert_values.push(unpack_int4(&w.raw_bytes(name)?));
        scales.extend(bf16_scale_bytes(
            &w.raw_bytes(scale_name)?,
            out_e * in_dim.div_ceil(group),
        ));
    }
    let mut words = Vec::with_capacity(in_dim / 8 * rows);
    for k_word in 0..in_dim / 8 {
        for expert in 0..experts {
            let values = &per_expert_values[expert];
            for row in 0..out_e {
                let start = row * in_dim + k_word * 8;
                words.push(pack8(&values[start..start + 8]));
            }
        }
    }
    Ok(PackedParts { words, scales })
}

/// Pack `count` same-shaped checkpoint tensors `[out, in_e]` as one logical
/// `[out, count * in_e]` payload, **k-major**: word `(col_word, row)` at index
/// `col_word * out + row`, where `col_word` covers eight global input columns
/// (expert-major, groups intact — `in_e` must be a group multiple). The down
/// kernel's consecutive threads take consecutive output rows, so this order
/// coalesces their reads. Scales stay row-major `[out, count * in_e / group]`.
pub fn packed_hconcat_kmajor_parts<B: Backend>(
    w: &impl WeightSource<B>,
    names: &[String],
    scale_names: &[String],
    per_expert: [usize; 2],
) -> Result<PackedParts> {
    let [out_dim, in_e] = per_expert;
    let group = GROUP_SIZE;
    assert!(
        in_e.is_multiple_of(group),
        "horizontal concat needs group-multiple expert widths"
    );
    let experts = names.len();
    let mut per_expert_values = Vec::with_capacity(experts);
    for name in names {
        per_expert_values.push(unpack_int4(&w.raw_bytes(name)?));
    }
    let mut per_expert_scales = Vec::with_capacity(experts);
    let mut scales = Vec::with_capacity(out_dim * experts * in_e.div_ceil(group));
    for (_name, scale_name) in names.iter().zip(scale_names) {
        per_expert_scales.push(bf16_scale_bytes(
            &w.raw_bytes(scale_name)?,
            out_dim * in_e.div_ceil(group),
        ));
    }
    for row in 0..out_dim {
        for expert in 0..experts {
            let scale_start = row * in_e.div_ceil(group);
            let scale_end = scale_start + in_e.div_ceil(group);
            scales.extend_from_slice(&per_expert_scales[expert][scale_start..scale_end]);
        }
    }
    let col_words = experts * in_e / 8;
    let mut words = Vec::with_capacity(col_words * out_dim);
    for col_word in 0..col_words {
        let expert = (col_word * 8) / in_e;
        let local = (col_word * 8) % in_e;
        let values = &per_expert_values[expert];
        for row in 0..out_dim {
            let start = row * in_e + local;
            words.push(pack8(&values[start..start + 8]));
        }
    }
    Ok(PackedParts { words, scales })
}

fn pack8(chunk: &[i8]) -> u32 {
    let mut packed = 0u32;
    for (nibble, value) in chunk.iter().enumerate() {
        packed |= (*value as u32 & 0xF) << (nibble * 4);
    }
    packed
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
    fn ndarray_checkpoint_keeps_experts_dequantized() {
        let bytes = fixture();
        let weights = omni_r2t2::weights::Weights::from_bytes(vec![bytes]).unwrap();
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let packed = load_packed::<NdArray>(
            &weights,
            "quant.weight",
            "quant.weight_scale",
            [64, 256],
            <NdArray as ExpertResidence>::load_mode(),
            Precision::F32,
            &device,
        )
        .unwrap();
        assert!(
            matches!(packed, PackedWeight::Dequantized(_)),
            "NdArray keeps expert weights dequantized at load"
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
    fn raw_cube_checkpoint_keeps_experts_packed() {
        use burn::backend::wgpu::{CubeBackend, WgpuRuntime};
        type CubeU8 = CubeBackend<WgpuRuntime, f32, i32, u8>;
        type CubeU32 = CubeBackend<WgpuRuntime, f32, i32, u32>;
        let bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/pack-quantized-int4.safetensors"),
        )
        .unwrap();
        let weights = omni_r2t2::weights::Weights::from_bytes(vec![bytes]).unwrap();
        let device = WgpuDevice::default();
        let packed = load_packed::<CubeU32>(
            &weights,
            "quant.weight",
            "quant.weight_scale",
            [64, 256],
            <CubeU32 as ExpertResidence>::load_mode(),
            Precision::F16,
            &device,
        )
        .unwrap();
        assert!(
            matches!(packed, PackedWeight::Quantized(_)),
            "raw Cube keeps int4 resident"
        );
        assert!(<CubeU8 as ExpertResidence>::load_mode() == LoadMode::Packed);
    }

    #[test]
    fn fusion_checkpoint_dequantizes_experts() {
        use burn::backend::wgpu::{CubeBackend, WgpuRuntime};
        use burn_fusion::Fusion;
        type FusedU8 = Fusion<CubeBackend<WgpuRuntime, f32, i32, u8>>;
        type FusedU32 = Fusion<CubeBackend<WgpuRuntime, f32, i32, u32>>;
        let bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/pack-quantized-int4.safetensors"),
        )
        .unwrap();
        let weights = omni_r2t2::weights::Weights::from_bytes(vec![bytes]).unwrap();
        let device = Default::default();
        let packed = load_packed::<FusedU8>(
            &weights,
            "quant.weight",
            "quant.weight_scale",
            [64, 256],
            <FusedU8 as ExpertResidence>::load_mode(),
            Precision::F32,
            &device,
        )
        .unwrap();
        assert!(
            matches!(packed, PackedWeight::Dequantized(_)),
            "Fusion dequantizes expert weights at load"
        );
        assert!(<FusedU32 as ExpertResidence>::load_mode() == LoadMode::Dequantize);
    }

    /// The MoE expert shapes and the single-token decode GEMV: the fused
    /// (packed-resident) matmul must survive cold autotune and stay exact.
    /// Regression for the vendored burn-cubecl tune fix that excludes the
    /// plane-parallel/unit-perpendicular GEMV candidates for quantized
    /// operands (their kernel expansion panics otherwise).
    #[test]
    fn fused_int4_matmul_covers_expert_shapes_and_gemv() {
        type B = Metal;
        let device = WgpuDevice::default();
        // as_chunks would need a newer MSRV than the workspace
        #[allow(clippy::chunks_exact_to_as_chunks)]
        let synth = |out: usize, input: usize| -> (Vec<u8>, Vec<f32>) {
            let group = GROUP_SIZE;
            let header = format!(
                concat!(
                    "{{\"quant.weight_packed\":{{\"dtype\":\"U8\",\"shape\":[{},{}],\"data_offsets\":[0,{}]}},",
                    "\"quant.weight_scale\":{{\"dtype\":\"BF16\",\"shape\":[{},{}],\"data_offsets\":[{},{}]}}}}"
                ),
                out,
                input / 2,
                out * input / 2,
                out,
                input / group,
                out * input / 2,
                out * input / 2 + out * input / group * 2
            );
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(header.as_bytes().as_ref());
            let mut weight = vec![0f32; out * input];
            let mut scales = vec![0f32; out * (input / group)];
            for r in 0..out {
                for g in 0..input / group {
                    let scale = 0.008 + ((r * 31 + g * 7) % 17) as f32 * 0.0007;
                    let stored = half::bf16::from_f32(scale).to_f32();
                    bytes.extend_from_slice(&half::bf16::from_f32(scale).to_bits().to_le_bytes());
                    scales[r * (input / group) + g] = stored;
                    for k in 0..group {
                        let c = g * group + k;
                        let q = (((r * 13 + c * 5) % 15) as i32 - 7) as f32;
                        weight[r * input + c] = q * stored;
                    }
                }
            }
            for r in 0..out {
                for c in 0..input / 2 {
                    let nibble = |index: usize| -> u8 {
                        let g = index / group;
                        ((weight[r * input + index] / scales[r * (input / group) + g]).round()
                            + 8.0) as u8
                    };
                    bytes.push((nibble(c * 2) & 0xF) | ((nibble(c * 2 + 1) & 0xF) << 4));
                }
            }
            (bytes, weight)
        };
        for (out, input, m) in [
            (512usize, 2048usize, 21usize),
            (2048, 512, 21),
            (512, 2048, 1),
        ] {
            let (bytes, weight) = synth(out, input);
            let weights = omni_r2t2::weights::Weights::from_bytes(vec![bytes]).unwrap();
            let packed = load_packed::<B>(
                &weights,
                "quant.weight_packed",
                "quant.weight_scale",
                [out, input],
                LoadMode::Packed,
                omni_r2t2::Precision::F16,
                &device,
            )
            .unwrap();
            let x_values: Vec<f32> = (0..21 * input)
                .map(|i| ((i % 29) as f32 - 14.0) * 0.05)
                .collect();
            let x = Tensor::<Metal, 2>::from_data(
                TensorData::new(x_values.clone(), [21, input]),
                &device,
            )
            .cast(FloatDType::F16)
            .slice([0..m, 0..input]);
            let got = packed
                .matmul_transpose(x)
                .into_data()
                .convert::<f32>()
                .as_slice::<f32>()
                .unwrap()
                .to_vec();
            let mut worst = 0f64;
            for i in 0..m {
                for j in 0..out {
                    let mut acc = 0f64;
                    for k in 0..input {
                        acc += x_values[i * input + k] as f64 * weight[j * input + k] as f64;
                    }
                    worst = worst.max((got[i * out + j] as f64 - acc).abs());
                }
            }
            // F16 activations with in-kernel dequant; exact to well under one
            // activation ULP step at these magnitudes.
            assert!(worst < 5e-2, "shape {out}x{input} m={m}: worst {worst}");
        }
    }

    #[test]
    fn dense_moe_matmul_timing_probe() {
        use std::time::Instant;
        type B = Metal;
        let device = WgpuDevice::default();
        let (experts, inter, hidden, m) = (256usize, 512usize, 2048usize, 16usize);
        let group = GROUP_SIZE;
        let synth = |out_dim: usize, in_dim: usize| -> Tensor<B, 2> {
            let scale = 0.01f32;
            let mut bytes = Vec::with_capacity(out_dim * in_dim / 2 + out_dim * in_dim / group * 4);
            for chunk in 0..out_dim * in_dim / 8 {
                bytes.extend_from_slice(&(0x12121212u32 + chunk as u32).to_le_bytes());
            }
            for _ in 0..out_dim * in_dim / group {
                bytes.extend_from_slice(&scale.to_le_bytes());
            }
            let data = TensorData::from_bytes_vec(
                bytes,
                [out_dim, in_dim],
                burn::tensor::DType::QFloat(scheme(group)),
            );
            Tensor::<B, 2>::from_data(data, &device)
        };
        let x = Tensor::<B, 2>::zeros([m, hidden], &device).cast(FloatDType::F16);
        let gate = synth(experts * inter, hidden);
        let down = synth(hidden, experts * inter);
        let act = Tensor::<B, 2>::zeros([m, experts * inter], &device).cast(FloatDType::F16);
        let run = |label: &str, mut f: Box<dyn FnMut() -> Tensor<B, 2>>| {
            let out = f(); // warm autotune
            let _ = out.into_data().convert::<f32>().as_slice::<f32>().unwrap()[0];
            let started = Instant::now();
            for _ in 0..10 {
                let _ = f();
            }
            let last = f();
            let values = last.into_data();
            let _ = values.convert::<f32>().as_slice::<f32>().unwrap()[0];
            eprintln!(
                "{label}: {:.2} ms/call",
                started.elapsed().as_secs_f64() * 1000.0 / 10.0
            );
        };
        run(
            "gate/up rhs-transposed [16,2048]x[131072,2048]^T",
            Box::new(|| x.clone().matmul(gate.clone().transpose())),
        );
        run(
            "down rhs-transposed [16,131072]x[2048,131072]^T",
            Box::new(|| act.clone().matmul(down.clone().transpose())),
        );
        let xt = x.clone().transpose();
        run(
            "gate/up lhs-quant [131072,2048]x[2048,16]",
            Box::new(|| gate.clone().matmul(xt.clone())),
        );
        let actt = act.clone().transpose();
        run(
            "down lhs-quant [2048,131072]x[131072,16]",
            Box::new(|| down.clone().matmul(actt.clone())),
        );
        let float_weight =
            Tensor::<B, 2>::zeros([experts * inter, hidden], &device).cast(FloatDType::F16);
        run(
            "float reference [131072,2048]x[2048,16]",
            Box::new(|| float_weight.clone().matmul(xt.clone())),
        );
        let one_row = x.clone().slice([0..1, 0..hidden]);
        run(
            "packed m=1 [1,2048]x[131072,2048]^T",
            Box::new(|| one_row.clone().matmul(gate.clone().transpose())),
        );
        {
            let _tile = burn_cubecl::kernel::matmul::DecodeTileRows::enter();
            run(
                "packed decode-tile m=16 [16,2048]x[131072,2048]^T",
                Box::new(|| x.clone().matmul(gate.clone().transpose())),
            );
        }
    }

    /// The concat loaders must reproduce exactly what the per-expert loader
    /// produces, element for element, on the CPU reference path.
    #[test]
    fn concat_loaders_match_per_expert_loader() {
        type B = burn::backend::NdArray;
        let (experts, inter, hidden, group) = (3usize, 128usize, 256usize, GROUP_SIZE);
        // The checkpoint stores bf16 scales; the loaders round-trip them, so
        // the expected values must use the rounded scale.
        let scale = |seed: usize, row: usize, g: usize| {
            half::bf16::from_f32(0.006 + ((seed + row * 31 + g * 7) % 19) as f32 * 0.0009).to_f32()
        };
        // Quantized integer stays in i32: `% 15 - 7` on usize underflows for
        // the negative half.
        let quantized =
            |seed: usize, r: usize, c: usize| -> i32 { ((seed + r * 13 + c * 5) % 15) as i32 - 7 };
        let value = |seed: usize, r: usize, c: usize| {
            quantized(seed, r, c) as f32 * scale(seed, r, c / group)
        };
        let pack = |seed: usize, rows: usize, cols: usize| -> Vec<u8> {
            let mut packed = Vec::new();
            for r in 0..rows {
                for c in 0..cols / 2 {
                    let nibble = |index: usize| {
                        (((value(seed, r, index) / scale(seed, r, index / group)) as f64).round()
                            as i32
                            + 8) as u8
                    };
                    packed.push((nibble(c * 2) & 0xF) | ((nibble(c * 2 + 1) & 0xF) << 4));
                }
            }
            packed
        };
        let pack_scales = |seed: usize, rows: usize, cols: usize| -> Vec<u8> {
            let mut scales = Vec::new();
            for r in 0..rows {
                for g in 0..cols / group {
                    scales.extend_from_slice(
                        &half::bf16::from_f32(scale(seed, r, g))
                            .to_bits()
                            .to_le_bytes(),
                    );
                }
            }
            scales
        };
        // gate/up: [inter, hidden] per expert; down: [hidden, inter].
        let mut entries: Vec<(String, String, [usize; 2], Vec<u8>)> = Vec::new();
        for e in 0..experts {
            for projection in ["gate", "up"] {
                entries.push((
                    format!("e{e}.{projection}.weight_packed"),
                    "U8".into(),
                    [inter, hidden / 2],
                    pack(e * 3, inter, hidden),
                ));
                entries.push((
                    format!("e{e}.{projection}.weight_scale"),
                    "BF16".into(),
                    [inter, hidden / group],
                    pack_scales(e * 3, inter, hidden),
                ));
            }
            entries.push((
                format!("e{e}.down.weight_packed"),
                "U8".into(),
                [hidden, inter / 2],
                pack(e * 5, hidden, inter),
            ));
            entries.push((
                format!("e{e}.down.weight_scale"),
                "BF16".into(),
                [hidden, inter / group],
                pack_scales(e * 5, hidden, inter),
            ));
        }
        let mut header = String::from("{");
        let mut offset = 0usize;
        for (name, dtype, shape, payload) in &entries {
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{},{}],\"data_offsets\":[{offset},{}]}},",
                shape[0],
                shape[1],
                offset + payload.len()
            ));
            offset += payload.len();
        }
        header.pop();
        header.push('}');
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        for (_, _, _, payload) in &entries {
            bytes.extend_from_slice(payload);
        }
        let weights = omni_r2t2::weights::Weights::from_bytes(vec![bytes]).unwrap();

        // Rebuild the logical row-major matrix from the k-major words:
        // word w covers (k_word, row); its nibble n is column k_word*8+n of
        // that row, scaled by scales[row][k/128].
        let dequantize = |parts: &PackedParts, rows: usize, cols: usize| -> Vec<f32> {
            let mut matrix = vec![0f32; rows * cols];
            for (index, word) in parts.words.iter().enumerate() {
                let k_word = index / rows;
                let row = index % rows;
                for nibble in 0..8 {
                    let k = k_word * 8 + nibble;
                    let raw = ((word >> (nibble * 4)) & 15) as i32;
                    let signed = raw - if raw >= 8 { 16 } else { 0 };
                    matrix[row * cols + k] =
                        signed as f32 * parts.scales[row * (cols / group) + k / group];
                }
            }
            matrix
        };
        let compare =
            |label: &str, concat: Vec<f32>, expected: &dyn Fn(usize, usize, usize) -> f32| {
                // concat rows are the concatenated logical matrix; expected maps
                // (expert, row, col).
                let [rows, cols] = if label == "down" {
                    [hidden, experts * inter]
                } else {
                    [experts * inter, hidden]
                };
                for row in 0..rows {
                    for col in 0..cols {
                        let at = row * cols + col;
                        if label == "down" {
                            let e = col / inter;
                            assert_eq!(
                                concat[at],
                                expected(e, row, col % inter),
                                "{label} expert {e} row {row} col {}",
                                col % inter
                            );
                        } else {
                            let e = row / inter;
                            assert_eq!(
                                concat[at],
                                expected(e, row % inter, col),
                                "{label} expert {e} row {} col {col}",
                                row % inter
                            );
                        }
                    }
                }
            };
        let gate_names: Vec<String> = (0..experts)
            .map(|e| format!("e{e}.gate.weight_packed"))
            .collect();
        let gate_scales: Vec<String> = (0..experts)
            .map(|e| format!("e{e}.gate.weight_scale"))
            .collect();
        let parts =
            packed_vconcat_kmajor_parts::<B>(&weights, &gate_names, &gate_scales, [inter, hidden])
                .unwrap();
        let gate = dequantize(&parts, experts * inter, hidden);
        compare("gate", gate, &|e, r, c| value(e * 3, r, c));
        let down_names: Vec<String> = (0..experts)
            .map(|e| format!("e{e}.down.weight_packed"))
            .collect();
        let down_scales: Vec<String> = (0..experts)
            .map(|e| format!("e{e}.down.weight_scale"))
            .collect();
        let parts =
            packed_hconcat_kmajor_parts::<B>(&weights, &down_names, &down_scales, [hidden, inter])
                .unwrap();
        let down = dequantize(&parts, hidden, experts * inter);
        compare("down", down, &|e, r, c| value(e * 5, r, c));
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
