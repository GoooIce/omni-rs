//! Unsloth INT8-ConvRot for the Qwen-Image-2.1-Turbo denoiser.
//!
//! Weights are stored per-row symmetric int8 in a group-256 Hadamard basis
//! (`convrot_hadamard_v1`). The forward rotates activations by that Hadamard,
//! then asks the shared int8 kernel to quantize each row and multiply. Which
//! lowering runs is decided inside that kernel.
use crate::{Error, Precision, Result};
use burn::tensor::{DType, Int, Tensor, TensorData, backend::Backend};
use omni_r2t2::weights::Weights;
use std::path::Path;

pub const GROUP: usize = 256;
const FILE_NAME: &str = "Qwen-Image-2.1-Turbo-INT8-ConvRot.safetensors";

pub struct ConvRotFile {
    weights: Weights,
}

impl ConvRotFile {
    pub fn open(root: &Path) -> Result<Option<Self>> {
        let path = root.join(FILE_NAME);
        if !path.is_file() {
            return Ok(None);
        }
        Ok(Some(Self {
            weights: Weights::open_file(&path)?,
        }))
    }

    pub fn linear<B: Backend>(
        &self,
        name: &str,
        input: usize,
        output: usize,
        hadamard: &Tensor<B, 2>,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Int8Linear<B>> {
        if !input.is_multiple_of(GROUP) || !input.is_multiple_of(4) {
            return Err(Error::Config(format!(
                "{name}: ConvRot 输入维度 {input} 不能按 group {GROUP} 旋转"
            )));
        }
        let (q_shape, codes) = self.weights.raw_i8(&format!("{name}._weight_qdata"))?;
        let (s_shape, scale) = self.weights.raw_f32(&format!("{name}._weight_scale"))?;
        let (z_shape, zero) = self.weights.raw_i8(&format!("{name}._weight_zero_point"))?;
        if q_shape != [output, input] || scale.len() != output || zero.len() != output {
            return Err(Error::Config(format!(
                "{name}: ConvRot 形状 {q_shape:?} / scale {s_shape:?} / zero {z_shape:?} 与 [{output}, {input}] 不符"
            )));
        }
        let packed = pack_i8(&codes);
        let zero: Vec<i32> = zero.into_iter().map(i32::from).collect();
        let words = packed.len();
        Ok(Int8Linear {
            input,
            output,
            packed: Tensor::<B, 1, Int>::from_data(TensorData::new(packed, [words]), device),
            scale: Tensor::<B, 1>::from_data(
                TensorData::new(scale, [output]),
                (device, DType::F32),
            ),
            zero: Tensor::<B, 1, Int>::from_data(TensorData::new(zero, [output]), device),
            hadamard: hadamard.clone(),
            precision,
        })
    }
}

pub struct Int8Linear<B: Backend> {
    input: usize,
    output: usize,
    packed: Tensor<B, 1, Int>,
    scale: Tensor<B, 1>,
    zero: Tensor<B, 1, Int>,
    hadamard: Tensor<B, 2>,
    precision: Precision,
}

impl<B: Backend> Int8Linear<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let dims = x.dims();
        let input = dims[D - 1];
        assert_eq!(input, self.input, "ConvRot 输入维度不匹配");
        let rows = dims.iter().take(D - 1).product();
        let flat = x.reshape([rows, input]);
        let rotated = rotate(flat.cast(burn::tensor::FloatDType::F32), &self.hadamard);
        let y = self.project(rotated);
        let mut out_dims = dims;
        out_dims[D - 1] = self.output;
        y.cast(self.precision.dtype()).reshape(out_dims)
    }

    fn project(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        omni_kernel::int8_matmul::try_int8_gemm(
            x,
            self.packed.clone(),
            self.scale.clone(),
            self.zero.clone(),
        )
        .expect("ConvRot int8 gemm")
    }
}

pub fn hadamard_tensor<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
    let values = hadamard(GROUP);
    Tensor::<B, 2>::from_data(
        TensorData::new(values, [GROUP, GROUP]),
        (device, DType::F32),
    )
}

fn rotate<B: Backend>(x: Tensor<B, 2>, hadamard: &Tensor<B, 2>) -> Tensor<B, 2> {
    let [rows, width] = x.dims();
    let groups = width / GROUP;
    x.reshape([rows * groups, GROUP])
        .matmul(hadamard.clone())
        .reshape([rows, width])
}

pub fn hadamard(size: usize) -> Vec<f32> {
    assert!(is_power_of_four(size), "ConvRot group 必须是 4 的幂");
    let scale = 2.0f32.powi(-((usize::BITS - size.leading_zeros() - 1) as i32 / 2));
    let mut matrix = vec![scale; size * size];
    for row in 0..size {
        for column in 0..size {
            if sign_negative(row, column, size) {
                matrix[row * size + column] = -scale;
            }
        }
    }
    matrix
}

fn is_power_of_four(size: usize) -> bool {
    size >= 4 && size.is_power_of_two() && (size.trailing_zeros() % 2 == 0)
}

/// Parity of the base-4 digits of `row ^ column` that equal 3. Matches Unsloth
/// `convrot_hadamard_v1`: `H4[a, b] = -1` iff `a ^ b == 3`, extended by Kronecker power.
fn sign_negative(row: usize, column: usize, size: usize) -> bool {
    let mut threes = (row ^ column) & ((row ^ column) >> 1) & 0x5555_5555_5555_5555;
    let mut parity = 0usize;
    for _ in 0..(usize::BITS - size.leading_zeros() - 1) / 2 {
        parity ^= threes & 1;
        threes >>= 2;
    }
    parity == 1
}

fn pack_i8(codes: &[i8]) -> Vec<i32> {
    codes
        .chunks_exact(4)
        .map(|chunk| {
            chunk.iter().enumerate().fold(0i32, |word, (lane, code)| {
                word | i32::from(*code as u8) << (8 * lane)
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "metal"))]
    use burn::backend::NdArray;
    use burn::tensor::DType;

    #[test]
    fn hadamard_is_symmetric_and_involutory() {
        let h = hadamard(GROUP);
        for row in 0..GROUP {
            for column in 0..GROUP {
                assert_eq!(h[row * GROUP + column], h[column * GROUP + row]);
            }
        }
        for row in 0..GROUP {
            for column in 0..GROUP {
                let mut dot = 0.0f32;
                for k in 0..GROUP {
                    dot += h[row * GROUP + k] * h[k * GROUP + column];
                }
                let expect = if row == column { 1.0 } else { 0.0 };
                assert!(
                    (dot - expect).abs() < 1.0e-4,
                    "H@H [{row},{column}] = {dot}"
                );
            }
        }
    }

    #[test]
    fn rotation_cancels_around_a_float_matmul() {
        let h = hadamard(4);
        let x = [1.0f32, 2.0, -1.0, 0.5, 0.25, -0.5, 1.5, -2.0];
        let w = [0.5f32, -1.0, 0.25, 2.0, 1.0, 0.0, -0.5, 0.75];
        let mut x_rot = [0.0; 8];
        let mut w_rot = [0.0; 8];
        for row in 0..2 {
            for column in 0..4 {
                let mut x_acc = 0.0;
                let mut w_acc = 0.0;
                for k in 0..4 {
                    x_acc += x[row * 4 + k] * h[k * 4 + column];
                    w_acc += w[row * 4 + k] * h[k * 4 + column];
                }
                x_rot[row * 4 + column] = x_acc;
                w_rot[row * 4 + column] = w_acc;
            }
        }
        for row in 0..2 {
            let mut direct = 0.0;
            let mut rotated = 0.0;
            for k in 0..4 {
                direct += x[0 * 4 + k] * w[row * 4 + k];
                rotated += x_rot[0 * 4 + k] * w_rot[row * 4 + k];
            }
            assert!((direct - rotated).abs() < 1.0e-4, "{direct} != {rotated}");
        }
    }

    #[test]
    #[cfg(not(feature = "metal"))]
    fn ndarray_linear_rotates_then_matches_the_int8_reference() {
        type Cpu = NdArray;
        let device = Default::default();
        let hadamard = hadamard_tensor::<Cpu>(&device);
        let input = GROUP;
        let output = 2;
        let rows = 2;
        let values: Vec<f32> = (0..rows * input)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
            .collect();
        let weights: Vec<i8> = (0..output * input)
            .map(|i| ((i * 13) % 31) as i8 - 15)
            .collect();
        let scale = vec![0.05, 0.2];
        let zero = vec![0, 2];
        let x = Tensor::<Cpu, 2>::from_data(TensorData::new(values, [rows, input]), &device);
        let linear = Int8Linear {
            input,
            output,
            packed: Tensor::<Cpu, 1, Int>::from_data(
                TensorData::new(pack_i8(&weights), [output * input / 4]),
                &device,
            ),
            scale: Tensor::<Cpu, 1>::from_data(
                TensorData::new(scale.clone(), [output]),
                (&device, DType::F32),
            ),
            zero: Tensor::<Cpu, 1, Int>::from_data(
                TensorData::new(zero.clone(), [output]),
                &device,
            ),
            hadamard: hadamard.clone(),
            precision: Precision::F32,
        };
        let rotated = rotate(x.clone(), &hadamard);
        let expected = dequantized(
            &rotated.into_data().to_vec::<f32>().unwrap(),
            input,
            &weights,
            &scale,
            &zero,
        );
        let got = linear.forward(x).into_data().to_vec::<f32>().unwrap();
        assert_eq!(got.len(), expected.len());
        for (got, expected) in got.iter().zip(&expected) {
            assert!((got - expected).abs() < 1.0e-2, "{got} != {expected}");
        }
    }

    fn dequantized(
        x: &[f32],
        width: usize,
        weights: &[i8],
        scale: &[f32],
        zero: &[i32],
    ) -> Vec<f32> {
        let rows = x.len() / width;
        let columns = scale.len();
        let mut out = vec![0.0; rows * columns];
        for row in 0..rows {
            let sample = &x[row * width..(row + 1) * width];
            let peak = sample
                .iter()
                .fold(0.0f32, |peak, value| peak.max(value.abs()));
            let row_scale = (peak / 127.0).max(1.0e-30);
            for column in 0..columns {
                let mut acc = 0.0f32;
                for k in 0..width {
                    let code = {
                        let value = sample[k] / row_scale;
                        let magnitude = value.abs();
                        let rounded = (magnitude + 0.5).floor();
                        let signed = if value >= 0.0 { rounded } else { -rounded };
                        signed.clamp(-128.0, 127.0)
                    };
                    let weight = half::f16::from_f32(
                        (i32::from(weights[column * width + k]) - zero[column]) as f32
                            * scale[column],
                    )
                    .to_f32();
                    acc += code * row_scale * weight;
                }
                out[row * columns + column] = acc;
            }
        }
        out
    }

    #[test]
    #[cfg(feature = "metal")]
    fn metal_linear_rotates_then_matches_dequantized_weights() {
        type Gpu = omni_kernel::Metal;
        let device = cubecl::wgpu::WgpuDevice::default();
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
            &device,
            Default::default(),
        );
        let hadamard = hadamard_tensor::<Gpu>(&device);
        let input = GROUP;
        let output = 2;
        let rows = 2;
        let values: Vec<f32> = (0..rows * input)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
            .collect();
        let weights: Vec<i8> = (0..output * input)
            .map(|i| ((i * 13) % 31) as i8 - 15)
            .collect();
        let scale = vec![0.05, 0.2];
        let zero = vec![0, 2];
        let x = Tensor::<Gpu, 2>::from_data(TensorData::new(values, [rows, input]), &device);
        let linear = Int8Linear {
            input,
            output,
            packed: Tensor::<Gpu, 1, Int>::from_data(
                TensorData::new(pack_i8(&weights), [output * input / 4]),
                &device,
            ),
            scale: Tensor::<Gpu, 1>::from_data(
                TensorData::new(scale.clone(), [output]),
                (&device, DType::F32),
            ),
            zero: Tensor::<Gpu, 1, Int>::from_data(
                TensorData::new(zero.clone(), [output]),
                &device,
            ),
            hadamard: hadamard.clone(),
            precision: Precision::F32,
        };
        let rotated = rotate(x.clone(), &hadamard);
        let expected = dequantized(
            &rotated.into_data().to_vec::<f32>().unwrap(),
            input,
            &weights,
            &scale,
            &zero,
        );
        let got = linear.forward(x).into_data().to_vec::<f32>().unwrap();
        assert_eq!(got.len(), expected.len());
        for (got, expected) in got.iter().zip(expected) {
            assert!((got - expected).abs() < 1.0e-2, "{got} != {expected}");
        }
    }
}
