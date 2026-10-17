//! Row-scaled int8 matrix multiply.
//!
//! Activations are quantized per row. Weights stay packed as four little-endian
//! int8 codes per `i32`. The dot product accumulates in `i32`, then each output
//! is scaled in `f32`. On Apple10 the CubeCL kernel lowers to a 64×64 int8
//! `matmul2d`, the measured faster path. Other devices, and
//! `OMNI_CONVROT_TENSOROPS=0`, keep the scalar CubeCL kernel. Weights stay
//! packed and are not expanded to FP16.
use crate::{Metal, Raw, RawTensor, Runtime};
use burn::tensor::{
    DType, FloatDType, Shape, Tensor as BurnTensor, TensorMetadata, TensorPrimitive,
    backend::Backend,
};
use burn_fusion::{
    FusionRuntime, FusionTensor,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use cubecl::{CubeCount, CubeDim, prelude::*, wgpu::WgpuRuntime};
use std::any::Any;

/// One cube per row, 256 threads. A 4096-wide denoiser row is not scanned by one thread.
#[cube(launch_unchecked)]
fn quantize_rows(
    x: &Tensor<f32>,
    packed: &mut Tensor<i32>,
    scale: &mut Tensor<f32>,
    sum: &mut Tensor<i32>,
    words: u32,
    #[comptime] write_sum: bool,
) {
    let mut peaks = SharedMemory::<f32>::new(256usize);
    let mut totals = SharedMemory::<i32>::new(256usize);
    let row = CUBE_POS_X as usize;
    let tx = UNIT_POS_X as usize;
    let count = words as usize;
    let width = count * 4;
    let stride0 = x.stride(0) as usize;
    let stride1 = x.stride(1) as usize;
    let mut peak = 0.0f32;
    let steps = (width + 255usize) / 256usize;
    let word_steps = (count + 255usize) / 256usize;
    for step in 0..steps {
        let column = tx + step * 256usize;
        if column < width {
            let value = x[row * stride0 + column * stride1];
            let magnitude = if value < 0.0 { -value } else { value };
            if magnitude > peak {
                peak = magnitude;
            }
        }
    }
    peaks[tx] = peak;
    sync_cube();
    if tx == 0usize {
        let mut maxp = 0.0f32;
        for i in 0..256usize {
            let value = peaks[i];
            if value > maxp {
                maxp = value;
            }
        }
        let floor = f32::new(1.0e-30_f32);
        let row_scale = if maxp / 127.0 > floor {
            maxp / 127.0
        } else {
            floor
        };
        scale[row] = row_scale;
        peaks[0] = row_scale;
    }
    sync_cube();
    let row_scale = peaks[0];
    let mut local = 0i32;
    for step in 0..word_steps {
        let word = tx + step * 256usize;
        if word < count {
            let base = row * stride0 + word * 4 * stride1;
            let q0 = quantize_i8(x[base] / row_scale);
            let q1 = quantize_i8(x[base + stride1] / row_scale);
            let q2 = quantize_i8(x[base + stride1 * 2] / row_scale);
            let q3 = quantize_i8(x[base + stride1 * 3] / row_scale);
            let packed_word =
                (q0 & 255) | ((q1 & 255) << 8) | ((q2 & 255) << 16) | ((q3 & 255) << 24);
            packed[row * count + word] = packed_word;
            local += sum4(packed_word);
        }
    }
    if comptime!(write_sum) {
        totals[tx] = local;
        sync_cube();
        if tx == 0usize {
            let mut total = 0i32;
            for i in 0..256usize {
                total += totals[i];
            }
            sum[row] = total;
        }
    }
}

#[cube]
fn quantize_i8(value: f32) -> i32 {
    // Half away from zero, matching `round_half_away` on the host.
    let magnitude = if value >= 0.0 { value } else { -value };
    let rounded = f32::floor(magnitude + 0.5);
    let signed = if value >= 0.0 { rounded } else { -rounded };
    let hi = f32::new(127.0_f32);
    let lo = f32::new(-128.0_f32);
    let clamped = if signed > hi {
        hi
    } else if signed < lo {
        lo
    } else {
        signed
    };
    i32::cast_from(clamped)
}

/// Sign-extend one little-endian byte. `shift` is 0, 8, 16, or 24.
#[cube]
fn sext(word: i32, shift: i32) -> i32 {
    let byte = (word >> shift) & 255;
    byte - ((byte >> 7) * 256)
}

#[cube]
fn dot4(x_word: i32, w_word: i32) -> i32 {
    let x0 = sext(x_word, 0);
    let x1 = sext(x_word, 8);
    let x2 = sext(x_word, 16);
    let x3 = sext(x_word, 24);
    let w0 = sext(w_word, 0);
    let w1 = sext(w_word, 8);
    let w2 = sext(w_word, 16);
    let w3 = sext(w_word, 24);
    x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3
}

#[cube]
fn sum4(word: i32) -> i32 {
    sext(word, 0) + sext(word, 8) + sext(word, 16) + sext(word, 24)
}

#[cube]
fn store(
    out: &mut Tensor<f32>,
    row: usize,
    column: usize,
    acc: i32,
    sum: i32,
    row_scale: f32,
    w_scale: &Tensor<f32>,
    w_zero: &Tensor<i32>,
) {
    let centered = acc - sum * w_zero[column];
    out[row * out.stride(0) + column * out.stride(1)] =
        f32::cast_from(centered) * row_scale * w_scale[column];
}

/// 64×64 device-tensor int8 `matmul2d`. `MatrixIdent::Accumulator` selects that lowering.
#[cube]
#[allow(unused_variables)]
fn mma_i8_64(
    x: &Tensor<i32>,
    w: &Tensor<i32>,
    y: &mut Tensor<i32>,
    #[comptime] rows: u32,
    #[comptime] cols: u32,
    #[comptime] inner: u32,
) {
    intrinsic!(|scope| {
        use cubecl::ir::{CoopMma, Instruction, Matrix, MatrixIdent, MatrixLayout};
        scope.register(Instruction::new(
            CoopMma::ExecuteManual {
                matrix: Matrix {
                    ident: MatrixIdent::Accumulator,
                    m: rows as usize,
                    n: cols as usize,
                    k: inner as usize,
                    storage: i32::as_type(scope).storage_type(),
                    layout: MatrixLayout::RowMajor,
                },
                registers_a: *x.expand,
                registers_b: *w.expand,
                registers_c: *y.expand,
            },
            *y.expand,
        ));
    });
}

#[cube(launch_unchecked)]
fn int8_dot_tile64(
    x_packed: &Tensor<i32>,
    w_packed: &Tensor<i32>,
    acc: &mut Tensor<i32>,
    x_scale: &Tensor<f32>,
    x_sum: &Tensor<i32>,
    w_scale: &Tensor<f32>,
    w_zero: &Tensor<i32>,
    out: &mut Tensor<f32>,
    #[comptime] rows: u32,
    #[comptime] cols: u32,
    #[comptime] inner: u32,
) {
    mma_i8_64(x_packed, w_packed, acc, rows, cols, inner);
    let rows = rows as usize;
    let cols = cols as usize;
    let tile_row = (CUBE_POS_Y as usize) * 64usize;
    let tile_col = (CUBE_POS_X as usize) * 64usize;
    let local = UNIT_POS_X as usize;
    for step in 0..32usize {
        let offset = local + step * 128usize;
        let row = tile_row + offset / 64usize;
        let col = tile_col + offset % 64usize;
        if row < rows && col < cols {
            store(
                out,
                row,
                col,
                acc[row * cols + col],
                x_sum[row],
                x_scale[row],
                w_scale,
                w_zero,
            );
        }
    }
}

/// Eight columns per unit, two partial sums per column. The extra accumulators
/// keep the int8 dot pipeline full on the 4096-wide denoiser projections.
#[cube(launch_unchecked)]
fn int8_dot(
    x_packed: &Tensor<i32>,
    x_scale: &Tensor<f32>,
    w_packed: &Tensor<i32>,
    w_scale: &Tensor<f32>,
    w_zero: &Tensor<i32>,
    out: &mut Tensor<f32>,
    words: u32,
) {
    let row = (CUBE_POS_Y as usize) * 16usize + UNIT_POS_Y as usize;
    let col0 = ((CUBE_POS_X as usize) * 16usize + UNIT_POS_X as usize) * 8usize;
    let count = words as usize;
    let columns = w_scale.shape(0);
    if row < x_scale.shape(0) && col0 + 7usize < columns {
        let mut a0 = 0i32;
        let mut a1 = 0i32;
        let mut a2 = 0i32;
        let mut a3 = 0i32;
        let mut a4 = 0i32;
        let mut a5 = 0i32;
        let mut a6 = 0i32;
        let mut a7 = 0i32;
        let mut b0 = 0i32;
        let mut b1 = 0i32;
        let mut b2 = 0i32;
        let mut b3 = 0i32;
        let mut b4 = 0i32;
        let mut b5 = 0i32;
        let mut b6 = 0i32;
        let mut b7 = 0i32;
        let mut sum = 0i32;
        let pairs = count / 2usize;
        for pair in 0..pairs {
            let word = pair * 2usize;
            let x0 = x_packed[row * count + word];
            let x1 = x_packed[row * count + word + 1usize];
            sum += sum4(x0);
            sum += sum4(x1);
            a0 += dot4(x0, w_packed[col0 * count + word]);
            b0 += dot4(x1, w_packed[col0 * count + word + 1usize]);
            a1 += dot4(x0, w_packed[(col0 + 1usize) * count + word]);
            b1 += dot4(x1, w_packed[(col0 + 1usize) * count + word + 1usize]);
            a2 += dot4(x0, w_packed[(col0 + 2usize) * count + word]);
            b2 += dot4(x1, w_packed[(col0 + 2usize) * count + word + 1usize]);
            a3 += dot4(x0, w_packed[(col0 + 3usize) * count + word]);
            b3 += dot4(x1, w_packed[(col0 + 3usize) * count + word + 1usize]);
            a4 += dot4(x0, w_packed[(col0 + 4usize) * count + word]);
            b4 += dot4(x1, w_packed[(col0 + 4usize) * count + word + 1usize]);
            a5 += dot4(x0, w_packed[(col0 + 5usize) * count + word]);
            b5 += dot4(x1, w_packed[(col0 + 5usize) * count + word + 1usize]);
            a6 += dot4(x0, w_packed[(col0 + 6usize) * count + word]);
            b6 += dot4(x1, w_packed[(col0 + 6usize) * count + word + 1usize]);
            a7 += dot4(x0, w_packed[(col0 + 7usize) * count + word]);
            b7 += dot4(x1, w_packed[(col0 + 7usize) * count + word + 1usize]);
        }
        if pairs * 2usize < count {
            let word = pairs * 2usize;
            let x0 = x_packed[row * count + word];
            sum += sum4(x0);
            a0 += dot4(x0, w_packed[col0 * count + word]);
            a1 += dot4(x0, w_packed[(col0 + 1usize) * count + word]);
            a2 += dot4(x0, w_packed[(col0 + 2usize) * count + word]);
            a3 += dot4(x0, w_packed[(col0 + 3usize) * count + word]);
            a4 += dot4(x0, w_packed[(col0 + 4usize) * count + word]);
            a5 += dot4(x0, w_packed[(col0 + 5usize) * count + word]);
            a6 += dot4(x0, w_packed[(col0 + 6usize) * count + word]);
            a7 += dot4(x0, w_packed[(col0 + 7usize) * count + word]);
        }
        a0 += b0;
        a1 += b1;
        a2 += b2;
        a3 += b3;
        a4 += b4;
        a5 += b5;
        a6 += b6;
        a7 += b7;
        let row_scale = x_scale[row];
        store(out, row, col0, a0, sum, row_scale, w_scale, w_zero);
        store(out, row, col0 + 1usize, a1, sum, row_scale, w_scale, w_zero);
        store(out, row, col0 + 2usize, a2, sum, row_scale, w_scale, w_zero);
        store(out, row, col0 + 3usize, a3, sum, row_scale, w_scale, w_zero);
        store(out, row, col0 + 4usize, a4, sum, row_scale, w_scale, w_zero);
        store(out, row, col0 + 5usize, a5, sum, row_scale, w_scale, w_zero);
        store(out, row, col0 + 6usize, a6, sum, row_scale, w_scale, w_zero);
        store(out, row, col0 + 7usize, a7, sum, row_scale, w_scale, w_zero);
    } else if row < x_scale.shape(0) && col0 < columns {
        let mut a0 = 0i32;
        let mut a1 = 0i32;
        let mut a2 = 0i32;
        let mut a3 = 0i32;
        let mut a4 = 0i32;
        let mut a5 = 0i32;
        let mut a6 = 0i32;
        let mut sum = 0i32;
        for word in 0..count {
            let x_word = x_packed[row * count + word];
            sum += sum4(x_word);
            a0 += dot4(x_word, w_packed[col0 * count + word]);
            if col0 + 1usize < columns {
                a1 += dot4(x_word, w_packed[(col0 + 1usize) * count + word]);
            }
            if col0 + 2usize < columns {
                a2 += dot4(x_word, w_packed[(col0 + 2usize) * count + word]);
            }
            if col0 + 3usize < columns {
                a3 += dot4(x_word, w_packed[(col0 + 3usize) * count + word]);
            }
            if col0 + 4usize < columns {
                a4 += dot4(x_word, w_packed[(col0 + 4usize) * count + word]);
            }
            if col0 + 5usize < columns {
                a5 += dot4(x_word, w_packed[(col0 + 5usize) * count + word]);
            }
            if col0 + 6usize < columns {
                a6 += dot4(x_word, w_packed[(col0 + 6usize) * count + word]);
            }
        }
        let row_scale = x_scale[row];
        store(out, row, col0, a0, sum, row_scale, w_scale, w_zero);
        if col0 + 1usize < columns {
            store(out, row, col0 + 1usize, a1, sum, row_scale, w_scale, w_zero);
        }
        if col0 + 2usize < columns {
            store(out, row, col0 + 2usize, a2, sum, row_scale, w_scale, w_zero);
        }
        if col0 + 3usize < columns {
            store(out, row, col0 + 3usize, a3, sum, row_scale, w_scale, w_zero);
        }
        if col0 + 4usize < columns {
            store(out, row, col0 + 4usize, a4, sum, row_scale, w_scale, w_zero);
        }
        if col0 + 5usize < columns {
            store(out, row, col0 + 5usize, a5, sum, row_scale, w_scale, w_zero);
        }
        if col0 + 6usize < columns {
            store(out, row, col0 + 6usize, a6, sum, row_scale, w_scale, w_zero);
        }
    }
}

fn empty_tensor(like: &RawTensor, shape: Shape, dtype: DType) -> RawTensor {
    let bytes = shape.iter().product::<usize>() * dtype.size();
    RawTensor::new_contiguous(
        like.client.clone(),
        like.device.clone(),
        shape,
        like.client.empty(bytes),
        dtype,
    )
}

fn launch(x: RawTensor, w_packed: RawTensor, w_scale: RawTensor, w_zero: RawTensor) -> RawTensor {
    let tile64 = {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            crate::int8_tensorops::enabled(&x)
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            false
        }
    };
    let x_shape = x.shape();
    let rows = x_shape[0];
    let width = x_shape[1];
    let columns = w_scale.shape()[0];
    let words = width / 4;
    let client = x.client.clone();
    let x_packed = empty_tensor(&x, Shape::new([rows * words]), DType::I32);
    let x_scale = empty_tensor(&x, Shape::new([rows]), DType::F32);
    let x_sum = empty_tensor(&x, Shape::new([rows]), DType::I32);
    let out = empty_tensor(&x, Shape::new([rows, columns]), DType::F32);
    let w_packed = burn_cubecl::kernel::into_contiguous(w_packed);
    let w_scale = burn_cubecl::kernel::into_contiguous(w_scale);
    let w_zero = burn_cubecl::kernel::into_contiguous(w_zero);
    unsafe {
        if tile64 {
            quantize_rows::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(rows as u32, 1, 1),
                CubeDim::new_1d(256),
                x.into_tensor_arg(),
                x_packed.clone().into_tensor_arg(),
                x_scale.clone().into_tensor_arg(),
                x_sum.clone().into_tensor_arg(),
                words as u32,
                true,
            );
            let acc = empty_tensor(&out, Shape::new([rows * columns]), DType::I32);
            int8_dot_tile64::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static((columns as u32).div_ceil(64), (rows as u32).div_ceil(64), 1),
                CubeDim::new_1d(128),
                x_packed.into_tensor_arg(),
                w_packed.into_tensor_arg(),
                acc.into_tensor_arg(),
                x_scale.into_tensor_arg(),
                x_sum.into_tensor_arg(),
                w_scale.into_tensor_arg(),
                w_zero.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                rows as u32,
                columns as u32,
                width as u32,
            );
            return out;
        }
        quantize_rows::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(rows as u32, 1, 1),
            CubeDim::new_1d(256),
            x.into_tensor_arg(),
            x_packed.clone().into_tensor_arg(),
            x_scale.clone().into_tensor_arg(),
            x_sum.into_tensor_arg(),
            words as u32,
            false,
        );
        int8_dot::launch_unchecked::<WgpuRuntime>(
            &out.client,
            CubeCount::Static(
                (columns as u32).div_ceil(8).div_ceil(16),
                (rows as u32).div_ceil(16),
                1,
            ),
            CubeDim::new_2d(16, 16),
            x_packed.into_tensor_arg(),
            x_scale.into_tensor_arg(),
            w_packed.into_tensor_arg(),
            w_scale.into_tensor_arg(),
            w_zero.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            words as u32,
        );
    }
    out
}

struct Int8Gemm {
    desc: CustomOpIr,
}

impl std::fmt::Debug for Int8Gemm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.desc.fmt(f)
    }
}

impl Operation<Runtime> for Int8Gemm {
    fn execute(&self, handles: &mut HandleContainer<<Runtime as FusionRuntime>::FusionHandle>) {
        let (inputs, [output]) = self.desc.as_fixed::<4, 1>();
        let x = handles.get_float_tensor::<Raw>(&inputs[0]);
        let w_packed = handles.get_int_tensor::<Raw>(&inputs[1]);
        let w_scale = handles.get_float_tensor::<Raw>(&inputs[2]);
        let w_zero = handles.get_int_tensor::<Raw>(&inputs[3]);
        handles.register_float_tensor::<Raw>(&output.id, launch(x, w_packed, w_scale, w_zero));
    }
}

fn fusion_tensor<B: Backend>(primitive: impl Any) -> Option<FusionTensor<Runtime>> {
    if std::any::TypeId::of::<B>() != std::any::TypeId::of::<Metal>() {
        return None;
    }
    let primitive: Box<dyn Any> = Box::new(primitive);
    primitive
        .downcast::<FusionTensor<Runtime>>()
        .ok()
        .map(|tensor| *tensor)
}

/// `x @ W^T` after per-row symmetric int8 quantization of `x`.
///
/// `w_packed` holds `n * (k / 4)` little-endian int8 groups. `w_scale` and
/// `w_zero` are per output row. Returns `None` when this backend is not the
/// Metal fusion runtime.
pub fn try_int8_gemm<B: Backend>(
    x: BurnTensor<B, 2>,
    w_packed: BurnTensor<B, 1, burn::tensor::Int>,
    w_scale: BurnTensor<B, 1>,
    w_zero: BurnTensor<B, 1, burn::tensor::Int>,
) -> Option<BurnTensor<B, 2>> {
    let [rows, width] = x.dims();
    let columns = w_scale.dims()[0];
    if width % 4 != 0 || w_packed.dims()[0] != columns * width / 4 || w_zero.dims()[0] != columns {
        return None;
    }
    let x = x.cast(FloatDType::F32);
    let w_scale = w_scale.cast(FloatDType::F32);
    let x = match x.into_primitive() {
        TensorPrimitive::Float(tensor) => fusion_tensor::<B>(tensor)?,
        TensorPrimitive::QFloat(_) => return None,
    };
    let w_scale = match w_scale.into_primitive() {
        TensorPrimitive::Float(tensor) => fusion_tensor::<B>(tensor)?,
        TensorPrimitive::QFloat(_) => return None,
    };
    let w_packed = fusion_tensor::<B>(w_packed.into_primitive())?;
    let w_zero = fusion_tensor::<B>(w_zero.into_primitive())?;
    let client = x.client.clone();
    let output = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([rows, columns]),
        DType::F32,
    );
    let inputs = [x, w_packed, w_scale, w_zero];
    let streams = OperationStreams::with_inputs(inputs.iter());
    let desc = CustomOpIr::new(
        "int8_scaled_matmul",
        &inputs.map(FusionTensor::into_ir),
        &[output],
    );
    let tensor = client
        .register(
            streams,
            OperationIr::Custom(desc.clone()),
            Int8Gemm { desc },
        )
        .output();
    Some(BurnTensor::from_primitive(TensorPrimitive::Float(
        fusion_out::<B>(tensor)?,
    )))
}

fn fusion_out<B: Backend>(tensor: FusionTensor<Runtime>) -> Option<B::FloatTensorPrimitive> {
    let tensor: Box<dyn Any> = Box::new(tensor);
    tensor
        .downcast::<B::FloatTensorPrimitive>()
        .ok()
        .map(|tensor| *tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    fn round_half_away(value: f32) -> i32 {
        let rounded = if value >= 0.0 {
            (value + 0.5).floor()
        } else {
            (value - 0.5).ceil()
        };
        rounded.clamp(-128.0, 127.0) as i32
    }

    /// Dequantize the same int8 weights in f32, then dot with the row-quantized
    /// activations. Checkpoint scales are f32. This is not the kernel's i32
    /// accumulator.
    fn dequantized_weight_reference(
        x: &[f32],
        rows: usize,
        width: usize,
        weights: &[i8],
        scale: &[f32],
        zero: &[i32],
    ) -> Vec<f32> {
        let columns = scale.len();
        let mut out = vec![0.0; rows * columns];
        for row in 0..rows {
            let sample = &x[row * width..(row + 1) * width];
            let peak = sample
                .iter()
                .fold(0.0f32, |peak, value| peak.max(value.abs()));
            let row_scale = (peak / 127.0).max(1.0e-30);
            let activations: Vec<f32> = sample
                .iter()
                .map(|value| round_half_away(value / row_scale) as f32 * row_scale)
                .collect();
            for column in 0..columns {
                let mut acc = 0.0f32;
                for k in 0..width {
                    let weight = (i32::from(weights[column * width + k]) - zero[column]) as f32
                        * scale[column];
                    acc += activations[k] * weight;
                }
                out[row * columns + column] = acc;
            }
        }
        out
    }

    fn run_gemm(
        device: &cubecl::wgpu::WgpuDevice,
        x: Vec<f32>,
        rows: usize,
        width: usize,
        weights: &[i8],
        scale: Vec<f32>,
        zero: Vec<i32>,
    ) -> Vec<f32> {
        let columns = scale.len();
        let packed: Vec<i32> = weights
            .chunks_exact(4)
            .map(|chunk| {
                chunk.iter().enumerate().fold(0i32, |word, (lane, code)| {
                    word | i32::from(*code as u8) << (8 * lane)
                })
            })
            .collect();
        let words = packed.len();
        let x = BurnTensor::<Metal, 2>::from_data(TensorData::new(x, [rows, width]), device);
        let w_packed = BurnTensor::<Metal, 1, burn::tensor::Int>::from_data(
            TensorData::new(packed, [words]),
            device,
        );
        let w_scale = BurnTensor::<Metal, 1>::from_data(TensorData::new(scale, [columns]), device);
        let w_zero = BurnTensor::<Metal, 1, burn::tensor::Int>::from_data(
            TensorData::new(zero, [columns]),
            device,
        );
        try_int8_gemm(x, w_packed, w_scale, w_zero)
            .expect("metal int8 gemm")
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    }

    fn assert_close(got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len());
        for (got, expected) in got.iter().zip(expected) {
            assert!((got - expected).abs() < 1.0e-3, "{got} != {expected}");
        }
    }

    fn tile_fixture() -> (Vec<f32>, Vec<i8>, Vec<f32>, Vec<i32>, Vec<f32>) {
        let rows = 64;
        let columns = 64;
        let width = 64;
        let mut x = vec![0.0; rows * width];
        x[..4].copy_from_slice(&[1.0, -2.0, 0.5, 4.0]);
        let mut weights = vec![0i8; columns * width];
        weights[..4].copy_from_slice(&[1, -2, 3, -4]);
        let scale = vec![0.25; columns];
        let mut zero = vec![0; columns];
        zero[0] = 1;
        let expected = dequantized_weight_reference(&x, rows, width, &weights, &scale, &zero);
        (x, weights, scale, zero, expected)
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn int8_gemm_matches_dequantized_weights_on_scalar_and_tile64() {
        let device = cubecl::wgpu::WgpuDevice::default();
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
            &device,
            Default::default(),
        );
        // Hand value: codes 32, -64, 16, 127 against weights 1, -2, 3, -4, zero 1.
        let x = vec![1.0, -2.0, 0.5, 4.0];
        let weights = vec![1i8, -2, 3, -4];
        let scale = vec![0.25];
        let zero = vec![1];
        let expected = dequantized_weight_reference(&x, 1, 4, &weights, &scale, &zero);
        assert!((expected[0] - (-411.0 / 127.0)).abs() < 1.0e-5);

        unsafe { std::env::set_var("OMNI_CONVROT_TENSOROPS", "0") };
        let got = run_gemm(&device, x, 1, 4, &weights, scale, zero);
        assert_close(&got, &expected);
        assert!((got[0] + 1.0 - expected[0]).abs() >= 1.0e-3);
        let (x, weights, scale, zero, expected) = tile_fixture();
        let got = run_gemm(&device, x, 64, 64, &weights, scale, zero);
        assert_close(&got, &expected);

        unsafe { std::env::remove_var("OMNI_CONVROT_TENSOROPS") };
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert!(
            crate::int8_tensorops::matmul2d_lowered(&device),
            "64×64 matmul2d lowering is off"
        );
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        panic!("64×64 matmul2d lowering is only built for Apple Silicon");
        let (x, weights, scale, zero, expected) = tile_fixture();
        assert!((expected[0] - (-411.0 / 127.0)).abs() < 1.0e-5);
        let got = run_gemm(&device, x, 64, 64, &weights, scale, zero);
        assert_close(&got, &expected);
        assert!((got[0] + 1.0 - expected[0]).abs() >= 1.0e-3);
    }
}
