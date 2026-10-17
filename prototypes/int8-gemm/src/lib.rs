//! CubeCL int8 gemm candidates. The production kernel stays in `omni-kernel`
//! until one of these is faster and still matches the row-scaled reference.
use burn::tensor::{
    DType, FloatDType, Shape, Tensor as BurnTensor, TensorMetadata, TensorPrimitive,
    backend::Backend,
};
use burn_cubecl::{CubeBackend, tensor::CubeTensor};
use burn_fusion::{
    Fusion, FusionBackend, FusionRuntime, FusionTensor,
    stream::{Operation, OperationStreams},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use cubecl::{CubeCount, CubeDim, prelude::*, wgpu::WgpuRuntime};
use std::any::Any;

pub type Raw = CubeBackend<WgpuRuntime, f32, i32, u8>;
pub type Metal = Fusion<Raw>;
type Runtime = <Raw as FusionBackend>::FusionRuntime;
type RawTensor = CubeTensor<WgpuRuntime>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// One output per unit. Weights are `[column, k_word]`, so a warp strides by K.
    Naive,
    /// One unit writes four consecutive columns and reuses each activation word.
    Reuse4,
    /// Weights are `[k_word, column]`. A 32-wide cube reads each K step contiguously.
    Coalesced,
    /// 16×32 shared-memory tile. Weights are `[k_word, column]`.
    Tiled,
    /// One unit writes eight consecutive columns and keeps eight independent accumulators.
    Wide,
    /// CubeCL kernel. Each SIMD group owns a 16×32 tile and calls the int8 TensorOps MMA.
    Mma,
    /// CubeCL kernel. Four SIMD groups run one 64×64 device-tensor `matmul2d`.
    Tile64,
}

/// One cube per row, 256 threads. The peak reduction fits in shared memory so a
/// 4096-wide row is not scanned by a single thread.
#[cube(launch_unchecked)]
fn quantize_rows(x: &Tensor<f32>, packed: &mut Tensor<i32>, scale: &mut Tensor<f32>, words: u32) {
    let mut peaks = SharedMemory::<f32>::new(256usize);
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
    for step in 0..word_steps {
        let word = tx + step * 256usize;
        if word < count {
            let base = row * stride0 + word * 4 * stride1;
            let q0 = quantize_i8(x[base] / row_scale);
            let q1 = quantize_i8(x[base + stride1] / row_scale);
            let q2 = quantize_i8(x[base + stride1 * 2] / row_scale);
            let q3 = quantize_i8(x[base + stride1 * 3] / row_scale);
            packed[row * count + word] =
                (q0 & 255) | ((q1 & 255) << 8) | ((q2 & 255) << 16) | ((q3 & 255) << 24);
        }
    }
}

#[cube]
fn quantize_i8(value: f32) -> i32 {
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

/// 16×32×16 int8 TensorOps MMA. Registers hold the int8 values; the compiler emits `matmul2d`.
#[cube]
#[allow(unused_variables)]
fn mma_i8(a: &Array<i32>, b: &Array<i32>, c: &mut Array<i32>) {
    intrinsic!(|scope| {
        use cubecl::ir::{CoopMma, Instruction, Matrix, MatrixIdent, MatrixLayout};
        scope.register(Instruction::new(
            CoopMma::ExecuteManual {
                matrix: Matrix {
                    ident: MatrixIdent::B,
                    m: 16,
                    n: 32,
                    k: 16,
                    storage: i32::as_type(scope).storage_type(),
                    layout: MatrixLayout::ColMajor,
                },
                registers_a: *a.expand,
                registers_b: *b.expand,
                registers_c: *c.expand,
            },
            *c.expand,
        ));
    });
}

#[cube]
fn write4(dst: &mut Array<i32>, base: usize, word: i32) {
    dst[base] = sext(word, 0);
    dst[base + 1usize] = sext(word, 8);
    dst[base + 2usize] = sext(word, 16);
    dst[base + 3usize] = sext(word, 24);
}

#[cube(launch_unchecked)]
fn int8_dot_mma(
    x_packed: &Tensor<i32>,
    x_scale: &Tensor<f32>,
    w_packed: &Tensor<i32>,
    w_scale: &Tensor<f32>,
    w_zero: &Tensor<i32>,
    out: &mut Tensor<f32>,
    words: u32,
) {
    let lane = UNIT_POS_PLANE;
    let row = (((lane >> 2) & 4) | ((lane >> 1) & 3)) as usize;
    let col = ((((lane >> 2) & 2) | (lane & 1)) * 4) as usize;
    let row0 = (CUBE_POS_Y as usize) * 16usize;
    let col0 = (CUBE_POS_X as usize) * 32usize;
    let count = words as usize;
    let rows = x_scale.shape(0);
    let columns = w_scale.shape(0);
    let width = count * 4usize;
    let m0 = row0 + row;
    let m1 = m0 + 8usize;
    let mut sum0 = 0i32;
    let mut sum1 = 0i32;
    if m0 < rows {
        for word in 0..count {
            sum0 += sum4(x_packed[m0 * count + word]);
        }
    }
    if m1 < rows {
        for word in 0..count {
            sum1 += sum4(x_packed[m1 * count + word]);
        }
    }
    let mut acc = Array::<i32>::new(16usize);
    #[unroll]
    for i in 0..16usize {
        acc[i] = 0i32;
    }
    let tiles = (width + 15usize) / 16usize;
    for tile in 0..tiles {
        let k0 = tile * 16usize;
        let mut frag_a = Array::<i32>::new(8usize);
        let mut frag_b = Array::<i32>::new(16usize);
        #[unroll]
        for r in 0..2usize {
            let am = row0 + row + r * 8usize;
            let ak = k0 + col;
            let mut aw = 0i32;
            if am < rows && ak < width {
                aw = x_packed[am * count + ak / 4usize];
            }
            write4(&mut frag_a, r * 4usize, aw);
            let n_lo = col0 + row + r * 8usize;
            let n_hi = n_lo + 16usize;
            let mut w_lo = 0i32;
            let mut w_hi = 0i32;
            if n_lo < columns && ak < width {
                w_lo = w_packed[n_lo * count + ak / 4usize];
            }
            if n_hi < columns && ak < width {
                w_hi = w_packed[n_hi * count + ak / 4usize];
            }
            write4(&mut frag_b, r * 4usize, w_lo);
            write4(&mut frag_b, 8usize + r * 4usize, w_hi);
        }
        mma_i8(&frag_a, &frag_b, &mut acc);
    }
    #[unroll]
    for t in 0..2usize {
        #[unroll]
        for r in 0..2usize {
            let m = row0 + row + r * 8usize;
            if m < rows {
                let row_sum = if r == 0usize { sum0 } else { sum1 };
                let row_scale = x_scale[m];
                #[unroll]
                for j in 0..4usize {
                    let n = col0 + t * 16usize + col + j;
                    if n < columns {
                        store(
                            out,
                            m,
                            n,
                            acc[t * 8usize + r * 4usize + j],
                            row_sum,
                            row_scale,
                            w_scale,
                            w_zero,
                        );
                    }
                }
            }
        }
    }
}

#[cube(launch_unchecked)]
fn int8_dot_naive(
    x_packed: &Tensor<i32>,
    x_scale: &Tensor<f32>,
    w_packed: &Tensor<i32>,
    w_scale: &Tensor<f32>,
    w_zero: &Tensor<i32>,
    out: &mut Tensor<f32>,
    words: u32,
) {
    let row = ABSOLUTE_POS_Y as usize;
    let column = ABSOLUTE_POS_X as usize;
    let count = words as usize;
    if row < x_scale.shape(0) && column < w_scale.shape(0) {
        let mut acc = 0i32;
        let mut sum = 0i32;
        for word in 0..count {
            let x_word = x_packed[row * count + word];
            acc += dot4(x_word, w_packed[column * count + word]);
            sum += sum4(x_word);
        }
        let centered = acc - sum * w_zero[column];
        out[row * out.stride(0) + column * out.stride(1)] =
            f32::cast_from(centered) * x_scale[row] * w_scale[column];
    }
}

#[cube(launch_unchecked)]
fn int8_dot_coalesced(
    x_packed: &Tensor<i32>,
    x_scale: &Tensor<f32>,
    w_packed: &Tensor<i32>,
    w_scale: &Tensor<f32>,
    w_zero: &Tensor<i32>,
    out: &mut Tensor<f32>,
    words: u32,
) {
    let row = ABSOLUTE_POS_Y as usize;
    let column = ABSOLUTE_POS_X as usize;
    let count = words as usize;
    if row < x_scale.shape(0) && column < w_scale.shape(0) {
        let columns = w_scale.shape(0);
        let mut acc = 0i32;
        let mut sum = 0i32;
        for word in 0..count {
            let x_word = x_packed[row * count + word];
            acc += dot4(x_word, w_packed[word * columns + column]);
            sum += sum4(x_word);
        }
        let centered = acc - sum * w_zero[column];
        out[row * out.stride(0) + column * out.stride(1)] =
            f32::cast_from(centered) * x_scale[row] * w_scale[column];
    }
}

/// 16×32 output tile. Weights are `[k_word, column]` so a warp load is contiguous.
/// The inner K loop stays rolled; unrolling the previous tile spilled and ran at 0.11 TFLOP/s.
#[cube(launch_unchecked)]
fn int8_dot_tiled(
    x_packed: &Tensor<i32>,
    x_scale: &Tensor<f32>,
    w_packed: &Tensor<i32>,
    w_scale: &Tensor<f32>,
    w_zero: &Tensor<i32>,
    out: &mut Tensor<f32>,
    words: u32,
) {
    let mut tile_x = SharedMemory::<i32>::new(256usize);
    let mut tile_w = SharedMemory::<i32>::new(512usize);
    let rows = x_scale.shape(0);
    let columns = w_scale.shape(0);
    let count = words as usize;
    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let tid = ty * 8usize + tx;
    let row0 = (CUBE_POS_Y as usize) * 16usize;
    let col_block = (CUBE_POS_X as usize) * 32usize;
    let row = row0 + ty;
    let col0 = col_block + tx * 4usize;
    let mut a0 = 0i32;
    let mut a1 = 0i32;
    let mut a2 = 0i32;
    let mut a3 = 0i32;
    let mut sum = 0i32;
    let tiles = (count + 15usize) / 16usize;
    for tile in 0..tiles {
        let k0 = tile * 16usize;
        for pass in 0..2usize {
            let element = tid + pass * 128usize;
            let local_row = element / 16usize;
            let local_word = element % 16usize;
            let x_row = row0 + local_row;
            let word = k0 + local_word;
            let mut xv = 0i32;
            if x_row < rows && word < count {
                xv = x_packed[x_row * count + word];
            }
            tile_x[local_row * 16usize + local_word] = xv;
        }
        for pass in 0..4usize {
            let element = tid + pass * 128usize;
            let local_k = element / 32usize;
            let local_col = element % 32usize;
            let word = k0 + local_k;
            let w_col = col_block + local_col;
            let mut wv = 0i32;
            if w_col < columns && word < count {
                wv = w_packed[word * columns + w_col];
            }
            tile_w[local_k * 32usize + local_col] = wv;
        }
        sync_cube();
        if row < rows {
            for kk in 0..16usize {
                let x_word = tile_x[ty * 16usize + kk];
                let base = kk * 32usize + tx * 4usize;
                a0 += dot4(x_word, tile_w[base]);
                a1 += dot4(x_word, tile_w[base + 1usize]);
                a2 += dot4(x_word, tile_w[base + 2usize]);
                a3 += dot4(x_word, tile_w[base + 3usize]);
                sum += sum4(x_word);
            }
        }
        sync_cube();
    }
    if row < rows {
        let row_scale = x_scale[row];
        if col0 < columns {
            store(out, row, col0, a0, sum, row_scale, w_scale, w_zero);
        }
        if col0 + 1usize < columns {
            store(out, row, col0 + 1usize, a1, sum, row_scale, w_scale, w_zero);
        }
        if col0 + 2usize < columns {
            store(out, row, col0 + 2usize, a2, sum, row_scale, w_scale, w_zero);
        }
        if col0 + 3usize < columns {
            store(out, row, col0 + 3usize, a3, sum, row_scale, w_scale, w_zero);
        }
    }
}

#[cube(launch_unchecked)]
fn int8_dot_wide(
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

#[cube(launch_unchecked)]
fn int8_dot_reuse4(
    x_packed: &Tensor<i32>,
    x_scale: &Tensor<f32>,
    w_packed: &Tensor<i32>,
    w_scale: &Tensor<f32>,
    w_zero: &Tensor<i32>,
    out: &mut Tensor<f32>,
    words: u32,
) {
    let row = ABSOLUTE_POS_Y as usize;
    let column0 = (ABSOLUTE_POS_X as usize) * 4;
    let count = words as usize;
    if row < x_scale.shape(0) {
        let mut acc0 = 0i32;
        let mut acc1 = 0i32;
        let mut acc2 = 0i32;
        let mut acc3 = 0i32;
        let mut sum = 0i32;
        let columns = w_scale.shape(0);
        for word in 0..count {
            let x_word = x_packed[row * count + word];
            sum += sum4(x_word);
            if column0 < columns {
                acc0 += dot4(x_word, w_packed[column0 * count + word]);
            }
            if column0 + 1 < columns {
                acc1 += dot4(x_word, w_packed[(column0 + 1) * count + word]);
            }
            if column0 + 2 < columns {
                acc2 += dot4(x_word, w_packed[(column0 + 2) * count + word]);
            }
            if column0 + 3 < columns {
                acc3 += dot4(x_word, w_packed[(column0 + 3) * count + word]);
            }
        }
        let row_scale = x_scale[row];
        if column0 < columns {
            store(out, row, column0, acc0, sum, row_scale, w_scale, w_zero);
        }
        if column0 + 1 < columns {
            store(out, row, column0 + 1, acc1, sum, row_scale, w_scale, w_zero);
        }
        if column0 + 2 < columns {
            store(out, row, column0 + 2, acc2, sum, row_scale, w_scale, w_zero);
        }
        if column0 + 3 < columns {
            store(out, row, column0 + 3, acc3, sum, row_scale, w_scale, w_zero);
        }
    }
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

fn launch(
    kernel: Kernel,
    x: RawTensor,
    w_packed: RawTensor,
    w_scale: RawTensor,
    w_zero: RawTensor,
    quant_repeats: u32,
    gemm_repeats: u32,
) -> RawTensor {
    let x_shape = x.shape();
    let rows = x_shape[0];
    let width = x_shape[1];
    let columns = w_scale.shape()[0];
    let words = width / 4;
    let client = x.client.clone();
    let x_packed = empty_tensor(&x, Shape::new([rows * words]), DType::I32);
    let x_scale = empty_tensor(&x, Shape::new([rows]), DType::F32);
    let out = empty_tensor(&x, Shape::new([rows, columns]), DType::F32);
    unsafe {
        for _ in 0..quant_repeats {
            quantize_rows::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(rows as u32, 1, 1),
                CubeDim::new_1d(256),
                x.clone().into_tensor_arg(),
                x_packed.clone().into_tensor_arg(),
                x_scale.clone().into_tensor_arg(),
                words as u32,
            );
        }
        for _ in 0..gemm_repeats {
            launch_dot(
                kernel, &x_packed, &x_scale, &w_packed, &w_scale, &w_zero, &out, rows, columns,
                words,
            );
        }
    }
    out
}

/// 64×64 device-tensor int8 `matmul2d`. `MatrixIdent::Accumulator` selects this lowering.
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
    #[comptime] rows: u32,
    #[comptime] cols: u32,
    #[comptime] inner: u32,
) {
    mma_i8_64(x_packed, w_packed, acc, rows, cols, inner);
}

#[cube(launch_unchecked)]
fn int8_row_sum(x_packed: &Tensor<i32>, sum: &mut Tensor<i32>, words: u32) {
    let row = ABSOLUTE_POS;
    let count = words as usize;
    if row < sum.shape(0) {
        let mut total = 0i32;
        for word in 0..count {
            total += sum4(x_packed[row * count + word]);
        }
        sum[row] = total;
    }
}

#[cube(launch_unchecked)]
fn int8_epilogue(
    acc: &Tensor<i32>,
    x_scale: &Tensor<f32>,
    x_sum: &Tensor<i32>,
    w_scale: &Tensor<f32>,
    w_zero: &Tensor<i32>,
    out: &mut Tensor<f32>,
    cols: u32,
) {
    let i = ABSOLUTE_POS as usize;
    let n = cols as usize;
    if i < x_scale.shape(0) * n {
        let row = i / n;
        let col = i % n;
        store(
            out,
            row,
            col,
            acc[i],
            x_sum[row],
            x_scale[row],
            w_scale,
            w_zero,
        );
    }
}

fn launch_dot(
    kernel: Kernel,
    x_packed: &RawTensor,
    x_scale: &RawTensor,
    w_packed: &RawTensor,
    w_scale: &RawTensor,
    w_zero: &RawTensor,
    out: &RawTensor,
    rows: usize,
    columns: usize,
    words: usize,
) {
    unsafe {
        match kernel {
            Kernel::Naive => int8_dot_naive::launch_unchecked::<WgpuRuntime>(
                &out.client,
                CubeCount::Static((columns as u32).div_ceil(16), (rows as u32).div_ceil(16), 1),
                CubeDim::new_2d(16, 16),
                x_packed.clone().into_tensor_arg(),
                x_scale.clone().into_tensor_arg(),
                w_packed.clone().into_tensor_arg(),
                w_scale.clone().into_tensor_arg(),
                w_zero.clone().into_tensor_arg(),
                out.clone().into_tensor_arg(),
                words as u32,
            ),
            Kernel::Reuse4 => int8_dot_reuse4::launch_unchecked::<WgpuRuntime>(
                &out.client,
                CubeCount::Static((columns as u32).div_ceil(64), (rows as u32).div_ceil(16), 1),
                CubeDim::new_2d(16, 16),
                x_packed.clone().into_tensor_arg(),
                x_scale.clone().into_tensor_arg(),
                w_packed.clone().into_tensor_arg(),
                w_scale.clone().into_tensor_arg(),
                w_zero.clone().into_tensor_arg(),
                out.clone().into_tensor_arg(),
                words as u32,
            ),
            Kernel::Tiled => int8_dot_tiled::launch_unchecked::<WgpuRuntime>(
                &out.client,
                CubeCount::Static((columns as u32).div_ceil(32), (rows as u32).div_ceil(16), 1),
                CubeDim::new_2d(8, 16),
                x_packed.clone().into_tensor_arg(),
                x_scale.clone().into_tensor_arg(),
                w_packed.clone().into_tensor_arg(),
                w_scale.clone().into_tensor_arg(),
                w_zero.clone().into_tensor_arg(),
                out.clone().into_tensor_arg(),
                words as u32,
            ),
            Kernel::Tile64 => {
                let acc = empty_tensor(out, Shape::new([rows * columns]), DType::I32);
                let x_sum = empty_tensor(out, Shape::new([rows]), DType::I32);
                {
                    int8_row_sum::launch_unchecked::<WgpuRuntime>(
                        &out.client,
                        CubeCount::Static((rows as u32).div_ceil(256), 1, 1),
                        CubeDim::new_1d(256),
                        x_packed.clone().into_tensor_arg(),
                        x_sum.clone().into_tensor_arg(),
                        words as u32,
                    );
                    int8_dot_tile64::launch_unchecked::<WgpuRuntime>(
                        &out.client,
                        CubeCount::Static(
                            (columns as u32).div_ceil(64),
                            (rows as u32).div_ceil(64),
                            1,
                        ),
                        CubeDim::new_1d(128),
                        x_packed.clone().into_tensor_arg(),
                        w_packed.clone().into_tensor_arg(),
                        acc.clone().into_tensor_arg(),
                        rows as u32,
                        columns as u32,
                        (words * 4) as u32,
                    );
                    int8_epilogue::launch_unchecked::<WgpuRuntime>(
                        &out.client,
                        CubeCount::Static((rows * columns).div_ceil(256) as u32, 1, 1),
                        CubeDim::new_1d(256),
                        acc.into_tensor_arg(),
                        x_scale.clone().into_tensor_arg(),
                        x_sum.into_tensor_arg(),
                        w_scale.clone().into_tensor_arg(),
                        w_zero.clone().into_tensor_arg(),
                        out.clone().into_tensor_arg(),
                        columns as u32,
                    );
                }
            }
            Kernel::Mma => int8_dot_mma::launch_unchecked::<WgpuRuntime>(
                &out.client,
                CubeCount::Static((columns as u32).div_ceil(32), (rows as u32).div_ceil(16), 1),
                CubeDim::new_1d(32),
                x_packed.clone().into_tensor_arg(),
                x_scale.clone().into_tensor_arg(),
                w_packed.clone().into_tensor_arg(),
                w_scale.clone().into_tensor_arg(),
                w_zero.clone().into_tensor_arg(),
                out.clone().into_tensor_arg(),
                words as u32,
            ),
            Kernel::Wide => int8_dot_wide::launch_unchecked::<WgpuRuntime>(
                &out.client,
                CubeCount::Static(
                    (columns as u32).div_ceil(8).div_ceil(16),
                    (rows as u32).div_ceil(16),
                    1,
                ),
                CubeDim::new_2d(16, 16),
                x_packed.clone().into_tensor_arg(),
                x_scale.clone().into_tensor_arg(),
                w_packed.clone().into_tensor_arg(),
                w_scale.clone().into_tensor_arg(),
                w_zero.clone().into_tensor_arg(),
                out.clone().into_tensor_arg(),
                words as u32,
            ),
            Kernel::Coalesced => int8_dot_coalesced::launch_unchecked::<WgpuRuntime>(
                &out.client,
                CubeCount::Static((columns as u32).div_ceil(32), (rows as u32).div_ceil(8), 1),
                CubeDim::new_2d(32, 8),
                x_packed.clone().into_tensor_arg(),
                x_scale.clone().into_tensor_arg(),
                w_packed.clone().into_tensor_arg(),
                w_scale.clone().into_tensor_arg(),
                w_zero.clone().into_tensor_arg(),
                out.clone().into_tensor_arg(),
                words as u32,
            ),
        }
    }
}

struct Int8Gemm {
    desc: CustomOpIr,
    kernel: Kernel,
    quant_repeats: u32,
    gemm_repeats: u32,
}

impl std::fmt::Debug for Int8Gemm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Int8Gemm")
            .field("kernel", &self.kernel)
            .field("desc", &self.desc)
            .finish()
    }
}

impl Operation<Runtime> for Int8Gemm {
    fn execute(&self, handles: &mut HandleContainer<<Runtime as FusionRuntime>::FusionHandle>) {
        let (inputs, [output]) = self.desc.as_fixed::<4, 1>();
        let x = handles.get_float_tensor::<Raw>(&inputs[0]);
        let w_packed = handles.get_int_tensor::<Raw>(&inputs[1]);
        let w_scale = handles.get_float_tensor::<Raw>(&inputs[2]);
        let w_zero = handles.get_int_tensor::<Raw>(&inputs[3]);
        handles.register_float_tensor::<Raw>(
            &output.id,
            launch(
                self.kernel,
                x,
                w_packed,
                w_scale,
                w_zero,
                self.quant_repeats,
                self.gemm_repeats,
            ),
        );
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

pub fn try_int8_gemm<B: Backend>(
    kernel: Kernel,
    x: BurnTensor<B, 2>,
    w_packed: BurnTensor<B, 1, burn::tensor::Int>,
    w_scale: BurnTensor<B, 1>,
    w_zero: BurnTensor<B, 1, burn::tensor::Int>,
) -> Option<BurnTensor<B, 2>> {
    try_int8_gemm_times(kernel, x, w_packed, w_scale, w_zero, 1, 1)
}

/// Repeat the quantize and gemm launches inside one fused op so a bench can
/// subtract the fixed upload and readback.
pub fn try_int8_gemm_times<B: Backend>(
    kernel: Kernel,
    x: BurnTensor<B, 2>,
    w_packed: BurnTensor<B, 1, burn::tensor::Int>,
    w_scale: BurnTensor<B, 1>,
    w_zero: BurnTensor<B, 1, burn::tensor::Int>,
    quant_repeats: u32,
    gemm_repeats: u32,
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
        "int8_gemm_probe",
        &inputs.map(FusionTensor::into_ir),
        &[output],
    );
    let tensor = client
        .register(
            streams,
            OperationIr::Custom(desc.clone()),
            Int8Gemm {
                desc,
                kernel,
                quant_repeats,
                gemm_repeats,
            },
        )
        .output();
    Some(BurnTensor::from_primitive(TensorPrimitive::Float(
        fusion_out::<B>(tensor)?,
    )))
}

/// `[column, k_word]` to `[k_word, column]` so a warp can read one K step contiguously.
pub fn transpose_packed(packed: &[i32], columns: usize, words: usize) -> Vec<i32> {
    let mut out = vec![0; packed.len()];
    for column in 0..columns {
        for word in 0..words {
            out[word * columns + column] = packed[column * words + word];
        }
    }
    out
}

fn fusion_out<B: Backend>(tensor: FusionTensor<Runtime>) -> Option<B::FloatTensorPrimitive> {
    let tensor: Box<dyn Any> = Box::new(tensor);
    tensor
        .downcast::<B::FloatTensorPrimitive>()
        .ok()
        .map(|tensor| *tensor)
}
