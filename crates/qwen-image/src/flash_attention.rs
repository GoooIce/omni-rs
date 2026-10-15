//! FlashAttention-2: tiled QK/PV and online softmax, expressed in Rust/CubeCL.
//! M5 cooperative-tensor lane coordinates follow Apple's published NAX layout
//! (also used by MLX). See ../THIRD_PARTY.md for source attribution and license.
//! No MLX shader source or runtime is compiled or called.
use burn::tensor::f16;
use cubecl::{
    ir::{CoopMma, Instruction, Matrix, MatrixIdent, MatrixLayout},
    prelude::*,
};

/// A narrowly typed compiler intrinsic for a 16x32x16 M5 TensorOps instruction.
/// Each SIMD lane owns 8 half A, 16 half B and 16 float accumulator elements.
#[cube]
#[allow(unused_variables)]
fn m5_mma(a: &Array<f16>, b: &Array<f16>, c: &mut Array<f32>, #[comptime] transpose_b: bool) {
    intrinsic!(|scope| {
        scope.register(Instruction::new(
            CoopMma::ExecuteManual {
                matrix: Matrix {
                    ident: MatrixIdent::B,
                    m: 16,
                    n: 32,
                    k: 16,
                    storage: f16::as_type(scope).storage_type(),
                    layout: if transpose_b {
                        MatrixLayout::ColMajor
                    } else {
                        MatrixLayout::RowMajor
                    },
                },
                registers_a: *a.expand,
                registers_b: *b.expand,
                registers_c: *c.expand,
            },
            *c.expand,
        ));
    });
}

#[cube(launch_unchecked)]
pub(crate) fn flash_attention(
    q: &Tensor<Vector<f16, Const<4>>>,
    k: &Tensor<Vector<f16, Const<4>>>,
    v: &Tensor<Vector<f16, Const<4>>>,
    out: &mut Tensor<f16>,
    offset: u32,
    #[comptime] causal: bool,
    #[comptime] align_q: bool,
    #[comptime] align_k: bool,
) {
    let lane = UNIT_POS_PLANE;
    // A 16x16 cooperative tensor: two rows, four adjacent columns per lane.
    let row = (((lane >> 2) & 4) | ((lane >> 1) & 3)) as usize;
    let col = ((((lane >> 2) & 2) | (lane & 1)) * 4) as usize;
    let plane = (UNIT_POS / 32) as usize;
    let query_start = CUBE_POS_X as usize * 64 + plane * 16;
    let head = CUBE_POS_Y as usize;
    let batch = CUBE_POS_Z as usize;
    let ql = q.shape(2);
    let kl = k.shape(2);
    let kv_head = head / (q.shape(1) / k.shape(1));
    let qb = batch * q.stride(0) + head * q.stride(1);
    let kb = batch * k.stride(0) + kv_head * k.stride(1);
    let vb = batch * v.stride(0) + kv_head * v.stride(1);
    let ob = batch * out.stride(0) + head * out.stride(1);
    let mut acc = Array::<f32>::new(64usize);
    let mut maximum = Array::<f32>::new(2usize);
    let mut denominator = Array::<f32>::new(2usize);
    #[unroll]
    for i in 0usize..2usize {
        maximum[i] = -3.402823466e38f32;
        denominator[i] = 0.0;
    }
    #[unroll]
    for i in 0usize..64usize {
        acc[i] = 0.0;
    }
    // Stream Q/K/V tiles directly into registers, as in the NAX reference.
    // This avoids shared-memory bank conflicts and threadgroup barriers.
    let mut end = kl;
    if comptime!(causal) {
        end = usize::min(kl, CUBE_POS_X as usize * 64 + 64 + offset as usize);
    }
    for block in 0usize..(end + 31) / 32 {
        let start = block * 32;
        let mut scores = Array::<f32>::new(16usize);
        #[unroll]
        for i in 0usize..16usize {
            scores[i] = 0.0;
        }
        // Q (16x128) times K^T (128x32), accumulating in FP32 registers.
        #[unroll]
        for t in 0usize..8usize {
            let mut a = Array::<f16>::new(8usize);
            let mut b = Array::<f16>::new(16usize);
            #[unroll]
            for r in 0usize..2usize {
                let qr = query_start + row + r * 8;
                let d = t * 16 + col;
                let kr = start + row + r * 8;
                let mut q4 = Vector::<f16, Const<4>>::new(f16::cast_from(0.0f32));
                let mut k0 = Vector::<f16, Const<4>>::new(f16::cast_from(0.0f32));
                let mut k1 = Vector::<f16, Const<4>>::new(f16::cast_from(0.0f32));
                if comptime!(align_q) || qr < ql {
                    q4 = q[(qb + qr * q.stride(2) + d) / 4];
                }
                if comptime!(align_k) || kr < kl {
                    k0 = k[(kb + kr * k.stride(2) + d) / 4];
                }
                if comptime!(align_k) || kr + 16 < kl {
                    k1 = k[(kb + (kr + 16) * k.stride(2) + d) / 4];
                }
                #[unroll]
                for j in 0usize..4usize {
                    a[r * 4 + j] = q4[j];
                    b[r * 4 + j] = k0[j];
                    b[8 + r * 4 + j] = k1[j];
                }
            }
            m5_mma(&a, &b, &mut scores, true);
        }
        let mut probabilities = Array::<f16>::new(16usize);
        #[unroll]
        for r in 0usize..2usize {
            let qr = query_start + row + r * 8;
            let mut peak = f32::new(-3.402823466e38f32);
            #[unroll]
            for t in 0usize..2usize {
                #[unroll]
                for j in 0usize..4usize {
                    let index = t * 8 + r * 4 + j;
                    let kr = start + t * 16 + col + j;
                    let mut score = scores[index] * 0.08838834764831845f32;
                    if kr >= kl || (comptime!(causal) && kr > qr + offset as usize) {
                        score = -3.402823466e38f32;
                    }
                    scores[index] = score;
                    peak = f32::max(peak, score);
                }
            }
            peak = f32::max(peak, plane_shuffle_xor(peak, 1));
            peak = f32::max(peak, plane_shuffle_xor(peak, 8));
            let next_max = f32::max(maximum[r], peak);
            let correction = f32::exp(maximum[r] - next_max);
            let mut sum = 0.0f32;
            #[unroll]
            for t in 0usize..2usize {
                #[unroll]
                for j in 0usize..4usize {
                    let index = t * 8 + r * 4 + j;
                    let kr = start + t * 16 + col + j;
                    let mut probability = f32::exp(scores[index] - next_max);
                    if kr >= kl || (comptime!(causal) && kr > qr + offset as usize) {
                        probability = 0.0;
                    }
                    probabilities[index] = f16::cast_from(probability);
                    sum += probability;
                }
            }
            sum += plane_shuffle_xor(sum, 1);
            sum += plane_shuffle_xor(sum, 8);
            denominator[r] = denominator[r] * correction + sum;
            maximum[r] = next_max;
            #[unroll]
            for t in 0usize..8usize {
                #[unroll]
                for j in 0usize..4usize {
                    let index = t * 8 + r * 4 + j;
                    acc[index] *= correction;
                }
            }
        }
        // P (16x32) times V (32x128), with the online accumulator kept in registers.
        #[unroll]
        for n in 0usize..4usize {
            let mut c = Array::<f32>::new(16usize);
            #[unroll]
            for i in 0usize..16usize {
                c[i] = acc[n * 16 + i];
            }
            #[unroll]
            for t in 0usize..2usize {
                let mut a = Array::<f16>::new(8usize);
                let mut b = Array::<f16>::new(16usize);
                #[unroll]
                for r in 0usize..2usize {
                    let kr = start + t * 16 + row + r * 8;
                    let vc = n * 32 + col;
                    let mut v0 = Vector::<f16, Const<4>>::new(f16::cast_from(0.0f32));
                    let mut v1 = Vector::<f16, Const<4>>::new(f16::cast_from(0.0f32));
                    if comptime!(align_k) || kr < kl {
                        v0 = v[(vb + kr * v.stride(2) + vc) / 4];
                        v1 = v[(vb + kr * v.stride(2) + vc + 16) / 4];
                    }
                    #[unroll]
                    for j in 0usize..4usize {
                        a[r * 4 + j] = probabilities[t * 8 + r * 4 + j];
                        b[r * 4 + j] = v0[j];
                        b[r * 4 + j + 8] = v1[j];
                    }
                }
                m5_mma(&a, &b, &mut c, false);
            }
            #[unroll]
            for i in 0usize..16usize {
                acc[n * 16 + i] = c[i];
            }
        }
    }
    #[unroll]
    for t in 0usize..8usize {
        #[unroll]
        for i in 0usize..8usize {
            let r = i / 4;
            let qr = query_start + row + r * 8;
            let d = t * 16 + col + i % 4;
            if qr < ql {
                out[ob + qr * 128 + d] = f16::cast_from(acc[t * 8 + i] / denominator[r]);
            }
        }
    }
}
