use super::*;
use crate::{Error, weights::Weights};
use burn::backend::NdArray;
use safetensors::{Dtype, tensor::TensorView};
type B = NdArray;

fn tensor<const D: usize>(values: Vec<f32>, shape: [usize; D]) -> Tensor<B, D> {
    Tensor::from_data(TensorData::new(values, shape), &Default::default())
}
fn values<const D: usize>(x: Tensor<B, D>) -> Vec<f32> {
    x.into_data().to_vec::<f32>().unwrap()
}
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= tolerance,
            "index={index}: {a} != {e}"
        );
    }
}
fn weights(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Weights {
    let bytes: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, _, values)| values.iter().flat_map(|x| x.to_le_bytes()).collect())
        .collect();
    let views: Vec<_> = tensors
        .iter()
        .zip(&bytes)
        .map(|((name, shape, _), bytes)| {
            (
                *name,
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    Weights::from_bytes(vec![safetensors::serialize(views, None).unwrap()]).unwrap()
}

#[test]
fn gelu_variants_match_independent_references() {
    let input: Vec<f32> = (0..513).map(|i| (i as f32 - 256.0) / 32.0).collect();
    let x = tensor(input.clone(), [513]);
    let reference = values(burn::tensor::activation::gelu(x.clone()));
    close(
        &values(gelu(x.clone(), GeluApproximation::Erf, FloatDType::F32)),
        &reference,
        2e-6,
    );
    let reference: Vec<_> = input
        .iter()
        .map(|&x| {
            let x = x as f64;
            (0.5 * x
                * (1.0 + ((x + 0.044715 * x.powi(3)) * (2.0 / std::f64::consts::PI).sqrt()).tanh()))
                as f32
        })
        .collect();
    close(
        &values(gelu(x, GeluApproximation::Tanh, FloatDType::F32)),
        &reference,
        2e-6,
    );
}

#[test]
fn linear_loading_preserves_checkpoint_orientation_and_optional_bias() {
    let w = weights(&[
        ("proj.weight", vec![2, 3], vec![1., 2., 3., 4., 5., 6.]),
        ("proj.bias", vec![2], vec![0.5, -0.25]),
    ]);
    for bias in [false, true] {
        let linear =
            Linear::<B>::load(&w, "proj", 3, 2, bias, Precision::F32, &Default::default()).unwrap();
        let expected = if bias {
            vec![14.5, 31.75, 5.5, 10.75]
        } else {
            vec![14., 32., 5., 11.]
        };
        close(
            &values(linear.forward(tensor(vec![1., 2., 3., 1., -1., 2.], [1, 2, 3]))),
            &expected,
            1e-6,
        );
    }
}

#[test]
fn normalization_policies_and_scaled_gemma_preserve_affine_results() {
    let w = weights(&[
        ("norm.weight", vec![3], vec![0.5, 1., 1.5]),
        ("norm.bias", vec![3], vec![0.1, -0.2, 0.3]),
    ]);
    let input = [1., 2., 4., -3., 1., 2.];
    for kind in [NormKind::Rms, NormKind::Gemma, NormKind::Layer] {
        let mut expected = Vec::new();
        for row in input.chunks(3) {
            let mean: f64 = if kind == NormKind::Layer {
                row.iter().sum::<f64>() / 3.0
            } else {
                0.0
            };
            let variance = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 3.0;
            for (i, &x) in row.iter().enumerate() {
                let scale = [0.5, 1., 1.5][i] + if kind == NormKind::Gemma { 1.0 } else { 0.0 };
                let bias = if kind == NormKind::Layer {
                    [0.1, -0.2, 0.3][i]
                } else {
                    0.0
                };
                expected.push(((x - mean) / (variance + 1e-5).sqrt() * scale + bias) as f32);
            }
        }
        for reduction in [NormReduction::Divide, NormReduction::MultiplyReciprocal] {
            let norm =
                Norm::<B>::load(&w, "norm", 3, 1e-5, kind, reduction, &Default::default()).unwrap();
            let x = tensor(input.iter().map(|&x| x as f32).collect(), [2, 3]);
            close(
                &values(norm.forward(x.clone(), FloatDType::F32)),
                &expected,
                5e-7,
            );
            if kind == NormKind::Gemma {
                let scaled =
                    norm.forward_scaled(x * (1.0 / 16.0), FloatDType::F32, 1.0 / 16.0, 0.25);
                close(
                    &values(scaled),
                    &expected.iter().map(|e| e * 0.25).collect::<Vec<_>>(),
                    2e-7,
                );
            }
        }
    }
}

#[test]
fn batched_gqa_offset_causal_and_windows_match_scalar_softmax() {
    let (batch, heads, kv_heads, k_len, width) = (2, 4, 2, 5, 2);
    let ks: Vec<f32> = (0..batch * kv_heads * k_len * width)
        .map(|i| (i as f32 * 0.19).cos())
        .collect();
    let vs: Vec<_> = ks.iter().map(|x| x * 0.7).collect();
    for (q_len, causal, offset, window) in [
        (3, false, 0, None),
        (3, true, 2, None),
        (1, true, 1, None),
        (3, true, 2, Some(2)),
        (3, false, 1, Some(3)),
    ] {
        let qs: Vec<f32> = (0..batch * heads * q_len * width)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let mut expected = Vec::new();
        for b in 0..batch {
            for head in 0..heads {
                for row in 0..q_len {
                    let pos = row + offset;
                    let visible: Vec<_> = (0..k_len)
                        .filter(|&col| {
                            if causal {
                                col <= pos && window.is_none_or(|w| pos - col < w)
                            } else {
                                window.is_none_or(|w: usize| {
                                    col + w.div_ceil(2) > pos && col <= pos + w / 2
                                })
                            }
                        })
                        .collect();
                    let scores: Vec<_> = visible
                        .iter()
                        .map(|&col| {
                            let dot: f64 = (0..width)
                                .map(|d| {
                                    qs[((b * heads + head) * q_len + row) * width + d] as f64
                                        * ks[((b * kv_heads + head / (heads / kv_heads)) * k_len
                                            + col)
                                            * width
                                            + d] as f64
                                })
                                .sum();
                            (dot / (width as f64).sqrt()).exp()
                        })
                        .collect();
                    let total: f64 = scores.iter().sum();
                    for d in 0..width {
                        expected.push(
                            visible
                                .iter()
                                .zip(&scores)
                                .map(|(&col, &s)| {
                                    s / total
                                        * vs[((b * kv_heads + head / (heads / kv_heads)) * k_len
                                            + col)
                                            * width
                                            + d] as f64
                                })
                                .sum::<f64>() as f32,
                        );
                    }
                }
            }
        }
        for kernel in [AttentionKernel::Automatic, AttentionKernel::ImageAligned] {
            let actual = attention(
                tensor(qs.clone(), [batch, heads, q_len, width]),
                tensor(ks.clone(), [batch, kv_heads, k_len, width]),
                tensor(vs.clone(), [batch, kv_heads, k_len, width]),
                AttentionOptions {
                    causal,
                    offset,
                    window,
                    kernel,
                },
            );
            close(&values(actual), &expected, 2e-6);
        }
    }
}

#[test]
fn rotary_layouts_share_math_without_reordering_positions() {
    let frequencies = [1.0, 0.01];
    let (cos, sin) = rope_tables::<B>(&frequencies, 2, 2, FloatDType::F32, &Default::default());
    let input = vec![1., 2., 3., 4., 5., 6., 7., 8.];
    let expected: Vec<_> = input
        .chunks(4)
        .enumerate()
        .flat_map(|(pos, row)| {
            (0..4).map(move |i| {
                let angle = (pos + 2) as f64 * frequencies[i % 2];
                (row[i] as f64 * angle.cos()
                    + if i < 2 {
                        -(row[i + 2] as f64)
                    } else {
                        row[i - 2] as f64
                    } * angle.sin()) as f32
            })
        })
        .collect();
    close(
        &values(rope(
            tensor(input.clone(), [1, 1, 2, 4]),
            &cos,
            &sin,
            RotaryLayout::SplitHalf,
        )),
        &expected,
        1e-6,
    );
    let angles = [2.0_f64, 2., 0.02, 0.02, 3., 3., 0.03, 0.03];
    let cos = tensor(
        angles.iter().map(|x| x.cos() as f32).collect(),
        [1, 1, 2, 4],
    );
    let sin = tensor(
        angles.iter().map(|x| x.sin() as f32).collect(),
        [1, 1, 2, 4],
    );
    let expected: Vec<_> = input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let rotated = if i % 2 == 0 {
                -input[i + 1]
            } else {
                input[i - 1]
            };
            (x as f64 * angles[i].cos() + rotated as f64 * angles[i].sin()) as f32
        })
        .collect();
    close(
        &values(rope(
            tensor(input, [1, 1, 2, 4]),
            &cos,
            &sin,
            RotaryLayout::Interleaved,
        )),
        &expected,
        1e-6,
    );
}

#[test]
fn embedding_crosses_shards_preserves_order_and_shares_output_projection() {
    let size = Embedding::<B>::SHARD_ROWS + 3;
    let data: Vec<f32> = (0..size).flat_map(|i| [i as f32, 1.0]).collect();
    let w = weights(&[("tokens", vec![size, 2], data)]);
    let embedding =
        Embedding::<B>::load(&w, "tokens", size, 2, Precision::F32, &Default::default()).unwrap();
    let ids = [8192, 8194, 1, 8191, 8193, 0];
    let expected: Vec<_> = ids.iter().flat_map(|&id| [id as f32, 1.0]).collect();
    close(&values(embedding.embed(&ids).unwrap()), &expected, 0.0);
    assert_eq!(embedding.forward(&ids).unwrap().dims(), [1, ids.len(), 2]);
    let logits = values(embedding.logits(tensor(vec![2., -1.], [1, 2])));
    close(
        &logits,
        &(0..size).map(|i| 2.0 * i as f32 - 1.0).collect::<Vec<_>>(),
        0.0,
    );
    let batched = values(embedding.logits(tensor(vec![2., -1., -3., 2.], [2, 2])));
    let expected: Vec<_> = (0..size)
        .map(|i| 2.0 * i as f32 - 1.0)
        .chain((0..size).map(|i| -3.0 * i as f32 + 2.0))
        .collect();
    close(&batched, &expected, 0.0);
    assert!(embedding.embed(&[]).is_err());
    assert!(embedding.embed(&[size as u32]).is_err());
    let mut uploads = 0;
    let result = Embedding::<B>::load_with(
        &w,
        "tokens",
        size,
        2,
        Precision::F32,
        &Default::default(),
        || {
            uploads += 1;
            if uploads == 2 {
                Err(Error::Interrupted)
            } else {
                Ok(())
            }
        },
    );
    assert!(matches!(result, Err(Error::Interrupted)));
    assert_eq!(uploads, 2);
}

#[test]
fn kv_cache_truncates_reuses_suffix_and_preserves_static_prefix() {
    let mut cache = KvCache::<B>::default();
    let initial = tensor(vec![1., 2., 3., 4., 5., 6.], [2, 1, 3, 1]);
    cache.append(initial.clone(), initial.clone() * 10.0, 0);
    let next = tensor(vec![7., 8.], [2, 1, 1, 1]);
    let (k, v) = cache.append(next.clone(), next * 10.0, 1);
    close(&values(k), &[1., 7., 4., 8.], 0.0);
    close(&values(v), &[10., 70., 40., 80.], 0.0);
    assert_eq!(cache.len(), 2);
    cache.retain_last(1);
    let next = tensor(vec![9., 10.], [2, 1, 1, 1]);
    let (k, v) = cache.append(next.clone(), next * 10.0, 1);
    close(&values(k.clone()), &[7., 9., 8., 10.], 0.0);
    let prefix = KvCache::copy_prefix(&k, &v, 1);
    let next = tensor(vec![11., 12.], [2, 1, 1, 1]);
    let (k, _) = prefix.concat(next.clone(), next * 10.0, prefix.len());
    close(&values(k), &[7., 11., 8., 12.], 0.0);
    assert_eq!(prefix.len(), 1);
    cache.clear();
    assert!(cache.is_empty());
}

#[test]
fn fp16_ranges_name_the_existing_model_headroom() {
    let x = tensor(vec![-70000., 65000., 70000.], [3]);
    close(
        &values(clamp_f16(x.clone(), Fp16Range::AsrHeadroom)),
        &[-64504., 64504., 64504.],
        0.0,
    );
    close(
        &values(clamp_f16(x, Fp16Range::Finite)),
        &[-65504., 65000., 65504.],
        0.0,
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub(super) fn metal_device() -> burn::backend::wgpu::WgpuDevice {
    use burn::backend::wgpu::{WgpuDevice, graphics, init_setup};
    static INIT: std::sync::Once = std::sync::Once::new();
    let device = WgpuDevice::default();
    // init_setup registers a process-wide runtime; a second registration panics.
    INIT.call_once(|| {
        init_setup::<graphics::Metal>(&device, Default::default());
    });
    device
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_fused_linear_gelu_matches_cpu() {
    use burn::backend::wgpu::Metal;
    fn linear<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 * 0.13).sin()).collect();
        let w: Vec<f32> = (0..64 * 32)
            .map(|i| (i as f32 * 0.07).cos() * 0.1)
            .collect();
        let b: Vec<f32> = (0..64).map(|i| i as f32 / 640.0).collect();
        module::linear(
            Tensor::from_data(TensorData::new(x, [1, 32]), device),
            Tensor::<B, 2>::from_data(TensorData::new(w, [64, 32]), device).transpose(),
            Some(Tensor::from_data(TensorData::new(b, [64]), device)),
        )
    }
    let expected = values(burn::tensor::activation::gelu(linear::<B>(
        &Default::default(),
    )));
    let device = metal_device();
    // Keep the linear result lazy: fusion is the regression case.
    let actual = gelu(
        linear::<Metal>(&device),
        GeluApproximation::Erf,
        FloatDType::F32,
    )
    .into_data()
    .to_vec::<f32>()
    .unwrap();
    close(&actual, &expected, 2e-6);
}
