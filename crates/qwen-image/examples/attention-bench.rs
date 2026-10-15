//! Compare the existing Burn policy and the Rust/CubeCL FlashAttention kernel on a real M5.

#[path = "support/parity.rs"]
mod parity;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("Requires an M5 Mac and --features metal");
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    use burn::{
        backend::wgpu::{Metal, WgpuDevice},
        tensor::{Bool, FloatDType, Tensor, TensorData, backend::Backend, f16, module},
    };
    use std::time::Instant;
    type B = Metal;
    // A shader/worker panic must fail the benchmark immediately; otherwise an
    // allocator can expose a previous output and produce a false parity pass.
    let panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        panic_hook(info);
        std::process::exit(1);
    }));
    let device = WgpuDevice::default();
    fn sample(i: usize, salt: u32) -> f32 {
        let mut x = (i as u32).wrapping_add(salt);
        x = (x ^ (x >> 16)).wrapping_mul(0x7feb352d);
        x = (x ^ (x >> 15)).wrapping_mul(0x846ca68b);
        ((x ^ (x >> 16)) >> 8) as f32 / 16777216.0 * 2.0 - 1.0
    }
    for (name, b, h, kh, ql, kl, causal, offset, layout, amplitude) in [
        ("text_gqa", 1, 32, 8, 433, 433, true, 0, 0, 1.0),
        ("dit_first_tail", 1, 32, 32, 275, 275, false, 0, 1, 1.0),
        ("dit_rectangular", 1, 32, 32, 360, 391, false, 31, 0, 1.0),
        ("dit_512", 1, 32, 32, 1024, 1043, false, 19, 0, 1.0),
        ("dit_1024", 1, 32, 32, 4096, 4115, false, 19, 0, 1.0),
        ("gqa_offset_causal", 2, 8, 2, 33, 64, true, 7, 1, 3.0),
        ("q1", 1, 4, 2, 1, 57, true, 56, 0, 1.0),
        ("strided_d", 1, 4, 4, 73, 91, false, 18, 2, 1.0),
        ("sharp_softmax", 1, 4, 4, 65, 97, true, 32, 0, 8.0),
        ("revisit_tail", 1, 32, 32, 324, 349, false, 25, 0, 1.0),
        ("aligned_q_and_k", 1, 32, 32, 1024, 2048, false, 0, 0, 1.0),
        ("aligned_causal_gqa", 1, 8, 2, 64, 128, true, 64, 0, 1.0),
        ("unaligned_d_offset", 1, 4, 4, 64, 66, false, 0, 3, 1.0),
        ("dense_q1_offset_mask", 1, 1, 1, 1, 57, true, 3, 3, 1.0),
    ] {
        let make = |heads, len, salt| {
            let stored_len = if layout == 1 { len + 2 } else { len };
            let stored_d = if layout == 3 { 130 } else { 128 };
            let mut x = Tensor::<B, 4>::from_data(
                TensorData::new(
                    (0..b * heads * stored_len * stored_d)
                        .map(|i| amplitude * sample(i, salt))
                        .collect::<Vec<_>>(),
                    [b, heads, stored_len, stored_d],
                ),
                &device,
            )
            .cast(FloatDType::F16);
            if layout == 1 {
                x = x.slice([0..b, 0..heads, 1..len + 1, 0..128]);
            }
            if layout == 3 {
                x = x.slice([0..b, 0..heads, 0..len, 1..129]);
            }
            if layout == 2 {
                x = x.reshape([b, heads, 128, len]).swap_dims(2, 3);
            }
            x
        };
        let q = make(h, ql, 42);
        let k = make(kh, kl, 1234);
        let v = make(kh, kl, 321);
        let data = |x: Tensor<B, 4>| x.into_data().convert::<f32>().to_vec::<f32>().unwrap();
        let qv = data(q.clone());
        let kv = data(k.clone());
        let vv = data(v.clone());
        let repeat = |x: Tensor<B, 4>| {
            if h == kh {
                x
            } else {
                x.reshape([b, kh, 1, kl, 128])
                    .repeat_dim(2, h / kh)
                    .reshape([b, h, kl, 128])
            }
        };
        let mask = if causal && (ql > 1 || offset + 1 < kl) {
            Some(Tensor::<B, 4, Bool>::from_data(
                TensorData::new(
                    (0..ql)
                        .flat_map(|r| (0..kl).map(move |c| c > r + offset))
                        .collect::<Vec<_>>(),
                    [1, 1, ql, kl],
                ),
                &device,
            ))
        } else {
            None
        };
        let mut outputs = vec![];
        let mut times = vec![];
        for fast in [false, true] {
            let mut elapsed = vec![];
            for run in 0..7 {
                B::sync(&device).unwrap();
                let start = Instant::now();
                let out = if fast {
                    omni_qwen_image::kernels::try_attention(&q, &k, &v, causal, offset)
                        .expect("Set OMNI_QWEN_RUST_ATTN=1 on an M5 Mac")
                } else if causal || ql % 256 != 0 {
                    module::attention_fallback(
                        q.clone(),
                        repeat(k.clone()),
                        repeat(v.clone()),
                        mask.clone(),
                        None,
                        Default::default(),
                    )
                } else {
                    module::attention(
                        q.clone(),
                        repeat(k.clone()),
                        repeat(v.clone()),
                        None,
                        None,
                        Default::default(),
                    )
                };
                B::sync(&device).unwrap();
                if run >= 2 {
                    elapsed.push(start.elapsed().as_secs_f64() * 1000.0);
                }
                if run == 6 {
                    outputs.push(data(out));
                }
            }
            elapsed.sort_by(f64::total_cmp);
            times.push(elapsed[elapsed.len() / 2]);
        }
        let mut err = 0.0;
        let mut norm = 0.0;
        let mut max_abs = 0.0_f32;
        for (a, c) in outputs[0].iter().zip(&outputs[1]) {
            assert!(a.is_finite() && c.is_finite());
            err += f64::from(a - c).powi(2);
            norm += f64::from(*a).powi(2);
            max_abs = max_abs.max((a - c).abs());
        }
        // Independent f64 softmax of FP16-rounded inputs, including both tails.
        let mut reference_samples = Vec::with_capacity(12 * 128);
        let mut candidate_samples = Vec::with_capacity(12 * 128);
        for s in 0..12 {
            let bi = s % b;
            let hi = (s * 13) % h;
            let ki = hi / (h / kh);
            let r = if s == 11 {
                ql - 1
            } else if s == 0 {
                0
            } else {
                (s * 997) % ql
            };
            let count = if causal { (r + offset + 1).min(kl) } else { kl };
            let scores = (0..count)
                .map(|c| {
                    (0..128)
                        .map(|d| {
                            f64::from(qv[((bi * h + hi) * ql + r) * 128 + d])
                                * f64::from(kv[((bi * kh + ki) * kl + c) * 128 + d])
                        })
                        .sum::<f64>()
                        / 128_f64.sqrt()
                })
                .collect::<Vec<_>>();
            let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores.iter().map(|s| (s - peak).exp()).collect::<Vec<_>>();
            let total: f64 = weights.iter().sum();
            for d in 0..128 {
                let e = weights
                    .iter()
                    .enumerate()
                    .map(|(c, w)| w * f64::from(vv[((bi * kh + ki) * kl + c) * 128 + d]))
                    .sum::<f64>()
                    / total;
                let e = f16::from_f64(e).to_f64();
                let a = f64::from(outputs[1][((bi * h + hi) * ql + r) * 128 + d]);
                reference_samples.push(e);
                candidate_samples.push(a);
            }
        }
        let cpu_error = parity::compare(&reference_samples, &candidate_samples).unwrap();
        let cpu_relative_l2 = cpu_error.relative_l2;
        let cpu_max_abs = cpu_error.max_abs;
        println!(
            "{}",
            serde_json::json!({"case":name,"batch":b,"heads":h,"kv_heads":kh,"q_len":ql,"kv_len":kl,
            "causal":causal,"offset":offset,"layout":layout,"amplitude":amplitude,
            "burn_ms":times[0],"rust_ms":times[1],"speedup":times[0]/times[1],
            "max_abs":max_abs,"relative_l2":(err/norm).sqrt(),"cpu_sample_max_abs":cpu_max_abs,"cpu_relative_l2":cpu_relative_l2})
        );
        assert!(
            cpu_error.passes(0.003, 0.04),
            "{name}: inaccurate Rust attention"
        );
    }
    let fp32 = Tensor::<B, 4>::zeros([1, 4, 64, 128], &device);
    assert!(omni_qwen_image::kernels::try_attention(&fp32, &fp32, &fp32, false, 0).is_none());
    let vision = Tensor::<B, 4>::zeros([1, 4, 64, 72], &device).cast(FloatDType::F16);
    assert!(omni_qwen_image::kernels::try_attention(&vision, &vision, &vision, false, 0).is_none());
    println!(
        "{}",
        serde_json::json!({"case":"unsupported_dtype_and_head_dim", "passed":true})
    );
}
