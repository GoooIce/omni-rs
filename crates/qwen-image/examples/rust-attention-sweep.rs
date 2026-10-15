//! Measure existing CubeK Rust FlashAttention tiling on the local Metal device.
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("Requires an M5 Mac and --features metal");
}
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    native::run();
}
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native {
    use burn::{
        backend::wgpu::WgpuDevice,
        tensor::{FloatDType, Tensor, TensorData, backend::Backend},
    };
    use burn_cubecl::{
        CubeBackend,
        kernel::attention::{AttentionStrategy, attention},
    };
    use cubecl::wgpu::WgpuRuntime;
    use cubek_attention::routines::blackbox_accelerated::BlackboxAcceleratedStrategy;
    use std::time::Instant;
    type B = CubeBackend<WgpuRuntime, f32, i32, u8>;

    pub(crate) fn run() {
        let device = WgpuDevice::default();
        let args: Vec<u8> = std::env::args()
            .skip(1)
            .map(|s| s.parse().unwrap())
            .collect();
        let configurations = if args.is_empty() {
            vec![(4, 1, 1)]
        } else {
            vec![(args[0], args[1], args[2])]
        };
        for (ql, kl) in [(1024, 1043), (4096, 4115)] {
            let make = |len, salt: u32| {
                Tensor::<B, 4>::from_data(
                    TensorData::new(
                        (0..32 * len * 128)
                            .map(|i| {
                                let mut x = (i as u32).wrapping_add(salt);
                                x = (x ^ (x >> 16)).wrapping_mul(0x7feb352d);
                                x = (x ^ (x >> 15)).wrapping_mul(0x846ca68b);
                                ((x ^ (x >> 16)) >> 8) as f32 / 16777216.0 * 2.0 - 1.0
                            })
                            .collect::<Vec<_>>(),
                        [1, 32, len, 128],
                    ),
                    &device,
                )
                .cast(FloatDType::F16)
                .into_primitive()
                .tensor()
            };
            let q = make(ql, 42);
            let k = make(kl, 1234);
            let v = make(kl, 321);
            // Independent implementation, outside the timing loop.
            let reference = attention(
                q.clone(),
                k.clone(),
                v.clone(),
                None,
                None,
                Default::default(),
                AttentionStrategy::Fallback,
            )
            .unwrap();
            let reference =
                Tensor::<B, 4>::from_primitive(burn::tensor::TensorPrimitive::Float(reference))
                    .into_data()
                    .convert::<f32>()
                    .to_vec::<f32>()
                    .unwrap();
            for (planes, qs, kvs) in configurations.iter().copied() {
                let mut times = vec![];
                let mut final_out = None;
                let mut error = None;
                for run in 0..5 {
                    B::sync(&device).unwrap();
                    let start = Instant::now();
                    match attention(
                        q.clone(),
                        k.clone(),
                        v.clone(),
                        None,
                        None,
                        Default::default(),
                        AttentionStrategy::FlashBlackboxAccelerated(BlackboxAcceleratedStrategy {
                            num_planes: planes,
                            seq_q: qs,
                            seq_kv: kvs,
                        }),
                    ) {
                        Ok(out) => {
                            B::sync(&device).unwrap();
                            if run >= 2 {
                                times.push(start.elapsed().as_secs_f64() * 1000.0);
                            }
                            final_out = Some(out);
                        }
                        Err(e) => {
                            error = Some(format!("{e:?}"));
                            break;
                        }
                    }
                }
                if let Some(error) = error {
                    println!(
                        "{}",
                        serde_json::json!({"ql":ql,"kl":kl,"planes":planes,"q":qs,"kv":kvs,"error":error})
                    );
                    continue;
                }
                let data = Tensor::<B, 4>::from_primitive(burn::tensor::TensorPrimitive::Float(
                    final_out.unwrap(),
                ))
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();
                let target = &reference;
                let norm: f64 = target.iter().map(|a| (*a as f64).powi(2)).sum();
                let err: f64 = data
                    .iter()
                    .zip(target.iter())
                    .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                    .sum();
                times.sort_by(f64::total_cmp);
                println!(
                    "{}",
                    serde_json::json!({"ql":ql,"kl":kl,"planes":planes,"q":qs,"kv":kvs,"ms":times[times.len()/2],"relative_l2":(err/norm).sqrt(),"finite":data.iter().all(|a|a.is_finite())})
                );
            }
        }
    }
}
