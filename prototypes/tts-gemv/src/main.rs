use breeze_gemv_probe::{B, gemv};
use burn::{
    backend::wgpu::{graphics, init_setup},
    tensor::{FloatDType, Tensor, TensorData, backend::Backend, module},
};
use std::time::Instant;

fn main() {
    let device = Default::default();
    let setup = init_setup::<graphics::Metal>(&device, Default::default());
    eprintln!("{:?}", setup.adapter.get_info());
    for (k, n) in [
        (1024, 256),
        (1024, 1024),
        (1024, 8192),
        (8192, 1024),
        (2048, 6144),
    ] {
        for (batch, tokens) in [(1, 1), (1, 2), (2, 1), (2, 2)] {
            let m = batch * tokens;
            let x = Tensor::<B, 3>::from_data(
                TensorData::new(
                    (0..m * k)
                        .map(|i| ((i * 17 % 97) as f32 - 48.0) / 139.13)
                        .collect::<Vec<_>>(),
                    [batch, tokens, k],
                ),
                &device,
            )
            .cast(FloatDType::F16);
            let w = Tensor::<B, 2>::from_data(
                TensorData::new(
                    (0..n * k)
                        .map(|i| ((i * 13 % 89) as f32 - 44.0) / 251.31)
                        .collect::<Vec<_>>(),
                    [n, k],
                ),
                &device,
            )
            .cast(FloatDType::F16)
            .transpose();
            let expected = module::linear(x.clone(), w.clone(), None)
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();
            let actual = gemv(x.clone().reshape([m, k]), w.clone())
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (*a - *b).abs())
                .fold(0.0f32, f32::max);
            assert!(error < 0.02, "{m}/{k}/{n}: {error}");
            // Alternate order; each run submits a batch and then drains once.
            for round in 0..8 {
                for custom in if round % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] } {
                    B::sync(&device).unwrap();
                    let mut outputs = Vec::with_capacity(100);
                    let t = Instant::now();
                    for _ in 0..100 {
                        let out = match custom {
                            2 => gemv(x.clone().reshape([m, k]), w.clone()),
                            1 => x.clone().reshape([m, k]).matmul(w.clone()),
                            _ => module::linear(x.clone(), w.clone(), None).reshape([m, n]),
                        };
                        outputs.push(out);
                    }
                    B::sync(&device).unwrap();
                    println!(
                        "{}",
                        serde_json::json!({"m":m,"batch":batch,"tokens":tokens,"k":k,"n":n,"custom":custom,"round":round,"us":t.elapsed().as_secs_f64()*1e4,"max_abs":error})
                    );
                    std::hint::black_box(outputs);
                }
            }
        }
    }
}
