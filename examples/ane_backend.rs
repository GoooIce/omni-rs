//! A normal Burn model on the ANE backend. No checkpoint or export tool is needed.
use burn::{
    backend::{Ane, AneDevice, AneOptions, FallbackPolicy, NdArray},
    module::Param,
    nn::Linear,
    tensor::{Tensor, TensorData, activation, backend::Backend},
};

fn network<B: Backend>(device: &B::Device) -> Linear<B> {
    Linear {
        weight: Param::from_tensor(Tensor::from_data(
            TensorData::new(
                (0..512 * 512)
                    .map(|i| (i as f32 * 0.031).cos() * 0.025)
                    .collect::<Vec<_>>(),
                [512, 512],
            ),
            device,
        )),
        bias: Some(Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![0.01; 512], [512]),
            device,
        ))),
    }
}
fn main() {
    let device = AneDevice::new(
        Default::default(),
        AneOptions {
            fallback: FallbackPolicy::Error,
            ..Default::default()
        },
    );
    let cpu = Default::default();
    let model = network::<Ane>(&device);
    let reference = network::<NdArray>(&cpu);
    let mut times = Vec::new();
    let mut max_error = 0f32;
    for step in 0..3 {
        let data = TensorData::new(
            (0..128 * 512)
                .map(|i| (i as f32 * 0.017 + step as f32 * 0.2).sin())
                .collect::<Vec<_>>(),
            [1, 128, 512],
        );
        let start = std::time::Instant::now();
        let output =
            activation::relu(model.forward(Tensor::<Ane, 3>::from_data(data.clone(), &device)))
                .into_data()
                .to_vec::<f32>()
                .unwrap();
        times.push(start.elapsed().as_secs_f64() * 1000.);
        let expected =
            activation::relu(reference.forward(Tensor::<NdArray, 3>::from_data(data, &cpu)))
                .into_data()
                .to_vec::<f32>()
                .unwrap();
        for (a, b) in output.iter().zip(&expected) {
            assert!(a.is_finite());
            max_error = max_error.max((a - b).abs());
        }
    }
    let stats = device.stats();
    assert!(max_error < 0.008);
    assert_eq!(stats.compiled_graphs, 1);
    assert_eq!(stats.cache_hits, 2);
    assert_eq!(stats.ane_graph_executions, 3);
    assert!(stats.ane_operations_planned > 0);
    assert_eq!(stats.ane_compute_operations_planned, 3);
    assert_eq!(stats.fallback_operations, 0);
    println!("{}",serde_json::to_string_pretty(&serde_json::json!({"backend":Ane::name(&device),"max_absolute_error":max_error,"inference_ms_cold_then_warm":times,"stats":stats})).unwrap());
    device.clear_cache();
}
