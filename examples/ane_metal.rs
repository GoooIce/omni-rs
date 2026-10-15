//! Verify generic Burn ANE execution with Metal FP16 storage and boolean readback.
use burn::{
    backend::{
        Ane, AneDevice, Metal,
        wgpu::{graphics, init_setup},
    },
    tensor::{FloatDType, Tensor, TensorData, activation},
};

fn main() {
    type B = Ane<Metal>;
    let device = AneDevice::<Metal>::default();
    init_setup::<graphics::Metal>(device.fallback_device(), Default::default());
    let x = Tensor::<B, 2>::from_data(
        TensorData::new(
            (0..128 * 512)
                .map(|i| if i / 512 % 2 == 0 { 0.125f32 } else { -0.125 })
                .collect::<Vec<_>>(),
            [128, 512],
        ),
        &device,
    )
    .cast(FloatDType::F16);
    let w = Tensor::<B, 2>::from_data(
        TensorData::new(vec![0.25f32; 512 * 512], [512, 512]),
        &device,
    )
    .cast(FloatDType::F16);
    let h = x.matmul(w);
    let mask = h.clone().lower_equal_elem(0.);
    let output = activation::relu(h)
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap();
    let mask = mask.into_data().convert::<bool>().to_vec::<bool>().unwrap();
    for (i, value) in output.iter().enumerate() {
        let negative = i / 512 % 2 != 0;
        assert_eq!(*value, if negative { 0. } else { 16. });
        assert_eq!(mask[i], negative);
    }
    let stats = device.stats();
    assert_eq!(stats.ane_graph_executions, 1);
    assert_eq!(stats.ane_compute_operations_planned, 1);
    // FP16 casts now remain inside the compiled ANE graph.
    assert_eq!(stats.fallback_operations, 0);
    println!("{}", serde_json::to_string_pretty(&stats).unwrap());
    device.clear_cache();
}
