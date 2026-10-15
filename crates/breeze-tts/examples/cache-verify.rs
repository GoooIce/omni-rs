//! Fixed-token model comparison; attention rounding cannot change later inputs.
use omni_breeze_tts::{Model, Precision};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use burn::backend::wgpu::{Metal, graphics, init_setup};
    let device = Default::default();
    init_setup::<graphics::Metal>(&device, Default::default());
    let model = Model::<Metal>::load(
        std::path::Path::new("models/Breeze-TTS-2"),
        Precision::F16,
        &device,
        |_, _| {},
    )?;
    for (text, instruction) in [
        ("你好，欢迎使用本地语音合成。", ""),
        (
            "请带上更新后的项目报告，明天上午九点开会。",
            "用温和自然的语气朗读。",
        ),
    ] {
        let (ids, dense) =
            model.trace_conditioned_with_paging(text, instruction, None, 375, false)?;
        let (other, paged) =
            model.trace_conditioned_with_paging(text, instruction, None, 375, true)?;
        assert_eq!(ids, other);
        for ((name, shape, expected), (other, actual_shape, actual)) in dense.into_iter().zip(paged)
        {
            assert_eq!(name, other);
            assert_eq!(shape, actual_shape);
            let diff: f64 = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum();
            let norm: f64 = expected.iter().map(|x| (*x as f64).powi(2)).sum();
            let relative = (diff / norm.max(1e-20)).sqrt();
            assert!(
                actual.iter().all(|x| x.is_finite()) && relative < 0.01,
                "{name}: relative L2 {relative}"
            );
            println!(
                "{}",
                serde_json::json!({"stage":name,"instruction":instruction,"relative_l2":relative})
            );
        }
    }
    println!("{}", serde_json::json!({"passed":true}));
    Ok(())
}
