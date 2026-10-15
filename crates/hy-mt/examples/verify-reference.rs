use omni_hy_mt::{Model, Options, Precision, Request, Translator};
use omni_r2t2::weights::Weights;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
};
#[derive(serde::Deserialize)]
struct Case {
    name: String,
    steps: Vec<Vec<u32>>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "metal")]
    type B = burn::backend::wgpu::Metal;
    #[cfg(all(not(feature = "metal"), feature = "webgpu"))]
    type B = burn::backend::wgpu::WebGpu;
    #[cfg(all(not(feature = "metal"), not(feature = "webgpu"), feature = "cpu"))]
    type B = burn::backend::NdArray;
    #[cfg(feature = "metal")]
    {
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
            &Default::default(),
            Default::default(),
        );
        if std::env::var_os("HY_MT_UNFUSED").is_some() {
            type Raw =
                burn::backend::wgpu::CubeBackend<burn::backend::wgpu::WgpuRuntime, f32, i32, u8>;
            return verify::<Raw>();
        }
    }
    verify::<B>()
}
fn verify<B: burn::tensor::backend::Backend>() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or("artifacts/hy-mt-reference".into()),
    );
    let precision = if std::env::args().nth(2).as_deref() == Some("f16") {
        Precision::F16
    } else {
        Precision::F32
    };
    let device = Default::default();
    let cancel = AtomicBool::new(false);
    let model = Model::<B>::from_directory(&path, precision, &device, &cancel, |_, _| {})?;
    let reference = Weights::open_named(&path, "reference")?;
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(path.join("references.json"))?)?;
    for case in cases {
        let mut history = model.empty_history();
        for (step, ids) in case.steps.iter().enumerate() {
            let actual = model
                .forward_tokens(ids, &mut history, &cancel)?
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();
            let expected = reference
                .tensor::<B, 2>(
                    &format!("{}.{step}", case.name),
                    [1, model.config().vocab_size],
                    Precision::F32,
                    &device,
                )?
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0_f32, f32::max);
            let argmax = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
            };
            assert_eq!(argmax(&actual), argmax(&expected));
            assert!(
                max_error
                    < if precision == Precision::F16 {
                        0.004
                    } else {
                        0.00005
                    },
                "{}.{step} max error {max_error}",
                case.name
            );
            println!(
                "{}",
                serde_json::json!({"case":case.name,"step":step,"max_abs_error":max_error})
            );
        }
    }
    let mut translator = Translator::new(model, 128);
    let mut request = Request {
        text: "t5 t6 t7 t8".into(),
        target_language: "English".into(),
        options: Options {
            temperature: 0.0,
            max_new_tokens: 8,
            max_context_tokens: 128,
            prefill_chunk: 8,
            ..Default::default()
        },
    };
    let cold = pollster::block_on(translator.translate(&request, &cancel, |_| {}))?;
    let warm = pollster::block_on(translator.translate(&request, &cancel, |_| {}))?;
    assert_eq!(cold.text, warm.text);
    assert!(warm.cached_tokens > 0);
    assert_eq!(cold.generated_tokens, warm.generated_tokens);
    request.text = "t5 t6 t7 t9".into();
    let branch = pollster::block_on(translator.translate(&request, &cancel, |_| {}))?;
    assert!(branch.cached_tokens > 0);
    request.options.use_cache = false;
    let uncached = pollster::block_on(translator.translate(&request, &cancel, |_| {}))?;
    assert_eq!(branch.text, uncached.text);
    assert_eq!(branch.generated_tokens, uncached.generated_tokens);
    let stats = translator.cache_stats();
    let interrupted = pollster::block_on(
        translator.translate(&request, &cancel, |_| cancel.store(true, Ordering::Release)),
    )?;
    assert_eq!(interrupted.finish_reason, "cancelled");
    assert_eq!(
        translator.cache_stats().resident_tokens,
        stats.resident_tokens
    );
    cancel.store(false, Ordering::Release);
    translator.clear_cache();
    assert_eq!(translator.cache_stats().resident_tokens, 0);
    println!(
        "{}",
        serde_json::json!({"radix_and_cancel":"passed","cold":cold,"warm":warm,"branch_cached_tokens":branch.cached_tokens})
    );
    Ok(())
}
