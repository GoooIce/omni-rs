//! Real-model serial/batch comparison, staggered admission, cancellation and
//! retry. Emits JSON suitable for archiving with the model revision and device.
use clap::Parser;
use omni_hy_mt::{BatchConfig, Model, Options, Precision, Request, Translator};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "models/Hy-MT2-7B")]
    model: std::path::PathBuf,
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    #[cfg(feature = "metal")]
    type B = burn::backend::wgpu::Metal;
    #[cfg(all(not(feature = "metal"), feature = "webgpu"))]
    type B = burn::backend::wgpu::WebGpu;
    #[cfg(all(not(feature = "metal"), not(feature = "webgpu"), feature = "cpu"))]
    type B = burn::backend::NdArray;
    #[cfg(feature = "metal")]
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
        &Default::default(),
        Default::default(),
    );
    let precision =
        if cfg!(feature = "cpu") && !cfg!(feature = "metal") && !cfg!(feature = "webgpu") {
            Precision::F32
        } else {
            Precision::F16
        };
    let stop = AtomicBool::new(false);
    let load = Instant::now();
    let model = Model::<B>::from_directory(
        &args.model,
        precision,
        &Default::default(),
        &stop,
        |_, _| {},
    )?;
    let load_ms = load.elapsed().as_secs_f64() * 1000.0;
    let mut translator = Translator::with_config(model, 4096, BatchConfig::default())?;
    let requests:Vec<_>=[
        ("分页注意力按需分配缓存页面，多个翻译请求可以共享已经计算过的前缀。","English"),
        ("Continuous batching admits new requests between decoding steps and releases completed requests immediately.","Chinese"),
        ("请保留原始文本，并将译文作为单独的结果显示。取消一个请求不会影响其他请求。","English"),
        ("The meeting starts tomorrow morning at nine. Please bring the updated project report.","Chinese"),
    ].into_iter().map(|(text,language)|Request{text:text.into(),target_language:language.into(),options:Options{temperature:0.0,max_new_tokens:args.max_new_tokens,use_cache:false,..Default::default()}}).collect();
    // Warm every batch width, including 2/3 after early completions. Otherwise
    // the shrinking batch pays first-use compilation in the timed interval.
    // Prefix reuse is disabled in both timings.
    eprintln!("warming batch widths 1..4");
    for width in 1..=requests.len() {
        for r in &requests[..width] {
            let mut r = r.clone();
            r.options.max_new_tokens = 2;
            translator
                .try_submit(&r, Arc::new(AtomicBool::new(false)))?
                .unwrap();
        }
        while !translator.is_idle() {
            for (_, r) in pollster::block_on(translator.step(|_, _| {})) {
                r?;
            }
        }
    }
    eprintln!("timing serial and concurrent translations");
    let started = Instant::now();
    let mut serial = Vec::new();
    for r in &requests {
        serial.push(pollster::block_on(translator.translate(r, &stop, |_| {}))?);
    }
    let serial_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let ids: Vec<_> = requests
        .iter()
        .map(|r| {
            translator
                .try_submit(r, Arc::new(AtomicBool::new(false)))
                .unwrap()
                .unwrap()
        })
        .collect();
    let mut concurrent = BTreeMap::new();
    while !translator.is_idle() {
        for (id, result) in pollster::block_on(translator.step(|_, _| {})) {
            concurrent.insert(id, result?);
        }
    }
    let batch_ms = started.elapsed().as_secs_f64() * 1000.0;
    for (expected, id) in serial.iter().zip(&ids) {
        let actual = &concurrent[id];
        assert_eq!(
            expected.text, actual.text,
            "greedy translation changed in batch"
        );
        assert_eq!(expected.generated_tokens, actual.generated_tokens);
        assert_eq!(expected.finish_reason, actual.finish_reason);
    }
    // A long prefill joins a live decode; another member is cancelled after a
    // streamed token. Long context exercises multiple softmax partitions.
    let a_stop = Arc::new(AtomicBool::new(false));
    let a = translator
        .try_submit(&requests[0], a_stop.clone())?
        .unwrap();
    assert!(pollster::block_on(translator.step(|_, _| {})).is_empty());
    let mut long = requests[1].clone();
    long.text = long.text.repeat(16);
    long.options.max_new_tokens = 16;
    let b = translator
        .try_submit(&long, Arc::new(AtomicBool::new(false)))?
        .unwrap();
    let c = translator
        .try_submit(&requests[2], Arc::new(AtomicBool::new(false)))?
        .unwrap();
    let mut staggered = BTreeMap::new();
    while !translator.is_idle() {
        for (id, result) in pollster::block_on(translator.step(|id, event| {
            if id == a && event.generated_tokens >= 2 {
                a_stop.store(true, Ordering::Release);
            }
        })) {
            staggered.insert(id, result?);
        }
    }
    assert_eq!(staggered[&a].finish_reason, "cancelled");
    assert_eq!(staggered[&c].text, serial[2].text);
    let long_single = pollster::block_on(translator.translate(&long, &stop, |_| {}))?;
    assert_eq!(staggered[&b].text, long_single.text);
    let mut cached = requests[0].clone();
    cached.options.use_cache = true;
    let cold = pollster::block_on(translator.translate(&cached, &stop, |_| {}))?;
    let warm = pollster::block_on(translator.translate(&cached, &stop, |_| {}))?;
    assert_eq!(cold.text, warm.text);
    assert!(warm.cached_tokens > 0);
    let stats = translator.batch_stats();
    translator.clear_cache();
    assert_eq!(translator.batch_stats().used_pages, 0);
    let tokens: usize = serial.iter().map(|r| r.generated_tokens).sum();
    println!(
        "{}",
        serde_json::json!({"passed":true,"model_revision":omni_hy_mt::MODEL_REVISION,"precision":precision,"load_ms":load_ms,"serial_ms":serial_ms,"batch_ms":batch_ms,"speedup":serial_ms/batch_ms,"output_tokens":tokens,"serial_tokens_per_second":tokens as f64*1000.0/serial_ms,"batch_tokens_per_second":tokens as f64*1000.0/batch_ms,"serial":serial,"concurrent":concurrent,"staggered":staggered,"cached_tokens":warm.cached_tokens,"batch_stats":stats,"pages_after_clear":translator.batch_stats().used_pages})
    );
    Ok(())
}
