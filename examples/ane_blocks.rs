//! Reproducible, model-independent FFN / attention-block benchmark and parity check.
use burn::{
    backend::{
        Ane, AneDevice, AneOptions, AneRuntime, FallbackPolicy, Metal, NdArray,
        wgpu::{graphics, init_setup},
    },
    tensor::{Tensor, TensorData, backend::Backend, module},
};
use clap::Parser;
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "ane")]
    backend: String,
    #[arg(long, default_value = "coreml", value_parser = ["coreml", "direct"])]
    ane_runtime: String,
    #[arg(long)]
    strict: bool,
    #[arg(long)]
    trace: bool,
    #[arg(long, default_value = "block")]
    case: String,
    #[arg(long, default_value_t = 32)]
    rows: usize,
    #[arg(long, default_value_t = 1024)]
    width: usize,
    #[arg(long, default_value_t = 4096)]
    hidden: usize,
    #[arg(long, default_value_t = 16)]
    heads: usize,
    #[arg(long, default_value_t = 6)]
    steps: usize,
    #[arg(long, default_value_t = 512)]
    max_ops: usize,
    /// FP32-equivalent constant bytes per graph; independent of cache budgets.
    #[arg(long, default_value_t = 256)]
    max_constant_mib: usize,
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    /// Run real encoder/decoder parity against Metal using this checkpoint.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Compare cached and fresh real encoder outputs across streaming boundaries.
    #[arg(long, requires = "model")]
    encoder_cache_probe: bool,
    #[arg(long, default_value_t = 32)]
    row_bucket: usize,
    #[arg(long)]
    pad_single_row: bool,
    #[arg(long)]
    fuse_single_token: bool,
    #[arg(long, default_value_t = 128)]
    kv_bucket: usize,
    #[arg(long, default_value = "prefer-small")]
    program_cache_policy: burn::backend::ane::ProgramCachePolicy,
    #[arg(long, default_value_t = 4096)]
    weight_cache_mib: usize,
    /// Keep whole decoder blocks fused for an ablation of attention separation.
    #[arg(long)]
    no_separate_attention: bool,
    /// Number of single-token decoder calls after prefill for model parity/timing.
    #[arg(long, default_value_t = 2)]
    model_steps: usize,
    /// Save full FP32 output arrays for comparisons across backend options.
    #[arg(long)]
    dump_dir: Option<PathBuf>,
    /// Retain and compare all decoder layer outputs (changes fusion, not a timing run).
    #[arg(long)]
    layer_trace: bool,
    /// Inspect individual activations inside the first decoder layer.
    #[arg(long, conflicts_with = "layer_trace")]
    activation_trace: bool,
    /// Save all requested stages before failing the original numerical guards.
    #[arg(long)]
    collect_failures: bool,
}

struct Block<B: Backend> {
    q: Tensor<B, 2>,
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    out: Tensor<B, 2>,
    up: Tensor<B, 2>,
    down: Tensor<B, 2>,
}
impl<B: Backend> Block<B> {
    fn new(a: &Args, device: &B::Device) -> Self {
        let w = |rows: usize, cols: usize, phase: f32| {
            Tensor::from_data(
                TensorData::new(
                    (0..rows * cols)
                        .map(|i| {
                            ((i as f32 * 0.731 + phase).sin() + (i as f32 * 0.177).cos())
                                * (0.15 / (rows as f32).sqrt())
                        })
                        .collect::<Vec<_>>(),
                    [rows, cols],
                ),
                device,
            )
        };
        Self {
            q: w(a.width, a.width, 0.),
            k: w(a.width, a.width, 0.3),
            v: w(a.width, a.width, 0.6),
            out: w(a.width, a.width, 0.9),
            up: w(a.width, a.hidden, 1.2),
            down: w(a.hidden, a.width, 1.5),
        }
    }
    fn forward(&self, a: &Args, mut x: Tensor<B, 2>) -> Tensor<B, 2> {
        if a.case == "block" {
            let h = norm(x.clone());
            let split = |t: Tensor<B, 2>| {
                t.reshape([1, a.rows, a.heads, a.width / a.heads])
                    .swap_dims(1, 2)
            };
            let h = module::attention(
                split(h.clone().matmul(self.q.clone())),
                split(h.clone().matmul(self.k.clone())),
                split(h.matmul(self.v.clone())),
                None,
                None,
                Default::default(),
            );
            x = x + h
                .swap_dims(1, 2)
                .reshape([a.rows, a.width])
                .matmul(self.out.clone());
        }
        let h = norm(x.clone()).matmul(self.up.clone()) + 0.01;
        x + gelu(h).matmul(self.down.clone())
    }
}
fn norm<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    let h = x.clone() - x.mean_dim(1);
    h.clone() / (h.square().mean_dim(1) + 1e-5).sqrt()
}
fn gelu<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    // Same elementary formulation used to avoid Burn Metal's fused erf issue.
    let scaled = x.clone() / std::f64::consts::SQRT_2;
    let negative = scaled.clone().lower_elem(0.);
    let magnitude = scaled.abs();
    let t = (magnitude.clone() * 0.3275911 + 1.).recip();
    let polynomial = ((((t.clone() * 1.061405429 - 1.453152027) * t.clone() + 1.421413741)
        * t.clone()
        - 0.284496736)
        * t.clone()
        + 0.254829592)
        * t;
    let erf = (polynomial * (-magnitude.square()).exp()).neg() + 1.;
    let erf = erf.clone().mask_where(negative, -erf);
    x * 0.5 * (erf + 1.)
}

fn run<B: Backend>(a: &Args, device: &B::Device) -> serde_json::Value {
    let block = Block::<B>::new(a, device);
    let cpu = Default::default();
    let reference = Block::<NdArray>::new(a, &cpu);
    B::sync(device).unwrap();
    let mut times = Vec::new();
    let mut max_error = 0f32;
    let mut max_relative_rmse = 0f64;
    for step in 0..a.steps {
        let data = TensorData::new(
            (0..a.rows * a.width)
                .map(|i| (i as f32 * 0.03 + step as f32 * 0.1).sin())
                .collect::<Vec<_>>(),
            [a.rows, a.width],
        );
        let input = Tensor::<B, 2>::from_data(data.clone(), device);
        B::sync(device).unwrap();
        let start = Instant::now();
        let out = block
            .forward(a, input)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        times.push(start.elapsed().as_secs_f64() * 1000.);
        // Numerical comparison is outside the timed region.
        let expected = reference
            .forward(a, Tensor::from_data(data, &cpu))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let mut error = 0.;
        let mut magnitude = 0.;
        for (actual, expected) in out.iter().zip(&expected) {
            assert!(actual.is_finite());
            max_error = max_error.max((actual - expected).abs());
            error += f64::from(actual - expected).powi(2);
            magnitude += f64::from(*expected).powi(2);
        }
        max_relative_rmse = max_relative_rmse.max((error / magnitude.max(1e-20)).sqrt());
    }
    assert!(
        max_error < 0.015 && max_relative_rmse < 0.01,
        "max abs {max_error}, relative RMSE {max_relative_rmse}"
    );
    serde_json::json!({"backend": B::name(device), "case": a.case, "rows": a.rows, "width": a.width, "hidden": a.hidden,
        "times_ms_cold_then_warm": times, "max_absolute_error": max_error, "max_relative_rmse": max_relative_rmse})
}
fn main() {
    let a = Args::parse();
    if let Some(path) = &a.model {
        probe_model(&a, path);
        return;
    }
    assert!(
        a.steps >= 2
            && a.rows > 0
            && a.width > 0
            && a.hidden > 0
            && a.heads > 0
            && a.width % a.heads == 0
    );
    assert!(matches!(a.case.as_str(), "ffn" | "block"));
    let result = match a.backend.as_str() {
        "ane" => {
            let device = AneDevice::<Metal>::new(
                Default::default(),
                AneOptions {
                    trace: a.trace,
                    program_cache_policy: a.program_cache_policy,
                    runtime: if a.ane_runtime == "direct" {
                        AneRuntime::Direct
                    } else {
                        AneRuntime::CoreMl
                    },
                    fallback: if a.strict {
                        FallbackPolicy::Error
                    } else {
                        FallbackPolicy::Allow
                    },
                    max_graph_operations: a.max_ops,
                    max_constant_bytes: a.max_constant_mib.saturating_mul(1024 * 1024),
                    cache_dir: a.cache_dir.clone(),
                    row_bucket: a.row_bucket,
                    pad_single_row: a.pad_single_row,
                    fuse_single_token_attention: a.fuse_single_token,
                    kv_bucket: a.kv_bucket,
                    max_cached_weight_bytes: a.weight_cache_mib * 1024 * 1024,
                    separate_cached_attention: !a.no_separate_attention,
                    ..Default::default()
                },
            );
            init_setup::<graphics::Metal>(device.fallback_device(), Default::default());
            let mut result = run::<Ane<Metal>>(&a, &device);
            let stats = device.stats();
            // A CPU/Metal fallback-only run must never masquerade as an ANE benchmark.
            assert!(
                stats.ane_compute_operations_planned > 0 || stats.direct_ane_graph_executions > 0,
                "{stats:?}"
            );
            result["stats"] = serde_json::to_value(stats).unwrap();
            device.clear_cache();
            result
        }
        "metal" => {
            let device = Default::default();
            init_setup::<graphics::Metal>(&device, Default::default());
            run::<Metal>(&a, &device)
        }
        _ => panic!("backend must be ane or metal"),
    };
    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}

fn probe_model(a: &Args, path: &std::path::Path) {
    use omni_r2t2::{
        Model, Precision,
        audio::MelFrontend,
        model::{decoder::DecoderCache, encoder::EncoderCache},
    };
    type B = Ane<Metal>;
    let device = AneDevice::<Metal>::new(
        Default::default(),
        AneOptions {
            trace: a.trace,
            program_cache_policy: a.program_cache_policy,
            runtime: if a.ane_runtime == "direct" {
                AneRuntime::Direct
            } else {
                AneRuntime::CoreMl
            },
            fallback: if a.strict {
                FallbackPolicy::Error
            } else {
                FallbackPolicy::Allow
            },
            max_graph_operations: a.max_ops,
            max_constant_bytes: a.max_constant_mib.saturating_mul(1024 * 1024),
            row_bucket: a.row_bucket,
            pad_single_row: a.pad_single_row,
            fuse_single_token_attention: a.fuse_single_token,
            kv_bucket: a.kv_bucket,
            max_cached_weight_bytes: a.weight_cache_mib * 1024 * 1024,
            separate_cached_attention: !a.no_separate_attention,
            max_cached_bytes: 8 * 1024 * 1024 * 1024,
            cache_dir: a.cache_dir.clone(),
            ..Default::default()
        },
    );
    let gpu = device.fallback_device().clone();
    init_setup::<graphics::Metal>(&gpu, Default::default());
    eprintln!("Loading both Burn backends for numerical comparison...");
    let ane = Model::<B>::from_directory(path, Precision::F16, &device).unwrap();
    B::sync(&device).unwrap();
    let metal = Model::<Metal>::from_directory(path, Precision::F16, &gpu).unwrap();
    <Metal as Backend>::sync(&gpu).unwrap();
    if a.encoder_cache_probe {
        probe_encoder_cache(a, &ane, &metal, &device);
        return;
    }
    let initialization_fallbacks = device.stats().fallback_operations;
    let mut previous_stats = device.stats();
    let mut violations = Vec::new();
    let mut compare = |name: &str, actual: Vec<f32>, expected: Vec<f32>, elapsed_ms: f64| {
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual.iter().all(|v| v.is_finite()),
            "{name}: non-finite output"
        );
        let error: f64 = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum();
        let magnitude: f64 = expected.iter().map(|v| f64::from(*v).powi(2)).sum();
        let rmse = (error / magnitude.max(1e-20)).sqrt();
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let top = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        let stats = device.stats();
        let delta = serde_json::json!({
            "compiled_graphs":stats.compiled_graphs-previous_stats.compiled_graphs,
            "cache_hits":stats.cache_hits-previous_stats.cache_hits,
            "constant_bytes_read":stats.constant_bytes_read-previous_stats.constant_bytes_read,
            "weight_cache_hits":stats.weight_cache_hits-previous_stats.weight_cache_hits,
            "kv_bucketed_graph_executions":stats.kv_bucketed_graph_executions-previous_stats.kv_bucketed_graph_executions,
            "prediction_ms":stats.prediction_ms-previous_stats.prediction_ms,
            "graph_build_ms":stats.graph_build_ms-previous_stats.graph_build_ms,
            "compile_load_ms":stats.compile_load_ms-previous_stats.compile_load_ms,
        });
        let report = serde_json::json!({"stage":name,"elapsed_ms":elapsed_ms,"delta":delta,"relative_rmse":rmse,"max_absolute_error":max,"ane_top1":top(&actual),"metal_top1":top(&expected),"initialization_fallback_operations":initialization_fallbacks,"inference_fallback_operations":stats.fallback_operations-initialization_fallbacks,"stats":stats});
        previous_stats = stats;
        println!("{report}");
        if let Some(dir) = &a.dump_dir {
            std::fs::create_dir_all(dir).unwrap();
            for (backend, values) in [("ane", &actual), ("metal", &expected)] {
                let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(dir.join(format!("{name}-{backend}.f32")), bytes).unwrap();
            }
        }
        if a.collect_failures {
            if rmse >= 0.05 {
                violations.push(format!("{name}: relative RMSE {rmse}"));
            }
        } else {
            assert!(rmse < 0.05, "{name}: relative RMSE {rmse}");
        }
        if name.starts_with("decoder")
            && !name.contains("-layer-")
            && !name.contains("-activation-")
        {
            if a.collect_failures {
                if top(&actual) != top(&expected) {
                    violations.push(format!("{name}: different greedy token"));
                }
                return;
            }
            assert_eq!(
                top(&actual),
                top(&expected),
                "{name}: different greedy token"
            );
        }
    };
    if a.case != "decoder" {
        let mut reader =
            hound::WavReader::open("artifacts/Confucius4-R2T2-upstream/resources/test.wav")
                .unwrap();
        let samples: Vec<f32> = reader
            .samples::<i16>()
            .take(5120)
            .map(|v| f32::from(v.unwrap()) / 32768.)
            .collect();
        let mut frontend = MelFrontend::new();
        frontend.push(&samples).unwrap();
        let mel = frontend.features();
        let expected = metal
            .diagnostics()
            .encoder
            .forward(&mel, &mut EncoderCache::default(), false)
            .unwrap()
            .embeddings
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        eprintln!("Comparing encoder at {} frames", mel.frames);
        let start = Instant::now();
        let actual = ane
            .diagnostics()
            .encoder
            .forward(&mel, &mut EncoderCache::default(), false)
            .unwrap()
            .embeddings
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
        compare(
            "encoder",
            actual,
            expected,
            start.elapsed().as_secs_f64() * 1000.,
        );
    }
    if a.case != "encoder" {
        let layers = ane.config().thinker_config.text_config.num_hidden_layers;
        let mut ac = DecoderCache::new(layers);
        let mut mc = DecoderCache::new(layers);
        let calls = std::iter::once((
            "decoder-prefill".to_string(),
            vec![100, 101, 102, 103, 104, 105, 106, 107],
        ))
        .chain(
            (0..a.model_steps).map(|i| (format!("decoder-step-{}", i + 1), vec![108 + i as u32])),
        );
        for (stage, tokens) in calls {
            let data = metal
                .diagnostics()
                .decoder
                .embed(&tokens)
                .unwrap()
                .into_data();
            let mut metal_layers = Vec::new();
            let mut ane_layers = Vec::new();
            let mut metal_activations = Vec::new();
            let mut ane_activations = Vec::new();
            let metal_input = Tensor::from_data(data.clone(), &gpu);
            let expected = if a.activation_trace {
                metal.diagnostics().decoder.forward_with_activation_trace(
                    metal_input,
                    &mut mc,
                    &mut metal_activations,
                )
            } else if a.layer_trace {
                metal.diagnostics().decoder.forward_with_layer_outputs(
                    metal_input,
                    &mut mc,
                    &mut metal_layers,
                )
            } else {
                metal.diagnostics().decoder.forward(metal_input, &mut mc)
            }
            .unwrap()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
            eprintln!("Comparing {stage}");
            let start = Instant::now();
            let ane_input = Tensor::from_data(data, &device);
            let actual = if a.activation_trace {
                ane.diagnostics().decoder.forward_with_activation_trace(
                    ane_input,
                    &mut ac,
                    &mut ane_activations,
                )
            } else if a.layer_trace {
                ane.diagnostics().decoder.forward_with_layer_outputs(
                    ane_input,
                    &mut ac,
                    &mut ane_layers,
                )
            } else {
                ane.diagnostics().decoder.forward(ane_input, &mut ac)
            }
            .unwrap()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .unwrap();
            compare(
                &stage,
                actual,
                expected,
                start.elapsed().as_secs_f64() * 1000.,
            );
            for (i, (actual, expected)) in ane_layers.into_iter().zip(metal_layers).enumerate() {
                compare(
                    &format!("{stage}-layer-{:02}", i + 1),
                    actual.into_data().convert::<f32>().to_vec().unwrap(),
                    expected.into_data().convert::<f32>().to_vec().unwrap(),
                    0.,
                );
            }
            for ((name, actual), (expected_name, expected)) in
                ane_activations.into_iter().zip(metal_activations)
            {
                assert_eq!(name, expected_name);
                compare(
                    &format!("{stage}-activation-{name}"),
                    actual.into_data().convert::<f32>().to_vec().unwrap(),
                    expected.into_data().convert::<f32>().to_vec().unwrap(),
                    0.,
                );
            }
        }
    }
    device.clear_cache();
    assert!(
        violations.is_empty(),
        "numerical guard failures: {violations:?}"
    );
}

fn probe_encoder_cache(
    a: &Args,
    ane: &omni_r2t2::Model<Ane<Metal>>,
    metal: &omni_r2t2::Model<Metal>,
    device: &AneDevice<Metal>,
) {
    use omni_r2t2::{audio::MelFrontend, model::encoder::EncoderCache};

    fn read<B: Backend>(tensor: Tensor<B, 2>) -> Vec<f32> {
        tensor.into_data().convert::<f32>().to_vec().unwrap()
    }
    fn compare(actual: &[f32], expected: &[f32]) -> serde_json::Value {
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().chain(expected).all(|v| v.is_finite()));
        let error: f64 = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum();
        let energy: f64 = expected.iter().map(|x| f64::from(*x).powi(2)).sum();
        serde_json::json!({
            "relative_rmse": (error / energy.max(1e-20)).sqrt(),
            "max_absolute_error": actual.iter().zip(expected).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max),
            "bitwise_equal": actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()),
        })
    }

    let mut reader =
        hound::WavReader::open("artifacts/Confucius4-R2T2-upstream/resources/test.wav").unwrap();
    assert_eq!(reader.spec().sample_rate, 16000);
    assert_eq!(reader.spec().channels, 1);
    let pcm: Vec<f32> = reader
        .samples::<i16>()
        .map(|x| f32::from(x.unwrap()) / 32768.)
        .collect();
    assert!(!pcm.is_empty());
    let mut frontend = MelFrontend::new();
    let mut cached = EncoderCache::default();
    let mut samples = 0;
    let initialized = device.stats();
    let mut failures = Vec::new();
    let mut total_reused_blocks = 0;
    // The repeated 320-frame step guarantees a cache hit even if a new loud
    // sample invalidated earlier prefixes. Longer stages also cross 8 seconds.
    for (index, frames) in [32, 102, 160, 202, 320, 320, 802, 850, 1001]
        .into_iter()
        .enumerate()
    {
        let next_samples = frames * 160;
        let input: Vec<_> = (samples..next_samples)
            .map(|i| pcm[i % pcm.len()])
            .collect();
        frontend.push(&input).unwrap();
        samples = next_samples;
        let mel = frontend.features();
        assert_eq!(mel.frames, frames);
        eprintln!("Comparing cached encoder at {frames} frames");
        let output = ane
            .diagnostics()
            .encoder
            .forward(&mel, &mut cached, true)
            .unwrap();
        let reused_blocks = output.reused_convolution_blocks;
        let reused_tokens = output.reused_tokens;
        total_reused_blocks += reused_blocks;
        let actual = read(output.embeddings);
        let fresh = read(
            ane.diagnostics()
                .encoder
                .forward(&mel, &mut EncoderCache::default(), false)
                .unwrap()
                .embeddings,
        );
        let reference = read(
            metal
                .diagnostics()
                .encoder
                .forward(&mel, &mut EncoderCache::default(), false)
                .unwrap()
                .embeddings,
        );
        let fresh_comparison = compare(&actual, &fresh);
        let metal_comparison = compare(&actual, &reference);
        let fresh_metal_comparison = compare(&fresh, &reference);
        let stats = device.stats();
        let fallback: std::collections::BTreeMap<_, _> = stats
            .fallback_reasons
            .iter()
            .filter_map(|(reason, &count)| {
                let count = count
                    - initialized
                        .fallback_reasons
                        .get(reason)
                        .copied()
                        .unwrap_or(0);
                (count > 0).then(|| (reason.clone(), count))
            })
            .collect();
        let direct = stats.direct_ane_graph_executions > initialized.direct_ane_graph_executions
            && stats.direct_ane_graph_executions - initialized.direct_ane_graph_executions
                == stats.ane_graph_executions - initialized.ane_graph_executions
            && stats.compilation_failures == initialized.compilation_failures
            && stats.failure_cache_hits == initialized.failure_cache_hits
            && fallback.keys().all(|reason| {
                matches!(
                    reason.as_str(),
                    "input token concatenation"
                        | "subgraph has no matrix multiplication or convolution"
                        | "unsupported operation: BaseFloat::Select"
                        | "unsupported operation: NumericFloat::ArgMax"
                )
            });
        let stage = format!("encoder-cache-{index}-{frames}");
        if !direct
            || fresh_comparison["relative_rmse"].as_f64().unwrap() >= 0.05
            || metal_comparison["relative_rmse"].as_f64().unwrap() >= 0.05
        {
            failures.push(stage.clone());
        }
        println!(
            "{}",
            serde_json::json!({
                "stage": stage, "frames": frames, "numerical_only": true,
                "reused_convolution_blocks": reused_blocks, "reused_tokens": reused_tokens,
                "cached_vs_fresh_ane": fresh_comparison, "cached_vs_metal": metal_comparison,
            "fresh_ane_vs_metal": fresh_metal_comparison,
                "inference_fallback_reasons": fallback, "direct_ane_pass": direct, "stats": stats,
            })
        );
        if let Some(dir) = &a.dump_dir {
            std::fs::create_dir_all(dir).unwrap();
            for (name, values) in [
                ("cached-ane", &actual),
                ("fresh-ane", &fresh),
                ("metal", &reference),
            ] {
                let bytes: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(dir.join(format!("{stage}-{name}.f32")), bytes).unwrap();
            }
        }
        if !a.collect_failures {
            assert!(failures.is_empty(), "{failures:?}");
        }
    }
    assert!(
        total_reused_blocks > 0,
        "the probe did not exercise convolution reuse"
    );
    assert!(failures.is_empty(), "{failures:?}");
}
