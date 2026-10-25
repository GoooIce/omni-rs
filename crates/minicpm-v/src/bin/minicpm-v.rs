use clap::{Parser, Subcommand};
use omni_minicpm_v::{
    chat::{self, Message, Role},
    config::ModelDirectory,
    doctor, generate,
    latent::LatentCache,
    model::{DecodeState, TextModel},
    multimodal::{self, Prompt},
};
use std::{
    io::Write,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

#[derive(Parser)]
#[command(version, about = "MiniCPM-V-4.7 local multimodal inference using Burn")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Verify the compiled backend and a downloaded model directory.
    Doctor {
        #[arg(long, default_value = "models/MiniCPM-V-4.7-1B")]
        model: PathBuf,
    },
    /// Generate text from a prompt.
    Generate(GenerateOptions),
    /// Compare dense KV decode with a fitted latent cache on the 1B checkpoint.
    LatentBench {
        #[arg(long, default_value = "models/MiniCPM-V-4.7-1B")]
        model: PathBuf,
        #[arg(long, default_value = "artifacts/minicpm-latent/factors.safetensors")]
        factors: PathBuf,
        /// Comma-separated latent ranks. Each must be at most the rank stored in the file.
        #[arg(long, default_value = "128,256")]
        ranks: String,
        #[arg(long, default_value_t = 32)]
        new_tokens: usize,
        /// Repeated-token prefill length. Zero skips it.
        #[arg(long, default_value_t = 0)]
        context: usize,
        /// Comma-separated prefill lengths. When set, only these prefills are timed.
        #[arg(long)]
        lengths: Option<String>,
        /// Tokens per prefill chunk. Bounds the attention matrix.
        #[arg(long, default_value_t = 64)]
        chunk: usize,
    },
}

#[derive(clap::Args)]
struct GenerateOptions {
    #[arg(long, default_value = "models/MiniCPM-V-4.7-1B")]
    model: PathBuf,
    #[arg(long)]
    prompt: String,
    /// One image (PNG/JPEG/WebP) for multimodal input.
    #[arg(long)]
    image: Option<PathBuf>,
    #[arg(long)]
    system: Option<String>,
    #[arg(long)]
    greedy: bool,
    #[arg(long, default_value_t = 512)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    #[arg(long, default_value_t = 20)]
    top_k: usize,
    #[arg(long, default_value_t = 0.6)]
    top_p: f32,
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// f16 on GPU, f32 for the CPU reference path.
    #[arg(long, default_value = "f16")]
    precision: String,
    #[arg(long)]
    json: bool,
    /// DSpark draft directory. Greedy text decoding then verifies a block of draft tokens.
    #[arg(long)]
    dspark: Option<PathBuf>,
    /// Verify DSpark blocks with the single-token decode kernels (bit-identical
    /// to plain decode, one block ≈ its tokens decoded one by one). The default
    /// batched verify matches vLLM-style scoring instead.
    #[arg(long, default_value_t = false)]
    dspark_match_decode: bool,
    /// Close the think block. Applies to plain decode and DSpark together.
    #[arg(long)]
    no_think: bool,
    /// Keep the think block open. Applies to plain decode and DSpark together.
    #[arg(long)]
    think: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Args { command } = Args::parse();
    match command {
        Command::Doctor { model } => doctor_command(model),
        Command::Generate(options) => generate_command(&options),
        Command::LatentBench {
            model,
            factors,
            ranks,
            new_tokens,
            context,
            lengths,
            chunk,
        } => latent_bench(
            model,
            factors,
            &ranks,
            new_tokens,
            context,
            lengths.as_deref(),
            chunk,
        ),
    }
}

fn doctor_command(model: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    check_backend()?;
    println!("Checking {}", model.display());
    let report = doctor::check(&model);
    for problem in &report.problems {
        println!("{problem}");
    }
    if !report.is_ok() {
        std::process::exit(1);
    }
    println!(
        "Model OK: {} files verified, {:.2} GiB hashed",
        report.verified_files,
        report.hashed_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}

fn generate_command(options: &GenerateOptions) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "metal")]
    type B = burn::backend::wgpu::Metal;
    #[cfg(all(not(feature = "metal"), feature = "webgpu"))]
    type B = burn::backend::wgpu::WebGpu;
    #[cfg(all(not(feature = "metal"), not(feature = "webgpu"), feature = "cpu"))]
    type B = burn::backend::NdArray;

    #[cfg(feature = "metal")]
    {
        use burn::backend::wgpu::{WgpuDevice, graphics, init_setup};
        let device = WgpuDevice::default();
        let setup = init_setup::<graphics::Metal>(&device, Default::default());
        eprintln!("Metal: {:?}", setup.adapter.get_info());
        run_generate::<B>(options, &device)
    }
    #[cfg(all(not(feature = "metal"), feature = "cpu"))]
    {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        run_generate::<B>(options, &device)
    }
}

fn run_generate<B: burn::tensor::backend::Backend + omni_minicpm_v::quant::ExpertResidence>(
    options: &GenerateOptions,
    device: &B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let directory = ModelDirectory::new(&options.model);
    let config = directory.config()?;
    let tokenizer = directory.tokenizer()?;
    let weights = omni_r2t2::weights::Weights::open(&options.model)?;

    let start = Instant::now();
    let precision = match options.precision.as_str() {
        "f16" => omni_r2t2::Precision::F16,
        "f32" => omni_r2t2::Precision::F32,
        other => return Err(format!("precision must be f16 or f32, got {other}").into()),
    };
    let model = TextModel::<B>::load(&weights, &config, precision, device)?;
    if options.think && options.no_think {
        return Err("--think and --no-think cannot be set together".into());
    }
    let thinking = if options.no_think { Some(false) } else { None };

    let prompt = match &options.image {
        Some(image) => {
            let vision = multimodal::Multimodal::<B>::load(&weights, &config, precision, device)?;
            vision
                .build_prompt(multimodal::Build::<B> {
                    tokenizer: &tokenizer,
                    model: &model,
                    config: &config,
                    model_dir: &options.model,
                    image,
                    prompt: &options.prompt,
                    enable_thinking: thinking,
                    device,
                    precision,
                })
                .map_err(|e| e.to_string())?
        }
        None => {
            let mut messages = Vec::new();
            if let Some(system) = &options.system {
                messages.push(Message::text(Role::System, system.clone()));
            }
            messages.push(Message::text(Role::User, options.prompt.clone()));
            let rendered = chat::render(&messages, true, thinking)?;
            Prompt::text(tokenizer.encode(&rendered))
        }
    };
    eprintln!(
        "Loading {} ({} prompt tokens)",
        options.model.display(),
        prompt.ids.len()
    );
    B::sync(device).map_err(|e| format!("weight initialization failed: {e:?}"))?;
    eprintln!("Weights loaded in {:.2}s", start.elapsed().as_secs_f32());

    let cancel = Arc::new(AtomicBool::new(false));
    let stop = cancel.clone();
    ctrlc::set_handler(move || stop.store(true, Ordering::Release))?;

    let generate_options = generate::Options {
        max_new_tokens: options.max_new_tokens,
        temperature: if options.greedy {
            0.0
        } else {
            options.temperature
        },
        top_k: options.top_k,
        top_p: options.top_p,
        repetition_penalty: options.repeat_penalty,
        presence_penalty: 0.0,
        seed: options.seed,
    };
    let json = options.json;
    let mut total_text = String::new();
    let mut emit = |token: generate::GeneratedToken| {
        total_text.push_str(&token.delta);
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "delta": token.delta,
                    "text": total_text,
                    "token_id": token.token,
                    "step_ms": token.step_ms,
                })
            );
        } else {
            print!("{}", token.delta);
            let _ = std::io::stdout().flush();
        }
        Ok(())
    };
    let decode = |ids: &[u32]| tokenizer.decode(ids);
    let outcome = if let Some(path) = &options.dspark {
        if generate_options.temperature != 0.0 {
            return Err("DSpark in this build verifies greedy decoding".into());
        }
        if config.text_config.num_hidden_layers <= 35 {
            return Err("DSpark needs the 40-layer MiniCPM-V-4.7-35B text tower".into());
        }
        let draft = omni_minicpm_v::dspark::Draft::load(path, precision, device)?;
        let numerics = if options.dspark_match_decode {
            omni_minicpm_v::model::VerifyNumerics::MatchingDecode
        } else {
            omni_minicpm_v::model::VerifyNumerics::Batched
        };
        let (outcome, spec) = omni_minicpm_v::dspark::generate(
            &model,
            &draft,
            &prompt,
            generate_options,
            &directory.eos_ids(),
            numerics,
            &cancel,
            &mut emit,
            decode,
        )?;
        eprintln!(
            "dspark cycles {} output {} acceptance_length {:.3}",
            spec.cycles,
            spec.output_tokens,
            spec.acceptance_length()
        );
        outcome
    } else {
        generate::generate(
            &model,
            &prompt,
            generate_options,
            &directory.eos_ids(),
            &cancel,
            &mut emit,
            decode,
        )?
    };
    if !json {
        println!();
    }
    eprintln!(
        "{:.0} ms; {} tokens; finish_reason: {}",
        outcome.compute_ms,
        outcome.tokens.len(),
        outcome.finish_reason.as_str()
    );
    if json {
        println!(
            "{}",
            serde_json::json!({
                "event": "done",
                "text": outcome.text,
                "tokens": outcome.tokens.len(),
                "finish_reason": outcome.finish_reason.as_str(),
                "compute_ms": outcome.compute_ms,
                "first_token_ms": outcome.first_token_ms,
            })
        );
    }
    Ok(())
}

fn latent_bench(
    model_dir: PathBuf,
    factors_path: PathBuf,
    ranks: &str,
    new_tokens: usize,
    context: usize,
    lengths: Option<&str>,
    chunk: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        use burn::backend::wgpu::{WgpuDevice, graphics, init_setup};
        let device = WgpuDevice::default();
        let setup = init_setup::<graphics::Metal>(&device, Default::default());
        eprintln!("Metal: {:?}", setup.adapter.get_info());
        run_latent_bench::<burn::backend::wgpu::Metal>(
            &model_dir,
            &factors_path,
            ranks,
            new_tokens,
            context,
            lengths,
            chunk,
            &device,
        )
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        let _ = (
            model_dir,
            factors_path,
            ranks,
            new_tokens,
            context,
            lengths,
            chunk,
        );
        Err("latent-bench measures the Metal path".into())
    }
}

fn run_latent_bench<B: burn::tensor::backend::Backend + omni_minicpm_v::quant::ExpertResidence>(
    model_dir: &PathBuf,
    factors_path: &PathBuf,
    ranks: &str,
    new_tokens: usize,
    context: usize,
    lengths: Option<&str>,
    chunk: usize,
    device: &B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    use burn::tensor::FloatDType;
    use omni_minicpm_v::latent::LatentFactors;

    let directory = ModelDirectory::new(model_dir);
    let config = directory.config()?;
    let text = &config.text_config;
    let tokenizer = directory.tokenizer()?;
    let weights = omni_r2t2::weights::Weights::open(model_dir)?;
    eprintln!("loading {}", model_dir.display());
    let model = TextModel::<B>::load(&weights, &config, omni_r2t2::Precision::F16, device)?;
    B::sync(device).map_err(|err| format!("weight initialization failed: {err:?}"))?;
    let bytes = std::fs::read(factors_path).map_err(|err| {
        format!(
            "read {}: {err}. Fit them with .venv-minicpm/bin/python scripts/fit_minicpm_latent.py",
            factors_path.display()
        )
    })?;
    let layers: Vec<usize> = text
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, kind)| *kind == "full_attention")
        .map(|(index, _)| index)
        .collect();
    let rope_dim =
        (text.head_dim as f64 * text.rope_parameters.partial_rotary_factor) as usize & !1;
    let max_rank = LatentFactors::<B>::stored_rank(&bytes, layers[0])?;
    let factors = LatentFactors::<B>::load(
        &bytes,
        &layers,
        text.hidden_size,
        text.num_key_value_heads,
        text.head_dim,
        rope_dim,
        max_rank,
        FloatDType::F16,
        device,
    )?;
    let ranks: Vec<usize> = ranks
        .split(',')
        .map(|rank| rank.trim().parse())
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(lengths) = lengths {
        return time_lengths::<B>(
            &model,
            device,
            &tokenizer,
            &factors,
            &layers,
            text.num_key_value_heads,
            rope_dim,
            &ranks,
            lengths,
            chunk,
            factors_path,
        );
    }
    let short = chat::render(
        &[Message::text(Role::User, "用三句话介绍杭州西湖。")],
        true,
        None,
    )?;
    let short_ids = tokenizer.encode(&short);
    let seed = tokenizer.encode("钱塘江从杭州湾入海。西湖在城西。");
    let mut long_ids = Vec::with_capacity(context);
    while long_ids.len() < context {
        long_ids.extend_from_slice(&seed);
    }
    long_ids.truncate(context);
    let eos = directory.eos_ids();
    let dense_bytes = layers.len() * text.num_key_value_heads * text.head_dim * 2 * 2;
    let mut report = Vec::new();

    for rank in ranks {
        let narrowed: Vec<_> = factors
            .iter()
            .map(|factor| factor.truncated(rank))
            .collect();
        let latent_bytes = layers.len()
            * ((rank + text.num_key_value_heads * rope_dim) * 2
                + text.num_key_value_heads * std::mem::size_of::<f32>());
        eprintln!(
            "rank {rank}: {} prompt tokens, {new_tokens} new tokens",
            short_ids.len()
        );
        let baseline = measure(
            &model,
            device,
            &short_ids,
            new_tokens,
            &eos,
            |state, ids| model.forward(ids, None, None, state),
        )?;
        let mut short_caches = fresh_caches::<B>(
            layers.len(),
            rank,
            text.num_key_value_heads,
            rope_dim,
            device,
        );
        let latent = measure(
            &model,
            device,
            &short_ids,
            new_tokens,
            &eos,
            |state, ids| {
                if state.length == 0 {
                    for cache in &mut short_caches {
                        *cache = LatentCache::new(rank, text.num_key_value_heads, rope_dim);
                    }
                }
                model.forward_latent(ids, None, None, state, &narrowed, &mut short_caches)
            },
        )?;
        let matched = baseline
            .tokens
            .iter()
            .zip(&latent.tokens)
            .take_while(|(left, right)| left == right)
            .count();
        let logit_gap = max_abs_diff(&baseline.prefill_logits, &latent.prefill_logits);
        println!(
            "rank {rank} short: matched {matched}/{} tokens, prefill {:.1} -> {:.1} ms, decode {:.2} -> {:.2} ms/token, kv {dense_bytes} -> {latent_bytes} B/token, logit gap {logit_gap:.4}",
            baseline.tokens.len().min(latent.tokens.len()),
            baseline.prefill_ms,
            latent.prefill_ms,
            baseline.decode_ms_per_token(),
            latent.decode_ms_per_token(),
        );
        println!(
            "  baseline: {}",
            tokenizer.decode(&baseline.tokens).replace('\n', " ")
        );
        println!(
            "  latent:   {}",
            tokenizer.decode(&latent.tokens).replace('\n', " ")
        );

        let mut entry = serde_json::json!({
            "rank": rank,
            "kv_bytes_per_token": {"dense": dense_bytes, "latent": latent_bytes},
            "short": {
                "prompt_tokens": short_ids.len(),
                "matched_tokens": matched,
                "baseline_tokens": baseline.tokens,
                "latent_tokens": latent.tokens,
                "baseline_text": tokenizer.decode(&baseline.tokens),
                "latent_text": tokenizer.decode(&latent.tokens),
                "prefill_ms": {"dense": baseline.prefill_ms, "latent": latent.prefill_ms},
                "decode_ms_per_token": {
                    "dense": baseline.decode_ms_per_token(),
                    "latent": latent.decode_ms_per_token(),
                },
                "prefill_logit_abs_max": logit_gap,
            },
        });
        if context > 0 {
            eprintln!("rank {rank}: prefill {context} tokens, then 8 decode steps");
            let long_base = measure(&model, device, &long_ids, 8, &eos, |state, ids| {
                model.forward(ids, None, None, state)
            })?;
            let mut long_caches = fresh_caches::<B>(
                layers.len(),
                rank,
                text.num_key_value_heads,
                rope_dim,
                device,
            );
            let long_latent = measure(&model, device, &long_ids, 8, &eos, |state, ids| {
                if state.length == 0 {
                    for cache in &mut long_caches {
                        *cache = LatentCache::new(rank, text.num_key_value_heads, rope_dim);
                    }
                }
                model.forward_latent(ids, None, None, state, &narrowed, &mut long_caches)
            })?;
            let long_gap = max_abs_diff(&long_base.prefill_logits, &long_latent.prefill_logits);
            let first_ok = long_base.tokens.first() == long_latent.tokens.first();
            println!(
                "rank {rank} context {context}: prefill {:.0} -> {:.0} ms ({:.2}x), decode {:.2} -> {:.2} ms/token ({:.2}x), first token {first_ok}, logit gap {long_gap:.4}",
                long_base.prefill_ms,
                long_latent.prefill_ms,
                long_base.prefill_ms / long_latent.prefill_ms.max(0.001),
                long_base.decode_ms_per_token(),
                long_latent.decode_ms_per_token(),
                long_base.decode_ms_per_token() / long_latent.decode_ms_per_token().max(0.001),
            );
            entry["long"] = serde_json::json!({
                "context": context,
                "prefill_ms": {"dense": long_base.prefill_ms, "latent": long_latent.prefill_ms},
                "decode_ms_per_token": {
                    "dense": long_base.decode_ms_per_token(),
                    "latent": long_latent.decode_ms_per_token(),
                },
                "first_token_match": first_ok,
                "prefill_logit_abs_max": long_gap,
            });
        }
        report.push(entry);
    }
    let path = factors_path.with_file_name("bench.json");
    std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

fn time_lengths<B: burn::tensor::backend::Backend>(
    model: &TextModel<B>,
    device: &B::Device,
    tokenizer: &omni_minicpm_v::tokenizer::MiniCpmTokenizer,
    factors: &[omni_minicpm_v::latent::LatentFactors<B>],
    layers: &[usize],
    kv_heads: usize,
    rope_dim: usize,
    ranks: &[usize],
    lengths: &str,
    chunk: usize,
    factors_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let lengths: Vec<usize> = lengths
        .split(',')
        .map(|item| item.trim().parse())
        .collect::<Result<Vec<_>, _>>()?;
    let seed = tokenizer.encode("钱塘江从杭州湾入海。西湖在城西。");
    let dense_bytes = layers.len() * kv_heads * 256 * 2 * 2;
    let mut report = Vec::new();
    for length in lengths {
        let mut ids = Vec::with_capacity(length);
        while ids.len() < length {
            ids.extend_from_slice(&seed);
        }
        ids.truncate(length);
        for &rank in ranks {
            let narrowed: Vec<_> = factors
                .iter()
                .map(|factor| factor.truncated(rank))
                .collect();
            let latent_bytes = layers.len()
                * ((rank + kv_heads * rope_dim) * 2 + kv_heads * std::mem::size_of::<f32>());
            eprintln!("dense prefill {length} tokens, chunk {chunk}");
            let dense = time_prefill(model, device, &ids, chunk, "dense", |state, piece| {
                model.forward(piece, None, None, state)
            })?;
            eprintln!("latent rank {rank} prefill {length} tokens, chunk {chunk}");
            let mut caches = fresh_caches::<B>(layers.len(), rank, kv_heads, rope_dim, device);
            let latent = time_prefill(model, device, &ids, chunk, "latent", |state, piece| {
                if state.length == 0 {
                    for cache in &mut caches {
                        *cache = LatentCache::new(rank, kv_heads, rope_dim);
                    }
                }
                model.forward_latent(piece, None, None, state, &narrowed, &mut caches)
            })?;
            let gap = max_abs_diff(&dense.logits, &latent.logits);
            println!(
                "length {length} rank {rank}: prefill {:.1}s -> {:.1}s ({:.2}x), decode {:.1} -> {:.1} ms, kv {:.2} -> {:.2} GiB, logit gap {gap:.4}",
                dense.prefill_ms / 1000.0,
                latent.prefill_ms / 1000.0,
                dense.prefill_ms / latent.prefill_ms.max(0.001),
                dense.decode_ms,
                latent.decode_ms,
                dense_bytes as f64 * length as f64 / 1024.0 / 1024.0 / 1024.0,
                latent_bytes as f64 * length as f64 / 1024.0 / 1024.0 / 1024.0,
            );
            report.push(serde_json::json!({
                "length": length,
                "rank": rank,
                "chunk": chunk,
                "prefill_ms": {"dense": dense.prefill_ms, "latent": latent.prefill_ms},
                "decode_ms": {"dense": dense.decode_ms, "latent": latent.decode_ms},
                "kv_gib": {
                    "dense": dense_bytes as f64 * length as f64 / 1024.0 / 1024.0 / 1024.0,
                    "latent": latent_bytes as f64 * length as f64 / 1024.0 / 1024.0 / 1024.0,
                },
                "final_logit_abs_max": gap,
            }));
        }
    }
    let path = factors_path.with_file_name("long-prefill.json");
    std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

struct PrefillTiming {
    prefill_ms: f64,
    decode_ms: f64,
    logits: Vec<f32>,
}

fn time_prefill<B, F>(
    model: &TextModel<B>,
    device: &B::Device,
    ids: &[u32],
    chunk: usize,
    label: &str,
    mut step: F,
) -> Result<PrefillTiming, Box<dyn std::error::Error>>
where
    B: burn::tensor::backend::Backend,
    F: FnMut(&mut DecodeState<B>, &[u32]) -> omni_r2t2::Result<burn::tensor::Tensor<B, 2>>,
{
    let warm_len = chunk.min(ids.len());
    {
        let mut state = model.new_state(device);
        let logits = step(&mut state, &ids[..warm_len]).map_err(|err| err.to_string())?;
        read_logits(logits)?;
    }
    let mut state = model.new_state(device);
    let started = Instant::now();
    let mut logits = Vec::new();
    let chunks = ids.len().div_ceil(chunk);
    for (index, piece) in ids.chunks(chunk).enumerate() {
        let next = step(&mut state, piece).map_err(|err| err.to_string())?;
        logits = read_logits(next)?;
        if index.is_multiple_of(8) || index + 1 == chunks {
            eprintln!(
                "  {label} chunk {}/{chunks} tokens {} {:.1}s",
                index + 1,
                state.length,
                started.elapsed().as_secs_f64()
            );
        }
    }
    let prefill_ms = started.elapsed().as_secs_f64() * 1000.0;
    // The first decode step compiles the seq=1 kernel. Time the second step.
    let discard = step(&mut state, &[ids[0]]).map_err(|err| err.to_string())?;
    read_logits(discard)?;
    let decode_started = Instant::now();
    let next = step(&mut state, &[ids[0]]).map_err(|err| err.to_string())?;
    read_logits(next)?;
    Ok(PrefillTiming {
        prefill_ms,
        decode_ms: decode_started.elapsed().as_secs_f64() * 1000.0,
        logits,
    })
}

fn fresh_caches<B: burn::tensor::backend::Backend>(
    layers: usize,
    rank: usize,
    kv_heads: usize,
    rope_dim: usize,
    _device: &B::Device,
) -> Vec<LatentCache<B>> {
    (0..layers)
        .map(|_| LatentCache::new(rank, kv_heads, rope_dim))
        .collect()
}

fn measure<B, F>(
    model: &TextModel<B>,
    device: &B::Device,
    prompt: &[u32],
    new_tokens: usize,
    eos: &[u32],
    mut step: F,
) -> Result<GreedyRun, Box<dyn std::error::Error>>
where
    B: burn::tensor::backend::Backend,
    F: FnMut(&mut DecodeState<B>, &[u32]) -> omni_r2t2::Result<burn::tensor::Tensor<B, 2>>,
{
    // One untimed pass pays for shader compilation. The returned pass is the warm one.
    decode_once(model, device, prompt, new_tokens, eos, &mut step)?;
    decode_once(model, device, prompt, new_tokens, eos, &mut step)
}

fn decode_once<B, F>(
    model: &TextModel<B>,
    device: &B::Device,
    prompt: &[u32],
    new_tokens: usize,
    eos: &[u32],
    step: &mut F,
) -> Result<GreedyRun, Box<dyn std::error::Error>>
where
    B: burn::tensor::backend::Backend,
    F: FnMut(&mut DecodeState<B>, &[u32]) -> omni_r2t2::Result<burn::tensor::Tensor<B, 2>>,
{
    let mut state = model.new_state(device);
    let started = Instant::now();
    let prefill_logits = read_logits(step(&mut state, prompt).map_err(|err| err.to_string())?)?;
    let prefill_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut tokens = Vec::new();
    let mut latest = prefill_logits.clone();
    let decode_started = Instant::now();
    let mut forwards = 0usize;
    while tokens.len() < new_tokens {
        let token = argmax(&latest);
        if eos.contains(&token) {
            break;
        }
        tokens.push(token);
        if tokens.len() == new_tokens {
            break;
        }
        latest = read_logits(step(&mut state, &[token]).map_err(|err| err.to_string())?)?;
        forwards += 1;
    }
    Ok(GreedyRun {
        tokens,
        prefill_logits,
        prefill_ms,
        decode_ms: decode_started.elapsed().as_secs_f64() * 1000.0,
        forwards,
    })
}

struct GreedyRun {
    tokens: Vec<u32>,
    prefill_logits: Vec<f32>,
    prefill_ms: f64,
    decode_ms: f64,
    forwards: usize,
}

impl GreedyRun {
    fn decode_ms_per_token(&self) -> f64 {
        if self.forwards == 0 {
            0.0
        } else {
            self.decode_ms / self.forwards as f64
        }
    }
}

fn read_logits<B: burn::tensor::backend::Backend>(
    logits: burn::tensor::Tensor<B, 2>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let data = pollster::block_on(logits.into_data_async())
        .map_err(|err| format!("logit readback failed: {err:?}"))?
        .convert::<f32>();
    Ok(data
        .as_slice::<f32>()
        .map_err(|err| format!("logit cast failed: {err:?}"))?
        .to_vec())
}

fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index as u32)
        .unwrap_or(0)
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn check_backend() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use burn::backend::wgpu::{WgpuDevice, graphics, init_setup};
        let device = WgpuDevice::default();
        let setup = init_setup::<graphics::Metal>(&device, Default::default());
        eprintln!("Metal: {:?}", setup.adapter.get_info());
        probe::<burn::backend::wgpu::Metal>(&device)
    }
    #[cfg(all(
        not(all(target_os = "macos", feature = "metal")),
        any(feature = "webgpu", feature = "cpu")
    ))]
    {
        #[cfg(feature = "webgpu")]
        type B = burn::backend::wgpu::WebGpu;
        #[cfg(all(not(feature = "webgpu"), feature = "cpu"))]
        type B = burn::backend::NdArray;
        probe::<B>(&Default::default())
    }
    #[cfg(not(any(
        all(target_os = "macos", feature = "metal"),
        feature = "webgpu",
        feature = "cpu"
    )))]
    Err("No backend enabled. Build with metal, webgpu or cpu.".into())
}

fn probe<B: burn::tensor::backend::Backend>(
    device: &B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    use burn::tensor::{Tensor, TensorData};

    let a = Tensor::<B, 2>::from_data(
        TensorData::new(vec![0.125_f32; 128 * 64], [128, 64]),
        device,
    );
    let b = Tensor::<B, 2>::from_data(TensorData::new(vec![0.5_f32; 64 * 64], [64, 64]), device);
    let start = Instant::now();
    let data = pollster::block_on(a.matmul(b).into_data_async())
        .map_err(|e| format!("backend execution failed: {e:?}"))?
        .convert::<f32>();
    let values = data
        .as_slice::<f32>()
        .map_err(|e| format!("readback failed: {e:?}"))?;
    if values.len() != 128 * 64
        || values
            .iter()
            .any(|&v| !v.is_finite() || (v - 4.0).abs() > 0.01)
    {
        return Err("matrix multiplication returned an incorrect result".into());
    }
    eprintln!(
        "Backend OK: {}, [128,64] x [64,64] matmul/readback {:.2} ms (includes cold compilation)",
        B::name(device),
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}
