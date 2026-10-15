use burn::tensor::{FloatDType, Tensor, TensorData, backend::Backend};
use clap::{Parser, Subcommand};
use omni_qwen_image::{
    Config, Error, Model, Options, Precision, Progress, Result, dimensions, dit, text, vae,
};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

#[derive(Parser)]
#[command(about = "Qwen-Image-2.1-Turbo native Rust/Burn inference")]
struct Args {
    #[arg(long, default_value = "models/Qwen-Image-2.1-Turbo")]
    model: PathBuf,
    #[arg(long, default_value = "metal")]
    backend: String,
    #[arg(long)]
    f32: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Generate {
        #[arg(long)]
        prompt: String,
        /// Avoid these concepts. Requires --true-cfg-scale > 1; "" is unconditional.
        #[arg(long)]
        negative_prompt: Option<String>,
        /// Turbo recommends 1 (off). CFG adds a second denoising branch.
        #[arg(long, alias = "cfg-scale", default_value_t = 1.0)]
        true_cfg_scale: f32,
        /// Official 2K preset. Cannot be combined with --width or --height.
        #[arg(long, conflicts_with_all = ["width", "height"],
            value_parser = clap::builder::PossibleValuesParser::new(dimensions::OFFICIAL_PRESETS.iter().map(|p| p.0)))]
        aspect_ratio: Option<String>,
        /// Output pixels (256–2752); rounded down to a multiple of 32.
        #[arg(long, default_value_t = dimensions::DEFAULT_SIDE)]
        width: usize,
        /// Output pixels (256–2752); rounded down to a multiple of 32.
        #[arg(long, default_value_t = dimensions::DEFAULT_SIDE)]
        height: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value = "qwen-image.png")]
        output: PathBuf,
        /// Reference image; repeat up to four times for multi-image editing.
        #[arg(long = "image")]
        images: Vec<PathBuf>,
        /// Per-reference area is approximately this value squared.
        #[arg(long, default_value_t = 1024)]
        reference_resolution: usize,
    },
    /// Component-level numerical checks against the official implementation.
    Trace {
        #[arg(long)]
        component: String,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

fn progress(p: Progress) {
    eprintln!("{}", serde_json::to_string(&p).unwrap());
}

fn read_tensor<B: Backend, const D: usize>(
    v: &Value,
    name: &str,
    p: Precision,
    d: &B::Device,
) -> Result<Tensor<B, D>> {
    let data: Vec<f32> = serde_json::from_value(v[name]["data"].clone())?;
    let shape: Vec<usize> = serde_json::from_value(v[name]["shape"].clone())?;
    let shape: [usize; D] = shape
        .try_into()
        .map_err(|_| Error::Config(format!("invalid tensor rank: {name}")))?;
    if shape.iter().product::<usize>() != data.len() {
        return Err(Error::Config(format!("invalid tensor length: {name}")));
    }
    Ok(Tensor::<B, D>::from_data(TensorData::new(data, shape), d).cast(p.dtype()))
}

fn array<B: Backend, const D: usize>(x: Tensor<B, D>) -> Result<Value> {
    let shape = x.dims().to_vec();
    let data = x
        .cast(FloatDType::F32)
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| Error::Config(format!("{e:?}")))?;
    if data.iter().any(|v| !v.is_finite()) {
        return Err(Error::Config("non-finite trace tensor".into()));
    }
    Ok(json!({"shape":shape,"data":data}))
}

fn run<B: Backend>(args: Args, device: B::Device, cancel: Arc<AtomicBool>) -> Result<()> {
    let p = if args.f32 {
        Precision::F32
    } else {
        Precision::F16
    };
    let start = Instant::now();
    match args.command {
        Command::Generate {
            prompt,
            negative_prompt,
            true_cfg_scale,
            aspect_ratio,
            width,
            height,
            seed,
            output,
            images,
            reference_resolution,
        } => {
            let (width, height) = aspect_ratio
                .as_deref()
                .and_then(dimensions::preset)
                .unwrap_or((width, height));
            let options = Options {
                prompt,
                negative_prompt,
                true_cfg_scale,
                width,
                height,
                seed,
                images,
                reference_resolution,
            };
            options.validate()?;
            let resolved = options.output_dimensions()?;
            if resolved != (width, height) {
                eprintln!(
                    "尺寸按官方规则向下对齐：{width} × {height} → {} × {}",
                    resolved.0, resolved.1
                );
            }
            if options.negative_prompt.is_some() && !options.guidance_enabled() {
                eprintln!("提示：CFG=1，负向提示词未生效；要启用请设置 --true-cfg-scale 大于 1。");
            } else if options.true_cfg_scale > 1.0 && options.negative_prompt.is_none() {
                eprintln!("提示：未提供 --negative-prompt，CFG 未启用；无条件引导可传入空字符串。");
            }
            omni_qwen_image::pipeline::validate_output_path(&output, &options.images)?;
            let mut model = Model::<B>::load(&args.model, p, &device, &cancel, progress)?;
            let load_seconds = start.elapsed().as_secs_f64();
            let result = model.generate(&options, &cancel, progress)?;
            result.save_png(&output)?;
            println!(
                "{}",
                json!({"load_seconds":load_seconds,"report":result.report,"output":output})
            );
        }
        Command::Trace {
            component,
            input,
            output,
        } => {
            let c = Config::load(&args.model)?;
            let input: Value = serde_json::from_slice(&std::fs::read(input)?)?;
            let mut tensors = serde_json::Map::new();
            let mut tokens = None;
            match component.as_str() {
                "vision" | "multimodal" | "multimodal_text" => {
                    use omni_qwen_image::{reference, vision};
                    let paths: Vec<PathBuf> = serde_json::from_value(input["images"].clone())?;
                    let resolution = input["reference_resolution"].as_u64().unwrap_or(256) as usize;
                    let model = if component == "multimodal_text" {
                        None
                    } else {
                        Some(vision::Vision::<B>::load(
                            &args.model,
                            c.encoder.vision_config.clone(),
                            p,
                            &device,
                            &cancel,
                            &mut progress,
                        )?)
                    };
                    let mut features = Vec::new();
                    let mut grids = Vec::new();
                    for (image_index, path) in paths.into_iter().enumerate() {
                        let image =
                            reference::prepare(&reference::open(&path)?, resolution, &cancel)?;
                        let h = image.grid_height;
                        let w = image.grid_width;
                        let patches = Tensor::<B, 3>::from_data(
                            TensorData::new(image.patches, [1, h * w, 1536]),
                            &device,
                        )
                        .cast(p.dtype());
                        if component == "vision" {
                            tensors.insert("patches".into(), array(patches.clone())?);
                            tensors.insert(
                                "rgba".into(),
                                array(
                                    Tensor::<B, 4>::from_data(
                                        TensorData::new(image.vae_pixels, [1, 4, h * 16, w * 16]),
                                        &device,
                                    )
                                    .cast(p.dtype()),
                                )?,
                            );
                        }
                        let feature = if let Some(model) = &model {
                            model.encode(
                                patches,
                                h,
                                w,
                                &cancel,
                                &mut progress,
                                &mut |name, x| {
                                    if component == "vision" {
                                        tensors.insert(name.to_owned(), array(x)?);
                                    }
                                    Ok(())
                                },
                            )?
                        } else {
                            let f = &input["features"][image_index];
                            vision::Features {
                                embedding: read_tensor(f, "embedding", p, &device)?,
                                deepstack: (0..3)
                                    .map(|i| read_tensor(f, &format!("deepstack_{i}"), p, &device))
                                    .collect::<Result<Vec<_>>>()?,
                                height: h / 2,
                                width: w / 2,
                            }
                        };
                        grids.push((feature.height, feature.width));
                        features.push(feature);
                    }
                    if component.starts_with("multimodal") {
                        let model = text::TextEncoder::<B>::load(
                            &args.model,
                            c.encoder.text_config,
                            p,
                            &device,
                            &cancel,
                            &mut progress,
                        )?;
                        let layout = model.layout(
                            input["prompt"]
                                .as_str()
                                .ok_or_else(|| Error::Config("missing prompt".into()))?,
                            &grids,
                        )?;
                        tokens = Some(layout.tokens.clone());
                        let positions: Vec<f32> = layout
                            .positions
                            .iter()
                            .flatten()
                            .map(|v| *v as f32)
                            .collect();
                        tensors.insert(
                            "positions".into(),
                            json!({"shape":[1,layout.tokens.len(),3],"data":positions}),
                        );
                        let encoded = model.encode_multimodal(
                            &layout,
                            &features,
                            &cancel,
                            &mut progress,
                            &mut |name, x| {
                                tensors.insert(name.to_owned(), array(x)?);
                                Ok(())
                            },
                        )?;
                        tensors.insert("text".into(), array(encoded)?);
                    }
                }
                "text" => {
                    let model = text::TextEncoder::<B>::load(
                        &args.model,
                        c.encoder.text_config,
                        p,
                        &device,
                        &cancel,
                        &mut progress,
                    )?;
                    let prompt = input["prompt"]
                        .as_str()
                        .ok_or_else(|| Error::Config("missing prompt".into()))?;
                    tokens = Some(model.tokenize(prompt)?);
                    let text = model.encode_with_trace(
                        prompt,
                        &cancel,
                        &mut progress,
                        &mut |name, x| {
                            tensors.insert(name.to_owned(), array(x)?);
                            Ok(())
                        },
                    )?;
                    tensors.insert("text".into(), array(text)?);
                }
                "dit" | "dit_edit" => {
                    let model = dit::Dit::<B>::load(
                        &args.model,
                        c.dit.clone(),
                        p,
                        &device,
                        &cancel,
                        &mut progress,
                    )?;
                    let mut cache = dit::Cache::new(c.dit.num_layers);
                    let z = read_tensor(&input, "latents", p, &device)?;
                    let text = read_tensor(&input, "text", p, &device)?;
                    let spans: Vec<text::ImageSpan> =
                        serde_json::from_value(input.get("spans").cloned().unwrap_or(json!([])))?;
                    let references = input["references"]
                        .as_array()
                        .map(|items| {
                            items
                                .iter()
                                .map(|v| {
                                    Ok(dit::ReferenceLatent {
                                        latents: read_tensor(v, "latents", p, &device)?,
                                        height: v["height"].as_u64().unwrap() as usize,
                                        width: v["width"].as_u64().unwrap() as usize,
                                    })
                                })
                                .collect::<Result<Vec<_>>>()
                        })
                        .transpose()?
                        .unwrap_or_default();
                    let h = input["height"].as_u64().unwrap() as usize;
                    let w = input["width"].as_u64().unwrap() as usize;
                    let first = model.predict_with_images(
                        z.clone(),
                        text.clone(),
                        &references,
                        &spans,
                        c.sigmas[0],
                        h,
                        w,
                        &mut cache,
                        &cancel,
                        &mut progress,
                    )?;
                    tensors.insert("first".into(), array(first.clone())?);
                    let next = omni_qwen_image::pipeline::euler_step(
                        z,
                        first,
                        c.sigmas[1] - c.sigmas[0],
                        p,
                    );
                    tensors.insert("next_latents".into(), array(next.clone())?);
                    let second = model.predict_with_images(
                        next.clone(),
                        text.clone(),
                        &references,
                        &spans,
                        c.sigmas[1],
                        h,
                        w,
                        &mut cache,
                        &cancel,
                        &mut progress,
                    )?;
                    tensors.insert("second".into(), array(second)?);
                    let mut fresh = dit::Cache::new(c.dit.num_layers);
                    tensors.insert(
                        "second_uncached".into(),
                        array(model.predict_with_images(
                            next,
                            text,
                            &references,
                            &spans,
                            c.sigmas[1],
                            h,
                            w,
                            &mut fresh,
                            &cancel,
                            &mut progress,
                        )?)?,
                    );
                }
                "vae" => {
                    let model = vae::Vae::<B>::load(
                        &args.model,
                        &c.vae,
                        p,
                        &device,
                        &cancel,
                        &mut progress,
                    )?;
                    tensors.insert(
                        "decoded".into(),
                        array(model.decode(
                            read_tensor(&input, "latents", p, &device)?,
                            &cancel,
                            &mut progress,
                        )?)?,
                    );
                    tensors.insert(
                        "encoded".into(),
                        array(model.encode(
                            read_tensor(&input, "rgba", p, &device)?,
                            &cancel,
                            &mut progress,
                        )?)?,
                    );
                }
                _ => {
                    return Err(Error::Config(
                        "component must be text, vision, multimodal, dit, dit_edit, or vae".into(),
                    ));
                }
            }
            if let Some(parent) = output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(
                output,
                serde_json::to_vec(
                    &json!({"component":component,"tokens":tokens,"tensors":tensors,"elapsed":start.elapsed().as_secs_f64()}),
                )?,
            )?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();
    ctrlc::set_handler(move || flag.store(true, Ordering::Release))
        .map_err(|e| Error::Config(e.to_string()))?;
    match args.backend.as_str() {
        #[cfg(feature = "metal")]
        "metal" => run::<burn::backend::wgpu::Metal>(args, Default::default(), cancel),
        #[cfg(feature = "webgpu")]
        "webgpu" => run::<burn::backend::wgpu::WebGpu>(args, Default::default(), cancel),
        #[cfg(feature = "cpu")]
        "cpu" => run::<burn::backend::NdArray>(args, Default::default(), cancel),
        _ => Err(Error::Config("所选后端未编译到此程序中".into())),
    }
}
