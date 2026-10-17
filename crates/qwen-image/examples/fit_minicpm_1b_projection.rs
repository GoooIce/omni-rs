//! Fit the MiniCPM-V-4.7-1B text-condition projection and leave the weights untracked.
//!
//! Caches the official conditions and the 1B last-layer sequences, then updates
//! the map on the fixed 16/8 split. The holdout error is on the condition tensor.

use burn::tensor::{FloatDType, backend::Backend};
use omni_qwen_image::text::TextEncoder;
use omni_qwen_image::{
    CONDITION_WIDTH, ConditionProjection, Config, MiniCpm1bText, Precision, Sequence,
    default_projection_weights_path, text_to_image_split,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

fn main() {
    let qwen = arg_path(1, "/Users/goooice/code/omni-rs/models/Qwen-Image-2.1-Turbo");
    let minicpm = arg_path(2, "/Users/goooice/code/omni-rs/models/MiniCPM-V-4.7-1B");
    let cache = arg_path(3, "artifacts/minicpm-v-4.7-1b/cache");
    let output = arg_path(
        4,
        default_projection_weights_path()
            .to_str()
            .unwrap_or("artifacts/minicpm-v-4.7-1b/text-condition-projection.bin"),
    );
    let (train, holdout) = text_to_image_split();
    let prompts: Vec<&str> = train
        .iter()
        .copied()
        .chain(holdout.iter().copied())
        .collect();
    std::fs::create_dir_all(&cache).expect("cache directory");

    let missing_official = (0..prompts.len()).any(|index| !cached(&cache, "official", index));
    let missing_last = (0..prompts.len()).any(|index| !cached(&cache, "last-layer", index));
    if missing_official {
        #[cfg(feature = "metal")]
        extract_official::<burn::backend::wgpu::Metal>(
            &qwen,
            &cache,
            &prompts,
            &Default::default(),
            Precision::F16,
        );
        #[cfg(all(not(feature = "metal"), feature = "cpu"))]
        extract_official::<burn::backend::NdArray>(
            &qwen,
            &cache,
            &prompts,
            &Default::default(),
            Precision::F32,
        );
        #[cfg(all(not(feature = "metal"), not(feature = "cpu")))]
        panic!("fit the projection with --features metal or --features cpu");
    }
    if missing_last {
        eprintln!("loading MiniCPM-V-4.7-1B from {}", minicpm.display());
        let encoder = MiniCpm1bText::load(&minicpm).expect("MiniCPM-V-4.7-1B");
        for (index, prompt) in prompts.iter().enumerate() {
            let path = cache.join(format!("last-layer-{index:02}.bin"));
            if path.is_file() {
                continue;
            }
            eprintln!("1B last layer {index} / {}", prompts.len());
            let sequence = encoder.last_layer(prompt).expect("last layer");
            write_sequence(&path, &sequence);
        }
    }

    let pairs: Vec<_> = (0..prompts.len())
        .map(|index| {
            (
                read_sequence(&cache.join(format!("last-layer-{index:02}.bin"))),
                read_sequence(&cache.join(format!("official-{index:02}.bin"))),
            )
        })
        .collect();
    let in_width = pairs[0].0.width();
    assert!(
        pairs.iter().all(|(input, target)| {
            input.width() == in_width && target.width() == CONDITION_WIDTH
        })
    );
    let init = ConditionProjection::deterministic_init(in_width).expect("init");
    eprintln!(
        "fitting {in_width} -> {CONDITION_WIDTH} on {} prompts",
        train.len()
    );
    let fitted = ConditionProjection::fit(&pairs[..train.len()]).expect("fit");
    let init_mse = init
        .mean_squared_error(&pairs[train.len()..])
        .expect("init mse");
    let fitted_mse = fitted
        .mean_squared_error(&pairs[train.len()..])
        .expect("fitted mse");
    println!("init_mse {init_mse:.8}");
    println!("fitted_mse {fitted_mse:.8}");
    if fitted_mse >= init_mse {
        eprintln!("holdout condition error did not fall below the untrained projection");
        std::process::exit(1);
    }
    fitted.save(&output).expect("save projection");
    println!("saved {}", output.display());
}

fn arg_path(index: usize, default: &str) -> PathBuf {
    std::env::args()
        .nth(index)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default))
}

fn cached(cache: &Path, kind: &str, index: usize) -> bool {
    cache.join(format!("{kind}-{index:02}.bin")).is_file()
}

fn extract_official<B: Backend>(
    root: &Path,
    cache: &Path,
    prompts: &[&str],
    device: &B::Device,
    precision: Precision,
) {
    eprintln!("loading official text encoder from {}", root.display());
    let config = Config::load(root).expect("qwen config");
    let cancel = AtomicBool::new(false);
    let encoder = TextEncoder::<B>::load(
        root,
        config.encoder.text_config.clone(),
        precision,
        device,
        &cancel,
        &mut |_| {},
    )
    .expect("text encoder");
    for (index, prompt) in prompts.iter().enumerate() {
        let path = cache.join(format!("official-{index:02}.bin"));
        if path.is_file() {
            continue;
        }
        eprintln!("official condition {index} / {}", prompts.len());
        let tensor = encoder
            .encode(prompt, &cancel, &mut |_| {})
            .expect("official condition");
        let dims = tensor.dims();
        if dims[2] != CONDITION_WIDTH {
            panic!(
                "official condition width {} is not {CONDITION_WIDTH}",
                dims[2]
            );
        }
        B::sync(device).expect("sync");
        let values = tensor
            .cast(FloatDType::F32)
            .into_data()
            .to_vec::<f32>()
            .expect("official values");
        write_sequence(
            &path,
            &Sequence::last_layer(dims[1], dims[2], values).expect("sequence"),
        );
    }
}

fn write_sequence(path: &Path, sequence: &Sequence) {
    let mut bytes = Vec::with_capacity(8 + sequence.values().len() * 4);
    bytes.extend_from_slice(&(sequence.length() as u32).to_le_bytes());
    bytes.extend_from_slice(&(sequence.width() as u32).to_le_bytes());
    for value in sequence.values() {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("cache parent");
    }
    std::fs::write(path, bytes).expect("write cache");
}

fn read_sequence(path: &Path) -> Sequence {
    let bytes = std::fs::read(path).expect("read cache");
    let length = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let width = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let mut values = Vec::with_capacity(length * width);
    for chunk in bytes[8..].chunks(4) {
        values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Sequence::last_layer(length, width, values).expect("cached sequence")
}
