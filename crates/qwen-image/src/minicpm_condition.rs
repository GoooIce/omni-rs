//! Text-only last layer of MiniCPM-V-4.7-1B, for the projection condition.
//!
//! Uses the selected GPU and explicit matmul/softmax attention for variable
//! prompt lengths. This path never opens an image or the JSON rewriter.

use crate::projection::Sequence;
use crate::{Error, Result};
use burn::tensor::backend::Backend;
use omni_minicpm_v::{
    chat::{self, Message, Role},
    config::ModelDirectory,
    model::TextModel,
    tokenizer::MiniCpmTokenizer,
};
use omni_r2t2::Precision;
use std::path::{Path, PathBuf};

#[cfg(any(feature = "wgpu", feature = "metal"))]
type Native = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;
#[cfg(not(any(feature = "wgpu", feature = "metal")))]
type Native = burn::backend::NdArray;

fn default_device<B: Backend>() -> B::Device {
    B::Device::default()
}

struct Loaded<B: Backend> {
    model: TextModel<B>,
    tokenizer: MiniCpmTokenizer,
    device: B::Device,
}

impl<B: Backend + omni_minicpm_v::quant::ExpertResidence> Loaded<B> {
    fn load(model_dir: &Path) -> Result<Self> {
        if !model_dir.join("config.json").is_file() {
            return Err(Error::Config(format!(
                "去噪观察缺少 MiniCPM-V-4.7-1B 检查点 {}，不能用未训练投影出图，也不会退回 JSON 改写",
                model_dir.display()
            )));
        }
        let directory = ModelDirectory::new(model_dir);
        let config = directory.config().map_err(|e| {
            Error::Config(format!(
                "读取 MiniCPM-V-4.7-1B 配置失败（{}）：{e}",
                model_dir.display()
            ))
        })?;
        let tokenizer = directory.tokenizer().map_err(|e| {
            Error::Config(format!(
                "读取 MiniCPM-V-4.7-1B 分词器失败（{}）：{e}",
                model_dir.display()
            ))
        })?;
        let weights = omni_r2t2::weights::Weights::open(model_dir).map_err(Error::Weights)?;
        let precision = Precision::F32;
        let device = B::Device::default();
        let model =
            TextModel::<B>::load(&weights, &config, precision, &device).map_err(Error::Weights)?;
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    fn last_layer(&self, prompt: &str) -> Result<Sequence> {
        self.last_layer_with(prompt, omni_r2t2::tensor::AttentionKernel::ImageAligned)
    }

    fn last_layer_with(
        &self,
        prompt: &str,
        kernel: omni_r2t2::tensor::AttentionKernel,
    ) -> Result<Sequence> {
        let rendered = chat::render(&[Message::text(Role::User, prompt)], false, None)
            .map_err(|e| Error::Config(format!("MiniCPM-V-4.7-1B 提示词渲染失败：{e}")))?;
        let ids = self.tokenizer.encode(&rendered);
        let mut state = self.model.new_state(&self.device);
        let hidden = self
            .model
            .last_layer_sequence_with(&ids, None, None, &mut state, kernel)
            .map_err(Error::Weights)?;
        let dims = hidden.dims();
        B::sync(&self.device).map_err(|e| Error::Config(e.to_string()))?;
        let values = hidden
            .cast(burn::tensor::FloatDType::F32)
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| Error::Config(format!("读取 MiniCPM-V-4.7-1B 最后一层失败：{e:?}")))?;
        Sequence::last_layer(dims[1], dims[2], values)
    }
}

pub struct MiniCpm1bText {
    inner: Loaded<Native>,
}

impl MiniCpm1bText {
    pub fn load(model_dir: &Path) -> Result<Self> {
        Ok(Self {
            inner: Loaded::<Native>::load(model_dir)?,
        })
    }

    /// Last-layer sequence for a text prompt. No image and no JSON rewrite.
    pub fn last_layer(&self, prompt: &str) -> Result<Sequence> {
        self.inner.last_layer(prompt)
    }
}

pub fn default_minicpm_1b_dir(qwen_root: &Path) -> PathBuf {
    if let Some(path) = std::env::var_os("OMNI_MINICPM_1B") {
        return PathBuf::from(path);
    }
    qwen_root
        .parent()
        .map(|parent| parent.join("MiniCPM-V-4.7-1B"))
        .unwrap_or_else(|| PathBuf::from("models/MiniCPM-V-4.7-1B"))
}

/// Text-only last layer of MiniCPM-V-4.7-35B-A3B-int4.
///
/// qwen-image's fused Metal wrapper panics when in-kernel int4 dequant and F16
/// activations disagree on dtype, so this loader uses the raw cube backend and
/// keeps int4 packed. A full NdArray load would dequantize those experts.
/// Full-attention layers use the image-aligned fallback: eager Metal autotune
/// can keep a stage size that does not divide the next prompt. This path never
/// opens an image or the JSON rewriter.
#[cfg(any(feature = "wgpu", feature = "metal"))]
type MiniCpm35bBackend = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;
#[cfg(not(any(feature = "wgpu", feature = "metal")))]
type MiniCpm35bBackend = burn::backend::NdArray;

pub struct MiniCpm35bText {
    inner: Loaded<MiniCpm35bBackend>,
}

impl MiniCpm35bText {
    pub fn load(model_dir: &Path) -> Result<Self> {
        if !model_dir.join("config.json").is_file() {
            return Err(Error::Config(format!(
                "去噪观察缺少 MiniCPM-V-4.7-35B-A3B-int4 检查点 {}，不能用未训练投影出图，也不会退回 JSON 改写",
                model_dir.display()
            )));
        }
        let directory = ModelDirectory::new(model_dir);
        let config = directory.config().map_err(|e| {
            Error::Config(format!(
                "读取 MiniCPM-V-4.7-35B-A3B-int4 配置失败（{}）：{e}",
                model_dir.display()
            ))
        })?;
        if config.text_config.hidden_size != crate::projection::MiniCpm35bProjection::HIDDEN {
            return Err(Error::Config(format!(
                "MiniCPM-V-4.7-35B-A3B-int4 的最后一层宽度是 {}，观察要求 2048",
                config.text_config.hidden_size
            )));
        }
        let tokenizer = directory.tokenizer().map_err(|e| {
            Error::Config(format!(
                "读取 MiniCPM-V-4.7-35B-A3B-int4 分词器失败（{}）：{e}",
                model_dir.display()
            ))
        })?;
        let weights = omni_r2t2::weights::Weights::open(model_dir).map_err(Error::Weights)?;
        #[cfg(any(feature = "wgpu", feature = "metal"))]
        let precision = Precision::F16;
        #[cfg(not(any(feature = "wgpu", feature = "metal")))]
        let precision = Precision::F32;
        let device = default_device::<MiniCpm35bBackend>();
        let model = TextModel::<MiniCpm35bBackend>::load(&weights, &config, precision, &device)
            .map_err(Error::Weights)?;
        Ok(Self {
            inner: Loaded {
                model,
                tokenizer,
                device,
            },
        })
    }

    /// Last-layer sequence for a text prompt. No image and no JSON rewrite.
    pub fn last_layer(&self, prompt: &str) -> Result<Sequence> {
        let rendered =
            chat::render(&[Message::text(Role::User, prompt)], false, None).map_err(|e| {
                Error::Config(format!("MiniCPM-V-4.7-35B-A3B-int4 提示词渲染失败：{e}"))
            })?;
        let ids = self.inner.tokenizer.encode(&rendered);
        let mut state = self.inner.model.new_state(&self.inner.device);
        let hidden = self
            .inner
            .model
            .last_layer_sequence_with(
                &ids,
                None,
                None,
                &mut state,
                omni_r2t2::tensor::AttentionKernel::ImageAligned,
            )
            .map_err(Error::Weights)?;
        let dims = hidden.dims();
        if dims[2] != crate::projection::MiniCpm35bProjection::HIDDEN {
            return Err(Error::Config(format!(
                "MiniCPM-V-4.7-35B-A3B-int4 的最后一层宽度是 {}，观察要求 2048",
                dims[2]
            )));
        }
        MiniCpm35bBackend::sync(&self.inner.device).map_err(|e| Error::Config(e.to_string()))?;
        let values = hidden
            .cast(burn::tensor::FloatDType::F32)
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| {
                Error::Config(format!(
                    "读取 MiniCPM-V-4.7-35B-A3B-int4 最后一层失败：{e:?}"
                ))
            })?;
        Sequence::last_layer(dims[1], dims[2], values)
    }
}

pub fn default_minicpm_35b_dir(qwen_root: &Path) -> PathBuf {
    if let Some(path) = std::env::var_os("OMNI_MINICPM_35B") {
        return PathBuf::from(path);
    }
    qwen_root
        .parent()
        .map(|parent| parent.join("MiniCPM-V-4.7-35B-A3B-int4"))
        .unwrap_or_else(|| PathBuf::from("models/MiniCPM-V-4.7-35B-A3B-int4"))
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
    use std::{
        any::TypeId,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    // Hidden activations precede RMSNorm. Allow FP32 reduction-order noise, but
    // reject missing values, nonfinite results and changes large enough to alter
    // the fitted projection's input.
    fn parity(candidate: &Sequence, reference: &Sequence) -> std::result::Result<f32, String> {
        if candidate.length() != reference.length() || candidate.width() != reference.width() {
            return Err("last-layer dimensions differ".into());
        }
        let mut max_error = 0.0_f32;
        for (&actual, &expected) in candidate.values().iter().zip(reference.values()) {
            if !actual.is_finite() || !expected.is_finite() {
                return Err("nonfinite last-layer value".into());
            }
            let error = (actual - expected).abs();
            if error > 2e-3 + 2e-4 * expected.abs() {
                return Err(format!("last-layer mismatch: {actual} vs {expected}"));
            }
            max_error = max_error.max(error);
        }
        Ok(max_error)
    }

    type PanicHook = dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync;

    struct PanicCounter {
        previous: Arc<PanicHook>,
        count: Arc<AtomicUsize>,
    }

    impl PanicCounter {
        fn install() -> Self {
            let previous: Arc<PanicHook> = std::panic::take_hook().into();
            let original = Arc::clone(&previous);
            let count = Arc::new(AtomicUsize::new(0));
            let observed = Arc::clone(&count);
            std::panic::set_hook(Box::new(move |info| {
                observed.fetch_add(1, Ordering::SeqCst);
                original(info);
            }));
            Self { previous, count }
        }
    }

    impl Drop for PanicCounter {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                let previous = Arc::clone(&self.previous);
                std::panic::set_hook(Box::new(move |info| previous(info)));
            }
        }
    }

    #[test]
    #[ignore = "local 1B checkpoint; run in release mode on Metal"]
    fn minicpm_1b_gpu_condition_matches_cpu_across_prompt_lengths() {
        assert_ne!(
            TypeId::of::<Native>(),
            TypeId::of::<burn::backend::NdArray>(),
            "the production 1B text conditioner must run on the selected GPU"
        );
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let model_dir = root.join("models/MiniCPM-V-4.7-1B");
        assert!(
            model_dir.join("config.json").is_file(),
            "missing 1B checkpoint"
        );
        let cache = tempfile::tempdir().unwrap();
        let mut runtime = CubeClRuntimeConfig::default();
        runtime.autotune.cache = CacheConfig::File(cache.path().to_path_buf());
        CubeClRuntimeConfig::set(runtime);
        let panics = PanicCounter::install();
        let gpu = MiniCpm1bText::load(&model_dir).unwrap();
        let cpu = Loaded::<burn::backend::NdArray>::load(&model_dir).unwrap();
        let prompts = [
            "茶",
            "一只白瓷茶壶放在木桌上，窗边有柔和的晨光。背景是竹林和远山，桌上还有一本书、一只杯子和几片落叶。镜头拉远，能看到完整的窗框。",
            "Draw a red fox in the snow.",
            "茶",
        ];
        let mut lengths = Vec::new();
        let mut first = None;
        for prompt in prompts {
            let reference = cpu
                .last_layer_with(prompt, omni_r2t2::tensor::AttentionKernel::Automatic)
                .unwrap();
            let actual = gpu.last_layer(prompt).unwrap();
            let max_error = parity(&actual, &reference).unwrap();
            lengths.push(actual.length());
            eprintln!(
                "1B condition: tokens={}, width={}, max_abs_error={max_error:.6}",
                actual.length(),
                actual.width()
            );
            if let Some(previous) = first.as_ref() {
                if prompt == prompts[0] {
                    parity(&actual, previous).unwrap();
                }
            } else {
                first = Some(actual.clone());
                // Verify the gate rejects a perturbed, truncated or nonfinite
                // candidate; a successful readback alone is not acceptance.
                let mut changed = actual.values().to_vec();
                changed[0] += 1.0;
                assert!(
                    parity(
                        &Sequence::last_layer(actual.length(), actual.width(), changed).unwrap(),
                        &reference
                    )
                    .is_err()
                );
                let mut nonfinite = actual.values().to_vec();
                nonfinite[0] = f32::NAN;
                assert!(
                    parity(
                        &Sequence::last_layer(actual.length(), actual.width(), nonfinite).unwrap(),
                        &reference
                    )
                    .is_err()
                );
                assert!(
                    parity(&Sequence::last_layer(1, 1, vec![0.0]).unwrap(), &reference).is_err()
                );
            }
        }
        assert!(lengths.windows(2).all(|pair| pair[0] != pair[1]));
        assert_eq!(lengths[0], lengths[3]);
        assert_eq!(panics.count.load(Ordering::SeqCst), 0, "caught GPU panics");
    }
}
