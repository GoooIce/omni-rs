//! Text-only last layer of MiniCPM-V-4.7-1B, for the projection condition.
//!
//! Eager Metal attention autotune can keep a stage size that does not divide the
//! next prompt. NdArray accepts every length. This path never opens an image or
//! the JSON rewriter.

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

type Native = burn::backend::NdArray;

struct Loaded<B: Backend> {
    model: TextModel<B>,
    tokenizer: MiniCpmTokenizer,
    device: B::Device,
}

impl<B: Backend> Loaded<B> {
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
        // NdArray cannot hold packed int4. This 1B checkpoint is dense; dequantize
        // is the CPU reference path either way.
        let quant = omni_minicpm_v::quant::LoadMode::Dequantize;
        let precision = Precision::F32;
        let device = B::Device::default();
        let model = TextModel::<B>::load_with(&weights, &config, precision, &device, Some(quant))
            .map_err(Error::Weights)?;
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    fn last_layer(&self, prompt: &str) -> Result<Sequence> {
        let rendered = chat::render(&[Message::text(Role::User, prompt)], false, None)
            .map_err(|e| Error::Config(format!("MiniCPM-V-4.7-1B 提示词渲染失败：{e}")))?;
        let ids = self.tokenizer.encode(&rendered);
        let mut state = self.model.new_state(&self.device);
        let hidden = self
            .model
            .last_layer_sequence(&ids, None, None, &mut state)
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
