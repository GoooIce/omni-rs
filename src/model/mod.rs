pub mod decoder;
pub mod encoder;
mod ops;

#[cfg(feature = "native")]
use crate::weights::Weights;
use crate::{ModelConfig, Precision, Result, text::TextProcessor, weights::WeightSource};
use burn::tensor::backend::Backend;
use decoder::Decoder;
use encoder::Encoder;

pub struct Model<B: Backend> {
    config: ModelConfig,
    encoder: Encoder<B>,
    decoder: Decoder<B>,
    text: TextProcessor,
    precision: Precision,
}

/// Read-only access for numerical comparisons and backend diagnostics.
/// Production transcription should use [`crate::Session`], which owns cache
/// validity, prompt construction and streaming state.
pub struct ModelDiagnostics<'a, B: Backend> {
    pub encoder: &'a Encoder<B>,
    pub decoder: &'a Decoder<B>,
    pub text: &'a TextProcessor,
}

impl<B: Backend> Model<B> {
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    pub fn diagnostics(&self) -> ModelDiagnostics<'_, B> {
        ModelDiagnostics {
            encoder: &self.encoder,
            decoder: &self.decoder,
            text: &self.text,
        }
    }

    pub(crate) fn encoder(&self) -> &Encoder<B> {
        &self.encoder
    }

    pub(crate) fn decoder(&self) -> &Decoder<B> {
        &self.decoder
    }

    pub(crate) fn text(&self) -> &TextProcessor {
        &self.text
    }

    pub fn load(
        config: ModelConfig,
        tokenizer: &[u8],
        weights: &impl WeightSource<B>,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Self> {
        Self::load_cancellable(
            config,
            tokenizer,
            weights,
            precision,
            device,
            &std::sync::atomic::AtomicBool::new(false),
            |_, _| {},
        )
    }

    pub fn load_cancellable(
        config: ModelConfig,
        tokenizer: &[u8],
        weights: &impl WeightSource<B>,
        precision: Precision,
        device: &B::Device,
        cancel: &std::sync::atomic::AtomicBool,
        progress: impl Fn(f64, &str),
    ) -> Result<Self> {
        use crate::{
            loading::{CancellableWeights, check},
            weights::{CheckpointLayout, CheckpointStage},
        };
        check(cancel)?;
        let layout = CheckpointLayout::new(&config)?;
        let encoder_layers = config.thinker_config.audio_config.encoder_layers;
        let decoder_layers = config.thinker_config.text_config.num_hidden_layers;
        let total = encoder_layers + decoder_layers + 4;
        let last = std::cell::Cell::new(0);
        let controlled = CancellableWeights::new(weights, cancel, |name: &str| {
            let Some(stage) = layout.stage(name) else {
                return;
            };
            let (step, message) = match stage {
                CheckpointStage::EncoderLayer(i) => {
                    (1 + i, format!("语音编码器 {}/{}", i + 1, encoder_layers))
                }
                CheckpointStage::AudioFeatures => (1 + encoder_layers, "语音特征层".into()),
                CheckpointStage::Vocabulary => (2 + encoder_layers, "文本词表".into()),
                CheckpointStage::DecoderLayer(i) => (
                    3 + encoder_layers + i,
                    format!("文本解码器 {}/{}", i + 1, decoder_layers),
                ),
                CheckpointStage::TextOutput => {
                    (3 + encoder_layers + decoder_layers, "文本输出层".into())
                }
            };
            if step > last.get() {
                last.set(step);
                progress(step as f64 / total as f64, &message);
            }
        });
        let weights = &controlled;
        config.validate()?;
        precision.validate_device::<B>(device)?;
        let text = TextProcessor::new(tokenizer, &config)?;
        let encoder = Encoder::load(
            weights,
            &config.thinker_config.audio_config,
            precision,
            device,
        )?;
        let decoder = Decoder::load(
            weights,
            &config.thinker_config.text_config,
            precision,
            device,
        )?;
        check(cancel)?;
        progress(1.0, "ASR 模型已就绪");
        check(cancel)?;
        Ok(Self {
            config,
            encoder,
            decoder,
            text,
            precision,
        })
    }

    #[cfg(feature = "native")]
    pub fn from_directory(
        path: impl AsRef<std::path::Path>,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Self> {
        Self::from_directory_cancellable(
            path,
            precision,
            device,
            &std::sync::atomic::AtomicBool::new(false),
            |_, _| {},
        )
    }

    #[cfg(feature = "native")]
    pub fn from_directory_cancellable(
        path: impl AsRef<std::path::Path>,
        precision: Precision,
        device: &B::Device,
        cancel: &std::sync::atomic::AtomicBool,
        progress: impl Fn(f64, &str),
    ) -> Result<Self> {
        use crate::loading::check;
        check(cancel)?;
        let path = path.as_ref();
        let config = ModelConfig::from_json(&std::fs::read(path.join("config.json"))?)?;
        check(cancel)?;
        let tokenizer = std::fs::read(path.join("tokenizer.json"))?;
        check(cancel)?;
        let weights = Weights::open(path)?;
        Self::load_cancellable(
            config, &tokenizer, &weights, precision, device, cancel, progress,
        )
    }

    /// Compatibility alias. Backend selection is controlled by B and its device.
    #[cfg(feature = "native")]
    pub fn from_directory_auto(
        path: impl AsRef<std::path::Path>,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Self> {
        Self::from_directory(path, precision, device)
    }

    /// Legacy model-specific experiment, separate from the generic burn-ane backend.
    /// Load the audio transformer onto ANE; the frontend and decoder use B.
    /// Requires a compiled export paired with this exact checkpoint.
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    pub fn from_directory_coreml(
        path: impl AsRef<std::path::Path>,
        coreml: impl AsRef<std::path::Path>,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Self> {
        let path = path.as_ref();
        let config = ModelConfig::from_json(&std::fs::read(path.join("config.json"))?)?;
        config.validate()?;
        precision.validate_device::<B>(device)?;
        let audio = &config.thinker_config.audio_config;
        let ane = crate::coreml::CoreMlEncoder::open(
            coreml.as_ref(),
            &path.join("model.safetensors"),
            audio,
        )?;
        eprintln!(
            "Core ML: {} operations assigned to ANE; GPU disabled for this graph",
            ane.ane_operations()
        );
        let weights = Weights::open(path)?;
        let encoder = Encoder::load_coreml(&weights, audio, precision, device, ane)?;
        let decoder = Decoder::load(
            &weights,
            &config.thinker_config.text_config,
            precision,
            device,
        )?;
        let text = TextProcessor::new(&std::fs::read(path.join("tokenizer.json"))?, &config)?;
        Ok(Self {
            config,
            encoder,
            decoder,
            text,
            precision,
        })
    }
}
