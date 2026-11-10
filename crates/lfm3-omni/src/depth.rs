//! Bidirectional speech-code depth decoder.
use crate::{
    config::{Config, TextConfig},
    sampling::AudioSampling,
    text::{Layer, norm},
};
use burn::tensor::{Int, Tensor, TensorData, activation::log_softmax, backend::Backend};
use omni_r2t2::{
    Error, Precision, Result,
    tensor::{self, Embedding, Linear, Norm},
    weights::{WeightSource, Weights},
};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
struct TextWidth {
    hidden_size: usize,
}
/// Depth decoder settings from an Omni checkpoint.
#[derive(Clone, Debug, Deserialize)]
pub struct DepthConfig {
    depth_config: TextConfig,
    text_config: TextWidth,
    num_codebooks: usize,
    codebook_size: usize,
}
impl DepthConfig {
    pub fn load(directory: &Path) -> Result<Self> {
        let config: Self = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        config.validate()?;
        Ok(config)
    }
    fn validate(&self) -> Result<()> {
        Config {
            model_type: "lfm3_omni".into(),
            text_config: self.depth_config.clone(),
        }
        .validate()?;
        if self.num_codebooks == 0
            || self.codebook_size == 0
            || self.text_config.hidden_size == 0
            || self.num_codebooks > self.depth_config.max_position_embeddings
            || !self
                .depth_config
                .layer_types
                .iter()
                .all(|kind| kind == "full_attention")
        {
            return Err(Error::Config(
                "invalid or unsupported LFM3 speech depth configuration".into(),
            ));
        }
        Ok(())
    }
}
struct Codebook<B: Backend> {
    embedding: Embedding<B>,
    norm: Norm<B>,
}
/// Scores codebooks for one speech frame, given its preceding language state.
pub struct DepthDecoder<B: Backend> {
    config: DepthConfig,
    precision: Precision,
    input: Linear<B>,
    mask: Tensor<B, 1>,
    layers: Vec<Layer<B>>,
    final_norm: Norm<B>,
    codebooks: Vec<Codebook<B>>,
}
impl<B: Backend> DepthDecoder<B> {
    pub fn load(directory: &Path, precision: Precision, device: &B::Device) -> Result<Self> {
        let config = DepthConfig::load(directory)?;
        let weights = Weights::open(directory)?;
        Self::from_weights(&weights, config, precision, device)
    }
    pub fn from_weights(
        w: &impl WeightSource<B>,
        config: DepthConfig,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Self> {
        config.validate()?;
        let c = &config.depth_config;
        let h = c.hidden_size;
        let prefix = "model.depth_decoder";
        let input = Linear::load(
            w,
            &format!("{prefix}.depth_linear"),
            config.text_config.hidden_size,
            config.num_codebooks * h,
            true,
            precision,
            device,
        )?;
        let mask = w.tensor(&format!("{prefix}.mask_embedding"), [h], precision, device)?;
        let layers = (0..c.num_hidden_layers)
            .map(|i| Layer::load_named(w, &format!("{prefix}.model"), c, i, precision, device))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = norm(
            w,
            &format!("{prefix}.model.embedding_norm"),
            h,
            c.norm_eps,
            device,
        )?;
        let codebooks = (0..config.num_codebooks)
            .map(|i| {
                let name = format!("{prefix}.depth_embeddings.{i}");
                Ok(Codebook {
                    embedding: Embedding::load(
                        w,
                        &format!("{name}.embedding.weight"),
                        config.codebook_size,
                        h,
                        precision,
                        device,
                    )?,
                    norm: norm(w, &format!("{name}.embedding_norm"), h, c.norm_eps, device)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config,
            precision,
            input,
            mask,
            layers,
            final_norm,
            codebooks,
        })
    }
    /// Logits `[num_codebooks, codebook_size]`. Masked slots use the learned mask embedding.
    pub fn logits(
        &self,
        hidden: Tensor<B, 2>,
        codes: &[u32],
        masked: &[bool],
    ) -> Result<Tensor<B, 2>> {
        let c = &self.config.depth_config;
        let n = self.config.num_codebooks;
        let h = c.hidden_size;
        if hidden.dims() != [1, self.config.text_config.hidden_size]
            || codes.len() != n
            || masked.len() != n
            || codes
                .iter()
                .any(|code| *code as usize >= self.config.codebook_size)
        {
            return Err(Error::Inference("invalid speech depth input".into()));
        }
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            values.push(if masked[i] {
                self.mask.clone().reshape([1, h])
            } else {
                self.codebooks[i].embedding.embed(&codes[i..i + 1])?
            });
        }
        let mut x = self
            .input
            .forward(hidden.cast(self.precision.dtype()))
            .reshape([1, n, h])
            + Tensor::cat(values, 0).reshape([1, n, h]);
        let dtype = tensor::dtype(&x);
        for layer in &self.layers {
            x = layer.forward(x, &mut layer.new_state(), 0, false);
        }
        x = self.final_norm.forward(x, dtype);
        let mut logits = Vec::with_capacity(n);
        for (i, book) in self.codebooks.iter().enumerate() {
            let value = x.clone().slice([0..1, i..i + 1, 0..h]).reshape([1, h]);
            logits.push(book.embedding.logits(book.norm.forward(value, dtype)));
        }
        Ok(Tensor::cat(logits, 0))
    }

    /// Sum masked codebook cross-entropies for one teacher-forced frame.
    ///
    /// The returned count lets callers combine multiple frames before dividing,
    /// matching the official loss when frames mask different numbers of books.
    pub fn masked_loss(
        &self,
        hidden: Tensor<B, 2>,
        codes: &[u32],
        masked: &[bool],
    ) -> Result<(Tensor<B, 1>, usize)> {
        let count = masked.iter().filter(|value| **value).count();
        if count == 0 {
            return Err(Error::Inference(
                "speech loss has no masked codebooks".into(),
            ));
        }
        let logits = self.logits(hidden, codes, masked)?;
        let device = logits.device();
        let n = codes.len();
        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(codes.iter().map(|code| i64::from(*code)).collect(), [n, 1]),
            &device,
        );
        let mask = Tensor::<B, 1>::from_data(
            TensorData::new(
                masked
                    .iter()
                    .map(|value| if *value { 1.0_f32 } else { 0.0 })
                    .collect(),
                [n],
            ),
            &device,
        );
        let selected = log_softmax(logits, 1).gather(1, targets).reshape([n]);
        Ok((-(selected * mask).sum(), count))
    }

    /// Reveal codebooks in the supplied order, using greedy selection on each pass.
    pub fn greedy_frame(
        &self,
        hidden: Tensor<B, 2>,
        order: &[usize],
        schedule: &[usize],
    ) -> Result<Vec<u32>> {
        self.greedy_frame_with_history(hidden, order, schedule, &[], 1.0)
    }

    /// As above, applying the generation repetition penalty to prior frames.
    pub fn greedy_frame_with_history(
        &self,
        hidden: Tensor<B, 2>,
        order: &[usize],
        schedule: &[usize],
        history: &[Vec<u32>],
        penalty: f32,
    ) -> Result<Vec<u32>> {
        self.sample_frame_with_history(
            hidden,
            order,
            schedule,
            history,
            AudioSampling {
                temperature: Some(0.0),
                top_k: None,
                repetition_penalty: penalty,
            },
            || 0.0,
        )
    }

    /// Reveal codebooks using official temperature/top-k sampling. A draw is
    /// requested for each selected codebook only when sampling is enabled.
    pub fn sample_frame_with_history(
        &self,
        hidden: Tensor<B, 2>,
        order: &[usize],
        schedule: &[usize],
        history: &[Vec<u32>],
        sampling: AudioSampling,
        mut draw: impl FnMut() -> f64,
    ) -> Result<Vec<u32>> {
        sampling.validate()?;
        let count = self.config.num_codebooks;
        let mut seen = vec![false; count];
        if order.len() != count
            || order
                .iter()
                .any(|&i| i >= count || std::mem::replace(&mut seen[i], true))
            || schedule.contains(&0)
            || schedule.iter().try_fold(0usize, |a, &b| a.checked_add(b)) != Some(count)
            || history.iter().any(|frame| {
                frame.len() != count
                    || frame
                        .iter()
                        .any(|&v| v as usize >= self.config.codebook_size)
            })
        {
            return Err(Error::Inference(
                "invalid speech codebook reveal order or schedule".into(),
            ));
        }
        let mut codes = vec![0; count];
        let mut masked = vec![true; count];
        let mut offset = 0;
        for &take in schedule {
            let logits = self
                .logits(hidden.clone(), &codes, &masked)?
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|e| Error::Inference(format!("speech code logits: {e}")))?;
            for &book in &order[offset..offset + take] {
                let row = &logits
                    [book * self.config.codebook_size..(book + 1) * self.config.codebook_size];
                let book_history: Vec<u32> = history.iter().map(|frame| frame[book]).collect();
                let value = if sampling.is_sampling() { draw() } else { 0.0 };
                codes[book] = sampling.sample_with_draw(row, &book_history, value)?;
            }
            for &book in &order[offset..offset + take] {
                masked[book] = false;
            }
            offset += take;
        }
        Ok(codes)
    }
}
