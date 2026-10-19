//! Breeze-TTS-2 native streaming inference. Python is only used by validation tools.
mod batch;
pub mod codec;
pub mod config;
pub mod enhancement;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod metal_depth;
mod ops;
pub mod profile;
pub mod reference;
mod routing;
pub mod sampling;
mod synthesis;
mod transformer;
pub use batch::{BatchStats, RequestId, SynthesisBatch};

use burn::tensor::{Bool, FloatDType, Int, Tensor, TensorData, backend::Backend};
pub use config::{CODEBOOKS, SAMPLE_RATE, SAMPLES_PER_FRAME};
use omni_r2t2::weights::Weights;
pub use omni_r2t2::{Error, Precision, Result};
use ops::*;
pub use reference::ReferenceAudio;
pub use sampling::Sampling;
pub type TraceTensor = (String, Vec<usize>, Vec<f32>);
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};
use transformer::*;

/// Depth kernel selection; unsupported backends/shapes use the stock operations.
/// All modes retain the existing weight precision and FP32 residual boundaries.
#[derive(Clone, Copy, Debug, Default, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DepthKernelMode {
    Stock,
    Gemv,
    Norm,
    #[default]
    GemvNorm,
}

pub struct Model<B: Backend> {
    pub config: config::ModelConfig,
    tokenizer: tokenizers::Tokenizer,
    text_embeddings: Embedding<B>,
    eoi: Tensor<B, 1>,
    text_encoder: Transformer<B>,
    text_projection: Linear<B>,
    audio_embeddings: Vec<Tensor<B, 2>>,
    backbone: Transformer<B>,
    head: Linear<B>,
    depth: Transformer<B>,
    depth_projection: Linear<B>,
    depth_head: Tensor<B, 3>,
    pub codec: codec::Codec<B>,
    precision: Precision,
    device: B::Device,
}

impl<B: Backend> Model<B> {
    /// Change only the depth kernels, allowing same-model interleaved measurements.
    pub fn set_depth_kernel_mode(&mut self, mode: DepthKernelMode) {
        self.depth.set_kernel_mode(mode);
        self.depth_projection.mode = mode;
    }
}

#[derive(Clone, Debug)]
pub struct SynthesisOptions {
    pub text: String,
    pub instruction: String,
    /// Official single CFG for instruction-following; 1 disables the second branch.
    pub guidance_scale: f32,
    pub sampling: Sampling,
    /// Safety ceiling, measured in 80 ms frames; a ceiling hit is reported explicitly.
    pub max_frames: usize,
    /// Number of new codec frames per output callback; never re-decodes old audio.
    pub chunk_frames: usize,
    /// Diagnostic fallback; device sampling avoids 15 CPU/GPU round trips per frame.
    pub cpu_depth_sampling: bool,
    /// Use paged KV for the long-lived backbone; depth/codec keep bounded caches.
    pub paged_attention: bool,
    /// Reference transcript and native Mimi/RVQ codes for voice cloning.
    pub reference: Option<ReferenceAudio>,
}
impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            text: String::new(),
            instruction: String::new(),
            guidance_scale: 4.0,
            sampling: Sampling::default(),
            max_frames: 750,
            chunk_frames: 2,
            cpu_depth_sampling: false,
            paged_attention: false,
            reference: None,
        }
    }
}
impl SynthesisOptions {
    fn validate(&self) -> Result<()> {
        if self.max_frames == 0
            || !(1..=32).contains(&self.chunk_frames)
            || !self.guidance_scale.is_finite()
            || !(1.0..=10.0).contains(&self.guidance_scale)
        {
            return Err(Error::Config("invalid synthesis limits/guidance".into()));
        }
        Ok(())
    }
}
#[derive(Debug, serde::Serialize)]
pub struct SynthesisReport {
    pub frames: usize,
    pub audio_seconds: f64,
    pub elapsed_seconds: f64,
    pub first_audio_seconds: Option<f64>,
    pub stopped: bool,
    pub reached_limit: bool,
    pub codes: Vec<[u32; CODEBOOKS]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<profile::Profile>,
}

struct Prompt {
    ids: Vec<u32>,
    target: Vec<u32>,
    reference_text: Option<Vec<u32>>,
}

impl<B: Backend> Model<B> {
    pub fn load(
        directory: &Path,
        precision: Precision,
        device: &B::Device,
        progress: impl FnMut(f64, &str),
    ) -> Result<Self> {
        Self::load_cancellable(
            directory,
            precision,
            device,
            &AtomicBool::new(false),
            progress,
        )
    }
    pub fn load_cancellable(
        directory: &Path,
        precision: Precision,
        device: &B::Device,
        cancel: &AtomicBool,
        mut progress: impl FnMut(f64, &str),
    ) -> Result<Self> {
        use omni_r2t2::{
            loading::{CancellableWeights, check},
            weights::WeightSource,
        };
        check(cancel)?;
        progress(0.0, "检查 Breeze-TTS-2 模型文件");
        check(cancel)?;
        let config = config::ModelConfig::load(directory)?;
        check(cancel)?;
        let tokenizer = tokenizers::Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        check(cancel)?;
        let weights = Weights::open(directory)?;
        let w = CancellableWeights::new(&weights, cancel, |_: &str| {});
        let t = &config.text_encoder_config;
        let b = &config.backbone_config;
        let dep = &config.depth_decoder_config;
        progress(0.02, "载入文本词表");
        let text_embeddings = Embedding::load(
            &w,
            "text_encoder.embed_tokens.weight",
            t.vocab_size,
            t.hidden_size,
            precision,
            device,
        )?;
        let eoi = w.tensor(
            "text_encoder.embed_tokens.eoi_embedding",
            [t.hidden_size],
            precision,
            device,
        )?;
        let text_encoder = Transformer::load(
            &w,
            "text_encoder",
            t,
            Architecture::Gemma,
            precision,
            device,
            &mut |i| {
                progress(
                    0.06 + 0.26 * i as f64 / t.num_hidden_layers as f64,
                    &format!("文本编码器 {i}/{}", t.num_hidden_layers),
                )
            },
        )?;
        let text_projection = Linear::load(
            &w,
            "text_encoder_proj",
            t.hidden_size,
            b.hidden_size,
            false,
            precision,
            device,
        )?;
        let audio_embeddings = (0..CODEBOOKS)
            .map(|i| {
                w.rows(
                    "depth_decoder.model.embed_tokens.weight",
                    [
                        config.num_codebooks * config.vocab_size,
                        config.audio_embed_size,
                    ],
                    i * config.vocab_size,
                    (i + 1) * config.vocab_size,
                    precision,
                    device,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let backbone = Transformer::load(
            &w,
            "backbone_model",
            b,
            Architecture::Qwen,
            precision,
            device,
            &mut |i| {
                progress(
                    0.33 + 0.32 * i as f64 / b.num_hidden_layers as f64,
                    &format!("语音生成器 {i}/{}", b.num_hidden_layers),
                )
            },
        )?;
        let head = Linear::load(
            &w,
            "lm_head",
            b.hidden_size,
            config.vocab_size + 1,
            false,
            precision,
            device,
        )?;
        let depth = Transformer::load(
            &w,
            "depth_decoder.model",
            dep,
            Architecture::Llama,
            precision,
            device,
            &mut |i| {
                progress(
                    0.65 + 0.19 * i as f64 / dep.num_hidden_layers as f64,
                    &format!("声学解码器 {i}/{}", dep.num_hidden_layers),
                )
            },
        )?;
        let depth_projection = Linear::load(
            &w,
            "depth_decoder.model.inputs_embeds_projector",
            config.audio_embed_size,
            dep.hidden_size,
            false,
            precision,
            device,
        )?;
        let depth_head = w.tensor(
            "depth_decoder.codebooks_head.weight",
            [CODEBOOKS - 1, dep.hidden_size, config.vocab_size],
            precision,
            device,
        )?;
        progress(0.86, "载入流式音频解码器");
        let codec =
            codec::Codec::load_cancellable(&directory.join("audio_tokenizer"), device, cancel)?;
        check(cancel)?;
        progress(1.0, "Breeze-TTS-2 已就绪");
        check(cancel)?;
        let mut model = Self {
            config,
            tokenizer,
            text_embeddings,
            eoi,
            text_encoder,
            text_projection,
            audio_embeddings,
            backbone,
            head,
            depth,
            depth_projection,
            depth_head,
            codec,
            precision,
            device: device.clone(),
        };
        model.set_depth_kernel_mode(DepthKernelMode::default());
        Ok(model)
    }

    /// Match upstream templates.py: tokenize each text segment with special tokens,
    /// render it, then tokenize the assembled prompt without adding specials again.
    pub fn prompt_ids(&self, text: &str, instruction: &str) -> Result<Vec<u32>> {
        if text.trim().is_empty() {
            return Err(Error::Config("请输入要合成的文本".into()));
        }
        let rendered = if instruction.trim().is_empty() {
            format!("[S0]{text}")
        } else {
            format!("[S0]<ins_bos>{instruction}<ins_eos>{text}")
        };
        let encoded = self
            .tokenizer
            .encode(rendered, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let rendered = self
            .tokenizer
            .decode(encoded.get_ids(), false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let ids = self
            .tokenizer
            .encode(rendered, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?
            .get_ids()
            .to_vec();
        if ids.is_empty() || ids.len() > 2048 {
            return Err(Error::Config(
                "文本和声音描述合计不得超过 2048 个 token，请分段合成".into(),
            ));
        }
        if ids.iter().any(|id| matches!(id, 262144 | 262145)) {
            return Err(Error::Config("文本不能包含音频占位符".into()));
        }
        Ok(ids)
    }

    fn text_input(&self, ids: &[u32]) -> Result<Tensor<B, 3>> {
        let c = &self.config.text_encoder_config;
        let scale = Tensor::<B, 1>::from_data([(c.hidden_size as f32).sqrt()], &self.device)
            .cast(self.precision.dtype());
        let mut x = self.text_embeddings.forward(ids)? * scale.unsqueeze::<3>();
        if ids.contains(&256000) {
            let mask = Tensor::<B, 3, Bool>::from_data(
                TensorData::new(
                    ids.iter().map(|&id| id == 256000).collect::<Vec<_>>(),
                    [1, ids.len(), 1],
                ),
                &self.device,
            );
            x = x.mask_where(mask, self.eoi.clone().reshape([1, 1, c.hidden_size]));
        }
        Ok(x)
    }
    pub fn encode_text(&self, ids: &[u32]) -> Result<Tensor<B, 3>> {
        let _routing = crate::routing::metal_phase::<B>(&self.device);
        Ok(self.text_projection.forward(
            self.text_encoder
                .forward(self.text_input(ids)?, &mut self.text_encoder.cache()),
        ))
    }
    fn prepare_prompt(
        &self,
        text: &str,
        instruction: &str,
        reference: Option<&ReferenceAudio>,
    ) -> Result<Prompt> {
        let target = self.prompt_ids(text, instruction)?;
        let mut ids = Vec::new();
        let reference_text = if let Some(reference) = reference {
            reference.validate()?;
            let tokens = self.prompt_ids(&reference.text, "")?;
            if tokens.len() + target.len() > 2048 {
                return Err(Error::Config(
                    "参考文字、目标文字和声音描述合计不得超过 2048 个 token".into(),
                ));
            }
            ids.extend_from_slice(&tokens);
            ids.extend(std::iter::repeat_n(262144, reference.codes.len()));
            ids.push(262145);
            Some(tokens)
        } else {
            None
        };
        ids.extend_from_slice(&target);
        if ids.len() >= self.config.backbone_config.max_position_embeddings {
            return Err(Error::Config("参考音频和文本超过模型上下文长度".into()));
        }
        Ok(Prompt {
            ids,
            target,
            reference_text,
        })
    }
    fn conditioning_input(
        &self,
        prompt: &Prompt,
        reference: Option<&ReferenceAudio>,
        cancel: &AtomicBool,
    ) -> Result<Option<Tensor<B, 3>>> {
        let _routing = crate::routing::metal_phase::<B>(&self.device);
        let encode = |ids: &[u32]| -> Result<Option<Tensor<B, 3>>> {
            Ok(self
                .text_encoder
                .forward_cancellable(
                    self.text_input(ids)?,
                    &mut self.text_encoder.cache(),
                    cancel,
                )
                .map(|x| self.text_projection.forward(x)))
        };
        let mut segments = Vec::new();
        if let (Some(ids), Some(reference)) = (&prompt.reference_text, reference) {
            let Some(text) = encode(ids)? else {
                return Ok(None);
            };
            segments.push(text);
            // Upstream fills the reference EOS with codebook_eos_token_id=0
            // in every codebook. This differs from generation's semantic EOS.
            let count = reference.codes.len() + 1;
            let embeddings = (0..CODEBOOKS)
                .map(|book| {
                    let ids: Vec<i32> = reference
                        .codes
                        .iter()
                        .map(|frame| frame[book] as i32)
                        .chain([0])
                        .collect();
                    self.audio_embeddings[book]
                        .clone()
                        .select(
                            0,
                            Tensor::from_data(TensorData::new(ids, [count]), &self.device),
                        )
                        .reshape([1, count, self.config.audio_embed_size])
                })
                .collect();
            segments.push(
                Tensor::cat(embeddings, 0)
                    .cast(FloatDType::F32)
                    .sum_dim(0)
                    .cast(self.precision.dtype()),
            );
        }
        // Each text segment has its own bidirectional attention and positions.
        let Some(target) = encode(&prompt.target)? else {
            return Ok(None);
        };
        segments.push(target);
        Ok(Some(Tensor::cat(segments, 1)))
    }
    fn prefill(
        &self,
        prompt: &Prompt,
        reference: Option<&ReferenceAudio>,
        cache: &mut TransformerCache<B>,
        cancel: &AtomicBool,
    ) -> Result<Option<Tensor<B, 3>>> {
        let _routing = crate::routing::metal_phase::<B>(&self.device);
        let Some(input) = self.conditioning_input(prompt, reference, cancel)? else {
            return Ok(None);
        };
        Ok(self
            .backbone
            .forward_cancellable(input, cache, cancel)
            .map(Self::last))
    }
    fn token(&self, id: u32) -> Tensor<B, 1, Int> {
        Tensor::from_data(TensorData::new(vec![id as i32], [1]), &self.device)
    }
    fn codebook_embedding(&self, codebook: usize, token: Tensor<B, 1, Int>) -> Tensor<B, 3> {
        self.audio_embeddings[codebook]
            .clone()
            .select(0, token)
            .reshape([1, 1, self.config.audio_embed_size])
    }
    fn last(x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, d] = x.dims();
        x.slice([0..b, n - 1..n, 0..d])
    }
    fn guided_logits(x: Tensor<B, 3>, guidance: f32) -> Tensor<B, 3> {
        let [b, n, d] = x.dims();
        debug_assert_eq!(n, 1);
        let x = x.cast(FloatDType::F32);
        if b == 2 {
            let cond = x.clone().slice([0..1, 0..1, 0..d]);
            let uncond = x.slice([1..2, 0..1, 0..d]);
            uncond.clone() + (cond - uncond) * guidance
        } else {
            x
        }
    }
    fn logits(x: Tensor<B, 3>, guidance: f32) -> Result<Vec<f32>> {
        let x = Self::guided_logits(x, guidance);
        pollster::block_on(x.into_data_async())
            .map_err(|e| Error::Inference(e.to_string()))?
            .to_vec::<f32>()
            .map_err(|e| Error::Inference(e.to_string()))
    }

    /// Diagnostic teacher-forced logits used to compare Rust against official PyTorch.
    pub fn trace(&self, ids: &[u32], first_codebook: u32) -> Result<Vec<TraceTensor>> {
        let text = self.encode_text(ids)?;
        self.trace_embeddings(text, first_codebook, false)
    }
    pub fn trace_conditioned(
        &self,
        text: &str,
        instruction: &str,
        reference: Option<&ReferenceAudio>,
        first_codebook: u32,
    ) -> Result<(Vec<u32>, Vec<TraceTensor>)> {
        self.trace_conditioned_with_paging(text, instruction, reference, first_codebook, false)
    }
    /// Teacher-forced diagnostic comparing backbone attention storage policies.
    pub fn trace_conditioned_with_paging(
        &self,
        text: &str,
        instruction: &str,
        reference: Option<&ReferenceAudio>,
        first_codebook: u32,
        paged_attention: bool,
    ) -> Result<(Vec<u32>, Vec<TraceTensor>)> {
        let prompt = self.prepare_prompt(text, instruction, reference)?;
        let input = self
            .conditioning_input(&prompt, reference, &AtomicBool::new(false))?
            .unwrap();
        Ok((
            prompt.ids,
            self.trace_embeddings(input, first_codebook, paged_attention)?,
        ))
    }
    /// Per-layer text encoder diagnostics for numerical backend comparisons.
    pub fn trace_text_encoder(
        &self,
        text: &str,
        instruction: &str,
    ) -> Result<(Vec<u32>, Vec<TraceTensor>)> {
        let ids = self.prompt_ids(text, instruction)?;
        let tensors = self.text_encoder.trace_layers(self.text_input(&ids)?);
        let trace = tensors
            .into_iter()
            .enumerate()
            .map(|(i, x)| {
                let shape = x.dims().to_vec();
                let data = pollster::block_on(x.cast(FloatDType::F32).into_data_async())
                    .map_err(|e| Error::Inference(e.to_string()))?
                    .to_vec::<f32>()
                    .map_err(|e| Error::Inference(e.to_string()))?;
                Ok((format!("text_stage_{i}"), shape, data))
            })
            .collect::<Result<_>>()?;
        Ok((ids, trace))
    }
    fn trace_embeddings(
        &self,
        text: Tensor<B, 3>,
        first_codebook: u32,
        paged_attention: bool,
    ) -> Result<Vec<TraceTensor>> {
        let mut backbone_cache = self.backbone.cache_with_paging(paged_attention);
        let hidden = Self::last(self.backbone.forward(text.clone(), &mut backbone_cache));
        let backbone_logits = self.head.forward(hidden.clone());
        let first = self.codebook_embedding(0, self.token(first_codebook));
        let mut frame_embeddings = vec![first.clone()];
        let input = self
            .depth_projection
            .forward(Tensor::cat(vec![hidden.clone(), first], 1));
        let mut depth_cache = self.depth.cache();
        let depth = Self::last(self.depth.forward(input, &mut depth_cache));
        let head = self
            .depth_head
            .clone()
            .slice([
                0..1,
                0..self.config.depth_decoder_config.hidden_size,
                0..self.config.vocab_size,
            ])
            .reshape([
                self.config.depth_decoder_config.hidden_size,
                self.config.vocab_size,
            ]);
        let logits = depth.matmul(head.unsqueeze());
        let mut all_logits = vec![logits.clone()];
        let mut frame: [u32; CODEBOOKS] = std::array::from_fn(|i| (i as u32 * 131) % 2048);
        frame[0] = first_codebook;
        for (i, &token) in frame.iter().enumerate().take(15).skip(1) {
            let input = self.codebook_embedding(i, self.token(token));
            frame_embeddings.push(input.clone());
            let output = self
                .depth
                .forward(self.depth_projection.forward(input), &mut depth_cache);
            let width = self.config.depth_decoder_config.hidden_size;
            let vocab = self.config.vocab_size;
            let head = self
                .depth_head
                .clone()
                .slice([i..i + 1, 0..width, 0..vocab])
                .reshape([1, width, vocab]);
            all_logits.push(output.matmul(head));
        }
        frame_embeddings.push(self.codebook_embedding(15, self.token(frame[15])));
        let embedding = Tensor::cat(frame_embeddings, 1)
            .cast(FloatDType::F32)
            .sum_dim(1)
            .cast(self.precision.dtype());
        let continuation = self.head.forward(
            self.backbone
                .forward(embedding.clone(), &mut backbone_cache),
        );
        [
            ("text", text),
            ("backbone_hidden", hidden),
            ("backbone_logits", backbone_logits),
            ("depth_logits", logits),
            ("depth_all_logits", Tensor::cat(all_logits, 1)),
            ("audio_embedding", embedding),
            ("continuation_logits", continuation),
        ]
        .into_iter()
        .map(|(name, x)| {
            let shape = x.dims().to_vec();
            let data = pollster::block_on(x.cast(FloatDType::F32).into_data_async())
                .map_err(|e| Error::Inference(e.to_string()))?
                .to_vec::<f32>()
                .map_err(|e| Error::Inference(e.to_string()))?;
            Ok((name.into(), shape, data))
        })
        .collect()
    }

    pub fn synthesize(
        &self,
        options: &SynthesisOptions,
        cancel: &AtomicBool,
        mut on_audio: impl FnMut(&[f32]) -> Result<()>,
    ) -> Result<SynthesisReport> {
        match synthesis::Synthesis::prepare(self, options, cancel, true)? {
            synthesis::Started::Running(request) => request.run_serial(self, cancel, &mut on_audio),
            synthesis::Started::Stopped(report) => Ok(report),
        }
    }
}
