//! A checkpoint-backed Omni model and its ordered text/media prompt path.
use crate::{
    audio,
    audio_encoder::{AudioEncoder, AudioEncoderConfig},
    chat::{ChatTemplate, RenderOptions, Tokenizer},
    codec::{Codec, CodecConfig},
    config::Config,
    depth::{DepthConfig, DepthDecoder},
    generate::{
        EndReason, GenerationOptions, GenerationResult, OmniEvent, OmniGenerationResult,
        OmniOptions, OutputMode, generate_greedy,
    },
    image::{self, ImageConfig},
    media::{self, MediaTokens},
    prepared::{MediaIds, PreparedMedia, bind_embeddings, bind_speech_frames},
    text::{TextCache, TextModel},
    video::{self, RgbFrame, VideoConfig},
    vision::{VisionEncoder, VisionEncoderConfig},
};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use omni_r2t2::{Error, Precision, Result, tensor::Embedding, weights::Weights};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Deserialize;
use std::{
    collections::VecDeque,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Deserialize)]
struct ModelSettings {
    text_config: TextSettings,
    num_codebooks: usize,
    codebook_size: usize,
    audio_out_token_id: u32,
    audio_start_token_id: u32,
    mixed_start_token_id: u32,
    text_end_token_id: u32,
    audio_eos_code: u32,
    interleaved_n_text: usize,
    interleaved_n_audio: usize,
    maskgit_schedule: Vec<usize>,
}
#[derive(Deserialize)]
struct TextSettings {
    hidden_size: usize,
}
#[derive(Deserialize)]
struct GenerationSettings {
    eos_token_id: serde_json::Value,
    #[serde(default)]
    audio_temperature: Option<f32>,
    #[serde(default)]
    audio_top_k: Option<usize>,
    #[serde(default)]
    audio_repetition_penalty: Option<f32>,
}
#[derive(Deserialize)]
struct ProcessorSettings {
    feature_extractor: FeatureSettings,
}
#[derive(Deserialize)]
struct FeatureSettings {
    feature_size: usize,
    sampling_rate: usize,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    log_floor: f64,
    normalization_offset: f64,
    normalization_divisor: f64,
    encoder_stride: usize,
}

/// Text and encoded-media embeddings ready for the language backbone.
pub struct PreparedPrompt<B: Backend> {
    pub ids: Vec<u32>,
    pub embeddings: Tensor<B, 3>,
}

/// A user turn may contain any ordered mixture of these byte-backed parts.
#[derive(Clone)]
pub enum InputPart {
    Text(String),
    Image(Vec<u8>),
    Audio(Vec<u8>),
    Video(Vec<u8>),
}

/// One loaded model. Request state is created separately for each conversation.
pub struct OmniModel<B: Backend> {
    pub text: TextModel<B>,
    pub audio: AudioEncoder<B>,
    pub vision: VisionEncoder<B>,
    pub depth: DepthDecoder<B>,
    pub codec: Codec<B>,
    pub tokenizer: Tokenizer,
    template: ChatTemplate,
    image_config: ImageConfig,
    video_config: VideoConfig,
    media_tokens: MediaTokens,
    media_ids: MediaIds,
    settings: ModelSettings,
    audio_sampling_defaults: crate::sampling::AudioSampling,
    eos_ids: Vec<u32>,
    codebook_embedding: Embedding<B>,
    device: B::Device,
}

/// Conversation-owned language and speech history. An interrupted or
/// length-limited answer is retained for display, but cannot be followed by a
/// new turn: clear it first, since it has no complete assistant end marker.
pub struct Conversation<B: Backend> {
    cache: TextCache<B>,
    state: ConversationState,
    tools: Option<Vec<u8>>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum ConversationState {
    Fresh,
    Complete,
    NeedsMessageEnd,
    Incomplete,
}

impl<B: Backend> Conversation<B> {
    pub fn new(model: &OmniModel<B>) -> Self {
        Self {
            cache: model.new_text_cache(),
            state: ConversationState::Fresh,
            tools: None,
        }
    }
    pub fn clear(&mut self) {
        self.cache.clear();
        self.state = ConversationState::Fresh;
        self.tools = None;
    }
    pub fn can_continue(&self) -> bool {
        self.state != ConversationState::Incomplete
    }

    /// Supply the complete initial messages, then only new user/tool messages.
    /// Generated speech codes and special tokens remain in the model cache.
    pub fn generate(
        &mut self,
        model: &OmniModel<B>,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        media: &[PreparedMedia<B>],
        options: &OmniOptions,
        cancelled: &AtomicBool,
        on_event: impl FnMut(&OmniEvent) -> bool,
    ) -> Result<OmniGenerationResult> {
        if self.state == ConversationState::Incomplete {
            return Err(Error::Inference("the previous answer was interrupted or reached its limit; clear the conversation before starting another turn".into()));
        }
        if cancelled.load(Ordering::Relaxed) {
            return Err(Error::Interrupted);
        }
        model.validate_omni_options(options)?;
        let tool_signature = serde_json::to_vec(&tools)?;
        if self.state != ConversationState::Fresh && self.tools.as_ref() != Some(&tool_signature) {
            return Err(Error::Inference(
                "tool definitions cannot change inside a cached conversation".into(),
            ));
        }
        if self.state == ConversationState::NeedsMessageEnd {
            let ids = model.tokenizer.encode("<|im_end|>")?;
            if ids.len() != 1 {
                return Err(Error::Tokenizer(
                    "invalid assistant message-end token".into(),
                ));
            }
            model.text.forward(&ids, &mut self.cache)?;
        }
        self.state = ConversationState::Incomplete;
        let result = model.generate_omni(
            messages,
            tools,
            media,
            &mut self.cache,
            options,
            cancelled,
            on_event,
        )?;
        self.tools = Some(tool_signature);
        self.state = match result.reason {
            EndReason::Stop(id) if id == model.settings.text_end_token_id => {
                ConversationState::NeedsMessageEnd
            }
            EndReason::Stop(_) => ConversationState::Complete,
            EndReason::Limit | EndReason::Cancelled => ConversationState::Incomplete,
        };
        Ok(result)
    }
}
impl<B: Backend> OmniModel<B> {
    fn validate_omni_options(
        &self,
        options: &OmniOptions,
    ) -> Result<crate::sampling::AudioSampling> {
        if options.max_slots == 0 {
            return Err(Error::Inference("invalid Omni generation options".into()));
        }
        let sampling = options.effective_audio_sampling(self.audio_sampling_defaults)?;
        if let Some(order) = &options.reveal_order {
            let mut seen = vec![false; self.settings.num_codebooks];
            if order.len() != seen.len()
                || order
                    .iter()
                    .any(|&index| index >= seen.len() || std::mem::replace(&mut seen[index], true))
            {
                return Err(Error::Inference(
                    "invalid speech codebook reveal order".into(),
                ));
            }
        }
        Ok(sampling)
    }
    pub fn load(directory: &Path, precision: Precision, device: &B::Device) -> Result<Self> {
        Self::load_inner(directory, precision, device, None)
    }

    /// Load the decoder from the exact codec bytes already used by the caller.
    pub fn load_with_codec_weights(
        directory: &Path,
        precision: Precision,
        device: &B::Device,
        weights: &Weights,
        config: CodecConfig,
    ) -> Result<Self> {
        Self::load_inner(directory, precision, device, Some((weights, config)))
    }

    fn load_inner(
        directory: &Path,
        precision: Precision,
        device: &B::Device,
        codec_snapshot: Option<(&Weights, CodecConfig)>,
    ) -> Result<Self> {
        precision.validate_device::<B>(device)?;
        let settings: ModelSettings =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        if settings.num_codebooks == 0
            || settings.codebook_size == 0
            || settings.audio_eos_code as usize >= settings.codebook_size
            || settings.interleaved_n_text == 0
            || settings.interleaved_n_audio == 0
            || settings.maskgit_schedule.contains(&0)
            || settings
                .maskgit_schedule
                .iter()
                .try_fold(0usize, |a, &b| a.checked_add(b))
                != Some(settings.num_codebooks)
        {
            return Err(Error::Config(
                "invalid Omni generation configuration".into(),
            ));
        }
        let processor: ProcessorSettings =
            serde_json::from_slice(&std::fs::read(directory.join("processor_config.json"))?)?;
        let f = processor.feature_extractor;
        if f.feature_size != 128
            || f.sampling_rate != 16_000
            || f.n_fft != 400
            || f.win_length != 400
            || f.hop_length != 160
            || f.encoder_stride != 8
            || f.log_floor != -12.0
            || f.normalization_offset != 4.0
            || f.normalization_divisor != 4.0
        {
            return Err(Error::Config(
                "unsupported LFM3 audio processor configuration".into(),
            ));
        }
        let generation: GenerationSettings =
            serde_json::from_slice(&std::fs::read(directory.join("generation_config.json"))?)?;
        let audio_sampling_defaults = crate::sampling::AudioSampling {
            temperature: generation.audio_temperature,
            top_k: generation.audio_top_k,
            repetition_penalty: generation.audio_repetition_penalty.unwrap_or(1.0),
        };
        audio_sampling_defaults.validate().map_err(|error| {
            Error::Config(format!(
                "invalid checkpoint speech sampling settings: {error}"
            ))
        })?;
        let eos_ids: Vec<u32> = match generation.eos_token_id {
            serde_json::Value::Array(values) => values
                .into_iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| u32::try_from(n).ok())
                        .ok_or_else(|| Error::Config("invalid end token ID".into()))
                })
                .collect::<Result<_>>()?,
            serde_json::Value::Number(value) => vec![
                value
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| Error::Config("invalid end token ID".into()))?,
            ],
            _ => return Err(Error::Config("missing generation end token ID".into())),
        };
        if eos_ids.is_empty() {
            return Err(Error::Config("missing generation end token ID".into()));
        }
        let text_config = Config::load(directory)?.text_config;
        let audio_config = AudioEncoderConfig::load(directory)?;
        let vision_config = VisionEncoderConfig::load(directory)?;
        let depth_config = DepthConfig::load(directory)?;
        let image_config = ImageConfig::load(directory)?;
        let video_config = VideoConfig::load(directory)?;
        let media_tokens = MediaTokens::load(directory)?;
        let media_ids = MediaIds::load(directory)?;
        let template = ChatTemplate::load(directory)?;
        let tokenizer = Tokenizer::load(directory)?;
        let weights = Weights::open(directory)?;
        let text = TextModel::from_weights(&weights, text_config, precision, device)?;
        let audio = AudioEncoder::from_weights(&weights, audio_config, precision, device)?;
        let vision = VisionEncoder::from_weights(&weights, vision_config, precision, device)?;
        let depth = DepthDecoder::from_weights(&weights, depth_config, precision, device)?;
        let codebook_count = settings
            .num_codebooks
            .checked_mul(settings.codebook_size)
            .ok_or_else(|| Error::Config("codebook size overflow".into()))?;
        let codebook_embedding = Embedding::load(
            &weights,
            "model.codebook_embedding.weight",
            codebook_count,
            settings.text_config.hidden_size,
            precision,
            device,
        )?;
        let codec = match codec_snapshot {
            Some((weights, config)) => Codec::from_weights(weights, config, precision, device)?,
            None => {
                let codec_dir = directory.join("audio_tokenizer");
                let config = CodecConfig::load(&codec_dir)?;
                let weights = Weights::open(&codec_dir)?;
                Codec::from_weights(&weights, config, precision, device)?
            }
        };
        Ok(Self {
            text,
            audio,
            vision,
            depth,
            codec,
            tokenizer,
            template,
            image_config,
            video_config,
            media_tokens,
            media_ids,
            settings,
            audio_sampling_defaults,
            eos_ids,
            codebook_embedding,
            device: device.clone(),
        })
    }
    pub fn new_text_cache(&self) -> TextCache<B> {
        self.text.new_cache()
    }
    pub fn audio_sampling_defaults(&self) -> crate::sampling::AudioSampling {
        self.audio_sampling_defaults
    }
    /// The checkpoint template's assistant header followed by its live-audio marker.
    pub fn live_transcription_prompt(&self) -> Result<Vec<u32>> {
        let header = self.template.render(&[], None, RenderOptions::default())?;
        let marker = self
            .tokenizer
            .decode(&[self.settings.mixed_start_token_id], false)?;
        self.tokenizer.encode(&format!("{header}{marker}"))
    }
    pub(crate) fn live_parameters(&self) -> ([u32; 4], [u32; 2], u32, &[u32], &B::Device) {
        (
            [
                self.media_ids.audio,
                self.settings.audio_out_token_id,
                self.media_ids.image,
                self.media_ids.video,
            ],
            [
                self.settings.audio_start_token_id,
                self.settings.mixed_start_token_id,
            ],
            self.settings.text_end_token_id,
            &self.eos_ids,
            &self.device,
        )
    }
    pub fn new_audio_stream(&self) -> crate::input_stream::AudioInputStream<B> {
        crate::input_stream::AudioInputStream::new(&self.audio, &self.device)
    }
    /// Decode and encode a user turn, preserving the supplied part order.
    pub fn prepare_parts(
        &self,
        parts: &[InputPart],
        cancelled: &AtomicBool,
    ) -> Result<(serde_json::Value, Vec<PreparedMedia<B>>)> {
        if parts.is_empty() {
            return Err(Error::Inference("user turn is empty".into()));
        }
        let mut content = Vec::with_capacity(parts.len());
        let mut media = Vec::new();
        for part in parts {
            if cancelled.load(Ordering::Relaxed) {
                return Err(Error::Interrupted);
            }
            match part {
                InputPart::Text(text) => {
                    content.push(serde_json::json!({"type":"text","text":text}))
                }
                InputPart::Image(bytes) => {
                    media.push(self.prepare_image(bytes)?);
                    content.push(serde_json::json!({"type":"image"}));
                }
                InputPart::Audio(bytes) => {
                    let pcm = crate::media_io::decode_audio_bytes(bytes, cancelled)?;
                    media.push(self.prepare_audio(&pcm, 16_000)?);
                    content.push(serde_json::json!({"type":"audio"}));
                }
                InputPart::Video(bytes) => {
                    let video =
                        crate::media_io::decode_video_bytes(bytes, &self.video_config, cancelled)?;
                    let soundtrack = video.audio.as_deref().map(|pcm| (pcm, 16_000));
                    media.push(self.prepare_video(
                        &video.frames,
                        video.indices,
                        video.fps,
                        soundtrack,
                    )?);
                    content.push(serde_json::json!({"type":"video"}));
                }
            }
        }
        Ok((serde_json::json!({"role":"user","content":content}), media))
    }
    pub fn prepare_image(&self, bytes: &[u8]) -> Result<PreparedMedia<B>> {
        let image = image::preprocess_encoded(bytes, &self.image_config)?;
        let count = image.grid_h * image.grid_w;
        let width = image.patches.len() / count;
        let patches =
            Tensor::from_data(TensorData::new(image.patches, [count, width]), &self.device);
        Ok(PreparedMedia::Image(self.vision.encode_frame(
            patches,
            image.grid_h,
            image.grid_w,
        )?))
    }
    pub fn prepare_audio(&self, samples: &[f32], sample_rate: u32) -> Result<PreparedMedia<B>> {
        if samples.is_empty() {
            return Err(Error::Audio("audio input is empty".into()));
        }
        let features = audio::extract(samples, sample_rate)?;
        let tensor = Tensor::from_data(
            TensorData::new(features.values, [1, 128, features.frames]),
            &self.device,
        );
        let encoded = self.audio.encode(tensor)?;
        let [_, tokens, width] = encoded.dims();
        Ok(PreparedMedia::Audio(encoded.reshape([tokens, width])))
    }
    /// Encode sampled RGB frames and their optional mono 16 kHz soundtrack.
    pub fn prepare_video(
        &self,
        frames: &[RgbFrame],
        frame_indices: Vec<usize>,
        fps: f64,
        soundtrack: Option<(&[f32], u32)>,
    ) -> Result<PreparedMedia<B>> {
        if frame_indices.len() != frames.len() {
            return Err(Error::Inference(
                "video frames and indices do not match".into(),
            ));
        }
        let patches = video::preprocess_video(frames, &self.video_config)?;
        let count = patches.grid_h * patches.grid_w;
        let mut encoded = Vec::with_capacity(patches.groups.len());
        for group in patches.groups {
            let width = group.len() / count;
            let tensor = Tensor::from_data(TensorData::new(group, [count, width]), &self.device);
            encoded.push(
                self.vision
                    .encode_frame(tensor, patches.grid_h, patches.grid_w)?,
            );
        }
        let soundtrack = if let Some((samples, rate)) = soundtrack {
            match self.prepare_audio(samples, rate)? {
                PreparedMedia::Audio(value) => Some(value),
                _ => unreachable!(),
            }
        } else {
            None
        };
        Ok(PreparedMedia::Video {
            frames: encoded,
            soundtrack,
            frame_indices,
            fps,
            temporal_patch_size: self.video_config.temporal_patch_size,
        })
    }
    pub fn prepare_prompt(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        media: &[PreparedMedia<B>],
    ) -> Result<PreparedPrompt<B>> {
        let rendered = self
            .template
            .render(messages, tools, RenderOptions::default())?;
        self.prepare_rendered(&rendered, messages, media)
    }
    fn prepare_suffix(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        media: &[PreparedMedia<B>],
    ) -> Result<PreparedPrompt<B>> {
        let suffix = self
            .template
            .render_suffix(messages, tools, RenderOptions::default())?;
        self.prepare_rendered(&format!("\n{suffix}"), messages, media)
    }
    fn prepare_rendered(
        &self,
        rendered: &str,
        messages: &[serde_json::Value],
        media: &[PreparedMedia<B>],
    ) -> Result<PreparedPrompt<B>> {
        let specs = media
            .iter()
            .map(PreparedMedia::spec)
            .collect::<Result<Vec<_>>>()?;
        let expanded = media::expand(rendered, &specs, &self.media_tokens)?;
        let ids = self.tokenizer.encode(&expanded.text)?;
        let embeddings = self.text.embed_tokens(&ids)?;
        let embeddings =
            bind_embeddings(&ids, embeddings, &expanded.segments, media, self.media_ids)?;
        let mut frames = Vec::new();
        for message in messages {
            if let Some(parts) = message.get("content").and_then(|value| value.as_array()) {
                for part in parts {
                    if part.get("type").and_then(|value| value.as_str()) == Some("audio_out") {
                        if message.get("role").and_then(|value| value.as_str()) != Some("assistant")
                        {
                            return Err(Error::Inference(
                                "speech output frames must belong to an assistant message".into(),
                            ));
                        }
                        let codes: Vec<u32> =
                            serde_json::from_value(part.get("codes").cloned().ok_or_else(
                                || Error::Inference("assistant speech frame lacks codes".into()),
                            )?)?;
                        frames.push(self.speech_frame_embedding(&codes)?);
                    }
                }
            }
        }
        let embeddings =
            bind_speech_frames(&ids, embeddings, self.settings.audio_out_token_id, &frames)?;
        Ok(PreparedPrompt { ids, embeddings })
    }
    /// Generate a text-only continuation. For a nonempty cache, messages must
    /// describe only the uncached suffix of the conversation.
    pub fn generate_text(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        media: &[PreparedMedia<B>],
        cache: &mut TextCache<B>,
        max_tokens: usize,
        cancelled: &AtomicBool,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<GenerationResult> {
        let prompt = if cache.is_empty() {
            self.prepare_prompt(messages, tools, media)?
        } else {
            self.prepare_suffix(messages, tools, media)?
        };
        let options = GenerationOptions {
            max_tokens,
            stop_tokens: self.eos_ids.clone(),
            suppress_tokens: vec![
                self.media_ids.image,
                self.media_ids.audio,
                self.media_ids.video,
                self.settings.audio_out_token_id,
                self.settings.audio_start_token_id,
                self.settings.mixed_start_token_id,
            ],
        };
        generate_greedy(
            &self.text,
            prompt.embeddings,
            cache,
            &options,
            cancelled,
            on_token,
        )
    }
    /// Run the checkpoint's full-context ASR prompt on an encoded recording.
    pub fn transcribe_audio(
        &self,
        audio: PreparedMedia<B>,
        language: &str,
        max_tokens: usize,
        cancelled: &AtomicBool,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<GenerationResult> {
        let language = language.trim();
        if language.is_empty()
            || language.len() > 64
            || !language
                .bytes()
                .all(|c| c.is_ascii_alphabetic() || c == b' ' || c == b'-')
        {
            return Err(Error::Inference(
                "ASR language must be an English language name".into(),
            ));
        }
        if !matches!(audio, PreparedMedia::Audio(_)) {
            return Err(Error::Audio("ASR requires encoded audio".into()));
        }
        let messages = [
            serde_json::json!({"role":"system","content":format!("Perform ASR in {language}.")}),
            serde_json::json!({"role":"user","content":[{"type":"audio"}]}),
        ];
        let mut cache = self.new_text_cache();
        self.generate_text(
            &messages,
            None,
            &[audio],
            &mut cache,
            max_tokens,
            cancelled,
            on_token,
        )
    }
    /// Continue an ASR transcription after a previously confirmed prefix.
    /// The callback receives only newly generated suffix tokens.
    pub fn transcribe_audio_after_prefix(
        &self,
        audio: PreparedMedia<B>,
        language: &str,
        prefix: &str,
        max_tokens: usize,
        cancelled: &AtomicBool,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<GenerationResult> {
        if prefix.is_empty() {
            return self.transcribe_audio(audio, language, max_tokens, cancelled, on_token);
        }
        let language = language.trim();
        if language.is_empty()
            || language.len() > 64
            || !language
                .bytes()
                .all(|c| c.is_ascii_alphabetic() || c == b' ' || c == b'-')
        {
            return Err(Error::Inference(
                "ASR language must be an English language name".into(),
            ));
        }
        if !matches!(audio, PreparedMedia::Audio(_)) {
            return Err(Error::Audio("ASR requires encoded audio".into()));
        }
        let messages = [
            serde_json::json!({"role":"system","content":format!("Perform ASR in {language}.")}),
            serde_json::json!({"role":"user","content":[{"type":"audio"}]}),
        ];
        let prompt = self.prepare_prompt(&messages, None, &[audio])?;
        let ids = self.tokenizer.encode(prefix)?;
        if ids.is_empty() {
            return Err(Error::Inference(
                "confirmed ASR prefix has no tokens".into(),
            ));
        }
        let embeddings = Tensor::cat(vec![prompt.embeddings, self.text.embed_tokens(&ids)?], 1);
        let options = GenerationOptions {
            max_tokens,
            stop_tokens: self.eos_ids.clone(),
            suppress_tokens: vec![
                self.media_ids.image,
                self.media_ids.audio,
                self.media_ids.video,
                self.settings.audio_out_token_id,
                self.settings.audio_start_token_id,
                self.settings.mixed_start_token_id,
            ],
        };
        generate_greedy(
            &self.text,
            embeddings,
            &mut self.new_text_cache(),
            &options,
            cancelled,
            on_token,
        )
    }
    /// Teacher-force speech frames and return the preceding hidden state for
    /// each frame, ready for the speech depth decoder's training loss.
    pub fn teacher_forced_speech_states(
        &self,
        messages: &[serde_json::Value],
        frames: &[Vec<u32>],
    ) -> Result<Tensor<B, 3>> {
        self.teacher_forced_speech_states_with_media(messages, &[], frames)
    }

    /// The same speech objective, with media bound through the normal prompt path.
    pub fn teacher_forced_speech_states_with_media(
        &self,
        messages: &[serde_json::Value],
        media: &[PreparedMedia<B>],
        frames: &[Vec<u32>],
    ) -> Result<Tensor<B, 3>> {
        if frames.is_empty() {
            return Err(Error::Audio(
                "speech training needs at least one frame".into(),
            ));
        }
        let prompt = self.prepare_prompt(messages, None, media)?;
        if prompt.ids.is_empty() {
            return Err(Error::Inference("speech training prompt is empty".into()));
        }
        let first = prompt.ids.len();
        let mut embeddings = Vec::with_capacity(frames.len() + 2);
        embeddings.push(prompt.embeddings);
        embeddings.push(
            self.text
                .embed_tokens(&[self.settings.audio_start_token_id])?,
        );
        for frame in frames {
            embeddings.push(self.speech_frame_embedding(frame)?);
        }
        let hidden = self
            .text
            .forward_embeddings(Tensor::cat(embeddings, 1), &mut self.new_text_cache())?;
        let width = hidden.dims()[2];
        Ok(hidden.slice([0..1, first..first + frames.len(), 0..width]))
    }

    /// Predict answer tokens after a prompt assembled by the inference path.
    pub fn teacher_forced_text_logits(
        &self,
        messages: &[serde_json::Value],
        media: &[PreparedMedia<B>],
        targets: &[u32],
    ) -> Result<Tensor<B, 3>> {
        let prompt = self.prepare_prompt(messages, None, media)?;
        self.text.teacher_forced_logits(prompt.embeddings, targets)
    }

    /// Number of codebooks, vocabulary size, and all-codebook end marker.
    pub fn speech_code_format(&self) -> (usize, usize, u32) {
        (
            self.settings.num_codebooks,
            self.settings.codebook_size,
            self.settings.audio_eos_code,
        )
    }

    pub fn speech_frame_embedding(&self, codes: &[u32]) -> Result<Tensor<B, 3>> {
        if codes.len() != self.settings.num_codebooks
            || codes
                .iter()
                .any(|id| *id as usize >= self.settings.codebook_size)
        {
            return Err(Error::Audio("invalid speech codebook frame".into()));
        }
        let ids: Vec<u32> = codes
            .iter()
            .enumerate()
            .map(|(i, id)| (i * self.settings.codebook_size + *id as usize) as u32)
            .collect();
        Ok(self.codebook_embedding.embed(&ids)?.sum_dim(0).reshape([
            1,
            1,
            self.settings.text_config.hidden_size,
        ]))
    }
}

fn publish(
    event: OmniEvent,
    result: &mut OmniGenerationResult,
    audio_token: u32,
    on_event: &mut impl FnMut(&OmniEvent) -> bool,
) -> bool {
    match &event {
        OmniEvent::TextToken(token) => result.tokens.push(*token),
        OmniEvent::SpeechCodes(codes) => {
            result.tokens.push(audio_token);
            result.audio_frames.push(codes.clone());
        }
        OmniEvent::Pcm { samples, .. } => result.samples.extend_from_slice(samples),
    }
    on_event(&event)
}

impl<B: Backend> OmniModel<B> {
    /// Generate ordered text, speech codes and 24 kHz audio chunks. `cache` and
    /// codec state are request-owned; returning false from `on_event` cancels.
    pub fn generate_omni(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        media: &[PreparedMedia<B>],
        cache: &mut TextCache<B>,
        options: &OmniOptions,
        cancelled: &AtomicBool,
        mut on_event: impl FnMut(&OmniEvent) -> bool,
    ) -> Result<OmniGenerationResult> {
        let mut result = OmniGenerationResult {
            tokens: Vec::new(),
            audio_frames: Vec::new(),
            samples: Vec::new(),
            reason: EndReason::Limit,
        };
        let audio_sampling = self.validate_omni_options(options)?;
        if cancelled.load(Ordering::Relaxed) {
            result.reason = EndReason::Cancelled;
            return Ok(result);
        }
        let prompt = if cache.is_empty() {
            self.prepare_prompt(messages, tools, media)?
        } else {
            self.prepare_suffix(messages, tools, media)?
        };
        let last = *prompt
            .ids
            .last()
            .ok_or_else(|| Error::Inference("empty prompt".into()))?;
        let mut mode = if options.mode == OutputMode::Auto {
            if last == self.settings.audio_start_token_id
                || last == self.settings.audio_out_token_id
            {
                OutputMode::Audio
            } else if last == self.settings.mixed_start_token_id {
                OutputMode::Interleaved
            } else {
                OutputMode::Text
            }
        } else {
            options.mode
        };
        let marker = match options.mode {
            OutputMode::Audio
                if last != self.settings.audio_start_token_id
                    && last != self.settings.audio_out_token_id =>
            {
                Some(self.settings.audio_start_token_id)
            }
            OutputMode::Interleaved if last != self.settings.mixed_start_token_id => {
                Some(self.settings.mixed_start_token_id)
            }
            _ => None,
        };
        let prompt_embeddings = if let Some(marker) = marker {
            Tensor::cat(
                vec![prompt.embeddings, self.text.embed_tokens(&[marker])?],
                1,
            )
        } else {
            prompt.embeddings
        };
        let hidden = self.text.forward_embeddings(prompt_embeddings, cache)?;
        let [_, n, width] = hidden.dims();
        let mut hidden = hidden.slice([0..1, n - 1..n, 0..width]);
        let mut rng = StdRng::seed_from_u64(options.seed);
        let mut history = VecDeque::<Vec<u32>>::new();
        let mut codec_stream = None;
        let mut slots_left = self.settings.interleaved_n_text;
        let mut speaking = false;
        let mut text_done = false;
        for _ in 0..options.max_slots {
            if cancelled.load(Ordering::Relaxed) {
                result.reason = EndReason::Cancelled;
                return Ok(result);
            }
            let interleaving = mode == OutputMode::Interleaved;
            let audio_step = mode == OutputMode::Audio || (interleaving && speaking);
            let mut events = Vec::new();
            let mut stop = None;
            let next_embedding = if audio_step {
                let mut order: Vec<usize> = (0..self.settings.num_codebooks).collect();
                if let Some(fixed) = &options.reveal_order {
                    order.clone_from(fixed);
                } else {
                    order.shuffle(&mut rng);
                }
                let prior: Vec<_> = history.iter().cloned().collect();
                let mut codes = self.depth.sample_frame_with_history(
                    hidden.clone().reshape([1, width]),
                    &order,
                    &self.settings.maskgit_schedule,
                    &prior,
                    audio_sampling,
                    || rng.random::<f64>(),
                )?;
                if codes.contains(&self.settings.audio_eos_code) {
                    codes.fill(self.settings.audio_eos_code);
                    mode = OutputMode::Text;
                    events.push(OmniEvent::SpeechCodes(codes.clone()));
                    if let Some(mut stream) = codec_stream.take() {
                        let tail = self.codec.decode_stream(&[], &mut stream, true)?;
                        if !tail.is_empty() {
                            events.push(OmniEvent::Pcm {
                                samples: tail,
                                sample_rate: self.codec.sample_rate(),
                            });
                        }
                    }
                } else {
                    if options.audio_repetition_window > 0 {
                        history.push_back(codes.clone());
                        if history.len() > options.audio_repetition_window {
                            history.pop_front();
                        }
                    }
                    events.push(OmniEvent::SpeechCodes(codes.clone()));
                    let stream = codec_stream.get_or_insert_with(|| self.codec.new_stream());
                    let chunk = self.codec.decode_stream(&[codes.clone()], stream, false)?;
                    if !chunk.is_empty() {
                        events.push(OmniEvent::Pcm {
                            samples: chunk,
                            sample_rate: self.codec.sample_rate(),
                        });
                    }
                }
                self.speech_frame_embedding(&codes)?
            } else {
                let logits = self
                    .text
                    .logits(hidden.clone())
                    .into_data()
                    .convert::<f32>()
                    .to_vec::<f32>()
                    .map_err(|e| Error::Inference(format!("read logits: {e}")))?;
                let mut suppress = vec![
                    self.media_ids.image,
                    self.media_ids.audio,
                    self.media_ids.video,
                    self.settings.audio_out_token_id,
                ];
                if options.mode == OutputMode::Text {
                    suppress.extend([
                        self.settings.audio_start_token_id,
                        self.settings.mixed_start_token_id,
                    ]);
                }
                let token = logits
                    .iter()
                    .enumerate()
                    .filter(|(i, v)| v.is_finite() && !suppress.contains(&(*i as u32)))
                    .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                    .ok_or_else(|| Error::Inference("no finite unsuppressed text logits".into()))?
                    .0 as u32;
                events.push(OmniEvent::TextToken(token));
                if token == self.settings.text_end_token_id {
                    if interleaving {
                        text_done = true;
                        speaking = true;
                    } else {
                        stop = Some(token);
                    }
                } else if self.eos_ids.contains(&token) {
                    stop = Some(token);
                } else if !interleaving && token == self.settings.audio_start_token_id {
                    mode = OutputMode::Audio;
                } else if !interleaving && token == self.settings.mixed_start_token_id {
                    mode = OutputMode::Interleaved;
                    slots_left = self.settings.interleaved_n_text;
                    speaking = false;
                    text_done = false;
                }
                self.text.embed_tokens(&[token])?
            };
            if interleaving && !text_done {
                slots_left -= 1;
                if slots_left == 0 {
                    speaking = !speaking;
                    slots_left = if speaking {
                        self.settings.interleaved_n_audio
                    } else {
                        self.settings.interleaved_n_text
                    };
                }
            }
            hidden = self.text.forward_embeddings(next_embedding, cache)?;
            for event in events {
                if !publish(
                    event,
                    &mut result,
                    self.settings.audio_out_token_id,
                    &mut on_event,
                ) || cancelled.load(Ordering::Relaxed)
                {
                    result.reason = EndReason::Cancelled;
                    return Ok(result);
                }
            }
            if let Some(token) = stop {
                result.reason = EndReason::Stop(token);
                break;
            }
        }
        if let Some(mut stream) = codec_stream {
            let tail = self.codec.decode_stream(&[], &mut stream, true)?;
            if !tail.is_empty()
                && !publish(
                    OmniEvent::Pcm {
                        samples: tail,
                        sample_rate: self.codec.sample_rate(),
                    },
                    &mut result,
                    self.settings.audio_out_token_id,
                    &mut on_event,
                )
            {
                result.reason = EndReason::Cancelled;
            }
        }
        Ok(result)
    }
}
