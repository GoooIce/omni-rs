//! Incremental transcription using the checkpoint's live audio/text protocol.
use crate::{
    TextCache,
    audio::{AudioFeatureStream, AudioFeatures},
    audio_encoder::AudioEncoderStream,
    model::OmniModel,
    prepared::PreparedMedia,
};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use omni_r2t2::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug)]
pub struct LiveOptions {
    pub max_new_tokens: usize,
    /// Additional tokens to suppress before the final input chunk.
    pub suppress_tokens: Vec<u32>,
    /// Additional tokens that stop the final input chunk.
    pub stop_tokens: Vec<u32>,
}
impl Default for LiveOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 48,
            suppress_tokens: Vec::new(),
            stop_tokens: Vec::new(),
        }
    }
}
pub struct LiveOutput {
    pub tokens: Vec<u32>,
    pub audio_tokens: usize,
    pub closed: bool,
    pub cancelled: bool,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    Unset,
    Pcm,
    Features,
}
/// One stream has separate audio encoder and text caches. A cancelled or final
/// stream is closed; create a new one for another recording.
pub struct LiveTranscription<B: Backend> {
    features: AudioFeatureStream,
    audio: AudioEncoderStream<B>,
    text: TextCache<B>,
    pending: Option<Tensor<B, 3>>,
    encoded: Vec<Tensor<B, 3>>,
    source: Source,
    samples: usize,
    finished: bool,
    audio_complete: bool,
}
impl<B: Backend> LiveTranscription<B> {
    /// `prompt` is the model-specific live-ASR prefix; it must not contain audio placeholders.
    pub fn new(model: &OmniModel<B>, prompt: &[u32]) -> Result<Self> {
        let (placeholders, _, _, _, _) = model.live_parameters();
        if prompt.is_empty() || prompt.iter().any(|id| placeholders.contains(id)) {
            return Err(Error::Inference(
                "live transcription prompt is empty or contains a media placeholder".into(),
            ));
        }
        Ok(Self {
            features: AudioFeatureStream::new(),
            audio: model.audio.new_stream(),
            text: model.new_text_cache(),
            pending: Some(model.text.embed_tokens(prompt)?),
            encoded: Vec::new(),
            source: Source::Unset,
            samples: 0,
            finished: false,
            audio_complete: false,
        })
    }
    pub fn push_pcm(
        &mut self,
        model: &OmniModel<B>,
        samples: &[f32],
        sample_rate: u32,
        is_final: bool,
        options: &LiveOptions,
        cancelled: &AtomicBool,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<LiveOutput> {
        self.validate(model, options, cancelled, Source::Pcm)?;
        let total = self
            .samples
            .checked_add(samples.len())
            .ok_or_else(|| Error::Audio("audio length overflow".into()))?;
        if is_final && total == 0 {
            return Err(Error::Audio("audio input is empty".into()));
        }
        let features = self.features.push(samples, sample_rate, is_final)?;
        self.samples = total;
        self.source = Source::Pcm;
        self.consume(model, features, is_final, options, cancelled, on_token)
    }
    /// Accepts normalized `[128, frames]` log-mel chunks using the same encoder
    /// and live decoder. Do not mix this method with `push_pcm` in one stream.
    pub fn push_features(
        &mut self,
        model: &OmniModel<B>,
        features: AudioFeatures,
        is_final: bool,
        options: &LiveOptions,
        cancelled: &AtomicBool,
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<LiveOutput> {
        self.validate(model, options, cancelled, Source::Features)?;
        if features.frames.checked_mul(128) != Some(features.values.len())
            || features.values.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Audio("invalid live audio features".into()));
        }
        self.source = Source::Features;
        self.consume(model, features, is_final, options, cancelled, on_token)
    }
    fn validate(
        &mut self,
        model: &OmniModel<B>,
        options: &LiveOptions,
        cancelled: &AtomicBool,
        source: Source,
    ) -> Result<()> {
        if self.finished {
            return Err(Error::Finished);
        }
        if self.source != Source::Unset && self.source != source {
            return Err(Error::Audio(
                "cannot mix PCM and feature chunks in one live stream".into(),
            ));
        }
        if options.max_new_tokens == 0
            || options
                .suppress_tokens
                .iter()
                .chain(&options.stop_tokens)
                .any(|id| *id as usize >= model.text.vocab_size())
        {
            return Err(Error::Inference(
                "invalid live transcription options".into(),
            ));
        }
        if cancelled.load(Ordering::Relaxed) {
            self.finished = true;
            return Err(Error::Interrupted);
        }
        Ok(())
    }
    fn consume(
        &mut self,
        model: &OmniModel<B>,
        features: AudioFeatures,
        is_final: bool,
        options: &LiveOptions,
        cancelled: &AtomicBool,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<LiveOutput> {
        // An error after encoder advancement invalidates this stream.
        self.finished = true;
        let (placeholders, starts, text_end, eos, device) = model.live_parameters();
        let input = (features.frames > 0).then(|| {
            Tensor::from_data(
                TensorData::new(features.values, [1, 128, features.frames]),
                device,
            )
        });
        let audio = model
            .audio
            .encode_stream(input, &mut self.audio, is_final)?;
        let count = audio.as_ref().map_or(0, |tensor| tensor.dims()[1]);
        if let Some(encoded) = &audio {
            self.encoded.push(encoded.clone());
        }
        let mut output = LiveOutput {
            tokens: Vec::new(),
            audio_tokens: count,
            closed: false,
            cancelled: false,
        };
        if cancelled.load(Ordering::Relaxed) {
            output.cancelled = true;
            return Ok(output);
        }
        if count == 0 && !is_final {
            self.finished = false;
            return Ok(output);
        }
        let input = match (self.pending.take(), audio) {
            (Some(prefix), Some(audio)) => Tensor::cat(vec![prefix, audio], 1),
            (Some(prefix), None) => prefix,
            (None, Some(audio)) => audio,
            (None, None) => return Err(Error::Audio("no live audio or pending prompt".into())),
        };
        let mut hidden = model.text.forward_embeddings(input, &mut self.text)?;
        for _ in 0..options.max_new_tokens {
            if cancelled.load(Ordering::Relaxed) {
                output.cancelled = true;
                return Ok(output);
            }
            let [_, n, width] = hidden.dims();
            let logits = model
                .text
                .logits(hidden.slice([0..1, n - 1..n, 0..width]))
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|error| Error::Inference(format!("read live logits: {error}")))?;
            let token = logits
                .iter()
                .enumerate()
                .filter(|(index, value)| {
                    let id = *index as u32;
                    value.is_finite()
                        && !placeholders.contains(&id)
                        && (is_final
                            || id == text_end
                            || (!starts.contains(&id)
                                && !eos.contains(&id)
                                && !options.suppress_tokens.contains(&id)))
                })
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .ok_or_else(|| Error::Inference("no finite live transcription logits".into()))?
                .0 as u32;
            if token == text_end
                || (is_final && (eos.contains(&token) || options.stop_tokens.contains(&token)))
            {
                output.closed = true;
                break;
            }
            hidden = model.text.forward(&[token], &mut self.text)?;
            output.tokens.push(token);
            if !on_token(token) || cancelled.load(Ordering::Relaxed) {
                output.cancelled = true;
                return Ok(output);
            }
        }
        if !is_final {
            self.pending = Some(model.text.embed_tokens(&[text_end])?);
            self.finished = false;
        } else {
            self.audio_complete = true;
        }
        Ok(output)
    }

    /// Reuse the encoded recording for a full-context answer after live input
    /// finishes. This permits a final transcription to replace provisional text.
    pub fn into_audio(self) -> Result<PreparedMedia<B>> {
        if !self.audio_complete || self.encoded.is_empty() {
            return Err(Error::Audio("live audio was not completed".into()));
        }
        let encoded = Tensor::cat(self.encoded, 1);
        let [_, tokens, width] = encoded.dims();
        Ok(PreparedMedia::Audio(encoded.reshape([tokens, width])))
    }
}
