use crate::{
    Error, Model, Result, StreamConfig,
    audio::{MelFrontend, SAMPLE_RATE},
    model::{decoder::DecoderCache, encoder::EncoderCache},
    text::TranscriptCommitter,
};
use burn::tensor::{Tensor, backend::Backend};
use serde::Serialize;
use std::time::Instant;
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
#[derive(Clone, Debug, Serialize)]
pub struct StreamEvent {
    /// Newly committed text. Concatenating deltas is always the final transcript.
    pub delta: String,
    pub text: String,
    pub language: String,
    pub audio_seconds: f64,
    pub compute_ms: f64,
    pub generated_tokens: usize,
    pub reused_audio_tokens: usize,
    pub is_final: bool,
    /// True if decoding exhausted the token budget rather than producing a stop.
    pub token_limit_reached: bool,
}

/// Audio after the first speech text that the model may keep answering
/// `language None` before that answer is committed anyway.
const UNKNOWN_LANGUAGE_PATIENCE_MS: usize = 8000;

/// A single streaming session. Model parameters are shared; caches are never shared
/// between streams. Call push sequentially and finish exactly once at end of input.
pub struct Session<B: Backend> {
    model: Arc<Model<B>>,
    config: StreamConfig,
    frontend: MelFrontend,
    encoder_cache: EncoderCache<B>,
    decoder_cache: DecoderCache<B>,
    pending: VecDeque<f32>,
    input_ended: bool,
    turn: Option<DecodeTurn<B>>,
    first: bool,
    finished: bool,
    consumed: usize,
    reported_samples: usize,
    transcript: TranscriptCommitter,
    stop: Arc<AtomicBool>,
}

impl<B: Backend> Session<B> {
    pub fn new(model: Arc<Model<B>>, config: StreamConfig) -> Result<Self> {
        config.validate(model.config())?;
        let layers = model.config().thinker_config.text_config.num_hidden_layers;
        let decoder_cache = DecoderCache::with_paging(layers, config.paged_attention);
        let transcript = TranscriptCommitter::new(
            config.text_postprocess,
            config.language.clone(),
            config.stable_tail_ms * (SAMPLE_RATE / 1000),
            UNKNOWN_LANGUAGE_PATIENCE_MS * (SAMPLE_RATE / 1000),
        );
        Ok(Self {
            model,
            config,
            frontend: MelFrontend::new(),
            encoder_cache: EncoderCache::default(),
            decoder_cache,
            pending: VecDeque::new(),
            input_ended: false,
            turn: None,
            first: true,
            finished: false,
            consumed: 0,
            reported_samples: 0,
            transcript,
            stop: Arc::new(AtomicBool::new(false)),
        })
    }
    pub fn text(&self) -> &str {
        self.transcript.text()
    }
    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    /// A separate handle can interrupt an awaited GPU readback without borrowing
    /// the session. Already submitted GPU work finishes; no new tokens are decoded.
    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }

    pub fn set_stop_handle(&mut self, stop: Arc<AtomicBool>) {
        self.stop = stop;
    }

    fn check_stop(&self) -> Result<()> {
        if self.stop.load(Ordering::Acquire) {
            Err(Error::Interrupted)
        } else {
            Ok(())
        }
    }

    /// End without decoding queued/pending audio or revising committed text.
    /// Call after any in-flight push/finish has returned. Idempotent.
    pub fn interrupt(&mut self) -> StreamEvent {
        self.stop.store(true, Ordering::Release);
        self.finished = true;
        self.pending.clear();
        self.turn = None;
        self.encoder_cache.clear();
        self.decoder_cache.clear();
        self.transcript.interrupt();
        StreamEvent {
            delta: String::new(),
            text: self.transcript.text().to_owned(),
            language: self.transcript.language().to_owned(),
            audio_seconds: self.reported_samples as f64 / SAMPLE_RATE as f64,
            compute_ms: 0.0,
            generated_tokens: 0,
            reused_audio_tokens: 0,
            is_final: true,
            token_limit_reached: false,
        }
    }

    pub async fn push(&mut self, pcm16k: &[f32]) -> Result<Vec<StreamEvent>> {
        self.receive(pcm16k, false)?;
        self.drain().await
    }

    pub async fn finish(&mut self) -> Result<StreamEvent> {
        self.receive(&[], true)?;
        let events = self.drain().await?;
        events
            .into_iter()
            .find(|event| event.is_final)
            .ok_or(Error::Interrupted)
    }

    fn append(&mut self, pcm: &[f32]) -> Result<()> {
        self.frontend
            .push(pcm)
            .map_err(|e| Error::Audio(e.into()))?;
        self.consumed += pcm.len();
        let maximum = self.config.window_seconds * SAMPLE_RATE;
        while self.frontend.samples() > maximum {
            let discard = maximum / 2;
            self.frontend.discard_prefix(discard);
            self.transcript.discard_prefix(discard);
            self.encoder_cache.clear();
            self.decoder_cache.clear();
        }
        Ok(())
    }

    fn begin_decode(&mut self, final_chunk: bool) -> Result<DecodeTurn<B>> {
        self.check_stop()?;
        let timer = Instant::now();
        // Final acoustic context is bounded separately from the rolling input
        // window. Do not append it as user audio: that would shift timestamps,
        // retire text at window boundaries and accumulate silence on retries.
        let mel = self.frontend.final_features(if final_chunk {
            self.config.final_padding_ms * (SAMPLE_RATE / 1000)
        } else {
            0
        });
        let encoded =
            self.model
                .encoder()
                .forward(&mel, &mut self.encoder_cache, self.config.reuse_cache)?;
        self.check_stop()?;
        let audio_len = encoded.embeddings.dims()[0];
        let hidden = self.model.config().thinker_config.text_config.hidden_size;
        let (before, after) = self.model.text().prompt(
            &self.config.context,
            self.config.language.as_deref(),
            self.transcript.prompt_prefix(),
        )?;
        let budget = if final_chunk {
            self.config.final_max_tokens
        } else {
            self.config.max_new_tokens
        };
        let prompt_len = before.len() + audio_len + after.len();
        if prompt_len + budget > self.config.max_context_tokens {
            return Err(Error::Inference(format!(
                "prompt ({prompt_len}) + generation ({budget}) exceeds max_context_tokens={}; reduce context/window size or raise the limit",
                self.config.max_context_tokens
            )));
        }
        let reused = if self.config.reuse_cache
            && self.decoder_cache.len >= before.len() + encoded.reused_tokens
        {
            before.len() + encoded.reused_tokens
        } else {
            0
        };
        self.decoder_cache.truncate(reused);
        let mut input = Vec::new();
        if reused == 0 {
            input.push(self.model.decoder().embed(&before)?);
        }
        let reused_audio = reused.saturating_sub(before.len());
        if reused_audio < audio_len {
            input.push(
                encoded
                    .embeddings
                    .slice([reused_audio..audio_len, 0..hidden]),
            );
        }
        input.push(self.model.decoder().embed(&after)?);
        let input = Tensor::cat(input, 0);
        Ok(DecodeTurn {
            input,
            cursor: 0,
            timer,
            final_chunk,
            reused_audio,
            budget,
            generated: Vec::new(),
            stopped: false,
        })
    }
    fn accept_token(&self, turn: &mut DecodeTurn<B>, token: u32) -> Result<bool> {
        if self.model.text().is_eos(token) {
            turn.stopped = true;
            return Ok(true);
        }
        turn.generated.push(token);
        if self.model.text().decode(&turn.generated)?.contains('|') {
            turn.stopped = true;
            return Ok(true);
        }
        Ok(turn.generated.len() == turn.budget)
    }
    fn commit_decode(&mut self, turn: DecodeTurn<B>) -> Result<StreamEvent> {
        let DecodeTurn {
            timer,
            final_chunk,
            reused_audio,
            budget,
            generated,
            stopped,
            ..
        } = turn;
        let generated_text = self.model.text().decode(&generated)?;
        let unfixed = self.config.unfixed_tokens;
        let text_processor = self.model.text();
        let delta = self.transcript.commit(
            &generated_text,
            self.consumed,
            stopped,
            final_chunk,
            |candidate| text_processor.stable_prefix(candidate, unfixed),
        )?;
        self.reported_samples = self.consumed;
        Ok(StreamEvent {
            delta,
            text: self.transcript.text().to_owned(),
            language: self.transcript.language().to_owned(),
            audio_seconds: self.consumed as f64 / SAMPLE_RATE as f64,
            compute_ms: timer.elapsed().as_secs_f64() * 1000.0,
            generated_tokens: generated.len(),
            reused_audio_tokens: reused_audio,
            is_final: final_chunk,
            token_limit_reached: !stopped && generated.len() == budget,
        })
    }
}

struct DecodeTurn<B: Backend> {
    input: Tensor<B, 2>,
    cursor: usize,
    timer: Instant,
    final_chunk: bool,
    reused_audio: usize,
    budget: usize,
    generated: Vec<u32>,
    stopped: bool,
}
impl<B: Backend> DecodeTurn<B> {
    fn next_input(
        &mut self,
        decoder: &crate::model::decoder::Decoder<B>,
        chunk: usize,
    ) -> Result<(Tensor<B, 2>, bool)> {
        let [n, width] = self.input.dims();
        if self.cursor < n {
            let end = (self.cursor + chunk).min(n);
            let input = self.input.clone().slice([self.cursor..end, 0..width]);
            self.cursor = end;
            Ok((input, end == n))
        } else {
            Ok((
                decoder.embed(&[*self.generated.last().expect("decoder has sampled a token")])?,
                true,
            ))
        }
    }
}
#[path = "stream_step.rs"]
mod step;
#[cfg(feature = "native")]
pub use crate::runner::batch::{BatchReport, BatchStats, TranscriptionBatch};
pub(crate) use step::{Advance, Phase};
