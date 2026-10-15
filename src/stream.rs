use crate::{
    Error, Model, Result, StreamConfig, TextPostprocess,
    audio::{MelFrontend, SAMPLE_RATE},
    model::{decoder::DecoderCache, encoder::EncoderCache},
    text::{StableTail, normalize, official_candidate, official_stable_text, parse_output},
};
use burn::tensor::{Tensor, backend::Backend};
use serde::Serialize;
#[cfg(not(feature = "wasm"))]
use std::time::Instant;
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
#[cfg(feature = "wasm")]
use web_time::Instant;

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

struct Span {
    end_sample: usize,
    text: String,
}

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
    base_sample: usize,
    fixed_raw: String,
    local_text: String,
    text: String,
    language: String,
    stable_tail: StableTail,
    spans: VecDeque<Span>,
    stop: Arc<AtomicBool>,
}

impl<B: Backend> Session<B> {
    pub fn new(model: Arc<Model<B>>, config: StreamConfig) -> Result<Self> {
        config.validate(model.config())?;
        let layers = model.config().thinker_config.text_config.num_hidden_layers;
        let decoder_cache = DecoderCache::with_paging(layers, config.paged_attention);
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
            base_sample: 0,
            fixed_raw: String::new(),
            local_text: String::new(),
            text: String::new(),
            language: String::new(),
            stable_tail: StableTail::default(),
            spans: VecDeque::new(),
            stop: Arc::new(AtomicBool::new(false)),
        })
    }
    pub fn text(&self) -> &str {
        &self.text
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
        StreamEvent {
            delta: String::new(),
            text: self.text.clone(),
            language: self.language.clone(),
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
            self.base_sample += discard;
            while self
                .spans
                .front()
                .is_some_and(|s| s.end_sample <= self.base_sample)
            {
                self.spans.pop_front();
            }
            self.local_text = self.spans.iter().map(|s| s.text.as_str()).collect();
            self.fixed_raw = crate::text::transcript_prefix(
                (self.config.language.is_none() && !self.language.is_empty())
                    .then_some(self.language.as_str()),
                &self.local_text,
            );
            self.encoder_cache.clear();
            self.decoder_cache.clear();
            self.stable_tail.clear();
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
            &self.fixed_raw,
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
        let official = self.config.text_postprocess == TextPostprocess::Official;
        let official_result = official.then(|| {
            official_candidate(
                &self.fixed_raw,
                &generated_text,
                self.config.language.as_deref(),
                final_chunk,
            )
        });
        let candidate = if let Some(result) = &official_result {
            result.raw.clone()
        } else {
            let candidate = normalize(&format!("{}{generated_text}", self.fixed_raw));
            if candidate.starts_with(&self.fixed_raw) {
                candidate
            } else {
                format!("{}{}", self.fixed_raw, normalize(&generated_text))
            }
        };
        let confirmed_tail = self.stable_tail.observe(
            &candidate,
            !final_chunk && stopped,
            self.consumed,
            self.config.stable_tail_ms * (SAMPLE_RATE / 1000),
        );
        let stable = if final_chunk || confirmed_tail {
            candidate
        } else {
            self.model
                .text()
                .stable_prefix(&candidate, self.config.unfixed_tokens)?
        };
        let (language, stable_text) = if let Some(result) = official_result {
            if !final_chunk && self.config.language.is_none() && !stable.contains("<asr_text>") {
                (String::new(), String::new())
            } else {
                let text = if final_chunk {
                    result.text
                } else {
                    official_stable_text(&stable)
                };
                (result.language, text)
            }
        } else {
            parse_output(&stable, self.config.language.as_deref())
        };
        let have_metadata = final_chunk
            || !official
            || self.config.language.is_some()
            || stable.contains("<asr_text>");
        let delta = if !have_metadata {
            // Match the no-reset Python path: partial language metadata is not
            // committed as a prompt prefix for the next chunk.
            String::new()
        } else if let Some(delta) = stable_text.strip_prefix(&self.local_text) {
            let delta = delta.to_owned();
            self.fixed_raw = if official {
                stable.trim().to_owned()
            } else {
                stable
            };
            self.local_text = stable_text;
            delta
        } else {
            // Re-tokenization can merge the last committed word with a new suffix.
            // Holding the previous prefix is safe; rewinding committed text is not.
            String::new()
        };
        if !language.is_empty() {
            self.language = language;
        }
        if !delta.is_empty() {
            self.text.push_str(&delta);
            self.spans.push_back(Span {
                end_sample: self.consumed,
                text: delta.clone(),
            });
        }
        self.reported_samples = self.consumed;
        Ok(StreamEvent {
            delta,
            text: self.text.clone(),
            language: self.language.clone(),
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
