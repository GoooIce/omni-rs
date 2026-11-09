//! Append-only confirmed text for the optional full-context ASR route.
use crate::{generate::EndReason, input_stream::AudioInputStream, model::OmniModel};
use burn::tensor::backend::Backend;
use omni_r2t2::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};

/// Holds the common prefix of successive transcription hypotheses.
/// Previously confirmed text is never revised.
#[derive(Default)]
pub struct ConfirmedText {
    text: String,
    confirmed: Vec<char>,
    previous: Vec<char>,
    finished: bool,
    conflict: bool,
}

fn punctuation(value: char) -> bool {
    matches!(
        value,
        '，' | '。' | '、' | '！' | '？' | '：' | '；' | ',' | '.' | '!' | '?' | ':' | ';'
    )
}

fn significant(value: &str) -> Vec<(char, usize)> {
    let mut result = Vec::new();
    let mut previous = None;
    let mut chars = value.char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        let numeric_separator = matches!(ch, '.' | ',')
            && previous.is_some_and(char::is_numeric)
            && chars.peek().is_some_and(|(_, next)| next.is_numeric());
        if !punctuation(ch) || numeric_separator {
            result.push((ch, index + ch.len_utf8()));
        }
        previous = Some(ch);
    }
    result
}

impl ConfirmedText {
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Return only newly confirmed text, holding back the most recent tail.
    pub fn observe(&mut self, candidate: &str) -> Result<String> {
        if self.finished {
            return Err(Error::Finished);
        }
        if self.conflict {
            return Err(Error::Inference(
                "final transcription conflict is unresolved".into(),
            ));
        }
        let candidate = candidate.replace('\u{fffd}', "");
        let chars = significant(&candidate);
        let current: Vec<char> = chars.iter().map(|(ch, _)| *ch).collect();
        let common = self
            .previous
            .iter()
            .zip(&current)
            .take_while(|(a, b)| a == b)
            .count();
        let mut delta = String::new();
        if current.starts_with(&self.confirmed) && common > self.confirmed.len() {
            let start = if self.confirmed.is_empty() {
                0
            } else {
                chars[self.confirmed.len() - 1].1
            };
            let end = chars[common - 1].1;
            delta.push_str(&candidate[start..end]);
            self.text.push_str(&delta);
            self.confirmed = current[..common].to_vec();
        }
        self.previous = current;
        Ok(delta)
    }

    /// Finalize when the new hypothesis agrees with committed characters.
    /// `None` signals a conflict; the committed text remains unchanged.
    pub fn finish(&mut self, candidate: &str) -> Result<Option<String>> {
        if self.finished {
            return Err(Error::Finished);
        }
        if self.conflict {
            return Err(Error::Inference(
                "final transcription conflict is unresolved".into(),
            ));
        }
        let candidate = candidate.replace('\u{fffd}', "");
        let chars = significant(&candidate);
        let current: Vec<char> = chars.iter().map(|(ch, _)| *ch).collect();
        if !current.starts_with(&self.confirmed) {
            self.conflict = true;
            return Ok(None);
        }
        let start = if self.confirmed.is_empty() {
            0
        } else {
            chars[self.confirmed.len() - 1].1
        };
        let delta = candidate[start..].to_owned();
        self.text.push_str(&delta);
        self.finished = true;
        Ok(Some(delta))
    }

    /// Complete a conflicting final hypothesis with a continuation generated
    /// from the already confirmed prefix.
    pub fn finish_with_suffix(&mut self, suffix: &str) -> Result<String> {
        if self.finished {
            return Err(Error::Finished);
        }
        if !self.conflict {
            return Err(Error::Inference(
                "no final transcription conflict to resolve".into(),
            ));
        }
        let suffix = suffix.replace('\u{fffd}', "");
        self.text.push_str(&suffix);
        self.finished = true;
        Ok(suffix)
    }
}

/// A newly confirmed, append-only portion of a transcription.
#[derive(Clone, Debug)]
pub struct StableEvent {
    pub delta: String,
    pub text: String,
    pub is_final: bool,
    /// The final unconstrained hypothesis disagreed with committed characters.
    pub conflict: bool,
}

pub struct StableOutput {
    pub text: String,
    /// `None` means more audio can be supplied.
    pub reason: Option<EndReason>,
    pub conflict: bool,
}

/// Optional ASR route: repeatedly decode completed audio and commit the common
/// prefix of successive hypotheses. Audio input is grouped into one-second
/// snapshots irrespective of the caller's packet boundaries.
pub struct StableTranscription<B: Backend> {
    audio: AudioInputStream<B>,
    text: ConfirmedText,
    pending: Vec<f32>,
    total_samples: usize,
    language: String,
    max_tokens: usize,
    finished: bool,
    conflict: bool,
}
impl<B: Backend> StableTranscription<B> {
    pub fn new(model: &OmniModel<B>, language: &str, max_tokens: usize) -> Result<Self> {
        let language = language.trim();
        if language.is_empty()
            || language.len() > 64
            || !language
                .bytes()
                .all(|c| c.is_ascii_alphabetic() || c == b' ' || c == b'-')
            || max_tokens == 0
        {
            return Err(Error::Inference(
                "invalid stable transcription options".into(),
            ));
        }
        Ok(Self {
            audio: model.new_audio_stream(),
            text: ConfirmedText::default(),
            pending: Vec::new(),
            total_samples: 0,
            language: language.to_owned(),
            max_tokens,
            finished: false,
            conflict: false,
        })
    }

    pub fn text(&self) -> &str {
        self.text.text()
    }

    fn output(&self, reason: Option<EndReason>) -> StableOutput {
        StableOutput {
            text: self.text.text().to_owned(),
            reason,
            conflict: self.conflict,
        }
    }

    /// Push 16 kHz mono PCM. Returning false from `on_event` cancels the stream;
    /// the returned output retains every previously delivered text delta.
    pub fn push_pcm(
        &mut self,
        model: &OmniModel<B>,
        samples: &[f32],
        sample_rate: u32,
        is_final: bool,
        cancelled: &AtomicBool,
        mut on_event: impl FnMut(&StableEvent) -> bool,
    ) -> Result<StableOutput> {
        if self.finished {
            return Err(Error::Finished);
        }
        if sample_rate != 16_000 || samples.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::Audio(
                "stable transcription expects finite 16 kHz mono PCM".into(),
            ));
        }
        let total = self
            .total_samples
            .checked_add(samples.len())
            .ok_or_else(|| Error::Audio("audio length overflow".into()))?;
        if total == 0 && is_final {
            return Err(Error::Audio("audio input is empty".into()));
        }
        if cancelled.load(Ordering::Acquire) {
            self.finished = true;
            return Ok(self.output(Some(EndReason::Cancelled)));
        }
        self.total_samples = total;
        self.pending.extend_from_slice(samples);
        let result = self.process(model, is_final, cancelled, &mut on_event);
        match result {
            Ok(reason) => Ok(self.output(reason)),
            Err(Error::Interrupted) if cancelled.load(Ordering::Acquire) => {
                self.finished = true;
                Ok(self.output(Some(EndReason::Cancelled)))
            }
            Err(error) => {
                self.finished = true;
                Err(error)
            }
        }
    }

    fn process(
        &mut self,
        model: &OmniModel<B>,
        is_final: bool,
        cancelled: &AtomicBool,
        on_event: &mut impl FnMut(&StableEvent) -> bool,
    ) -> Result<Option<EndReason>> {
        while self.pending.len() >= 16_000 {
            let tail = self.pending.split_off(16_000);
            let chunk = std::mem::replace(&mut self.pending, tail);
            let count = self
                .audio
                .push(&model.audio, &chunk, 16_000, false, cancelled)?;
            if count > 0
                && let Some(reason) = self.decode(model, false, cancelled, on_event)?
            {
                self.finished = true;
                return Ok(Some(reason));
            }
        }
        if is_final {
            let tail = std::mem::take(&mut self.pending);
            self.audio
                .push(&model.audio, &tail, 16_000, true, cancelled)?;
            let reason = self
                .decode(model, true, cancelled, on_event)?
                .ok_or_else(|| Error::Inference("stable transcription did not finish".into()))?;
            self.finished = true;
            return Ok(Some(reason));
        }
        Ok(None)
    }

    fn decode(
        &mut self,
        model: &OmniModel<B>,
        is_final: bool,
        cancelled: &AtomicBool,
        on_event: &mut impl FnMut(&StableEvent) -> bool,
    ) -> Result<Option<EndReason>> {
        let Some(audio) = self.audio.snapshot()? else {
            return if is_final {
                Err(Error::Audio("no encoded audio".into()))
            } else {
                Ok(None)
            };
        };
        let result =
            model.transcribe_audio(audio, &self.language, self.max_tokens, cancelled, |_| {
                !cancelled.load(Ordering::Acquire)
            })?;
        if result.reason == EndReason::Cancelled {
            return Ok(Some(EndReason::Cancelled));
        }
        let candidate = model.tokenizer.decode(&result.tokens, true)?;
        if !is_final {
            let delta = self.text.observe(&candidate)?;
            if !delta.is_empty()
                && !on_event(&StableEvent {
                    delta,
                    text: self.text.text().to_owned(),
                    is_final: false,
                    conflict: false,
                })
            {
                return Ok(Some(EndReason::Cancelled));
            }
            return Ok(None);
        }
        let (delta, reason) = if let Some(delta) = self.text.finish(&candidate)? {
            (delta, result.reason)
        } else {
            self.conflict = true;
            let audio = self
                .audio
                .snapshot()?
                .ok_or_else(|| Error::Audio("no encoded audio".into()))?;
            let forced = model.transcribe_audio_after_prefix(
                audio,
                &self.language,
                self.text.text(),
                self.max_tokens,
                cancelled,
                |_| !cancelled.load(Ordering::Acquire),
            )?;
            if forced.reason == EndReason::Cancelled {
                return Ok(Some(EndReason::Cancelled));
            }
            let suffix = model.tokenizer.decode(&forced.tokens, true)?;
            (self.text.finish_with_suffix(&suffix)?, forced.reason)
        };
        if !on_event(&StableEvent {
            delta,
            text: self.text.text().to_owned(),
            is_final: true,
            conflict: self.conflict,
        }) {
            return Ok(Some(EndReason::Cancelled));
        }
        Ok(Some(reason))
    }
}
