//! Request-owned incremental PCM preprocessing and audio encoding.
use crate::{
    audio::AudioFeatureStream,
    audio_encoder::{AudioEncoder, AudioEncoderStream},
    prepared::PreparedMedia,
};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use omni_r2t2::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};

/// Accepts mono 16 kHz PCM in arbitrary chunks and encodes completed audio
/// tokens while input continues. Finalization preserves the padded tail.
/// Discard a cancelled stream and start a fresh one for the next utterance.
pub struct AudioInputStream<B: Backend> {
    features: AudioFeatureStream,
    encoder: AudioEncoderStream<B>,
    device: B::Device,
    chunks: Vec<Tensor<B, 3>>,
    samples: usize,
    finished: bool,
    failed: bool,
}
impl<B: Backend> AudioInputStream<B> {
    pub fn new(encoder: &AudioEncoder<B>, device: &B::Device) -> Self {
        Self {
            features: AudioFeatureStream::new(),
            encoder: encoder.new_stream(),
            device: device.clone(),
            chunks: Vec::new(),
            samples: 0,
            finished: false,
            failed: false,
        }
    }
    /// Returns the number of newly encoded tokens, including before EOF.
    pub fn push(
        &mut self,
        encoder: &AudioEncoder<B>,
        samples: &[f32],
        sample_rate: u32,
        is_final: bool,
        cancelled: &AtomicBool,
    ) -> Result<usize> {
        if self.finished || self.failed {
            return Err(Error::Finished);
        }
        if cancelled.load(Ordering::Relaxed) {
            self.failed = true;
            return Err(Error::Interrupted);
        }
        let total = self
            .samples
            .checked_add(samples.len())
            .ok_or_else(|| Error::Audio("audio length overflow".into()))?;
        if total == 0 && is_final {
            return Err(Error::Audio("audio input is empty".into()));
        }
        let features = self.features.push(samples, sample_rate, is_final)?;
        let input = (features.frames > 0).then(|| {
            Tensor::from_data(
                TensorData::new(features.values, [1, 128, features.frames]),
                &self.device,
            )
        });
        let encoded = match encoder.encode_stream(input, &mut self.encoder, is_final) {
            Ok(encoded) => encoded,
            Err(error) => {
                self.failed = true;
                return Err(error);
            }
        };
        if cancelled.load(Ordering::Relaxed) {
            self.failed = true;
            return Err(Error::Interrupted);
        }
        let count = encoded.as_ref().map_or(0, |value| value.dims()[1]);
        if let Some(encoded) = encoded {
            self.chunks.push(encoded);
        }
        self.samples = total;
        self.finished = is_final;
        Ok(count)
    }
    pub fn finish(self) -> Result<PreparedMedia<B>> {
        if !self.finished || self.failed || self.chunks.is_empty() {
            return Err(Error::Audio("audio stream was not completed".into()));
        }
        let embeddings = Tensor::cat(self.chunks, 1);
        let [_, tokens, width] = embeddings.dims();
        Ok(PreparedMedia::Audio(embeddings.reshape([tokens, width])))
    }

    pub(crate) fn samples(&self) -> usize {
        self.samples
    }

    /// Return completed audio embeddings without flushing or advancing the stream.
    pub fn snapshot(&self) -> Result<Option<PreparedMedia<B>>> {
        if self.failed {
            return Err(Error::Finished);
        }
        if self.chunks.is_empty() {
            return Ok(None);
        }
        let embeddings = Tensor::cat(self.chunks.clone(), 1);
        let [_, tokens, width] = embeddings.dims();
        Ok(Some(PreparedMedia::Audio(
            embeddings.reshape([tokens, width]),
        )))
    }
}
