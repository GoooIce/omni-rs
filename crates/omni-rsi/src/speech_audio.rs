//! Audio-conditioned answer training. Dataset admission and publication are separate.
use crate::speech_training::{
    SpeechOptimizer, speech_loss_with_media, text_loss_with_media, validate_speech_frames,
};
use burn::tensor::{Tensor, backend::AutodiffBackend};
use omni_lfm3::model::OmniModel;
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};

/// One already-labelled example. The answer is never inserted into the prompt.
pub struct AudioAnswer<'a> {
    samples: &'a [f32],
    instruction: &'a str,
    answer: &'a str,
    frames: Option<&'a [Vec<u32>]>,
}

impl<'a> AudioAnswer<'a> {
    pub fn new(
        samples: &'a [f32],
        sample_rate: u32,
        instruction: &'a str,
        answer: &'a str,
        frames: &'a [Vec<u32>],
    ) -> Result<Self, String> {
        Self::with_frames(samples, sample_rate, instruction, answer, Some(frames))
    }

    /// Train the answer text without requiring a synthesized speech target.
    pub fn new_text_only(
        samples: &'a [f32],
        sample_rate: u32,
        instruction: &'a str,
        answer: &'a str,
    ) -> Result<Self, String> {
        Self::with_frames(samples, sample_rate, instruction, answer, None)
    }

    fn with_frames(
        samples: &'a [f32],
        sample_rate: u32,
        instruction: &'a str,
        answer: &'a str,
        frames: Option<&'a [Vec<u32>]>,
    ) -> Result<Self, String> {
        if sample_rate != 16_000
            || samples.is_empty()
            || samples.len() > 20 * 16_000
            || samples
                .iter()
                .any(|sample| !sample.is_finite() || sample.abs() > 1.0)
            || instruction.trim().is_empty()
            || instruction.len() > 2048
            || instruction.contains("<|")
            || answer.trim().is_empty()
            || answer.len() > 2048
            || answer.contains("<|")
        {
            return Err("invalid audio answer example".into());
        }
        Ok(Self {
            samples,
            instruction,
            answer,
            frames,
        })
    }

    fn messages(&self) -> Vec<serde_json::Value> {
        vec![serde_json::json!({"role":"user", "content":[
            {"type":"text", "text":self.instruction}, {"type":"audio"}
        ]})]
    }
}

#[derive(Debug, Serialize)]
pub struct AudioTrainingStep {
    pub text_loss: f64,
    /// Zero when `speech_supervised` is false.
    pub speech_loss: f64,
    pub speech_end_loss: f64,
    pub speech_supervised: bool,
    pub gradient_norm: f64,
}

fn scalar<B: AutodiffBackend>(value: Tensor<B, 1>) -> Result<f64, String> {
    let values = value
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|error| error.to_string())?;
    let value = *values.first().ok_or("empty audio training loss")?;
    if !value.is_finite() {
        return Err("non-finite audio training loss".into());
    }
    Ok(f64::from(value))
}

/// Apply one bounded Burn update. No checkpoint or candidate is published here.
/// Callers must provide independently admitted data and persist protocol-bound
/// optimizer state before this primitive is used by the idle scheduler.
pub fn train_step<B: AutodiffBackend>(
    model: &mut OmniModel<B>,
    optimizer: &mut SpeechOptimizer<B>,
    example: &AudioAnswer<'_>,
    seed: u64,
    stop: &AtomicBool,
) -> Result<AudioTrainingStep, String> {
    if stop.load(Ordering::Acquire) {
        return Err("interrupted".into());
    }
    if let Some(frames) = example.frames {
        validate_speech_frames(model, frames)?;
    }
    let mut targets = model
        .tokenizer
        .encode(example.answer)
        .map_err(|error| error.to_string())?;
    let eos = model
        .tokenizer
        .encode("<|im_end|>")
        .map_err(|error| error.to_string())?;
    if targets.is_empty() || targets.len() > 256 || eos.len() != 1 || targets.contains(&eos[0]) {
        return Err("invalid audio answer tokens".into());
    }
    targets.push(eos[0]);
    let messages = example.messages();
    let media = [model
        .prepare_audio(example.samples, 16_000)
        .map_err(|error| error.to_string())?];
    optimizer.adapter().install(&mut model.text, 23)?;
    let text = text_loss_with_media(model, &messages, &media, &targets)?;
    let text_loss = scalar(text.clone())?;
    let (loss, speech_loss, speech_end_loss) = if let Some(frames) = example.frames {
        let speech = speech_loss_with_media(model, &messages, &media, frames, seed)?;
        let speech_loss = scalar(speech.content.clone())?;
        let speech_end_loss = scalar(speech.end.clone())?;
        (
            text + speech.content + speech.end * 0.01,
            speech_loss,
            speech_end_loss,
        )
    } else {
        (text, 0.0, 0.0)
    };
    if stop.load(Ordering::Acquire) {
        return Err("interrupted".into());
    }
    let gradient_norm = optimizer.update(loss, 0.0001)?;
    Ok(AudioTrainingStep {
        text_loss,
        speech_loss,
        speech_end_loss,
        speech_supervised: example.frames.is_some(),
        gradient_norm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_requires_bounded_finite_audio_and_no_control_tokens() {
        let samples = [0.1, -0.1];
        let frames = vec![vec![1; 12]; 5];
        assert!(AudioAnswer::new(&samples, 16_000, "加一回答", "十三", &frames).is_ok());
        assert!(AudioAnswer::new(&samples, 8_000, "加一回答", "十三", &frames).is_err());
        assert!(AudioAnswer::new(&[f32::NAN], 16_000, "加一回答", "十三", &frames).is_err());
        assert!(AudioAnswer::new(&[1.1], 16_000, "加一回答", "十三", &frames).is_err());
        assert!(AudioAnswer::new(&samples, 16_000, "<|audio|>", "十三", &frames).is_err());
        assert!(AudioAnswer::new(&samples, 16_000, "加一回答", "<|im_end|>", &frames).is_err());
        let sample = AudioAnswer::new(&samples, 16_000, "加一回答", "十三", &frames).unwrap();
        assert!(
            !serde_json::to_string(&sample.messages())
                .unwrap()
                .contains("十三")
        );
    }
}
