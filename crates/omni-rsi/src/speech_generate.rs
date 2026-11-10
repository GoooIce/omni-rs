//! Generate and verify fixed speech cases with native models.

use crate::{
    speech_asr,
    speech_evaluation::{SpeechCase, SpeechEvaluation},
    speech_ingest::write_pcm,
    speech_training::SpeechOptimizer,
};
use burn::tensor::backend::{AutodiffBackend, Backend};
use omni_lfm3::{
    generate::{OmniEvent, OmniOptions, OutputMode},
    model::OmniModel,
    speech_adapter::SpeechObservation,
};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

pub struct GenerationContext<'a> {
    pub asr: &'a Path,
    pub binary: &'a Path,
    pub state: &'a Path,
    pub stop: &'a AtomicBool,
}

pub fn development<B: AutodiffBackend>(
    model: &mut OmniModel<B>,
    optimizer: &SpeechOptimizer<B>,
    evaluation: &SpeechEvaluation,
    epoch: usize,
    context: &GenerationContext<'_>,
) -> Result<Vec<SpeechObservation>, String> {
    let (a, b) = optimizer.export()?;
    compare(
        model,
        (&a, &b),
        &evaluation.dev,
        &format!("epoch-{epoch}"),
        context,
    )
}

pub fn compare<B: Backend>(
    model: &mut OmniModel<B>,
    adapter: (&[f32], &[f32]),
    cases: &[SpeechCase],
    label: &str,
    context: &GenerationContext<'_>,
) -> Result<Vec<SpeechObservation>, String> {
    let safe_name = |value: &str| {
        !value.is_empty()
            && value.len() <= 64
            && value.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_'
            })
    };
    if !safe_name(label) || cases.iter().any(|case| !safe_name(&case.id)) {
        return Err("invalid speech evaluation output name".into());
    }
    let directory = context.state.join("evaluation").join(label);
    std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let (_, _, eos) = model.speech_code_format();
    let mut observations = Vec::with_capacity(cases.len() * 2);
    for case in cases {
        let messages = if case.category == "reading" {
            vec![
                serde_json::json!({"role":"system","content":"Perform TTS in Chinese."}),
                serde_json::json!({"role":"user","content":case.prompt}),
            ]
        } else {
            vec![serde_json::json!({"role":"user","content":case.prompt})]
        };
        for kind in ["baseline", "candidate"] {
            if context.stop.load(Ordering::Acquire) {
                return Err("interrupted".into());
            }
            if kind == "baseline" {
                model
                    .text
                    .clear_conv_lora(23)
                    .map_err(|error| error.to_string())?;
            } else {
                model
                    .text
                    .install_conv_lora(23, 4, adapter.0, adapter.1)
                    .map_err(|error| error.to_string())?;
            }
            let options = OmniOptions {
                mode: OutputMode::Audio,
                max_slots: 180,
                seed: 42,
                audio_temperature: Some(0.4),
                audio_repetition_penalty: 1.2,
                ..Default::default()
            };
            let mut eos_seen = false;
            let result = model
                .generate_omni(
                    &messages,
                    None,
                    &[],
                    &mut model.new_text_cache(),
                    &options,
                    context.stop,
                    |event| {
                        if context.stop.load(Ordering::Acquire) {
                            return false;
                        }
                        match event {
                            OmniEvent::SpeechCodes(codes)
                                if codes.iter().all(|code| *code == eos) =>
                            {
                                eos_seen = true;
                                true
                            }
                            OmniEvent::Pcm { .. } | OmniEvent::TextToken(_) if eos_seen => false,
                            _ => true,
                        }
                    },
                )
                .map_err(|error| error.to_string())?;
            if context.stop.load(Ordering::Acquire) {
                return Err("interrupted".into());
            }
            let stopped = result
                .audio_frames
                .last()
                .is_some_and(|frame| frame.iter().all(|code| *code == eos));
            let frames = result
                .audio_frames
                .len()
                .saturating_sub(usize::from(stopped));
            let mut transcript = String::new();
            let valid = frames >= 7
                && result
                    .audio_frames
                    .iter()
                    .take(frames)
                    .all(|frame| frame.iter().all(|code| *code < eos))
                && !result.samples.is_empty()
                && result.samples.iter().all(|sample| sample.is_finite())
                && result.samples.iter().any(|sample| sample.abs() >= 0.001);
            if valid {
                let path = directory.join(format!("{}-{kind}.wav", case.id));
                write_pcm(&path, &result.samples, 24_000)?;
                let event =
                    speech_asr::transcribe(&path, context.asr, context.binary, context.stop)?;
                transcript = speech_asr::simplified(&event.text);
            }
            observations.push(SpeechObservation {
                sample: case.id.clone(),
                category: case.category.clone(),
                kind: kind.into(),
                frames,
                stopped,
                transcript,
            });
        }
    }
    Ok(observations)
}
