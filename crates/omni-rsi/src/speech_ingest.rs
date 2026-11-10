//! Native, recoverable admission of explicitly submitted speech.

use crate::{
    speech_asr::{self, NATIVE_VERIFIER, VerifierFingerprints},
    speech_data::{SpeechDataset, file_hash},
    speech_evaluation::SpeechEvaluation,
};
use burn::tensor::backend::Backend;
use omni_lfm3::speech_inbox::{PendingSpeech, SpeechClaim, SpeechOutcome, SpeechWorker};
use omni_lfm3::{
    Precision,
    codec::{Codec, CodecEncoder},
};
use omni_r2t2::audio::Resampler;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

pub enum VerificationError {
    Rejected(String),
    Retry(String),
}

/// Separates native model work from queue ownership and atomic publication.
pub trait SpeechVerifier {
    fn fingerprints(&self) -> &VerifierFingerprints;
    fn model_hash(&self) -> &str;
    fn codec_hash(&self) -> &str;
    fn prepare(
        &self,
        entry: &PendingSpeech,
        destination: &Path,
        stop: &AtomicBool,
    ) -> Result<Value, VerificationError>;
    fn recheck(&self, stop: &AtomicBool) -> Result<(), String>;
}

#[derive(Debug, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum IngestResult {
    Empty,
    Rejected { id: String, reason: String },
    Accepted { id: String, recovered: bool },
}

fn check_models(data: &SpeechDataset, verifier: &impl SpeechVerifier) -> Result<(), String> {
    if data
        .train
        .iter()
        .chain(&data.dev)
        .chain(&data.test)
        .any(|item| {
            item.record.teacher_model_sha256 != verifier.model_hash()
                || item.record.codec_hash() != verifier.codec_hash()
        })
    {
        return Err("speech dataset does not match the current model and codec".into());
    }
    Ok(())
}

fn reject(
    worker: &SpeechWorker,
    entry: &PendingSpeech,
    reason: String,
    category: &str,
) -> Result<IngestResult, String> {
    worker
        .finish(entry, SpeechOutcome::Rejected, category)
        .map_err(|error| error.to_string())?;
    Ok(IngestResult::Rejected {
        id: entry.id.clone(),
        reason,
    })
}

fn private_dir(path: &Path) -> Result<(), String> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path).map_err(|error| error.to_string())
}

/// Claim one recording, verify it, and publish only after all data gates pass.
pub fn ingest_one<V: SpeechVerifier>(
    root: &Path,
    data: &Path,
    make_verifier: impl FnOnce() -> Result<V, String>,
    stop: &AtomicBool,
) -> Result<IngestResult, String> {
    let worker = SpeechWorker::open(root).map_err(|error| error.to_string())?;
    let entry = match worker.claim().map_err(|error| error.to_string())? {
        None => return Ok(IngestResult::Empty),
        Some(SpeechClaim::Rejected { id, reason }) => {
            return Ok(IngestResult::Rejected { id, reason });
        }
        Some(SpeechClaim::Valid(entry)) => entry,
    };
    let evaluation = SpeechEvaluation::fixed()?;
    if evaluation.is_reserved_pcm(&entry.audio)? {
        return reject(
            &worker,
            &entry,
            "reserved evaluation audio".into(),
            "reserved evaluation audio",
        );
    }
    let online = root.join("online-data");
    private_dir(&online)?;
    let verifier = make_verifier()?;
    let combined =
        SpeechDataset::load_for_native(data, Some(&online), None, verifier.fingerprints())?;
    check_models(&combined, &verifier)?;
    evaluation.check_data(&combined)?;
    evaluation.check_online_originals(&online)?;
    let published = online.join(&entry.id);
    if published.exists() {
        worker
            .finish(&entry, SpeechOutcome::Accepted, "")
            .map_err(|error| error.to_string())?;
        return Ok(IngestResult::Accepted {
            id: entry.id,
            recovered: true,
        });
    }
    let temporary = tempfile::Builder::new()
        .prefix(".speech-stage-")
        .tempdir_in(root)
        .map_err(|error| error.to_string())?;
    let staging = temporary.path().join(&entry.id);
    fs::create_dir(&staging).map_err(|error| error.to_string())?;
    let record = match verifier.prepare(&entry, &staging, stop) {
        Ok(record) => record,
        Err(VerificationError::Rejected(reason)) => {
            return reject(&worker, &entry, reason, "verification");
        }
        Err(VerificationError::Retry(reason)) => return Err(reason),
    };
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(staging.join("record.json"))
        .map_err(|error| error.to_string())?;
    serde_json::to_writer(&mut output, &record).map_err(|error| error.to_string())?;
    output.sync_all().map_err(|error| error.to_string())?;
    for name in ["source.wav", "audio.wav"] {
        File::open(staging.join(name))
            .and_then(|file| file.sync_all())
            .map_err(|error| error.to_string())?;
    }
    let checked = match SpeechDataset::load_for_native(
        data,
        Some(&online),
        Some(&staging),
        verifier.fingerprints(),
    ) {
        Ok(data) => data,
        Err(reason) => return reject(&worker, &entry, reason, "verification"),
    };
    if let Err(reason) =
        check_models(&checked, &verifier).and_then(|_| evaluation.check_data(&checked))
    {
        return reject(&worker, &entry, reason, "verification");
    }
    verifier.recheck(stop)?;
    File::open(&staging)
        .and_then(|file| file.sync_all())
        .map_err(|error| error.to_string())?;
    fs::rename(&staging, &published).map_err(|error| error.to_string())?;
    File::open(&online)
        .and_then(|file| file.sync_all())
        .map_err(|error| error.to_string())?;
    worker
        .finish(&entry, SpeechOutcome::Accepted, "")
        .map_err(|error| error.to_string())?;
    Ok(IngestResult::Accepted {
        id: entry.id,
        recovered: false,
    })
}

/// Local R2T2 plus the project's Burn LiquidCodec implementation.
pub struct LocalVerifier<B: Backend> {
    model: PathBuf,
    asr: PathBuf,
    binary: PathBuf,
    device: B::Device,
    precision: Precision,
    ids: VerifierFingerprints,
    model_hash: String,
    codec_hash: String,
}

impl<B: Backend> LocalVerifier<B> {
    pub fn new(
        model: &Path,
        asr: &Path,
        binary: &Path,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Self, String> {
        let model = model.canonicalize().map_err(|error| error.to_string())?;
        let asr = asr.canonicalize().map_err(|error| error.to_string())?;
        let binary = binary.canonicalize().map_err(|error| error.to_string())?;
        let ids = speech_asr::fingerprints(&asr, &binary)?;
        let model_hash = file_hash(&model.join("model.safetensors"))?;
        let codec_hash = file_hash(&model.join("audio_tokenizer/model.safetensors"))?;
        Ok(Self {
            model,
            asr,
            binary,
            device: device.clone(),
            precision,
            ids,
            model_hash,
            codec_hash,
        })
    }
}

pub(crate) fn write_pcm(path: &Path, samples: &[f32], rate: u32) -> Result<(), String> {
    if samples.is_empty() || samples.iter().any(|value| !value.is_finite()) {
        return Err("invalid audio for PCM output".into());
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut output = hound::WavWriter::create(path, spec).map_err(|error| error.to_string())?;
    for sample in samples {
        let value = (sample * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
        output
            .write_sample(value)
            .map_err(|error| error.to_string())?;
    }
    output.finalize().map_err(|error| error.to_string())
}

pub(crate) fn read_pcm(path: &Path, expected_rate: u32) -> Result<Vec<f32>, String> {
    let file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    crate::speech_pcm::decode_mono_pcm(std::io::BufReader::new(file), expected_rate)
}

impl<B: Backend> SpeechVerifier for LocalVerifier<B> {
    fn fingerprints(&self) -> &VerifierFingerprints {
        &self.ids
    }
    fn model_hash(&self) -> &str {
        &self.model_hash
    }
    fn codec_hash(&self) -> &str {
        &self.codec_hash
    }

    fn prepare(
        &self,
        entry: &PendingSpeech,
        destination: &Path,
        stop: &AtomicBool,
    ) -> Result<Value, VerificationError> {
        let reject = |message: String| VerificationError::Rejected(message);
        let retry = |message: String| VerificationError::Retry(message);
        let check_stop = || {
            if stop.load(Ordering::Acquire) {
                Err(retry("interrupted".into()))
            } else {
                Ok(())
            }
        };
        check_stop()?;
        let original = read_pcm(&entry.audio, 16_000).map_err(reject)?;
        if original.is_empty()
            || (original
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>()
                / original.len() as f64)
                .sqrt()
                < 0.001
        {
            return Err(reject("silent or invalid audio".into()));
        }
        let initial =
            speech_asr::transcribe(&entry.audio, &self.asr, &self.binary, stop).map_err(retry)?;
        let transcript = speech_asr::simplified(&initial.text).trim().to_owned();
        if !(2..=200).contains(&transcript.chars().count())
            || !speech_asr::assess(&transcript, &initial).passed
        {
            return Err(reject(
                "source ASR did not produce enough Chinese speech".into(),
            ));
        }
        let mut resampler = Resampler::new(16_000, 24_000).map_err(|error| retry(error.into()))?;
        let source = resampler
            .push(&original, true)
            .map_err(|error| retry(error.into()))?;
        let source_path = destination.join("source.wav");
        write_pcm(&source_path, &source, 24_000).map_err(retry)?;
        let source = read_pcm(&source_path, 24_000).map_err(reject)?;
        check_stop()?;
        let directory = self.model.join("audio_tokenizer");
        let encoder = CodecEncoder::<B>::load(&directory, self.precision, &self.device)
            .map_err(|error| retry(error.to_string()))?;
        let codes = encoder
            .encode(&source)
            .map_err(|error| reject(error.to_string()))?;
        if !(4..200).contains(&codes.len())
            || codes
                .iter()
                .any(|frame| frame.len() != 12 || frame.iter().any(|code| *code >= 2016))
        {
            return Err(reject("invalid LiquidCodec frames".into()));
        }
        if encoder
            .encode(&source)
            .map_err(|error| retry(error.to_string()))?
            != codes
        {
            return Err(retry("LiquidCodec encoding is not repeatable".into()));
        }
        check_stop()?;
        let decoder = Codec::<B>::load(&directory, self.precision, &self.device)
            .map_err(|error| retry(error.to_string()))?;
        let decoded = decoder
            .decode(&codes)
            .map_err(|error| reject(error.to_string()))?;
        if decoded.is_empty()
            || decoded
                .iter()
                .any(|value| !value.is_finite() || value.abs() > 1.0)
        {
            return Err(reject("invalid decoded speech".into()));
        }
        let audio = destination.join("audio.wav");
        write_pcm(&audio, &decoded, 24_000).map_err(retry)?;
        let restored = read_pcm(&audio, 24_000).map_err(reject)?;
        if restored.len() != decoded.len() {
            return Err(reject("decoded speech length is invalid".into()));
        }
        let codec_error = restored
            .iter()
            .zip(&decoded)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        if codec_error > 0.0001 {
            return Err(reject("decoded speech changed while saving".into()));
        }
        let event = speech_asr::transcribe(&audio, &self.asr, &self.binary, stop).map_err(retry)?;
        let comparison = speech_asr::assess(&transcript, &event);
        if !comparison.passed {
            return Err(reject("source and round-trip ASR disagree".into()));
        }
        let mut frames = codes;
        frames.push(vec![2016; 12]);
        let frame_bytes = serde_json::to_vec(&frames).map_err(|error| retry(error.to_string()))?;
        let frame_hash = format!("{:x}", Sha256::digest(frame_bytes));
        check_stop()?;
        Ok(serde_json::json!({
            "id":entry.id,"split":"train","task":"tts","question":transcript,"answer":transcript,
            "label_source":"r2t2-pseudo-label","frames":frames,"audio":"audio.wav",
            "audio_sha256":file_hash(&audio).map_err(retry)?,"teacher_model_sha256":self.model_hash,"verified":true,
            "speech_teacher":{"method":"user-speech-v1","fingerprint":entry.id,"audio":"source.wav",
                              "audio_sha256":file_hash(&source_path).map_err(retry)?},
            "verification":{"method":NATIVE_VERIFIER,"kind":"codec-roundtrip-self-consistency",
                            "asr_sha256":self.ids.native,"transcript":comparison.transcript,
                            "reported_language":comparison.reported_language,"han_fraction":comparison.han_fraction,
                            "similarity":comparison.similarity,"codec_max_error":codec_error,"codec_sha256":self.codec_hash,
                            "frame_sha256":frame_hash,"source_frame_sha256":frame_hash}
        }))
    }

    fn recheck(&self, stop: &AtomicBool) -> Result<(), String> {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        let ids = speech_asr::fingerprints(&self.asr, &self.binary)?;
        if ids.legacy != self.ids.legacy
            || ids.native != self.ids.native
            || file_hash(&self.model.join("model.safetensors"))? != self.model_hash
            || file_hash(&self.model.join("audio_tokenizer/model.safetensors"))? != self.codec_hash
        {
            return Err("model or verifier changed while processing speech".into());
        }
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod pcm_tests {
    use super::read_pcm;

    #[test]
    fn speech_pcm_decoding_preserves_existing_format_and_value_rules() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("speech.wav");
        let mut writer = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 24_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for value in [0_i16, 8192, i16::MIN] {
            writer.write_sample(value).unwrap();
        }
        writer.finalize().unwrap();
        assert_eq!(read_pcm(&path, 24_000).unwrap(), [0.0, 0.25, -1.0]);
        assert!(read_pcm(&path, 16_000).is_err());

        let mut writer = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 24_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        writer.write_sample(1.5_f32).unwrap();
        writer.finalize().unwrap();
        assert_eq!(read_pcm(&path, 24_000).unwrap(), [1.5]);

        let mut writer = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 24_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        writer.write_sample(f32::NAN).unwrap();
        writer.finalize().unwrap();
        assert!(read_pcm(&path, 24_000).is_err());
    }
}
