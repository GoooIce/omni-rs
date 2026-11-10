//! Prepare codec targets from an explicitly supplied local answer WAV.
//!
//! The caller supplies authorization and verifies the answer separately. This
//! module does not infer a transcript, meaning, or training eligibility.

use crate::{fingerprint::codec_fingerprint, speech_data::file_hash};
use burn::tensor::backend::Backend;
use omni_lfm3::{Precision, codec::CodecEncoder};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Cursor, Read},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

const SAMPLE_RATE: u32 = 24_000;
const MAX_WAV_BYTES: u64 = 32 * 1024 * 1024;
const MAX_SAMPLES: usize = SAMPLE_RATE as usize * 20;
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;

#[derive(Deserialize)]
struct ModelCodeFormat {
    num_codebooks: usize,
    codebook_size: usize,
    audio_eos_code: u32,
}

#[derive(Deserialize)]
struct CodecCodeFormat {
    n_q: usize,
    fsq_levels: Vec<u32>,
    sampling_rate: u32,
}

/// JSON fields may be copied into a rule-data plan record. Provenance and
/// frames are read-only so an injected result cannot be relabelled as native.
///
/// ```compile_fail
/// fn relabel(target: &mut omni_rsi::speech_audio_target::EncodedTarget) {
///     target.encoding = "local_liquidcodec_f32";
/// }
/// ```
#[derive(Debug, Serialize)]
pub struct EncodedTarget {
    frames: Vec<Vec<u32>>,
    target_codec_sha256: String,
    target_audio_sha256: String,
    codec_fingerprint_sha256: String,
    model_config_sha256: String,
    sample_rate: u32,
    /// Identifies test injection separately from the local CodecEncoder path.
    encoding: &'static str,
    /// Codec execution does not verify the answer, speaker, consent or quality.
    quality_evidence: &'static str,
    runtime_backend: String,
    preparation_sha256: String,
}

impl EncodedTarget {
    pub fn frames(&self) -> &[Vec<u32>] {
        &self.frames
    }
    pub fn target_codec_sha256(&self) -> &str {
        &self.target_codec_sha256
    }
    pub fn target_audio_sha256(&self) -> &str {
        &self.target_audio_sha256
    }
    pub fn codec_fingerprint_sha256(&self) -> &str {
        &self.codec_fingerprint_sha256
    }
    pub fn model_config_sha256(&self) -> &str {
        &self.model_config_sha256
    }
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    pub fn encoding(&self) -> &str {
        self.encoding
    }
    pub fn quality_evidence(&self) -> &str {
        self.quality_evidence
    }
    pub fn runtime_backend(&self) -> &str {
        &self.runtime_backend
    }
    pub fn preparation_sha256(&self) -> &str {
        &self.preparation_sha256
    }
}

fn interrupted(stop: &AtomicBool) -> Result<(), String> {
    if stop.load(Ordering::Acquire) {
        Err("interrupted".into())
    } else {
        Ok(())
    }
}

fn bounded_read(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let file = fs::File::open(path).map_err(|error| error.to_string())?;
    let metadata = file.metadata().map_err(|error| error.to_string())?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > limit {
        return Err(format!("invalid or oversized file: {}", path.display()));
    }
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() as u64 != metadata.len() || bytes.len() as u64 > limit {
        return Err(format!("file changed while reading: {}", path.display()));
    }
    Ok(bytes)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn preparation_fingerprint() -> String {
    let mut hash = Sha256::new();
    hash.update(b"omni-rsi-target-preparation-v1\0");
    for bytes in [
        include_bytes!("speech_audio_target.rs").as_slice(),
        include_bytes!("fingerprint.rs").as_slice(),
        include_bytes!("../../lfm3-omni/src/codec.rs").as_slice(),
        include_bytes!("../../lfm3-omni/src/config.rs").as_slice(),
        include_bytes!("../../lfm3-omni/src/text.rs").as_slice(),
        include_bytes!("../../../src/weights.rs").as_slice(),
        include_bytes!("../Cargo.lock").as_slice(),
    ] {
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    format!("{:x}", hash.finalize())
}

fn target_pcm(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let mut wav = hound::WavReader::new(Cursor::new(bytes)).map_err(|error| error.to_string())?;
    let spec = wav.spec();
    if spec.channels != 1
        || spec.sample_rate != SAMPLE_RATE
        || wav.duration() as usize > MAX_SAMPLES
    {
        return Err("target WAV must be bounded mono 24 kHz PCM".into());
    }
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => wav
            .samples::<i16>()
            .map(|sample| {
                sample
                    .map(|value| f32::from(value) / 32768.0)
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Float, 32) => wav
            .samples::<f32>()
            .map(|sample| sample.map_err(|error| error.to_string()))
            .collect::<Result<_, _>>()?,
        _ => return Err("target WAV must be PCM16 or float32".into()),
    };
    if samples.is_empty()
        || samples.len() > MAX_SAMPLES
        || samples
            .iter()
            .any(|sample| !sample.is_finite() || sample.abs() > 1.0)
    {
        return Err("invalid target WAV samples".into());
    }
    Ok(samples)
}

/// Pipeline seam for an injected encoder. Its output is explicitly marked as
/// injected and must not be treated as evidence that the local codec ran.
pub fn encode_target_with(
    model: &Path,
    wav: &Path,
    stop: &AtomicBool,
    mut encode: impl FnMut(&Path, &[f32]) -> Result<Vec<Vec<u32>>, String>,
) -> Result<EncodedTarget, String> {
    interrupted(stop)?;
    let model_config = model.join("config.json");
    let codec = model.join("audio_tokenizer");
    let wav_bytes = bounded_read(wav, MAX_WAV_BYTES)?;
    let wav_before = sha256(&wav_bytes);
    if sha256(&bounded_read(wav, MAX_WAV_BYTES)?) != wav_before {
        return Err("target WAV changed while reading".into());
    }
    let samples = target_pcm(&wav_bytes)?;
    let model_bytes = bounded_read(&model_config, MAX_CONFIG_BYTES)?;
    let model_hash = sha256(&model_bytes);
    let model_format: ModelCodeFormat =
        serde_json::from_slice(&model_bytes).map_err(|error| error.to_string())?;
    let codec_hash = codec_fingerprint(&codec, stop)?;
    let codec_bytes = bounded_read(&codec.join("config.json"), MAX_CONFIG_BYTES)?;
    let codec_format: CodecCodeFormat =
        serde_json::from_slice(&codec_bytes).map_err(|error| error.to_string())?;
    let code_count = codec_format
        .fsq_levels
        .iter()
        .try_fold(1_u32, |product, level| {
            (*level >= 2).then(|| product.checked_mul(*level)).flatten()
        })
        .ok_or("invalid codec levels")?;
    if codec_format.fsq_levels.is_empty()
        || codec_format.sampling_rate != SAMPLE_RATE
        || codec_format.n_q == 0
        || codec_format.n_q != model_format.num_codebooks
        || model_format.audio_eos_code < code_count
        || model_format.audio_eos_code as usize >= model_format.codebook_size
    {
        return Err("model and codec codebook formats do not match".into());
    }
    let target_codec_sha256 = if codec.join("model.safetensors.index.json").is_file() {
        codec_hash.clone()
    } else {
        file_hash(&codec.join("model.safetensors"))?
    };
    interrupted(stop)?;
    let mut frames = encode(&codec, &samples)?;
    interrupted(stop)?;
    if !(4..200).contains(&frames.len())
        || frames.iter().any(|frame| {
            frame.len() != model_format.num_codebooks
                || frame.iter().any(|code| *code >= code_count)
        })
    {
        return Err("invalid encoded target frames".into());
    }
    if encode(&codec, &samples)? != frames {
        return Err("unstable codec output for the same target WAV".into());
    }
    interrupted(stop)?;
    frames.push(vec![
        model_format.audio_eos_code;
        model_format.num_codebooks
    ]);
    if sha256(&bounded_read(wav, MAX_WAV_BYTES)?) != wav_before
        || sha256(&bounded_read(&model_config, MAX_CONFIG_BYTES)?) != model_hash
        || codec_fingerprint(&codec, stop)? != codec_hash
    {
        return Err("target WAV or codec changed during encoding".into());
    }
    interrupted(stop)?;
    Ok(EncodedTarget {
        frames,
        target_codec_sha256,
        target_audio_sha256: wav_before,
        codec_fingerprint_sha256: codec_hash,
        model_config_sha256: model_hash,
        sample_rate: SAMPLE_RATE,
        encoding: "caller_supplied_unverified",
        quality_evidence: "not_assessed",
        runtime_backend: "caller_supplied_unverified".into(),
        preparation_sha256: preparation_fingerprint(),
    })
}

/// Load the exact local LiquidCodec encoder and emit validated target frames.
pub fn encode_target<B: Backend>(
    model: &Path,
    wav: &Path,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<EncodedTarget, String> {
    let mut encoder: Option<CodecEncoder<B>> = None;
    let mut result = encode_target_with(model, wav, stop, |codec, samples| {
        if encoder.is_none() {
            encoder = Some(
                CodecEncoder::<B>::load(codec, Precision::F32, device)
                    .map_err(|error| error.to_string())?,
            );
        }
        let encoder = encoder.as_ref().ok_or("missing codec encoder")?;
        if encoder.sample_rate() != SAMPLE_RATE {
            return Err("codec sample rate does not match target WAV".into());
        }
        encoder.encode(samples).map_err(|error| error.to_string())
    })?;
    result.encoding = "local_liquidcodec_f32";
    result.runtime_backend = B::name(device);
    Ok(result)
}
