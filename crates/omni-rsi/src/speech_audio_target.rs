//! Prepare codec targets from an explicitly supplied local answer WAV.
//!
//! The caller supplies authorization and verifies the answer separately. This
//! module does not infer a transcript, meaning, or training eligibility.

use crate::fingerprint::{
    CodecSnapshot, codec_fingerprint_for_encoding, read_regular_bounded,
    target_codec_hash_for_encoding,
};
use crate::speech_target_format::{MAX_TARGET_CODEBOOKS, TARGET_SAMPLE_RATE, TargetCodeFormat};
use burn::tensor::backend::Backend;
use omni_lfm3::{
    Precision,
    codec::{CodecConfig, CodecEncoder},
};
use omni_r2t2::weights::Weights;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    io::Cursor,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

const SAMPLE_RATE: u32 = TARGET_SAMPLE_RATE;
const MAX_WAV_BYTES: u64 = 32 * 1024 * 1024;
const MAX_SAMPLES: usize = SAMPLE_RATE as usize * 20;
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_TARGET_CODES: usize = 200 * MAX_TARGET_CODEBOOKS;

/// JSON fields may be copied into a rule-data plan record. The Rust fields are
/// read-only after construction; serialized fields are editable declarations,
/// not authenticated evidence of execution or of a WAV-to-frames relation.
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

fn bounded_read(path: &Path, limit: u64, stop: &AtomicBool) -> Result<Vec<u8>, String> {
    let bytes = read_regular_bounded(path, limit, stop)?;
    if bytes.is_empty() {
        return Err(format!("invalid or oversized file: {}", path.display()));
    }
    Ok(bytes)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn preparation_sources() -> Vec<(&'static str, &'static [u8])> {
    macro_rules! source {
        ($name:literal, $path:literal) => {
            ($name, include_bytes!($path).as_slice())
        };
    }
    vec![
        source!("speech_audio_target.rs", "speech_audio_target.rs"),
        source!("speech_target_format.rs", "speech_target_format.rs"),
        source!("fingerprint.rs", "fingerprint.rs"),
        source!("speech_pcm.rs", "speech_pcm.rs"),
        source!("lfm3/codec.rs", "../../lfm3-omni/src/codec.rs"),
        source!("lfm3/config.rs", "../../lfm3-omni/src/config.rs"),
        source!("lfm3/text.rs", "../../lfm3-omni/src/text.rs"),
        source!("lfm3/lib.rs", "../../lfm3-omni/src/lib.rs"),
        source!("src/lib.rs", "../../../src/lib.rs"),
        source!("src/config.rs", "../../../src/config.rs"),
        source!("src/weights.rs", "../../../src/weights.rs"),
        source!("src/weights/layout.rs", "../../../src/weights/layout.rs"),
        source!("src/tensor/mod.rs", "../../../src/tensor/mod.rs"),
        source!("src/tensor/norm.rs", "../../../src/tensor/norm.rs"),
        source!("src/tensor/cache.rs", "../../../src/tensor/cache.rs"),
        source!(
            "src/tensor/embedding.rs",
            "../../../src/tensor/embedding.rs"
        ),
        source!("vendor/burn/lib.rs", "../../../vendor/burn/src/lib.rs"),
        source!(
            "vendor/burn/backend.rs",
            "../../../vendor/burn/src/backend.rs"
        ),
        source!("vendor/burn/Cargo.toml", "../../../vendor/burn/Cargo.toml"),
        source!("Cargo.toml", "../Cargo.toml"),
        source!("lfm3/Cargo.toml", "../../lfm3-omni/Cargo.toml"),
        source!("root/Cargo.toml", "../../../Cargo.toml"),
        source!("Cargo.lock", "../../../Cargo.lock"),
    ]
}

fn hash_preparation_sources(sources: &[(&str, &[u8])]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"omni-rsi-target-preparation-v2\0");
    for (name, bytes) in sources {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    format!("{:x}", hash.finalize())
}

fn preparation_fingerprint() -> String {
    hash_preparation_sources(&preparation_sources())
}

pub(crate) fn target_pcm(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let samples = crate::speech_pcm::decode_mono_pcm(Cursor::new(bytes), SAMPLE_RATE)?;
    if samples.is_empty()
        || samples.len() > MAX_SAMPLES
        || samples.iter().any(|sample| sample.abs() > 1.0)
    {
        return Err("invalid target WAV samples".into());
    }
    Ok(samples)
}

/// Pipeline seam for an injected encoder. Its initial output is marked as
/// injected and must not be treated as evidence that the local codec ran.
pub fn encode_target_with(
    model: &Path,
    wav: &Path,
    stop: &AtomicBool,
    mut encode: impl FnMut(&Path, &[f32]) -> Result<Vec<Vec<u32>>, String>,
) -> Result<EncodedTarget, String> {
    encode_target_impl(model, wav, stop, |codec, samples, _, _, _| {
        encode(codec, samples)
    })
}

fn encode_target_impl(
    model: &Path,
    wav: &Path,
    stop: &AtomicBool,
    mut encode: impl FnMut(&Path, &[f32], &str, &str, &[u8]) -> Result<Vec<Vec<u32>>, String>,
) -> Result<EncodedTarget, String> {
    interrupted(stop)?;
    let model_config = model.join("config.json");
    let codec = model.join("audio_tokenizer");
    let wav_bytes = bounded_read(wav, MAX_WAV_BYTES, stop)?;
    let wav_before = sha256(&wav_bytes);
    if sha256(&bounded_read(wav, MAX_WAV_BYTES, stop)?) != wav_before {
        return Err("target WAV changed while reading".into());
    }
    let samples = target_pcm(&wav_bytes)?;
    let model_bytes = bounded_read(&model_config, MAX_CONFIG_BYTES, stop)?;
    let model_hash = sha256(&model_bytes);
    let codec_bytes = bounded_read(&codec.join("config.json"), MAX_CONFIG_BYTES, stop)?;
    let format = TargetCodeFormat::from_bytes(&model_bytes, &codec_bytes)?;
    let codec_hash = codec_fingerprint_for_encoding(&codec, stop)?;
    let target_codec_sha256 = target_codec_hash_for_encoding(&codec, stop)?;
    interrupted(stop)?;
    let mut frames = encode(
        &codec,
        &samples,
        &codec_hash,
        &target_codec_sha256,
        &codec_bytes,
    )?;
    interrupted(stop)?;
    if !(4..200).contains(&frames.len())
        || frames
            .iter()
            .try_fold(format.num_codebooks, |total, frame| {
                total.checked_add(frame.len())
            })
            .is_none_or(|total| total > MAX_TARGET_CODES)
        || frames.iter().any(|frame| {
            frame.len() != format.num_codebooks
                || frame.iter().any(|code| *code >= format.code_count)
        })
    {
        return Err("invalid encoded target frames".into());
    }
    if encode(
        &codec,
        &samples,
        &codec_hash,
        &target_codec_sha256,
        &codec_bytes,
    )? != frames
    {
        return Err("unstable codec output for the same target WAV".into());
    }
    interrupted(stop)?;
    frames.push(vec![format.eos_code; format.num_codebooks]);
    if sha256(&bounded_read(wav, MAX_WAV_BYTES, stop)?) != wav_before
        || sha256(&bounded_read(&model_config, MAX_CONFIG_BYTES, stop)?) != model_hash
        || codec_fingerprint_for_encoding(&codec, stop)? != codec_hash
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
    let mut result = encode_target_impl(
        model,
        wav,
        stop,
        |codec, samples, fingerprint, target_hash, config_bytes| {
            if encoder.is_none() {
                let snapshot = CodecSnapshot::capture(codec, stop)?;
                if snapshot.fingerprint() != fingerprint
                    || snapshot.target_codec_hash() != target_hash
                {
                    return Err("codec changed while capturing encoder".into());
                }
                let (captured_config, shards) = snapshot.into_parts();
                if captured_config != config_bytes {
                    return Err("codec configuration changed while capturing encoder".into());
                }
                let config =
                    CodecConfig::from_bytes(&captured_config).map_err(|error| error.to_string())?;
                let weights = Weights::from_bytes(shards).map_err(|error| error.to_string())?;
                encoder = Some(
                    CodecEncoder::<B>::from_weights(&weights, config, Precision::F32, device)
                        .map_err(|error| error.to_string())?,
                );
            }
            let encoder = encoder.as_ref().ok_or("missing codec encoder")?;
            if encoder.sample_rate() != SAMPLE_RATE {
                return Err("codec sample rate does not match target WAV".into());
            }
            encoder.encode(samples).map_err(|error| error.to_string())
        },
    )?;
    result.encoding = "local_liquidcodec_f32";
    result.runtime_backend = B::name(device);
    Ok(result)
}

#[cfg(test)]
mod preparation_tests {
    use super::{hash_preparation_sources, preparation_fingerprint, preparation_sources};

    #[test]
    fn preparation_identity_changes_with_shared_numerical_and_pcm_sources() {
        let original = preparation_fingerprint();
        for name in [
            "src/tensor/norm.rs",
            "src/tensor/mod.rs",
            "src/tensor/cache.rs",
            "speech_pcm.rs",
        ] {
            let mut sources = preparation_sources();
            let entry = sources
                .iter_mut()
                .find(|(path, _)| *path == name)
                .expect("preparation source must be bound");
            entry.1 = b"independently changed source";
            assert_ne!(original, hash_preparation_sources(&sources), "{name}");
        }
    }
}
