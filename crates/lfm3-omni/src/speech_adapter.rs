//! Load a speech LoRA only after checking its model binding and development evidence.
use omni_r2t2::{Error, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
};
use unicode_normalization::UnicodeNormalization;

const EVALUATION: &[u8] = include_bytes!("../../../docs/eval/chinese-speech-generation-v1.json");
const TARGET_LAYER: usize = 23;
const RANK: usize = 4;

fn invalid(message: &str) -> Error {
    Error::Checkpoint(message.into())
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_bounded(path: &Path, max: u64) -> Result<Vec<u8>> {
    let info = fs::symlink_metadata(path)?;
    if !info.file_type().is_file() || info.len() > max {
        return Err(invalid("invalid adapter package file"));
    }
    Ok(fs::read(path)?)
}

#[derive(Deserialize)]
struct Case {
    id: String,
    category: String,
    reference: String,
}
#[derive(Deserialize)]
struct Evaluation {
    dev: Vec<Case>,
}
#[derive(Deserialize)]
struct Observation {
    sample: String,
    category: String,
    kind: String,
    frames: usize,
    stopped: bool,
    transcript: String,
}
#[derive(Deserialize)]
struct Manifest {
    version: u32,
    base_hash: String,
    layer: usize,
    rank: usize,
    width: usize,
    weights_sha256: String,
    evaluation_sha256: String,
    asr_sha256: String,
    data_sha256: String,
    results: Vec<Observation>,
}
#[derive(Deserialize)]
struct Weights {
    a: Vec<f32>,
    b: Vec<f32>,
}

/// A package that passed the fixed development gate for one exact frozen base.
pub struct VerifiedAdapter {
    pub layer: usize,
    pub rank: usize,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}
impl VerifiedAdapter {
    pub fn load(directory: &Path, base_hash: &str, width: usize) -> Result<Self> {
        let manifest: Manifest =
            serde_json::from_slice(&read_bounded(&directory.join("manifest.json"), 512 * 1024)?)?;
        if manifest.version != 1
            || manifest.base_hash != base_hash
            || manifest.layer != TARGET_LAYER
            || manifest.rank != RANK
            || manifest.width != width
            || manifest.evaluation_sha256 != hash(EVALUATION)
            || manifest.asr_sha256.len() != 64
            || !manifest
                .asr_sha256
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
            || manifest.data_sha256.len() != 64
            || !manifest
                .data_sha256
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err(invalid(
                "speech adapter model, target or evaluation mismatch",
            ));
        }
        let bytes = read_bounded(&directory.join("adapter.json"), 8 * 1024 * 1024)?;
        if hash(&bytes) != manifest.weights_sha256 {
            return Err(invalid("speech adapter weights hash mismatch"));
        }
        let weights: Weights = serde_json::from_slice(&bytes)?;
        let size = width
            .checked_mul(RANK)
            .ok_or_else(|| invalid("speech adapter size overflow"))?;
        if weights.a.len() != size
            || weights.b.len() != size
            || weights
                .a
                .iter()
                .chain(&weights.b)
                .any(|value| !value.is_finite())
        {
            return Err(invalid("invalid speech adapter tensors"));
        }
        let evaluation: Evaluation = serde_json::from_slice(EVALUATION)?;
        if !passes(&evaluation.dev, &manifest.results) {
            return Err(invalid("speech adapter development gate failed"));
        }
        Ok(Self {
            layer: TARGET_LAYER,
            rank: RANK,
            a: weights.a,
            b: weights.b,
        })
    }
}

fn normalized(text: &str) -> Vec<char> {
    text.nfkc()
        .flat_map(char::to_lowercase)
        .filter(|c| c.is_alphanumeric())
        .collect()
}
fn errors(reference: &[char], observed: &[char]) -> usize {
    let mut previous: Vec<_> = (0..=observed.len()).collect();
    for (i, left) in reference.iter().enumerate() {
        let mut current = vec![i + 1];
        for (j, right) in observed.iter().enumerate() {
            current.push(
                (previous[j + 1] + 1)
                    .min(current[j] + 1)
                    .min(previous[j] + usize::from(left != right)),
            );
        }
        previous = current;
    }
    previous[observed.len()]
}
fn passes(cases: &[Case], results: &[Observation]) -> bool {
    if cases.len() < 10 || results.len() != 2 * cases.len() {
        return false;
    }
    let mut lookup = HashMap::new();
    for result in results {
        if result.transcript.len() > 4096
            || lookup
                .insert((result.sample.as_str(), result.kind.as_str()), result)
                .is_some()
        {
            return false;
        }
    }
    let mut counts = [0usize; 2];
    let mut totals = [[0usize; 2]; 2];
    for case in cases {
        let group = match case.category.as_str() {
            "reading" => 0,
            "dialogue" => 1,
            _ => return false,
        };
        counts[group] += 1;
        let Some(base) = lookup.get(&(case.id.as_str(), "baseline")) else {
            return false;
        };
        let Some(candidate) = lookup.get(&(case.id.as_str(), "candidate")) else {
            return false;
        };
        let reference = normalized(&case.reference);
        let candidate_text = normalized(&candidate.transcript);
        if base.category != case.category
            || candidate.category != case.category
            || reference.is_empty()
            || !candidate.stopped
            || candidate.frames == 0
            || candidate_text.is_empty()
        {
            return false;
        }
        totals[group][0] += errors(&reference, &normalized(&base.transcript));
        totals[group][1] += errors(&reference, &candidate_text);
    }
    let baseline = totals[0][0] + totals[1][0];
    let candidate = totals[0][1] + totals[1][1];
    counts.iter().all(|count| *count >= 5)
        && baseline > 0
        && candidate * 10 <= baseline * 9
        && totals.iter().all(|group| group[1] <= group[0])
}

fn codec_files(root: &Path, directory: &Path, names: &mut Vec<String>) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            codec_files(root, &path, names)?;
        } else if entry.file_type()?.is_file()
            && (path
                .extension()
                .is_some_and(|v| v == "py" || v == "safetensors")
                || path.file_name().is_some_and(|v| v == "config.json"))
        {
            names.push(
                path.strip_prefix(root)
                    .map_err(|_| invalid("invalid codec path"))?
                    .to_str()
                    .ok_or_else(|| invalid("non-UTF8 codec path"))?
                    .replace('\\', "/"),
            );
        }
    }
    Ok(())
}

/// Match the training protocol's SHA-256 over base weights, configuration and codec.
pub fn source_fingerprint(root: &Path) -> Result<String> {
    let mut names: Vec<String> = [
        "model.safetensors",
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_file() && entry.path().extension().is_some_and(|v| v == "py") {
            names.push(
                entry
                    .file_name()
                    .into_string()
                    .map_err(|_| invalid("non-UTF8 model path"))?,
            );
        }
    }
    codec_files(root, &root.join("audio_tokenizer"), &mut names)?;
    names.sort();
    let mut digest = Sha256::new();
    digest.update(b"omni-rsi-chinese-speech-v1\0");
    let mut buffer = vec![0u8; 1024 * 1024];
    for name in names {
        let path: PathBuf = root.join(&name);
        let mut file = fs::File::open(&path)?;
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update(file.metadata()?.len().to_le_bytes());
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}
