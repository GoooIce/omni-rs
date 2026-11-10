//! Native text normalization and evidence checks for the local R2T2 verifier.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{OnceLock, atomic::AtomicBool},
    time::Duration,
};

pub const LEGACY_VERIFIER: &str = "omni-r2t2-auto-metal-f16-exact-tuning-v2";
pub const NATIVE_VERIFIER: &str = "omni-r2t2-native-metal-f16-exact-tuning-v3";
const LEGACY_IMPL_SHA256: &str = "4e30c38952b0c217ac0249f80f5d134a75bac262bf25ba29997ad18bb65a74ac";
const REQUIRED_FILES: &[&str] = &[
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "preprocessor_config.json",
];
const OPTIONAL_FILES: &[&str] = &[
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "vocab.json",
    "merges.txt",
];

#[derive(Clone, Debug)]
pub struct VerifierFingerprints {
    pub legacy: String,
    pub native: String,
}

fn hash_file(path: &Path) -> Result<[u8; 32], String> {
    let mut file = std::fs::File::open(path).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let n = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(digest.finalize().into())
}

/// Fingerprint both historical evidence and the native verifier independently.
pub fn fingerprints(model: &Path, binary: &Path) -> Result<VerifierFingerprints, String> {
    let model = model.canonicalize().map_err(|error| error.to_string())?;
    let binary = binary.canonicalize().map_err(|error| error.to_string())?;
    if !model.is_dir() || !binary.is_file() {
        return Err("missing R2T2 model or executable".into());
    }
    let mut files: Vec<(&str, PathBuf)> = REQUIRED_FILES
        .iter()
        .map(|name| (*name, model.join(name)))
        .collect();
    files.extend(OPTIONAL_FILES.iter().filter_map(|name| {
        let path = model.join(name);
        path.is_file().then_some((*name, path))
    }));
    files.push(("r2t2-executable", binary));
    let mut legacy = Sha256::new();
    let mut native = Sha256::new();
    legacy.update(format!("{LEGACY_VERIFIER}\0").as_bytes());
    native.update(format!("{NATIVE_VERIFIER}\0").as_bytes());
    for (name, path) in files {
        let hash = hash_file(&path)?;
        for digest in [&mut legacy, &mut native] {
            digest.update(name.as_bytes());
            digest.update([0]);
            digest.update(hash);
        }
    }
    legacy.update(b"verifier-implementation\0");
    let old_impl = (0..32)
        .map(|i| {
            u8::from_str_radix(&LEGACY_IMPL_SHA256[i * 2..i * 2 + 2], 16)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    legacy.update(old_impl);
    for (name, contents) in [
        (
            "verifier-implementation",
            include_bytes!("speech_asr.rs").as_slice(),
        ),
        (
            "opencc-phrases",
            include_bytes!("../data/opencc/TSPhrases.txt").as_slice(),
        ),
        (
            "opencc-characters",
            include_bytes!("../data/opencc/TSCharacters.txt").as_slice(),
        ),
    ] {
        native.update(name.as_bytes());
        native.update([0]);
        native.update(Sha256::digest(contents));
    }
    Ok(VerifierFingerprints {
        legacy: format!("{:x}", legacy.finalize()),
        native: format!("{:x}", native.finalize()),
    })
}

/// Invoke the project's native R2T2 executable with bounded cancellation and output.
pub fn transcribe(
    audio: &Path,
    model: &Path,
    binary: &Path,
    stop: &AtomicBool,
) -> Result<AsrEvent, String> {
    let mut stdout = tempfile::tempfile().map_err(|error| error.to_string())?;
    let mut stderr = tempfile::tempfile().map_err(|error| error.to_string())?;
    let mut command = Command::new(binary.canonicalize().map_err(|error| error.to_string())?);
    command
        .arg("transcribe")
        .arg("--model")
        .arg(model.canonicalize().map_err(|error| error.to_string())?)
        .arg("--audio")
        .arg(audio.canonicalize().map_err(|error| error.to_string())?)
        .args([
            "--language",
            "auto",
            "--backend",
            "metal",
            "--precision",
            "f16",
            "--chunk-ms",
            "2000",
            "--json",
        ])
        .env("CUBECL_AUTOTUNE_LEVEL", "full")
        .stdin(Stdio::null())
        .stdout(Stdio::from(
            stdout.try_clone().map_err(|error| error.to_string())?,
        ))
        .stderr(Stdio::from(
            stderr.try_clone().map_err(|error| error.to_string())?,
        ));
    let status = if std::env::var_os("OMNI_RSI_SUPERVISED_WORKER").is_some() {
        crate::child::run_inherited_with_timeout(
            command,
            stop,
            Duration::from_secs(8),
            Duration::from_secs(300),
        )?
    } else {
        crate::child::run_with_timeout(
            command,
            stop,
            Duration::from_secs(10),
            Duration::from_secs(300),
        )?
    };
    if stdout.metadata().map_err(|error| error.to_string())?.len() > 4 * 1024 * 1024
        || stderr.metadata().map_err(|error| error.to_string())?.len() > 4 * 1024 * 1024
    {
        return Err("R2T2 output is oversized".into());
    }
    stdout
        .seek(SeekFrom::Start(0))
        .map_err(|error| error.to_string())?;
    stderr
        .seek(SeekFrom::Start(0))
        .map_err(|error| error.to_string())?;
    let mut output = String::new();
    let mut errors = String::new();
    stdout
        .read_to_string(&mut output)
        .map_err(|error| error.to_string())?;
    stderr
        .read_to_string(&mut errors)
        .map_err(|error| error.to_string())?;
    if !status.success() || errors.contains("panicked at") {
        let tail: String = errors
            .chars()
            .rev()
            .take(2500)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        return Err(format!("R2T2 failed ({status}): {tail}"));
    }
    final_event(&output)
}

struct Dictionary {
    phrases: HashMap<String, String>,
    characters: HashMap<char, String>,
    max_phrase: usize,
}

fn dictionary() -> &'static Dictionary {
    static DATA: OnceLock<Dictionary> = OnceLock::new();
    DATA.get_or_init(|| {
        let phrases: HashMap<String, String> = include_str!("../data/opencc/TSPhrases.txt")
            .lines()
            .filter_map(|line| line.split_once('\t'))
            .map(|(key, value)| {
                (
                    key.to_owned(),
                    value.split_whitespace().next().unwrap_or("").to_owned(),
                )
            })
            .collect();
        let characters = include_str!("../data/opencc/TSCharacters.txt")
            .lines()
            .filter_map(|line| line.split_once('\t'))
            .filter_map(|(key, value)| {
                Some((
                    key.chars().next()?,
                    value.split_whitespace().next()?.to_owned(),
                ))
            })
            .collect();
        let max_phrase = phrases
            .keys()
            .map(|key| key.chars().count())
            .max()
            .unwrap_or(0);
        Dictionary {
            phrases,
            characters,
            max_phrase,
        }
    })
}

/// Match the existing OpenCC t2s phrase-first, longest-match conversion.
pub fn simplified(input: &str) -> String {
    fn convert(chars: &[char], dictionary: &Dictionary, out: &mut String) {
        for length in (2..=dictionary.max_phrase.min(chars.len())).rev() {
            for start in 0..=chars.len() - length {
                let key: String = chars[start..start + length].iter().collect();
                if let Some(replacement) = dictionary.phrases.get(&key) {
                    convert(&chars[..start], dictionary, out);
                    out.push_str(replacement);
                    convert(&chars[start + length..], dictionary, out);
                    return;
                }
            }
        }
        for ch in chars {
            if let Some(replacement) = dictionary.characters.get(ch) {
                out.push_str(replacement);
            } else {
                out.push(*ch);
            }
        }
    }
    let mut out = String::with_capacity(input.len());
    convert(&input.chars().collect::<Vec<_>>(), dictionary(), &mut out);
    out
}

/// Python SequenceMatcher's matching-block ratio, including its autojunk rule.
pub fn matching_ratio(reference: &[char], observed: &[char]) -> f64 {
    if reference.is_empty() && observed.is_empty() {
        return 1.0;
    }
    let mut positions: HashMap<char, Vec<usize>> = HashMap::new();
    for (index, ch) in observed.iter().enumerate() {
        positions.entry(*ch).or_default().push(index);
    }
    if observed.len() >= 200 {
        let limit = observed.len() / 100 + 1;
        positions.retain(|_, indices| indices.len() <= limit);
    }
    let mut total = 0;
    let mut pending = vec![(0, reference.len(), 0, observed.len())];
    while let Some((alo, ahi, blo, bhi)) = pending.pop() {
        let (mut best_a, mut best_b, mut best_len) = (alo, blo, 0);
        let mut previous = vec![0_usize; observed.len() + 1];
        for (i, ch) in reference.iter().enumerate().take(ahi).skip(alo) {
            let mut current = vec![0_usize; observed.len() + 1];
            if let Some(indices) = positions.get(ch) {
                for &j in indices.iter().filter(|&&j| j >= blo && j < bhi) {
                    let length = previous[j] + 1;
                    current[j + 1] = length;
                    if length > best_len {
                        best_a = i + 1 - length;
                        best_b = j + 1 - length;
                        best_len = length;
                    }
                }
            }
            previous = current;
        }
        while best_a > alo && best_b > blo && reference[best_a - 1] == observed[best_b - 1] {
            best_a -= 1;
            best_b -= 1;
            best_len += 1;
        }
        while best_a + best_len < ahi
            && best_b + best_len < bhi
            && reference[best_a + best_len] == observed[best_b + best_len]
        {
            best_len += 1;
        }
        if best_len > 0 {
            total += best_len;
            if alo < best_a && blo < best_b {
                pending.push((alo, best_a, blo, best_b));
            }
            if best_a + best_len < ahi && best_b + best_len < bhi {
                pending.push((best_a + best_len, ahi, best_b + best_len, bhi));
            }
        }
    }
    2.0 * total as f64 / (reference.len() + observed.len()) as f64
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AsrEvent {
    pub text: String,
    #[serde(default)]
    pub language: String,
}

pub fn final_event(output: &str) -> Result<AsrEvent, String> {
    let events: Vec<serde_json::Value> = output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()
        .map_err(|error| format!("invalid R2T2 JSON: {error}"))?;
    if events.iter().any(|event| !event.is_object())
        || events
            .iter()
            .filter(|event| event.get("is_final") == Some(&serde_json::Value::Bool(true)))
            .count()
            != 1
    {
        return Err("R2T2 did not return exactly one final event".into());
    }
    let event = events.last().ok_or("R2T2 returned no event")?;
    if event.get("is_final") != Some(&serde_json::Value::Bool(true))
        || event.get("cancelled") != Some(&serde_json::Value::Bool(false))
        || event.get("token_limit_reached") != Some(&serde_json::Value::Bool(false))
    {
        return Err("R2T2 transcription was cancelled, truncated, or incomplete".into());
    }
    serde_json::from_value(event.clone()).map_err(|error| format!("invalid R2T2 event: {error}"))
}

#[derive(Clone, Debug, Serialize)]
pub struct Assessment {
    pub transcript: String,
    pub reported_language: String,
    pub han_fraction: f64,
    pub similarity: f64,
    pub passed: bool,
}

pub fn assess(reference: &str, event: &AsrEvent) -> Assessment {
    fn han(ch: &char) -> bool {
        ('\u{4e00}'..='\u{9fff}').contains(ch)
    }
    let reference: Vec<_> = simplified(reference).chars().filter(han).collect();
    let normalized = simplified(&event.text);
    let observed: Vec<_> = normalized.chars().filter(han).collect();
    let spoken = normalized.chars().filter(|ch| ch.is_alphanumeric()).count();
    let han_fraction = if spoken == 0 {
        0.0
    } else {
        observed.len() as f64 / spoken as f64
    };
    let similarity = matching_ratio(&reference, &observed);
    let passed =
        reference.len() >= 2 && observed.len() >= 2 && han_fraction >= 0.8 && similarity >= 0.75;
    Assessment {
        transcript: event.text.clone(),
        reported_language: event.language.clone(),
        han_fraction,
        similarity,
        passed,
    }
}
