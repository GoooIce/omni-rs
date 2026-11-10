//! Fixed speech evaluation and held-out recording separation.

use crate::speech_data::{SpeechDataset, file_hash};
use omni_lfm3::speech_adapter::{SpeechObservation, passes_development};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};
use unicode_normalization::UnicodeNormalization;

const EVALUATION: &[u8] = include_bytes!("../../../docs/eval/chinese-speech-generation-v1.json");
const SOURCE: &[u8] = include_bytes!("../../../docs/eval/mandarin-v1.json");

#[derive(Clone, Debug, Deserialize)]
pub struct SpeechCase {
    pub id: String,
    pub category: String,
    pub prompt: String,
    pub reference: String,
    #[serde(default)]
    source_case: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct ReservedAudio {
    source_case: String,
    samples: usize,
    pcm_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SpeechEvaluation {
    pub dev: Vec<SpeechCase>,
    pub test: Vec<SpeechCase>,
    source_manifest: String,
    source_manifest_sha256: String,
    reserved_audio: Vec<ReservedAudio>,
}

fn normalized(text: &str) -> String {
    text.nfkc()
        .flat_map(char::to_lowercase)
        .filter(|ch| ch.is_alphanumeric())
        .collect()
}

fn hex_id(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl SpeechEvaluation {
    pub fn fixed() -> Result<Self, String> {
        let plan: Self = serde_json::from_slice(EVALUATION).map_err(|error| error.to_string())?;
        let mut ids = HashSet::new();
        let mut texts = HashMap::new();
        for (split, cases, minimum) in [("dev", &plan.dev, 10), ("test", &plan.test, 30)] {
            if cases.len() < minimum
                || ["reading", "dialogue"].iter().any(|category| {
                    cases
                        .iter()
                        .filter(|case| &case.category == category)
                        .count()
                        < 5
                })
            {
                return Err("insufficient fixed speech evaluation coverage".into());
            }
            for case in cases {
                let valid_id = !case.id.is_empty()
                    && case.id.len() <= 64
                    && case.id.bytes().enumerate().all(|(i, byte)| {
                        byte.is_ascii_lowercase()
                            || byte.is_ascii_digit()
                            || (i > 0 && (byte == b'_' || byte == b'-'))
                    });
                let prompt = normalized(&case.prompt);
                let reference = normalized(&case.reference);
                if !valid_id
                    || !ids.insert(&case.id)
                    || !matches!(case.category.as_str(), "reading" | "dialogue")
                    || prompt.is_empty()
                    || reference.is_empty()
                    || (case.category == "reading" && prompt != reference)
                {
                    return Err("invalid fixed speech evaluation case".into());
                }
                for text in [prompt, reference] {
                    if texts.get(&text).is_some_and(|previous| *previous != split) {
                        return Err("fixed speech evaluation leaks across splits".into());
                    }
                    texts.insert(text, split);
                }
            }
        }
        if plan.source_manifest != "docs/eval/mandarin-v1.json"
            || format!("{:x}", Sha256::digest(SOURCE)) != plan.source_manifest_sha256
        {
            return Err("speech evaluation source manifest changed".into());
        }
        let source: serde_json::Value =
            serde_json::from_slice(SOURCE).map_err(|error| error.to_string())?;
        let cases = source
            .get("cases")
            .and_then(|value| value.as_array())
            .ok_or("invalid source manifest")?;
        let references: HashMap<_, _> = cases
            .iter()
            .filter_map(|case| Some((case.get("id")?.as_str()?, case.get("reference")?.as_str()?)))
            .collect();
        let sourced: Vec<_> = plan
            .test
            .iter()
            .filter_map(|case| case.source_case.as_ref().map(|id| (id, &case.reference)))
            .collect();
        if sourced
            .iter()
            .any(|(id, reference)| references.get(id.as_str()) != Some(&reference.as_str()))
        {
            return Err("held-out speech reference changed".into());
        }
        let reserved_ids: HashSet<_> = plan
            .reserved_audio
            .iter()
            .map(|item| item.source_case.as_str())
            .collect();
        let sourced_ids: HashSet<_> = sourced.iter().map(|(id, _)| id.as_str()).collect();
        if reserved_ids.len() != plan.reserved_audio.len()
            || reserved_ids != sourced_ids
            || plan
                .reserved_audio
                .iter()
                .any(|item| !(6400..=240_000).contains(&item.samples) || !hex_id(&item.pcm_sha256))
        {
            return Err("invalid reserved speech fingerprints".into());
        }
        Ok(plan)
    }

    pub fn fingerprint(&self) -> String {
        format!("{:x}", Sha256::digest(EVALUATION))
    }

    /// Reject training and monitoring text overlapping any fixed evaluation.
    pub fn check_data(&self, data: &SpeechDataset) -> Result<(), String> {
        let reserved: HashSet<_> = self
            .dev
            .iter()
            .chain(&self.test)
            .flat_map(|case| [normalized(&case.prompt), normalized(&case.reference)])
            .collect();
        for item in data.train.iter().chain(&data.dev) {
            if reserved.contains(&normalized(&item.record.question))
                || reserved.contains(&normalized(&item.record.answer))
            {
                return Err("speech training text leaks into fixed evaluation".into());
            }
        }
        Ok(())
    }

    pub fn check_training_text(&self, question: &str, answer: &str) -> Result<(), String> {
        let reserved: HashSet<_> = self
            .dev
            .iter()
            .chain(&self.test)
            .flat_map(|case| [normalized(&case.prompt), normalized(&case.reference)])
            .collect();
        if reserved.contains(&normalized(question)) || reserved.contains(&normalized(answer)) {
            return Err("speech training text leaks into fixed evaluation".into());
        }
        Ok(())
    }

    pub fn is_reserved_pcm(&self, path: &Path) -> Result<bool, String> {
        let mut audio = hound::WavReader::open(path).map_err(|error| error.to_string())?;
        let spec = audio.spec();
        if spec.channels != 1
            || spec.sample_rate != 16_000
            || spec.bits_per_sample != 16
            || spec.sample_format != hound::SampleFormat::Int
        {
            return Err("invalid original speech audio".into());
        }
        let mut digest = Sha256::new();
        let mut samples = 0;
        for sample in audio.samples::<i16>() {
            digest.update(sample.map_err(|error| error.to_string())?.to_le_bytes());
            samples += 1;
        }
        let digest = format!("{:x}", digest.finalize());
        Ok(self
            .reserved_audio
            .iter()
            .any(|item| item.samples == samples && item.pcm_sha256 == digest))
    }

    /// Recheck the original audio of every admitted online record at resume.
    pub fn check_online_originals(&self, online: &Path) -> Result<(), String> {
        let root = online
            .parent()
            .ok_or("online speech directory has no parent")?;
        for entry in fs::read_dir(online).map_err(|error| error.to_string())? {
            let entry = entry.map_err(|error| error.to_string())?;
            let id = entry
                .file_name()
                .into_string()
                .map_err(|_| "invalid online speech directory")?;
            if id.starts_with('.') {
                continue;
            }
            if !hex_id(&id)
                || !entry
                    .file_type()
                    .map_err(|error| error.to_string())?
                    .is_dir()
            {
                return Err("invalid online speech directory".into());
            }
            let paths = [
                root.join("processed").join(&id).join("audio.wav"),
                root.join("processing").join(&id).join("audio.wav"),
            ];
            let audio = paths
                .iter()
                .find(|path| {
                    fs::symlink_metadata(path).is_ok_and(|metadata| {
                        metadata.is_file() && !metadata.file_type().is_symlink()
                    })
                })
                .ok_or("original online speech is missing")?;
            if file_hash(audio)? != id {
                return Err("original online speech changed".into());
            }
            if self.is_reserved_pcm(audio)? {
                return Err("online speech matches reserved evaluation audio".into());
            }
        }
        Ok(())
    }

    pub fn passes_development(&self, results: &[SpeechObservation]) -> Result<bool, String> {
        passes_development(results).map_err(|error| error.to_string())
    }

    /// Verify the exact human recordings backing the held-out reading cases.
    pub fn verify_human_sources(
        &self,
        repository: &Path,
    ) -> Result<Vec<serde_json::Value>, String> {
        let root = repository
            .canonicalize()
            .map_err(|error| error.to_string())?;
        let source: serde_json::Value =
            serde_json::from_slice(SOURCE).map_err(|error| error.to_string())?;
        let cases = source
            .get("cases")
            .and_then(|value| value.as_array())
            .ok_or("invalid source manifest")?;
        let mut checked = Vec::new();
        for case in self.test.iter().filter(|case| case.source_case.is_some()) {
            let id = case.source_case.as_deref().ok_or("missing human source")?;
            let source = cases
                .iter()
                .find(|item| item.get("id").and_then(|value| value.as_str()) == Some(id))
                .ok_or("missing human source")?;
            let name = source
                .get("audio")
                .and_then(|value| value.as_str())
                .ok_or("missing human audio path")?;
            let expected = source
                .get("sha256")
                .and_then(|value| value.as_str())
                .ok_or("missing human audio fingerprint")?;
            let audio = root
                .join(name)
                .canonicalize()
                .map_err(|error| error.to_string())?;
            if !audio.starts_with(&root) || file_hash(&audio)? != expected {
                return Err(
                    "human reference audio is missing, changed, or outside the project".into(),
                );
            }
            let mut wav = hound::WavReader::open(&audio).map_err(|error| error.to_string())?;
            let spec = wav.spec();
            if spec.channels != 1
                || spec.sample_rate != 16_000
                || spec.bits_per_sample != 16
                || spec.sample_format != hound::SampleFormat::Int
            {
                return Err("invalid human reference audio format".into());
            }
            let mut hash = Sha256::new();
            let mut count = 0;
            for sample in wav.samples::<i16>() {
                let value = i32::from(sample.map_err(|error| error.to_string())?);
                let magnitude = (value.abs() * 32767 + 16384) / 32768;
                let converted = if value < 0 { -magnitude } else { magnitude } as i16;
                hash.update(converted.to_le_bytes());
                count += 1;
            }
            let reserved = self
                .reserved_audio
                .iter()
                .find(|item| item.source_case == id)
                .ok_or("missing reserved speech fingerprint")?;
            if count != reserved.samples || format!("{:x}", hash.finalize()) != reserved.pcm_sha256
            {
                return Err("human reference capture fingerprint mismatch".into());
            }
            checked.push(serde_json::json!({"id":id,"audio_sha256":expected}));
        }
        Ok(checked)
    }

    /// Report held-out error rates without producing a promotion decision.
    pub fn summarize(
        &self,
        cases: &[SpeechCase],
        results: &[SpeechObservation],
    ) -> Result<serde_json::Value, String> {
        fn distance(left: &[char], right: &[char]) -> usize {
            let mut previous: Vec<_> = (0..=right.len()).collect();
            for (i, a) in left.iter().enumerate() {
                let mut current = vec![i + 1];
                for (j, b) in right.iter().enumerate() {
                    current.push(
                        (previous[j + 1] + 1)
                            .min(current[j] + 1)
                            .min(previous[j] + usize::from(a != b)),
                    );
                }
                previous = current;
            }
            previous[right.len()]
        }
        if results.len() != cases.len() * 2 {
            return Err("incomplete speech evaluation observations".into());
        }
        let mut observations = HashMap::new();
        for result in results {
            if observations
                .insert((result.sample.as_str(), result.kind.as_str()), result)
                .is_some()
            {
                return Err("duplicate speech evaluation observation".into());
            }
        }
        let mut groups =
            HashMap::<&str, [usize; 3]>::from([("reading", [0; 3]), ("dialogue", [0; 3])]);
        let mut baseline_stopped = 0;
        let mut candidate_stopped = 0;
        let mut candidate_valid = true;
        for case in cases {
            let baseline = observations
                .get(&(case.id.as_str(), "baseline"))
                .ok_or("missing baseline speech")?;
            let candidate = observations
                .get(&(case.id.as_str(), "candidate"))
                .ok_or("missing candidate speech")?;
            if baseline.category != case.category || candidate.category != case.category {
                return Err("speech category mismatch".into());
            }
            baseline_stopped += usize::from(baseline.stopped);
            candidate_stopped += usize::from(candidate.stopped);
            candidate_valid &= candidate.stopped
                && candidate.frames > 0
                && !normalized(&candidate.transcript).is_empty();
            let reference: Vec<_> = normalized(&case.reference).chars().collect();
            if reference.is_empty() {
                return Err("empty speech evaluation reference".into());
            }
            let values = groups
                .get_mut(case.category.as_str())
                .ok_or("invalid speech category")?;
            values[0] += distance(
                &reference,
                &normalized(&baseline.transcript).chars().collect::<Vec<_>>(),
            );
            values[1] += distance(
                &reference,
                &normalized(&candidate.transcript)
                    .chars()
                    .collect::<Vec<_>>(),
            );
            values[2] += reference.len();
        }
        let baseline: usize = groups.values().map(|values| values[0]).sum();
        let candidate: usize = groups.values().map(|values| values[1]).sum();
        let count: usize = groups.values().map(|values| values[2]).sum();
        if count == 0 {
            return Err("empty speech evaluation".into());
        }
        let totals: HashMap<_, _> = groups.into_iter().map(|(name, values)| (name, serde_json::json!({
            "baseline_errors":values[0],"candidate_errors":values[1],"reference_characters":values[2]
        }))).collect();
        Ok(
            serde_json::json!({"baseline_cer":baseline as f64 / count as f64,
            "candidate_cer":candidate as f64 / count as f64,"baseline_stopped":baseline_stopped,
            "candidate_stopped":candidate_stopped,"candidate_valid":candidate_valid,
            "groups":totals,"results":results}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_pcm_and_changed_original_are_refused() {
        let directory = tempfile::tempdir().unwrap();
        let audio = directory.path().join("speech.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&audio, spec).unwrap();
        let mut hash = Sha256::new();
        for _ in 0..6400 {
            writer.write_sample(1234_i16).unwrap();
            hash.update(1234_i16.to_le_bytes());
        }
        writer.finalize().unwrap();
        let mut evaluation = SpeechEvaluation::fixed().unwrap();
        evaluation.reserved_audio[0].samples = 6400;
        evaluation.reserved_audio[0].pcm_sha256 = format!("{:x}", hash.finalize());
        assert!(evaluation.is_reserved_pcm(&audio).unwrap());
        let id = file_hash(&audio).unwrap();
        let online = directory.path().join("online-data");
        fs::create_dir_all(online.join(&id)).unwrap();
        let archived = directory.path().join("processed").join(&id);
        fs::create_dir_all(&archived).unwrap();
        fs::copy(&audio, archived.join("audio.wav")).unwrap();
        assert!(evaluation.check_online_originals(&online).is_err());
        fs::remove_file(archived.join("audio.wav")).unwrap();
        assert!(evaluation.check_online_originals(&online).is_err());
    }
}
