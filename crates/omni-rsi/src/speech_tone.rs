//! Data and evaluation contracts for observable speech expression.

use crate::speech_data::file_hash;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    path::Path,
};
use unicode_normalization::UnicodeNormalization;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToneSplit {
    Train,
    Dev,
    Holdout,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToneAudio {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToneSourceKind {
    Human,
    Synthetic,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToneSource {
    pub name: String,
    pub license: String,
    pub kind: ToneSourceKind,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", content = "label", rename_all = "snake_case")]
pub enum AnnotationDecision {
    Label(String),
    Uncertain,
    Abstain,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToneAnnotation {
    pub annotator_id: String,
    pub source: String,
    pub decision: AnnotationDecision,
}

/// An independently sourced reference for an observable expression label.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToneReference {
    pub label: String,
    pub source: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToneRecord {
    pub id: String,
    pub split: ToneSplit,
    pub audio: ToneAudio,
    pub text: String,
    pub text_family: String,
    pub speaker_id: String,
    pub session_id: String,
    pub derived_family: String,
    pub source: ToneSource,
    pub annotations: Vec<ToneAnnotation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<ToneReference>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToneDataset {
    pub version: u32,
    pub labels: Vec<String>,
    pub records: Vec<ToneRecord>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct AnnotationSummary {
    pub votes: BTreeMap<String, usize>,
    pub uncertain: usize,
    pub abstained: usize,
}

impl ToneDataset {
    /// Check the dataset before using it for training or evaluation.
    pub fn validate(&self, root: &Path) -> Result<(), String> {
        self.validate_with_audio(root).map(|_| ())
    }

    fn validate_with_audio(&self, root: &Path) -> Result<HashMap<String, String>, String> {
        let root = root
            .canonicalize()
            .map_err(|e| format!("dataset root: {e}"))?;
        let labels: HashSet<_> = self.labels.iter().collect();
        if self.version != 1
            || self.records.is_empty()
            || self.labels.is_empty()
            || labels.len() != self.labels.len()
            || self.labels.iter().any(|label| !valid_key(label))
        {
            return Err("invalid tone dataset version, records or labels".into());
        }
        let mut ids = HashSet::new();
        let mut groups = HashMap::new();
        let mut texts = HashMap::new();
        let mut audio_samples = HashMap::new();
        let mut fingerprints = HashMap::new();
        for record in &self.records {
            if !ids.insert(&record.id)
                || [
                    &record.id,
                    &record.speaker_id,
                    &record.session_id,
                    &record.derived_family,
                    &record.text_family,
                ]
                .iter()
                .any(|value| !valid_key(value))
                || record.text.trim().is_empty()
                || record.source.name.trim().is_empty()
                || record.source.license.trim().is_empty()
                || record.annotations.is_empty()
            {
                return Err(format!("invalid tone record {}", record.id));
            }
            let text = normalized_text(&record.text);
            if text.is_empty()
                || texts
                    .insert(text, record.split)
                    .is_some_and(|split| split != record.split)
            {
                return Err("empty or cross-split text".into());
            }
            let audio_samples_hash = validate_audio(&root, &record.audio)?;
            if audio_samples
                .insert(audio_samples_hash.clone(), record.split)
                .is_some_and(|split| split != record.split)
            {
                return Err("audio samples cross splits".into());
            }
            fingerprints.insert(record.id.clone(), audio_samples_hash);
            let mut annotators = HashSet::new();
            for annotation in &record.annotations {
                if !valid_key(&annotation.annotator_id)
                    || annotation.source.trim().is_empty()
                    || !annotators.insert(&annotation.annotator_id)
                {
                    return Err(format!("invalid or duplicate annotator in {}", record.id));
                }
                if let AnnotationDecision::Label(label) = &annotation.decision
                    && !labels.contains(label)
                {
                    return Err(format!("unknown annotation label in {}", record.id));
                }
            }
            if let Some(reference) = &record.reference
                && (!labels.contains(&reference.label) || reference.source.trim().is_empty())
            {
                return Err(format!("invalid reference in {}", record.id));
            }
            for (kind, value) in [
                ("speaker", record.speaker_id.as_str()),
                ("session", record.session_id.as_str()),
                ("derived family", record.derived_family.as_str()),
                ("text family", record.text_family.as_str()),
                ("audio", record.audio.sha256.as_str()),
            ] {
                if groups
                    .insert((kind, value), record.split)
                    .is_some_and(|split| split != record.split)
                {
                    return Err(format!("{kind} {value} crosses splits"));
                }
            }
        }
        Ok(fingerprints)
    }

    /// Count the recorded decisions without resolving disagreement.
    pub fn annotation_summary(&self, record_id: &str) -> Result<AnnotationSummary, String> {
        let record = self
            .records
            .iter()
            .find(|record| record.id == record_id)
            .ok_or_else(|| format!("unknown record {record_id}"))?;
        let mut summary = AnnotationSummary::default();
        for annotation in &record.annotations {
            match &annotation.decision {
                AnnotationDecision::Label(label) => {
                    *summary.votes.entry(label.clone()).or_default() += 1;
                }
                AnnotationDecision::Uncertain => summary.uncertain += 1,
                AnnotationDecision::Abstain => summary.abstained += 1,
            }
        }
        Ok(summary)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToneInput {
    Full,
    AudioMasked,
    TextMasked,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToneCase {
    pub id: String,
    pub record_id: String,
    pub input: ToneInput,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TonePairKind {
    SameTextDifferentExpression,
    SameExpressionDifferentText,
    AudioMasked,
    TextMasked,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TonePair {
    pub id: String,
    pub kind: TonePairKind,
    pub original: String,
    pub variant: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PredictionOutcome {
    Label { label: String, confidence: f64 },
    Abstain { reason: String },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TonePrediction {
    pub case_id: String,
    pub outcome: PredictionOutcome,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToneEvaluationPlan {
    pub run_id: String,
    pub split: ToneSplit,
    pub cases: Vec<ToneCase>,
    pub pairs: Vec<TonePair>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToneClassMetrics {
    pub support: usize,
    pub correct: usize,
    pub recall: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CalibrationBin {
    pub index: usize,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub count: usize,
    pub mean_confidence: Option<f64>,
    pub accuracy: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Calibration {
    pub count: usize,
    pub ece: Option<f64>,
    pub bins: Vec<CalibrationBin>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToneMetrics {
    pub total: usize,
    pub reference_count: usize,
    pub predicted: usize,
    pub abstained: usize,
    pub coverage: f64,
    pub abstention_rate: f64,
    pub per_class: BTreeMap<String, ToneClassMetrics>,
    pub macro_recall: Option<f64>,
    pub calibration: Calibration,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToneResultRow {
    pub case: ToneCase,
    pub record: ToneRecord,
    pub prediction: TonePrediction,
    pub correct: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TonePairResult {
    pub pair: TonePair,
    pub changed: Option<bool>,
    pub both_correct: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TonePairMetrics {
    pub total: usize,
    pub comparable: usize,
    pub changed: usize,
    pub unchanged: usize,
    pub both_correct: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToneReport {
    pub run_id: String,
    pub split: ToneSplit,
    pub metrics: ToneMetrics,
    pub by_input: BTreeMap<ToneInput, ToneMetrics>,
    pub pairs: Vec<TonePairResult>,
    pub pair_metrics: BTreeMap<TonePairKind, TonePairMetrics>,
    pub rows: Vec<ToneResultRow>,
}

impl ToneEvaluationPlan {
    /// Summarize caller-supplied predictions; this never runs a model.
    pub fn evaluate(
        &self,
        data: &ToneDataset,
        root: &Path,
        predictions: &[TonePrediction],
    ) -> Result<ToneReport, String> {
        let audio_samples = data.validate_with_audio(root)?;
        if !valid_key(&self.run_id) || self.cases.is_empty() {
            return Err("invalid evaluation plan".into());
        }
        let records: HashMap<_, _> = data.records.iter().map(|r| (r.id.as_str(), r)).collect();
        let labels: HashSet<_> = data.labels.iter().map(String::as_str).collect();
        let mut by_case = HashMap::new();
        for prediction in predictions {
            if by_case
                .insert(prediction.case_id.as_str(), prediction)
                .is_some()
            {
                return Err("duplicate prediction".into());
            }
            match &prediction.outcome {
                PredictionOutcome::Label { label, confidence }
                    if labels.contains(label.as_str())
                        && confidence.is_finite()
                        && (0.0..=1.0).contains(confidence) => {}
                PredictionOutcome::Abstain { reason } if !reason.trim().is_empty() => {}
                _ => return Err("invalid prediction".into()),
            }
        }
        let mut ids = HashSet::new();
        let mut inputs = HashSet::new();
        let mut rows = Vec::with_capacity(self.cases.len());
        for case in &self.cases {
            if !valid_key(&case.id)
                || !ids.insert(case.id.as_str())
                || !inputs.insert((case.record_id.as_str(), case.input))
            {
                return Err("invalid, duplicate or repeated case input".into());
            }
            let record = records
                .get(case.record_id.as_str())
                .filter(|record| record.split == self.split)
                .ok_or("case record missing from evaluation split")?;
            let prediction = by_case
                .remove(case.id.as_str())
                .ok_or("missing prediction")?;
            let correct = match (&record.reference, &prediction.outcome) {
                (Some(reference), PredictionOutcome::Label { label, .. }) => {
                    Some(reference.label == *label)
                }
                (Some(_), PredictionOutcome::Abstain { .. }) => Some(false),
                (None, _) => None,
            };
            rows.push(ToneResultRow {
                case: case.clone(),
                record: (*record).clone(),
                prediction: prediction.clone(),
                correct,
            });
        }
        if !by_case.is_empty() {
            return Err("prediction has no evaluation case".into());
        }
        let indexed: HashMap<_, _> = rows.iter().map(|row| (row.case.id.as_str(), row)).collect();
        let mut pair_ids = HashSet::new();
        let mut seen_pairs = BTreeSet::new();
        let mut pairs = Vec::new();
        let mut pair_metrics: BTreeMap<TonePairKind, TonePairMetrics> = BTreeMap::new();
        for pair in &self.pairs {
            let endpoints = if pair.original < pair.variant {
                (pair.original.as_str(), pair.variant.as_str())
            } else {
                (pair.variant.as_str(), pair.original.as_str())
            };
            if !valid_key(&pair.id)
                || !pair_ids.insert(&pair.id)
                || pair.original == pair.variant
                || !seen_pairs.insert((pair.kind, endpoints))
            {
                return Err("invalid or duplicate counterfactual pair".into());
            }
            let original = indexed
                .get(pair.original.as_str())
                .ok_or("missing original case")?;
            let variant = indexed
                .get(pair.variant.as_str())
                .ok_or("missing variant case")?;
            validate_pair(pair.kind, original, variant, &audio_samples)?;
            let changed = match (&original.prediction.outcome, &variant.prediction.outcome) {
                (
                    PredictionOutcome::Label { label: first, .. },
                    PredictionOutcome::Label { label: second, .. },
                ) => Some(first != second),
                _ => None,
            };
            let both_correct = changed.and(match (original.correct, variant.correct) {
                (Some(first), Some(second)) => Some(first && second),
                _ => None,
            });
            let stats = pair_metrics.entry(pair.kind).or_default();
            stats.total += 1;
            if let Some(changed) = changed {
                stats.comparable += 1;
                if changed {
                    stats.changed += 1;
                } else {
                    stats.unchanged += 1;
                }
            }
            stats.both_correct += usize::from(both_correct == Some(true));
            pairs.push(TonePairResult {
                pair: pair.clone(),
                changed,
                both_correct,
            });
        }
        let mut by_input = BTreeMap::new();
        for input in [
            ToneInput::Full,
            ToneInput::AudioMasked,
            ToneInput::TextMasked,
        ] {
            let selected: Vec<_> = rows
                .iter()
                .filter(|row| row.case.input == input)
                .cloned()
                .collect();
            if !selected.is_empty() {
                by_input.insert(input, summarize(&selected, &data.labels));
            }
        }
        Ok(ToneReport {
            run_id: self.run_id.clone(),
            split: self.split,
            metrics: summarize(&rows, &data.labels),
            by_input,
            pairs,
            pair_metrics,
            rows,
        })
    }
}

fn validate_pair(
    kind: TonePairKind,
    original: &ToneResultRow,
    variant: &ToneResultRow,
    audio_samples: &HashMap<String, String>,
) -> Result<(), String> {
    let first = &original.record;
    let second = &variant.record;
    let valid = match kind {
        TonePairKind::SameTextDifferentExpression | TonePairKind::SameExpressionDifferentText => {
            let references = first.reference.as_ref().zip(second.reference.as_ref());
            let same_text = first.text == second.text;
            let different_text = normalized_text(&first.text) != normalized_text(&second.text);
            let different_audio = audio_samples.get(&first.id) != audio_samples.get(&second.id);
            let relation = references.is_some_and(|(a, b)| match kind {
                TonePairKind::SameTextDifferentExpression => same_text && a.label != b.label,
                TonePairKind::SameExpressionDifferentText => different_text && a.label == b.label,
                _ => unreachable!(),
            });
            original.case.input == ToneInput::Full
                && variant.case.input == ToneInput::Full
                && different_audio
                && relation
        }
        TonePairKind::AudioMasked => {
            first.id == second.id
                && original.case.input == ToneInput::Full
                && variant.case.input == ToneInput::AudioMasked
        }
        TonePairKind::TextMasked => {
            first.id == second.id
                && original.case.input == ToneInput::Full
                && variant.case.input == ToneInput::TextMasked
        }
    };
    if !valid {
        return Err(format!("invalid {kind:?} counterfactual pair"));
    }
    Ok(())
}

fn summarize(rows: &[ToneResultRow], labels: &[String]) -> ToneMetrics {
    let mut per_class: BTreeMap<_, _> = labels
        .iter()
        .map(|label| {
            (
                label.clone(),
                ToneClassMetrics {
                    support: 0,
                    correct: 0,
                    recall: None,
                },
            )
        })
        .collect();
    let mut predicted = 0;
    let mut reference_count = 0;
    let mut bin_sums = [(0usize, 0.0f64, 0usize); 10];
    for row in rows {
        if matches!(row.prediction.outcome, PredictionOutcome::Label { .. }) {
            predicted += 1;
        }
        if let Some(reference) = &row.record.reference {
            reference_count += 1;
            let class = per_class
                .get_mut(&reference.label)
                .expect("validated label");
            class.support += 1;
            if row.correct == Some(true) {
                class.correct += 1;
            }
            if let PredictionOutcome::Label { confidence, .. } = row.prediction.outcome {
                let index = ((confidence * 10.0) as usize).min(9);
                let bin = &mut bin_sums[index];
                bin.0 += 1;
                bin.1 += confidence;
                bin.2 += usize::from(row.correct == Some(true));
            }
        }
    }
    let mut recall_sum = 0.0;
    let mut represented = 0;
    for class in per_class.values_mut() {
        if class.support > 0 {
            let recall = class.correct as f64 / class.support as f64;
            class.recall = Some(recall);
            recall_sum += recall;
            represented += 1;
        }
    }
    let count: usize = bin_sums.iter().map(|bin| bin.0).sum();
    let mut weighted_error = 0.0;
    let bins = bin_sums
        .iter()
        .enumerate()
        .map(|(index, &(n, confidence_sum, correct))| {
            let (mean_confidence, accuracy) = if n > 0 {
                let confidence = confidence_sum / n as f64;
                let accuracy = correct as f64 / n as f64;
                weighted_error += n as f64 * (confidence - accuracy).abs();
                (Some(confidence), Some(accuracy))
            } else {
                (None, None)
            };
            CalibrationBin {
                index,
                lower_bound: index as f64 / 10.0,
                upper_bound: (index + 1) as f64 / 10.0,
                count: n,
                mean_confidence,
                accuracy,
            }
        })
        .collect();
    ToneMetrics {
        total: rows.len(),
        reference_count,
        predicted,
        abstained: rows.len() - predicted,
        coverage: predicted as f64 / rows.len() as f64,
        abstention_rate: (rows.len() - predicted) as f64 / rows.len() as f64,
        per_class,
        macro_recall: (represented > 0).then(|| recall_sum / represented as f64),
        calibration: Calibration {
            count,
            ece: (count > 0).then(|| weighted_error / count as f64),
            bins,
        },
    }
}

pub(crate) fn valid_key(value: &str) -> bool {
    !value.is_empty() && value.trim() == value
}

pub(crate) fn normalized_text(text: &str) -> String {
    text.nfkc()
        .flat_map(char::to_lowercase)
        .filter(|ch| ch.is_alphanumeric())
        .collect()
}

fn validate_audio(root: &Path, audio: &ToneAudio) -> Result<String, String> {
    if audio.path.trim().is_empty()
        || Path::new(&audio.path).is_absolute()
        || audio.sha256.len() != 64
        || !audio
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("invalid audio path or SHA-256".into());
    }
    let path = root
        .join(&audio.path)
        .canonicalize()
        .map_err(|e| format!("audio is missing: {e}"))?;
    if !path.starts_with(root) || !path.is_file() {
        return Err("audio must be a file inside the dataset root".into());
    }
    if file_hash(&path)? != audio.sha256 {
        return Err("audio SHA-256 does not match the file".into());
    }
    let mut reader = hound::WavReader::open(path).map_err(|e| format!("invalid WAV: {e}"))?;
    let spec = reader.spec();
    let expected_samples = u64::from(reader.duration()) * u64::from(spec.channels);
    if spec.sample_rate == 0 || spec.channels == 0 || expected_samples == 0 {
        return Err("audio WAV is empty or has an invalid format".into());
    }
    let mut samples_hash = Sha256::new();
    samples_hash.update(spec.sample_rate.to_le_bytes());
    samples_hash.update(spec.channels.to_le_bytes());
    samples_hash.update(spec.bits_per_sample.to_le_bytes());
    samples_hash.update([match spec.sample_format {
        hound::SampleFormat::Int => 0,
        hound::SampleFormat::Float => 1,
    }]);
    let mut count = 0u64;
    match spec.sample_format {
        hound::SampleFormat::Int => {
            for sample in reader.samples::<i32>() {
                let sample = sample.map_err(|e| format!("invalid WAV sample: {e}"))?;
                samples_hash.update(sample.to_le_bytes());
                count += 1;
            }
        }
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                let sample = sample.map_err(|e| format!("invalid WAV sample: {e}"))?;
                if !sample.is_finite() {
                    return Err("nonfinite WAV sample".into());
                }
                samples_hash.update((if sample == 0.0 { 0.0 } else { sample }).to_le_bytes());
                count += 1;
            }
        }
    }
    if count != expected_samples {
        return Err("truncated WAV samples".into());
    }
    Ok(format!("{:x}", samples_hash.finalize()))
}
