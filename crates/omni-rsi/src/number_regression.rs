//! A bounded, separately scored Chinese text regression. This is not eligibility evidence.

use crate::number_v3_evaluation::{DecodedText, TextEnd};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

const MAX_MANIFEST: usize = 256 * 1024;
const MAX_REPORT: usize = 2 * 1024 * 1024;
const MAX_CASES: usize = 128;
const MAX_OUTPUT: usize = 4096;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestFile {
    version: u32,
    group: String,
    scoring: String,
    normal_end: String,
    decode: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    prompt: String,
    reference: String,
}

/// Validated frozen inputs. References are never passed to the inference callback.
pub struct ChineseManifest {
    file: ManifestFile,
    sha256: String,
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

impl ChineseManifest {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > MAX_MANIFEST {
            return Err("Chinese regression manifest exceeds size limit".into());
        }
        let file: ManifestFile = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if file.version != 1
            || file.group != "chinese"
            || file.scoring != "exact-trim-v1"
            || file.normal_end != "registered-stop-token-v1"
            || file.decode != "v3-greedy-v1"
            || file.cases.is_empty()
            || file.cases.len() > MAX_CASES
        {
            return Err("unsupported Chinese regression manifest or case count".into());
        }
        let mut ids = HashSet::new();
        let mut prompts = HashSet::new();
        for case in &file.cases {
            if case.id.is_empty()
                || case.id.len() > 64
                || !case
                    .id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
                || !ids.insert(&case.id)
                || !prompts.insert(&case.prompt)
                || case.prompt.is_empty()
                || case.prompt.len() > 2048
                || case.reference.is_empty()
                || case.reference.len() > 256
                || case.reference.trim() != case.reference
                || case.prompt.contains(&case.reference)
                || case.prompt.contains("<|")
                || case
                    .prompt
                    .chars()
                    .any(|c| c.is_control() && c != '\n' && c != '\t')
                || case.reference.chars().any(char::is_control)
            {
                return Err("invalid or duplicate Chinese regression case".into());
            }
        }
        Ok(Self {
            file,
            sha256: hash(bytes),
        })
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    Baseline,
    Candidate,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "token",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum End {
    Stop(u32),
    Limit,
    Cancelled,
}

impl From<TextEnd> for End {
    fn from(end: TextEnd) -> Self {
        match end {
            TextEnd::Stop(token) => Self::Stop(token),
            TextEnd::Limit => Self::Limit,
            TextEnd::Cancelled => Self::Cancelled,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    arm: Arm,
    case_id: String,
    prompt_sha256: String,
    raw_text: String,
    end: End,
    normal_end: bool,
    correct: bool,
}

#[derive(Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Counts {
    correct: usize,
    normal_end: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Report {
    version: u32,
    group: String,
    status: String,
    candidate_eligible: bool,
    execution: String,
    manifest_sha256: String,
    baseline: Counts,
    candidate: Counts,
    correct_delta: i64,
    normal_end_delta: i64,
    observations: Vec<Observation>,
}

fn diagnostic(
    manifest: &ChineseManifest,
    rows: &[Observation],
    current: (Arm, &Case),
    outcome: &str,
    failure: Option<&str>,
) -> String {
    let failure = failure.map(|text| text.chars().take(1024).collect::<String>());
    serde_json::json!({"status":"insufficient_evidence", "candidate_eligible":false,
        "outcome":outcome, "manifest_sha256":manifest.sha256, "observations":rows,
        "current":{"arm":current.0,"case_id":current.1.id,"prompt_sha256":hash(current.1.prompt.as_bytes())},
        "failure":failure})
    .to_string()
}

pub(crate) fn check_stops(stops: &[u32]) -> Result<(), String> {
    if stops.is_empty()
        || stops.len() > 64
        || stops.iter().collect::<HashSet<_>>().len() != stops.len()
    {
        return Err("invalid registered stop tokens".into());
    }
    Ok(())
}

fn score(case: &Case, raw: &str, end: &End, stops: &[u32]) -> (bool, bool) {
    let normal = matches!(end, End::Stop(token) if stops.contains(token));
    (normal, normal && raw.trim() == case.reference)
}

pub(crate) fn verify(
    manifest: &ChineseManifest,
    stops: &[u32],
    report: &Report,
) -> Result<(), String> {
    check_stops(stops)?;
    if report.version != 1
        || report.group != "chinese"
        || report.status != "insufficient_evidence"
        || report.candidate_eligible
        || report.manifest_sha256 != manifest.sha256
        || !matches!(
            report.execution.as_str(),
            "injected_unverified" | "local_rust_burn_unverified"
        )
        || report.observations.len() != manifest.file.cases.len() * 2
    {
        return Err("invalid Chinese regression report header or case count".into());
    }
    let cases: HashMap<_, _> = manifest
        .file
        .cases
        .iter()
        .map(|c| (c.id.as_str(), c))
        .collect();
    let mut seen = HashSet::new();
    let (mut baseline, mut candidate) = (Counts::default(), Counts::default());
    for row in &report.observations {
        if matches!(&row.end, End::Cancelled) {
            return Err("cancelled observation cannot form a complete report".into());
        }
        let case = cases.get(row.case_id.as_str()).ok_or("unknown case")?;
        if !seen.insert((row.arm, row.case_id.as_str()))
            || row.raw_text.len() > MAX_OUTPUT
            || row.prompt_sha256 != hash(case.prompt.as_bytes())
        {
            return Err("invalid or duplicate observation".into());
        }
        let (normal, correct) = score(case, &row.raw_text, &row.end, stops);
        if normal != row.normal_end || correct != row.correct {
            return Err("observation score mismatch".into());
        }
        let counts = match row.arm {
            Arm::Baseline => &mut baseline,
            Arm::Candidate => &mut candidate,
        };
        counts.normal_end += usize::from(normal);
        counts.correct += usize::from(correct);
    }
    if baseline != report.baseline
        || candidate != report.candidate
        || report.correct_delta != candidate.correct as i64 - baseline.correct as i64
        || report.normal_end_delta != candidate.normal_end as i64 - baseline.normal_end as i64
    {
        return Err("Chinese regression summary mismatch".into());
    }
    Ok(())
}

/// Independently recomputes the result of a serialized report against the frozen manifest.
pub fn recheck(manifest: &ChineseManifest, stops: &[u32], bytes: &[u8]) -> Result<(), String> {
    if bytes.len() > MAX_REPORT {
        return Err("Chinese regression report exceeds size limit".into());
    }
    let report: Report = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    verify(manifest, stops, &report)
}

/// Injected runs test wiring only; this API cannot mark them as actual inference.
pub fn compare_with<F>(
    manifest: &ChineseManifest,
    stops: &[u32],
    stop: &AtomicBool,
    infer: F,
) -> Result<serde_json::Value, String>
where
    F: FnMut(Arm, &str, &str) -> Result<DecodedText, String>,
{
    compare(manifest, stops, stop, "injected_unverified", infer)
}

pub(crate) fn compare_real<F>(
    manifest: &ChineseManifest,
    stops: &[u32],
    stop: &AtomicBool,
    infer: F,
) -> Result<serde_json::Value, String>
where
    F: FnMut(Arm, &str, &str) -> Result<DecodedText, String>,
{
    compare(manifest, stops, stop, "local_rust_burn_unverified", infer)
}

fn compare<F>(
    manifest: &ChineseManifest,
    stops: &[u32],
    stop: &AtomicBool,
    execution: &str,
    mut infer: F,
) -> Result<serde_json::Value, String>
where
    F: FnMut(Arm, &str, &str) -> Result<DecodedText, String>,
{
    check_stops(stops)?;
    let mut rows = Vec::new();
    let mut observation_bytes = 0_usize;
    let (mut baseline, mut candidate) = (Counts::default(), Counts::default());
    for arm in [Arm::Baseline, Arm::Candidate] {
        for case in &manifest.file.cases {
            if stop.load(Ordering::Acquire) {
                return Err(diagnostic(
                    manifest,
                    &rows,
                    (arm, case),
                    "cancelled_unverified",
                    None,
                ));
            }
            let decoded = match infer(arm, &case.id, &case.prompt) {
                Ok(decoded) => decoded,
                Err(error) => {
                    return Err(diagnostic(
                        manifest,
                        &rows,
                        (arm, case),
                        if stop.load(Ordering::Acquire) {
                            "cancelled_unverified"
                        } else {
                            "failed_unverified"
                        },
                        Some(&error),
                    ));
                }
            };
            let cancelled = matches!(&decoded.end, TextEnd::Cancelled);
            if decoded.text.len() > MAX_OUTPUT {
                return Err(diagnostic(
                    manifest,
                    &rows,
                    (arm, case),
                    if cancelled {
                        "cancelled_unverified"
                    } else {
                        "failed_unverified"
                    },
                    Some("decoded text exceeds size limit; last output omitted"),
                ));
            }
            let end = End::from(decoded.end);
            let (normal_end, correct) = score(case, &decoded.text, &end, stops);
            let counts = match arm {
                Arm::Baseline => &mut baseline,
                Arm::Candidate => &mut candidate,
            };
            counts.normal_end += usize::from(normal_end);
            counts.correct += usize::from(correct);
            let row = Observation {
                arm,
                case_id: case.id.clone(),
                prompt_sha256: hash(case.prompt.as_bytes()),
                raw_text: decoded.text,
                end,
                normal_end,
                correct,
            };
            observation_bytes = observation_bytes
                .saturating_add(serde_json::to_vec(&row).map_err(|e| e.to_string())?.len() + 1);
            if observation_bytes > MAX_REPORT - 32 * 1024 {
                return Err(diagnostic(
                    manifest,
                    &rows,
                    (arm, case),
                    if cancelled {
                        "cancelled_unverified"
                    } else {
                        "failed_unverified"
                    },
                    Some("observation budget exceeded; last output omitted"),
                ));
            }
            rows.push(row);
            if cancelled || stop.load(Ordering::Acquire) {
                return Err(diagnostic(
                    manifest,
                    &rows,
                    (arm, case),
                    "cancelled_unverified",
                    None,
                ));
            }
        }
    }
    let report = Report {
        version: 1,
        group: "chinese".into(),
        status: "insufficient_evidence".into(),
        candidate_eligible: false,
        execution: execution.into(),
        manifest_sha256: manifest.sha256.clone(),
        correct_delta: candidate.correct as i64 - baseline.correct as i64,
        normal_end_delta: candidate.normal_end as i64 - baseline.normal_end as i64,
        baseline,
        candidate,
        observations: rows,
    };
    verify(manifest, stops, &report)?;
    serde_json::to_value(report).map_err(|e| e.to_string())
}
