//! Offline text-only exploratory screening; it does not establish audio truth.

use crate::{alimeeting_fs, alimeeting_textgrid, speech_number_evaluation::answer_number};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{self, Write},
    path::Path,
};
use unicode_normalization::UnicodeNormalization;

const MAX_INTERVALS: usize = 1_000_000;
const MAX_ROWS: usize = 100_000;
const MAX_OUTPUT_BYTES: usize = 64 * 1024 * 1024;
const MAX_ACCUMULATED_REPORT_BYTES: usize = 32 * 1024 * 1024;
// Fixed charges keep the acceptance boundary the same across architectures.
const SOURCE_BUDGET_OVERHEAD: usize = 320;
const ROW_BUDGET_OVERHEAD: usize = 512;

#[derive(Serialize)]
struct SourceFile {
    relative_path: String,
    bytes: usize,
    sha256: String,
    spk_suffix: String,
}

#[derive(Serialize)]
struct Candidate {
    source_file: String,
    recording_id: String,
    spk_suffix: String,
    tier_name: String,
    tier_index: usize,
    interval_index: usize,
    start_seconds: f64,
    end_seconds: f64,
    mark: String,
    candidate_number: u8,
}

#[derive(Serialize)]
struct Deferred {
    source_file: String,
    recording_id: String,
    spk_suffix: String,
    tier_name: String,
    tier_index: usize,
    interval_index: usize,
    start_seconds: f64,
    end_seconds: f64,
    mark: String,
    reason: &'static str,
    reasons: &'static [&'static str],
}

const _: () = assert!(2 * std::mem::size_of::<SourceFile>() + 128 <= SOURCE_BUDGET_OVERHEAD);
const _: () = assert!(2 * std::mem::size_of::<Candidate>() + 128 <= ROW_BUDGET_OVERHEAD);
const _: () = assert!(2 * std::mem::size_of::<Deferred>() + 128 <= ROW_BUDGET_OVERHEAD);

#[derive(Serialize)]
struct Summary {
    source_files: usize,
    interval_count: usize,
    candidate_count: usize,
    distinct_tier_ids: usize,
    tier_id_counts: BTreeMap<String, usize>,
    distinct_recording_tier_ids: usize,
    recording_tier_counts: Vec<RecordingTierCount>,
    distinct_source_spk_suffixes: usize,
    source_spk_file_counts: BTreeMap<String, usize>,
    source_unknown_spk_files: usize,
    distinct_candidate_spk_suffixes: usize,
    candidate_spk_counts: BTreeMap<String, usize>,
    candidate_unknown_spk_count: usize,
    distinct_deferred_spk_suffixes: usize,
    deferred_spk_counts: BTreeMap<String, usize>,
    deferred_unknown_spk_count: usize,
    deferred_adjacent_count: usize,
    deferred_ambiguous_notation_count: usize,
}

#[derive(Serialize)]
struct RecordingTierCount {
    recording_id: String,
    tier_name: String,
    candidate_count: usize,
}

#[derive(Serialize)]
struct Report {
    kind: &'static str,
    rule: &'static str,
    context_policy: &'static str,
    limitations: &'static [&'static str],
    tool_sha256: String,
    source_sha256: String,
    source_files: Vec<SourceFile>,
    summary: Summary,
    candidates: Vec<Candidate>,
    deferred: Vec<Deferred>,
}

// The scoring parser accepts answer labels and many extra forms. This lexical
// gate first restricts it to the documented whole-number-v1 vocabulary.
fn whole_number(mark: &str) -> Option<u8> {
    let normalized: String = mark.nfkc().collect();
    let text = normalized
        .trim_matches(|ch: char| ch.is_whitespace() || ",，。.!！?？、:：;；".contains(ch));
    if text.is_empty() {
        return None;
    }
    let allowed = if text.bytes().all(|byte| byte.is_ascii_digit()) {
        text.len() <= 2 && (text.len() == 1 || !text.starts_with('0'))
    } else {
        let digits = "一二三四五六七八九";
        let tens = "二三四五六七八九";
        let chars: Vec<_> = text.chars().collect();
        match chars.as_slice() {
            [ch] => "零〇一二两三四五六七八九十".contains(*ch),
            [a, b] => {
                (*a == '十' && digits.contains(*b))
                    || (*b == '十' && (*a == '一' || tens.contains(*a)))
            }
            [a, b, c] => *b == '十' && (*a == '一' || tens.contains(*a)) && digits.contains(*c),
            _ => false,
        }
    };
    if !allowed {
        return None;
    }
    answer_number(text)
        .and_then(|number| u8::try_from(number).ok())
        .filter(|number| *number <= 99)
}

// NFKC can convert circled or superscript characters to digits, and stripping
// a leading period can turn a decimal fragment into an integer. Keep those
// lexical matches visible for review without assigning a candidate number.
fn ambiguous_notation(mark: &str) -> bool {
    let chinese_digits = "零〇一二两三四五六七八九十";
    if mark.chars().any(|ch| {
        if ch.is_ascii_digit() || ('０'..='９').contains(&ch) || chinese_digits.contains(ch) {
            return false;
        }
        ch.to_string()
            .nfkc()
            .any(|normalized| normalized.is_ascii_digit() || chinese_digits.contains(normalized))
    }) {
        return true;
    }
    let normalized: String = mark.nfkc().collect();
    let text = normalized
        .trim_matches(|ch: char| ch.is_whitespace() || ",，。!！?？、:：;；".contains(ch));
    text.starts_with('.')
        && text
            .trim_start_matches(|ch: char| ch == '.' || ch.is_whitespace())
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_digit() || chinese_digits.contains(ch))
}

fn recording_id(relative: &str) -> Result<String, String> {
    let filename = relative
        .rsplit('/')
        .next()
        .ok_or("missing TextGrid filename")?;
    let (stem, extension) = filename
        .rsplit_once('.')
        .ok_or("invalid TextGrid filename")?;
    if !extension.eq_ignore_ascii_case("TextGrid")
        || stem.is_empty()
        || stem.len() > 200
        || !stem
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        return Err(format!("unsupported recording filename: {relative}"));
    }
    Ok(stem.to_owned())
}

// This is a filename-derived release label, not a verified human identity.
fn spk_suffix(recording: &str) -> &str {
    if let Some((prefix, suffix)) = recording.rsplit_once('_')
        && !prefix.is_empty()
        && let Some(digits) = suffix.strip_prefix("SPK")
        && !digits.is_empty()
        && digits.bytes().all(|byte| byte.is_ascii_digit())
    {
        suffix
    } else {
        "unknown"
    }
}

struct LimitedWriter {
    bytes: Vec<u8>,
}

#[derive(Default)]
struct AccumulationBudget {
    bytes: usize,
}

impl AccumulationBudget {
    fn reserve(&mut self, fixed: usize, strings: &[&str]) -> Result<(), String> {
        let added = strings
            .iter()
            .try_fold(fixed, |total, value| total.checked_add(value.len()))
            .ok_or("accumulated report data exceeds limit")?;
        let total = self
            .bytes
            .checked_add(added)
            .ok_or("accumulated report data exceeds limit")?;
        if total > MAX_ACCUMULATED_REPORT_BYTES {
            return Err("accumulated report data exceeds limit".into());
        }
        self.bytes = total;
        Ok(())
    }
}

fn tool_fingerprint() -> String {
    let sources: &[(&str, &[u8])] = &[
        (
            "alimeeting_screen.rs",
            include_bytes!("alimeeting_screen.rs"),
        ),
        (
            "alimeeting_textgrid.rs",
            include_bytes!("alimeeting_textgrid.rs"),
        ),
        ("alimeeting_fs.rs", include_bytes!("alimeeting_fs.rs")),
        (
            "speech_number_evaluation.rs",
            include_bytes!("speech_number_evaluation.rs"),
        ),
        (
            "alimeeting-screen.rs",
            include_bytes!("bin/alimeeting-screen.rs"),
        ),
        ("Cargo.toml", include_bytes!("../Cargo.toml")),
        ("Cargo.lock", include_bytes!("../Cargo.lock")),
    ];
    let mut hash = Sha256::new();
    hash.update(b"omni-rsi/alimeeting-screen-tool/v1\0");
    for (name, bytes) in sources {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    format!("{:x}", hash.finalize())
}

impl Write for LimitedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_OUTPUT_BYTES.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other("output exceeds limit"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Screen a caller-supplied licensed training TextGrid directory and publish a
/// new JSON report. Source identity covers observed text bytes, not audio or
/// real-person identity. The caller is responsible for the source's licence.
pub fn screen(input: &Path, output: &Path) -> Result<(), String> {
    let root = alimeeting_fs::open_directory(input)?;
    let (output_dir, output_name) = alimeeting_fs::output_parent(&root, output)?;
    let mut source_files = Vec::new();
    let mut candidates = Vec::new();
    let mut deferred = Vec::new();
    let mut seen_recordings = BTreeSet::new();
    let mut interval_count = 0_usize;
    let mut report_budget = AccumulationBudget::default();
    alimeeting_fs::visit_inputs(&root, |relative, bytes| {
        let recording = recording_id(&relative)?;
        let spk_suffix = spk_suffix(&recording);
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        // Account for the source record, recording set and summary keys before
        // retaining copies. The factor covers spare Vec capacity; the extra
        // allowance covers allocation and map-node overhead conservatively.
        report_budget.reserve(
            SOURCE_BUDGET_OVERHEAD,
            &[&relative, &recording, spk_suffix, spk_suffix, &sha256],
        )?;
        if !seen_recordings.insert(recording.clone()) {
            return Err(format!("duplicate recording identifier: {recording}"));
        }
        let text = alimeeting_textgrid::decode(&bytes)?;
        let tiers =
            alimeeting_textgrid::parse(&text).map_err(|error| format!("{relative}: {error}"))?;
        source_files.push(SourceFile {
            relative_path: relative.clone(),
            bytes: bytes.len(),
            sha256,
            spk_suffix: spk_suffix.to_owned(),
        });
        for (tier_offset, tier) in tiers.iter().enumerate() {
            interval_count = interval_count
                .checked_add(tier.intervals.len())
                .ok_or("interval count overflow")?;
            if interval_count > MAX_INTERVALS {
                return Err("total intervals exceed limit".into());
            }
            for (interval_offset, interval) in tier.intervals.iter().enumerate() {
                let Some(number) = whole_number(&interval.mark) else {
                    continue;
                };
                let neighbor_has_text = interval_offset
                    .checked_sub(1)
                    .and_then(|index| tier.intervals.get(index))
                    .is_some_and(|neighbor| !neighbor.mark.trim().is_empty())
                    || tier
                        .intervals
                        .get(interval_offset + 1)
                        .is_some_and(|neighbor| !neighbor.mark.trim().is_empty());
                if candidates.len().saturating_add(deferred.len()) >= MAX_ROWS {
                    return Err("candidate row limit exceeded".into());
                }
                // Per-row copies and worst-case unique summary keys are
                // budgeted before allocation, including spare Vec capacity.
                report_budget.reserve(
                    ROW_BUDGET_OVERHEAD,
                    &[
                        &relative,
                        &recording,
                        &recording,
                        spk_suffix,
                        spk_suffix,
                        &tier.name,
                        &tier.name,
                        &tier.name,
                        &interval.mark,
                    ],
                )?;
                let reasons: &'static [&'static str] =
                    match (ambiguous_notation(&interval.mark), neighbor_has_text) {
                        (true, true) => {
                            &["ambiguous_numeric_notation", "adjacent_nonempty_interval"]
                        }
                        (true, false) => &["ambiguous_numeric_notation"],
                        (false, true) => &["adjacent_nonempty_interval"],
                        (false, false) => &[],
                    };
                if let Some(&reason) = reasons.first() {
                    deferred.push(Deferred {
                        source_file: relative.clone(),
                        recording_id: recording.clone(),
                        spk_suffix: spk_suffix.to_owned(),
                        tier_name: tier.name.clone(),
                        tier_index: tier_offset + 1,
                        interval_index: interval_offset + 1,
                        start_seconds: interval.start,
                        end_seconds: interval.end,
                        mark: interval.mark.clone(),
                        reason,
                        reasons,
                    });
                } else {
                    candidates.push(Candidate {
                        source_file: relative.clone(),
                        recording_id: recording.clone(),
                        spk_suffix: spk_suffix.to_owned(),
                        tier_name: tier.name.clone(),
                        tier_index: tier_offset + 1,
                        interval_index: interval_offset + 1,
                        start_seconds: interval.start,
                        end_seconds: interval.end,
                        mark: interval.mark.clone(),
                        candidate_number: number,
                    });
                }
            }
        }
        Ok(())
    })?;
    source_files.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    candidates.sort_by(|a, b| {
        (&a.source_file, a.tier_index, a.interval_index).cmp(&(
            &b.source_file,
            b.tier_index,
            b.interval_index,
        ))
    });
    deferred.sort_by(|a, b| {
        (&a.source_file, a.tier_index, a.interval_index).cmp(&(
            &b.source_file,
            b.tier_index,
            b.interval_index,
        ))
    });
    let mut hash = Sha256::new();
    hash.update(b"omni-rsi/alimeeting-textgrid-source/v1\0");
    for file in &source_files {
        hash.update((file.relative_path.len() as u64).to_le_bytes());
        hash.update(file.relative_path.as_bytes());
        hash.update((file.bytes as u64).to_le_bytes());
        hash.update(file.sha256.as_bytes());
    }
    let mut tier_id_counts = BTreeMap::new();
    let mut recording_tier_counts = BTreeMap::new();
    let mut source_spk_file_counts = BTreeMap::new();
    let mut source_unknown_spk_files = 0;
    for source in &source_files {
        if source.spk_suffix == "unknown" {
            source_unknown_spk_files += 1;
        } else {
            *source_spk_file_counts
                .entry(source.spk_suffix.clone())
                .or_insert(0) += 1;
        }
    }
    let mut candidate_spk_counts = BTreeMap::new();
    let mut candidate_unknown_spk_count = 0;
    let mut deferred_spk_counts = BTreeMap::new();
    let mut deferred_unknown_spk_count = 0;
    for row in &deferred {
        if row.spk_suffix == "unknown" {
            deferred_unknown_spk_count += 1;
        } else {
            *deferred_spk_counts
                .entry(row.spk_suffix.clone())
                .or_insert(0) += 1;
        }
    }
    for candidate in &candidates {
        *tier_id_counts
            .entry(candidate.tier_name.clone())
            .or_insert(0) += 1;
        *recording_tier_counts
            .entry((candidate.recording_id.clone(), candidate.tier_name.clone()))
            .or_insert(0) += 1;
        if candidate.spk_suffix == "unknown" {
            candidate_unknown_spk_count += 1;
        } else {
            *candidate_spk_counts
                .entry(candidate.spk_suffix.clone())
                .or_insert(0) += 1;
        }
    }
    let distinct_tier_ids = candidates
        .iter()
        .map(|item| &item.tier_name)
        .collect::<BTreeSet<_>>()
        .len();
    let distinct_recording_tier_ids = recording_tier_counts.len();
    let recording_tier_counts = recording_tier_counts
        .into_iter()
        .map(
            |((recording_id, tier_name), candidate_count)| RecordingTierCount {
                recording_id,
                tier_name,
                candidate_count,
            },
        )
        .collect();
    let report = Report {
        kind: "exploratory_text_candidates",
        rule: "whole-number-v1",
        context_policy: "adjacent text and ambiguous numeric notation deferred for review",
        limitations: &[
            "text candidates only; no independent human listening",
            "tier labels are not verified unique people",
            "recording and tier pairs are local source labels, not verified unique people",
            "SPK filename suffixes are release labels, not verified independent people",
            "audio content, completeness and overlap are not checked",
            "time gaps between annotated intervals are not evidence of silence",
            "Arabic numeral text does not distinguish integer from digit-by-digit speech",
            "not evidence of model improvement or holdout passage",
            "source licence and training-only provenance are caller responsibilities",
            "provided directory does not prove complete corpus coverage",
            "only the documented labelled-long TextGrid subset is supported",
            "tool hash identifies compiled-in source text and lockfile, not execution attestation",
        ],
        tool_sha256: tool_fingerprint(),
        source_sha256: format!("{:x}", hash.finalize()),
        summary: Summary {
            source_files: source_files.len(),
            interval_count,
            candidate_count: candidates.len(),
            distinct_tier_ids,
            tier_id_counts,
            distinct_recording_tier_ids,
            recording_tier_counts,
            distinct_source_spk_suffixes: source_spk_file_counts.len(),
            source_spk_file_counts,
            source_unknown_spk_files,
            distinct_candidate_spk_suffixes: candidate_spk_counts.len(),
            candidate_spk_counts,
            candidate_unknown_spk_count,
            distinct_deferred_spk_suffixes: deferred_spk_counts.len(),
            deferred_spk_counts,
            deferred_unknown_spk_count,
            deferred_adjacent_count: deferred
                .iter()
                .filter(|row| row.reasons.contains(&"adjacent_nonempty_interval"))
                .count(),
            deferred_ambiguous_notation_count: deferred
                .iter()
                .filter(|row| row.reasons.contains(&"ambiguous_numeric_notation"))
                .count(),
        },
        source_files,
        candidates,
        deferred,
    };
    let mut output_bytes = LimitedWriter { bytes: Vec::new() };
    serde_json::to_writer_pretty(&mut output_bytes, &report).map_err(|error| error.to_string())?;
    output_bytes
        .write_all(b"\n")
        .map_err(|error| error.to_string())?;
    alimeeting_fs::publish(&output_dir, &output_name, &output_bytes.bytes)
}
