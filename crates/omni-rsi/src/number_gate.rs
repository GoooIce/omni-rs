//! Independent paired statistics for the audio-number development protocol.

use std::collections::HashSet;

/// Content identities supplied by the caller; they are not verified here.
#[derive(Clone, Debug)]
pub struct GateBinding {
    pub rule_sha256: String,
    pub train_sha256: String,
    pub dev_sha256: String,
    /// Commitment to the holdout metadata, never its answers or media.
    pub holdout_sha256: String,
    pub grouping_sha256: String,
    pub baseline_sha256: String,
    pub candidate_sha256: String,
    pub codec_sha256: String,
    pub verifier_sha256: String,
    pub comparison_sha256: String,
    pub evaluation_code_sha256: String,
    pub parameters_sha256: String,
    pub observations_sha256: String,
}

/// A declaration, not proof that a check was actually performed.
#[derive(Clone, Debug, Default)]
pub enum DeclaredCheck {
    #[default]
    Missing,
    Passed {
        artifact_sha256: String,
    },
    Failed {
        artifact_sha256: String,
    },
}

#[derive(Clone, Debug, Default)]
pub struct DevelopmentChecks {
    pub blind_isolated_data: DeclaredCheck,
    pub independent_human_references: DeclaredCheck,
    pub original_scoring: DeclaredCheck,
    pub swapped_scoring: DeclaredCheck,
    pub silence_scoring: DeclaredCheck,
    pub output_speech_and_asr: DeclaredCheck,
    pub normal_end_all_conditions: DeclaredCheck,
    pub chinese_regression: DeclaredCheck,
    pub english_regression: DeclaredCheck,
    pub other_omni_regression: DeclaredCheck,
    pub reading_regression: DeclaredCheck,
    pub dialogue_regression: DeclaredCheck,
}

#[derive(Clone, Debug)]
pub struct DevelopmentInput {
    pub binding: GateBinding,
    pub pairs: Vec<PairedCase>,
    pub checks: DevelopmentChecks,
}

/// No formal pass is available without an adapter that verifies real evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GateStatus {
    InsufficientEvidence,
}

#[derive(Clone, Debug)]
pub struct DevelopmentReport {
    pub status: GateStatus,
    pub statistics: Option<PairedStatistics>,
    pub missing: Vec<String>,
    pub failures: Vec<String>,
}

fn sha256_id(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Evaluate an untrusted development precheck. This cannot freeze a candidate,
/// authorize reading a holdout, or establish that declared evidence exists.
pub fn evaluate_development(input: &DevelopmentInput) -> Result<DevelopmentReport, String> {
    let binding = &input.binding;
    if [
        &binding.rule_sha256,
        &binding.train_sha256,
        &binding.dev_sha256,
        &binding.holdout_sha256,
        &binding.grouping_sha256,
        &binding.baseline_sha256,
        &binding.candidate_sha256,
        &binding.codec_sha256,
        &binding.verifier_sha256,
        &binding.comparison_sha256,
        &binding.evaluation_code_sha256,
        &binding.parameters_sha256,
        &binding.observations_sha256,
    ]
    .iter()
    .any(|hash| !sha256_id(hash))
    {
        return Err("invalid development binding".into());
    }
    let checks = &input.checks;
    let mut missing =
        vec!["real execution, source and frozen protocol verification are not connected".into()];
    let mut failures = Vec::new();
    for (name, check) in [
        ("blind_isolated_data", &checks.blind_isolated_data),
        (
            "independent_human_references",
            &checks.independent_human_references,
        ),
        ("original_scoring", &checks.original_scoring),
        ("swapped_scoring", &checks.swapped_scoring),
        ("silence_scoring", &checks.silence_scoring),
        ("output_speech_and_asr", &checks.output_speech_and_asr),
        (
            "normal_end_all_conditions",
            &checks.normal_end_all_conditions,
        ),
        ("chinese_regression", &checks.chinese_regression),
        ("english_regression", &checks.english_regression),
        ("other_omni_regression", &checks.other_omni_regression),
        ("reading_regression", &checks.reading_regression),
        ("dialogue_regression", &checks.dialogue_regression),
    ] {
        match check {
            DeclaredCheck::Missing => missing.push(name.into()),
            DeclaredCheck::Passed { artifact_sha256 } if sha256_id(artifact_sha256) => {}
            DeclaredCheck::Failed { artifact_sha256 } if sha256_id(artifact_sha256) => {
                failures.push(name.into())
            }
            _ => return Err(format!("invalid declared check: {name}")),
        }
    }
    let statistics = if input.pairs.is_empty() {
        missing.push("paired cases".into());
        None
    } else {
        let stats = paired_statistics(&input.pairs)?;
        if stats.n < 30 {
            missing.push("at least 30 independent paired cases".into());
        }
        if stats.interval_95.is_none() {
            missing.push("independent speakers, sessions, recordings and families".into());
        }
        if stats.n >= 30 && stats.interval_95.is_some() {
            if ((stats.candidate_only as i128 - stats.baseline_only as i128) * 20) < stats.n as i128
            {
                failures.push("accuracy improvement below five percentage points".into());
            }
            if stats.interval_95.is_some_and(|(lower, _)| lower <= 0.0) {
                failures.push("paired 95% confidence lower bound is not positive".into());
            }
        }
        let baseline_ends = input
            .pairs
            .iter()
            .filter(|pair| pair.baseline.normal_end)
            .count();
        let candidate_ends = input
            .pairs
            .iter()
            .filter(|pair| pair.candidate.normal_end)
            .count();
        if candidate_ends < baseline_ends {
            failures.push("normal-end rate regressed".into());
        }
        Some(stats)
    };
    Ok(DevelopmentReport {
        status: GateStatus::InsufficientEvidence,
        statistics,
        missing,
        failures,
    })
}

/// A caller's rule-based judgment and the model's termination status.
#[derive(Clone, Debug)]
pub struct PairedOutcome {
    pub correct: bool,
    pub normal_end: bool,
}

/// Both arms evaluated against the same original recording.
#[derive(Clone, Debug)]
pub struct PairedCase {
    pub case_id: String,
    pub original_id: String,
    pub family_id: String,
    pub speaker_id: String,
    pub session_id: String,
    pub baseline: PairedOutcome,
    pub candidate: PairedOutcome,
}

#[derive(Clone, Debug)]
pub struct PairedStatistics {
    pub n: usize,
    pub baseline_correct: usize,
    pub candidate_correct: usize,
    pub both_correct: usize,
    pub both_wrong: usize,
    pub candidate_only: usize,
    pub baseline_only: usize,
    pub baseline_normal_end: usize,
    pub candidate_normal_end: usize,
    pub improvement: f64,
    pub interval_95: Option<(f64, f64)>,
}

/// Calculate a conservative 95% interval using Bonferroni-adjusted exact
/// binomial intervals for candidate-only and baseline-only proportions.
pub fn paired_statistics(pairs: &[PairedCase]) -> Result<PairedStatistics, String> {
    if pairs.is_empty() {
        return Err("empty paired comparison".into());
    }
    let n = pairs.len();
    if n > 10_000 {
        return Err("too many paired cases".into());
    }
    let mut ids = HashSet::new();
    for pair in pairs {
        if [
            &pair.case_id,
            &pair.original_id,
            &pair.family_id,
            &pair.speaker_id,
            &pair.session_id,
        ]
        .iter()
        .any(|id| {
            id.trim().is_empty()
                || id.trim() != id.as_str()
                || id.len() > 256
                || id.chars().any(char::is_control)
        }) || !ids.insert(&pair.case_id)
        {
            return Err("invalid or duplicate paired case identity".into());
        }
    }
    let mut candidate_only = 0;
    let mut baseline_only = 0;
    let mut both_correct = 0;
    let mut both_wrong = 0;
    let mut baseline_normal_end = 0;
    let mut candidate_normal_end = 0;
    for pair in pairs {
        let baseline = pair.baseline.correct && pair.baseline.normal_end;
        let candidate = pair.candidate.correct && pair.candidate.normal_end;
        candidate_only += usize::from(candidate && !baseline);
        baseline_only += usize::from(baseline && !candidate);
        both_correct += usize::from(baseline && candidate);
        both_wrong += usize::from(!baseline && !candidate);
        baseline_normal_end += usize::from(pair.baseline.normal_end);
        candidate_normal_end += usize::from(pair.candidate.normal_end);
    }
    let independent = pairs
        .iter()
        .map(|pair| &pair.speaker_id)
        .collect::<HashSet<_>>()
        .len()
        == n
        && pairs
            .iter()
            .map(|pair| &pair.session_id)
            .collect::<HashSet<_>>()
            .len()
            == n
        && pairs
            .iter()
            .map(|pair| &pair.original_id)
            .collect::<HashSet<_>>()
            .len()
            == n
        && pairs
            .iter()
            .map(|pair| &pair.family_id)
            .collect::<HashSet<_>>()
            .len()
            == n;
    let interval_95 = independent.then(|| {
        let (candidate_lower, candidate_upper) = exact_binomial(n, candidate_only);
        let (baseline_lower, baseline_upper) = exact_binomial(n, baseline_only);
        (
            candidate_lower - baseline_upper,
            candidate_upper - baseline_lower,
        )
    });
    Ok(PairedStatistics {
        n,
        baseline_correct: both_correct + baseline_only,
        candidate_correct: both_correct + candidate_only,
        both_correct,
        both_wrong,
        candidate_only,
        baseline_only,
        baseline_normal_end,
        candidate_normal_end,
        improvement: (candidate_only as f64 - baseline_only as f64) / n as f64,
        interval_95,
    })
}

// Each two-sided interval has error 0.025; each individual tail is 0.0125.
fn exact_binomial(n: usize, successes: usize) -> (f64, f64) {
    const TAIL: f64 = 0.0125;
    let mut lower = 0.0;
    if successes > 0 {
        let (mut left, mut right) = (0.0, 1.0);
        for _ in 0..64 {
            let middle = (left + right) / 2.0;
            if binomial_tail(n, successes, middle, true) < TAIL {
                left = middle;
            } else {
                right = middle;
            }
        }
        lower = (left + right) / 2.0;
    }
    let mut upper = 1.0;
    if successes < n {
        let (mut left, mut right) = (0.0, 1.0);
        for _ in 0..64 {
            let middle = (left + right) / 2.0;
            if binomial_tail(n, successes, middle, false) > TAIL {
                left = middle;
            } else {
                right = middle;
            }
        }
        upper = (left + right) / 2.0;
    }
    (lower, upper)
}

fn binomial_tail(n: usize, k: usize, p: f64, above: bool) -> f64 {
    if p <= 0.0 {
        return if !above || k == 0 { 1.0 } else { 0.0 };
    }
    if p >= 1.0 {
        return if above || k == n { 1.0 } else { 0.0 };
    }
    let mut log_probability = n as f64 * (-p).ln_1p();
    let log_odds = p.ln() - (-p).ln_1p();
    let mut sum = 0.0;
    for i in 0..=n {
        if (above && i >= k) || (!above && i <= k) {
            sum += log_probability.exp();
        }
        if i < n {
            log_probability += ((n - i) as f64).ln() - ((i + 1) as f64).ln() + log_odds;
        }
    }
    sum.min(1.0)
}
