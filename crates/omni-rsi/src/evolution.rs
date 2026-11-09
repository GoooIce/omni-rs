//! Verified chart rounds, counterfactual supervision and reader-head updates.
use crate::{
    chart::{ChartProgram, ChartSpec, Question, Split, render, verify},
    model::FrozenReader,
    reader::{Example, ReaderHead},
    storage::{Evaluation, Metrics, Package, Store, atomic_write, digest},
};
use burn::tensor::backend::Backend;
use image::{DynamicImage, ImageFormat, RgbImage};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use std::{
    io::Cursor,
    sync::atomic::{AtomicBool, Ordering},
};

pub trait Oracle {
    fn write_program(
        &self,
        spec: ChartSpec,
        stop: &AtomicBool,
    ) -> Result<Option<ChartProgram>, String>;
    fn features(
        &self,
        image: &RgbImage,
        question: Question,
        label: bool,
        stop: &AtomicBool,
    ) -> Result<Example, String>;
    fn verify_source(&self, _fingerprint: &str, _stop: &AtomicBool) -> Result<(), String> {
        Ok(())
    }
}
impl<B: Backend> Oracle for FrozenReader<B> {
    fn write_program(
        &self,
        spec: ChartSpec,
        stop: &AtomicBool,
    ) -> Result<Option<ChartProgram>, String> {
        FrozenReader::write_program(self, spec, stop)
    }
    fn features(
        &self,
        image: &RgbImage,
        question: Question,
        label: bool,
        stop: &AtomicBool,
    ) -> Result<Example, String> {
        FrozenReader::features(self, image, question, label, stop)
    }
    fn verify_source(&self, fingerprint: &str, stop: &AtomicBool) -> Result<(), String> {
        FrozenReader::verify_source(self, fingerprint, stop)
    }
}

#[derive(Clone, Debug)]
pub struct RoundOptions {
    pub families: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub seed: u64,
}
impl Default for RoundOptions {
    fn default() -> Self {
        Self {
            families: 6,
            epochs: 100,
            learning_rate: 0.05,
            seed: 261003002,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilyStats {
    checks: u32,
    errors: u32,
    writer_attempts: u32,
    writer_rejects: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Progress {
    format_version: u32,
    model_fingerprint: String,
    round: u64,
    candidate: String,
    stats: Vec<FamilyStats>,
    replay: Vec<ReplaySample>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifiedSample {
    pub spec: ChartSpec,
    pub program: ChartProgram,
    pub image_sha256: String,
    pub counterfactual: bool,
    pub label: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplaySample {
    example: Example,
    sample: VerifiedSample,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoundReport {
    pub round: u64,
    pub candidate: String,
    pub promoted: bool,
    /// False means the process ended before the promotion decision was recorded.
    pub promotion_finalized: bool,
    pub writer_attempts: usize,
    pub writer_rejected: usize,
    pub train_examples: usize,
    pub new_examples: usize,
    pub baseline: Evaluation,
    pub dev: Evaluation,
    pub test: Evaluation,
    pub samples: Vec<VerifiedSample>,
}

fn stop_if_requested(stop: &AtomicBool) -> Result<(), String> {
    if stop.load(Ordering::Acquire) {
        Err("interrupted".into())
    } else {
        Ok(())
    }
}

fn specs(split: Split) -> Vec<ChartSpec> {
    let mut result = Vec::new();
    for left in 1..=9 {
        for right in left + 1..=9 {
            let spec = ChartSpec::new(left, right).expect("valid family");
            if spec.split() == split {
                result.push(spec);
            }
        }
    }
    result
}

fn select_families(stats: &[FamilyStats], options: &RoundOptions, round: u64) -> Vec<ChartSpec> {
    let mut rng = StdRng::seed_from_u64(options.seed ^ round);
    let mut pool = specs(Split::Train);
    pool.shuffle(&mut rng);
    let count = options.families.min(pool.len());
    let explore = count.div_ceil(5).max(1);
    let mut selected = Vec::with_capacity(count);
    for _ in 0..explore.min(count) {
        selected.push(pool.pop().expect("training pool"));
    }
    while selected.len() < count {
        let weights: Vec<f64> = pool
            .iter()
            .map(|spec| {
                let item = &stats[spec.family()];
                (1.0 + 2.0 * (f64::from(item.errors) + 1.0) / (f64::from(item.checks) + 2.0))
                    .min(3.0)
            })
            .collect();
        let mut draw = rng.random::<f64>() * weights.iter().sum::<f64>();
        let mut index = weights.len() - 1;
        for (i, weight) in weights.iter().enumerate() {
            if draw < *weight {
                index = i;
                break;
            }
            draw -= weight;
        }
        selected.push(pool.swap_remove(index));
    }
    selected
}

fn verified_example<O: Oracle>(
    oracle: &O,
    spec: ChartSpec,
    program: ChartProgram,
    counterfactual: bool,
    stop: &AtomicBool,
) -> Result<(Example, VerifiedSample), String> {
    stop_if_requested(stop)?;
    let image = render(&program)?;
    verify(&spec, &program, &image)?;
    let mut png = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(image.clone())
        .write_to(&mut png, ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    let example = oracle.features(&image, spec.question, spec.label(), stop)?;
    if example.label != spec.label() {
        return Err("reader label disagrees with verified chart".into());
    }
    Ok((
        example,
        VerifiedSample {
            spec,
            program,
            image_sha256: digest(&png.into_inner()),
            counterfactual,
            label: spec.label(),
        },
    ))
}

fn evaluation_set<O: Oracle>(
    oracle: &O,
    split: Split,
    stop: &AtomicBool,
) -> Result<Vec<Example>, String> {
    let mut result = Vec::new();
    for spec in specs(split) {
        for question in [Question::Taller, Question::Twice] {
            for current in [
                spec.with_question(question),
                ChartSpec::new(spec.right, spec.left)?.with_question(question),
            ] {
                let program = ChartProgram {
                    left: current.left,
                    right: current.right,
                };
                result.push(verified_example(oracle, current, program, false, stop)?.0);
            }
        }
    }
    Ok(result)
}

fn evaluate(head: &ReaderHead, examples: &[Example]) -> Result<Evaluation, String> {
    let empty = || Metrics {
        yes_total: 0,
        no_total: 0,
        yes_correct: 0,
        no_correct: 0,
    };
    let mut combined = empty();
    let mut by_question = [empty(), empty()];
    for (index, example) in examples.iter().enumerate() {
        let correct = (head.probability(example)? >= 0.5) == example.label;
        for metrics in [&mut combined, &mut by_question[(index % 4) / 2]] {
            if example.label {
                metrics.yes_total += 1;
                metrics.yes_correct += usize::from(correct);
            } else {
                metrics.no_total += 1;
                metrics.no_correct += usize::from(correct);
            }
        }
    }
    combined.validate()?;
    for task in &by_question {
        task.validate()?;
    }
    Ok(Evaluation {
        combined,
        by_question,
    })
}

fn checkpoint_digest(stats: &[FamilyStats], replay: &[ReplaySample]) -> Result<String, String> {
    Ok(digest(
        &serde_json::to_vec(&(stats, replay)).map_err(|e| e.to_string())?,
    ))
}

/// A round is published only after verified data, training, and both evaluations
/// complete. Interruption leaves the previous progress and active head intact.
pub fn run_round<O: Oracle>(
    oracle: &O,
    store: &Store,
    fingerprint: &str,
    options: &RoundOptions,
    stop: &AtomicBool,
) -> Result<RoundReport, String> {
    if fingerprint.is_empty()
        || options.families == 0
        || options.families > 24
        || options.epochs == 0
        || options.epochs > 1_000
        || !options.learning_rate.is_finite()
        || options.learning_rate <= 0.0
        || options.learning_rate > 1.0
    {
        return Err("invalid evolution options".into());
    }
    stop_if_requested(stop)?;
    let progress_path = store.root().join("progress.json");
    let progress: Option<Progress> = if progress_path.exists() {
        Some(
            serde_json::from_slice(&std::fs::read(&progress_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    if let Some(progress) = &progress {
        if progress.format_version != 1
            || progress.model_fingerprint != fingerprint
            || progress.stats.len() != 36
            || progress.replay.len() > 256
            || progress
                .stats
                .iter()
                .any(|item| item.errors > item.checks || item.writer_rejects > item.writer_attempts)
        {
            return Err("invalid or incompatible evolution progress".into());
        }
        let previous = store.load_candidate(&progress.candidate, fingerprint)?;
        if previous.round != progress.round {
            return Err("evolution progress does not match candidate".into());
        }
        if checkpoint_digest(&progress.stats, &progress.replay)? != previous.checkpoint_digest {
            return Err("evolution checkpoint digest mismatch".into());
        }
        for item in &progress.replay {
            if item.sample.spec.split() != Split::Train
                || item.example.label != item.sample.label
                || item.sample.label != item.sample.spec.label()
            {
                return Err("invalid replay label or split".into());
            }
            let image = render(&item.sample.program)?;
            verify(&item.sample.spec, &item.sample.program, &image)?;
            let mut png = Cursor::new(Vec::new());
            DynamicImage::ImageRgb8(image)
                .write_to(&mut png, ImageFormat::Png)
                .map_err(|e| e.to_string())?;
            if digest(&png.into_inner()) != item.sample.image_sha256 {
                return Err("replay render hash mismatch".into());
            }
            previous.head.probability(&item.example)?;
        }
        // Resuming must not silently promote a candidate: the user may have
        // deliberately rolled back after this checkpoint was written.
    }
    let round = progress.as_ref().map_or(1, |p| p.round.saturating_add(1));
    if round == u64::MAX {
        return Err("evolution round counter overflow".into());
    }
    let mut stats = progress
        .as_ref()
        .map_or_else(|| vec![FamilyStats::default(); 36], |p| p.stats.clone());
    let selected = select_families(&stats, options, round);
    let mut train = Vec::new();
    let mut samples = Vec::new();
    let mut rejected = 0;
    for spec in &selected {
        stop_if_requested(stop)?;
        let stat = &mut stats[spec.family()];
        stat.writer_attempts = stat.writer_attempts.saturating_add(1);
        let Some(program) = oracle.write_program(*spec, stop)? else {
            rejected += 1;
            stat.writer_rejects = stat.writer_rejects.saturating_add(1);
            continue;
        };
        let image = match render(&program) {
            Ok(image) => image,
            Err(_) => {
                rejected += 1;
                stat.writer_rejects = stat.writer_rejects.saturating_add(1);
                continue;
            }
        };
        if verify(spec, &program, &image).is_err() {
            rejected += 1;
            stat.writer_rejects = stat.writer_rejects.saturating_add(1);
            continue;
        }
        for question in [Question::Taller, Question::Twice] {
            let (example, record) =
                verified_example(oracle, spec.with_question(question), program, false, stop)?;
            train.push(example);
            samples.push(record);
            let opposite = ChartSpec::new(spec.right, spec.left)?.with_question(question);
            let mirror = ChartProgram {
                left: program.right,
                right: program.left,
            };
            let (example, record) = verified_example(oracle, opposite, mirror, true, stop)?;
            train.push(example);
            samples.push(record);
        }
    }
    if train.len() < 2 {
        return Err("no verified program pairs available for training".into());
    }
    let width = train[0].features.len();
    let baseline_head = ReaderHead::new(width)?;
    let previous = match store.load_active(fingerprint)? {
        Some(active) => Some(active),
        None => progress
            .as_ref()
            .map(|p| store.load_candidate(&p.candidate, fingerprint))
            .transpose()?,
    };
    let mut head = previous.map_or_else(|| baseline_head.clone(), |package| package.head);
    if head.weights.len() != width {
        return Err("reader feature width changed".into());
    }
    for (example, sample) in train.iter().zip(&samples) {
        let stat = &mut stats[sample.spec.family()];
        stat.checks = stat.checks.saturating_add(1);
        if (head.probability(example)? >= 0.5) != example.label {
            stat.errors = stat.errors.saturating_add(1);
        }
    }
    let new_examples = train.len();
    let mut replay = progress
        .as_ref()
        .map_or_else(Vec::new, |p| p.replay.clone());
    for (example, sample) in train.into_iter().zip(samples.iter().cloned()) {
        if let Some(existing) = replay
            .iter_mut()
            .find(|item| item.sample.spec == sample.spec)
        {
            *existing = ReplaySample { example, sample };
        } else {
            replay.push(ReplaySample { example, sample });
        }
    }
    if replay.len() > 256 {
        return Err("verified replay limit exceeded".into());
    }
    let train: Vec<_> = replay.iter().map(|item| item.example.clone()).collect();
    for _ in 0..options.epochs {
        stop_if_requested(stop)?;
        head.train_epoch(&train, options.learning_rate, 0.0001)?;
    }
    let dev_set = evaluation_set(oracle, Split::Dev, stop)?;
    let test_set = evaluation_set(oracle, Split::Test, stop)?;
    let baseline = evaluate(&baseline_head, &dev_set)?;
    let dev = evaluate(&head, &dev_set)?;
    let test = evaluate(&head, &test_set)?;
    stop_if_requested(stop)?;
    oracle.verify_source(fingerprint, stop)?;
    stop_if_requested(stop)?;
    let package = Package::new(
        fingerprint.into(),
        round,
        head,
        dev.clone(),
        baseline.clone(),
        checkpoint_digest(&stats, &replay)?,
    )?;
    let candidate = store.stage(&package)?;
    stop_if_requested(stop)?;
    let mut report = RoundReport {
        round,
        candidate: candidate.clone(),
        promoted: false,
        promotion_finalized: false,
        writer_attempts: selected.len(),
        writer_rejected: rejected,
        train_examples: train.len(),
        new_examples,
        baseline,
        dev,
        test,
        samples,
    };
    let report_path = store
        .root()
        .join("rounds")
        .join(format!("{candidate}.json"));
    atomic_write(
        &report_path,
        &serde_json::to_vec(&report).map_err(|e| e.to_string())?,
    )?;
    stop_if_requested(stop)?;
    let progress = Progress {
        format_version: 1,
        model_fingerprint: fingerprint.into(),
        round,
        candidate: candidate.clone(),
        stats,
        replay,
    };
    atomic_write(
        &progress_path,
        &serde_json::to_vec(&progress).map_err(|e| e.to_string())?,
    )?;
    stop_if_requested(stop)?;
    report.promoted = store.promote(&candidate, fingerprint)?;
    report.promotion_finalized = true;
    atomic_write(
        &report_path,
        &serde_json::to_vec(&report).map_err(|e| e.to_string())?,
    )?;
    Ok(report)
}
