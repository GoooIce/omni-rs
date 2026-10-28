//! Calibrated verify-size ledger for DSpark (`MINICPM_DSPARK_LEDGER`,
//! default off).
//!
//! The AIMD [`crate::dspark::Ladder`] moves one rung on streaks; this ledger
//! instead measures acceptance and cost per allowed block size and picks the
//! size that maximizes expected accepted tokens minus the time those tokens
//! cost:
//!
//! ```text
//! n* = argmax_n  E(n) - rate * cost[n]
//! ```
//!
//! `E(n)` is the expected number of accepted tokens for a chain of `n` draft
//! tokens, bonus included. Before a size has measurements it comes from the
//! per-depth acceptance posterior `p_d = (accepts_d + 1) / (trials_d + 2)`:
//! `E(n) = 1 + sum_{d=1..n} prod_{k=1..d} p_k`. Once the size has untrimmed
//! samples, its accepted-token EWMA replaces the posterior estimate.
//! `cost[n]` is the verify phase's host wall clock EWMA in milliseconds and
//! `rate` the recent decode rate in tokens/ms, so the objective is in
//! token-equivalents.
//!
//! Cost basis: **host-wall-clock** (decision #65). This repository has no
//! command-buffer GPU timestamps; the verify timer covers the host call into
//! the verify forward, including async submission, not a device-side
//! execution time.
//!
//! Allowed sizes are the chain lengths `[3, 8, 16]`, the existing
//! `BLOCK_LADDER`. The set is deliberately finite because each verify shape
//! pays a one-time matmul/attention autotune that then replays from the
//! persistent cache; an unbounded integer search would keep paying for new
//! shapes. The ledger searches inside that set and never invents a shape.
//!
//! Exploration: every allowed size is sampled once in ascending order before
//! the argmax runs, then the least recently sampled non-argmax size is forced
//! every [`EXPLORE_PERIOD`] cycles so an unpicked size keeps a live estimate.
//!
//! Estimates can persist as JSON keyed by model + drafter + precision + block
//! policy + chip fingerprint. A missing, corrupt or differently keyed file
//! starts from fresh estimates; the ledger does not fail a run over storage.
//! Nothing is read or written while `MINICPM_DSPARK_LEDGER` is off.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The chain sizes the ledger may choose from: the adaptive ladder's finite
/// autotune shape set (3/8/16 draft tokens).
pub const ALLOWED_SIZES: [usize; 3] = crate::dspark::BLOCK_LADDER;

/// The cost basis label attached to every ledger decision (decision #65).
pub const COST_BASIS: &str = "host-wall-clock";

/// Weight of the newest sample in every EWMA.
pub const EWMA_ALPHA: f64 = 0.5;

/// Cycles between forced exploration samples.
pub const EXPLORE_PERIOD: u64 = 8;

/// One allowed size's running measurements.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SizeRecord {
    /// Untrimmed cycles observed at this size (the acceptance samples).
    pub samples: u64,
    /// EWMA of accepted draft tokens per untrimmed cycle.
    pub accepted: f64,
    /// Cycles whose verify cost was measured (trimmed cycles count here).
    pub cost_samples: u64,
    /// EWMA of verify rows per measured cycle.
    pub rows: f64,
    /// EWMA of verify-phase host milliseconds per measured cycle.
    pub verify_ms: f64,
    /// Cycle index of the most recent measurement.
    pub last_cycle: u64,
}

/// The ledger's persisted estimates.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LedgerState {
    /// Completed cycles.
    pub cycles: u64,
    /// Per-depth acceptance counts (`depth_accepts[d - 1]`).
    pub depth_accepts: Vec<u64>,
    /// Per-depth trial counts (`depth_trials[d - 1]`): a depth is tried when
    /// the chain reaches it, i.e. the previous depth was accepted.
    pub depth_trials: Vec<u64>,
    /// Per-size measurements, in allowed-set order.
    pub sizes: Vec<(usize, SizeRecord)>,
    /// EWMA decode rate in tokens per millisecond, bonus included.
    pub rate: f64,
    /// Samples behind `rate` (keeps the first sample exact).
    pub rate_samples: u64,
}

impl LedgerState {
    fn fresh(allowed: &[usize]) -> Self {
        Self {
            sizes: allowed
                .iter()
                .map(|&size| (size, SizeRecord::default()))
                .collect(),
            ..Self::default()
        }
    }
}

/// One cycle's observation handed to [`Ledger::observe`].
pub struct CycleObservation {
    /// The allowed chain length the cycle used.
    pub size: usize,
    /// Verify rows in the block, bonus included.
    pub rows: usize,
    /// Accepted draft tokens (bonus excluded).
    pub accepted: usize,
    /// Verify-phase host wall clock in milliseconds.
    pub verify_ms: f64,
    /// Whole-cycle host wall clock in milliseconds.
    pub cycle_ms: f64,
    /// The cycle was trimmed by the emission room; acceptance is not a
    /// layout signal then (cost still is).
    pub trimmed: bool,
}

/// Persistence key: model + drafter + precision + block policy + chip.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerKey {
    pub model: String,
    pub drafter: String,
    pub precision: String,
    /// `adaptive` or `fixed:<n>`.
    pub block: String,
    /// Chip fingerprint, `MINICPM_DSPARK_CHIP` or target arch/OS.
    pub chip: String,
}

impl LedgerKey {
    pub fn new(
        model: impl Into<String>,
        drafter: impl Into<String>,
        precision: impl Into<String>,
        block: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            drafter: drafter.into(),
            precision: precision.into(),
            block: block.into(),
            chip: chip_fingerprint(),
        }
    }

    /// A short stable file stem for this key.
    pub fn fingerprint(&self) -> String {
        let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
        for part in [
            self.model.as_str(),
            self.drafter.as_str(),
            self.precision.as_str(),
            self.block.as_str(),
            self.chip.as_str(),
        ] {
            sha2::Digest::update(&mut hasher, part.as_bytes());
            sha2::Digest::update(&mut hasher, [0u8]);
        }
        let digest = sha2::Digest::finalize(hasher);
        digest[..16]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

fn chip_fingerprint() -> String {
    std::env::var("MINICPM_DSPARK_CHIP")
        .unwrap_or_else(|_| format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS))
}

/// The checkpoint identity a run's ledger is keyed under. Only read when
/// `MINICPM_DSPARK_LEDGER=1`.
pub struct RunIdentity<'a> {
    pub model: &'a str,
    pub drafter: &'a str,
    pub precision: &'a str,
}

/// Measured estimates plus the size rule over one allowed set.
pub struct Ledger {
    allowed: Vec<usize>,
    state: LedgerState,
    key: LedgerKey,
    path: PathBuf,
    /// Size returned by [`Ledger::choose`] and awaiting its observation.
    pending: Option<usize>,
}

/// Format version of the persisted estimate file.
const STATE_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct Persisted {
    version: u32,
    key: LedgerKey,
    state: LedgerState,
}

impl Ledger {
    /// A fresh in-memory ledger (no persisted estimates).
    pub fn new(allowed: Vec<usize>) -> Self {
        Self {
            state: LedgerState::fresh(&allowed),
            allowed,
            key: LedgerKey::default(),
            path: PathBuf::new(),
            pending: None,
        }
    }

    /// Load persisted estimates for `key`; missing/corrupt files fall back to
    /// fresh estimates.
    pub fn load(allowed: Vec<usize>, key: LedgerKey) -> Self {
        let path = ledger_path(&key);
        Self::load_path(&path, allowed, key)
    }

    fn load_path(path: &Path, allowed: Vec<usize>, key: LedgerKey) -> Self {
        let mut ledger = Self {
            state: LedgerState::fresh(&allowed),
            allowed,
            key: key.clone(),
            path: path.to_path_buf(),
            pending: None,
        };
        let Ok(text) = std::fs::read_to_string(path) else {
            return ledger;
        };
        match serde_json::from_str::<Persisted>(&text) {
            Ok(persisted) if persisted.version == STATE_VERSION && persisted.key == key => {
                ledger.state = reconciled(persisted.state, &ledger.allowed);
            }
            Ok(_) => {}
            Err(error) => eprintln!("dspark ledger: ignoring {}: {error}", path.display()),
        }
        ledger
    }

    fn save_path(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        let persisted = Persisted {
            version: STATE_VERSION,
            key: self.key.clone(),
            state: self.state.clone(),
        };
        let text = serde_json::to_string_pretty(&persisted).map_err(std::io::Error::other)?;
        std::fs::write(path, text)
    }

    pub fn allowed(&self) -> &[usize] {
        &self.allowed
    }

    pub fn cycles(&self) -> u64 {
        self.state.cycles
    }

    pub fn rate(&self) -> f64 {
        self.state.rate
    }

    pub fn stats(&self, size: usize) -> Option<&SizeRecord> {
        self.state
            .sizes
            .iter()
            .find(|(allowed, _)| *allowed == size)
            .map(|(_, record)| record)
    }

    /// The next cycle's chain length.
    pub fn choose(&mut self) -> usize {
        let index = if let Some(index) = self
            .state
            .sizes
            .iter()
            .position(|(_, record)| record.samples == 0)
        {
            // The initial sweep: every allowed size is measured once, in
            // ascending order, before the argmax may repeat a size.
            index
        } else if self.state.cycles > 0 && self.state.cycles % EXPLORE_PERIOD == 0 {
            self.explore_index()
        } else {
            self.argmax_index()
        };
        self.pending = Some(index);
        self.allowed[index]
    }

    /// The allowed-set index maximizing `E(n) - rate * cost[n]`; ties keep
    /// the smaller size.
    fn argmax_index(&self) -> usize {
        let mut best = 0;
        for index in 1..self.allowed.len() {
            if self.score(self.allowed[index]) > self.score(self.allowed[best]) {
                best = index;
            }
        }
        best
    }

    /// The least recently measured size other than the argmax (the argmax
    /// itself when it is the only allowed size).
    fn explore_index(&self) -> usize {
        let argmax = self.argmax_index();
        let mut best = argmax;
        let mut best_cycle = u64::MAX;
        for index in 0..self.allowed.len() {
            if self.allowed.len() > 1 && index == argmax {
                continue;
            }
            let last = self.state.sizes[index].1.last_cycle;
            if last < best_cycle || (last == best_cycle && index < best) {
                best = index;
                best_cycle = last;
            }
        }
        best
    }

    /// Fold one completed cycle in.
    pub fn observe(&mut self, observation: CycleObservation) {
        let cycle = self.state.cycles;
        let index = self
            .state
            .sizes
            .iter()
            .position(|(size, _)| *size == observation.size)
            .expect("ledger observed a size outside the allowed set");
        if let Some(pending) = self.pending.take() {
            debug_assert_eq!(
                pending, index,
                "observed size does not match the layout choose() returned"
            );
        }
        let record = &mut self.state.sizes[index].1;
        record.rows = ewma(record.rows, observation.rows as f64, record.cost_samples);
        record.verify_ms = ewma(record.verify_ms, observation.verify_ms, record.cost_samples);
        record.cost_samples += 1;
        record.last_cycle = cycle;
        if !observation.trimmed {
            record.accepted = ewma(record.accepted, observation.accepted as f64, record.samples);
            record.samples += 1;
        }
        if !observation.trimmed && observation.cycle_ms > 0.0 {
            // A cycle contributes its emitted tokens, bonus included, over
            // its whole host wall clock.
            let sample = (observation.accepted as f64 + 1.0) / observation.cycle_ms;
            self.state.rate = ewma(self.state.rate, sample, self.state.rate_samples);
            self.state.rate_samples += 1;
        }
        if !observation.trimmed {
            if self.state.depth_trials.len() < observation.size {
                self.state.depth_trials.resize(observation.size, 0);
                self.state.depth_accepts.resize(observation.size, 0);
            }
            for depth in 1..=observation.size {
                if observation.accepted + 1 >= depth {
                    self.state.depth_trials[depth - 1] += 1;
                }
                if observation.accepted >= depth {
                    self.state.depth_accepts[depth - 1] += 1;
                }
            }
        }
        self.state.cycles += 1;
    }

    /// Laplace/Beta posterior acceptance probability at `depth` (1-based).
    pub fn depth_probability(&self, depth: usize) -> f64 {
        let accepts = self
            .state
            .depth_accepts
            .get(depth - 1)
            .copied()
            .unwrap_or(0) as f64;
        let trials = self.state.depth_trials.get(depth - 1).copied().unwrap_or(0) as f64;
        (accepts + 1.0) / (trials + 2.0)
    }

    /// Expected accepted tokens for a chain of `size`, bonus included.
    pub fn expected(&self, size: usize) -> f64 {
        if let Some(record) = self.stats(size).filter(|record| record.samples > 0) {
            return 1.0 + record.accepted;
        }
        let mut total = 1.0;
        let mut prefix = 1.0;
        for depth in 1..=size {
            prefix *= self.depth_probability(depth);
            total += prefix;
        }
        total
    }

    /// `E(size) - rate * cost[size]`, in token-equivalents.
    pub fn score(&self, size: usize) -> f64 {
        self.expected(size) - self.state.rate * self.cost(size)
    }

    fn cost(&self, size: usize) -> f64 {
        self.stats(size)
            .filter(|record| record.cost_samples > 0)
            .map(|record| record.verify_ms)
            .unwrap_or(0.0)
    }

    /// Persist the estimates; storage failures are reported and ignored.
    pub fn save(&self) {
        if self.path.as_os_str().is_empty() {
            return;
        }
        if let Err(error) = self.save_path(&self.path) {
            eprintln!(
                "dspark ledger: cannot write {}: {error}",
                self.path.display()
            );
        }
    }
}

/// Keep only the allowed sizes' records, in allowed-set order.
fn reconciled(state: LedgerState, allowed: &[usize]) -> LedgerState {
    let sizes = allowed
        .iter()
        .map(|&size| {
            let record = state
                .sizes
                .iter()
                .find(|(stored, _)| *stored == size)
                .map(|(_, record)| record.clone())
                .unwrap_or_default();
            (size, record)
        })
        .collect();
    LedgerState { sizes, ..state }
}

/// Exponentially weighted mean; the first sample is taken as-is.
fn ewma(previous: f64, sample: f64, samples: u64) -> f64 {
    if samples == 0 {
        sample
    } else {
        EWMA_ALPHA * sample + (1.0 - EWMA_ALPHA) * previous
    }
}

fn ledger_path(key: &LedgerKey) -> PathBuf {
    if let Some(path) = std::env::var_os("MINICPM_DSPARK_LEDGER_PATH") {
        return PathBuf::from(path);
    }
    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok().map(|dir| dir.join("target")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("dspark-ledger")
        .join(format!("{}.json", key.fingerprint()))
}

#[cfg(test)]
mod tests {
    use super::{CycleObservation, EXPLORE_PERIOD, Ledger, LedgerKey};
    use std::path::PathBuf;

    fn temp_path() -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "omni-dspark-ledger-{}-{nanos}.json",
            std::process::id()
        ))
    }

    fn observation(size: usize, accepted: usize) -> CycleObservation {
        cycle(size, accepted, 10.0, 100.0)
    }

    fn cycle(size: usize, accepted: usize, verify_ms: f64, cycle_ms: f64) -> CycleObservation {
        CycleObservation {
            size,
            rows: size + 1,
            accepted,
            verify_ms,
            cycle_ms,
            trimmed: false,
        }
    }

    #[test]
    fn expected_acceptance_follows_the_chain_formula() {
        // No observations: Beta(1,1) makes every depth 1/2, so the chain
        // expectation is 1 + 1/2 + 1/4 + 1/8 at size 3.
        let ledger = Ledger::new(vec![3, 8, 16]);
        assert!((ledger.expected(1) - 1.5).abs() < 1e-12);
        assert!((ledger.expected(2) - 1.75).abs() < 1e-12);
        assert!((ledger.expected(3) - 1.875).abs() < 1e-12);

        // A full accept at size 8 measures size 8 itself, and its depth
        // counts feed the formula for any size that has no measurements yet.
        let mut ledger = Ledger::new(vec![3, 8, 16]);
        ledger.observe(observation(8, 8));
        assert!((ledger.depth_probability(1) - 2.0 / 3.0).abs() < 1e-12);
        assert!((ledger.depth_probability(9) - 0.5).abs() < 1e-12);
        let p = 2.0 / 3.0;
        let size_four = 1.0 + p + p * p + p * p * p + p * p * p * p;
        assert!((ledger.expected(4) - size_four).abs() < 1e-12);
        // A measured size uses its accepted EWMA instead: 1 bonus + 8.
        assert!((ledger.expected(8) - 9.0).abs() < 1e-12);
    }

    #[test]
    fn observations_feed_ewma_estimates() {
        let mut ledger = Ledger::new(vec![3]);
        ledger.observe(CycleObservation {
            size: 3,
            rows: 4,
            accepted: 1,
            verify_ms: 10.0,
            cycle_ms: 100.0,
            trimmed: false,
        });
        // The first sample is taken as-is: accepted 1, verify 10 ms, and
        // (1 accepted + 1 bonus) / 100 ms = 0.02 tokens/ms.
        let record = ledger.stats(3).unwrap();
        assert_eq!(record.samples, 1);
        assert!((record.accepted - 1.0).abs() < 1e-12);
        assert!((record.rows - 4.0).abs() < 1e-12);
        assert!((record.verify_ms - 10.0).abs() < 1e-12);
        assert!((ledger.rate() - 0.02).abs() < 1e-12);

        ledger.observe(CycleObservation {
            size: 3,
            rows: 4,
            accepted: 3,
            verify_ms: 20.0,
            cycle_ms: 100.0,
            trimmed: false,
        });
        // Alpha 0.5: accepted (1 + 3) / 2 = 2, verify (10 + 20) / 2 = 15,
        // rate (0.02 + 0.04) / 2 = 0.03.
        let record = ledger.stats(3).unwrap();
        assert!((record.accepted - 2.0).abs() < 1e-12);
        assert!((record.verify_ms - 15.0).abs() < 1e-12);
        assert!((ledger.rate() - 0.03).abs() < 1e-12);

        ledger.observe(CycleObservation {
            size: 3,
            rows: 4,
            accepted: 3,
            verify_ms: 30.0,
            cycle_ms: 50.0,
            trimmed: false,
        });
        // accepted (2 + 3) / 2 = 2.5, verify (15 + 30) / 2 = 22.5, and the
        // rate sample is 4 tokens / 50 ms = 0.08 -> 0.055.
        let record = ledger.stats(3).unwrap();
        assert!((record.accepted - 2.5).abs() < 1e-12);
        assert!((record.verify_ms - 22.5).abs() < 1e-12);
        assert!((ledger.rate() - 0.055).abs() < 1e-12);
        assert_eq!(ledger.cycles(), 3);

        // A trimmed cycle carries no acceptance signal, but its cost counts.
        ledger.observe(CycleObservation {
            size: 3,
            rows: 4,
            accepted: 0,
            verify_ms: 25.0,
            cycle_ms: 50.0,
            trimmed: true,
        });
        let record = ledger.stats(3).unwrap();
        assert_eq!(record.samples, 3);
        assert!((record.accepted - 2.5).abs() < 1e-12);
        assert!((record.verify_ms - 23.75).abs() < 1e-12);
        assert!((ledger.rate() - 0.055).abs() < 1e-12);
        assert_eq!(ledger.cycles(), 4);
    }

    #[test]
    fn size_rule_maximizes_expected_tokens_minus_cost() {
        // One sweep over all three sizes. E(n) = 1 + measured accepted:
        // 4, 7 and 8 tokens. The rate EWMA lands at (0.04 + 0.07 + 0.08) / 2
        // blended twice = 0.0675 tokens/ms.
        let mut cheap_verify = Ledger::new(vec![3, 8, 16]);
        cheap_verify.observe(cycle(3, 3, 5.0, 100.0));
        cheap_verify.observe(cycle(8, 6, 10.0, 100.0));
        cheap_verify.observe(cycle(16, 7, 40.0, 100.0));
        assert!((cheap_verify.rate() - 0.0675).abs() < 1e-12);
        // 4 - 0.3375 = 3.6625, 7 - 0.675 = 6.325, 8 - 2.7 = 5.3.
        assert_eq!(cheap_verify.choose(), 8);

        // Same acceptance, verify costs an order of magnitude apart: the
        // same rule keeps the cheap size.
        let mut dear_verify = Ledger::new(vec![3, 8, 16]);
        dear_verify.observe(cycle(3, 3, 5.0, 100.0));
        dear_verify.observe(cycle(8, 6, 1000.0, 100.0));
        dear_verify.observe(cycle(16, 7, 4000.0, 100.0));
        assert_eq!(dear_verify.choose(), 3);
    }

    #[test]
    fn exploration_samples_every_allowed_size() {
        let mut ledger = Ledger::new(vec![3, 8, 16]);
        let mut sweep = Vec::new();
        for _ in 0..3 {
            let size = ledger.choose();
            sweep.push(size);
            ledger.observe(cycle(size, size, 8.0, 100.0));
        }
        // The fresh ledger ramps conservatively through the whole set.
        assert_eq!(sweep, vec![3, 8, 16]);
        for size in [3, 8, 16] {
            assert_eq!(ledger.stats(size).unwrap().samples, 1);
        }

        // Full accepts make the largest size the argmax; the periodic forced
        // sample must still revisit a non-argmax size and refresh it.
        let mut choices = Vec::new();
        for _ in 0..EXPLORE_PERIOD as usize + 1 {
            let size = ledger.choose();
            choices.push(size);
            ledger.observe(cycle(size, size, 8.0, 100.0));
        }
        assert!(
            choices.iter().any(|&size| size != 16),
            "no exploration sample over {choices:?}"
        );
        assert_eq!(ledger.stats(3).unwrap().samples, 2);
    }

    #[test]
    fn persisted_estimates_round_trip() {
        let path = temp_path();
        let key = LedgerKey::new("model-a", "draft-a", "f16", "adaptive");
        let mut ledger = Ledger::load_path(&path, vec![3, 8, 16], key.clone());
        // Missing file: fresh estimates.
        assert_eq!(ledger.cycles(), 0);
        ledger.observe(cycle(3, 2, 5.0, 100.0));
        ledger.observe(cycle(8, 6, 30.0, 100.0));
        let expected_three = ledger.expected(3);
        let rate = ledger.rate();
        let record_eight = ledger.stats(8).unwrap().clone();
        ledger.save_path(&path).unwrap();

        let loaded = Ledger::load_path(&path, vec![3, 8, 16], key.clone());
        assert_eq!(loaded.cycles(), 2);
        assert!((loaded.expected(3) - expected_three).abs() < 1e-12);
        assert!((loaded.rate() - rate).abs() < 1e-12);
        let loaded_record = loaded.stats(8).unwrap();
        assert_eq!(loaded_record.samples, record_eight.samples);
        assert!((loaded_record.accepted - record_eight.accepted).abs() < 1e-12);
        assert!((loaded_record.verify_ms - record_eight.verify_ms).abs() < 1e-12);

        // A different checkpoint set must not inherit these estimates.
        let other = Ledger::load_path(
            &path,
            vec![3, 8, 16],
            LedgerKey::new("model-b", "draft-a", "f16", "adaptive"),
        );
        assert_eq!(other.cycles(), 0);
        assert!((other.rate() - 0.0).abs() < 1e-12);

        // A corrupt file is non-fatal.
        std::fs::write(&path, "{ not json").unwrap();
        let corrupt = Ledger::load_path(&path, vec![3, 8, 16], key);
        assert_eq!(corrupt.cycles(), 0);
        let _ = std::fs::remove_file(&path);
    }
}
