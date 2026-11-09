//! Immutable candidate packages and validated atomic promotion.
use crate::reader::ReaderHead;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

pub const QUESTION_VERSION: &str = "two-bar-comparisons-v2";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    pub yes_total: usize,
    pub no_total: usize,
    pub yes_correct: usize,
    pub no_correct: usize,
}
impl Metrics {
    pub fn validate(&self) -> Result<(), String> {
        if self.yes_total == 0
            || self.no_total == 0
            || self.yes_correct > self.yes_total
            || self.no_correct > self.no_total
            || self.yes_total.checked_add(self.no_total).is_none()
        {
            return Err("invalid split metrics".into());
        }
        Ok(())
    }
    pub fn correct(&self) -> usize {
        self.yes_correct + self.no_correct
    }
    pub fn total(&self) -> usize {
        self.yes_total + self.no_total
    }
    pub fn improves_on(&self, baseline: &Self) -> Result<bool, String> {
        self.validate()?;
        baseline.validate()?;
        if (self.yes_total, self.no_total) != (baseline.yes_total, baseline.no_total) {
            return Err("reader evaluations use different dev splits".into());
        }
        let minimum = self.total().div_ceil(20).max(1);
        Ok(self.correct() >= baseline.correct().saturating_add(minimum)
            && self.yes_correct.saturating_add(1) >= baseline.yes_correct
            && self.no_correct.saturating_add(1) >= baseline.no_correct)
    }

    fn does_not_regress_on(&self, baseline: &Self) -> Result<bool, String> {
        self.validate()?;
        baseline.validate()?;
        if (self.yes_total, self.no_total) != (baseline.yes_total, baseline.no_total) {
            return Err("reader evaluations use different task splits".into());
        }
        Ok(self.correct() >= baseline.correct())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evaluation {
    pub combined: Metrics,
    /// Taller, then twice-as-tall; each holds both chart orientations.
    pub by_question: [Metrics; 2],
}
impl Evaluation {
    fn validate(&self) -> Result<(), String> {
        self.combined.validate()?;
        for task in &self.by_question {
            task.validate()?;
        }
        let sum = |get: fn(&Metrics) -> usize| -> Option<usize> {
            get(&self.by_question[0]).checked_add(get(&self.by_question[1]))
        };
        if sum(|m| m.yes_total) != Some(self.combined.yes_total)
            || sum(|m| m.no_total) != Some(self.combined.no_total)
            || sum(|m| m.yes_correct) != Some(self.combined.yes_correct)
            || sum(|m| m.no_correct) != Some(self.combined.no_correct)
        {
            return Err("reader task metrics do not match combined metrics".into());
        }
        Ok(())
    }

    fn improves_on(&self, baseline: &Self) -> Result<bool, String> {
        self.validate()?;
        baseline.validate()?;
        if !self.combined.improves_on(&baseline.combined)? {
            return Ok(false);
        }
        for (candidate, reference) in self.by_question.iter().zip(&baseline.by_question) {
            if !candidate.does_not_regress_on(reference)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Package {
    pub format_version: u32,
    pub model_fingerprint: String,
    pub question_version: String,
    pub round: u64,
    pub head: ReaderHead,
    pub dev: Evaluation,
    pub baseline: Evaluation,
    /// Digest of the replay and sampling state used for this candidate.
    pub checkpoint_digest: String,
}
impl Package {
    pub fn new(
        model_fingerprint: String,
        round: u64,
        head: ReaderHead,
        dev: Evaluation,
        baseline: Evaluation,
        checkpoint_digest: String,
    ) -> Result<Self, String> {
        let package = Self {
            format_version: 2,
            model_fingerprint,
            question_version: QUESTION_VERSION.into(),
            round,
            head,
            dev,
            baseline,
            checkpoint_digest,
        };
        package.validate(&package.model_fingerprint)?;
        Ok(package)
    }
    pub fn validate(&self, model_fingerprint: &str) -> Result<(), String> {
        if self.format_version != 2
            || self.model_fingerprint.is_empty()
            || self.model_fingerprint != model_fingerprint
            || self.question_version != QUESTION_VERSION
            || self.checkpoint_digest.len() != 64
            || !self
                .checkpoint_digest
                .bytes()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        {
            return Err("reader package does not match this model or question".into());
        }
        self.head.validate()?;
        self.dev.validate()?;
        self.baseline.validate()?;
        if (self.dev.combined.yes_total, self.dev.combined.no_total)
            != (
                self.baseline.combined.yes_total,
                self.baseline.combined.no_total,
            )
        {
            return Err("reader evaluation splits differ".into());
        }
        Ok(())
    }
}

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().ok_or("missing publication directory")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let mut temp = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    temp.write_all(bytes)
        .and_then(|_| temp.as_file().sync_all())
        .map_err(|e| e.to_string())?;
    temp.persist(path).map_err(|e| e.error.to_string())?;
    #[cfg(unix)]
    fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|e| e.to_string())?;
    Ok(())
}

pub struct Store {
    root: PathBuf,
}
impl Store {
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_owned(),
        }
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn stage(&self, package: &Package) -> Result<String, String> {
        package.validate(&package.model_fingerprint)?;
        let bytes = serde_json::to_vec(package).map_err(|e| e.to_string())?;
        let id = digest(&bytes);
        let path = self.root.join("candidates").join(format!("{id}.json"));
        if path.exists() {
            if fs::read(&path).map_err(|e| e.to_string())? != bytes {
                return Err("candidate hash collision or corruption".into());
            }
        } else {
            atomic_write(&path, &bytes)?;
        }
        Ok(id)
    }
    pub fn load_candidate(&self, id: &str, fingerprint: &str) -> Result<Package, String> {
        if id.len() != 64
            || !id
                .bytes()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        {
            return Err("invalid candidate identifier".into());
        }
        let bytes = fs::read(self.root.join("candidates").join(format!("{id}.json")))
            .map_err(|e| e.to_string())?;
        if digest(&bytes) != id {
            return Err("candidate hash mismatch".into());
        }
        let package: Package = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        package.validate(fingerprint)?;
        Ok(package)
    }
    pub fn load_active(&self, fingerprint: &str) -> Result<Option<Package>, String> {
        let path = self.root.join("active.json");
        if !path.exists() {
            return Ok(None);
        }
        let id: String = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        self.load_candidate(&id, fingerprint).map(Some)
    }
    pub fn promote(&self, id: &str, fingerprint: &str) -> Result<bool, String> {
        let candidate = self.load_candidate(id, fingerprint)?;
        if !candidate.dev.improves_on(&candidate.baseline)? {
            return Ok(false);
        }
        if let Some(active) = self.load_active(fingerprint)? {
            if !candidate.dev.improves_on(&active.dev)? {
                return Ok(false);
            }
            let previous = fs::read(self.root.join("active.json")).map_err(|e| e.to_string())?;
            atomic_write(&self.root.join("previous.json"), &previous)?;
        }
        atomic_write(
            &self.root.join("active.json"),
            &serde_json::to_vec(id).map_err(|e| e.to_string())?,
        )?;
        Ok(true)
    }

    /// Swap the active and previous pointers after verifying both candidates.
    /// Callers serialize this operation with the scheduler's runner lease.
    pub fn rollback(&self, fingerprint: &str) -> Result<String, String> {
        let read_id = |name: &str| -> Result<String, String> {
            let bytes = fs::read(self.root.join(name)).map_err(|e| e.to_string())?;
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())
        };
        let active = read_id("active.json")?;
        let previous = read_id("previous.json")?;
        if active == previous {
            return Err("active and previous candidates are identical".into());
        }
        self.load_candidate(&active, fingerprint)?;
        self.load_candidate(&previous, fingerprint)?;
        atomic_write(
            &self.root.join("active.json"),
            &serde_json::to_vec(&previous).map_err(|e| e.to_string())?,
        )?;
        atomic_write(
            &self.root.join("previous.json"),
            &serde_json::to_vec(&active).map_err(|e| e.to_string())?,
        )?;
        Ok(previous)
    }
}
