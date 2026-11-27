//! Explicitly submitted, completed user speech awaiting independent verification.
//!
//! This inbox does not start a microphone, perform recognition, or mark an
//! utterance as training data. A directory is published only after its WAV and
//! manifest are complete; the content-addressed ID makes retries idempotent.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

const RATE: u32 = 16_000;
const MIN_SAMPLES: usize = 6_400;
const MAX_SAMPLES: usize = 16_000 * 15;
const MAX_PENDING: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Completion {
    Normal,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Receipt {
    Queued { id: String },
    Duplicate { id: String },
    Cancelled,
    Rejected { reason: String },
}

#[derive(Clone, Debug)]
pub struct PendingSpeech {
    pub id: String,
    pub source: String,
    pub sample_rate: u32,
    pub samples: usize,
    pub audio: PathBuf,
}

#[derive(Deserialize, Serialize)]
struct Manifest {
    version: u8,
    id: String,
    source: String,
    sample_rate: u32,
    samples: usize,
    audio_sha256: String,
}

pub struct SpeechInbox {
    directory: PathBuf,
}

/// One exclusively claimed learning utterance, or a corrupt entry already
/// retained in the rejection archive.
pub enum SpeechClaim {
    Valid(PendingSpeech),
    Rejected { id: String, reason: String },
}

#[derive(Clone, Copy)]
pub enum SpeechOutcome {
    Accepted,
    Rejected,
}

impl SpeechOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
        }
    }

    fn directory(self) -> &'static str {
        match self {
            Self::Accepted => "processed",
            Self::Rejected => "rejected",
        }
    }
}

/// One consumer owns the processing directory until this lease is dropped.
pub struct SpeechWorker {
    root: PathBuf,
    _lease: WorkerLease,
}

/// A bounded, in-memory copy of one explicitly enabled live utterance.
#[derive(Default)]
pub struct SpeechCapture {
    samples: Vec<f32>,
    rejection: Option<&'static str>,
}

impl SpeechCapture {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, samples: &[f32]) {
        if self.rejection.is_some() {
            return;
        }
        if samples.len() > MAX_SAMPLES.saturating_sub(self.samples.len()) {
            self.samples.clear();
            self.rejection = Some("speech exceeds the 15-second learning limit");
        } else if samples
            .iter()
            .any(|sample| !sample.is_finite() || sample.abs() > 1.0)
        {
            self.samples.clear();
            self.rejection = Some("speech contains invalid or out-of-range PCM");
        } else {
            self.samples.extend_from_slice(samples);
        }
    }

    pub fn finish(self, inbox: &SpeechInbox, completion: Completion) -> io::Result<Receipt> {
        if completion == Completion::Cancelled {
            return Ok(Receipt::Cancelled);
        }
        if let Some(reason) = self.rejection {
            return Ok(Receipt::Rejected {
                reason: reason.into(),
            });
        }
        inbox.submit(&self.samples, completion)
    }
}

/// Cooperate with the idle RSI scheduler while a live request owns the device.
pub struct ForegroundLease {
    _foreground: crate::lease::FileLease,
    _device: crate::lease::FileLease,
}

impl ForegroundLease {
    pub fn acquire(
        root: &Path,
        stop: &std::sync::atomic::AtomicBool,
        timeout: std::time::Duration,
    ) -> io::Result<Self> {
        use std::sync::atomic::Ordering;
        fs::create_dir_all(root)?;
        let foreground = crate::lease::FileLease::try_acquire(&root.join("foreground.lock"))?
            .ok_or_else(|| io::Error::new(io::ErrorKind::WouldBlock, "foreground lease busy"))?;
        let started = std::time::Instant::now();
        loop {
            if stop.load(Ordering::Acquire) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "live request cancelled",
                ));
            }
            if let Some(device) = crate::lease::FileLease::try_acquire(&root.join("device.lock"))? {
                return Ok(Self {
                    _foreground: foreground,
                    _device: device,
                });
            }
            if started.elapsed() >= timeout {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "background device work did not yield",
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}

/// Read-only view of queued speech-learning utterances across inbox directories.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpeechQueueCounts {
    pub inbox: usize,
    pub processing: usize,
    pub online_data: usize,
    pub rejected: usize,
}

impl SpeechQueueCounts {
    pub fn inspect(root: &Path) -> io::Result<Self> {
        Ok(Self {
            inbox: Self::count_entries(&root.join("inbox"))?,
            processing: Self::count_entries(&root.join("processing"))?,
            online_data: Self::count_entries(&root.join("online-data"))?,
            rejected: Self::count_entries(&root.join("rejected"))?,
        })
    }

    pub fn count_entries(directory: &Path) -> io::Result<usize> {
        let entries = match fs::read_dir(directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(error),
        };
        let mut count = 0;
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            if name.to_str().is_some_and(valid_hex_id) && entry.file_type()?.is_dir() {
                count += 1;
            }
        }
        Ok(count)
    }
}

fn valid_hex_id(name: &str) -> bool {
    name.len() == 64
        && name
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

struct QueueLock(#[allow(dead_code)] crate::lease::FileLease);
impl QueueLock {
    fn try_acquire(directory: &Path) -> io::Result<Option<Self>> {
        Self::try_acquire_file(&directory.join(".queue.lock"))
    }

    fn try_acquire_file(path: &Path) -> io::Result<Option<Self>> {
        Ok(crate::lease::FileLease::try_acquire(path)?.map(Self))
    }
}

struct WorkerLease(#[allow(dead_code)] QueueLock);

impl WorkerLease {
    fn acquire(root: &Path) -> io::Result<Self> {
        QueueLock::try_acquire_file(&root.join(".ingest.lock"))?
            .map(Self)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "speech processor is already running",
                )
            })
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn sha256(path: &Path) -> io::Result<String> {
    let mut digest = Sha256::new();
    let mut input = File::open(path)?;
    let mut block = [0; 64 * 1024];
    loop {
        let count = input.read(&mut block)?;
        if count == 0 {
            break;
        }
        digest.update(&block[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn valid_id(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

impl SpeechInbox {
    pub fn open(root: impl AsRef<Path>) -> io::Result<Self> {
        let directory = root.as_ref().join("inbox");
        fs::create_dir_all(&directory)?;
        Ok(Self {
            directory: directory.canonicalize()?,
        })
    }

    /// Submit mono 16 kHz PCM only after normal end of the user's input.
    pub fn submit(&self, samples: &[f32], completion: Completion) -> io::Result<Receipt> {
        if completion == Completion::Cancelled {
            return Ok(Receipt::Cancelled);
        }
        if !(MIN_SAMPLES..=MAX_SAMPLES).contains(&samples.len()) {
            return Ok(Receipt::Rejected {
                reason: "speech must be between 0.4 and 15 seconds".into(),
            });
        }
        if samples
            .iter()
            .any(|value| !value.is_finite() || value.abs() > 1.0)
        {
            return Ok(Receipt::Rejected {
                reason: "speech contains invalid or out-of-range PCM".into(),
            });
        }
        let Some(_lock) = QueueLock::try_acquire(&self.directory)? else {
            return Ok(Receipt::Rejected {
                reason: "speech inbox is busy".into(),
            });
        };
        let temporary = tempfile::Builder::new()
            .prefix(".speech-")
            .tempdir_in(&self.directory)?;
        let audio = temporary.path().join("audio.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut wav = hound::WavWriter::create(&audio, spec).map_err(|e| invalid(e.to_string()))?;
        for sample in samples {
            wav.write_sample((sample * i16::MAX as f32).round() as i16)
                .map_err(|e| invalid(e.to_string()))?;
        }
        wav.finalize().map_err(|e| invalid(e.to_string()))?;
        File::open(&audio)?.sync_all()?;
        let id = sha256(&audio)?;
        if self.get(&id)?.is_some() {
            return Ok(Receipt::Duplicate { id });
        }
        if let Some(root) = self.directory.parent() {
            for name in ["processing", "processed", "rejected"] {
                let archive = Self {
                    directory: root.join(name),
                };
                if archive.get(&id)?.is_some() {
                    return Ok(Receipt::Duplicate { id });
                }
            }
        }
        if self.pending()?.len() >= MAX_PENDING {
            return Ok(Receipt::Rejected {
                reason: "speech inbox is full".into(),
            });
        }
        let manifest = Manifest {
            version: 1,
            id: id.clone(),
            source: "user".into(),
            sample_rate: RATE,
            samples: samples.len(),
            audio_sha256: id.clone(),
        };
        let mut file = File::create(temporary.path().join("manifest.json"))?;
        serde_json::to_writer(&mut file, &manifest).map_err(|e| invalid(e.to_string()))?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        File::open(temporary.path())?.sync_all()?;
        let destination = self.directory.join(&id);
        match fs::rename(temporary.path(), &destination) {
            Ok(()) => {
                File::open(&self.directory)?.sync_all()?;
                Ok(Receipt::Queued { id })
            }
            Err(error) if destination.exists() => {
                self.get(&id)?.ok_or(error)?;
                Ok(Receipt::Duplicate { id })
            }
            Err(error) => Err(error),
        }
    }

    /// Read and validate one fully published entry.
    pub fn get(&self, id: &str) -> io::Result<Option<PendingSpeech>> {
        if !valid_id(id) {
            return Err(invalid("invalid speech ID"));
        }
        let directory = self.directory.join(id);
        let metadata = match fs::symlink_metadata(&directory) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(invalid("speech entry is not a directory"));
        }
        let manifest_path = directory.join("manifest.json");
        let audio = directory.join("audio.wav");
        let manifest_metadata = fs::symlink_metadata(&manifest_path)?;
        let audio_metadata = fs::symlink_metadata(&audio)?;
        if !manifest_metadata.is_file()
            || manifest_metadata.len() > 4096
            || !audio_metadata.is_file()
            || audio_metadata.len() > (MAX_SAMPLES * 2 + 4096) as u64
        {
            return Err(invalid("speech entry has an invalid file"));
        }
        let manifest: Manifest = serde_json::from_reader(File::open(manifest_path)?)
            .map_err(|e| invalid(e.to_string()))?;
        if manifest.version != 1
            || manifest.id != id
            || manifest.source != "user"
            || manifest.sample_rate != RATE
            || !(MIN_SAMPLES..=MAX_SAMPLES).contains(&manifest.samples)
            || manifest.audio_sha256 != id
            || sha256(&audio)? != id
        {
            return Err(invalid("speech entry failed integrity validation"));
        }
        let mut wav = hound::WavReader::open(&audio).map_err(|e| invalid(e.to_string()))?;
        if wav.spec().channels != 1
            || wav.spec().sample_rate != RATE
            || wav.spec().bits_per_sample != 16
            || wav.spec().sample_format != hound::SampleFormat::Int
            || wav.duration() as usize != manifest.samples
        {
            return Err(invalid("speech entry has an invalid WAV format"));
        }
        let mut decoded = 0;
        for sample in wav.samples::<i16>() {
            sample.map_err(|error| invalid(error.to_string()))?;
            decoded += 1;
        }
        if decoded != manifest.samples {
            return Err(invalid("speech entry has a truncated WAV"));
        }
        Ok(Some(PendingSpeech {
            id: manifest.id,
            source: manifest.source,
            sample_rate: manifest.sample_rate,
            samples: manifest.samples,
            audio,
        }))
    }

    pub fn pending(&self) -> io::Result<Vec<PendingSpeech>> {
        let mut ids = Vec::new();
        for entry in fs::read_dir(&self.directory)? {
            let entry = entry?;
            let name = entry.file_name();
            if let Some(name) = name.to_str().filter(|name| valid_id(name)) {
                ids.push(name.to_owned());
            }
        }
        ids.sort_unstable();
        ids.into_iter()
            .map(|id| {
                self.get(&id)?
                    .ok_or_else(|| invalid("speech entry disappeared"))
            })
            .collect()
    }
}

fn speech_ids(directory: &Path) -> io::Result<Vec<String>> {
    let mut ids = Vec::new();
    for entry in fs::read_dir(directory)? {
        let name = entry?.file_name();
        if let Some(name) = name.to_str().filter(|name| valid_id(name)) {
            ids.push(name.to_owned());
        }
    }
    ids.sort_unstable();
    Ok(ids)
}

fn private_dir(path: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)
}

impl SpeechWorker {
    /// Acquire the existing cross-language worker lease without starting any model.
    pub fn open(root: impl AsRef<Path>) -> io::Result<Self> {
        private_dir(root.as_ref())?;
        let root = root.as_ref().canonicalize()?;
        for name in ["inbox", "processing", "processed", "rejected", "duplicates"] {
            private_dir(&root.join(name))?;
        }
        let lease = WorkerLease::acquire(&root)?;
        Ok(Self {
            root,
            _lease: lease,
        })
    }

    /// Resume interrupted work first; otherwise claim and validate one entry.
    pub fn claim(&self) -> io::Result<Option<SpeechClaim>> {
        let processing = self.root.join("processing");
        if let Some(id) = speech_ids(&processing)?.into_iter().next() {
            return self.validate_claim(id).map(Some);
        }
        let inbox = self.root.join("inbox");
        let Some(lock) = QueueLock::try_acquire(&inbox)? else {
            return Ok(None);
        };
        for id in speech_ids(&inbox)? {
            let duplicate = ["processed", "rejected"]
                .iter()
                .any(|name| self.root.join(name).join(&id).exists());
            let destination = if duplicate {
                self.root
                    .join("duplicates")
                    .join(format!("{id}-{:032x}", rand::random::<u128>()))
            } else {
                processing.join(&id)
            };
            fs::rename(inbox.join(&id), &destination)?;
            File::open(&inbox)?.sync_all()?;
            File::open(destination.parent().expect("queue entry has a parent"))?.sync_all()?;
            if !duplicate {
                drop(lock);
                return self.validate_claim(id).map(Some);
            }
        }
        Ok(None)
    }

    fn validate_claim(&self, id: String) -> io::Result<SpeechClaim> {
        let processing = SpeechInbox {
            directory: self.root.join("processing"),
        };
        match processing.get(&id) {
            Ok(Some(entry)) => Ok(SpeechClaim::Valid(entry)),
            Ok(None) => Err(invalid("claimed speech entry disappeared")),
            Err(error) => {
                self.finish_entry(&id, SpeechOutcome::Rejected, "integrity")?;
                Ok(SpeechClaim::Rejected {
                    id,
                    reason: error.to_string(),
                })
            }
        }
    }

    pub fn finish(
        &self,
        entry: &PendingSpeech,
        outcome: SpeechOutcome,
        reason: &str,
    ) -> io::Result<()> {
        if !valid_id(&entry.id)
            || entry.audio
                != self
                    .root
                    .join("processing")
                    .join(&entry.id)
                    .join("audio.wav")
        {
            return Err(invalid("speech claim belongs to a different queue"));
        }
        self.finish_entry(&entry.id, outcome, reason)
    }

    fn finish_entry(&self, id: &str, outcome: SpeechOutcome, reason: &str) -> io::Result<()> {
        if !valid_id(id) {
            return Err(invalid("invalid speech queue completion"));
        }
        let processing = self.root.join("processing");
        let entry = processing.join(id);
        let target_directory = self.root.join(outcome.directory());
        let target = target_directory.join(id);
        if target.exists() {
            return Err(invalid("speech result already exists"));
        }
        let metadata = fs::symlink_metadata(&entry)?;
        if metadata.is_dir() && !metadata.file_type().is_symlink() {
            let result = entry.join("result.json");
            if result.exists() {
                let metadata = fs::symlink_metadata(&result)?;
                if !metadata.is_file() || metadata.len() > 4096 {
                    return Err(invalid("speech result has an invalid file"));
                }
                let value: serde_json::Value = serde_json::from_reader(File::open(&result)?)
                    .map_err(|error| invalid(error.to_string()))?;
                if value.get("outcome").and_then(|value| value.as_str()) != Some(outcome.as_str()) {
                    return Err(invalid("speech completion conflicts with previous outcome"));
                }
            } else {
                let mut temporary = tempfile::Builder::new()
                    .prefix(".result-")
                    .tempfile_in(&entry)?;
                let value = serde_json::json!({"outcome": outcome.as_str(), "reason": reason.chars().take(200).collect::<String>()});
                serde_json::to_writer(&mut temporary, &value)
                    .map_err(|error| invalid(error.to_string()))?;
                temporary.as_file().sync_all()?;
                temporary.persist(&result).map_err(|error| error.error)?;
            }
            File::open(&entry)?.sync_all()?;
        }
        fs::rename(&entry, &target)?;
        File::open(&processing)?.sync_all()?;
        File::open(&target_directory)?.sync_all()?;
        Ok(())
    }
}
