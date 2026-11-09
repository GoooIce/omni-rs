//! Explicitly submitted, completed user speech awaiting independent verification.
//!
//! This inbox does not start a microphone, perform recognition, or mark an
//! utterance as training data. A directory is published only after its WAV and
//! manifest are complete; the content-addressed ID makes retries idempotent.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
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
#[cfg(unix)]
pub struct ForegroundLease {
    _foreground: File,
    _device: File,
}

#[cfg(unix)]
impl ForegroundLease {
    pub fn acquire(
        root: &Path,
        stop: &std::sync::atomic::AtomicBool,
        timeout: std::time::Duration,
    ) -> io::Result<Self> {
        use std::os::fd::AsRawFd;
        use std::sync::atomic::Ordering;
        fs::create_dir_all(root)?;
        let open = |name| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(root.join(name))
        };
        let foreground = open("foreground.lock")?;
        if unsafe { libc::flock(foreground.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let device = open("device.lock")?;
        let started = std::time::Instant::now();
        loop {
            if stop.load(Ordering::Acquire) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "live request cancelled",
                ));
            }
            if unsafe { libc::flock(device.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
                return Ok(Self {
                    _foreground: foreground,
                    _device: device,
                });
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::WouldBlock {
                return Err(error);
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

#[cfg(not(unix))]
pub struct ForegroundLease;
#[cfg(not(unix))]
impl ForegroundLease {
    pub fn acquire(
        _: &Path,
        _: &std::sync::atomic::AtomicBool,
        _: std::time::Duration,
    ) -> io::Result<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "cooperative device leases are unavailable",
        ))
    }
}

#[cfg(unix)]
struct QueueLock(File);
#[cfg(unix)]
impl QueueLock {
    fn try_acquire(directory: &Path) -> io::Result<Option<Self>> {
        use std::os::fd::AsRawFd;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(directory.join(".queue.lock"))?;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            return Ok(Some(Self(file)));
        }
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::WouldBlock {
            Ok(None)
        } else {
            Err(error)
        }
    }
}
#[cfg(unix)]
impl Drop for QueueLock {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}

#[cfg(not(unix))]
struct QueueLock;
#[cfg(not(unix))]
impl QueueLock {
    fn try_acquire(_: &Path) -> io::Result<Option<Self>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "speech inbox locking is unavailable",
        ))
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
        if fs::symlink_metadata(&manifest_path)?
            .file_type()
            .is_symlink()
            || fs::symlink_metadata(&audio)?.file_type().is_symlink()
        {
            return Err(invalid("speech entry contains a symbolic link"));
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
        let wav = hound::WavReader::open(&audio).map_err(|e| invalid(e.to_string()))?;
        if wav.spec().channels != 1
            || wav.spec().sample_rate != RATE
            || wav.spec().bits_per_sample != 16
            || wav.spec().sample_format != hound::SampleFormat::Int
            || wav.duration() as usize != manifest.samples
        {
            return Err(invalid("speech entry has an invalid WAV format"));
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
