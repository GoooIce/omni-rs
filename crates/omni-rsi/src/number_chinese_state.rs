//! Descriptor-anchored cooperative state access for Chinese development runs.

use crate::{
    lease::Lease,
    number_v3_voice_recheck::{open_child, open_directory, read_child},
    speech_audio_checkpoint::{AudioCheckpoint, AudioProtocol},
};
use burn::tensor::backend::AutodiffBackend;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    ffi::OsStr,
    fs::File,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pointer {
    sha256: String,
}

pub(super) struct ChineseState {
    path: PathBuf,
    root: File,
    v3: File,
    conditioned: File,
    checkpoints: File,
    leases: [Lease; 4],
}

#[cfg(unix)]
fn same_directory(first: &File, second: &File) -> Result<bool, String> {
    use std::os::unix::fs::MetadataExt;
    let a = first.metadata().map_err(|e| e.to_string())?;
    let b = second.metadata().map_err(|e| e.to_string())?;
    Ok(a.is_dir() && b.is_dir() && a.dev() == b.dev() && a.ino() == b.ino())
}

#[cfg(not(unix))]
fn same_directory(_: &File, _: &File) -> Result<bool, String> {
    Ok(false)
}

impl ChineseState {
    pub(super) fn acquire(path: &Path, stop: &AtomicBool) -> Result<Self, String> {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        let root = open_directory(path)?;
        let v3 = open_child(&root, OsStr::new("number-audio-v3"), true)?;
        let conditioned = open_child(&v3, OsStr::new("audio-conditioned-v1"), true)?;
        let checkpoints = open_child(&conditioned, OsStr::new("checkpoints"), true)?;
        let foreground = Lease::try_acquire_at(&root, "foreground.lock")?
            .ok_or("another RSI foreground request is active")?;
        let device = Lease::try_acquire_at(&root, "device.lock")?
            .ok_or("another RSI process owns the device lease")?;
        let training = Lease::try_acquire_at(&root, "audio-training.lock")?
            .ok_or("another audio trainer owns this state")?;
        let checkpoint = Lease::try_acquire_at(&v3, "audio-training.lock")?
            .ok_or("another audio trainer owns this state")?;
        let state = Self {
            path: path.into(),
            root,
            v3,
            conditioned,
            checkpoints,
            leases: [foreground, device, training, checkpoint],
        };
        state.verify(stop)?;
        Ok(state)
    }

    pub(super) fn verify(&self, stop: &AtomicBool) -> Result<(), String> {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        let root = open_directory(&self.path)?;
        let v3 = open_child(&root, OsStr::new("number-audio-v3"), true)?;
        let conditioned = open_child(&v3, OsStr::new("audio-conditioned-v1"), true)?;
        let checkpoints = open_child(&conditioned, OsStr::new("checkpoints"), true)?;
        if !same_directory(&self.root, &root)?
            || !same_directory(&self.v3, &v3)?
            || !same_directory(&self.conditioned, &conditioned)?
            || !same_directory(&self.checkpoints, &checkpoints)?
        {
            return Err("state directory changed during Chinese evaluation".into());
        }
        for (lease, directory, name) in [
            (&self.leases[0], &root, "foreground.lock"),
            (&self.leases[1], &root, "device.lock"),
            (&self.leases[2], &root, "audio-training.lock"),
            (&self.leases[3], &v3, "audio-training.lock"),
        ] {
            lease.verify_at(directory, name)?;
        }
        Ok(())
    }

    pub(super) fn checkpoint<B: AutodiffBackend>(
        &self,
        expected: &AudioProtocol,
        device: &B::Device,
        stop: &AtomicBool,
    ) -> Result<(String, AudioCheckpoint, PathBuf), String> {
        self.verify(stop)?;
        let mut remaining = 256 * 1024 * 1024 + 4096;
        let pointer = read_child(
            &self.conditioned,
            OsStr::new("latest.json"),
            4096,
            &mut remaining,
            stop,
        )?;
        let pointer: Pointer = serde_json::from_slice(&pointer).map_err(|e| e.to_string())?;
        let id = pointer.sha256;
        if id.len() != 64
            || !id
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("invalid v3 candidate pointer".into());
        }
        let name = format!("{id}.json");
        let bytes = read_child(
            &self.checkpoints,
            OsStr::new(&name),
            256 * 1024 * 1024,
            &mut remaining,
            stop,
        )?;
        if format!("{:x}", Sha256::digest(&bytes)) != id {
            return Err("v3 candidate content mismatch".into());
        }
        let checkpoint: AudioCheckpoint =
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if checkpoint.protocol() != expected || checkpoint.progress().updates == 0 {
            return Err(
                "v3 checkpoint does not match current training identity or has no updates".into(),
            );
        }
        let normalized =
            serde_json::to_vec(&serde_json::to_value(&checkpoint).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if format!("{:x}", Sha256::digest(&normalized)) != id {
            return Err("loaded checkpoint identity differs from the candidate pointer".into());
        }
        drop(checkpoint.restore::<B>(device)?);
        self.verify(stop)?;
        let path = self
            .path
            .join("number-audio-v3/audio-conditioned-v1/checkpoints")
            .join(name);
        Ok((id, checkpoint, path))
    }
}
