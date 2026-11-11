//! Checkpoints for audio-conditioned training, isolated from self-reading.
use crate::{
    fingerprint::{codec_fingerprint, target_codec_hash},
    number_audio_v3::v3_audio_implementation_hash_for_backend,
    speech_audio_data::AudioDataset,
    speech_checkpoint::{atomic_json, implementation_hash},
    speech_training::{SpeechOptimizer, SpeechOptimizerSnapshot},
};
use burn::tensor::backend::AutodiffBackend;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

/// The dataset digest must cover the immutable manifest and every referenced
/// input/target file; admission and digest construction belong to its loader.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AudioProtocol {
    pub version: u32,
    pub base_hash: String,
    pub codec_hash: String,
    pub dataset_hash: String,
    pub implementation_hash: String,
    pub backend: String,
    pub width: usize,
    pub train_records: usize,
    pub mask_seed: u64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct AudioProgress {
    pub epoch: usize,
    pub next_index: usize,
    pub updates: usize,
}

/// Integrity and implementation identity only; this is not a restorable checkpoint
/// and does not establish that the current model or dataset matches it.
pub struct AudioCheckpointSummary {
    pub protocol_version: u32,
    pub progress: AudioProgress,
    pub checkpoint_sha256: String,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct AudioCheckpoint {
    protocol: AudioProtocol,
    progress: AudioProgress,
    optimizer: SpeechOptimizerSnapshot,
}

fn hex_id(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn validate(protocol: &AudioProtocol, progress: &AudioProgress) -> Result<(), String> {
    let expected_implementation = match protocol.version {
        1 => audio_implementation_hash_for_backend(&protocol.backend)?,
        2 => v3_audio_implementation_hash_for_backend(&protocol.backend)?,
        _ => return Err("invalid audio training protocol or progress".into()),
    };
    if protocol.implementation_hash != expected_implementation {
        return Err("audio checkpoint implementation changed; existing state is untouched; use its original binary".into());
    }
    if [
        &protocol.base_hash,
        &protocol.codec_hash,
        &protocol.dataset_hash,
        &protocol.implementation_hash,
    ]
    .iter()
    .any(|value| !hex_id(value))
        || protocol.backend.is_empty()
        || protocol.backend.len() > 256
        || !(4..=65_536).contains(&protocol.width)
        || !(1..=1_000_000).contains(&protocol.train_records)
        || progress.next_index >= protocol.train_records
        || progress
            .epoch
            .checked_mul(protocol.train_records)
            .and_then(|n| n.checked_add(progress.next_index))
            != Some(progress.updates)
    {
        return Err("invalid audio training protocol or progress".into());
    }
    Ok(())
}

fn audio_shared_sources() -> Vec<(&'static str, &'static [u8])> {
    macro_rules! source {
        ($name:literal, $path:literal) => {
            ($name, include_bytes!($path).as_slice())
        };
    }
    vec![
        source!("speech_audio.rs", "speech_audio.rs"),
        source!("speech_audio_checkpoint.rs", "speech_audio_checkpoint.rs"),
        source!("speech_audio_data.rs", "speech_audio_data.rs"),
        source!("speech_audio_round.rs", "speech_audio_round.rs"),
        source!("audio_idle.rs", "audio_idle.rs"),
        source!("lease.rs", "lease.rs"),
        source!("fingerprint.rs", "fingerprint.rs"),
        source!("audio_main.rs", "audio_main.rs"),
        source!("Cargo.toml", "../Cargo.toml"),
        source!("build.rs", "../build.rs"),
        source!("build_backend.rs", "../build_backend.rs"),
        source!("build_environment.rs", "../build_environment.rs"),
        source!("build_invocation.rs", "../build_invocation.rs"),
        source!("root/Cargo.toml", "../../../Cargo.toml"),
        source!("lfm3/Cargo.toml", "../../lfm3-omni/Cargo.toml"),
        source!("lfm3/media_io.rs", "../../lfm3-omni/src/media_io.rs"),
        source!("src/input.rs", "../../../src/input.rs"),
        source!("src/config.rs", "../../../src/config.rs"),
        source!("src/weights.rs", "../../../src/weights.rs"),
        source!("src/weights/layout.rs", "../../../src/weights/layout.rs"),
        source!("src/tensor/mod.rs", "../../../src/tensor/mod.rs"),
        source!("src/tensor/norm.rs", "../../../src/tensor/norm.rs"),
        source!(
            "src/tensor/embedding.rs",
            "../../../src/tensor/embedding.rs"
        ),
        source!("src/tensor/cache.rs", "../../../src/tensor/cache.rs"),
        source!(
            "src/tensor/decode_cache.rs",
            "../../../src/tensor/decode_cache.rs"
        ),
        source!("lfm3/config.rs", "../../lfm3-omni/src/config.rs"),
        source!("lfm3/audio.rs", "../../lfm3-omni/src/audio.rs"),
        source!(
            "lfm3/audio_encoder.rs",
            "../../lfm3-omni/src/audio_encoder.rs"
        ),
        source!("lfm3/chat.rs", "../../lfm3-omni/src/chat.rs"),
        source!("lfm3/media.rs", "../../lfm3-omni/src/media.rs"),
        source!("lfm3/prepared.rs", "../../lfm3-omni/src/prepared.rs"),
    ]
}

pub(crate) fn hash_audio_sources(domain: &[u8], parent: &str, sources: &[(&str, &[u8])]) -> String {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update((parent.len() as u64).to_le_bytes());
    hash.update(parent.as_bytes());
    for (name, bytes) in sources {
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    format!("{:x}", hash.finalize())
}

/// Bind legacy audio training only to its actual shared execution sources.
pub fn audio_implementation_hash() -> Result<String, String> {
    #[cfg(feature = "cpu")]
    let backend = "autodiff<ndarray>";
    #[cfg(not(feature = "cpu"))]
    let backend = "autodiff<cubecl<wgpu<msl>>>";
    audio_implementation_hash_for_backend(backend)
}

/// Reject unknown, disabled or unverified backends instead of naming a digest.
pub fn audio_implementation_hash_for_backend(backend: &str) -> Result<String, String> {
    let backend_hash = checked_backend_graph(
        backend,
        env!("OMNI_RSI_BACKEND_CPU"),
        env!("OMNI_RSI_BACKEND_GPU"),
    )?;
    let mut sources = audio_shared_sources();
    sources.push(("patched-numerical-backend", backend_hash.as_bytes()));
    Ok(hash_audio_sources(
        b"omni-audio-legacy-v2\0",
        &implementation_hash(),
        &sources,
    ))
}

fn checked_backend_graph<'a>(backend: &str, cpu: &'a str, gpu: &'a str) -> Result<&'a str, String> {
    let backend_hash = match backend {
        "autodiff<ndarray>" | "ndarray" => cpu,
        "autodiff<cubecl<wgpu<msl>>>" | "cubecl<wgpu<msl>>" => gpu,
        _ => return Err("unsupported audio training backend".into()),
    };
    if backend_hash == "backend-unverified-context" {
        return Err(
            "cannot verify Cargo invocation directory or Cargo home for audio training identity"
                .into(),
        );
    }
    if !hex_id(backend_hash) {
        return Err(
            "audio training backend is not enabled or its source identity is unavailable".into(),
        );
    }
    Ok(backend_hash)
}

impl AudioProtocol {
    /// Build the identical model/data binding for training and evaluation.
    pub fn for_dataset(
        model: &Path,
        dataset: &AudioDataset,
        backend: String,
        stop: &AtomicBool,
    ) -> Result<Self, String> {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        let implementation_hash = audio_implementation_hash_for_backend(&backend)?;
        let base_hash = omni_lfm3::speech_adapter::source_fingerprint(model)
            .map_err(|error| error.to_string())?;
        let codec_hash = audio_codec_hash(model, stop)?;
        let mut full_codec_hash = None;
        for index in 0..dataset.len() {
            let record = dataset.record(index).ok_or("missing audio record")?;
            if record
                .target_codec_sha256
                .as_ref()
                .is_some_and(|hash| hash != &codec_hash)
            {
                return Err("audio target codec does not match the current model".into());
            }
            if let Some(expected) = &record.codec_fingerprint_sha256 {
                if full_codec_hash.is_none() {
                    full_codec_hash =
                        Some(codec_fingerprint(&model.join("audio_tokenizer"), stop)?);
                }
                if full_codec_hash.as_ref() != Some(expected) {
                    return Err(
                        "audio target codec fingerprint does not match the current model".into(),
                    );
                }
            }
        }
        let config: serde_json::Value = serde_json::from_slice(
            &fs::read(model.join("config.json")).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let width = config
            .pointer("/text_config/hidden_size")
            .and_then(serde_json::Value::as_u64)
            .and_then(|width| usize::try_from(width).ok())
            .filter(|width| (4..=65_536).contains(width))
            .ok_or("invalid audio model width")?;
        Ok(Self {
            version: 1,
            base_hash,
            codec_hash,
            dataset_hash: dataset.digest().into(),
            implementation_hash,
            backend,
            width,
            train_records: dataset.len(),
            mask_seed: 159,
        })
    }

    /// Check the files again after loading or evaluating the model.
    pub fn verify_model(&self, model: &Path, stop: &AtomicBool) -> Result<(), String> {
        if omni_lfm3::speech_adapter::source_fingerprint(model)
            .map_err(|error| error.to_string())?
            != self.base_hash
            || audio_codec_hash(model, stop)? != self.codec_hash
        {
            return Err("audio model or codec changed".into());
        }
        Ok(())
    }

    /// Derive the mask seed from checkpointed progress, including after resume.
    pub fn seed_for_step(&self, progress: &AudioProgress) -> Result<u64, String> {
        validate(self, progress)?;
        Ok(self.mask_seed.wrapping_add(progress.updates as u64))
    }
}

// Preserve the established target and checkpoint digest for unsharded codecs.
pub(crate) fn audio_codec_hash(model: &Path, stop: &AtomicBool) -> Result<String, String> {
    target_codec_hash(&model.join("audio_tokenizer"), stop)
}

impl AudioCheckpoint {
    pub fn protocol(&self) -> &AudioProtocol {
        &self.protocol
    }

    pub fn progress(&self) -> &AudioProgress {
        &self.progress
    }

    pub fn new<B: AutodiffBackend>(
        protocol: AudioProtocol,
        progress: AudioProgress,
        optimizer: &SpeechOptimizer<B>,
    ) -> Result<Self, String> {
        validate(&protocol, &progress)?;
        let (a, b) = optimizer.adapter().weights();
        if a.dims() != [4, protocol.width]
            || b.dims() != [protocol.width, 4]
            || B::name(&a.device()) != protocol.backend
        {
            return Err("audio optimizer does not match its protocol".into());
        }
        Ok(Self {
            protocol,
            progress,
            optimizer: optimizer.snapshot()?,
        })
    }

    /// Persist content first, then atomically switch its hash-addressed pointer.
    pub fn save(&self, state: &Path) -> Result<String, String> {
        validate(&self.protocol, &self.progress)?;
        let directory = state.join("audio-conditioned-v1");
        let checkpoints = directory.join("checkpoints");
        fs::create_dir_all(&checkpoints).map_err(|error| error.to_string())?;
        let value = serde_json::to_value(self).map_err(|error| error.to_string())?;
        let bytes = serde_json::to_vec(&value).map_err(|error| error.to_string())?;
        let id = format!("{:x}", Sha256::digest(&bytes));
        let path = checkpoints.join(format!("{id}.json"));
        if path.exists() {
            let info = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
            if !info.is_file() || fs::read(&path).map_err(|error| error.to_string())? != bytes {
                return Err("existing audio checkpoint is corrupt".into());
            }
        } else {
            atomic_json(&path, &value)?;
        }
        atomic_json(
            &directory.join("latest.json"),
            &serde_json::json!({"sha256":id}),
        )?;
        Ok(id)
    }

    /// Inspect integrity and implementation identity without authorizing a restore.
    /// Model and data matching still requires `load` with an expected protocol.
    ///
    /// A preflight result cannot restore optimizer state without `load(expected)`:
    /// ```compile_fail
    /// # use omni_rsi::speech_audio_checkpoint::AudioCheckpoint;
    /// # use burn::tensor::backend::AutodiffBackend;
    /// fn restore_from_preflight<B: AutodiffBackend>(state: &std::path::Path, device: &B::Device) {
    ///     let info = AudioCheckpoint::inspect(state).unwrap().unwrap();
    ///     let _ = info.restore::<B>(device);
    /// }
    /// ```
    /// A checkpoint obtained through the checked API supports the same generic call:
    /// ```no_run
    /// # use omni_rsi::speech_audio_checkpoint::AudioCheckpoint;
    /// # use burn::tensor::backend::AutodiffBackend;
    /// fn restore_checked<B: AutodiffBackend>(checkpoint: &AudioCheckpoint, device: &B::Device) {
    ///     let _ = checkpoint.restore::<B>(device);
    /// }
    /// fn inspect_only(state: &std::path::Path) {
    ///     let info = AudioCheckpoint::inspect(state).unwrap().unwrap();
    ///     let _ = info.protocol_version;
    /// }
    /// ```
    pub fn inspect(state: &Path) -> Result<Option<AudioCheckpointSummary>, String> {
        Ok(
            Self::read_validated(state)?.map(|(checkpoint, checkpoint_sha256)| {
                AudioCheckpointSummary {
                    protocol_version: checkpoint.protocol.version,
                    progress: checkpoint.progress,
                    checkpoint_sha256,
                }
            }),
        )
    }

    fn read_validated(state: &Path) -> Result<Option<(Self, String)>, String> {
        let directory = state.join("audio-conditioned-v1");
        let pointer = directory.join("latest.json");
        let info = match fs::symlink_metadata(&pointer) {
            Ok(info) => info,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.to_string()),
        };
        if !info.is_file() || info.len() > 4096 {
            return Err("invalid audio checkpoint pointer".into());
        }
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(pointer).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
        let id = value
            .get("sha256")
            .and_then(|value| value.as_str())
            .filter(|id| hex_id(id))
            .ok_or("invalid audio checkpoint pointer")?
            .to_owned();
        let path = directory.join("checkpoints").join(format!("{id}.json"));
        let info = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
        if !info.is_file() || info.len() > 256 * 1024 * 1024 {
            return Err("invalid audio checkpoint file".into());
        }
        let bytes = fs::read(path).map_err(|error| error.to_string())?;
        if format!("{:x}", Sha256::digest(&bytes)) != id {
            return Err("audio checkpoint hash mismatch".into());
        }
        let checkpoint: Self = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        validate(&checkpoint.protocol, &checkpoint.progress)?;
        Ok(Some((checkpoint, id)))
    }

    /// Reject old formats, changed source/data/model, and damaged checkpoint files.
    pub fn load(state: &Path, expected: &AudioProtocol) -> Result<Option<Self>, String> {
        validate(expected, &AudioProgress::default())?;
        let Some((checkpoint, _)) = Self::read_validated(state)? else {
            return Ok(None);
        };
        if &checkpoint.protocol != expected {
            return Err(
                "audio checkpoint does not match the current model, data or protocol".into(),
            );
        }
        Ok(Some(checkpoint))
    }

    pub fn restore<B: AutodiffBackend>(
        &self,
        device: &B::Device,
    ) -> Result<SpeechOptimizer<B>, String> {
        validate(&self.protocol, &self.progress)?;
        if B::name(device) != self.protocol.backend {
            return Err("audio checkpoint backend does not match its protocol".into());
        }
        let optimizer = SpeechOptimizer::restore(self.optimizer.clone(), device)?;
        let (a, b) = optimizer.adapter().weights();
        if a.dims() != [4, self.protocol.width] || b.dims() != [self.protocol.width, 4] {
            return Err("audio optimizer does not match its checkpoint protocol".into());
        }
        Ok(optimizer)
    }
}

#[cfg(test)]
mod backend_identity_tests {
    use super::checked_backend_graph;

    #[test]
    fn unknown_disabled_and_unverified_graphs_fail_closed() {
        let cpu = "1".repeat(64);
        let gpu = "2".repeat(64);
        assert_eq!(
            checked_backend_graph("autodiff<ndarray>", &cpu, &gpu).unwrap(),
            cpu
        );
        assert_eq!(
            checked_backend_graph("autodiff<cubecl<wgpu<msl>>>", &cpu, &gpu).unwrap(),
            gpu
        );
        assert!(checked_backend_graph("custom-unknown-backend", &cpu, &gpu).is_err());
        assert!(
            checked_backend_graph("autodiff<cubecl<wgpu<msl>>>", &cpu, "backend-not-enabled")
                .is_err()
        );
        assert!(
            checked_backend_graph("autodiff<ndarray>", "backend-unverified-context", &gpu).is_err()
        );
        assert!(checked_backend_graph("autodiff<cubecl<wgpu<spirv>>>", &cpu, &gpu).is_err());
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::backend::Backend,
    };
    type Cpu = Autodiff<NdArray<f32>>;

    fn protocol() -> AudioProtocol {
        AudioProtocol {
            version: 1,
            base_hash: "1".repeat(64),
            codec_hash: "2".repeat(64),
            dataset_hash: "3".repeat(64),
            implementation_hash: audio_implementation_hash().unwrap(),
            backend: Cpu::name(&Default::default()),
            width: 16,
            train_records: 2,
            mask_seed: 159,
        }
    }

    #[test]
    fn v3_only_sources_do_not_change_the_legacy_training_identity() {
        let mut shared = audio_shared_sources();
        shared.push((
            "patched-numerical-backend",
            env!("OMNI_RSI_BACKEND_CPU").as_bytes(),
        ));
        let v3 = crate::number_audio_v3::v3_audio_sources();
        let native = implementation_hash();
        let legacy = hash_audio_sources(b"omni-audio-legacy-v2\0", &native, &shared);
        let v3_hash = hash_audio_sources(b"omni-audio-v3-v2\0", &legacy, &v3);
        assert_eq!(legacy, audio_implementation_hash().unwrap());
        assert_eq!(
            v3_hash,
            v3_audio_implementation_hash_for_backend("autodiff<ndarray>").unwrap()
        );

        for name in [
            "number_v3_target.rs",
            "number_audio_v3.rs",
            "number_development.rs",
            "speech_target_format.rs",
        ] {
            let mut changed_v3 = v3.clone();
            changed_v3
                .iter_mut()
                .find(|(path, _)| *path == name)
                .unwrap()
                .1 = b"changed v3 source";
            assert_ne!(
                hash_audio_sources(b"omni-audio-v3-v2\0", &legacy, &changed_v3),
                v3_hash,
                "{name}"
            );
            assert_eq!(
                hash_audio_sources(b"omni-audio-legacy-v2\0", &native, &shared),
                legacy
            );
        }

        for name in [
            "Cargo.toml",
            "src/config.rs",
            "src/weights.rs",
            "src/weights/layout.rs",
            "src/tensor/norm.rs",
            "speech_audio.rs",
        ] {
            let mut changed_shared = shared.clone();
            changed_shared
                .iter_mut()
                .find(|(path, _)| *path == name)
                .unwrap()
                .1 = b"changed shared implementation";
            let new_legacy =
                hash_audio_sources(b"omni-audio-legacy-v2\0", &native, &changed_shared);
            assert_ne!(new_legacy, legacy, "{name}");
            assert_ne!(
                hash_audio_sources(b"omni-audio-v3-v2\0", &new_legacy, &v3),
                v3_hash,
                "{name}"
            );
        }

        let new_native = "0".repeat(64);
        let new_legacy = hash_audio_sources(b"omni-audio-legacy-v2\0", &new_native, &shared);
        assert_ne!(new_legacy, legacy);
        assert_ne!(
            hash_audio_sources(b"omni-audio-v3-v2\0", &new_legacy, &v3),
            v3_hash
        );
    }

    #[test]
    fn checkpoint_version_cannot_use_the_other_paths_implementation_identity() {
        let device = Default::default();
        let optimizer = SpeechOptimizer::<Cpu>::new(16, 4, 1, &device).unwrap();
        let mut value = protocol();
        assert!(AudioCheckpoint::new(value.clone(), AudioProgress::default(), &optimizer).is_ok());
        value.version = 2;
        assert!(AudioCheckpoint::new(value.clone(), AudioProgress::default(), &optimizer).is_err());
        value.implementation_hash =
            v3_audio_implementation_hash_for_backend(&value.backend).unwrap();
        assert!(AudioCheckpoint::new(value.clone(), AudioProgress::default(), &optimizer).is_ok());
        value.version = 1;
        assert!(AudioCheckpoint::new(value, AudioProgress::default(), &optimizer).is_err());
    }

    fn update(optimizer: &mut SpeechOptimizer<Cpu>) {
        let (a, b) = optimizer.adapter().weights();
        optimizer
            .update(
                a.square().mean() + b.add_scalar(1.0).square().mean(),
                0.0001,
            )
            .unwrap();
    }

    #[test]
    fn checkpoint_is_isolated_and_rejects_changed_or_corrupt_state() {
        let root = tempfile::tempdir().unwrap();
        let protocol = protocol();
        let mut optimizer = SpeechOptimizer::<Cpu>::new(16, 4, 1, &Default::default()).unwrap();
        update(&mut optimizer);
        let progress = AudioProgress {
            epoch: 1,
            next_index: 1,
            updates: 3,
        };
        let checkpoint =
            AudioCheckpoint::new(protocol.clone(), progress.clone(), &optimizer).unwrap();
        assert_eq!(protocol.seed_for_step(&progress).unwrap(), 162);
        let id = checkpoint.save(root.path()).unwrap();
        let restored = AudioCheckpoint::load(root.path(), &protocol)
            .unwrap()
            .unwrap();
        assert_eq!(restored.progress, progress);
        let mut resumed = restored.restore::<Cpu>(&Default::default()).unwrap();
        assert_eq!(resumed.export().unwrap(), optimizer.export().unwrap());
        update(&mut optimizer);
        update(&mut resumed);
        assert_eq!(resumed.export().unwrap(), optimizer.export().unwrap());
        assert!(!root.path().join("latest.json").exists());
        let mut changed = protocol.clone();
        changed.dataset_hash = "4".repeat(64);
        assert!(AudioCheckpoint::load(root.path(), &changed).is_err());
        fs::write(
            root.path()
                .join("audio-conditioned-v1/checkpoints")
                .join(format!("{id}.json")),
            b"{}",
        )
        .unwrap();
        assert!(AudioCheckpoint::load(root.path(), &protocol).is_err());
        assert!(
            AudioCheckpoint::new(
                protocol,
                AudioProgress {
                    epoch: 0,
                    next_index: 1,
                    updates: 0
                },
                &optimizer
            )
            .is_err()
        );
    }
}
