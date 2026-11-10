//! Checkpoints for audio-conditioned training, isolated from self-reading.
use crate::{
    fingerprint::codec_fingerprint,
    number_audio_v3::NumberAudioV3,
    speech_audio_data::AudioDataset,
    speech_checkpoint::{atomic_json, implementation_hash},
    speech_data::file_hash,
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
    if protocol.implementation_hash != audio_implementation_hash() {
        return Err("audio checkpoint implementation changed; existing state is untouched; use its original binary".into());
    }
    if protocol.version != 1
        || [
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

/// Hash every part of the native audio-conditioned objective and prompt path.
pub fn audio_implementation_hash() -> String {
    let mut hash = Sha256::new();
    for bytes in [
        implementation_hash().as_bytes(),
        include_bytes!("speech_audio.rs").as_slice(),
        include_bytes!("speech_audio_checkpoint.rs").as_slice(),
        include_bytes!("speech_audio_data.rs").as_slice(),
        include_bytes!("speech_audio_round.rs").as_slice(),
        include_bytes!("audio_main.rs").as_slice(),
        include_bytes!("../Cargo.toml").as_slice(),
        include_bytes!("../../lfm3-omni/src/media_io.rs").as_slice(),
        include_bytes!("../../../src/input.rs").as_slice(),
        include_bytes!("../../lfm3-omni/src/audio.rs").as_slice(),
        include_bytes!("../../lfm3-omni/src/audio_encoder.rs").as_slice(),
        include_bytes!("../../lfm3-omni/src/chat.rs").as_slice(),
        include_bytes!("../../lfm3-omni/src/media.rs").as_slice(),
        include_bytes!("../../lfm3-omni/src/prepared.rs").as_slice(),
    ] {
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    format!("{:x}", hash.finalize())
}

impl AudioProtocol {
    /// Build a checkpoint protocol for the isolated v3 training snapshot.
    pub fn for_v3(
        model: &Path,
        dataset: &NumberAudioV3,
        backend: String,
        stop: &AtomicBool,
    ) -> Result<Self, String> {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        let base_hash =
            omni_lfm3::speech_adapter::source_fingerprint(model).map_err(|e| e.to_string())?;
        let full_codec_hash = codec_fingerprint(&model.join("audio_tokenizer"), stop)?;
        let codec_hash = audio_codec_hash(model, stop)?;
        if dataset
            .development()
            .target_model_fingerprints()
            .is_some_and(|(base, codec)| base != base_hash || codec != full_codec_hash)
        {
            return Err("v3 target model or codec changed after data loading".into());
        }
        for record in dataset
            .development()
            .train()
            .iter()
            .chain(dataset.development().dev())
        {
            if record
                .speech_target()
                .is_some_and(|target| target.target_codec_sha256() != codec_hash)
            {
                return Err("v3 target codec does not match the current model".into());
            }
        }
        let config: serde_json::Value = serde_json::from_slice(
            &fs::read(model.join("config.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
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
            implementation_hash: audio_implementation_hash(),
            backend,
            width,
            train_records: dataset.len(),
            mask_seed: 159,
        })
    }

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
        let base_hash = omni_lfm3::speech_adapter::source_fingerprint(model)
            .map_err(|error| error.to_string())?;
        let codec_hash = audio_codec_hash(model, stop)?;
        for index in 0..dataset.len() {
            let record = dataset.record(index).ok_or("missing audio record")?;
            if record
                .target_codec_sha256
                .as_ref()
                .is_some_and(|hash| hash != &codec_hash)
            {
                return Err("audio target codec does not match the current model".into());
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
            implementation_hash: audio_implementation_hash(),
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
fn audio_codec_hash(model: &Path, stop: &AtomicBool) -> Result<String, String> {
    let codec = model.join("audio_tokenizer");
    if codec.join("model.safetensors.index.json").is_file() {
        codec_fingerprint(&codec, stop)
    } else {
        file_hash(&codec.join("model.safetensors"))
    }
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

    /// Reject old formats, changed source/data/model, and damaged checkpoint files.
    pub fn load(state: &Path, expected: &AudioProtocol) -> Result<Option<Self>, String> {
        validate(expected, &AudioProgress::default())?;
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
            .ok_or("invalid audio checkpoint pointer")?;
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
            implementation_hash: audio_implementation_hash(),
            backend: Cpu::name(&Default::default()),
            width: 16,
            train_records: 2,
            mask_seed: 159,
        }
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
