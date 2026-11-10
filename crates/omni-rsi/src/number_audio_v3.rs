//! Bind separated v3 train/dev data to the existing audio training primitive.
//! This is data admission, not evidence of quality or verified authorization.

use crate::{
    fingerprint::codec_fingerprint,
    number_development::{DevelopmentData, DevelopmentRecord},
    speech_audio::AudioAnswer,
    speech_audio_checkpoint::{
        AudioProtocol, audio_codec_hash, audio_implementation_hash_for_backend, hash_audio_sources,
    },
};

pub(crate) fn v3_audio_sources() -> Vec<(&'static str, &'static [u8])> {
    vec![
        (
            "number_v3_target.rs",
            include_bytes!("number_v3_target.rs").as_slice(),
        ),
        (
            "number_audio_v3.rs",
            include_bytes!("number_audio_v3.rs").as_slice(),
        ),
        (
            "number_development.rs",
            include_bytes!("number_development.rs").as_slice(),
        ),
        (
            "number_blind_holdout.rs",
            include_bytes!("number_blind_holdout.rs").as_slice(),
        ),
        (
            "speech_audio_target.rs",
            include_bytes!("speech_audio_target.rs").as_slice(),
        ),
        (
            "speech_target_format.rs",
            include_bytes!("speech_target_format.rs").as_slice(),
        ),
    ]
}

pub(crate) fn v3_audio_implementation_hash_for_backend(backend: &str) -> String {
    hash_audio_sources(
        b"omni-audio-v3-v2\0",
        &audio_implementation_hash_for_backend(backend),
        &v3_audio_sources(),
    )
}
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

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
        crate::number_development::check_artifact_tree(model, stop)?;
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
            version: 2,
            base_hash,
            codec_hash,
            dataset_hash: dataset.digest().into(),
            implementation_hash: v3_audio_implementation_hash_for_backend(&backend),
            backend,
            width,
            train_records: dataset.len(),
            mask_seed: 159,
        })
    }
}

const INSTRUCTION: &str = "把听到的数字加一，用中文回答。";

pub struct NumberAudioV3 {
    data: DevelopmentData,
    answers: Vec<String>,
    digest: String,
}

impl NumberAudioV3 {
    /// Loads only train/dev and public holdout metadata; no holdout permit exists here.
    pub fn load(root: &Path, model: &Path, stop: &AtomicBool) -> Result<Self, String> {
        let data = DevelopmentData::load_with_model(root, model, stop)?;
        let answers = data
            .train()
            .iter()
            .map(|record| {
                crate::speech_audio_data::number_text(record.answer())
                    .ok_or_else(|| "v3 rule answer outside 0..=100".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let digest = data_identity(&data, stop)?;
        Ok(Self {
            data,
            answers,
            digest,
        })
    }

    pub fn len(&self) -> usize {
        self.data.train().len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.train().is_empty()
    }
    pub fn digest(&self) -> &str {
        &self.digest
    }
    pub fn development(&self) -> &DevelopmentData {
        &self.data
    }

    /// Build exactly the object consumed by the native `train_step`.
    pub fn example(&self, index: usize) -> Result<AudioAnswer<'_>, String> {
        let row = self
            .data
            .train()
            .get(index)
            .ok_or("missing v3 training record")?;
        let answer = self
            .answers
            .get(index)
            .ok_or("missing v3 training answer")?;
        match row.speech_target() {
            Some(target) => {
                AudioAnswer::new(row.samples(), 16_000, INSTRUCTION, answer, target.frames())
            }
            None => AudioAnswer::new_text_only(row.samples(), 16_000, INSTRUCTION, answer),
        }
    }
}

fn feed(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn bind_record(
    hash: &mut Sha256,
    record: &DevelopmentRecord,
    stop: &AtomicBool,
) -> Result<(), String> {
    feed(hash, record.id().as_bytes());
    feed(hash, &record.spoken_number().to_le_bytes());
    feed(hash, &record.answer().to_le_bytes());
    feed(
        hash,
        &serde_json::to_vec(record.source()).map_err(|e| e.to_string())?,
    );
    feed(
        hash,
        &serde_json::to_vec(record.declaration()).map_err(|e| e.to_string())?,
    );
    feed(
        hash,
        &serde_json::to_vec(&record.speech_target()).map_err(|e| e.to_string())?,
    );
    hash.update((record.samples().len() as u64).to_le_bytes());
    for (index, sample) in record.samples().iter().enumerate() {
        if index % 4096 == 0 && stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        hash.update(sample.to_bits().to_le_bytes());
    }
    Ok(())
}

fn data_identity(data: &DevelopmentData, stop: &AtomicBool) -> Result<String, String> {
    let mut hash = Sha256::new();
    hash.update(b"omni-rsi-number-v3-training-dataset-v1\0");
    for bytes in [
        include_bytes!("number_audio_v3.rs").as_slice(),
        include_bytes!("number_development.rs").as_slice(),
        include_bytes!("number_blind_holdout.rs").as_slice(),
        include_bytes!("speech_audio_target.rs").as_slice(),
        include_bytes!("speech_target_format.rs").as_slice(),
        include_bytes!("fingerprint.rs").as_slice(),
    ] {
        feed(&mut hash, bytes);
    }
    for value in data.manifest_fingerprints() {
        feed(&mut hash, value.as_bytes());
    }
    feed(&mut hash, data.blind_fingerprint().as_bytes());
    for (split, records) in [
        (b"train".as_slice(), data.train()),
        (b"dev".as_slice(), data.dev()),
    ] {
        feed(&mut hash, split);
        hash.update((records.len() as u64).to_le_bytes());
        for record in records {
            bind_record(&mut hash, record, stop)?;
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}
