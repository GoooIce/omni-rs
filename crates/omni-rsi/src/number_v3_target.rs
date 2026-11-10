//! Recompute v3 speech supervision before any training checkpoint is published.
use crate::{
    fingerprint::CodecSnapshot,
    number_audio_v3::NumberAudioV3,
    number_development::{MediaBudget, SpeechTarget, read_model_config, read_target_snapshot},
    speech_audio_checkpoint::AudioProtocol,
    speech_audio_target::target_pcm,
    speech_target_format::{TARGET_SAMPLE_RATE, TargetCodeFormat},
};
use burn::tensor::backend::{AutodiffBackend, Backend};
use omni_lfm3::{
    Precision,
    codec::{CodecConfig, CodecEncoder},
};
use omni_r2t2::weights::Weights;
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

// Limit the one-target STFT working array to 64 MiB of FP32 features.
const MAX_STFT_FEATURES: usize = 16 * 1024 * 1024;

pub(crate) struct VerifiedCodec {
    pub(crate) config: CodecConfig,
    pub(crate) weights: Weights,
}

fn cancelled(stop: &AtomicBool) -> Result<(), String> {
    if stop.load(Ordering::Acquire) {
        Err("interrupted".into())
    } else {
        Ok(())
    }
}

fn check_encoding_claim(target: &SpeechTarget) -> Result<(), String> {
    if target.encoding() != "local_liquidcodec_f32"
        || target.runtime_backend() == "caller_supplied_unverified"
    {
        return Err("unverified speech target is not eligible for training".into());
    }
    Ok(())
}

fn compare_frames<B: Backend>(
    encoder: &CodecEncoder<B>,
    samples: &[f32],
    expected: &[Vec<u32>],
    eos: u32,
    books: usize,
    stop: &AtomicBool,
) -> Result<(), String> {
    cancelled(stop)?;
    if expected
        .last()
        .is_none_or(|frame| frame.len() != books || frame.iter().any(|code| *code != eos))
    {
        return Err("invalid target EOS".into());
    }
    let actual = encoder.encode(samples).map_err(|e| e.to_string())?;
    cancelled(stop)?;
    if actual.as_slice() != &expected[..expected.len() - 1] {
        return Err("target frames do not match local codec re-encoding".into());
    }
    Ok(())
}

fn verify_target<B: Backend>(
    root: &Path,
    partition: &str,
    target: &SpeechTarget,
    encoder: &CodecEncoder<B>,
    format: (u32, usize),
    budget: &mut MediaBudget,
    stop: &AtomicBool,
) -> Result<(), String> {
    let samples = target_samples(root, partition, target, budget, stop)?;
    compare_frames(encoder, &samples, target.frames(), format.0, format.1, stop)
}

fn target_samples(
    root: &Path,
    partition: &str,
    target: &SpeechTarget,
    budget: &mut MediaBudget,
    stop: &AtomicBool,
) -> Result<Vec<f32>, String> {
    check_encoding_claim(target)?;
    let bytes = read_target_snapshot(root, partition, target, budget, stop)?;
    budget.reserve_target_decode(&bytes)?;
    target_pcm(&bytes)
}

#[cfg(all(test, unix, feature = "cpu"))]
thread_local! {
    static TEST_BUDGET: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
    static TEST_REENCODED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(all(test, unix, feature = "cpu"))]
pub(crate) fn test_mock_reencoding_budget(remaining: Option<usize>) -> usize {
    TEST_BUDGET.with(|value| value.set(remaining));
    TEST_REENCODED.with(|count| count.replace(0))
}

/// Only real local encoding can return the codec snapshot used by training.
pub(crate) fn verify<B: AutodiffBackend>(
    data: &NumberAudioV3,
    root: &Path,
    model: &Path,
    protocol: &AudioProtocol,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<Option<VerifiedCodec>, String> {
    let development = data.development();
    let targets = || {
        [("train", development.train()), ("dev", development.dev())]
            .into_iter()
            .flat_map(|(partition, records)| {
                records
                    .iter()
                    .filter_map(move |row| row.speech_target().map(|target| (partition, target)))
            })
    };
    let mut count = 0usize;
    for (_, target) in targets() {
        cancelled(stop)?;
        count += 1;
        check_encoding_claim(target)?;
    }
    if count == 0 {
        return Ok(None);
    }

    let snapshot = CodecSnapshot::capture(&model.join("audio_tokenizer"), stop)?;
    let (base, full_codec) = development
        .target_model_fingerprints()
        .ok_or("missing v3 model and codec binding")?;
    if base != protocol.base_hash
        || full_codec != snapshot.fingerprint()
        || protocol.codec_hash != snapshot.target_codec_hash()
    {
        return Err("v3 model or codec changed before target verification".into());
    }
    let full_codec = snapshot.fingerprint().to_owned();
    let target_codec = snapshot.target_codec_hash().to_owned();
    let (config_bytes, shards) = snapshot.into_parts();
    let model_bytes = read_model_config(model, stop)?;
    let model_hash = format!("{:x}", Sha256::digest(&model_bytes));
    let format = TargetCodeFormat::from_bytes(&model_bytes, &config_bytes)?;
    let config = CodecConfig::from_bytes(&config_bytes).map_err(|e| e.to_string())?;
    let mut media_budget = development.media_budget();
    #[cfg(all(test, unix, feature = "cpu"))]
    if let Some(remaining) = TEST_BUDGET.with(|value| value.get()) {
        media_budget.test_remaining_bytes(remaining);
    }

    // Check identities and resource limits before constructing the encoder.
    for (partition, target) in targets() {
        cancelled(stop)?;
        if target.codec_fingerprint_sha256() != full_codec
            || target.target_codec_sha256() != target_codec
            || target.model_config_sha256() != model_hash
        {
            return Err("target does not match the captured model or codec".into());
        }
        let bytes = read_target_snapshot(root, partition, target, &mut media_budget, stop)?;
        media_budget.reserve_target_decode(&bytes)?;
        let samples = target_pcm(&bytes)?;
        cancelled(stop)?;
        let (frames, features) = config
            .encoder_workload(samples.len())
            .map_err(|e| e.to_string())?;
        if !(4..200).contains(&frames)
            || features > MAX_STFT_FEATURES
            || target.frames().len() != frames + 1
        {
            return Err("target codec frame count or STFT workload exceeds limit".into());
        }
    }

    // This test-only codec exercises the real snapshot/decode budget path and
    // can never return a VerifiedCodec or authorize training.
    #[cfg(all(test, unix, feature = "cpu"))]
    if TEST_BUDGET.with(|value| value.get()).is_some() {
        for (partition, target) in targets() {
            let samples = target_samples(root, partition, target, &mut media_budget, stop)?;
            if samples.is_empty() {
                return Err("empty mock target".into());
            }
            TEST_REENCODED.with(|count| count.set(count.get() + 1));
        }
        return Err("test mock cannot authorize training".into());
    }

    cancelled(stop)?;
    let weights = Weights::from_bytes(shards).map_err(|e| e.to_string())?;
    let encoder = CodecEncoder::<B::InnerBackend>::from_weights(
        &weights,
        config.clone(),
        Precision::F32,
        device,
    )
    .map_err(|e| e.to_string())?;
    if encoder.sample_rate() != TARGET_SAMPLE_RATE {
        return Err("invalid codec sample rate".into());
    }
    for (partition, target) in targets() {
        cancelled(stop)?;
        verify_target(
            root,
            partition,
            target,
            &encoder,
            (format.eos_code, format.num_codebooks),
            &mut media_budget,
            stop,
        )?;
    }
    drop(encoder);
    cancelled(stop)?;
    Ok(Some(VerifiedCodec { config, weights }))
}

#[cfg(all(test, unix, feature = "cpu"))]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use omni_lfm3::config::TextConfig;
    use serde::Deserialize;
    use serde_json::{Value, json};
    use std::{fs, os::unix::fs::symlink};

    #[derive(Deserialize)]
    struct Reference {
        cases: Vec<Case>,
    }
    #[derive(Deserialize)]
    struct Case {
        samples: Vec<f32>,
        frames: Vec<Vec<u32>>,
    }

    #[test]
    fn reencoding_budget_failure_preserves_new_and_resumed_checkpoint_state() {
        use crate::{
            number_v3_fixture::repeated_target_budget_fixture,
            speech_audio_round::{AudioRoundOptions, train_v3_audio_round},
        };
        use burn::backend::{Autodiff, NdArray};
        type Cpu = Autodiff<NdArray<f32>>;
        let root = repeated_target_budget_fixture(113, None, 2);
        let model = root.path().join("model");
        let wav_size = fs::metadata(root.path().join("train-target-media/answer.wav"))
            .unwrap()
            .len() as usize;
        for resume in [false, true] {
            let state = root
                .path()
                .join(if resume { "resume-state" } else { "new-state" });
            let directory = state.join("number-audio-v3/audio-conditioned-v1");
            let pointer = directory.join("latest.json");
            if resume {
                fs::create_dir_all(&directory).unwrap();
                fs::write(&pointer, b"existing checkpoint pointer").unwrap();
            }
            let options = AudioRoundOptions {
                model: &model,
                data: root.path(),
                state: &state,
                epochs: 1,
                max_updates: 1,
                resume,
            };
            test_mock_reencoding_budget(Some(wav_size * 3));
            let error =
                train_v3_audio_round::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
                    .err()
                    .unwrap();
            let encoded = test_mock_reencoding_budget(None);
            assert!(error.contains("media byte limit"), "{error}");
            assert_eq!(
                encoded, 1,
                "preflight and first reencoding must complete before exhaustion"
            );
            if resume {
                assert_eq!(fs::read(&pointer).unwrap(), b"existing checkpoint pointer");
                assert!(!directory.join("checkpoints").exists());
            } else {
                assert!(!directory.exists());
            }
        }
        let state = root.path().join("mock-success-state");
        let options = AudioRoundOptions {
            model: &model,
            data: root.path(),
            state: &state,
            epochs: 1,
            max_updates: 1,
            resume: false,
        };
        test_mock_reencoding_budget(Some(wav_size * 4));
        let error =
            train_v3_audio_round::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
                .err()
                .unwrap();
        assert_eq!(test_mock_reencoding_budget(None), 2);
        assert!(error.contains("mock cannot authorize"), "{error}");
        assert!(!state.join("number-audio-v3/audio-conditioned-v1").exists());
    }

    #[test]
    fn actual_codec_checks_the_owned_wav_frames_and_eos() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../lfm3-omni/tests/fixtures/codec");
        let reference: Reference =
            serde_json::from_slice(&fs::read(fixture.join("expected-encode.json")).unwrap())
                .unwrap();
        let backbone: TextConfig =
            serde_json::from_slice(&fs::read(fixture.join("backbone.json")).unwrap()).unwrap();
        let config = CodecConfig::load(&fixture)
            .unwrap()
            .with_backbone(backbone)
            .unwrap();
        let weights =
            Weights::from_bytes(vec![fs::read(fixture.join("synthetic.bin")).unwrap()]).unwrap();
        let encoder = CodecEncoder::<NdArray<f32>>::from_weights(
            &weights,
            config.clone(),
            Precision::F32,
            &Default::default(),
        )
        .unwrap();
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("train-target-media")).unwrap();
        let wav = root.path().join("train-target-media/answer.wav");
        let write_wav = |samples: &[f32]| {
            let mut writer = hound::WavWriter::create(
                &wav,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 24_000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();
            for sample in samples {
                writer.write_sample(*sample).unwrap();
            }
            writer.finalize().unwrap();
        };
        let case = &reference.cases[1];
        write_wav(&case.samples);
        let mut frames = case.frames.clone();
        frames.push(vec![12; 3]);
        let digest = |path: &Path| format!("{:x}", Sha256::digest(fs::read(path).unwrap()));
        let mut value = json!({"frames":frames,"target_codec_sha256":"a".repeat(64),
            "target_audio_sha256":digest(&wav),"codec_fingerprint_sha256":"b".repeat(64),
            "model_config_sha256":"c".repeat(64),"sample_rate":24000,
            "encoding":"local_liquidcodec_f32","quality_evidence":"not_assessed",
            "runtime_backend":"different-preparation-backend","preparation_sha256":"d".repeat(64),"audio_file":"answer.wav",
            "source":{"kind":"licensed-synthetic","source":"fixture","license":"CC0-1.0"},
            "source_group":{"speaker_sha256":"1".repeat(64),"session_sha256":"2".repeat(64),"recording_sha256":"3".repeat(64),"family_sha256":"4".repeat(64)}});
        let target = |v: &Value| serde_json::from_value::<SpeechTarget>(v.clone()).unwrap();
        let stop = AtomicBool::new(false);
        let mut budget = MediaBudget::default();
        assert_eq!(
            config.encoder_workload(case.samples.len()).unwrap().0,
            case.frames.len()
        );
        verify_target(
            root.path(),
            "train",
            &target(&value),
            &encoder,
            (12, 3),
            &mut budget,
            &stop,
        )
        .unwrap();

        value["frames"][0][0] = ((case.frames[0][0] + 1) % 12).into();
        assert!(
            verify_target(
                root.path(),
                "train",
                &target(&value),
                &encoder,
                (12, 3),
                &mut budget,
                &stop
            )
            .unwrap_err()
            .contains("do not match")
        );
        value["frames"] = json!(frames);
        value["frames"][case.frames.len()][0] = 0.into();
        assert!(
            verify_target(
                root.path(),
                "train",
                &target(&value),
                &encoder,
                (12, 3),
                &mut budget,
                &stop
            )
            .unwrap_err()
            .contains("EOS")
        );
        value["frames"] = json!(frames);

        value["frames"][0].as_array_mut().unwrap().pop();
        assert!(
            verify_target(
                root.path(),
                "train",
                &target(&value),
                &encoder,
                (12, 3),
                &mut budget,
                &stop
            )
            .unwrap_err()
            .contains("do not match")
        );
        value["frames"] = json!(frames);

        write_wav(&reference.cases[2].samples);
        assert!(
            verify_target(
                root.path(),
                "train",
                &target(&value),
                &encoder,
                (12, 3),
                &mut budget,
                &stop
            )
            .unwrap_err()
            .contains("changed")
        );
        value["target_audio_sha256"] = digest(&wav).into();
        assert!(
            verify_target(
                root.path(),
                "train",
                &target(&value),
                &encoder,
                (12, 3),
                &mut budget,
                &stop
            )
            .unwrap_err()
            .contains("do not match")
        );
        let replacement = root.path().join("replacement.wav");
        fs::copy(&wav, &replacement).unwrap();
        fs::remove_file(&wav).unwrap();
        symlink(&replacement, &wav).unwrap();
        assert!(
            verify_target(
                root.path(),
                "train",
                &target(&value),
                &encoder,
                (12, 3),
                &mut budget,
                &stop
            )
            .is_err()
        );
        stop.store(true, Ordering::Release);
        assert_eq!(
            verify_target(
                root.path(),
                "train",
                &target(&value),
                &encoder,
                (12, 3),
                &mut budget,
                &stop
            )
            .unwrap_err(),
            "interrupted"
        );
    }
}
