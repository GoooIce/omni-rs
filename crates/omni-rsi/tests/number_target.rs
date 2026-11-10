use omni_rsi::speech_audio_target::encode_target_with;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

fn fixture() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    let codec = model.join("audio_tokenizer");
    fs::create_dir_all(&codec).unwrap();
    fs::write(
        model.join("config.json"),
        br#"{"num_codebooks":12,"codebook_size":2017,"audio_eos_code":2016,"text_config":{"hidden_size":16}}"#,
    )
    .unwrap();
    for name in [
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ] {
        fs::write(model.join(name), b"fixture").unwrap();
    }
    fs::write(
        codec.join("config.json"),
        br#"{"n_q":12,"fsq_levels":[8,7,6,6],"sampling_rate":24000}"#,
    )
    .unwrap();
    fs::write(codec.join("model.safetensors"), b"fixture-codec").unwrap();
    let wav = root.path().join("answer.wav");
    write_pcm(&wav);
    (root, model, wav)
}

fn write_pcm(path: &Path) {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for sample in [0_i16, 8192, -8192, 4096] {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
}

#[test]
fn encoded_target_contains_the_original_hash_codec_binding_and_one_eos() {
    let (_root, model, wav) = fixture();
    let expected_wav_hash = format!("{:x}", Sha256::digest(fs::read(&wav).unwrap()));
    let result = encode_target_with(&model, &wav, &AtomicBool::new(false), |_, samples| {
        assert_eq!(samples, [0.0, 0.25, -0.25, 0.125]);
        Ok(vec![vec![17; 12], vec![42; 12], vec![8; 12], vec![10; 12]])
    })
    .unwrap();
    assert_eq!(result.target_audio_sha256(), expected_wav_hash);
    assert_eq!(
        result.target_codec_sha256(),
        "ffcc069676f8971f7b94d8f661463b81afec83836ae80285edd022524400477c"
    );
    assert_eq!(result.frames().len(), 5);
    assert_eq!(result.frames()[0], vec![17; 12]);
    assert_eq!(result.frames()[4], vec![2016; 12]);
    assert!(
        result.frames()[..4]
            .iter()
            .flatten()
            .all(|code| *code < 2016)
    );
    assert_eq!(result.encoding(), "caller_supplied_unverified");
    assert_eq!(result.quality_evidence(), "not_assessed");
    assert_eq!(result.runtime_backend(), "caller_supplied_unverified");
    assert_eq!(result.preparation_sha256().len(), 64);
}

#[test]
fn unstable_codec_output_is_rejected() {
    let (_root, model, wav) = fixture();
    let mut calls = 0_u32;
    let error = encode_target_with(&model, &wav, &AtomicBool::new(false), |_, _| {
        calls += 1;
        Ok(vec![vec![calls; 12]; 4])
    })
    .unwrap_err();
    assert!(error.contains("unstable"), "{error}");
}

#[test]
fn cancellation_before_or_during_encoding_returns_no_target() {
    let (_root, model, wav) = fixture();
    let stop = AtomicBool::new(true);
    let error = encode_target_with(&model, &wav, &stop, |_, _| {
        panic!("encoder must not run after cancellation")
    })
    .unwrap_err();
    assert!(error.contains("interrupted"));

    stop.store(false, Ordering::Release);
    let error = encode_target_with(&model, &wav, &stop, |_, _| {
        stop.store(true, Ordering::Release);
        Ok(vec![vec![1; 12]; 4])
    })
    .unwrap_err();
    assert!(error.contains("interrupted"));
}

#[test]
fn invalid_or_out_of_bounds_wav_is_rejected_before_encoding() {
    let (_root, model, wav) = fixture();
    let stop = AtomicBool::new(false);
    let reject = || {
        assert!(
            encode_target_with(&model, &wav, &stop, |_, _| {
                panic!("invalid input reached encoder")
            })
            .is_err()
        );
    };

    fs::write(&wav, b"not a WAV").unwrap();
    reject();
    for (channels, rate, sample_count) in [
        (1, 24_000, 0),
        (2, 24_000, 4),
        (1, 16_000, 4),
        (1, 24_000, 480_001),
    ] {
        let mut writer = hound::WavWriter::create(
            &wav,
            hound::WavSpec {
                channels,
                sample_rate: rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..sample_count {
            writer.write_sample(1_i16).unwrap();
        }
        writer.finalize().unwrap();
        reject();
    }
    for value in [f32::NAN, f32::INFINITY, 1.01] {
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
        writer.write_sample(value).unwrap();
        writer.finalize().unwrap();
        reject();
    }
    write_pcm(&wav);
    let mut truncated = fs::read(&wav).unwrap();
    truncated.pop();
    fs::write(&wav, truncated).unwrap();
    reject();
    fs::File::create(&wav)
        .unwrap()
        .set_len(32 * 1024 * 1024 + 1)
        .unwrap();
    reject();
}

#[test]
fn source_wav_and_each_codec_binding_must_stay_unchanged() {
    for changed in ["wav", "codec_config", "codec_weights", "model_config"] {
        let (_root, model, wav) = fixture();
        let path = match changed {
            "wav" => wav.clone(),
            "codec_config" => model.join("audio_tokenizer/config.json"),
            "codec_weights" => model.join("audio_tokenizer/model.safetensors"),
            _ => model.join("config.json"),
        };
        let mut modified = false;
        let error = encode_target_with(&model, &wav, &AtomicBool::new(false), |_, _| {
            if !modified {
                fs::write(&path, b"changed during encoding").unwrap();
                modified = true;
            }
            Ok(vec![vec![1; 12]; 4])
        })
        .unwrap_err();
        assert!(error.contains("changed"), "{changed}: {error}");
    }
}

#[test]
fn changing_a_codec_shard_during_encoding_is_rejected() {
    let (_root, model, wav) = fixture();
    let codec = model.join("audio_tokenizer");
    let shard = codec.join("part.safetensors");
    fs::write(&shard, b"first shard").unwrap();
    fs::write(
        codec.join("model.safetensors.index.json"),
        br#"{"weight_map":{"encoder.weight":"part.safetensors"}}"#,
    )
    .unwrap();
    let mut modified = false;
    let error = encode_target_with(&model, &wav, &AtomicBool::new(false), |_, _| {
        if !modified {
            fs::write(&shard, b"changed shard").unwrap();
            modified = true;
        }
        Ok(vec![vec![1; 12]; 4])
    })
    .unwrap_err();
    assert!(error.contains("changed"), "{error}");
}

#[test]
fn malformed_codec_frames_or_embedded_end_markers_are_rejected() {
    let (_root, model, wav) = fixture();
    let mut short_frame = vec![vec![1; 12]; 4];
    short_frame[0].pop();
    let mut premature_end = vec![vec![1; 12]; 4];
    premature_end[1][2] = 2016;
    let mut out_of_range = vec![vec![1; 12]; 4];
    out_of_range[1][2] = 2017;
    for frames in [
        vec![],
        vec![vec![1; 12]; 3],
        vec![vec![1; 12]; 200],
        short_frame,
        premature_end,
        out_of_range,
    ] {
        let error = encode_target_with(&model, &wav, &AtomicBool::new(false), |_, _| {
            Ok(frames.clone())
        })
        .unwrap_err();
        assert!(error.contains("frames"), "{error}");
    }
}

#[test]
fn incompatible_model_and_codec_formats_are_rejected_before_encoding() {
    for (model_config, codec_config) in [
        (
            r#"{"num_codebooks":11,"codebook_size":2017,"audio_eos_code":2016}"#,
            r#"{"n_q":12,"fsq_levels":[8,7,6,6],"sampling_rate":24000}"#,
        ),
        (
            r#"{"num_codebooks":12,"codebook_size":2017,"audio_eos_code":2015}"#,
            r#"{"n_q":12,"fsq_levels":[8,7,6,6],"sampling_rate":24000}"#,
        ),
        (
            r#"{"num_codebooks":12,"codebook_size":2016,"audio_eos_code":2016}"#,
            r#"{"n_q":12,"fsq_levels":[8,7,6,6],"sampling_rate":24000}"#,
        ),
        (
            r#"{"num_codebooks":12,"codebook_size":2017,"audio_eos_code":2016}"#,
            r#"{"n_q":12,"fsq_levels":[8,7,0,6],"sampling_rate":24000}"#,
        ),
        (
            r#"{"num_codebooks":12,"codebook_size":2017,"audio_eos_code":2016}"#,
            r#"{"n_q":12,"fsq_levels":[8,7,6,6],"sampling_rate":16000}"#,
        ),
    ] {
        let (_root, model, wav) = fixture();
        fs::write(model.join("config.json"), model_config).unwrap();
        fs::write(model.join("audio_tokenizer/config.json"), codec_config).unwrap();
        let error = encode_target_with(&model, &wav, &AtomicBool::new(false), |_, _| {
            panic!("incompatible format reached encoder")
        })
        .unwrap_err();
        assert!(error.contains("codec"), "{error}");
    }
}

#[test]
fn target_fields_fit_rule_plans_and_the_training_codec_binding() {
    use omni_rsi::{
        speech_audio_checkpoint::AudioProtocol,
        speech_audio_data::{AudioDataset, generate_rule_manifest},
    };
    use serde_json::json;

    for sharded in [false, true] {
        let (root, model, wav) = fixture();
        if sharded {
            let codec = model.join("audio_tokenizer");
            fs::write(codec.join("part.safetensors"), b"fixture-shard").unwrap();
            fs::write(
                codec.join("model.safetensors.index.json"),
                br#"{"weight_map":{"encoder.weight":"part.safetensors"}}"#,
            )
            .unwrap();
        }
        let stop = AtomicBool::new(false);
        let target =
            encode_target_with(&model, &wav, &stop, |_, _| Ok(vec![vec![1; 12]; 4])).unwrap();
        let expected_codec = target.target_codec_sha256().to_owned();
        let data = root.path().join("data");
        fs::create_dir(&data).unwrap();
        let mut records = Vec::new();
        for (index, split, number, spoken) in [
            (1_i16, "train", 12, "十二"),
            (2, "dev", 15, "十五"),
            (3, "test", 18, "十八"),
        ] {
            let name = format!("{split}.wav");
            let mut writer = hound::WavWriter::create(
                data.join(&name),
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 16_000,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )
            .unwrap();
            for sample in [0_i16, index * 4096, -index * 4096, 0] {
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
            records.push(json!({"id":split, "split":split, "input_audio":name, "spoken_number":number,
                "authorization":{"kind":"licensed-synthetic","source":"pipeline fixture","license":"CC0-1.0"},
                "provenance":{"speaker_id":format!("speaker-{split}"),"session_id":format!("session-{split}"),
                    "recording_id":format!("recording-{split}"),"family_id":format!("family-{split}"),
                    "teacher_sha256":"1".repeat(64),"verifier_sha256":"2".repeat(64),
                    "verified_text":spoken,"created_at":"2026-11-10T00:00:00Z"}}));
        }
        for (key, value) in serde_json::to_value(target).unwrap().as_object().unwrap() {
            records[0][key] = value.clone();
        }
        generate_rule_manifest(&data, &json!({"version":1,"records":records}), &stop).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(data.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(
            manifest["records"][0]["encoding"],
            "caller_supplied_unverified"
        );
        assert_eq!(manifest["records"][0]["quality_evidence"], "not_assessed");
        assert_eq!(manifest["records"][0]["frames"][4], json!(vec![2016; 12]));
        let dataset = AudioDataset::load(&data, &stop).unwrap();
        let protocol =
            AudioProtocol::for_dataset(&model, &dataset, "fixture".into(), &stop).unwrap();
        assert_eq!(protocol.codec_hash, expected_codec, "sharded={sharded}");
    }
}

#[cfg(feature = "cpu")]
#[test]
fn official_entry_does_not_accept_a_hash_as_a_loaded_codec() {
    use burn::backend::NdArray;
    use omni_rsi::speech_audio_target::encode_target;
    let (_root, model, wav) = fixture();
    let result =
        encode_target::<NdArray<f32>>(&model, &wav, &Default::default(), &AtomicBool::new(false));
    assert!(
        result.is_err(),
        "a fake checkpoint was reported as an executed codec"
    );
}

#[cfg(feature = "cpu")]
#[test]
fn float_wav_reaches_the_independently_referenced_fixture_codec() {
    use burn::backend::NdArray;
    use omni_lfm3::{
        Precision,
        codec::{CodecConfig, CodecEncoder},
        config::TextConfig,
    };
    use omni_r2t2::weights::Weights;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Reference {
        cases: Vec<Case>,
    }
    #[derive(Deserialize)]
    struct Case {
        samples: Vec<f32>,
        frames: Vec<Vec<u32>>,
    }
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../lfm3-omni/tests/fixtures/codec");
    let reference: Reference =
        serde_json::from_slice(&fs::read(source.join("expected-encode.json")).unwrap()).unwrap();
    let backbone: TextConfig =
        serde_json::from_slice(&fs::read(source.join("backbone.json")).unwrap()).unwrap();
    let config = CodecConfig::load(&source)
        .unwrap()
        .with_backbone(backbone)
        .unwrap();
    let weights = Weights::open_file(&source.join("synthetic.bin")).unwrap();
    let encoder = CodecEncoder::<NdArray<f32>>::from_weights(
        &weights,
        config,
        Precision::F32,
        &Default::default(),
    )
    .unwrap();
    let (_root, model, wav) = fixture();
    fs::copy(
        source.join("config.json"),
        model.join("audio_tokenizer/config.json"),
    )
    .unwrap();
    fs::copy(
        source.join("synthetic.bin"),
        model.join("audio_tokenizer/model.safetensors"),
    )
    .unwrap();
    fs::write(
        model.join("config.json"),
        br#"{"num_codebooks":3,"codebook_size":13,"audio_eos_code":12}"#,
    )
    .unwrap();
    let case = &reference.cases[1];
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
    for sample in &case.samples {
        writer.write_sample(*sample).unwrap();
    }
    writer.finalize().unwrap();
    let result = encode_target_with(&model, &wav, &AtomicBool::new(false), |_, samples| {
        encoder.encode(samples).map_err(|error| error.to_string())
    })
    .unwrap();
    assert_eq!(&result.frames()[..result.frames().len() - 1], case.frames);
    assert_eq!(result.frames().last().unwrap(), &vec![12; 3]);
    assert_eq!(result.encoding(), "caller_supplied_unverified");
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires a local official LFM3 checkpoint and independent codec reference"]
fn official_codec_target_matches_independent_reference() {
    use burn::backend::NdArray;
    use omni_rsi::speech_audio_target::encode_target;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Reference {
        samples: Vec<f32>,
        frames: Vec<Vec<u32>>,
    }
    let model = std::path::PathBuf::from(std::env::var("LFM3_MODEL").expect("set LFM3_MODEL"));
    let reference =
        std::env::var("LFM3_CODEC_ENCODE_REFERENCE").expect("set LFM3_CODEC_ENCODE_REFERENCE");
    let reference: Reference = serde_json::from_slice(&fs::read(reference).unwrap()).unwrap();
    let root = tempfile::tempdir().unwrap();
    let wav = root.path().join("reference.wav");
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
    for sample in &reference.samples {
        writer.write_sample(*sample).unwrap();
    }
    writer.finalize().unwrap();
    let result =
        encode_target::<NdArray<f32>>(&model, &wav, &Default::default(), &AtomicBool::new(false))
            .unwrap();
    assert_eq!(
        &result.frames()[..result.frames().len() - 1],
        reference.frames
    );
    assert_eq!(result.frames().last().unwrap(), &vec![2016; 12]);
    assert_eq!(result.encoding(), "local_liquidcodec_f32");
    assert_eq!(result.quality_evidence(), "not_assessed");
}
