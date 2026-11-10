use omni_rsi::{
    speech_data::file_hash,
    speech_tone::{ToneDataset, ToneEvaluationPlan, ToneInput},
    speech_tone_input::{ToneInputLimits, prepare_development_inputs},
};
use serde_json::{Value, json};
use std::{path::Path, sync::atomic::AtomicBool};

fn record(root: &Path, split: &str, id: &str, samples: Option<&[i16]>) -> Value {
    let path = format!("{split}/{id}.wav");
    let hash = if let Some(samples) = samples {
        std::fs::create_dir_all(root.join(split)).unwrap();
        let mut wav = hound::WavWriter::create(
            root.join(&path),
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for sample in samples {
            wav.write_sample(*sample).unwrap();
        }
        wav.finalize().unwrap();
        file_hash(&root.join(&path)).unwrap()
    } else {
        "f".repeat(64)
    };
    json!({
        "id":id, "split":split, "audio":{"path":path,"sha256":hash},
        "text":format!("测试文本{id}"),"text_family":format!("text-{id}"),
        "speaker_id":format!("speaker-{id}"),"session_id":format!("session-{id}"),
        "derived_family":format!("family-{id}"),
        "source":{"name":"synthetic","license":"fixture","kind":"synthetic"},
        "annotations":[{"annotator_id":"rater","source":"fixture",
            "decision":{"kind":"label","label":"steady"}}]
    })
}

fn dataset(records: Vec<Value>) -> ToneDataset {
    serde_json::from_value(json!({"version":1, "labels":["steady"], "records":records})).unwrap()
}

fn plan(split: &str, cases: Vec<Value>) -> ToneEvaluationPlan {
    serde_json::from_value(
        json!({"run_id":"synthetic-run", "split":split, "cases":cases, "pairs":[]}),
    )
    .unwrap()
}

#[cfg(unix)]
fn reference_pcm_hash(samples: &[f32]) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(b"omni-rsi/tone-pcm/v1\0");
    hash.update(16_000_u32.to_be_bytes());
    hash.update((samples.len() as u64).to_be_bytes());
    for sample in samples {
        hash.update((if *sample == 0.0 { 0 } else { sample.to_bits() }).to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[cfg(unix)]
#[test]
fn development_inputs_materialize_masks_without_opening_holdout() {
    let root = tempfile::tempdir().unwrap();
    let samples: Vec<_> = (0..320)
        .map(|n| if n % 2 == 0 { 8000 } else { -4000 })
        .collect();
    let development = record(root.path(), "dev", "dev-a", Some(&samples));
    let holdout = record(root.path(), "holdout", "secret", None);
    let data = dataset(vec![development.clone(), holdout]);
    let mut plan = plan(
        "dev",
        vec![
            json!({"id":"full","record_id":"dev-a","input":"full"}),
            json!({"id":"audio-mask","record_id":"dev-a","input":"audio_masked"}),
            json!({"id":"text-mask","record_id":"dev-a","input":"text_masked"}),
        ],
    );
    plan.pairs = serde_json::from_value(json!([
        {"id":"audio-pair","kind":"audio_masked","original":"full","variant":"audio-mask"},
        {"id":"text-pair","kind":"text_masked","original":"full","variant":"text-mask"}
    ]))
    .unwrap();
    let prepared = prepare_development_inputs(
        &data,
        &plan,
        root.path(),
        &ToneInputLimits::default(),
        &AtomicBool::new(false),
    )
    .unwrap();
    let inputs = prepared.cases();
    assert_eq!(inputs.len(), 3);
    let full = &inputs[0];
    let audio = &inputs[1];
    let text = &inputs[2];
    assert_eq!(full.condition(), ToneInput::Full);
    assert_eq!(audio.condition(), ToneInput::AudioMasked);
    assert_eq!(text.condition(), ToneInput::TextMasked);
    assert_eq!(full.sample_rate(), 16_000);
    assert_eq!(full.samples().len(), 320);
    assert_eq!(audio.samples().len(), full.samples().len());
    assert_eq!(text.samples(), full.samples());
    assert!(full.samples().iter().any(|sample| *sample != 0.0));
    assert!(audio.samples().iter().all(|sample| *sample == 0.0));
    assert_eq!(full.instruction(), text.instruction());
    assert_eq!(full.transcript(), Some("测试文本dev-a"));
    assert_eq!(audio.transcript(), full.transcript());
    assert_eq!(text.transcript(), None);
    assert_eq!(
        full.source_audio_sha256(),
        development["audio"]["sha256"].as_str().unwrap()
    );
    assert_ne!(full.input_sha256(), audio.input_sha256());
    assert_ne!(full.input_sha256(), text.input_sha256());
    assert_eq!(full.pcm_sha256(), text.pcm_sha256());
    assert_ne!(full.pcm_sha256(), audio.pcm_sha256());
    assert_eq!(prepared.holdout_pcm_independence(), "unverified");
    assert_eq!(prepared.execution_provenance(), "not_run");
    assert!(!format!("{prepared:?}").contains("测试文本dev-a"));
}

#[test]
fn preflight_rejects_a_holdout_case_before_opening_development_audio() {
    let root = tempfile::tempdir().unwrap();
    let mut development = record(root.path(), "dev", "dev-a", None);
    development["audio"]["sha256"] = json!("a".repeat(64));
    let holdout = record(root.path(), "holdout", "secret", None);
    let data = dataset(vec![development, holdout]);
    let plan = plan(
        "dev",
        vec![
            json!({"id":"dev-first","record_id":"dev-a","input":"full"}),
            json!({"id":"secret-second","record_id":"secret","input":"full"}),
        ],
    );
    let error = prepare_development_inputs(
        &data,
        &plan,
        root.path(),
        &ToneInputLimits::default(),
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(
        error.contains("holdout"),
        "unexpected error before holdout preflight: {error}"
    );
    assert!(!root.path().join("holdout").exists());
}

#[test]
fn preflight_rejects_public_group_collisions_before_any_audio() {
    let root = tempfile::tempdir().unwrap();
    for field in [
        "speaker_id",
        "session_id",
        "derived_family",
        "text_family",
        "audio",
    ] {
        let mut development = record(root.path(), "dev", "dev-a", None);
        development["audio"]["sha256"] = json!("a".repeat(64));
        let mut holdout = record(root.path(), "holdout", "secret", None);
        if field == "audio" {
            holdout["audio"]["sha256"] = development["audio"]["sha256"].clone();
        } else {
            holdout[field] = development[field].clone();
        }
        let data = dataset(vec![development, holdout]);
        let plan = plan(
            "dev",
            vec![json!({"id":"dev-first","record_id":"dev-a","input":"full"})],
        );
        let error = prepare_development_inputs(
            &data,
            &plan,
            root.path(),
            &ToneInputLimits::default(),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(
            error.contains("cross"),
            "{field} reached audio before metadata rejection: {error}"
        );
    }
}

#[test]
fn malformed_declarations_are_rejected_before_the_root_is_opened() {
    let root = tempfile::tempdir().unwrap();
    let mut development = record(root.path(), "dev", "dev-a", None);
    development["audio"]["sha256"] = json!("a".repeat(64));
    let mut training = record(root.path(), "train", "train-a", None);
    training["audio"]["sha256"] = json!("b".repeat(64));
    let holdout = record(root.path(), "holdout", "secret", None);
    let base = json!({"version":1,"labels":["steady"],"records":[development,training,holdout]});
    let mut variants = Vec::new();
    let mut bad = base.clone();
    bad["version"] = json!(2);
    variants.push(bad);
    let mut bad = base.clone();
    bad["labels"] = json!(["steady", "steady"]);
    variants.push(bad);
    let mut bad = base.clone();
    bad["records"][1]["id"] = bad["records"][0]["id"].clone();
    variants.push(bad);
    let mut bad = base.clone();
    bad["records"][0]["audio"]["path"] = json!("dev/../holdout/secret.wav");
    variants.push(bad);
    let mut bad = base.clone();
    bad["records"][2]["audio"]["path"] = json!("/private/secret.wav");
    variants.push(bad);
    let mut bad = base.clone();
    bad["records"][0]["audio"]["sha256"] = json!("not-a-hash");
    variants.push(bad);
    let mut bad = base;
    bad["records"][0]["text"] = json!("ＴＥＳＴ！");
    bad["records"][1]["text"] = json!("test");
    variants.push(bad);
    let plan = plan(
        "dev",
        vec![json!({"id":"dev-first","record_id":"dev-a","input":"full"})],
    );
    for (index, value) in variants.into_iter().enumerate() {
        let data: ToneDataset = serde_json::from_value(value).unwrap();
        let error = prepare_development_inputs(
            &data,
            &plan,
            &root.path().join("missing-root"),
            &ToneInputLimits::default(),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(
            !error.contains("No such file"),
            "variant {index} reached the filesystem: {error}"
        );
    }
}

#[cfg(unix)]
#[test]
fn rejects_symlinks_at_each_audio_path_component() {
    use std::os::unix::fs::symlink;
    let root = tempfile::tempdir().unwrap();
    let original = record(root.path(), "dev", "dev-a", Some(&[100, -100, 200, -200]));
    let plan = plan(
        "dev",
        vec![json!({"id":"case","record_id":"dev-a","input":"full"})],
    );
    symlink("dev-a.wav", root.path().join("dev/link.wav")).unwrap();
    let mut linked = original.clone();
    linked["audio"]["path"] = json!("dev/link.wav");
    let result = prepare_development_inputs(
        &dataset(vec![linked]),
        &plan,
        root.path(),
        &ToneInputLimits::default(),
        &AtomicBool::new(false),
    );
    assert!(result.is_err(), "a symlinked WAV was accepted");

    let outside = tempfile::tempdir().unwrap();
    std::fs::copy(
        root.path().join("dev/dev-a.wav"),
        outside.path().join("alias.wav"),
    )
    .unwrap();
    symlink(outside.path(), root.path().join("dev/outside")).unwrap();
    let mut escaped = original;
    escaped["audio"]["path"] = json!("dev/outside/alias.wav");
    let result = prepare_development_inputs(
        &dataset(vec![escaped]),
        &plan,
        root.path(),
        &ToneInputLimits::default(),
        &AtomicBool::new(false),
    );
    assert!(result.is_err(), "a symlinked parent was accepted");
}

#[cfg(unix)]
#[test]
fn bounded_preparation_rejects_per_file_and_aggregate_resource_overflow() {
    let root = tempfile::tempdir().unwrap();
    let one = record(root.path(), "dev", "one", Some(&vec![1000; 320]));
    let two = record(root.path(), "dev", "two", Some(&vec![2000; 320]));
    let first_size = std::fs::metadata(root.path().join("dev/one.wav"))
        .unwrap()
        .len() as usize;
    let second_size = std::fs::metadata(root.path().join("dev/two.wav"))
        .unwrap()
        .len() as usize;
    let data = dataset(vec![one, two]);
    let exact_plan = plan(
        "dev",
        vec![
            json!({"id":"one-full","record_id":"one","input":"full"}),
            json!({"id":"one-silent","record_id":"one","input":"audio_masked"}),
            json!({"id":"one-no-text","record_id":"one","input":"text_masked"}),
        ],
    );
    let exact = ToneInputLimits {
        max_file_bytes: first_size,
        max_total_bytes: first_size,
        max_record_samples: 320,
        max_total_samples: 320,
    };
    assert_eq!(
        prepare_development_inputs(
            &data,
            &exact_plan,
            root.path(),
            &exact,
            &AtomicBool::new(false)
        )
        .unwrap()
        .cases()
        .len(),
        3
    );
    let plan = plan(
        "dev",
        vec![
            json!({"id":"one-case","record_id":"one","input":"full"}),
            json!({"id":"two-case","record_id":"two","input":"full"}),
        ],
    );
    let limits = ToneInputLimits {
        max_file_bytes: first_size - 1,
        ..ToneInputLimits::default()
    };
    let error =
        prepare_development_inputs(&data, &plan, root.path(), &limits, &AtomicBool::new(false))
            .unwrap_err();
    assert!(error.contains("limit"), "{error}");
    let limits = ToneInputLimits {
        max_total_bytes: first_size + second_size - 1,
        ..ToneInputLimits::default()
    };
    let error =
        prepare_development_inputs(&data, &plan, root.path(), &limits, &AtomicBool::new(false))
            .unwrap_err();
    assert!(error.contains("limit"), "{error}");
    let limits = ToneInputLimits {
        max_total_samples: 639,
        ..ToneInputLimits::default()
    };
    let error =
        prepare_development_inputs(&data, &plan, root.path(), &limits, &AtomicBool::new(false))
            .unwrap_err();
    assert!(error.contains("limit"), "{error}");
    let limits = ToneInputLimits {
        max_file_bytes: usize::MAX,
        ..ToneInputLimits::default()
    };
    let error = prepare_development_inputs(
        &data,
        &plan,
        &root.path().join("missing"),
        &limits,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(error.contains("limit"), "{error}");
}

#[cfg(unix)]
#[test]
fn stereo_24khz_wav_produces_equal_length_16khz_masks() {
    let root = tempfile::tempdir().unwrap();
    let mut development = record(root.path(), "dev", "dev-a", None);
    std::fs::create_dir(root.path().join("dev")).unwrap();
    let path = root.path().join("dev/dev-a.wav");
    let mut wav = hound::WavWriter::create(
        &path,
        hound::WavSpec {
            channels: 2,
            sample_rate: 24_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for _ in 0..480 {
        wav.write_sample(8000_i16).unwrap();
        wav.write_sample(4000_i16).unwrap();
    }
    wav.finalize().unwrap();
    development["audio"]["sha256"] = json!(file_hash(&path).unwrap());
    let plan = plan(
        "dev",
        vec![
            json!({"id":"full","record_id":"dev-a","input":"full"}),
            json!({"id":"audio","record_id":"dev-a","input":"audio_masked"}),
            json!({"id":"text","record_id":"dev-a","input":"text_masked"}),
        ],
    );
    let prepared = prepare_development_inputs(
        &dataset(vec![development]),
        &plan,
        root.path(),
        &ToneInputLimits::default(),
        &AtomicBool::new(false),
    )
    .unwrap();
    let inputs = prepared.cases();
    assert_eq!(inputs[0].sample_rate(), 16_000);
    assert_eq!(inputs[0].samples().len(), 320);
    assert_eq!(inputs[1].samples().len(), 320);
    assert_eq!(inputs[2].samples(), inputs[0].samples());
    assert!(inputs[1].samples().iter().all(|sample| *sample == 0.0));
}

#[cfg(unix)]
#[test]
fn sinc_overshoot_remains_unchanged_in_all_prepared_inputs() {
    let root = tempfile::tempdir().unwrap();
    let mut development = record(root.path(), "dev", "wave", None);
    std::fs::create_dir(root.path().join("dev")).unwrap();
    let path = root.path().join("dev/wave.wav");
    let mut wav = hound::WavWriter::create(
        &path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for n in 0..240 {
        wav.write_sample(if n < 120 { 32767_i16 } else { -32767_i16 })
            .unwrap();
    }
    wav.finalize().unwrap();
    development["audio"]["sha256"] = json!(file_hash(&path).unwrap());
    let bytes = std::fs::read(&path).unwrap();
    let stop = AtomicBool::new(false);
    let decoded = omni_lfm3::media_io::decode_wav(&bytes, &stop).unwrap();
    let peak = decoded
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0_f32, f32::max);
    assert_eq!(decoded.len(), 160);
    assert!(peak > 1.13 && peak < 1.14, "unexpected sinc peak {peak}");
    let features = omni_lfm3::audio::extract(&decoded, 16_000).expect("extract finite overshoot");
    assert!(features.frames > 0 && features.values.iter().all(|sample| sample.is_finite()));
    let expected_hash = reference_pcm_hash(&decoded);
    eprintln!(
        "review counterexample: peak={peak} samples={} pcm_sha256={expected_hash}",
        decoded.len()
    );
    let plan = plan(
        "dev",
        vec![
            json!({"id":"full","record_id":"wave","input":"full"}),
            json!({"id":"audio","record_id":"wave","input":"audio_masked"}),
            json!({"id":"text","record_id":"wave","input":"text_masked"}),
        ],
    );
    let limits = ToneInputLimits {
        max_file_bytes: bytes.len(),
        max_total_bytes: bytes.len(),
        max_record_samples: 160,
        max_total_samples: 160,
    };
    let prepared = prepare_development_inputs(
        &dataset(vec![
            development,
            record(root.path(), "holdout", "secret", None),
        ]),
        &plan,
        root.path(),
        &limits,
        &stop,
    )
    .unwrap();
    let inputs = prepared.cases();
    assert!(
        inputs[0]
            .samples()
            .iter()
            .zip(&decoded)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert_eq!(inputs[0].samples().len(), decoded.len());
    assert_eq!(inputs[2].samples(), inputs[0].samples());
    assert_eq!(inputs[0].pcm_sha256(), expected_hash);
    assert_eq!(inputs[2].pcm_sha256(), expected_hash);
    assert_eq!(inputs[1].samples().len(), decoded.len());
    assert!(inputs[1].samples().iter().all(|sample| *sample == 0.0));
    assert_ne!(inputs[1].pcm_sha256(), expected_hash);
    assert_eq!(inputs[0].sample_rate(), 16_000);
    assert_eq!(inputs[1].sample_rate(), 16_000);
    assert_eq!(inputs[2].sample_rate(), 16_000);
    assert_eq!(inputs[0].transcript(), Some("测试文本wave"));
    assert_eq!(inputs[2].transcript(), None);
    assert_eq!(prepared.holdout_pcm_independence(), "unverified");
}

#[cfg(unix)]
#[test]
fn pcm16_rate_and_channel_boundaries_preserve_decoder_samples_and_budgets() {
    for (rate, channels) in [(16_000, 1), (16_000, 2), (24_000, 1), (24_000, 2)] {
        let root = tempfile::tempdir().unwrap();
        let mut development = record(root.path(), "dev", "wave", None);
        std::fs::create_dir(root.path().join("dev")).unwrap();
        let path = root.path().join("dev/wave.wav");
        let mut wav = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels,
                sample_rate: rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        let frames = rate / 100;
        for n in 0..frames {
            for _ in 0..channels {
                wav.write_sample(if n < frames / 2 { i16::MAX } else { i16::MIN })
                    .unwrap();
            }
        }
        wav.finalize().unwrap();
        let file_hash = file_hash(&path).unwrap();
        development["audio"]["sha256"] = json!(file_hash);
        let bytes = std::fs::read(&path).unwrap();
        let stop = AtomicBool::new(false);
        let decoded = omni_lfm3::media_io::decode_wav(&bytes, &stop).unwrap();
        assert_eq!(decoded.len(), 160, "{rate} Hz / {channels} channel(s)");
        if rate == 24_000 {
            assert!(
                decoded.iter().any(|sample| sample.abs() > 1.0),
                "expected sinc overshoot"
            );
        } else {
            assert!(decoded.contains(&-1.0), "PCM16 lower endpoint was changed");
            assert!(
                decoded.contains(&(32767.0 / 32768.0)),
                "PCM16 upper endpoint was changed"
            );
        }
        let hash = reference_pcm_hash(&decoded);
        let plan = plan(
            "dev",
            vec![
                json!({"id":"full","record_id":"wave","input":"full"}),
                json!({"id":"audio","record_id":"wave","input":"audio_masked"}),
                json!({"id":"text","record_id":"wave","input":"text_masked"}),
            ],
        );
        let data = dataset(vec![
            development,
            record(root.path(), "holdout", "secret", None),
        ]);
        let exact = ToneInputLimits {
            max_file_bytes: bytes.len(),
            max_total_bytes: bytes.len(),
            max_record_samples: 160,
            max_total_samples: 160,
        };
        let prepared =
            prepare_development_inputs(&data, &plan, root.path(), &exact, &stop).unwrap();
        let inputs = prepared.cases();
        assert_eq!(inputs.len(), 3);
        assert!(
            inputs[0]
                .samples()
                .iter()
                .zip(&decoded)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!(inputs[0].samples().len(), decoded.len());
        assert_eq!(inputs[2].samples(), inputs[0].samples());
        assert_eq!(inputs[0].pcm_sha256(), hash);
        assert_eq!(inputs[2].pcm_sha256(), hash);
        assert_eq!(inputs[0].source_audio_sha256(), file_hash);
        assert_eq!(inputs[1].samples().len(), decoded.len());
        assert!(
            inputs[1]
                .samples()
                .iter()
                .all(|sample| sample.to_bits() == 0)
        );
        assert_eq!(inputs[1].pcm_sha256(), reference_pcm_hash(&vec![0.0; 160]));
        assert_ne!(inputs[1].pcm_sha256(), hash);
        for input in inputs {
            assert_eq!(input.sample_rate(), 16_000);
        }
        assert_eq!(inputs[0].transcript(), Some("测试文本wave"));
        assert_eq!(inputs[1].transcript(), inputs[0].transcript());
        assert_eq!(inputs[2].transcript(), None);
        assert_eq!(inputs[0].instruction(), inputs[2].instruction());
        for limits in [
            ToneInputLimits {
                max_record_samples: 159,
                ..exact
            },
            ToneInputLimits {
                max_total_samples: 159,
                ..exact
            },
            ToneInputLimits {
                max_file_bytes: bytes.len() - 1,
                ..exact
            },
            ToneInputLimits {
                max_total_bytes: bytes.len() - 1,
                ..exact
            },
        ] {
            let error =
                prepare_development_inputs(&data, &plan, root.path(), &limits, &stop).unwrap_err();
            assert!(
                error.contains("limit"),
                "{rate} Hz / {channels} channels: {error}"
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn float_input_boundaries_remain_strict_before_resampling() {
    for (rate, channels) in [(16_000, 1), (16_000, 2), (24_000, 1), (24_000, 2)] {
        for value in [
            -1.0_f32,
            1.0,
            -1.000_001,
            1.000_001,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let root = tempfile::tempdir().unwrap();
            let mut development = record(root.path(), "dev", "wave", None);
            std::fs::create_dir(root.path().join("dev")).unwrap();
            let path = root.path().join("dev/wave.wav");
            let mut wav = hound::WavWriter::create(
                &path,
                hound::WavSpec {
                    channels,
                    sample_rate: rate,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();
            let frames = rate / 100;
            let valid = value.is_finite() && value.abs() <= 1.0;
            for n in 0..frames {
                for channel in 0..channels {
                    let sample = if n < frames / 2 { value } else { -value };
                    // Invalid stereo channels may average to zero; validate each input channel.
                    wav.write_sample(if !valid && channel == 1 {
                        -sample
                    } else {
                        sample
                    })
                    .unwrap();
                }
            }
            wav.finalize().unwrap();
            development["audio"]["sha256"] = json!(file_hash(&path).unwrap());
            let plan = plan(
                "dev",
                vec![json!({"id":"full","record_id":"wave","input":"full"})],
            );
            let result = prepare_development_inputs(
                &dataset(vec![development]),
                &plan,
                root.path(),
                &ToneInputLimits::default(),
                &AtomicBool::new(false),
            );
            if valid {
                let prepared = result.expect("finite boundary float rejected");
                let decoded = omni_lfm3::media_io::decode_wav(
                    &std::fs::read(&path).unwrap(),
                    &AtomicBool::new(false),
                )
                .unwrap();
                assert_eq!(prepared.cases()[0].samples(), decoded);
                assert_eq!(
                    prepared.cases()[0].pcm_sha256(),
                    reference_pcm_hash(&decoded)
                );
                if rate == 24_000 {
                    assert!(decoded.iter().any(|sample| sample.abs() > 1.0));
                }
            } else {
                let error = result.unwrap_err();
                assert!(
                    error.contains("invalid tone WAV sample"),
                    "{rate} Hz / {channels} channels / {value}: {error}"
                );
            }
        }
    }
}

#[cfg(unix)]
#[test]
fn audio_safety_rejects_tampered_truncated_and_nonfinite_wav() {
    let root = tempfile::tempdir().unwrap();
    let original = record(root.path(), "dev", "dev-a", Some(&[100, -100, 200, -200]));
    let path = root.path().join("dev/dev-a.wav");
    let bytes = std::fs::read(&path).unwrap();
    let plan = plan(
        "dev",
        vec![json!({"id":"case","record_id":"dev-a","input":"full"})],
    );
    let prepare = |record: Value| {
        prepare_development_inputs(
            &dataset(vec![record]),
            &plan,
            root.path(),
            &ToneInputLimits::default(),
            &AtomicBool::new(false),
        )
    };
    std::fs::write(&path, b"changed").unwrap();
    assert!(prepare(original.clone()).unwrap_err().contains("hash"));
    let mut changed = original.clone();
    changed["audio"]["sha256"] = json!(file_hash(&path).unwrap());
    assert!(prepare(changed).is_err(), "malformed WAV was accepted");
    std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
    let mut truncated = original.clone();
    truncated["audio"]["sha256"] = json!(file_hash(&path).unwrap());
    assert!(prepare(truncated).is_err(), "truncated WAV was accepted");
    let mut wav = hound::WavWriter::create(
        &path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .unwrap();
    wav.write_sample(f32::NAN).unwrap();
    wav.finalize().unwrap();
    let mut nonfinite = original;
    nonfinite["audio"]["sha256"] = json!(file_hash(&path).unwrap());
    assert!(prepare(nonfinite).is_err(), "nonfinite WAV was accepted");
}

#[test]
fn audio_safety_cancellation_precedes_any_filesystem_access() {
    let root = tempfile::tempdir().unwrap();
    let data = dataset(vec![record(root.path(), "dev", "dev-a", None)]);
    let plan = plan(
        "dev",
        vec![json!({"id":"case","record_id":"dev-a","input":"full"})],
    );
    let error = prepare_development_inputs(
        &data,
        &plan,
        &root.path().join("missing-root"),
        &ToneInputLimits::default(),
        &AtomicBool::new(true),
    )
    .unwrap_err();
    assert!(error.contains("interrupted"), "{error}");
}

#[test]
fn invalid_pair_references_are_rejected_before_opening_audio() {
    let root = tempfile::tempdir().unwrap();
    let data = dataset(vec![record(root.path(), "dev", "dev-a", None)]);
    let mut plan = plan(
        "dev",
        vec![json!({"id":"case","record_id":"dev-a","input":"full"})],
    );
    plan.pairs = serde_json::from_value(json!([{
        "id":"bad-pair","kind":"audio_masked","original":"case","variant":"../holdout/secret"
    }]))
    .unwrap();
    let error = prepare_development_inputs(
        &data,
        &plan,
        &root.path().join("missing-root"),
        &ToneInputLimits::default(),
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(
        error.contains("pair"),
        "invalid pair reached the filesystem: {error}"
    );
}

#[cfg(unix)]
#[test]
fn input_identity_binds_case_run_and_public_partition_declarations() {
    let root = tempfile::tempdir().unwrap();
    let dev = record(root.path(), "dev", "dev-a", Some(&[500, -500, 300, -300]));
    let holdout = record(root.path(), "holdout", "secret", None);
    let plan = plan(
        "dev",
        vec![json!({"id":"case-a","record_id":"dev-a","input":"full"})],
    );
    let prepare = |data: &ToneDataset, plan: &ToneEvaluationPlan| {
        prepare_development_inputs(
            data,
            plan,
            root.path(),
            &ToneInputLimits::default(),
            &AtomicBool::new(false),
        )
        .unwrap()
        .cases()[0]
            .input_sha256()
            .to_owned()
    };
    let data = dataset(vec![dev.clone(), holdout.clone()]);
    let reference = prepare_development_inputs(
        &data,
        &plan,
        root.path(),
        &ToneInputLimits::default(),
        &AtomicBool::new(false),
    )
    .unwrap();
    // Independently computed SHA-256 over the documented domain, 16 kHz,
    // four samples, and their little-endian f32 representation.
    assert_eq!(
        reference.cases()[0].pcm_sha256(),
        "22dc0831a4b1f8bd6573eea7b82204a6d5811005cc93b4fc7d81356637328617"
    );
    let first = prepare(&data, &plan);
    assert_eq!(first, prepare(&data, &plan));
    assert_eq!(
        first,
        prepare(&dataset(vec![holdout.clone(), dev.clone()]), &plan)
    );
    let mut different_case = plan.clone();
    different_case.cases[0].id = "case-b".into();
    assert_ne!(first, prepare(&data, &different_case));
    let mut different_run = plan.clone();
    different_run.run_id = "another-run".into();
    assert_ne!(first, prepare(&data, &different_run));
    let mut different_holdout = holdout;
    different_holdout["speaker_id"] = json!("different-public-speaker");
    assert_ne!(
        first,
        prepare(&dataset(vec![dev, different_holdout]), &plan)
    );
}

#[cfg(unix)]
#[test]
fn holdout_public_groups_need_no_private_content() {
    let root = tempfile::tempdir().unwrap();
    let dev = record(root.path(), "dev", "dev-a", Some(&[500, -500, 300, -300]));
    let mut holdout = record(root.path(), "holdout", "secret", None);
    holdout["audio"] = json!({"path":"", "sha256":""});
    holdout["text"] = json!("");
    holdout["annotations"] = json!([]);
    holdout["source"] = json!({"name":"", "license":"", "kind":"synthetic"});
    let plan = plan(
        "dev",
        vec![json!({"id":"case-a","record_id":"dev-a","input":"full"})],
    );
    let prepared = prepare_development_inputs(
        &dataset(vec![dev, holdout]),
        &plan,
        root.path(),
        &ToneInputLimits::default(),
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(prepared.cases().len(), 1);
    assert_eq!(prepared.holdout_pcm_independence(), "unverified");
}

#[cfg(unix)]
#[test]
fn only_requested_recordings_are_opened_and_fifos_are_rejected() {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let root = tempfile::tempdir().unwrap();
    let selected = record(root.path(), "dev", "selected", Some(&[120, -120, 80, -80]));
    let mut unselected = record(root.path(), "dev", "unused", None);
    unselected["audio"]["path"] = json!("dev/fifo.wav");
    let path = root.path().join("dev/fifo.wav");
    let name = CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let data = dataset(vec![selected, unselected]);
    let selected_plan = plan(
        "dev",
        vec![json!({"id":"selected-case","record_id":"selected","input":"full"})],
    );
    assert_eq!(
        prepare_development_inputs(
            &data,
            &selected_plan,
            root.path(),
            &ToneInputLimits::default(),
            &AtomicBool::new(false)
        )
        .unwrap()
        .cases()
        .len(),
        1
    );
    let fifo_plan = plan(
        "dev",
        vec![json!({"id":"fifo-case","record_id":"unused","input":"full"})],
    );
    let error = prepare_development_inputs(
        &data,
        &fifo_plan,
        root.path(),
        &ToneInputLimits::default(),
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(error.contains("regular file"), "{error}");
}

#[cfg(unix)]
#[test]
fn train_plan_opens_only_its_selected_train_recording() {
    let root = tempfile::tempdir().unwrap();
    let train = record(
        root.path(),
        "train",
        "train-a",
        Some(&[900, -900, 600, -600]),
    );
    let mut dev = record(root.path(), "dev", "unavailable-dev", None);
    dev["audio"]["sha256"] = json!("a".repeat(64));
    let mut holdout = record(root.path(), "holdout", "secret", None);
    holdout["audio"] = json!({"path":"", "sha256":""});
    let plan = plan(
        "train",
        vec![json!({"id":"train-case","record_id":"train-a","input":"full"})],
    );
    let prepared = prepare_development_inputs(
        &dataset(vec![train, dev, holdout]),
        &plan,
        root.path(),
        &ToneInputLimits::default(),
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(prepared.cases()[0].record_id(), "train-a");
}

#[cfg(not(unix))]
#[test]
fn input_preparation_fails_closed_where_nofollow_is_unsupported() {
    let root = tempfile::tempdir().unwrap();
    let dev = record(root.path(), "dev", "dev-a", Some(&[100, -100]));
    let holdout = record(root.path(), "holdout", "secret", None);
    let plan = plan(
        "dev",
        vec![json!({"id":"case","record_id":"dev-a","input":"full"})],
    );
    let error = prepare_development_inputs(
        &dataset(vec![dev, holdout]),
        &plan,
        root.path(),
        &ToneInputLimits::default(),
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(error.contains("unsupported"), "{error}");
    assert!(!root.path().join("holdout").exists());
}

#[test]
fn invalid_case_ids_and_holdout_split_fail_before_the_root_is_opened() {
    let root = tempfile::tempdir().unwrap();
    let data = dataset(vec![record(root.path(), "dev", "dev-a", None)]);
    let base = plan(
        "dev",
        vec![json!({"id":"case","record_id":"dev-a","input":"full"})],
    );
    let mut malicious = base.clone();
    malicious.cases[0].record_id = "../../holdout/secret".into();
    let mut duplicate = base.clone();
    duplicate.cases.push(duplicate.cases[0].clone());
    let mut holdout = base;
    holdout.split = omni_rsi::speech_tone::ToneSplit::Holdout;
    for plan in [malicious, duplicate, holdout] {
        let error = prepare_development_inputs(
            &data,
            &plan,
            &root.path().join("missing-root"),
            &ToneInputLimits::default(),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(
            !error.contains("No such file"),
            "case validation reached the filesystem: {error}"
        );
    }
}

#[test]
fn oversized_annotation_label_hits_metadata_limit_before_lookup() {
    let root = tempfile::tempdir().unwrap();
    let mut development = record(root.path(), "dev", "dev-a", None);
    development["annotations"][0]["decision"]["label"] = json!("x".repeat(4 * 1024 * 1024 + 1));
    let plan = plan(
        "dev",
        vec![json!({"id":"case","record_id":"dev-a","input":"full"})],
    );
    let error = prepare_development_inputs(
        &dataset(vec![development]),
        &plan,
        &root.path().join("missing"),
        &ToneInputLimits::default(),
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(
        error.contains("limit"),
        "oversized label was processed without the budget: {error}"
    );
}

#[cfg(unix)]
#[test]
fn text_mask_is_independent_of_transcript_and_prepared_pcm_is_a_snapshot() {
    let root = tempfile::tempdir().unwrap();
    let original = record(root.path(), "dev", "dev-a", Some(&[500, -500, 300, -300]));
    let plan = plan(
        "dev",
        vec![
            json!({"id":"full","record_id":"dev-a","input":"full"}),
            json!({"id":"masked","record_id":"dev-a","input":"text_masked"}),
        ],
    );
    let prepare = |record: Value| {
        prepare_development_inputs(
            &dataset(vec![record]),
            &plan,
            root.path(),
            &ToneInputLimits::default(),
            &AtomicBool::new(false),
        )
        .unwrap()
    };
    let first = prepare(original.clone());
    let mut changed_text = original.clone();
    changed_text["text"] = json!("另一个转写");
    let second = prepare(changed_text);
    assert_ne!(
        first.cases()[0].input_sha256(),
        second.cases()[0].input_sha256()
    );
    assert_eq!(
        first.cases()[1].input_sha256(),
        second.cases()[1].input_sha256()
    );
    assert_eq!(first.cases()[1].transcript(), None);
    let before = first.cases()[0].samples().to_vec();
    std::fs::write(
        root.path().join("dev/dev-a.wav"),
        b"modified after preparation",
    )
    .unwrap();
    assert_eq!(first.cases()[0].samples(), before);
    let result = prepare_development_inputs(
        &dataset(vec![original]),
        &plan,
        root.path(),
        &ToneInputLimits::default(),
        &AtomicBool::new(false),
    );
    assert!(result.unwrap_err().contains("hash"));
}
