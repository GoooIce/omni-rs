use omni_rsi::number_development::{
    DevelopmentData, RegistrationOptions, preregister, verify_registration,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::atomic::AtomicBool};

fn source(speaker: char, session: char, recording: char, family: char) -> Value {
    json!({"speaker_sha256":speaker.to_string().repeat(64), "session_sha256":session.to_string().repeat(64),
        "recording_sha256":recording.to_string().repeat(64), "family_sha256":family.to_string().repeat(64)})
}

fn case(root: &Path, split: &str, id: &str, number: u32, src: Value, amplitude: i16) -> Value {
    let filename = format!("{id}.wav");
    let path = root.join(format!("{split}-media")).join(&filename);
    let mut writer = hound::WavWriter::create(
        &path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for sample in [0_i16, amplitude, -amplitude, 0] {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
    json!({"id":id, "audio_file":filename, "audio_sha256":format!("{:x}",Sha256::digest(fs::read(path).unwrap())),
        "spoken_number":number, "answer":number+1, "source":src,
        "declaration":{"kind":"licensed-synthetic","source":"fixture","license":"CC0-1.0"}})
}

fn write_json(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}

fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    fs::write(
        root.path().join("layout.v3"),
        b"omni-rsi-number-train-dev-v3\n",
    )
    .unwrap();
    fs::create_dir(root.path().join("train-media")).unwrap();
    fs::create_dir(root.path().join("dev-media")).unwrap();
    let train = case(
        root.path(),
        "train",
        "train-a",
        7,
        source('1', '2', '3', '4'),
        4096,
    );
    let first = case(
        root.path(),
        "dev",
        "dev-a",
        12,
        source('5', '6', '7', '8'),
        8192,
    );
    let second = case(
        root.path(),
        "dev",
        "dev-b",
        15,
        source('9', 'a', 'b', 'c'),
        12_288,
    );
    write_json(
        &root.path().join("train.groups.v3.json"),
        &json!({"version":3,"partition":"train",
        "records":[{"id":train["id"],"source":train["source"]}]}),
    );
    write_json(
        &root.path().join("dev.groups.v3.json"),
        &json!({"version":3,"partition":"dev",
        "records":[{"id":first["id"],"source":first["source"]},{"id":second["id"],"source":second["source"]}]}),
    );
    write_json(
        &root.path().join("train.v3.json"),
        &json!({"version":3,"partition":"train","rule":"number-plus-one-v1","records":[train]}),
    );
    write_json(
        &root.path().join("dev.v3.json"),
        &json!({"version":3,"partition":"dev","rule":"number-plus-one-v1","records":[first,second]}),
    );
    write_json(
        &root.path().join("holdout-public.json"),
        &json!({"version":2,"content_sha256":"f".repeat(64),
        "sources":[source('d','e','f','0')]}),
    );
    root
}

fn artifacts(root: &Path) {
    let model = root.join("model");
    let codec = model.join("audio_tokenizer");
    fs::create_dir_all(&codec).unwrap();
    for name in [
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "model.safetensors",
    ] {
        fs::write(model.join(name), format!("synthetic {name}")).unwrap();
    }
    fs::write(codec.join("config.json"), b"synthetic codec config").unwrap();
    fs::write(codec.join("model.safetensors"), b"synthetic codec weights").unwrap();
    fs::write(root.join("candidate.json"), b"synthetic candidate").unwrap();
    fs::write(root.join("evaluator.bin"), b"synthetic evaluator").unwrap();
    fs::create_dir(root.join("regression-inputs")).unwrap();
    let groups: Vec<_> = ["chinese", "english", "other_omni", "reading", "dialogue"]
        .iter()
        .map(|name| {
            let file = format!("{name}.json");
            write_json(
                &root.join("regression-inputs").join(&file),
                &json!({"cases":[format!("synthetic {name}")]}),
            );
            json!({"name":name,"manifest_file":file})
        })
        .collect();
    write_json(
        &root.join("regression-plan.json"),
        &json!({"version":1,"criterion":"no_degradation","groups":groups}),
    );
}

fn options(root: &Path) -> RegistrationOptions<'_> {
    RegistrationOptions {
        data_root: root,
        model_dir: root.join("model"),
        candidate_file: root.join("candidate.json"),
        evaluator_file: root.join("evaluator.bin"),
        backend: "cpu-fixture".into(),
        r2t2: None,
        breeze: None,
    }
}

#[cfg(unix)]
#[test]
fn legacy_combined_manifest_is_rejected_without_opening_it() {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("manifest.json");
    let name = CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let error = DevelopmentData::load(root.path(), &AtomicBool::new(false))
        .err()
        .expect("accepted a legacy combined dataset");
    assert!(error.contains("v3"), "{error}");
}

#[cfg(unix)]
#[test]
fn isolated_train_dev_loads_without_opening_any_holdout_secret_file() {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let root = fixture();
    let private = root.path().join("holdout");
    fs::create_dir(&private).unwrap();
    for name in ["manifest.json", "salt.bin", "holdout.wav"] {
        let path = private.join(name);
        let name = CString::new(path.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    }
    let data = DevelopmentData::load(root.path(), &AtomicBool::new(false)).unwrap();
    assert_eq!(data.train().len(), 1);
    assert_eq!(data.dev().len(), 2);
    assert_eq!(data.dev()[0].answer(), 13);
    assert_eq!(data.dev()[1].answer(), 16);
    assert_eq!(data.blind_fingerprint().len(), 64);
}

#[test]
fn any_source_group_shared_between_partitions_is_rejected_before_media_access() {
    for field in [
        "speaker_sha256",
        "session_sha256",
        "recording_sha256",
        "family_sha256",
    ] {
        for (edited, other) in [("dev", "train"), ("dev", "holdout"), ("train", "holdout")] {
            let root = fixture();
            let dev_path = root.path().join(format!("{edited}.groups.v3.json"));
            let mut dev: Value = serde_json::from_slice(&fs::read(&dev_path).unwrap()).unwrap();
            let other_path = root.path().join(if other == "train" {
                "train.groups.v3.json"
            } else {
                "holdout-public.json"
            });
            let compared: Value = serde_json::from_slice(&fs::read(other_path).unwrap()).unwrap();
            dev["records"][0]["source"][field] = if other == "train" {
                compared["records"][0]["source"][field].clone()
            } else {
                compared["sources"][0][field].clone()
            };
            write_json(&dev_path, &dev);
            // An invalid media path proves that overlap is rejected before opening media.
            fs::remove_file(root.path().join(if edited == "dev" {
                "dev-media/dev-a.wav"
            } else {
                "train-media/train-a.wav"
            }))
            .unwrap();
            let error = DevelopmentData::load(root.path(), &AtomicBool::new(false))
                .err()
                .unwrap();
            assert!(error.contains(field), "{edited} {other} {field}: {error}");
        }
    }
}

#[cfg(unix)]
#[test]
fn overlapping_holdout_group_is_rejected_before_opening_train_dev_answers() {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let root = fixture();
    let group_path = root.path().join("dev.groups.v3.json");
    let mut groups: Value = serde_json::from_slice(&fs::read(&group_path).unwrap()).unwrap();
    let public: Value =
        serde_json::from_slice(&fs::read(root.path().join("holdout-public.json")).unwrap())
            .unwrap();
    groups["records"][0]["source"]["speaker_sha256"] =
        public["sources"][0]["speaker_sha256"].clone();
    write_json(&group_path, &groups);
    let answers = root.path().join("dev.v3.json");
    fs::remove_file(&answers).unwrap();
    let name = CString::new(answers.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let error = DevelopmentData::load(root.path(), &AtomicBool::new(false))
        .err()
        .unwrap();
    assert!(error.contains("speaker_sha256"), "{error}");
}

#[test]
fn reencoded_identical_pcm_cannot_cross_train_and_development() {
    let root = fixture();
    let path = root.path().join("dev-media/dev-a.wav");
    let mut writer = hound::WavWriter::create(
        &path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .unwrap();
    for sample in [0.0_f32, 0.125, -0.125, 0.0] {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
    let dev_path = root.path().join("dev.v3.json");
    let mut dev: Value = serde_json::from_slice(&fs::read(&dev_path).unwrap()).unwrap();
    dev["records"][0]["audio_sha256"] =
        format!("{:x}", Sha256::digest(fs::read(path).unwrap())).into();
    write_json(&dev_path, &dev);
    let error = DevelopmentData::load(root.path(), &AtomicBool::new(false))
        .err()
        .unwrap();
    assert!(error.contains("PCM crosses"), "{error}");
}

#[test]
fn identical_pcm_within_each_partition_requires_the_same_label() {
    for (split, number, amplitude) in [("train", 7_u32, 0.125_f32), ("dev", 12_u32, 0.25_f32)] {
        let root = fixture();
        let path = root.path().join(format!("{split}-media/duplicate.wav"));
        let mut writer = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        for sample in [0.0_f32, amplitude, -amplitude, 0.0] {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        let manifest_path = root.path().join(format!("{split}.v3.json"));
        let group_path = root.path().join(format!("{split}.groups.v3.json"));
        let mut manifest: Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        let mut groups: Value = serde_json::from_slice(&fs::read(&group_path).unwrap()).unwrap();
        let mut duplicate = manifest["records"][0].clone();
        duplicate["id"] = format!("{split}-duplicate").into();
        duplicate["audio_file"] = "duplicate.wav".into();
        duplicate["audio_sha256"] = format!("{:x}", Sha256::digest(fs::read(path).unwrap())).into();
        duplicate["spoken_number"] = (number + 1).into();
        duplicate["answer"] = (number + 2).into();
        groups["records"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id":duplicate["id"],"source":duplicate["source"]}));
        manifest["records"].as_array_mut().unwrap().push(duplicate);
        write_json(&group_path, &groups);
        write_json(&manifest_path, &manifest);
        let error = DevelopmentData::load(root.path(), &AtomicBool::new(false))
            .err()
            .unwrap();
        assert!(error.contains("contradictory"), "{split}: {error}");

        let last = manifest["records"]
            .as_array_mut()
            .unwrap()
            .last_mut()
            .unwrap();
        last["spoken_number"] = number.into();
        last["answer"] = (number + 1).into();
        write_json(&manifest_path, &manifest);
        assert!(
            DevelopmentData::load(root.path(), &AtomicBool::new(false)).is_ok(),
            "{split}"
        );
    }
}

#[cfg(unix)]
#[test]
fn decoded_sample_limit_is_shared_and_stops_before_later_media() {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};

    let root = fixture();
    for (split, count, number, src, amplitude) in [
        ("train", 30, 7, source('1', '2', '3', '4'), 4096_i16),
        ("dev", 32, 12, source('5', '6', '7', '8'), 8192_i16),
    ] {
        let media_dir = root.path().join(format!("{split}-media"));
        let first = media_dir.join("long-0.wav");
        let mut writer = hound::WavWriter::create(
            &first,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..(16_000 * 20) {
            writer.write_sample(amplitude).unwrap();
        }
        writer.finalize().unwrap();
        let digest = format!("{:x}", Sha256::digest(fs::read(&first).unwrap()));
        let mut records = Vec::new();
        let mut groups = Vec::new();
        for index in 0..count {
            let id = format!("{split}-{index}");
            let filename = format!("long-{index}.wav");
            if index > 0 {
                let path = media_dir.join(&filename);
                if split == "dev" && index == count - 1 {
                    // Opening this later file would fail, instead of reporting the sample limit.
                    let name = CString::new(path.as_os_str().as_bytes()).unwrap();
                    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
                } else {
                    fs::hard_link(&first, &path).unwrap();
                }
            }
            records.push(json!({"id":id,"audio_file":filename,"audio_sha256":digest,
                "spoken_number":number,"answer":number+1,"source":src,
                "declaration":{"kind":"licensed-synthetic","source":"fixture","license":"CC0-1.0"}}));
            groups.push(json!({"id":id,"source":src}));
        }
        write_json(
            &root.path().join(format!("{split}.v3.json")),
            &json!({
            "version":3,"partition":split,"rule":"number-plus-one-v1","records":records}),
        );
        write_json(
            &root.path().join(format!("{split}.groups.v3.json")),
            &json!({
            "version":3,"partition":split,"records":groups}),
        );
    }
    let error = DevelopmentData::load(root.path(), &AtomicBool::new(false))
        .err()
        .unwrap();
    assert!(error.contains("decoded sample limit"), "{error}");
}

#[test]
fn missing_authorization_declaration_is_not_a_valid_training_record() {
    let root = fixture();
    let path = root.path().join("train.v3.json");
    let mut train: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    train["records"][0]["declaration"]
        .as_object_mut()
        .unwrap()
        .remove("license");
    write_json(&path, &train);
    let error = DevelopmentData::load(root.path(), &AtomicBool::new(false))
        .err()
        .unwrap();
    assert!(error.contains("authorization"), "{error}");
}

#[test]
fn preregistration_freezes_real_inputs_once_without_creating_a_pass() {
    let root = fixture();
    artifacts(root.path());
    let options = options(root.path());
    let destination = root.path().join("frozen");
    let stop = AtomicBool::new(false);
    let frozen = preregister(&options, &destination, &stop).unwrap();
    assert_eq!(frozen.status(), "insufficient_evidence");
    assert!(!frozen.candidate_eligible());
    assert_eq!(frozen.data().train().len(), 1);
    assert_eq!(frozen.data().dev().len(), 2);
    let original = fs::read(destination.join("protocol.json")).unwrap();
    verify_registration(&options, &destination, &stop).unwrap();
    assert!(preregister(&options, &destination, &stop).is_err());
    assert_eq!(
        fs::read(destination.join("protocol.json")).unwrap(),
        original
    );
}

#[test]
fn registered_development_rejects_changed_media_and_public_commitment() {
    let root = fixture();
    artifacts(root.path());
    let options = options(root.path());
    let destination = root.path().join("frozen");
    let stop = AtomicBool::new(false);
    preregister(&options, &destination, &stop).unwrap();
    let protocol_bytes = fs::read(destination.join("protocol.json")).unwrap();

    let public_path = root.path().join("holdout-public.json");
    let original_public = fs::read(&public_path).unwrap();
    let mut public: Value = serde_json::from_slice(&original_public).unwrap();
    public["content_sha256"] = "0".repeat(64).into();
    write_json(&public_path, &public);
    assert!(verify_registration(&options, &destination, &stop).is_err());
    fs::write(&public_path, original_public).unwrap();

    let dev_path = root.path().join("dev.v3.json");
    let mut dev: Value = serde_json::from_slice(&fs::read(&dev_path).unwrap()).unwrap();
    dev["records"][0] = case(
        root.path(),
        "dev",
        "dev-a",
        12,
        source('5', '6', '7', '8'),
        3000,
    );
    write_json(&dev_path, &dev);
    assert!(verify_registration(&options, &destination, &stop).is_err());
    assert_eq!(
        fs::read(destination.join("protocol.json")).unwrap(),
        protocol_bytes
    );
}

#[test]
fn registration_rechecks_model_codec_candidate_evaluator_and_regression_inputs() {
    let root = fixture();
    artifacts(root.path());
    let options = options(root.path());
    let destination = root.path().join("frozen");
    let stop = AtomicBool::new(false);
    preregister(&options, &destination, &stop).unwrap();
    for name in [
        "model/model.safetensors",
        "model/audio_tokenizer/config.json",
        "candidate.json",
        "evaluator.bin",
        "regression-inputs/chinese.json",
    ] {
        let path = root.path().join(name);
        let original = fs::read(&path).unwrap();
        fs::write(
            &path,
            if name.ends_with("chinese.json") {
                b"{\"cases\":[\"changed\"]}".as_slice()
            } else {
                b"changed".as_slice()
            },
        )
        .unwrap();
        assert!(
            verify_registration(&options, &destination, &stop).is_err(),
            "{name}"
        );
        fs::write(path, original).unwrap();
    }
    let mut changed = crate::options(root.path());
    changed.backend = "different-backend".into();
    assert!(verify_registration(&changed, &destination, &stop).is_err());
    verify_registration(&options, &destination, &stop).unwrap();
}

#[test]
fn registration_binds_training_source_fingerprint_and_index_selected_shards() {
    let root = fixture();
    artifacts(root.path());
    let model = root.path().join("model");
    let codec = model.join("audio_tokenizer");
    fs::create_dir(model.join("nested")).unwrap();
    fs::create_dir(codec.join("nested")).unwrap();
    fs::write(
        model.join("nested/actual-base.safetensors"),
        b"synthetic selected base shard",
    )
    .unwrap();
    fs::write(
        codec.join("nested/actual-codec.safetensors"),
        b"synthetic selected codec shard",
    )
    .unwrap();
    write_json(
        &model.join("model.safetensors.index.json"),
        &json!({"weight_map":{"a":"nested/actual-base.safetensors"}}),
    );
    write_json(
        &codec.join("model.safetensors.index.json"),
        &json!({"weight_map":{"b":"nested/actual-codec.safetensors"}}),
    );
    let options = options(root.path());
    let destination = root.path().join("frozen");
    let stop = AtomicBool::new(false);
    preregister(&options, &destination, &stop).unwrap();

    for shard in [
        model.join("nested/actual-base.safetensors"),
        codec.join("nested/actual-codec.safetensors"),
    ] {
        let original = fs::read(&shard).unwrap();
        fs::write(&shard, b"changed selected shard").unwrap();
        assert!(
            verify_registration(&options, &destination, &stop).is_err(),
            "{}",
            shard.display()
        );
        fs::write(&shard, original).unwrap();
    }
    verify_registration(&options, &destination, &stop).unwrap();
    let envelope: Value =
        serde_json::from_slice(&fs::read(destination.join("protocol.json")).unwrap()).unwrap();
    assert_eq!(
        envelope["protocol"]["base_model_sha256"],
        omni_lfm3::speech_adapter::source_fingerprint(&model).unwrap()
    );
    assert_eq!(
        envelope["protocol"]["codec_sha256"],
        omni_rsi::fingerprint::codec_fingerprint(&codec, &stop).unwrap()
    );
}

#[test]
fn forged_pass_flag_and_corrupt_protocol_are_rejected() {
    let root = fixture();
    artifacts(root.path());
    let options = options(root.path());
    let destination = root.path().join("frozen");
    let stop = AtomicBool::new(false);
    preregister(&options, &destination, &stop).unwrap();
    let path = destination.join("protocol.json");
    let original = fs::read(&path).unwrap();
    let mut report: Value = serde_json::from_slice(&original).unwrap();
    report["candidate_eligible"] = true.into();
    write_json(&path, &report);
    assert!(verify_registration(&options, &destination, &stop).is_err());
    report["candidate_eligible"] = false.into();
    report["protocol"]["candidate_sha256"] = "0".repeat(64).into();
    write_json(&path, &report);
    assert!(verify_registration(&options, &destination, &stop).is_err());
}

#[test]
fn cancellation_before_registration_does_not_create_a_destination() {
    let root = fixture();
    artifacts(root.path());
    let options = options(root.path());
    let destination = root.path().join("frozen");
    let error = preregister(&options, &destination, &AtomicBool::new(true))
        .err()
        .unwrap();
    assert!(error.contains("interrupted"), "{error}");
    assert!(!destination.exists());
}

#[cfg(unix)]
#[test]
fn train_dev_media_symlink_cannot_open_a_holdout_file() {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let root = fixture();
    let holdout = root.path().join("holdout");
    fs::create_dir(&holdout).unwrap();
    let secret = holdout.join("secret.wav");
    let name = CString::new(secret.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let dev_audio = root.path().join("dev-media/dev-a.wav");
    fs::remove_file(&dev_audio).unwrap();
    std::os::unix::fs::symlink(&secret, &dev_audio).unwrap();
    assert!(DevelopmentData::load(root.path(), &AtomicBool::new(false)).is_err());
}

#[cfg(unix)]
#[test]
fn model_and_codec_fingerprints_reject_symlinked_artifacts() {
    for name in [
        "model/config.json",
        "model/model.safetensors",
        "model/audio_tokenizer/config.json",
        "model/audio_tokenizer/model.safetensors",
    ] {
        let root = fixture();
        artifacts(root.path());
        let target = root.path().join("unrelated-synthetic-file");
        fs::write(&target, b"not a model artifact").unwrap();
        let artifact = root.path().join(name);
        fs::remove_file(&artifact).unwrap();
        std::os::unix::fs::symlink(target, artifact).unwrap();
        let error = preregister(
            &options(root.path()),
            &root.path().join("frozen"),
            &AtomicBool::new(false),
        )
        .err()
        .expect("followed a symlink when fingerprinting an artifact");
        assert!(!error.is_empty(), "{name}");
    }
}

#[cfg(unix)]
#[test]
fn verifier_and_teacher_fingerprints_reject_symlinked_artifacts() {
    use omni_rsi::number_development::ToolPaths;
    for teacher in [false, true] {
        let root = fixture();
        artifacts(root.path());
        let model = root.path().join("tool-model");
        fs::create_dir(&model).unwrap();
        for name in [
            "model.safetensors",
            "tokenizer.json",
            "preprocessor_config.json",
        ] {
            fs::write(model.join(name), b"synthetic").unwrap();
        }
        let target = root.path().join("unrelated-synthetic-file");
        fs::write(&target, b"unrelated").unwrap();
        std::os::unix::fs::symlink(target, model.join("config.json")).unwrap();
        let binary = root.path().join("tool-binary");
        fs::write(&binary, b"synthetic binary").unwrap();
        let mut options = options(root.path());
        let tool = Some(ToolPaths { model, binary });
        if teacher {
            options.breeze = tool;
        } else {
            options.r2t2 = tool;
        }
        assert!(
            preregister(
                &options,
                &root.path().join("frozen"),
                &AtomicBool::new(false)
            )
            .is_err(),
            "followed a symlink in the {} model",
            if teacher { "Breeze" } else { "R2T2" }
        );
    }
}

#[test]
fn answer_manifest_must_match_the_prechecked_group_manifest() {
    let root = fixture();
    let path = root.path().join("dev.v3.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    manifest["records"][0]["source"]["speaker_sha256"] = "f".repeat(64).into();
    write_json(&path, &manifest);
    let error = DevelopmentData::load(root.path(), &AtomicBool::new(false))
        .err()
        .unwrap();
    assert!(error.contains("group manifest"), "{error}");
}

#[test]
fn legacy_or_extra_sensitive_public_metadata_is_rejected() {
    let root = fixture();
    let path = root.path().join("holdout-public.json");
    let mut public: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    public["version"] = 1.into();
    write_json(&path, &public);
    assert!(DevelopmentData::load(root.path(), &AtomicBool::new(false)).is_err());
    public["version"] = 2.into();
    public["answer"] = 13.into();
    write_json(&path, &public);
    assert!(DevelopmentData::load(root.path(), &AtomicBool::new(false)).is_err());
}

#[test]
fn competing_registrations_cannot_overwrite_each_other() {
    use std::sync::Barrier;
    let root = fixture();
    artifacts(root.path());
    let options = options(root.path());
    let destination = root.path().join("frozen");
    let stop = AtomicBool::new(false);
    let barrier = Barrier::new(4);
    let winners = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    preregister(&options, &destination, &stop).is_ok()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .filter(|ok| *ok)
            .count()
    });
    assert_eq!(winners, 1);
    assert!(
        !verify_registration(&options, &destination, &stop)
            .unwrap()
            .candidate_eligible()
    );
}

#[cfg(unix)]
#[test]
fn crash_after_directory_reservation_cannot_be_silently_reused() {
    use std::{
        process::{Command, Stdio},
        time::{Duration, Instant},
    };
    const CHILD: &str = "OMNI_RSI_PREREG_CRASH_CHILD";
    if let Some(path) = std::env::var_os(CHILD) {
        let root = Path::new(&path);
        let options = options(root);
        let _ = preregister(&options, &root.join("frozen"), &AtomicBool::new(false));
        std::process::exit(0);
    }
    let root = fixture();
    artifacts(root.path());
    // A sparse synthetic file makes the second real hashing pass long enough
    // for the parent to kill the process after the atomic directory creation.
    fs::OpenOptions::new()
        .write(true)
        .open(root.path().join("candidate.json"))
        .unwrap()
        .set_len(128 * 1024 * 1024)
        .unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("crash_after_directory_reservation_cannot_be_silently_reused")
        .env(CHILD, root.path())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let destination = root.path().join("frozen");
    let started = Instant::now();
    while !destination.exists() {
        assert!(
            child.try_wait().unwrap().is_none(),
            "child exited before reservation"
        );
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "reservation timed out"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    child.kill().unwrap();
    assert!(!child.wait().unwrap().success());
    let options = options(root.path());
    let stop = AtomicBool::new(false);
    assert!(preregister(&options, &destination, &stop).is_err());
    assert!(!destination.join("protocol.json").exists());
    assert!(verify_registration(&options, &destination, &stop).is_err());
}
