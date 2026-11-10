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

fn tagged_source(tag: &str) -> Value {
    let id = |kind: &str| format!("{:x}", Sha256::digest(format!("{tag}-{kind}")));
    json!({"speaker_sha256":id("speaker"),"session_sha256":id("session"),
        "recording_sha256":id("recording"),"family_sha256":id("family")})
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

fn speech_target_fixture(root: &Path) -> Value {
    artifacts(root);
    let model = root.join("model");
    let codec = model.join("audio_tokenizer");
    fs::write(
        model.join("config.json"),
        br#"{"num_codebooks":12,"codebook_size":2017,"audio_eos_code":2016}"#,
    )
    .unwrap();
    fs::write(
        codec.join("config.json"),
        br#"{"n_q":12,"fsq_levels":[8,7,6,6],"sampling_rate":24000}"#,
    )
    .unwrap();
    fs::create_dir(root.join("train-target-media")).unwrap();
    let wav = root.join("train-target-media/answer.wav");
    let mut writer = hound::WavWriter::create(
        &wav,
        hound::WavSpec {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for sample in [0_i16, 4096, -4096, 0] {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
    let encoded = omni_rsi::speech_audio_target::encode_target_with(
        &model,
        &wav,
        &AtomicBool::new(false),
        |_, _| Ok(vec![vec![1; 12], vec![2; 12], vec![3; 12], vec![4; 12]]),
    )
    .unwrap();
    let mut target = serde_json::to_value(encoded).unwrap();
    target["audio_file"] = "answer.wav".into();
    target["source"] =
        json!({"kind":"licensed-synthetic","source":"target fixture","license":"CC0-1.0"});
    target["source_group"] = json!({"speaker_sha256":"01".repeat(32),"session_sha256":"02".repeat(32),
        "recording_sha256":"03".repeat(32),"family_sha256":"04".repeat(32)});
    let path = root.join("train.v3.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    manifest["records"][0]["speech_target"] = target.clone();
    write_json(&path, &manifest);
    let group_path = root.join("train.groups.v3.json");
    let mut groups: Value = serde_json::from_slice(&fs::read(&group_path).unwrap()).unwrap();
    groups["records"][0]["target_source"] = target["source_group"].clone();
    write_json(&group_path, &groups);
    target
}

#[cfg(unix)]
#[test]
fn speech_targets_survive_model_bound_load_and_freeze_without_holdout_access() {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let root = fixture();
    let expected = speech_target_fixture(root.path());
    let private = root.path().join("holdout");
    fs::create_dir(&private).unwrap();
    for name in ["manifest.json", "salt.bin", "holdout.wav"] {
        let name = CString::new(private.join(name).as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    }
    let stop = AtomicBool::new(false);
    assert!(DevelopmentData::load(root.path(), &stop).is_err());
    let data =
        DevelopmentData::load_with_model(root.path(), &root.path().join("model"), &stop).unwrap();
    let target = data.train()[0].speech_target().unwrap();
    assert_eq!(serde_json::to_value(target).unwrap(), expected);
    assert_eq!(target.frames().last().unwrap(), &vec![2016; 12]);
    assert!(data.dev()[0].speech_target().is_none());
    let registration =
        preregister(&options(root.path()), &root.path().join("frozen"), &stop).unwrap();
    assert_eq!(
        serde_json::to_value(registration.data().train()[0].speech_target().unwrap()).unwrap(),
        expected
    );
    assert!(!registration.candidate_eligible());
}

fn replace_train_target(root: &Path, target: Value) {
    let path = root.join("train.v3.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    manifest["records"][0]["speech_target"] = target;
    write_json(&path, &manifest);
}

fn repeat_train_targets(root: &Path, target: &Value, count: usize) {
    let manifest_path = root.join("train.v3.json");
    let groups_path = root.join("train.groups.v3.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let mut groups: Value = serde_json::from_slice(&fs::read(&groups_path).unwrap()).unwrap();
    let base = manifest["records"][0].clone();
    let mut records = Vec::new();
    let mut identities = Vec::new();
    for i in 0..count {
        let mut row = base.clone();
        let name = format!("copy-{i}.wav");
        fs::hard_link(
            root.join("train-media/train-a.wav"),
            root.join("train-media").join(&name),
        )
        .unwrap();
        row["id"] = format!("train-{i}").into();
        row["audio_file"] = name.into();
        row["speech_target"] = target.clone();
        identities.push(
            json!({"id":row["id"],"source":row["source"],"target_source":target["source_group"]}),
        );
        records.push(row);
    }
    manifest["records"] = records.into();
    groups["records"] = identities.into();
    write_json(&manifest_path, &manifest);
    write_json(&groups_path, &groups);
}

#[cfg(unix)]
#[test]
fn speech_frames_require_model_shape_codec_codes_and_one_final_eos() {
    let root = fixture();
    let target = speech_target_fixture(root.path());
    let stop = AtomicBool::new(false);
    let model = root.path().join("model");
    for invalid in [
        json!([vec![1; 12], vec![2; 12], vec![3; 12], vec![2016; 12]]),
        json!(vec![vec![1_u32; 12]; 201]),
        json!([
            vec![1; 11],
            vec![2; 12],
            vec![3; 12],
            vec![4; 12],
            vec![2016; 12]
        ]),
        json!([
            vec![2016; 12],
            vec![2; 12],
            vec![3; 12],
            vec![4; 12],
            vec![2016; 12]
        ]),
        json!([
            vec![1; 12],
            vec![2; 12],
            vec![3; 12],
            vec![4; 12],
            vec![1; 12]
        ]),
        json!([
            vec![1; 12],
            vec![2; 12],
            vec![3; 12],
            vec![4; 12],
            vec![
                2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 1
            ]
        ]),
        json!([
            vec![2017; 12],
            vec![2; 12],
            vec![3; 12],
            vec![4; 12],
            vec![2016; 12]
        ]),
    ] {
        let mut bad = target.clone();
        bad["frames"] = invalid;
        replace_train_target(root.path(), bad);
        assert!(DevelopmentData::load_with_model(root.path(), &model, &stop).is_err());
    }
    replace_train_target(root.path(), target);
    assert!(DevelopmentData::load_with_model(root.path(), &model, &stop).is_ok());
}

#[cfg(unix)]
#[test]
fn incomplete_target_is_rejected_and_text_only_is_explicit() {
    let root = fixture();
    let target = speech_target_fixture(root.path());
    let model = root.path().join("model");
    let stop = AtomicBool::new(false);
    for field in [
        "frames",
        "target_codec_sha256",
        "target_audio_sha256",
        "codec_fingerprint_sha256",
        "model_config_sha256",
        "audio_file",
        "source",
        "source_group",
        "preparation_sha256",
    ] {
        let mut bad = target.clone();
        bad.as_object_mut().unwrap().remove(field);
        replace_train_target(root.path(), bad);
        assert!(
            DevelopmentData::load_with_model(root.path(), &model, &stop).is_err(),
            "{field}"
        );
    }
    let mut bad = target;
    bad["quality_evidence"] = "passed".into();
    replace_train_target(root.path(), bad);
    assert!(DevelopmentData::load_with_model(root.path(), &model, &stop).is_err());
    replace_train_target(root.path(), Value::Null);
    let group_path = root.path().join("train.groups.v3.json");
    let mut groups: Value = serde_json::from_slice(&fs::read(&group_path).unwrap()).unwrap();
    groups["records"][0]
        .as_object_mut()
        .unwrap()
        .remove("target_source");
    write_json(&group_path, &groups);
    let data = DevelopmentData::load(root.path(), &stop).unwrap();
    assert!(data.train()[0].speech_target().is_none());
}

#[cfg(unix)]
#[test]
fn registered_target_rechecks_frames_source_audio_and_codec_files() {
    let root = fixture();
    let target = speech_target_fixture(root.path());
    let options = options(root.path());
    let destination = root.path().join("frozen");
    let stop = AtomicBool::new(false);
    preregister(&options, &destination, &stop).unwrap();
    let mut changed = target.clone();
    changed["frames"][0][0] = 5.into();
    replace_train_target(root.path(), changed);
    assert!(DevelopmentData::load_with_model(root.path(), &options.model_dir, &stop).is_ok());
    assert!(verify_registration(&options, &destination, &stop).is_err());
    let mut changed = target.clone();
    changed["source"]["source"] = "another source".into();
    replace_train_target(root.path(), changed);
    assert!(DevelopmentData::load_with_model(root.path(), &options.model_dir, &stop).is_ok());
    assert!(verify_registration(&options, &destination, &stop).is_err());
    replace_train_target(root.path(), target.clone());

    let wav_path = root.path().join("train-target-media/answer.wav");
    let original = fs::read(&wav_path).unwrap();
    let mut changed_wav = original.clone();
    *changed_wav.last_mut().unwrap() ^= 1;
    fs::write(&wav_path, &changed_wav).unwrap();
    let mut changed = target.clone();
    changed["target_audio_sha256"] = format!("{:x}", Sha256::digest(&changed_wav)).into();
    replace_train_target(root.path(), changed);
    assert!(DevelopmentData::load_with_model(root.path(), &options.model_dir, &stop).is_ok());
    assert!(verify_registration(&options, &destination, &stop).is_err());
    fs::write(wav_path, original).unwrap();
    replace_train_target(root.path(), target);

    for name in [
        "train-target-media/answer.wav",
        "model/audio_tokenizer/config.json",
        "model/audio_tokenizer/model.safetensors",
    ] {
        let path = root.path().join(name);
        let original = fs::read(&path).unwrap();
        let mut bytes = original.clone();
        bytes.push(b' ');
        fs::write(&path, bytes).unwrap();
        assert!(
            verify_registration(&options, &destination, &stop).is_err(),
            "{name}"
        );
        fs::write(path, original).unwrap();
    }
    verify_registration(&options, &destination, &stop).unwrap();
}

#[cfg(unix)]
#[test]
fn indexed_codec_target_uses_the_full_actual_codec_fingerprint() {
    let root = fixture();
    let mut target = speech_target_fixture(root.path());
    let codec = root.path().join("model/audio_tokenizer");
    fs::create_dir(codec.join("parts")).unwrap();
    fs::write(
        codec.join("parts/encoder.safetensors"),
        b"synthetic indexed codec",
    )
    .unwrap();
    write_json(
        &codec.join("model.safetensors.index.json"),
        &json!({"weight_map":{"encoder.weight":"parts/encoder.safetensors"}}),
    );
    let stop = AtomicBool::new(false);
    let fingerprint = omni_rsi::fingerprint::codec_fingerprint(&codec, &stop).unwrap();
    target["target_codec_sha256"] = fingerprint.clone().into();
    target["codec_fingerprint_sha256"] = fingerprint.into();
    replace_train_target(root.path(), target);
    let options = options(root.path());
    let destination = root.path().join("frozen");
    preregister(&options, &destination, &stop).unwrap();
    fs::write(codec.join("parts/encoder.safetensors"), b"changed shard").unwrap();
    assert!(verify_registration(&options, &destination, &stop).is_err());
}

#[cfg(unix)]
#[test]
fn target_validation_checks_actual_model_and_codec_formats_not_only_claimed_hashes() {
    let root = fixture();
    let target = speech_target_fixture(root.path());
    let model = root.path().join("model");
    let codec = model.join("audio_tokenizer");
    let stop = AtomicBool::new(false);
    let model_path = model.join("config.json");
    let codec_path = codec.join("config.json");
    let model_original = fs::read(&model_path).unwrap();
    let codec_original = fs::read(&codec_path).unwrap();
    for (model_value, codec_value) in [
        (
            json!({"num_codebooks":11,"codebook_size":2017,"audio_eos_code":2016}),
            json!({"n_q":12,"fsq_levels":[8,7,6,6],"sampling_rate":24000}),
        ),
        (
            json!({"num_codebooks":12,"codebook_size":2017,"audio_eos_code":2015}),
            json!({"n_q":12,"fsq_levels":[8,7,6,6],"sampling_rate":24000}),
        ),
        (
            json!({"num_codebooks":12,"codebook_size":2017,"audio_eos_code":2016}),
            json!({"n_q":12,"fsq_levels":[8,7,6,6],"sampling_rate":16000}),
        ),
        (
            json!({"num_codebooks":12,"codebook_size":2017,"audio_eos_code":2016}),
            json!({"n_q":12,"fsq_levels":[2,2],"sampling_rate":24000}),
        ),
    ] {
        write_json(&model_path, &model_value);
        write_json(&codec_path, &codec_value);
        let mut updated = target.clone();
        updated["model_config_sha256"] =
            format!("{:x}", Sha256::digest(fs::read(&model_path).unwrap())).into();
        updated["codec_fingerprint_sha256"] =
            omni_rsi::fingerprint::codec_fingerprint(&codec, &stop)
                .unwrap()
                .into();
        replace_train_target(root.path(), updated);
        assert!(
            DevelopmentData::load_with_model(root.path(), &model, &stop).is_err(),
            "model {model_value} codec {codec_value}"
        );
        assert!(
            omni_rsi::speech_audio_target::encode_target_with(
                &model,
                &root.path().join("train-target-media/answer.wav"),
                &stop,
                |_, _| Ok(vec![vec![1; 12], vec![2; 12], vec![3; 12], vec![4; 12]]),
            )
            .is_err(),
            "target encoder accepted model {model_value} codec {codec_value}"
        );
    }
    fs::write(model_path, model_original).unwrap();
    fs::write(codec_path, codec_original).unwrap();
    replace_train_target(root.path(), target);
    assert!(DevelopmentData::load_with_model(root.path(), &model, &stop).is_ok());
}

#[cfg(unix)]
#[test]
fn model_bound_targets_enforce_the_encoder_codebook_width_limit() {
    let root = fixture();
    let target = speech_target_fixture(root.path());
    let model = root.path().join("model");
    let codec = model.join("audio_tokenizer");
    let stop = AtomicBool::new(false);
    for books in [128_usize, 129] {
        let model_config =
            json!({"num_codebooks":books,"codebook_size":2017,"audio_eos_code":2016});
        let codec_config = json!({"n_q":books,"fsq_levels":[8,7,6,6],"sampling_rate":24000});
        write_json(&model.join("config.json"), &model_config);
        write_json(&codec.join("config.json"), &codec_config);
        let mut changed = target.clone();
        changed["model_config_sha256"] = format!(
            "{:x}",
            Sha256::digest(fs::read(model.join("config.json")).unwrap())
        )
        .into();
        changed["codec_fingerprint_sha256"] =
            omni_rsi::fingerprint::codec_fingerprint(&codec, &stop)
                .unwrap()
                .into();
        changed["frames"] = json!([
            vec![1_u32; books],
            vec![2; books],
            vec![3; books],
            vec![4; books],
            vec![2016; books]
        ]);
        replace_train_target(root.path(), changed);
        let result = DevelopmentData::load_with_model(root.path(), &model, &stop);
        if books == 128 {
            assert!(result.is_ok());
        } else {
            let error = result
                .err()
                .expect("unsupported codebook width was accepted");
            assert!(
                error.contains("codec") || error.contains("codebook"),
                "{error}"
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn target_audio_symlink_cannot_open_holdout_media() {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let root = fixture();
    speech_target_fixture(root.path());
    fs::create_dir(root.path().join("holdout")).unwrap();
    let secret = root.path().join("holdout/secret.wav");
    let name = CString::new(secret.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let target = root.path().join("train-target-media/answer.wav");
    fs::remove_file(&target).unwrap();
    std::os::unix::fs::symlink(secret, target).unwrap();
    assert!(
        DevelopmentData::load_with_model(
            root.path(),
            &root.path().join("model"),
            &AtomicBool::new(false)
        )
        .is_err()
    );
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

#[cfg(unix)]
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

#[cfg(unix)]
#[test]
fn target_source_groups_are_checked_before_any_case_or_media_read() {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    for field in [
        "speaker_sha256",
        "session_sha256",
        "recording_sha256",
        "family_sha256",
    ] {
        for (edited, other) in [
            ("train", "dev"),
            ("dev", "train"),
            ("train", "holdout"),
            ("dev", "holdout"),
        ] {
            let root = fixture();
            let path = root.path().join(format!("{edited}.groups.v3.json"));
            let mut groups: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            let other_path = root.path().join(if other == "holdout" {
                "holdout-public.json".to_owned()
            } else {
                format!("{other}.groups.v3.json")
            });
            let other_groups: Value =
                serde_json::from_slice(&fs::read(other_path).unwrap()).unwrap();
            let compared = if other == "holdout" {
                &other_groups["sources"][0]
            } else {
                &other_groups["records"][0]["source"]
            };
            let mut target = json!({"speaker_sha256":"01".repeat(32),"session_sha256":"02".repeat(32),
                "recording_sha256":"03".repeat(32),"family_sha256":"04".repeat(32)});
            target[field] = compared[field].clone();
            groups["records"][0]["target_source"] = target;
            write_json(&path, &groups);
            for split in ["train", "dev"] {
                let cases = root.path().join(format!("{split}.v3.json"));
                fs::remove_file(&cases).unwrap();
                let name = CString::new(cases.as_os_str().as_bytes()).unwrap();
                assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
            }
            let error = DevelopmentData::load(root.path(), &AtomicBool::new(false))
                .err()
                .unwrap();
            assert!(error.contains(field), "{edited}/{other}/{field}: {error}");
        }
    }
    let root = fixture();
    let path = root.path().join("train.groups.v3.json");
    let mut groups: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    groups["records"][0]["target_source"] = json!({"speaker_sha256":"invalid","session_sha256":"02".repeat(32),
        "recording_sha256":"03".repeat(32),"family_sha256":"04".repeat(32)});
    write_json(&path, &groups);
    let cases = root.path().join("train.v3.json");
    fs::remove_file(&cases).unwrap();
    let name = CString::new(cases.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let error = DevelopmentData::load(root.path(), &AtomicBool::new(false))
        .err()
        .unwrap();
    assert!(error.contains("group identity"), "{error}");
}

#[cfg(unix)]
#[test]
fn target_case_must_match_its_prechecked_group_before_media_access() {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let root = fixture();
    let mut target = speech_target_fixture(root.path());
    let source = json!({"speaker_sha256":"01".repeat(32),"session_sha256":"02".repeat(32),
        "recording_sha256":"03".repeat(32),"family_sha256":"04".repeat(32)});
    target["source_group"] = source.clone();
    replace_train_target(root.path(), target.clone());
    let path = root.path().join("train.groups.v3.json");
    let mut groups: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    groups["records"][0]["target_source"] = source;
    write_json(&path, &groups);
    let stop = AtomicBool::new(false);
    let model = root.path().join("model");
    assert!(DevelopmentData::load_with_model(root.path(), &model, &stop).is_ok());

    target["source_group"]["speaker_sha256"] = "05".repeat(32).into();
    replace_train_target(root.path(), target.clone());
    let wav = root.path().join("train-target-media/answer.wav");
    fs::remove_file(&wav).unwrap();
    let name = CString::new(wav.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let error = DevelopmentData::load_with_model(root.path(), &model, &stop)
        .err()
        .unwrap();
    assert!(error.contains("group manifest"), "{error}");

    replace_train_target(root.path(), Value::Null);
    let error = DevelopmentData::load_with_model(root.path(), &model, &stop)
        .err()
        .unwrap();
    assert!(error.contains("group manifest"), "{error}");
    groups["records"][0]
        .as_object_mut()
        .unwrap()
        .remove("target_source");
    write_json(&path, &groups);
    replace_train_target(root.path(), target);
    let error = DevelopmentData::load_with_model(root.path(), &model, &stop)
        .err()
        .unwrap();
    assert!(error.contains("group manifest"), "{error}");
}

#[cfg(unix)]
#[test]
fn target_sources_cannot_share_groups_across_partitions_before_cases_are_opened() {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    for field in [
        "speaker_sha256",
        "session_sha256",
        "recording_sha256",
        "family_sha256",
    ] {
        let root = fixture();
        let mut train_source = tagged_source("train-target");
        let dev_source = tagged_source("dev-target");
        train_source[field] = dev_source[field].clone();
        for (split, target_source) in [("train", train_source), ("dev", dev_source)] {
            let path = root.path().join(format!("{split}.groups.v3.json"));
            let mut groups: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            groups["records"][0]["target_source"] = target_source;
            write_json(&path, &groups);
            let cases = root.path().join(format!("{split}.v3.json"));
            fs::remove_file(&cases).unwrap();
            let name = CString::new(cases.as_os_str().as_bytes()).unwrap();
            assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        }
        let error = DevelopmentData::load(root.path(), &AtomicBool::new(false))
            .err()
            .unwrap();
        assert!(error.contains(field), "{field}: {error}");
    }
}

#[cfg(unix)]
#[test]
fn target_wav_and_reencoded_pcm_cannot_cross_train_dev() {
    for (reencoded, duplicate) in [(false, true), (true, true), (true, false)] {
        let root = fixture();
        let mut target = speech_target_fixture(root.path());
        fs::create_dir(root.path().join("dev-target-media")).unwrap();
        let wav = root.path().join("dev-target-media/answer.wav");
        if reencoded {
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
            let amplitude = if duplicate { 0.125_f32 } else { 0.25_f32 };
            for sample in [0.0_f32, amplitude, -amplitude, -0.0] {
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
        } else {
            fs::copy(root.path().join("train-target-media/answer.wav"), &wav).unwrap();
        }
        target["target_audio_sha256"] =
            format!("{:x}", Sha256::digest(fs::read(&wav).unwrap())).into();
        if reencoded {
            assert_ne!(
                target["target_audio_sha256"],
                format!(
                    "{:x}",
                    Sha256::digest(
                        fs::read(root.path().join("train-target-media/answer.wav")).unwrap()
                    )
                )
            );
        }
        target["source_group"] = json!({"speaker_sha256":"a1".repeat(32),"session_sha256":"a2".repeat(32),
            "recording_sha256":"a3".repeat(32),"family_sha256":"a4".repeat(32)});
        let cases_path = root.path().join("dev.v3.json");
        let mut cases: Value = serde_json::from_slice(&fs::read(&cases_path).unwrap()).unwrap();
        cases["records"][0]["speech_target"] = target.clone();
        write_json(&cases_path, &cases);
        let groups_path = root.path().join("dev.groups.v3.json");
        let mut groups: Value = serde_json::from_slice(&fs::read(&groups_path).unwrap()).unwrap();
        groups["records"][0]["target_source"] = target["source_group"].clone();
        write_json(&groups_path, &groups);
        let result = DevelopmentData::load_with_model(
            root.path(),
            &root.path().join("model"),
            &AtomicBool::new(false),
        );
        if duplicate {
            let error = result
                .err()
                .expect("duplicate target audio crossed train/dev");
            assert!(
                error.contains("target") && error.contains("cross"),
                "{reencoded}: {error}"
            );
        } else {
            assert!(result.is_ok());
        }
    }
}

#[cfg(unix)]
#[test]
fn target_wav_cannot_reappear_as_a_cross_partition_input() {
    for (target_in_train, mode) in [true, false].into_iter().flat_map(|direction| {
        ["raw", "metadata", "float", "resampled16k", "different"]
            .into_iter()
            .map(move |mode| (direction, mode))
    }) {
        let root = fixture();
        let mut target = speech_target_fixture(root.path());
        let original = root.path().join("train-target-media/answer.wav");
        let input_split = if target_in_train { "dev" } else { "train" };
        if !target_in_train {
            fs::create_dir(root.path().join("dev-target-media")).unwrap();
            fs::copy(&original, root.path().join("dev-target-media/answer.wav")).unwrap();
            target["source_group"] = json!({"speaker_sha256":"a1".repeat(32),"session_sha256":"a2".repeat(32),
                "recording_sha256":"a3".repeat(32),"family_sha256":"a4".repeat(32)});
            let path = root.path().join("dev.v3.json");
            let mut cases: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            cases["records"][0]["speech_target"] = target.clone();
            write_json(&path, &cases);
            let path = root.path().join("dev.groups.v3.json");
            let mut groups: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            groups["records"][0]["target_source"] = target["source_group"].clone();
            write_json(&path, &groups);
            replace_train_target(root.path(), Value::Null);
            let path = root.path().join("train.groups.v3.json");
            let mut groups: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            groups["records"][0]
                .as_object_mut()
                .unwrap()
                .remove("target_source");
            write_json(&path, &groups);
        }
        let input = root
            .path()
            .join(format!("{input_split}-media/{input_split}-a.wav"));
        match mode {
            "raw" => {
                fs::copy(&original, &input).unwrap();
            }
            "metadata" => {
                let mut bytes = fs::read(&original).unwrap();
                bytes.extend_from_slice(b"JUNK\x04\0\0\0meta");
                let size = u32::try_from(bytes.len() - 8).unwrap();
                bytes[4..8].copy_from_slice(&size.to_le_bytes());
                fs::write(&input, bytes).unwrap();
            }
            "resampled16k" => {
                let samples = omni_lfm3::media_io::decode_wav(
                    &fs::read(&original).unwrap(),
                    &AtomicBool::new(false),
                )
                .unwrap();
                let mut writer = hound::WavWriter::create(
                    &input,
                    hound::WavSpec {
                        channels: 1,
                        sample_rate: 16_000,
                        bits_per_sample: 32,
                        sample_format: hound::SampleFormat::Float,
                    },
                )
                .unwrap();
                for sample in samples {
                    writer.write_sample(sample).unwrap();
                }
                writer.finalize().unwrap();
            }
            _ => {
                let amplitude = if mode == "different" {
                    0.25_f32
                } else {
                    0.125_f32
                };
                let mut writer = hound::WavWriter::create(
                    &input,
                    hound::WavSpec {
                        channels: 1,
                        sample_rate: 24_000,
                        bits_per_sample: 32,
                        sample_format: hound::SampleFormat::Float,
                    },
                )
                .unwrap();
                for sample in [0.0_f32, amplitude, -amplitude, -0.0] {
                    writer.write_sample(sample).unwrap();
                }
                writer.finalize().unwrap();
            }
        }
        if mode != "raw" {
            assert_ne!(fs::read(&input).unwrap(), fs::read(&original).unwrap());
        }
        let path = root.path().join(format!("{input_split}.v3.json"));
        let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        manifest["records"][0]["audio_sha256"] =
            format!("{:x}", Sha256::digest(fs::read(input).unwrap())).into();
        write_json(&path, &manifest);
        let result = DevelopmentData::load_with_model(
            root.path(),
            &root.path().join("model"),
            &AtomicBool::new(false),
        );
        if mode == "different" {
            assert!(result.is_ok(), "different audio should remain usable");
        } else {
            let error = result
                .err()
                .expect("target WAV was accepted as another split's input");
            assert!(
                error.contains("target") && error.contains("cross"),
                "{target_in_train}/{mode}: {error}"
            );
        }
    }
}

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
#[test]
fn oversized_wav_duration_is_rejected_before_decoding_its_truncated_body() {
    let root = fixture();
    let path = root.path().join("train-media/train-a.wav");
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
    for _ in 0..(16_000 * 20 + 1) {
        writer.write_sample(0_i16).unwrap();
    }
    writer.finalize().unwrap();
    let mut wav = fs::read(&path).unwrap();
    wav.truncate(44);
    fs::write(&path, &wav).unwrap();
    let manifest_path = root.path().join("train.v3.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    manifest["records"][0]["audio_sha256"] = format!("{:x}", Sha256::digest(&wav)).into();
    write_json(&manifest_path, &manifest);

    let error = DevelopmentData::load(root.path(), &AtomicBool::new(false))
        .err()
        .unwrap();
    assert!(error.contains("duration"), "{error}");
}

#[cfg(unix)]
#[test]
fn repeated_target_audio_counts_against_the_shared_duration_budget() {
    let root = fixture();
    let mut target = speech_target_fixture(root.path());
    let wav = root.path().join("train-target-media/answer.wav");
    let mut writer = hound::WavWriter::create(
        &wav,
        hound::WavSpec {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for _ in 0..(24_000 * 20) {
        writer.write_sample(4096_i16).unwrap();
    }
    writer.finalize().unwrap();
    target["target_audio_sha256"] = format!("{:x}", Sha256::digest(fs::read(&wav).unwrap())).into();

    repeat_train_targets(root.path(), &target, 61);
    let error = DevelopmentData::load_with_model(
        root.path(),
        &root.path().join("model"),
        &AtomicBool::new(false),
    )
    .err()
    .expect("repeated target audio bypassed the budget");
    assert!(error.contains("sample limit"), "{error}");
}

#[cfg(unix)]
#[test]
fn combined_input_and_target_duration_accepts_the_limit_and_rejects_above_it() {
    let root = fixture();
    let mut target = speech_target_fixture(root.path());
    let make_wav = |path: &Path, count: usize| {
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
        for _ in 0..count {
            writer.write_sample(4096_i16).unwrap();
        }
        writer.finalize().unwrap();
        format!("{:x}", Sha256::digest(fs::read(path).unwrap()))
    };
    let full = root.path().join("train-target-media/answer.wav");
    target["target_audio_sha256"] = make_wav(&full, 24_000 * 20).into();
    repeat_train_targets(root.path(), &target, 60);
    let last = root.path().join("train-target-media/last.wav");
    // 60 train inputs and two dev inputs have 4 samples each at 16 kHz.
    // Removing 372 samples at 24 kHz leaves exactly 248 samples for them.
    let frames = 24_000 * 20 - 372;
    let mut final_target = target;
    final_target["audio_file"] = "last.wav".into();
    final_target["target_audio_sha256"] = make_wav(&last, frames).into();
    let path = root.path().join("train.v3.json");
    let mut cases: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    cases["records"][59]["speech_target"] = final_target.clone();
    write_json(&path, &cases);
    let model = root.path().join("model");
    let stop = AtomicBool::new(false);
    assert!(DevelopmentData::load_with_model(root.path(), &model, &stop).is_ok());

    final_target["target_audio_sha256"] = make_wav(&last, frames + 3).into();
    cases["records"][59]["speech_target"] = final_target;
    write_json(&path, &cases);
    let error = DevelopmentData::load_with_model(root.path(), &model, &stop)
        .err()
        .unwrap();
    assert!(error.contains("sample limit"), "{error}");
}

#[cfg(unix)]
#[test]
fn repeated_target_media_counts_against_the_shared_byte_budget() {
    let root = fixture();
    let mut target = speech_target_fixture(root.path());
    let wav = root.path().join("train-target-media/answer.wav");
    fs::OpenOptions::new()
        .write(true)
        .open(&wav)
        .unwrap()
        .set_len(32 * 1024 * 1024)
        .unwrap();
    target["target_audio_sha256"] = format!("{:x}", Sha256::digest(fs::read(&wav).unwrap())).into();
    repeat_train_targets(root.path(), &target, 4);
    let error = DevelopmentData::load_with_model(
        root.path(),
        &root.path().join("model"),
        &AtomicBool::new(false),
    )
    .err()
    .expect("repeated target bytes bypassed the budget");
    assert!(error.contains("media byte limit"), "{error}");
}

#[cfg(unix)]
#[test]
fn combined_input_and_target_bytes_accept_the_limit_and_reject_before_overread() {
    let root = fixture();
    let mut target = speech_target_fixture(root.path());
    let full = root.path().join("train-target-media/answer.wav");
    let file_limit = 32 * 1024 * 1024_u64;
    fs::OpenOptions::new()
        .write(true)
        .open(&full)
        .unwrap()
        .set_len(file_limit)
        .unwrap();
    target["target_audio_sha256"] =
        format!("{:x}", Sha256::digest(fs::read(&full).unwrap())).into();
    repeat_train_targets(root.path(), &target, 4);
    let input_bytes = 4 * fs::metadata(root.path().join("train-media/train-a.wav"))
        .unwrap()
        .len()
        + fs::metadata(root.path().join("dev-media/dev-a.wav"))
            .unwrap()
            .len()
        + fs::metadata(root.path().join("dev-media/dev-b.wav"))
            .unwrap()
            .len();
    let last = root.path().join("train-target-media/last.wav");
    fs::copy(&full, &last).unwrap();
    let last_len = file_limit - input_bytes;
    fs::OpenOptions::new()
        .write(true)
        .open(&last)
        .unwrap()
        .set_len(last_len)
        .unwrap();
    let mut last_target = target;
    last_target["audio_file"] = "last.wav".into();
    last_target["target_audio_sha256"] =
        format!("{:x}", Sha256::digest(fs::read(&last).unwrap())).into();
    let path = root.path().join("train.v3.json");
    let mut cases: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    cases["records"][3]["speech_target"] = last_target;
    write_json(&path, &cases);
    let stop = AtomicBool::new(false);
    let model = root.path().join("model");
    assert!(DevelopmentData::load_with_model(root.path(), &model, &stop).is_ok());

    let last_input = root.path().join("dev-media/dev-b.wav");
    let input_len = fs::metadata(&last_input).unwrap().len();
    fs::OpenOptions::new()
        .write(true)
        .open(&last_input)
        .unwrap()
        .set_len(input_len + 1)
        .unwrap();
    let error = DevelopmentData::load_with_model(root.path(), &model, &stop)
        .err()
        .unwrap();
    assert!(error.contains("media byte limit"), "{error}");
}

#[cfg(unix)]
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

#[cfg(unix)]
#[test]
fn licensed_human_reference_remains_unverified_without_fictitious_consent() {
    let root = fixture();
    artifacts(root.path());
    let path = root.path().join("train.v3.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let declaration = json!({"kind":"licensed-human","source":"public corpus fixture",
        "license":"CC-BY-4.0","published_prompt":"published reference text"});
    manifest["records"][0]["declaration"] = declaration.clone();
    write_json(&path, &manifest);
    let stop = AtomicBool::new(false);
    let data = DevelopmentData::load(root.path(), &stop).unwrap();
    assert_eq!(
        data.train()[0].declaration()["published_prompt"],
        "published reference text"
    );
    assert_eq!(data.train()[0].reference_status(), "unverified");
    assert!(!data.train()[0].declaration().contains_key("consent"));
    assert!(
        !data.train()[0]
            .declaration()
            .contains_key("human_reference")
    );
    let destination = root.path().join("frozen");
    let registered = preregister(&options(root.path()), &destination, &stop).unwrap();
    assert_eq!(registered.status(), "insufficient_evidence");
    assert!(!registered.candidate_eligible());
    let envelope: Value =
        serde_json::from_slice(&fs::read(destination.join("protocol.json")).unwrap()).unwrap();
    assert_eq!(
        envelope["protocol"]["policy"]["reference_verification"],
        "unverified"
    );

    for invalid in [
        json!({"kind":"licensed-human","license":"CC-BY-4.0"}),
        json!({"kind":"licensed-human","source":"   ","license":"CC-BY-4.0"}),
        json!({"kind":"licensed-human","source":"corpus"}),
        json!({"kind":"licensed-human","source":"corpus","license":"  "}),
        json!({"kind":"licensed-human","source":"corpus","license":"CC-BY-4.0","reference_status":"verified"}),
    ] {
        manifest["records"][0]["declaration"] = invalid;
        write_json(&path, &manifest);
        assert!(DevelopmentData::load(root.path(), &stop).is_err());
    }
}

#[cfg(unix)]
#[test]
fn licensed_human_target_source_is_a_declaration_not_a_pass() {
    let root = fixture();
    let mut target = speech_target_fixture(root.path());
    target["source"] = json!({"kind":"licensed-human","source":"public corpus fixture",
        "license":"CC-BY-4.0","published_prompt":"published reference text"});
    replace_train_target(root.path(), target.clone());
    let stop = AtomicBool::new(false);
    let model = root.path().join("model");
    let data = DevelopmentData::load_with_model(root.path(), &model, &stop).unwrap();
    assert_eq!(
        data.train()[0].speech_target().unwrap().source()["kind"],
        "licensed-human"
    );
    let registered =
        preregister(&options(root.path()), &root.path().join("frozen"), &stop).unwrap();
    assert!(!registered.candidate_eligible());
    target["source"]["reference_status"] = "verified".into();
    replace_train_target(root.path(), target);
    assert!(DevelopmentData::load_with_model(root.path(), &model, &stop).is_err());
}

#[cfg(unix)]
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

#[cfg(unix)]
#[test]
fn registered_swap_view_is_the_mapping_bound_to_the_protocol_hash() {
    let root = fixture();
    artifacts(root.path());
    let options = options(root.path());
    let destination = root.path().join("frozen");
    let stop = AtomicBool::new(false);
    let registered = preregister(&options, &destination, &stop).unwrap();
    let envelope_path = destination.join("protocol.json");
    let mut envelope: Value = serde_json::from_slice(&fs::read(&envelope_path).unwrap()).unwrap();
    let view = registered.swaps();
    assert_eq!(view.protocol_sha256(), registered.protocol_sha256());
    assert_eq!(
        view.protocol_sha256(),
        envelope["protocol_sha256"].as_str().unwrap()
    );
    let stored = envelope["protocol"]["swaps"].as_array().unwrap();
    assert_eq!(view.entries().len(), stored.len());
    for (entry, expected) in view.entries().iter().zip(stored) {
        assert_eq!(entry.case_id(), expected["case_id"].as_str().unwrap());
        assert_eq!(
            entry.audio_case_id(),
            expected["audio_case_id"].as_str().unwrap()
        );
        let records = registered.data().dev();
        let case = records
            .iter()
            .find(|row| row.id() == entry.case_id())
            .unwrap();
        let audio = records
            .iter()
            .find(|row| row.id() == entry.audio_case_id())
            .unwrap();
        assert_ne!(case.answer(), audio.answer());
    }
    let verified = verify_registration(&options, &destination, &stop).unwrap();
    assert_eq!(verified.swaps().protocol_sha256(), view.protocol_sha256());
    assert_eq!(
        verified
            .swaps()
            .entries()
            .iter()
            .map(|entry| (entry.case_id(), entry.audio_case_id()))
            .collect::<Vec<_>>(),
        view.entries()
            .iter()
            .map(|entry| (entry.case_id(), entry.audio_case_id()))
            .collect::<Vec<_>>()
    );
    assert!(!verified.candidate_eligible());

    envelope["protocol"]["swaps"][0]["audio_case_id"] = "dev-a".into();
    write_json(&envelope_path, &envelope);
    assert!(verify_registration(&options, &destination, &stop).is_err());
}

#[cfg(unix)]
#[test]
fn preregistration_records_missing_and_declared_origin_without_verifying_it() {
    let root = fixture();
    let mut target = speech_target_fixture(root.path());
    target["source"]["created_at"] = "2026-11-10T12:00:00Z".into();
    target["source"]["teacher_sha256"] = "a".repeat(64).into();
    target["source"]["verifier_sha256"] = "b".repeat(64).into();
    replace_train_target(root.path(), target);
    let path = root.path().join("train.v3.json");
    let mut cases: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    cases["records"][0]["declaration"]["created_at"] = "2026-11-09T12:00:00Z".into();
    cases["records"][0]["declaration"]["teacher_sha256"] = "c".repeat(64).into();
    write_json(&path, &cases);
    let dev_path = root.path().join("dev.v3.json");
    let mut dev_cases: Value = serde_json::from_slice(&fs::read(&dev_path).unwrap()).unwrap();
    dev_cases["records"][0]["declaration"]["created_at"] = "unknown".into();
    dev_cases["records"][0]["declaration"]["teacher_sha256"] = "not-a-digest".into();
    dev_cases["records"][1]["declaration"]["created_at"] = "2024-02-29T12:00:00.5+08:00".into();
    dev_cases["records"][1]["declaration"]["teacher_sha256"] = "d".repeat(64).into();
    dev_cases["records"][1]["declaration"]["verifier_sha256"] = "e".repeat(64).into();
    write_json(&dev_path, &dev_cases);
    let dest = root.path().join("frozen");
    let frozen = preregister(&options(root.path()), &dest, &AtomicBool::new(false)).unwrap();
    assert!(!frozen.candidate_eligible());
    let envelope: Value =
        serde_json::from_slice(&fs::read(dest.join("protocol.json")).unwrap()).unwrap();
    let input = &envelope["protocol"]["train_cases"][0]["source_provenance"];
    assert_eq!(input["status"], "incomplete");
    assert_eq!(input["created_at"], "2026-11-09T12:00:00Z");
    assert_eq!(input["teacher_sha256"], "c".repeat(64));
    assert!(input["verifier_sha256"].is_null());
    assert_eq!(input["missing"], json!(["verifier_sha256"]));
    let target = &envelope["protocol"]["train_cases"][0]["target_provenance"];
    assert_eq!(target["status"], "declared_unverified");
    assert_eq!(target["missing"], json!([]));
    let dev = &envelope["protocol"]["dev_cases"][0]["source_provenance"];
    assert_eq!(dev["status"], "incomplete");
    assert!(
        dev["created_at"].is_null()
            && dev["teacher_sha256"].is_null()
            && dev["verifier_sha256"].is_null()
    );
    let dated = &envelope["protocol"]["dev_cases"][1]["source_provenance"];
    assert_eq!(dated["status"], "declared_unverified");
    assert_eq!(dated["created_at"], "2024-02-29T12:00:00.5+08:00");
    assert!(envelope["protocol"]["r2t2_sha256"].is_null());
    assert!(envelope["protocol"]["breeze_sha256"].is_null());
}

#[cfg(unix)]
#[test]
fn preregistration_requires_a_distinct_answer_one_to_one_audio_swap() {
    for (numbers, feasible) in [
        (
            (0..30)
                .map(|i| if i == 29 { 15 } else { 12 })
                .collect::<Vec<_>>(),
            false,
        ),
        (vec![12, 12, 15, 15], true),
        (vec![12, 12, 15, 15, 18], true),
    ] {
        let root = fixture();
        artifacts(root.path());
        let mut records = Vec::new();
        let mut groups = Vec::new();
        for (i, number) in numbers.iter().enumerate() {
            let id = format!("swap-{i}");
            let record = case(
                root.path(),
                "dev",
                &id,
                *number,
                tagged_source(&id),
                1000 + i as i16,
            );
            groups.push(json!({"id":record["id"],"source":record["source"]}));
            records.push(record);
        }
        write_json(
            &root.path().join("dev.v3.json"),
            &json!({"version":3,"partition":"dev",
            "rule":"number-plus-one-v1","records":records}),
        );
        write_json(
            &root.path().join("dev.groups.v3.json"),
            &json!({"version":3,"partition":"dev","records":groups}),
        );
        let dest = root.path().join("frozen");
        let result = preregister(&options(root.path()), &dest, &AtomicBool::new(false));
        if !feasible {
            let error = result.err().expect("29:1 distribution must not be frozen");
            assert!(error.contains("one-to-one"), "{error}");
            assert!(!dest.exists());
            continue;
        }
        assert!(!result.unwrap().candidate_eligible());
        let envelope: Value =
            serde_json::from_slice(&fs::read(dest.join("protocol.json")).unwrap()).unwrap();
        let swaps = envelope["protocol"]["swaps"].as_array().unwrap();
        let cases = envelope["protocol"]["dev_cases"].as_array().unwrap();
        assert_eq!(
            envelope["protocol"]["policy"]["audio_swaps"]["mapping"],
            "different-answer-bijection-v1"
        );
        assert_eq!(
            envelope["protocol"]["policy"]["audio_swaps"]["distinct_pcm"],
            numbers.len()
        );
        let mut selected = std::collections::HashSet::new();
        let mut selected_pcm = std::collections::HashSet::new();
        for swap in swaps {
            let case_id = swap["case_id"].as_str().unwrap();
            let audio_id = swap["audio_case_id"].as_str().unwrap();
            assert!(selected.insert(audio_id));
            let answer = |id: &str| {
                cases.iter().find(|row| row["id"] == id).unwrap()["answer"]
                    .as_u64()
                    .unwrap()
            };
            assert_ne!(answer(case_id), answer(audio_id));
            let audio = cases.iter().find(|row| row["id"] == audio_id).unwrap();
            assert!(selected_pcm.insert(audio["pcm_sha256"].as_str().unwrap()));
            assert!(audio["source"]["speaker_sha256"].as_str().is_some());
        }
        assert_eq!(selected.len(), numbers.len());
    }
}

#[cfg(unix)]
#[test]
fn repeated_development_pcm_cannot_be_counted_as_distinct_swap_audio() {
    let root = fixture();
    artifacts(root.path());
    let cases_path = root.path().join("dev.v3.json");
    let groups_path = root.path().join("dev.groups.v3.json");
    let mut cases: Value = serde_json::from_slice(&fs::read(&cases_path).unwrap()).unwrap();
    let mut groups: Value = serde_json::from_slice(&fs::read(&groups_path).unwrap()).unwrap();
    fs::hard_link(
        root.path().join("dev-media/dev-a.wav"),
        root.path().join("dev-media/duplicate.wav"),
    )
    .unwrap();
    let mut duplicate = cases["records"][0].clone();
    duplicate["id"] = "dev-duplicate".into();
    duplicate["audio_file"] = "duplicate.wav".into();
    let other = case(
        root.path(),
        "dev",
        "dev-c",
        15,
        tagged_source("dev-c"),
        13_000,
    );
    for row in [duplicate, other] {
        groups["records"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id":row["id"],"source":row["source"]}));
        cases["records"].as_array_mut().unwrap().push(row);
    }
    write_json(&cases_path, &cases);
    write_json(&groups_path, &groups);
    let stop = AtomicBool::new(false);
    assert!(DevelopmentData::load(root.path(), &stop).is_ok());
    let dest = root.path().join("frozen");
    let error = preregister(&options(root.path()), &dest, &stop)
        .err()
        .expect("repeated PCM was counted as independent swap audio");
    assert!(error.contains("unique") && error.contains("PCM"), "{error}");
    assert!(!dest.exists());
}

#[cfg(unix)]
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

#[cfg(unix)]
#[test]
fn older_preregistration_is_rejected_without_overwriting_its_protocol() {
    let root = fixture();
    artifacts(root.path());
    let options = options(root.path());
    let destination = root.path().join("frozen");
    let stop = AtomicBool::new(false);
    preregister(&options, &destination, &stop).unwrap();
    let path = destination.join("protocol.json");
    let mut envelope: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    for split in ["train_cases", "dev_cases"] {
        for case in envelope["protocol"][split].as_array_mut().unwrap() {
            case.as_object_mut().unwrap().remove("source_provenance");
            case.as_object_mut().unwrap().remove("target_provenance");
        }
    }
    envelope["protocol"]["policy"]
        .as_object_mut()
        .unwrap()
        .remove("audio_swaps");
    let bytes = serde_json::to_vec(&envelope["protocol"]).unwrap();
    let mut hash = Sha256::new();
    hash.update(b"omni-rsi-number-development-preregistration-v1\0");
    hash.update((bytes.len() as u64).to_be_bytes());
    hash.update(bytes);
    envelope["protocol_sha256"] = format!("{:x}", hash.finalize()).into();
    write_json(&path, &envelope);
    let original = fs::read(&path).unwrap();
    let error = verify_registration(&options, &destination, &stop)
        .err()
        .unwrap();
    assert!(error.contains("changed"), "{error}");
    assert_eq!(fs::read(&path).unwrap(), original);
}

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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
fn indexed_codec_fingerprint_rejects_symlinked_index_and_shard_directory() {
    for index_is_symlink in [false, true] {
        let root = fixture();
        artifacts(root.path());
        let codec = root.path().join("model/audio_tokenizer");
        let outside = root.path().join("unrelated-codec");
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("shard.safetensors"), b"synthetic shard").unwrap();
        let index = codec.join("model.safetensors.index.json");
        if index_is_symlink {
            write_json(
                &outside.join("index.json"),
                &json!({"weight_map":{"a":"shard.safetensors"}}),
            );
            std::os::unix::fs::symlink(outside.join("index.json"), &index).unwrap();
        } else {
            write_json(
                &index,
                &json!({"weight_map":{"a":"parts/shard.safetensors"}}),
            );
            std::os::unix::fs::symlink(&outside, codec.join("parts")).unwrap();
        }
        let error = preregister(
            &options(root.path()),
            &root.path().join("frozen"),
            &AtomicBool::new(false),
        )
        .err()
        .expect("followed a symlink in the indexed codec");
        assert!(
            error.contains("symlink") || error.contains("loop"),
            "{error}"
        );
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(not(unix))]
#[test]
fn unsupported_platform_fails_closed_before_data_or_registration_access() {
    let root = tempfile::tempdir().unwrap();
    let stop = AtomicBool::new(false);
    let model = root.path().join("model");
    let data_error = DevelopmentData::load(root.path(), &stop).err().unwrap();
    assert!(data_error.contains("unsupported"), "{data_error}");
    let target_error = DevelopmentData::load_with_model(root.path(), &model, &stop)
        .err()
        .unwrap();
    assert!(target_error.contains("unsupported"), "{target_error}");
    let destination = root.path().join("frozen");
    let error = preregister(&options(root.path()), &destination, &stop)
        .err()
        .unwrap();
    assert!(error.contains("unsupported"), "{error}");
    assert!(!destination.exists());
    let error = verify_registration(&options(root.path()), &destination, &stop)
        .err()
        .unwrap();
    assert!(error.contains("unsupported"), "{error}");
}
