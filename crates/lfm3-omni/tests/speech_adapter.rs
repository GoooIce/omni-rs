use omni_lfm3::speech_adapter::VerifiedAdapter;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

fn digest(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn package(directory: &Path) -> Value {
    let evaluation = include_bytes!("../../../docs/eval/chinese-speech-generation-v1.json");
    let plan: Value = serde_json::from_slice(evaluation).unwrap();
    let mut observations = Vec::new();
    for case in plan["dev"].as_array().unwrap() {
        let reference = case["reference"].as_str().unwrap();
        for (kind, transcript) in [
            ("baseline", reference.chars().skip(1).collect::<String>()),
            ("candidate", reference.to_owned()),
        ] {
            observations.push(
                json!({"sample":case["id"], "category":case["category"], "kind":kind,
                                     "frames":12, "stopped":true, "transcript":transcript}),
            );
        }
    }
    let weights =
        serde_json::to_vec(&json!({"a":vec![0.02_f32; 64],"b":vec![0.01_f32; 64]})).unwrap();
    fs::write(directory.join("adapter.json"), &weights).unwrap();
    let manifest = json!({"version":1,"base_hash":"a".repeat(64),"layer":23,"rank":4,"width":16,
                          "asr_sha256":"d".repeat(64),
                          "data_sha256":"f".repeat(64),
                          "weights_sha256":digest(&weights),"evaluation_sha256":digest(evaluation),
                          "results":observations});
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    manifest
}

#[test]
fn only_matching_untampered_packages_with_a_passing_gate_are_accepted() {
    let root = tempfile::tempdir().unwrap();
    let mut manifest = package(root.path());
    let loaded = VerifiedAdapter::load(root.path(), &"a".repeat(64), 16).unwrap();
    assert_eq!(
        (loaded.layer, loaded.rank, loaded.a.len(), loaded.b.len()),
        (23, 4, 64, 64)
    );
    assert!(VerifiedAdapter::load(root.path(), &"b".repeat(64), 16).is_err());
    manifest["results"][1]["stopped"] = false.into();
    fs::write(
        root.path().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    assert!(VerifiedAdapter::load(root.path(), &"a".repeat(64), 16).is_err());
    let mut manifest = package(root.path());
    manifest["asr_sha256"] = "invalid".into();
    fs::write(
        root.path().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    assert!(VerifiedAdapter::load(root.path(), &"a".repeat(64), 16).is_err());
    package(root.path());
    fs::write(root.path().join("adapter.json"), b"{}").unwrap();
    assert!(VerifiedAdapter::load(root.path(), &"a".repeat(64), 16).is_err());
}

#[test]
#[ignore = "requires the local full model and an independently computed Python fingerprint"]
fn full_model_fingerprint_matches_python() {
    let directory = std::env::var_os("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required");
    let expected =
        std::env::var("LFM3_EXPECTED_SOURCE_HASH").expect("Python source hash is required");
    assert_eq!(
        omni_lfm3::speech_adapter::source_fingerprint(Path::new(&directory)).unwrap(),
        expected
    );
}

#[test]
fn a_missing_or_duplicate_evaluation_result_cannot_certify_an_adapter() {
    let root = tempfile::tempdir().unwrap();
    let mut manifest = package(root.path());
    let entries = manifest["results"].as_array_mut().unwrap();
    entries[1] = entries[0].clone();
    fs::write(
        root.path().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    assert!(VerifiedAdapter::load(root.path(), &"a".repeat(64), 16).is_err());
}

#[test]
fn model_fingerprint_matches_the_training_protocol() {
    let root = tempfile::tempdir().unwrap();
    let names = [
        "model.safetensors",
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "helper.py",
        "audio_tokenizer/config.json",
        "audio_tokenizer/model.safetensors",
        "audio_tokenizer/nested/codec.py",
    ];
    for name in names {
        let path = root.path().join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, format!("{name}\n")).unwrap();
    }
    assert_eq!(
        omni_lfm3::speech_adapter::source_fingerprint(root.path()).unwrap(),
        "0b18187cb93ae794dd516170f2da8b01a0406eea316ce32701d39263987ffb57"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let codec = root.path().join("audio_tokenizer");
        fs::rename(
            codec.join("config.json"),
            root.path().join("codec-config.data"),
        )
        .unwrap();
        symlink("../codec-config.data", codec.join("config.json")).unwrap();
        let first = omni_lfm3::speech_adapter::source_fingerprint(root.path()).unwrap();
        fs::write(
            root.path().join("codec-config.data"),
            b"changed codec config",
        )
        .unwrap();
        assert_ne!(
            first,
            omni_lfm3::speech_adapter::source_fingerprint(root.path()).unwrap()
        );
        fs::rename(
            codec.join("model.safetensors"),
            root.path().join("codec-weights.data"),
        )
        .unwrap();
        symlink("../codec-weights.data", codec.join("model.safetensors")).unwrap();
        let first = omni_lfm3::speech_adapter::source_fingerprint(root.path()).unwrap();
        fs::write(
            root.path().join("codec-weights.data"),
            b"changed codec weights",
        )
        .unwrap();
        assert_ne!(
            first,
            omni_lfm3::speech_adapter::source_fingerprint(root.path()).unwrap()
        );
    }
}

#[test]
fn indexed_model_fingerprint_tracks_the_index_and_loaded_shards() {
    let root = tempfile::tempdir().unwrap();
    for name in [
        "model.safetensors",
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "audio_tokenizer/config.json",
        "audio_tokenizer/model.safetensors",
        "model-01.safetensors",
        "model-02.safetensors",
    ] {
        let path = root.path().join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, format!("{name}\n")).unwrap();
    }
    let index = root.path().join("model.safetensors.index.json");
    fs::write(
        &index,
        br#"{"weight_map":{"a":"model-01.safetensors","b":"model-02.safetensors"}}"#,
    )
    .unwrap();
    let fingerprint = || omni_lfm3::speech_adapter::source_fingerprint(root.path());
    let first = fingerprint().unwrap();
    fs::write(root.path().join("model.safetensors"), b"unused single file").unwrap();
    assert_eq!(first, fingerprint().unwrap());
    fs::write(root.path().join("model-02.safetensors"), b"changed shard").unwrap();
    let changed = fingerprint().unwrap();
    assert_ne!(first, changed);
    fs::write(
        &index,
        br#"{ "weight_map":{"a":"model-01.safetensors","b":"model-02.safetensors"}}"#,
    )
    .unwrap();
    assert_ne!(changed, fingerprint().unwrap());
    fs::remove_file(root.path().join("model.safetensors")).unwrap();
    assert!(fingerprint().is_ok());
    fs::remove_file(root.path().join("model-01.safetensors")).unwrap();
    assert!(fingerprint().is_err());
}

#[test]
fn indexed_codec_fingerprint_tracks_the_index_and_loaded_shards() {
    let root = tempfile::tempdir().unwrap();
    for name in [
        "model.safetensors",
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "audio_tokenizer/config.json",
        "audio_tokenizer/part.safetensors",
        "audio_tokenizer/model.safetensors",
    ] {
        let path = root.path().join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, format!("{name}\n")).unwrap();
    }
    let codec = root.path().join("audio_tokenizer");
    let index = codec.join("model.safetensors.index.json");
    fs::write(&index, br#"{"weight_map":{"a":"part.safetensors"}}"#).unwrap();
    let fingerprint = || omni_lfm3::speech_adapter::source_fingerprint(root.path());
    let first = fingerprint().unwrap();
    fs::write(codec.join("model.safetensors"), b"unused single file").unwrap();
    assert_eq!(first, fingerprint().unwrap());
    fs::write(&index, br#"{ "weight_map":{"a":"part.safetensors"}}"#).unwrap();
    let changed_index = fingerprint().unwrap();
    assert_ne!(first, changed_index);
    fs::write(codec.join("part.safetensors"), b"changed shard").unwrap();
    assert_ne!(changed_index, fingerprint().unwrap());
    fs::remove_file(codec.join("model.safetensors")).unwrap();
    assert!(fingerprint().is_ok());
    fs::remove_file(codec.join("part.safetensors")).unwrap();
    assert!(fingerprint().is_err());
}

#[test]
fn rust_loads_the_python_exported_synthetic_package() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/speech_adapter");
    let adapter = VerifiedAdapter::load(&directory, &"a".repeat(64), 16).unwrap();
    assert_eq!(adapter.a, vec![0.1; 64]);
    assert_eq!(adapter.b, vec![0.2; 64]);
}
