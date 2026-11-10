use omni_rsi::speech_seed::teacher_fingerprint;
use std::fs;

#[test]
fn native_teacher_fingerprint_binds_weights_and_executable() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    fs::create_dir(&model).unwrap();
    fs::write(model.join("model.safetensors"), "first weights").unwrap();
    fs::write(model.join("config.json"), "{}").unwrap();
    let binary = root.path().join("breeze-tts");
    fs::write(&binary, "first binary").unwrap();
    let first = teacher_fingerprint(&model, &binary).unwrap();
    fs::write(&binary, "second binary").unwrap();
    let second = teacher_fingerprint(&model, &binary).unwrap();
    assert_ne!(first, second);
    fs::write(model.join("model.safetensors"), "second weights").unwrap();
    assert_ne!(second, teacher_fingerprint(&model, &binary).unwrap());
}

#[cfg(all(feature = "cpu", unix))]
#[test]
#[ignore = "requires local LFM3 codec weights and an existing Breeze audio fixture"]
fn native_seed_uses_codec_and_resumes_without_overwriting() {
    use burn::backend::NdArray;
    use omni_rsi::speech_seed::{SeedOptions, prepare};
    use std::{os::unix::fs::PermissionsExt, path::PathBuf, sync::atomic::AtomicBool};

    let path = |name| PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("set {name}")));
    let model = path("LFM3_MODEL");
    let answers = path("LFM3_SPEECH_DATA");
    let fixture = answers.join("source/greeting.wav").canonicalize().unwrap();
    let root = tempfile::tempdir().unwrap();
    let data = root.path().join("new-data");
    let tts_model = root.path().join("breeze");
    let asr = root.path().join("asr");
    fs::create_dir(&tts_model).unwrap();
    fs::create_dir(&asr).unwrap();
    fs::write(tts_model.join("model.safetensors"), "fixture").unwrap();
    for filename in [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
    ] {
        fs::write(asr.join(filename), filename).unwrap();
    }
    let tts_binary = root.path().join("breeze-tts");
    let escaped = fixture.to_str().unwrap().replace('\'', "'\\''");
    let script = format!(
        "#!/bin/sh\nwhile [ \"$#\" -gt 0 ]; do if [ \"$1\" = --output ]; then shift; dest=$1; fi; shift; done\ncp '{escaped}' \"$dest\"\nprintf '%s\\n' '{{\"stopped\":false,\"reached_limit\":false}}'\n"
    );
    fs::write(&tts_binary, script).unwrap();
    fs::set_permissions(&tts_binary, fs::Permissions::from_mode(0o700)).unwrap();
    let asr_binary = root.path().join("r2t2");
    fs::write(&asr_binary, "#!/bin/sh\nprintf '%s\\n' '{\"text\":\"你好，我是你的朋友。\",\"language\":\"Chinese\",\"is_final\":true,\"cancelled\":false,\"token_limit_reached\":false}'\n").unwrap();
    fs::set_permissions(&asr_binary, fs::Permissions::from_mode(0o700)).unwrap();
    let options = SeedOptions {
        model: &model,
        data: &data,
        answers_from: Some(&answers),
        only: Some("greeting"),
        tts_model: &tts_model,
        tts_binary: &tts_binary,
        asr: &asr,
        asr_binary: &asr_binary,
    };
    let stop = AtomicBool::new(false);
    let result = prepare::<NdArray<f32>>(&options, &Default::default(), &stop).unwrap();
    assert_eq!(result[0].state, "verified");
    let record_path = data.join("samples/greeting/record.json");
    let bytes = fs::read(&record_path).unwrap();
    let record: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(record["verified"], true);
    assert_eq!(
        prepare::<NdArray<f32>>(&options, &Default::default(), &stop).unwrap()[0].state,
        "existing"
    );
    assert_eq!(bytes, fs::read(&record_path).unwrap());
    let mut altered = record;
    altered["frames"][0][0] = serde_json::json!(9999);
    fs::write(&record_path, serde_json::to_vec(&altered).unwrap()).unwrap();
    assert!(prepare::<NdArray<f32>>(&options, &Default::default(), &stop).is_err());
}
