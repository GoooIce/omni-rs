#![cfg(feature = "native")]

use std::process::Command;

#[test]
fn audio_training_has_an_explicit_native_entry_and_resume_flag() {
    let output = Command::new(env!("CARGO_BIN_EXE_omni-rsi-audio"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    for option in [
        "--model",
        "--data",
        "--state",
        "--epochs",
        "--max-updates",
        "--resume",
    ] {
        assert!(help.contains(option), "missing {option}");
    }
    assert!(!help.contains("--python"));
}

#[test]
fn v3_training_is_an_explicit_cli_choice() {
    let output = Command::new(env!("CARGO_BIN_EXE_omni-rsi-audio"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    assert!(help.contains("--number-v3"), "{help}");
}

#[cfg(unix)]
#[test]
fn v3_cli_resume_uses_its_own_checkpoint_namespace() {
    use std::fs;
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let legacy = state.join("audio-conditioned-v1");
    fs::create_dir_all(&legacy).unwrap();
    fs::write(legacy.join("marker"), b"unchanged").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_omni-rsi-audio"))
        .env("DDUST_RUN_REGISTRY", root.path().join("registry.json"))
        .arg("--model")
        .arg(root.path())
        .arg("--data")
        .arg(root.path())
        .arg("--state")
        .arg(&state)
        .args(["--number-v3", "--resume"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no audio training state to resume"));
    assert_eq!(fs::read(legacy.join("marker")).unwrap(), b"unchanged");
    assert!(state.join("number-audio-v3").is_dir());
}

#[cfg(unix)]
#[test]
#[ignore = "requires an explicit local LFM3 checkpoint and a v3 dataset with speech then text targets"]
fn v3_real_cli_trains_speech_and_text_then_resumes() {
    use omni_rsi::number_audio_v3::NumberAudioV3;
    use std::{fs, path::PathBuf, sync::atomic::AtomicBool};
    let model = PathBuf::from(std::env::var_os("LFM3_MODEL").expect("LFM3_MODEL"));
    let source =
        PathBuf::from(std::env::var_os("LFM3_NUMBER_V3_DATA").expect("LFM3_NUMBER_V3_DATA"));
    let root = tempfile::tempdir().unwrap();
    let data = root.path().join("v3-data");
    fs::create_dir(&data).unwrap();
    for name in [
        "layout.v3",
        "holdout-public.json",
        "train.groups.v3.json",
        "dev.groups.v3.json",
        "train.v3.json",
        "dev.v3.json",
    ] {
        fs::copy(source.join(name), data.join(name)).unwrap();
    }
    for name in [
        "train-media",
        "dev-media",
        "train-target-media",
        "dev-target-media",
    ] {
        let from = source.join(name);
        if !from.exists() {
            continue;
        }
        let to = data.join(name);
        fs::create_dir(&to).unwrap();
        for entry in fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            assert!(entry.file_type().unwrap().is_file());
            fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
        }
    }
    // A declaration about preparation on another backend is not evidence and
    // must not reject frames that the current codec actually reproduces.
    let train_path = data.join("train.v3.json");
    let mut train: serde_json::Value =
        serde_json::from_slice(&fs::read(&train_path).unwrap()).unwrap();
    assert_eq!(
        train["records"][0]["speech_target"]["encoding"],
        "local_liquidcodec_f32"
    );
    train["records"][0]["speech_target"]["runtime_backend"] = "different-historical-backend".into();
    fs::write(train_path, serde_json::to_vec(&train).unwrap()).unwrap();
    let snapshot = NumberAudioV3::load(&data, &model, &AtomicBool::new(false)).unwrap();
    assert!(snapshot.len() >= 2);
    assert!(snapshot.example(0).unwrap().frames().is_some());
    assert!(snapshot.example(1).unwrap().frames().is_none());
    let state = root.path().join("state");
    let run = |resume: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_omni-rsi-audio"));
        command
            .env("DDUST_RUN_REGISTRY", root.path().join("registry.json"))
            .arg("--model")
            .arg(&model)
            .arg("--data")
            .arg(&data)
            .arg("--state")
            .arg(&state)
            .args(["--number-v3", "--epochs", "1", "--max-updates", "1"]);
        if resume {
            command.arg("--resume");
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice::<serde_json::Value>(&output.stdout).unwrap()
    };
    assert_eq!(run(false)["progress"]["updates"], 1);
    assert_eq!(run(true)["progress"]["updates"], 2);
    assert!(
        state
            .join("number-audio-v3/audio-conditioned-v1/latest.json")
            .is_file()
    );
    assert!(!state.join("audio-conditioned-v1/latest.json").exists());
}

#[cfg(unix)]
#[test]
fn audio_training_respects_the_ddust_global_device_lease() {
    use std::{fs::OpenOptions, os::fd::AsRawFd};
    let root = tempfile::tempdir().unwrap();
    let lock = OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.path().join("mps.lock"))
        .unwrap();
    assert_eq!(
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    let output = Command::new(env!("CARGO_BIN_EXE_omni-rsi-audio"))
        .env("DDUST_RUN_REGISTRY", root.path().join("registry.json"))
        .args([
            "--model",
            root.path().to_str().unwrap(),
            "--data",
            root.path().to_str().unwrap(),
            "--state",
            root.path().join("state").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("ddust accelerator lease is occupied")
    );
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires the official LFM3 checkpoint and audio reference"]
fn native_cli_trains_with_the_supplied_speech_targets() {
    use sha2::{Digest, Sha256};
    use std::{fs, path::PathBuf};
    let model = PathBuf::from(std::env::var("LFM3_MODEL").unwrap());
    let reference: serde_json::Value = serde_json::from_slice(
        &fs::read(std::env::var("LFM3_AUDIO_TRAIN_REFERENCE").unwrap()).unwrap(),
    )
    .unwrap();
    let root = tempfile::tempdir().unwrap();
    let data = root.path().join("data");
    let state = root.path().join("state");
    fs::create_dir(&data).unwrap();
    let wav = data.join("synthetic.wav");
    let mut writer = hound::WavWriter::create(
        &wav,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for i in 0..6400 {
        writer
            .write_sample(((i * 37 % 127 - 63) * 4) as i16)
            .unwrap();
    }
    writer.finalize().unwrap();
    let audio_hash = format!("{:x}", Sha256::digest(fs::read(&wav).unwrap()));
    let codec_hash =
        omni_rsi::speech_data::file_hash(&model.join("audio_tokenizer/model.safetensors")).unwrap();
    let manifest = serde_json::json!({"version":1,"records":[{
        "id":"fixture", "split":"train", "input_audio":"synthetic.wav", "input_sha256":audio_hash,
        "authorization":{"kind":"licensed-synthetic","source":"deterministic test waveform","license":"CC0-1.0"},
        "instruction":"把听到的数字加一，用中文回答。", "answer":reference["answer"],
        "frames":reference["frames"], "target_codec_sha256":codec_hash
    }]});
    fs::write(
        data.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_omni-rsi-audio"))
        .env("PATH", root.path())
        .env("DDUST_RUN_REGISTRY", root.path().join("registry.json"))
        .arg("--model")
        .arg(model)
        .arg("--data")
        .arg(data)
        .arg("--state")
        .arg(state)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["progress"]["updates"], 1);
    assert_eq!(result["status"], "complete");
    assert_eq!(result["checkpoint"].as_str().unwrap().len(), 64);
}
