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
