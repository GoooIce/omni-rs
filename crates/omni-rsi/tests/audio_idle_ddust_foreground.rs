#![cfg(all(unix, feature = "native", feature = "cpu"))]

use omni_rsi::lease::Lease;
use sha2::{Digest, Sha256};
use std::{
    fs,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

struct Reap(Child);
impl Drop for Reap {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn foreground_audio_holds_configured_ddust_lock_before_loading_synthetic_data() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    for name in ["home", "ddust", "model", "data"] {
        fs::create_dir(root.join(name)).unwrap();
    }
    let registry = root.join("ddust/registry.json");
    let lock = root.join("ddust/mps.lock");
    fs::write(&registry, br#"{"version":1,"runs":{}}"#).unwrap();
    let wav = root.join("data/input.wav");
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
    for index in 0..24_000 * 20 {
        writer
            .write_sample(if index % 2 == 0 { 4096_i16 } else { -4096_i16 })
            .unwrap();
    }
    writer.finalize().unwrap();
    let hash = format!("{:x}", Sha256::digest(fs::read(&wav).unwrap()));
    let records: Vec<_> = (0..60).map(|index| serde_json::json!({
        "id":format!("synthetic-{index}"),"split":"train","input_audio":"input.wav",
        "input_sha256":hash,"authorization":{"kind":"licensed-synthetic","source":"fixture","license":"CC0-1.0"},
        "instruction":"加一回答","answer":"十三"
    })).collect();
    fs::write(
        root.join("data/manifest.json"),
        serde_json::to_vec(&serde_json::json!({"version":1,"records":records})).unwrap(),
    )
    .unwrap();
    let state = root.join("state");
    let child = Command::new(env!("CARGO_BIN_EXE_omni-rsi-audio"))
        .env("HOME", root.join("home"))
        .env("DDUST_RUN_REGISTRY", &registry)
        .arg("--model")
        .arg(root.join("model"))
        .arg("--data")
        .arg(root.join("data"))
        .arg("--state")
        .arg(&state)
        .args(["--epochs", "1", "--max-updates", "1"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut child = Reap(child);
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "synthetic foreground exited before observation"
        );
        if state.join("audio-training.lock").exists()
            && Lease::try_acquire(&state.join("audio-training.lock"))
                .unwrap()
                .is_none()
            && Lease::try_acquire(&state.join("device.lock"))
                .unwrap()
                .is_none()
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "foreground did not start training with the device"
        );
        thread::sleep(Duration::from_millis(2));
    }
    let existed_before_peer = lock.exists();
    let competitor = Lease::try_acquire(&lock).unwrap();
    let peer_acquired = competitor.is_some();
    drop(competitor);
    let pid = i32::try_from(child.0.id()).unwrap();
    assert_eq!(unsafe { libc::kill(pid, libc::SIGINT) }, 0);
    let deadline = Instant::now() + Duration::from_secs(8);
    while child.0.try_wait().unwrap().is_none() {
        assert!(
            Instant::now() < deadline,
            "synthetic foreground did not stop"
        );
        thread::sleep(Duration::from_millis(10));
    }
    println!(
        "foreground lock existed before peer={existed_before_peer}; peer acquired={peer_acquired}"
    );
    assert!(
        existed_before_peer,
        "foreground failed to create the configured ddust lock"
    );
    assert!(
        !peer_acquired,
        "foreground did not hold the configured ddust lease"
    );
    assert!(
        Lease::try_acquire(&lock).unwrap().is_some(),
        "ddust lease was not released"
    );
    assert!(!state.join("audio-conditioned-v1/latest.json").exists());
}

fn run_foreground_with_missing_data(
    root: &std::path::Path,
    registry: Option<&std::path::Path>,
) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_omni-rsi-audio"));
    command
        .env("HOME", root.join("home"))
        .arg("--model")
        .arg(root.join("absent-model"))
        .arg("--data")
        .arg(root.join("absent-data"))
        .arg("--state")
        .arg(root.join("state"));
    if let Some(registry) = registry {
        command.env("DDUST_RUN_REGISTRY", registry);
    } else {
        command.env_remove("DDUST_RUN_REGISTRY");
    }
    command.output().unwrap()
}

#[test]
fn configured_registry_missing_parent_fails_closed_without_creating_it() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("home")).unwrap();
    let registry = root.path().join("absent-ddust/registry.json");
    let output = run_foreground_with_missing_data(root.path(), Some(&registry));
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("ddust"), "{error}");
    assert!(!root.path().join("absent-ddust").exists());
    assert!(
        Lease::try_acquire(&root.path().join("state/device.lock"))
            .unwrap()
            .is_some()
    );
}

#[test]
fn configured_registry_occupied_and_unwritable_fail_closed() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("home")).unwrap();
    let ddust = root.path().join("ddust");
    fs::create_dir(&ddust).unwrap();
    let registry = ddust.join("registry.json");
    let lock = ddust.join("mps.lock");
    let held = Lease::try_acquire(&lock).unwrap().unwrap();
    let output = run_foreground_with_missing_data(root.path(), Some(&registry));
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("ddust accelerator lease is occupied")
    );
    assert!(
        Lease::try_acquire(&root.path().join("state/device.lock"))
            .unwrap()
            .is_some()
    );
    drop(held);
    assert!(Lease::try_acquire(&lock).unwrap().is_some());

    if unsafe { libc::geteuid() } != 0 {
        let readonly = root.path().join("read-only-ddust");
        fs::create_dir(&readonly).unwrap();
        fs::set_permissions(&readonly, fs::Permissions::from_mode(0o500)).unwrap();
        let output =
            run_foreground_with_missing_data(root.path(), Some(&readonly.join("registry.json")));
        fs::set_permissions(&readonly, fs::Permissions::from_mode(0o700)).unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("ddust"));
        assert!(!readonly.join("mps.lock").exists());
    } else {
        eprintln!("permission check skipped: effective user is root");
    }
}

#[test]
fn default_project_absent_is_skipped_but_uninitialized_project_fails_closed() {
    let root = tempfile::tempdir().unwrap();
    let home = root.path().join("home");
    fs::create_dir(&home).unwrap();
    let project = home.join("code/ddust");
    let absent = run_foreground_with_missing_data(root.path(), None);
    assert!(!absent.status.success());
    assert!(!String::from_utf8_lossy(&absent.stderr).contains("ddust"));
    assert!(!project.exists());

    fs::create_dir_all(&project).unwrap();
    let uninitialized = run_foreground_with_missing_data(root.path(), None);
    assert!(!uninitialized.status.success());
    assert!(
        String::from_utf8_lossy(&uninitialized.stderr).contains("default ddust project exists")
    );
    assert!(!project.join(".ddust").exists());

    fs::create_dir(project.join(".ddust")).unwrap();
    let initialized = run_foreground_with_missing_data(root.path(), None);
    assert!(!initialized.status.success());
    assert!(!String::from_utf8_lossy(&initialized.stderr).contains("ddust"));
    let lock = project.join(".ddust/mps.lock");
    assert!(lock.exists());
    assert!(Lease::try_acquire(&lock).unwrap().is_some());
}

#[test]
fn foreground_rejects_fifo_ddust_lock_without_waiting_or_leaking_device() {
    use std::{
        ffi::CString,
        os::unix::{ffi::OsStrExt, fs::FileTypeExt},
    };
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir(root.join("home")).unwrap();
    let ddust = root.join("ddust");
    fs::create_dir(&ddust).unwrap();
    let lock = ddust.join("mps.lock");
    let lock_name = CString::new(lock.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(lock_name.as_ptr(), 0o600) }, 0);
    let stderr = root.join("stderr.log");
    let child = Command::new(env!("CARGO_BIN_EXE_omni-rsi-audio"))
        .env("HOME", root.join("home"))
        .env("DDUST_RUN_REGISTRY", ddust.join("registry.json"))
        .arg("--model")
        .arg(root.join("absent-model"))
        .arg("--data")
        .arg(root.join("absent-data"))
        .arg("--state")
        .arg(root.join("state"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(fs::File::create(&stderr).unwrap()))
        .spawn()
        .unwrap();
    let mut child = Reap(child);
    let deadline = Instant::now() + Duration::from_secs(2);
    let exited = loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            break Some(status);
        }
        if Instant::now() >= deadline {
            break None;
        }
        thread::sleep(Duration::from_millis(10));
    };
    if exited.is_none() {
        child.0.kill().unwrap();
        child.0.wait().unwrap();
    }
    let error = fs::read_to_string(&stderr).unwrap();
    assert!(
        exited.is_some(),
        "opening FIFO ddust lock blocked foreground and held the device"
    );
    assert!(!exited.unwrap().success());
    assert!(error.contains("ddust"), "{error}");
    assert!(fs::symlink_metadata(lock).unwrap().file_type().is_fifo());
    assert!(
        Lease::try_acquire(&root.join("state/device.lock"))
            .unwrap()
            .is_some()
    );
}
