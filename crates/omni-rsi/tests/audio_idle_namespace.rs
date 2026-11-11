#![cfg(all(unix, feature = "native", feature = "cpu"))]

use burn::backend::{Autodiff, NdArray};
use omni_rsi::speech_audio_round::{AudioRoundOptions, train_audio_round, train_v3_audio_round};
use std::{fs, path::Path, process::Command, sync::atomic::AtomicBool};

type Cpu = Autodiff<NdArray<f32>>;

fn setup(root: &Path, v3: bool, alias: bool) -> (std::path::PathBuf, std::path::PathBuf) {
    let state = root.join("state");
    let data = root.join("data");
    fs::create_dir_all(&state).unwrap();
    fs::create_dir_all(&data).unwrap();
    fs::write(data.join("manifest.json"), b"invalid synthetic manifest").unwrap();
    let marker = state.join(if v3 || alias {
        "audio-idle-v3-started"
    } else {
        "audio-idle-legacy-started"
    });
    fs::write(&marker, b"must remain untouched").unwrap();
    let call_state = if alias {
        let nested = state.join("number-audio-v3");
        fs::create_dir(&nested).unwrap();
        nested
    } else {
        state
    };
    (call_state, marker)
}

#[test]
fn review_public_training_fresh_rejects_idle_reservations_before_reading_data() {
    for (v3, alias) in [(false, false), (true, false), (false, true)] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let (state, marker) = setup(root, v3, alias);
        let options = AudioRoundOptions {
            model: root,
            data: &root.join("data"),
            state: &state,
            epochs: 1,
            max_updates: 1,
            resume: false,
        };
        let error = if v3 {
            train_v3_audio_round::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
        } else {
            train_audio_round::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
        }
        .err()
        .expect("reserved state accepted");
        assert!(
            error.contains("already exists") || error.contains("reserved"),
            "v3={v3} alias={alias}: {error}"
        );
        assert_eq!(fs::read(marker).unwrap(), b"must remain untouched");
        let namespace = if v3 {
            state.join("number-audio-v3")
        } else {
            state
        };
        assert!(!namespace.join("audio-conditioned-v1").exists());
    }
}

#[test]
fn review_foreground_fresh_rejects_idle_reservations() {
    for (v3, alias) in [(false, false), (true, false), (false, true)] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let (state, marker) = setup(root, v3, alias);
        let mut command = Command::new(env!("CARGO_BIN_EXE_omni-rsi-audio"));
        command
            .env(
                "DDUST_RUN_REGISTRY",
                root.join("synthetic-registry/runs.json"),
            )
            .arg("--model")
            .arg(root)
            .arg("--data")
            .arg(root.join("data"))
            .arg("--state")
            .arg(&state)
            .args(["--epochs", "1", "--max-updates", "1"]);
        if v3 {
            command.arg("--number-v3");
        }
        let output = command.output().unwrap();
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success());
        assert!(
            error.contains("already exists") || error.contains("reserved"),
            "v3={v3} alias={alias}: {error}"
        );
        assert_eq!(fs::read(marker).unwrap(), b"must remain untouched");
        let namespace = if v3 {
            state.join("number-audio-v3")
        } else {
            state
        };
        assert!(!namespace.join("audio-conditioned-v1").exists());
    }
}

fn check_supervised_resume_scope(test: &str, mode: &str) {
    use omni_rsi::{
        audio_idle_state::AudioSupervision, lease::Lease,
        speech_audio_round::train_supervised_audio_round,
    };
    use std::os::{fd::AsRawFd, unix::net::UnixStream};
    const ROLE: &str = "AUDIO_REVIEW_SUPERVISION_ROLE";
    if std::env::var_os(ROLE).is_none() {
        let temp = tempfile::tempdir().unwrap();
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture"])
            .env(ROLE, "1")
            .env(
                "DDUST_RUN_REGISTRY",
                temp.path().join("synthetic-registry/runs.json"),
            )
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let a = temp.path().join("a");
    let b = temp.path().join("b");
    fs::create_dir_all(a.join("audio-conditioned-v1")).unwrap();
    fs::create_dir_all(b.join("audio-conditioned-v1")).unwrap();
    fs::create_dir(temp.path().join("data")).unwrap();
    fs::write(
        temp.path().join("data/manifest.json"),
        b"invalid synthetic manifest",
    )
    .unwrap();
    let runner = Lease::try_acquire(&a.join("runner.lock")).unwrap().unwrap();
    let device = Lease::try_acquire(&a.join("device.lock")).unwrap().unwrap();
    let (parent, child) = UnixStream::pair().unwrap();
    let supervision = AudioSupervision::validate(
        &a,
        runner.as_raw_fd(),
        device.as_raw_fd(),
        child.as_raw_fd(),
    )
    .unwrap();
    let _keepalive = if mode == "dead" {
        drop(parent);
        None
    } else {
        Some(parent)
    };
    let _device = if mode == "no-device" {
        drop(device);
        None
    } else {
        Some(device)
    };
    let state = if mode == "other" { &b } else { &a };
    let data = temp.path().join("data");
    let options = AudioRoundOptions {
        model: temp.path(),
        data: &data,
        state,
        epochs: 1,
        max_updates: 1,
        resume: true,
    };
    let error = train_supervised_audio_round::<Cpu>(
        &options,
        &Default::default(),
        &AtomicBool::new(false),
        false,
        &supervision,
    )
    .err()
    .expect("invalid supervision accepted");
    match mode {
        "other" | "dead" => assert!(error.contains("supervisor or state changed"), "{error}"),
        "no-device" => assert!(error.contains("inherited lease"), "{error}"),
        "valid" => assert!(error.contains("invalid audio dataset manifest"), "{error}"),
        _ => unreachable!(),
    }
    assert!(!state.join("audio-conditioned-v1/latest.json").exists());
}

#[test]
fn review_supervised_resume_rejects_different_state() {
    check_supervised_resume_scope("review_supervised_resume_rejects_different_state", "other");
}

#[test]
fn review_supervised_resume_rejects_closed_socket() {
    check_supervised_resume_scope("review_supervised_resume_rejects_closed_socket", "dead");
}

#[test]
fn review_supervised_resume_rejects_lost_device_lease() {
    check_supervised_resume_scope(
        "review_supervised_resume_rejects_lost_device_lease",
        "no-device",
    );
}

#[test]
fn review_supervised_resume_allows_same_live_scope_to_reach_data_validation() {
    check_supervised_resume_scope(
        "review_supervised_resume_allows_same_live_scope_to_reach_data_validation",
        "valid",
    );
}

#[test]
fn review_v3_reservation_blocks_alternate_symlink_alias() {
    use std::os::unix::fs::symlink;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    omni_rsi::audio_idle_state::reserve_fresh(root, true).unwrap();
    let alias = root.join("different-name");
    symlink(root.join("number-audio-v3"), &alias).unwrap();
    let error = omni_rsi::audio_idle_state::check_fresh(&alias, false).unwrap_err();
    assert!(error.contains("reserved"), "{error}");
    assert!(
        omni_rsi::audio_idle_state::check_fresh(root, false).is_ok(),
        "independent legacy namespace must remain usable"
    );
}

#[test]
fn review_actual_supervised_worker_enters_its_reserved_fresh_namespace() {
    use omni_rsi::lease::Lease;
    use std::os::{
        fd::AsRawFd,
        unix::{net::UnixStream, process::CommandExt},
    };
    for v3 in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let state = root.join("state");
        let data = root.join("data");
        fs::create_dir_all(&state).unwrap();
        fs::create_dir(&data).unwrap();
        fs::write(data.join("manifest.json"), b"invalid synthetic manifest").unwrap();
        omni_rsi::audio_idle_state::reserve_fresh(&state, v3).unwrap();
        let runner = Lease::try_acquire(&state.join("runner.lock"))
            .unwrap()
            .unwrap();
        let device = Lease::try_acquire(&state.join("device.lock"))
            .unwrap()
            .unwrap();
        let (_parent, worker) = UnixStream::pair().unwrap();
        let fds = [runner.as_raw_fd(), device.as_raw_fd(), worker.as_raw_fd()];
        let mut command = Command::new(env!("CARGO_BIN_EXE_omni-rsi"));
        command
            .env(
                "DDUST_RUN_REGISTRY",
                root.join("synthetic-registry/runs.json"),
            )
            .arg("--model")
            .arg(root)
            .arg("--state")
            .arg(&state)
            .args(["audio-work", "--data"])
            .arg(&data)
            .args(["--fresh", "--epochs", "1", "--updates-per-round", "1"])
            .arg("--runner-fd")
            .arg(fds[0].to_string())
            .arg("--device-fd")
            .arg(fds[1].to_string())
            .arg("--supervisor-fd")
            .arg(fds[2].to_string());
        if v3 {
            command.arg("--number-v3");
        }
        unsafe {
            command.pre_exec(move || {
                for fd in fds {
                    let flags = libc::fcntl(fd, libc::F_GETFD);
                    if flags < 0 || libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
        let output = command.output().unwrap();
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success());
        assert!(
            !error.contains("supervised audio worker rejected") && !error.contains("reservation"),
            "v3={v3}: {error}"
        );
        assert!(
            state.join("audio-training.lock").exists(),
            "worker never entered training: {error}"
        );
        let namespace = if v3 {
            state.join("number-audio-v3")
        } else {
            state
        };
        assert!(!namespace.join("audio-conditioned-v1").exists());
    }
}
