use super::*;
use crate::{AudioStart, WatchConfig};
use burn::{
    backend::{Autodiff, NdArray},
    tensor::backend::Backend,
};
use omni_rsi::{
    idle::{IdlePolicy, IdleSnapshot},
    lease::Lease,
    speech_audio_checkpoint::{
        AudioCheckpoint, AudioProgress, AudioProtocol, audio_implementation_hash,
    },
    speech_training::SpeechOptimizer,
};
use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

type Cpu = Autodiff<NdArray<f32>>;

#[allow(dead_code)]
#[path = "../tests/support/number_v3_fixture.rs"]
mod v3_fixture;

fn idle(ready: bool) -> IdleSnapshot {
    IdleSnapshot {
        idle_seconds: Some(if ready { 301.0 } else { 1.0 }),
        ac_power: Some(true),
        low_power: Some(false),
        thermal_nominal: Some(true),
        memory_normal: Some(true),
        free_bytes: Some(20 * 1024 * 1024 * 1024),
        competing_process: Some(false),
    }
}

fn job(root: &Path) -> AudioJob {
    AudioJob {
        data: root.to_path_buf(),
        number_v3: false,
        epochs: 2,
        updates_per_round: 1,
        start: AudioStart {
            fresh: true,
            resume: false,
        },
    }
}

fn write_checkpoint(stage: &Path, progress: AudioProgress) -> serde_json::Value {
    let device = Default::default();
    let optimizer = SpeechOptimizer::<Cpu>::new(16, 4, 761, &device).unwrap();
    let protocol = AudioProtocol {
        version: 1,
        base_hash: "1".repeat(64),
        codec_hash: "2".repeat(64),
        dataset_hash: "3".repeat(64),
        implementation_hash: audio_implementation_hash().unwrap(),
        backend: Cpu::name(&device),
        width: 16,
        train_records: 2,
        mask_seed: 159,
    };
    let checkpoint = AudioCheckpoint::new(protocol.clone(), progress.clone(), &optimizer).unwrap();
    let id = checkpoint.save(stage).unwrap();
    serde_json::json!({"status":"partial", "progress":progress, "protocol":protocol, "checkpoint":id})
}

fn program(root: &Path, text: &str) -> std::path::PathBuf {
    let path = root.join("mock-audio-worker.sh");
    fs::write(&path, text).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    path
}

fn wait_file(path: &Path) {
    let deadline = Instant::now() + Duration::from_secs(8);
    while !path.exists() {
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {}",
            path.display()
        );
        thread::sleep(Duration::from_millis(10));
    }
}

fn wait_for_device(root: &Path) {
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        if let Some(lease) = Lease::try_acquire(&root.join("device.lock")).unwrap() {
            drop(lease);
            break;
        }
        assert!(
            Instant::now() < deadline,
            "worker group retained the device"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn audio_route_launches_worker_yields_and_resumes_only_the_complete_checkpoint() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_owned();
    write_checkpoint(
        &root.join("stage-one"),
        AudioProgress {
            epoch: 0,
            next_index: 1,
            updates: 1,
        },
    );
    let output = write_checkpoint(
        &root.join("stage-two"),
        AudioProgress {
            epoch: 1,
            next_index: 0,
            updates: 2,
        },
    );
    fs::write(
        root.join("mock-output.json"),
        serde_json::to_vec(&output).unwrap(),
    )
    .unwrap();
    let executable = program(
        &root,
        r#"#!/bin/sh
previous=
for arg in "$@"; do
  [ "$previous" = "--state" ] && state="$arg"
  [ "$previous" = "--runner-fd" ] && runner="$arg"
  [ "$previous" = "--device-fd" ] && device="$arg"
  [ "$previous" = "--supervisor-fd" ] && supervisor="$arg"
  previous="$arg"
done
[ -e "/dev/fd/$runner" ] || exit 91
[ -e "/dev/fd/$device" ] || exit 92
[ -e "/dev/fd/$supervisor" ] || exit 96
if [ ! -f "$state/first-attempt" ]; then
  printf '%s\n' "$@" > "$state/args-first"
  cp -R "$state/stage-one/audio-conditioned-v1" "$state/audio-conditioned-v1" || exit 93
  touch "$state/first-attempt"
  printf '%s' "$$" > "$state/worker-pid"
  sleep 30 &
  child=$!
  trap 'wait "$child" 2>/dev/null; exit 130' INT TERM
  wait "$child"
  exit 94
fi
printf '%s\n' "$@" > "$state/args-second"
cp -R "$state/stage-two/audio-conditioned-v1/." "$state/audio-conditioned-v1/" || exit 95
cat "$state/mock-output.json"
"#,
    );
    let ready = Arc::new(AtomicBool::new(false));
    let probe = ready.clone();
    let shutdown = Arc::new(AtomicBool::new(false));
    let thread_root = root.clone();
    let runner = thread::spawn(move || {
        let mut config = WatchConfig::new(
            &thread_root,
            1,
            IdlePolicy::default(),
            shutdown,
            "audio mock",
        );
        config.probe = Arc::new(move |_| idle(probe.load(Ordering::Acquire)));
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(5);
        config.worker_grace = Duration::from_millis(500);
        let job = job(&thread_root);
        watch_audio(config, &thread_root, &job, &executable)
    });
    thread::sleep(Duration::from_millis(50));
    assert!(!root.join("first-attempt").exists());
    ready.store(true, Ordering::Release);
    wait_file(&root.join("worker-pid"));
    let pid: i32 = fs::read_to_string(root.join("worker-pid"))
        .unwrap()
        .parse()
        .unwrap();
    let foreground = Lease::try_acquire(&root.join("foreground.lock"))
        .unwrap()
        .unwrap();
    wait_for_device(&root);
    assert_eq!(
        unsafe { libc::kill(pid, 0) },
        -1,
        "worker process survived device release"
    );
    assert!(!root.join("args-second").exists());
    assert!(
        fs::read_to_string(root.join("args-first"))
            .unwrap()
            .contains("--fresh\n")
    );
    drop(foreground);
    runner.join().unwrap().unwrap();
    let args = fs::read_to_string(root.join("args-second")).unwrap();
    assert!(args.contains("--resume\n"), "{args}");
    assert!(!args.contains("--fresh\n"), "{args}");
    let state: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("status.json")).unwrap()).unwrap();
    assert_eq!(state["phase"], "partial");
    assert_eq!(state["completed_this_session"], 1);
    assert!(root.join("audio-idle-legacy-started").is_file());
}

#[test]
fn inherited_leases_and_supervisor_survive_exec_and_detect_loss() {
    use std::{
        os::{fd::AsRawFd, unix::net::UnixStream},
        process::Command,
    };
    const CHILD: &str = "OMNI_AUDIO_IDLE_TEST_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let root = std::path::PathBuf::from(std::env::var_os("OMNI_AUDIO_IDLE_TEST_ROOT").unwrap());
        let fd = |name| std::env::var(name).unwrap().parse::<i32>().unwrap();
        let proof = AudioProof {
            runner_fd: fd("OMNI_AUDIO_IDLE_TEST_RUNNER"),
            device_fd: fd("OMNI_AUDIO_IDLE_TEST_DEVICE"),
            supervisor_fd: fd("OMNI_AUDIO_IDLE_TEST_SOCKET"),
            ddust_fd: Some(fd("OMNI_AUDIO_IDLE_TEST_DDUST")),
        };
        let stream = validate_supervision(&root, proof, Some(&root.join("mps.lock"))).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let monitor = monitor_supervisor(stop.clone(), stream);
        fs::write(root.join("validated"), b"ok").unwrap();
        let deadline = Instant::now() + Duration::from_secs(8);
        while !stop.load(Ordering::Acquire) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            stop.load(Ordering::Acquire),
            "lost supervisor was not detected"
        );
        drop(monitor);
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let runner = Lease::try_acquire(&root.join("runner.lock"))
        .unwrap()
        .unwrap();
    let device = Lease::try_acquire(&root.join("device.lock"))
        .unwrap()
        .unwrap();
    let ddust = Lease::try_acquire(&root.join("mps.lock")).unwrap().unwrap();
    let (parent, worker) = UnixStream::pair().unwrap();
    let proof = AudioProof {
        runner_fd: runner.as_raw_fd(),
        device_fd: device.as_raw_fd(),
        supervisor_fd: worker.as_raw_fd(),
        ddust_fd: Some(ddust.as_raw_fd()),
    };
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "audio_idle::tests::inherited_leases_and_supervisor_survive_exec_and_detect_loss",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env("OMNI_AUDIO_IDLE_TEST_ROOT", root)
        .env("OMNI_AUDIO_IDLE_TEST_RUNNER", proof.runner_fd.to_string())
        .env("OMNI_AUDIO_IDLE_TEST_DEVICE", proof.device_fd.to_string())
        .env(
            "OMNI_AUDIO_IDLE_TEST_DDUST",
            proof.ddust_fd.unwrap().to_string(),
        )
        .env(
            "OMNI_AUDIO_IDLE_TEST_SOCKET",
            proof.supervisor_fd.to_string(),
        );
    configure_inherited(&mut command, proof);
    let child = thread::spawn(move || {
        let _worker = worker;
        omni_rsi::child::run(command, &AtomicBool::new(false), Duration::from_secs(2))
    });
    wait_file(&root.join("validated"));
    assert!(
        Lease::try_acquire(&root.join("device.lock"))
            .unwrap()
            .is_none()
    );
    assert!(
        Lease::try_acquire(&root.join("mps.lock"))
            .unwrap()
            .is_none()
    );
    drop(parent);
    assert!(child.join().unwrap().unwrap().success());
    assert!(
        Lease::try_acquire(&root.join("device.lock"))
            .unwrap()
            .is_none()
    );
    drop(device);
    drop(ddust);
    assert!(
        Lease::try_acquire(&root.join("device.lock"))
            .unwrap()
            .is_some()
    );
}

#[test]
fn cancellation_before_a_complete_checkpoint_stops_without_restarting_or_clearing_state() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_owned();
    let executable = program(
        &root,
        r#"#!/bin/sh
previous=
for arg in "$@"; do
  [ "$previous" = "--state" ] && state="$arg"
  previous="$arg"
done
touch "$state/worker-started"
sleep 30 &
child=$!
trap 'wait "$child" 2>/dev/null; exit 130' INT TERM
wait "$child"
"#,
    );
    let ready = Arc::new(AtomicBool::new(true));
    let probe = ready.clone();
    let thread_root = root.clone();
    let runner = thread::spawn(move || {
        let mut config = WatchConfig::new(
            &thread_root,
            2,
            IdlePolicy::default(),
            Arc::new(AtomicBool::new(false)),
            "audio mock",
        );
        config.probe = Arc::new(move |_| idle(probe.load(Ordering::Acquire)));
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(5);
        config.worker_grace = Duration::from_millis(500);
        watch_audio(config, &thread_root, &job(&thread_root), &executable)
    });
    wait_file(&root.join("worker-started"));
    ready.store(false, Ordering::Release);
    let error = runner.join().unwrap().unwrap_err();
    assert!(error.contains("no complete audio checkpoint"), "{error}");
    assert!(root.join("audio-idle-legacy-started").is_file());
    assert!(!root.join("audio-conditioned-v1/latest.json").exists());
    assert!(preflight(&root, &job(&root)).is_err());
    let state: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("status.json")).unwrap()).unwrap();
    assert_eq!(state["phase"], "error");
}

#[test]
fn output_overflow_stops_the_worker_before_exit_and_releases_the_device() {
    use std::sync::mpsc;
    for stderr in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_owned();
        let write = if stderr {
            "dd if=/dev/zero bs=4096 count=32 1>&2 2>/dev/null"
        } else {
            "dd if=/dev/zero bs=4096 count=300 2>/dev/null"
        };
        let executable = program(
            &root,
            &format!(
                r#"#!/bin/sh
previous=
for arg in "$@"; do
  [ "$previous" = "--state" ] && state="$arg"
  previous="$arg"
done
printf '%s' "$$" > "$state/overflow-pid"
{write}
sleep 30 &
child=$!
trap 'wait "$child" 2>/dev/null; exit 130' INT TERM
wait "$child"
"#
            ),
        );
        let shutdown = Arc::new(AtomicBool::new(false));
        let thread_shutdown = shutdown.clone();
        let thread_root = root.clone();
        let (sender, receiver) = mpsc::channel();
        let runner = thread::spawn(move || {
            let mut config = WatchConfig::new(
                &thread_root,
                1,
                IdlePolicy::default(),
                thread_shutdown,
                "audio mock",
            );
            config.probe = Arc::new(|_| idle(true));
            config.ddust_lease = || Ok(None);
            config.wait_unit = Duration::from_millis(5);
            config.watchdog_interval = Duration::from_millis(5);
            config.worker_grace = Duration::from_millis(500);
            sender
                .send(watch_audio(
                    config,
                    &thread_root,
                    &job(&thread_root),
                    &executable,
                ))
                .unwrap();
        });
        wait_file(&root.join("overflow-pid"));
        let pid: i32 = fs::read_to_string(root.join("overflow-pid"))
            .unwrap()
            .parse()
            .unwrap();
        let result = match receiver.recv_timeout(Duration::from_secs(4)) {
            Ok(result) => result,
            Err(error) => {
                shutdown.store(true, Ordering::Release);
                let _ = receiver.recv_timeout(Duration::from_secs(12));
                runner.join().unwrap();
                panic!("overflow did not stop a live worker: {error}");
            }
        };
        runner.join().unwrap();
        let error = result.unwrap_err();
        assert!(error.contains("output limit"), "{error}");
        wait_for_device(&root);
        assert_eq!(
            unsafe { libc::kill(pid, 0) },
            -1,
            "worker survived output rejection"
        );
    }
}

#[test]
fn bounded_output_accepts_the_limit_and_rejects_one_extra_byte() {
    use std::{io::Write, os::unix::net::UnixStream};
    for (length, should_exceed) in [(16, false), (17, true)] {
        let (reader, mut writer) = UnixStream::pair().unwrap();
        reader.set_nonblocking(true).unwrap();
        writer.write_all(&vec![7_u8; length]).unwrap();
        drop(writer);
        let stop = AtomicBool::new(false);
        let finished = AtomicBool::new(false);
        let exceeded = AtomicBool::new(false);
        let bytes = collect_worker_output(reader, 16, &stop, &finished, &exceeded).unwrap();
        assert_eq!(exceeded.load(Ordering::Acquire), should_exceed);
        assert_eq!(stop.load(Ordering::Acquire), should_exceed);
        assert!(bytes.len() <= 16);
        if !should_exceed {
            assert_eq!(bytes, vec![7_u8; 16]);
        }
    }
}

#[test]
fn v3_audio_route_counts_partial_then_complete_without_touching_legacy_state() {
    use omni_rsi::number_audio_v3::NumberAudioV3;
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_owned();
    let legacy = root.join("audio-conditioned-v1");
    fs::create_dir(&legacy).unwrap();
    fs::write(legacy.join("keep"), b"legacy").unwrap();
    let data_root = v3_fixture::fixture();
    let model = data_root.path().join("model");
    let stop = AtomicBool::new(false);
    let data = NumberAudioV3::load(data_root.path(), &model, &stop).unwrap();
    let device = Default::default();
    let protocol = AudioProtocol::for_v3(&model, &data, Cpu::name(&device), &stop).unwrap();
    let optimizer = SpeechOptimizer::<Cpu>::new(16, 4, 761, &device).unwrap();
    for (name, progress, status) in [
        (
            "one",
            AudioProgress {
                epoch: 0,
                next_index: 1,
                updates: 1,
            },
            "partial",
        ),
        (
            "two",
            AudioProgress {
                epoch: 1,
                next_index: 0,
                updates: 2,
            },
            "complete",
        ),
    ] {
        let checkpoint =
            AudioCheckpoint::new(protocol.clone(), progress.clone(), &optimizer).unwrap();
        let id = checkpoint
            .save(&root.join(format!("stage-{name}/number-audio-v3")))
            .unwrap();
        let report = serde_json::json!({"status":status,"progress":progress,"protocol":protocol,"checkpoint":id});
        fs::write(
            root.join(format!("report-{name}.json")),
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
    }
    let executable = program(
        &root,
        r#"#!/bin/sh
previous=
for arg in "$@"; do
  [ "$previous" = "--state" ] && state="$arg"
  previous="$arg"
done
mkdir -p "$state/number-audio-v3"
if [ ! -f "$state/first-done" ]; then
  printf '%s\n' "$@" > "$state/v3-args-first"
  cp -R "$state/stage-one/number-audio-v3/audio-conditioned-v1" "$state/number-audio-v3/" || exit 91
  touch "$state/first-done"
  cat "$state/report-one.json"
else
  printf '%s\n' "$@" > "$state/v3-args-second"
  cp -R "$state/stage-two/number-audio-v3/audio-conditioned-v1/." "$state/number-audio-v3/audio-conditioned-v1/" || exit 92
  cat "$state/report-two.json"
fi
"#,
    );
    let mut config = WatchConfig::new(
        &root,
        2,
        IdlePolicy::default(),
        Arc::new(AtomicBool::new(false)),
        "audio v3 mock",
    );
    config.probe = Arc::new(|_| idle(true));
    config.ddust_lease = || Ok(None);
    config.wait_unit = Duration::from_millis(5);
    config.watchdog_interval = Duration::from_millis(5);
    config.worker_grace = Duration::from_millis(500);
    let mut job = job(&root);
    job.number_v3 = true;
    job.epochs = 1;
    watch_audio(config, &root, &job, &executable).unwrap();
    let first = fs::read_to_string(root.join("v3-args-first")).unwrap();
    let second = fs::read_to_string(root.join("v3-args-second")).unwrap();
    assert!(
        first.contains("--number-v3\n") && first.contains("--fresh\n"),
        "{first}"
    );
    assert!(
        second.contains("--number-v3\n") && second.contains("--resume\n"),
        "{second}"
    );
    assert_eq!(fs::read(legacy.join("keep")).unwrap(), b"legacy");
    assert!(root.join("audio-idle-v3-started").is_file());
    assert!(!root.join("audio-idle-legacy-started").exists());
    let status: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("status.json")).unwrap()).unwrap();
    assert_eq!(status["phase"], "complete");
    assert_eq!(status["completed_this_session"], 2);
}

#[test]
fn audio_route_stops_during_non_idle_wait_without_starting_a_worker() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_owned();
    let executable = program(&root, "#!/bin/sh\nexit 99\n");
    let shutdown = Arc::new(AtomicBool::new(false));
    let thread_shutdown = shutdown.clone();
    let thread_root = root.clone();
    let runner = thread::spawn(move || {
        let mut config = WatchConfig::new(
            &thread_root,
            1,
            IdlePolicy::default(),
            thread_shutdown,
            "audio mock",
        );
        config.probe = Arc::new(|_| idle(false));
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(5);
        watch_audio(config, &thread_root, &job(&thread_root), &executable)
    });
    wait_file(&root.join("status.json"));
    shutdown.store(true, Ordering::Release);
    runner.join().unwrap().unwrap();
    let status: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("status.json")).unwrap()).unwrap();
    assert_eq!(status["phase"], "stopped");
    assert_eq!(status["completed_this_session"], 0);
    assert!(!root.join("audio-idle-legacy-started").exists());
    assert!(!root.join("audio-conditioned-v1").exists());
}

#[test]
fn audio_stop_during_worker_keeps_the_last_complete_checkpoint() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_owned();
    let report = write_checkpoint(
        &root.join("stage-one"),
        AudioProgress {
            epoch: 0,
            next_index: 1,
            updates: 1,
        },
    );
    let expected = report["checkpoint"].as_str().unwrap().to_owned();
    let executable = program(
        &root,
        r#"#!/bin/sh
previous=
for arg in "$@"; do
  [ "$previous" = "--state" ] && state="$arg"
  previous="$arg"
done
cp -R "$state/stage-one/audio-conditioned-v1" "$state/audio-conditioned-v1" || exit 91
printf '%s' "$$" > "$state/stop-worker-pid"
sleep 30 &
child=$!
trap 'wait "$child" 2>/dev/null; exit 130' INT TERM
wait "$child"
"#,
    );
    let shutdown = Arc::new(AtomicBool::new(false));
    let thread_shutdown = shutdown.clone();
    let thread_root = root.clone();
    let runner = thread::spawn(move || {
        let mut config = WatchConfig::new(
            &thread_root,
            2,
            IdlePolicy::default(),
            thread_shutdown,
            "audio mock",
        );
        config.probe = Arc::new(|_| idle(true));
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(5);
        config.worker_grace = Duration::from_millis(500);
        watch_audio(config, &thread_root, &job(&thread_root), &executable)
    });
    wait_file(&root.join("stop-worker-pid"));
    let pid: i32 = fs::read_to_string(root.join("stop-worker-pid"))
        .unwrap()
        .parse()
        .unwrap();
    shutdown.store(true, Ordering::Release);
    runner.join().unwrap().unwrap();
    assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
    assert_eq!(
        AudioCheckpoint::inspect(&root)
            .unwrap()
            .unwrap()
            .checkpoint_sha256,
        expected
    );
    wait_for_device(&root);
    let status: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("status.json")).unwrap()).unwrap();
    assert_eq!(status["phase"], "stopped");
    assert_eq!(status["completed_this_session"], 0);
}
