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
        };
        let supervision = validate_supervision(&root, proof).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let monitor = supervision.monitor(stop.clone()).unwrap();
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
    let (parent, worker) = UnixStream::pair().unwrap();
    let proof = AudioProof {
        runner_fd: runner.as_raw_fd(),
        device_fd: device.as_raw_fd(),
        supervisor_fd: worker.as_raw_fd(),
    };
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "audio_idle::tests::inherited_leases_and_supervisor_survive_exec_and_detect_loss",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env(
            "DDUST_RUN_REGISTRY",
            root.join("synthetic-registry/runs.json"),
        )
        .env("OMNI_AUDIO_IDLE_TEST_ROOT", root)
        .env("OMNI_AUDIO_IDLE_TEST_RUNNER", proof.runner_fd.to_string())
        .env("OMNI_AUDIO_IDLE_TEST_DEVICE", proof.device_fd.to_string())
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
    drop(parent);
    assert!(child.join().unwrap().unwrap().success());
    assert!(
        Lease::try_acquire(&root.join("device.lock"))
            .unwrap()
            .is_none()
    );
    drop(device);
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
    let reports = root.join("idle-round-reports");
    fs::create_dir(&reports).unwrap();
    fs::write(
        reports.join("round-existing.json"),
        b"preserve existing report",
    )
    .unwrap();
    fs::write(reports.join(".pending-interrupted.tmp"), b"incomplete").unwrap();
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
    assert_eq!(
        fs::read(reports.join("round-existing.json")).unwrap(),
        b"preserve existing report"
    );
    assert_eq!(
        fs::read(reports.join(".pending-interrupted.tmp")).unwrap(),
        b"incomplete"
    );
    let generated: Vec<_> = fs::read_dir(&reports)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .filter(|path| path.file_name().unwrap() != "round-existing.json")
        .collect();
    assert_eq!(generated.len(), 2);
    for path in generated {
        let name = path.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with("round-") && name.ends_with(".json"));
        let report: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(report["quality"], "not_assessed");
    }
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

#[test]
fn review_v3_started_marker_blocks_legacy_alias_before_worker_launch() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let alias = root.join("number-audio-v3");
    fs::create_dir(&alias).unwrap();
    let marker = root.join("audio-idle-v3-started");
    fs::write(&marker, b"reserved-v3").unwrap();
    let executable = program(
        &alias,
        &format!(
            "#!/bin/sh\ntouch '{}'\nexit 1\n",
            alias.join("worker-started").display()
        ),
    );
    let mut config = WatchConfig::new(
        &alias,
        1,
        IdlePolicy::default(),
        Arc::new(AtomicBool::new(false)),
        "alias mock",
    );
    config.probe = Arc::new(|_| idle(true));
    config.ddust_lease = || Ok(None);
    config.wait_unit = Duration::from_millis(5);
    config.watchdog_interval = Duration::from_millis(5);
    config.worker_grace = Duration::from_millis(500);
    let error = watch_audio(config, root, &job(root), &executable).unwrap_err();
    assert!(
        error.contains("already exists") || error.contains("reserved"),
        "{error}"
    );
    assert!(!alias.join("worker-started").exists());
    assert!(!alias.join("audio-idle-legacy-started").exists());
    assert_eq!(fs::read(&marker).unwrap(), b"reserved-v3");
}

#[test]
fn review_full_scheduler_stderr_never_blocks_foreground_yield() {
    use std::{
        io::{ErrorKind, Read, Write},
        os::{fd::OwnedFd, unix::net::UnixStream},
        process::{Child, Command, Stdio},
    };
    const ROLE: &str = "AUDIO_REVIEW_STDERR_ROLE";
    if std::env::var(ROLE).as_deref() == Ok("scheduler") {
        let root = std::path::PathBuf::from(std::env::var_os("AUDIO_REVIEW_ROOT").unwrap());
        let executable = program(
            &root,
            r#"#!/bin/sh
previous=
for arg in "$@"; do
  [ "$previous" = "--state" ] && state="$arg"
  previous="$arg"
done
printf '%s' "$$" > "$state/stderr-worker-pid"
sleep 30 &
child=$!
trap 'wait "$child" 2>/dev/null; exit 130' INT TERM
wait "$child"
"#,
        );
        let mut config = WatchConfig::new(
            &root,
            1,
            IdlePolicy::default(),
            Arc::new(AtomicBool::new(false)),
            "synthetic blocked stderr",
        );
        config.probe = Arc::new(|_| idle(true));
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(5);
        config.worker_grace = Duration::from_millis(500);
        let result = watch_audio(config, &root, &job(&root), &executable);
        if let Err(error) = result {
            assert!(error.contains("no complete audio checkpoint"), "{error}");
        }
        return;
    }
    struct Reap(Child);
    impl Drop for Reap {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    fs::create_dir(root.join("ddust")).unwrap();
    let (mut reader, mut writer) = UnixStream::pair().unwrap();
    writer.set_nonblocking(true).unwrap();
    let mut filled = 0;
    loop {
        match writer.write(&[b'x'; 8192]) {
            Ok(0) => panic!("unexpected zero while filling stderr"),
            Ok(n) => filled += n,
            Err(error) if error.kind() == ErrorKind::WouldBlock => break,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => panic!("cannot fill synthetic stderr: {error}"),
        }
    }
    writer.set_nonblocking(false).unwrap();
    let child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "audio_idle::tests::review_full_scheduler_stderr_never_blocks_foreground_yield",
            "--nocapture",
        ])
        .env(ROLE, "scheduler")
        .env("AUDIO_REVIEW_ROOT", root)
        .env("DDUST_RUN_REGISTRY", root.join("ddust/registry.json"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(OwnedFd::from(writer)))
        .spawn()
        .unwrap();
    let mut child = Reap(child);
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        if let Ok(bytes) = fs::read(root.join("status.json"))
            && serde_json::from_slice::<serde_json::Value>(&bytes)
                .is_ok_and(|status| status["phase"] == "running")
        {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "scheduler did not reach running phase"
        );
        thread::sleep(Duration::from_millis(10));
    }
    let deadline = Instant::now() + Duration::from_secs(8);
    let worker_pid = loop {
        if let Ok(text) = fs::read_to_string(root.join("stderr-worker-pid"))
            && let Ok(pid) = text.parse::<i32>()
        {
            break pid;
        }
        assert!(Instant::now() < deadline, "mock worker did not start");
        thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(
        unsafe { libc::kill(worker_pid, 0) },
        0,
        "mock worker did not start"
    );
    let foreground = Lease::try_acquire(&root.join("foreground.lock"))
        .unwrap()
        .unwrap();
    fs::write(root.join("stop.request"), b"stop").unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    let released = loop {
        if let Some(lease) = Lease::try_acquire(&root.join("device.lock")).unwrap() {
            drop(lease);
            break true;
        }
        if Instant::now() >= deadline {
            break false;
        }
        thread::sleep(Duration::from_millis(10));
    };
    if released {
        assert_eq!(
            unsafe { libc::kill(worker_pid, 0) },
            -1,
            "worker survived device release"
        );
    }
    reader.set_nonblocking(true).unwrap();
    let mut buffer = [0_u8; 8192];
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        match reader.read(&mut buffer) {
            Ok(_) => {}
            Err(error) if error.kind() == ErrorKind::WouldBlock => {}
            Err(error) => panic!("cannot drain synthetic stderr: {error}"),
        }
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(status.success(), "scheduler exited {status}");
            break;
        }
        assert!(
            Instant::now() < deadline,
            "scheduler did not stop after stderr was drained"
        );
        thread::sleep(Duration::from_millis(5));
    }
    drop(foreground);
    println!("stderr filled={filled}, device released before drain={released}");
    assert!(
        released,
        "full scheduler stderr blocked foreground device release"
    );
}

#[test]
fn review_watchdog_probe_panic_with_blocked_stderr_still_yields() {
    use std::{
        io::{ErrorKind, Read, Write},
        os::{fd::OwnedFd, unix::net::UnixStream},
        process::{Child, Command, Stdio},
    };
    const ROLE: &str = "AUDIO_REVIEW_PANIC_ROLE";
    if std::env::var_os(ROLE).is_some() {
        // Reproduce the CLI's startup ordering before starting scheduler threads.
        crate::install_watchdog_panic_hook();
        let root = std::path::PathBuf::from(std::env::var_os("AUDIO_REVIEW_ROOT").unwrap());
        let executable = program(
            &root,
            r#"#!/bin/sh
previous=
for arg in "$@"; do
  [ "$previous" = "--state" ] && state="$arg"
  previous="$arg"
done
printf '%s' "$$" > "$state/panic-worker-pid"
sleep 30 &
child=$!
trap 'wait "$child" 2>/dev/null; exit 130' INT TERM
wait "$child"
"#,
        );
        let mut config = WatchConfig::new(
            &root,
            1,
            IdlePolicy::default(),
            Arc::new(AtomicBool::new(false)),
            "panic probe mock",
        );
        config.probe = Arc::new(|root| {
            assert!(
                !root.join("panic-probe").exists(),
                "synthetic idle probe failure"
            );
            idle(true)
        });
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(5);
        config.worker_grace = Duration::from_millis(500);
        let error = watch_audio(config, &root, &job(&root), &executable).unwrap_err();
        assert!(error.contains("no complete audio checkpoint"), "{error}");
        assert!(error.contains("idle probe failed"), "{error}");
        return;
    }
    struct Reap(Child);
    impl Drop for Reap {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let (mut reader, mut writer) = UnixStream::pair().unwrap();
    writer.set_nonblocking(true).unwrap();
    loop {
        match writer.write(&[b'x'; 8192]) {
            Ok(0) => panic!("unexpected zero while filling stderr"),
            Ok(_) => {}
            Err(error) if error.kind() == ErrorKind::WouldBlock => break,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => panic!("cannot fill stderr: {error}"),
        }
    }
    writer.set_nonblocking(false).unwrap();
    let mut child = Reap(
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "audio_idle::tests::review_watchdog_probe_panic_with_blocked_stderr_still_yields",
                "--nocapture",
            ])
            .env(ROLE, "1")
            .env("AUDIO_REVIEW_ROOT", root)
            .env(
                "DDUST_RUN_REGISTRY",
                root.join("synthetic-registry/runs.json"),
            )
            .stdout(Stdio::null())
            .stderr(Stdio::from(OwnedFd::from(writer)))
            .spawn()
            .unwrap(),
    );
    wait_file(&root.join("panic-worker-pid"));
    fs::write(root.join("panic-probe"), b"panic").unwrap();
    thread::sleep(Duration::from_millis(200));
    let foreground = Lease::try_acquire(&root.join("foreground.lock"))
        .unwrap()
        .unwrap();
    fs::write(root.join("stop.request"), b"stop").unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    let released = loop {
        if Lease::try_acquire(&root.join("device.lock"))
            .unwrap()
            .is_some()
        {
            break true;
        }
        if Instant::now() >= deadline {
            break false;
        }
        thread::sleep(Duration::from_millis(10));
    };
    reader.set_nonblocking(true).unwrap();
    let mut buffer = [0_u8; 8192];
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        match reader.read(&mut buffer) {
            Ok(_) => {}
            Err(error) if error.kind() == ErrorKind::WouldBlock => {}
            Err(error) => panic!("cannot drain stderr: {error}"),
        }
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(status.success(), "{status}");
            break;
        }
        assert!(
            Instant::now() < deadline,
            "scheduler did not exit after stderr drained"
        );
        thread::sleep(Duration::from_millis(5));
    }
    drop(foreground);
    assert!(
        released,
        "watchdog panic hook blocked device release on full stderr"
    );
}

#[test]
fn review_fifo_foreground_lock_cannot_block_worker_cancellation() {
    use std::{
        ffi::CString,
        os::unix::{ffi::OsStrExt, fs::OpenOptionsExt},
        process::{Child, Command, Stdio},
    };
    const ROLE: &str = "AUDIO_REVIEW_FOREGROUND_FIFO";
    if std::env::var_os(ROLE).is_some() {
        let root = std::path::PathBuf::from(std::env::var_os("AUDIO_REVIEW_ROOT").unwrap());
        let executable = program(
            &root,
            r#"#!/bin/sh
previous=
for arg in "$@"; do
  [ "$previous" = "--state" ] && state="$arg"
  previous="$arg"
done
touch "$state/fifo-worker-started"
sleep 30 &
child=$!
trap 'wait "$child" 2>/dev/null; exit 130' INT TERM
wait "$child"
"#,
        );
        let mut config = WatchConfig::new(
            &root,
            1,
            IdlePolicy::default(),
            Arc::new(AtomicBool::new(false)),
            "fifo foreground mock",
        );
        config.probe = Arc::new(|_| idle(true));
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(5);
        config.worker_grace = Duration::from_millis(500);
        let error = watch_audio(config, &root, &job(&root), &executable).unwrap_err();
        assert!(error.contains("no complete audio checkpoint"), "{error}");
        return;
    }
    struct Reap(Child);
    impl Drop for Reap {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let mut child = Reap(
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "audio_idle::tests::review_fifo_foreground_lock_cannot_block_worker_cancellation",
                "--nocapture",
            ])
            .env(ROLE, "1")
            .env("AUDIO_REVIEW_ROOT", root)
            .env(
                "DDUST_RUN_REGISTRY",
                root.join("synthetic-registry/runs.json"),
            )
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    wait_file(&root.join("fifo-worker-started"));
    let fifo = root.join("pending-fifo");
    let name = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let foreground = root.join("foreground.lock");
    fs::rename(&fifo, &foreground).unwrap();
    thread::sleep(Duration::from_millis(200));
    fs::write(root.join("stop.request"), b"stop").unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    let released = loop {
        if Lease::try_acquire(&root.join("device.lock"))
            .unwrap()
            .is_some()
        {
            break true;
        }
        if Instant::now() >= deadline {
            break false;
        }
        thread::sleep(Duration::from_millis(10));
    };
    // Unblock the known-bad baseline before evaluating the assertion.
    let _reader = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK)
        .open(&foreground)
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(status.success(), "{status}");
            break;
        }
        assert!(
            Instant::now() < deadline,
            "scheduler did not exit after opening FIFO reader"
        );
        thread::sleep(Duration::from_millis(10));
    }
    assert!(released, "foreground FIFO blocked device release");
}

fn check_blocked_output_does_not_prevent_scheduler_stop(selected: &str) {
    use std::{
        io::{ErrorKind, Read, Write},
        os::{
            fd::AsRawFd,
            unix::{net::UnixStream, process::CommandExt},
        },
        process::{Child, Command, Stdio},
    };
    const ROLE: &str = "AUDIO_REVIEW_OUTPUT_ROLE";
    if let Ok(role) = std::env::var(ROLE) {
        let root = std::path::PathBuf::from(std::env::var_os("AUDIO_REVIEW_ROOT").unwrap());
        let executable = if role == "report" {
            let report = write_checkpoint(
                &root.join("stage"),
                AudioProgress {
                    epoch: 0,
                    next_index: 1,
                    updates: 1,
                },
            );
            fs::write(
                root.join("mock-output.json"),
                serde_json::to_vec(&report).unwrap(),
            )
            .unwrap();
            program(
                &root,
                r#"#!/bin/sh
previous=
for arg in "$@"; do
  [ "$previous" = "--state" ] && state="$arg"
  previous="$arg"
done
cp -R "$state/stage/audio-conditioned-v1" "$state/audio-conditioned-v1" || exit 93
cat "$state/mock-output.json"
"#,
            )
        } else {
            program(&root, "#!/bin/sh\nexit 97\n")
        };
        let fd: i32 = std::env::var("AUDIO_REVIEW_OUTPUT_FD")
            .unwrap()
            .parse()
            .unwrap();
        let target = if role == "report" {
            libc::STDOUT_FILENO
        } else {
            libc::STDERR_FILENO
        };
        assert_eq!(unsafe { libc::dup2(fd, target) }, target);
        let mut config = WatchConfig::new(
            &root,
            1,
            IdlePolicy::default(),
            Arc::new(AtomicBool::new(false)),
            "output mock",
        );
        config.probe = Arc::new(move |_| idle(role == "report"));
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(5);
        config.worker_grace = Duration::from_millis(500);
        watch_audio(config, &root, &job(&root), &executable).unwrap();
        // Do not let the test harness itself write to the deliberately full fd.
        std::process::exit(0);
    }
    struct Reap(Child);
    impl Drop for Reap {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let role = selected;
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let (mut reader, mut writer) = UnixStream::pair().unwrap();
    // This socket is created by the test and has not been shared with the child yet.
    writer.set_nonblocking(true).unwrap();
    loop {
        match writer.write(&[b'x'; 8192]) {
            Ok(0) => panic!("unexpected zero while filling output"),
            Ok(_) => {}
            Err(error) if error.kind() == ErrorKind::WouldBlock => break,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => panic!("cannot fill output: {error}"),
        }
    }
    writer.set_nonblocking(false).unwrap();
    let fd = writer.as_raw_fd();
    let mut command = Command::new(std::env::current_exe().unwrap());
    let test = if role == "waiting" {
        "audio_idle::tests::review_blocked_waiting_output_never_prevents_stop"
    } else {
        "audio_idle::tests::review_blocked_round_output_never_prevents_stop"
    };
    command
        .args(["--exact", test, "--nocapture"])
        .env(ROLE, role)
        .env("AUDIO_REVIEW_ROOT", root)
        .env("AUDIO_REVIEW_OUTPUT_FD", fd.to_string())
        .env(
            "DDUST_RUN_REGISTRY",
            root.join("synthetic-registry/runs.json"),
        )
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    unsafe {
        command.pre_exec(move || {
            let flags = libc::fcntl(fd, libc::F_GETFD);
            if flags < 0 || libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = Reap(command.spawn().unwrap());
    drop(writer);
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        let phase = fs::read(root.join("status.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .and_then(|value| value["phase"].as_str().map(str::to_owned));
        let ready = if role == "waiting" {
            phase.as_deref() == Some("waiting")
        } else {
            phase.as_deref() == Some("running")
                && root.join("audio-conditioned-v1/latest.json").exists()
                && Lease::try_acquire(&root.join("device.lock"))
                    .unwrap()
                    .is_some()
        };
        if ready {
            break;
        }
        assert!(
            child.0.try_wait().unwrap().is_none(),
            "{role} scheduler exited before observation"
        );
        assert!(
            Instant::now() < deadline,
            "{role} scheduler did not reach output"
        );
        thread::sleep(Duration::from_millis(5));
    }
    fs::write(root.join("stop.request"), b"stop").unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    let released = loop {
        if Lease::try_acquire(&root.join("runner.lock"))
            .unwrap()
            .is_some()
        {
            break true;
        }
        if Instant::now() >= deadline {
            break false;
        }
        thread::sleep(Duration::from_millis(10));
    };
    let deadline = Instant::now() + Duration::from_secs(2);
    let exited_before_drain = loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(status.success(), "{role}: {status}");
            break true;
        }
        if Instant::now() >= deadline {
            break false;
        }
        thread::sleep(Duration::from_millis(10));
    };
    reader.set_nonblocking(true).unwrap();
    let mut buffer = [0_u8; 8192];
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        match reader.read(&mut buffer) {
            Ok(_) => {}
            Err(error) if error.kind() == ErrorKind::WouldBlock => {}
            Err(error) => panic!("cannot drain {role} output: {error}"),
        }
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(status.success(), "{role}: {status}");
            break;
        }
        assert!(
            Instant::now() < deadline,
            "{role} scheduler did not stop after drain"
        );
        thread::sleep(Duration::from_millis(5));
    }
    println!("{role}: runner released before draining output={released}");
    assert!(
        released,
        "blocked {role} output prevented stop from releasing runner"
    );
    assert!(
        exited_before_drain,
        "blocked {role} output prevented process exit"
    );
    if role == "report" {
        let mut report = write_checkpoint(
            &root.join("resume-stage"),
            AudioProgress {
                epoch: 2,
                next_index: 0,
                updates: 4,
            },
        );
        report["status"] = "complete".into();
        fs::write(
            root.join("resume-output.json"),
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
        let executable = program(
            root,
            r#"#!/bin/sh
previous=
for arg in "$@"; do
  [ "$previous" = "--state" ] && state="$arg"
  previous="$arg"
done
cp -R "$state/resume-stage/audio-conditioned-v1/." "$state/audio-conditioned-v1/" || exit 93
cat "$state/resume-output.json"
"#,
        );
        let mut config = WatchConfig::new(
            root,
            1,
            IdlePolicy::default(),
            Arc::new(AtomicBool::new(false)),
            "resume after blocked output",
        );
        config.probe = Arc::new(|_| idle(true));
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(5);
        config.worker_grace = Duration::from_millis(500);
        let mut resume = job(root);
        resume.start = AudioStart {
            fresh: false,
            resume: true,
        };
        watch_audio(config, root, &resume, &executable).unwrap();
        let status: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("status.json")).unwrap()).unwrap();
        assert_eq!(status["phase"], "complete");
    } else {
        assert!(
            preflight(root, &job(root)).is_ok(),
            "waiting did not start a worker or reserve state"
        );
    }
}

#[test]
fn review_blocked_waiting_output_never_prevents_stop() {
    check_blocked_output_does_not_prevent_scheduler_stop("waiting");
}

#[test]
fn review_blocked_round_output_never_prevents_stop() {
    check_blocked_output_does_not_prevent_scheduler_stop("report");
}

#[test]
fn review_final_round_report_is_recoverable_when_stdout_is_blocked() {
    use std::{
        io::{ErrorKind, Read, Write},
        os::{
            fd::AsRawFd,
            unix::{net::UnixStream, process::CommandExt},
        },
        process::{Child, Command, Stdio},
    };
    const ROLE: &str = "AUDIO_REVIEW_FINAL_REPORT";
    if std::env::var_os(ROLE).is_some() {
        let root = std::path::PathBuf::from(std::env::var_os("AUDIO_REVIEW_ROOT").unwrap());
        let mut report = write_checkpoint(
            &root.join("stage"),
            AudioProgress {
                epoch: 2,
                next_index: 0,
                updates: 4,
            },
        );
        report["status"] = "complete".into();
        fs::write(
            root.join("mock-output.json"),
            serde_json::to_vec(&report).unwrap(),
        )
        .unwrap();
        let executable = program(
            &root,
            r#"#!/bin/sh
previous=
for arg in "$@"; do
  [ "$previous" = "--state" ] && state="$arg"
  previous="$arg"
done
cp -R "$state/stage/audio-conditioned-v1" "$state/audio-conditioned-v1" || exit 93
cat "$state/mock-output.json"
"#,
        );
        let fd: i32 = std::env::var("AUDIO_REVIEW_OUTPUT_FD")
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(
            unsafe { libc::dup2(fd, libc::STDOUT_FILENO) },
            libc::STDOUT_FILENO
        );
        let mut config = WatchConfig::new(
            &root,
            1,
            IdlePolicy::default(),
            Arc::new(AtomicBool::new(false)),
            "final report mock",
        );
        config.probe = Arc::new(|_| idle(true));
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(5);
        config.worker_grace = Duration::from_millis(500);
        watch_audio(config, &root, &job(&root), &executable).unwrap();
        std::process::exit(0);
    }
    struct Reap(Child);
    impl Drop for Reap {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let (mut reader, mut writer) = UnixStream::pair().unwrap();
    writer.set_nonblocking(true).unwrap();
    loop {
        match writer.write(&[b'x'; 8192]) {
            Ok(0) => panic!("unexpected zero while filling stdout"),
            Ok(_) => {}
            Err(error) if error.kind() == ErrorKind::WouldBlock => break,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => panic!("cannot fill stdout: {error}"),
        }
    }
    writer.set_nonblocking(false).unwrap();
    let fd = writer.as_raw_fd();
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "audio_idle::tests::review_final_round_report_is_recoverable_when_stdout_is_blocked",
            "--nocapture",
        ])
        .env(ROLE, "1")
        .env("AUDIO_REVIEW_ROOT", root)
        .env("AUDIO_REVIEW_OUTPUT_FD", fd.to_string())
        .env(
            "DDUST_RUN_REGISTRY",
            root.join("synthetic-registry/runs.json"),
        )
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    unsafe {
        command.pre_exec(move || {
            let flags = libc::fcntl(fd, libc::F_GETFD);
            if flags < 0 || libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = Reap(command.spawn().unwrap());
    drop(writer);
    let deadline = Instant::now() + Duration::from_secs(8);
    let status = loop {
        if let Ok(bytes) = fs::read(root.join("status.json"))
            && let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes)
            && value["phase"] == "complete"
        {
            break Some(value);
        }
        if Instant::now() >= deadline {
            break None;
        }
        thread::sleep(Duration::from_millis(10));
    };
    let deadline = Instant::now() + Duration::from_secs(2);
    let runner_released = loop {
        if Lease::try_acquire(&root.join("runner.lock"))
            .unwrap()
            .is_some()
        {
            break true;
        }
        if Instant::now() >= deadline {
            break false;
        }
        thread::sleep(Duration::from_millis(10));
    };
    let deadline = Instant::now() + Duration::from_secs(2);
    let exited_before_drain = loop {
        if let Some(exit) = child.0.try_wait().unwrap() {
            assert!(exit.success(), "{exit}");
            break true;
        }
        if Instant::now() >= deadline {
            break false;
        }
        thread::sleep(Duration::from_millis(10));
    };
    let report = fs::read_dir(root.join("idle-round-reports"))
        .ok()
        .and_then(|mut entries| entries.next())
        .and_then(Result::ok)
        .and_then(|entry| {
            let path = entry.path();
            let bytes = fs::read(&path).ok()?;
            Some((
                path,
                serde_json::from_slice::<serde_json::Value>(&bytes).ok()?,
            ))
        });
    reader.set_nonblocking(true).unwrap();
    let mut buffer = [0_u8; 8192];
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        match reader.read(&mut buffer) {
            Ok(_) => {}
            Err(error) if error.kind() == ErrorKind::WouldBlock => {}
            Err(error) => panic!("cannot drain stdout: {error}"),
        }
        if let Some(exit) = child.0.try_wait().unwrap() {
            assert!(exit.success(), "{exit}");
            break;
        }
        assert!(
            Instant::now() < deadline,
            "test child did not finish after drain"
        );
        thread::sleep(Duration::from_millis(5));
    }
    let status = status.expect("final status did not complete while stdout was blocked");
    let detail = status["detail"].as_str().unwrap();
    assert!(
        detail.contains("live report output unconfirmed") && detail.contains("idle-round-reports"),
        "{detail}"
    );
    assert!(runner_released, "runner was still held");
    assert!(exited_before_drain, "scheduler process did not exit");
    let (path, report) = report.expect("no recoverable round report");
    assert!(detail.contains(&path.display().to_string()), "{detail}");
    assert_eq!(report["training_status"], "complete");
    assert_eq!(report["quality"], "not_assessed");
    assert_eq!(report["progress"]["updates"], 4);
}

#[test]
fn review_blocked_idle_probe_cancels_and_reaps_running_worker() {
    use std::process::{Child, Command, Stdio};
    use std::sync::atomic::AtomicUsize;
    const ROLE: &str = "AUDIO_390_BLOCKED_PROBE";
    if std::env::var_os(ROLE).is_some() {
        let root = std::path::PathBuf::from(std::env::var_os("AUDIO_390_ROOT").unwrap());
        let executable = program(
            &root,
            r#"#!/bin/sh
previous=
for arg in "$@"; do
  [ "$previous" = "--state" ] && state="$arg"
  previous="$arg"
done
printf '%s' "$$" > "$state/probe-worker-pid"
sleep 30 &
child=$!
trap 'wait "$child" 2>/dev/null; exit 130' INT TERM
wait "$child"
"#,
        );
        let calls = Arc::new(AtomicUsize::new(0));
        let mut config = WatchConfig::new(
            &root,
            2,
            IdlePolicy::default(),
            Arc::new(AtomicBool::new(false)),
            "blocked probe mock",
        );
        config.probe = Arc::new(move |path| {
            if calls.fetch_add(1, Ordering::AcqRel) > 0 {
                fs::write(path.join("probe-blocked"), b"blocked").unwrap();
                while !path.join("probe-release").exists() {
                    thread::sleep(Duration::from_millis(5));
                }
            }
            idle(true)
        });
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(5);
        config.worker_grace = Duration::from_millis(300);
        let _ = watch_audio(config, &root, &job(&root), &executable);
        return;
    }
    struct Reap(Child);
    impl Drop for Reap {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let mut scheduler = Reap(
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "audio_idle::tests::review_blocked_idle_probe_cancels_and_reaps_running_worker",
                "--nocapture",
            ])
            .env(ROLE, "1")
            .env("AUDIO_390_ROOT", root)
            .env(
                "DDUST_RUN_REGISTRY",
                root.join("synthetic-registry/runs.json"),
            )
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    wait_file(&root.join("probe-blocked"));
    let deadline = Instant::now() + Duration::from_secs(8);
    let worker = loop {
        if let Ok(text) = fs::read_to_string(root.join("probe-worker-pid"))
            && let Ok(pid) = text.parse::<i32>()
        {
            break pid;
        }
        assert!(Instant::now() < deadline, "mock worker did not start");
        thread::sleep(Duration::from_millis(5));
    };
    let foreground = Lease::try_acquire(&root.join("foreground.lock"))
        .unwrap()
        .unwrap();
    fs::write(root.join("stop.request"), b"stop").unwrap();
    let deadline = Instant::now() + Duration::from_secs(4);
    let released = loop {
        if Lease::try_acquire(&root.join("device.lock"))
            .unwrap()
            .is_some()
        {
            break true;
        }
        if Instant::now() >= deadline {
            break false;
        }
        thread::sleep(Duration::from_millis(10));
    };
    let worker_gone = unsafe { libc::kill(worker, 0) } == -1;
    fs::write(root.join("probe-release"), b"release").unwrap();
    drop(foreground);
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        if let Some(status) = scheduler.0.try_wait().unwrap() {
            assert!(status.success(), "{status}");
            break;
        }
        if Instant::now() >= deadline {
            unsafe {
                libc::kill(-worker, libc::SIGKILL);
            }
            panic!("scheduler did not finish after releasing mock probe");
        }
        thread::sleep(Duration::from_millis(10));
    }
    println!("device_released_before_probe_release={released}; worker_reaped={worker_gone}");
    assert!(
        released,
        "blocked idle probe retained the device after stop"
    );
    assert!(
        worker_gone,
        "device released before worker process was reaped"
    );
}
