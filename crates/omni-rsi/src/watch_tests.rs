use super::*;
use std::{
    fs,
    sync::mpsc::{self, Receiver},
    thread,
};

#[derive(Debug, PartialEq, Eq)]
enum Event {
    Started(usize),
    Cancelled(usize),
}

fn mock_snapshot(idle: bool) -> IdleSnapshot {
    IdleSnapshot {
        idle_seconds: Some(if idle { 301.0 } else { 1.0 }),
        ac_power: Some(true),
        low_power: Some(false),
        thermal_nominal: Some(true),
        memory_normal: Some(true),
        free_bytes: Some(20 * 1024 * 1024 * 1024),
        competing_process: Some(false),
    }
}

fn receive(events: &Receiver<Event>) -> Event {
    events
        .recv_timeout(Duration::from_secs(4))
        .expect("scheduler event timed out")
}

fn wait_for_status(root: &Path, phase: &str) -> serde_json::Value {
    let deadline = Instant::now() + Duration::from_secs(4);
    loop {
        if let Ok(bytes) = fs::read(root.join("status.json")) {
            let status: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            if status["phase"] == phase {
                return status;
            }
        }
        assert!(Instant::now() < deadline, "scheduler did not reach {phase}");
        thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn mock_idle_scheduler_yields_and_resumes_without_model_or_quality_metrics() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().to_owned();
    let thread_root = root.clone();
    let ready = Arc::new(AtomicBool::new(false));
    let probe_ready = ready.clone();
    let shutdown = Arc::new(AtomicBool::new(false));
    let thread_shutdown = shutdown.clone();
    let (events_tx, events_rx) = mpsc::channel();

    let runner = thread::spawn(move || {
        let mut config = WatchConfig::new(
            &thread_root,
            1,
            IdlePolicy::default(),
            thread_shutdown,
            "mock work",
        );
        config.probe = Arc::new(move |_| mock_snapshot(probe_ready.load(Ordering::Acquire)));
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(5);
        let checkpoint = thread_root.join("mock-checkpoint");
        let mut attempt = 0;
        watch(
            config,
            || Ok(None),
            |cancel| {
                attempt += 1;
                if attempt > 1 {
                    assert_eq!(
                        fs::read_to_string(&checkpoint).unwrap(),
                        (attempt - 1).to_string()
                    );
                }
                fs::write(&checkpoint, attempt.to_string()).unwrap();
                events_tx.send(Event::Started(attempt)).unwrap();
                if attempt < 3 {
                    let deadline = Instant::now() + Duration::from_secs(4);
                    while !cancel.load(Ordering::Acquire) && Instant::now() < deadline {
                        thread::sleep(Duration::from_millis(5));
                    }
                    assert!(
                        cancel.load(Ordering::Acquire),
                        "mock work was not cancelled"
                    );
                    events_tx.send(Event::Cancelled(attempt)).unwrap();
                    return Err("mock work interrupted".into());
                }
                Ok(json!({"complete":true}))
            },
        )
    });

    let status = wait_for_status(&root, "waiting");
    assert_eq!(status["detail"], "recent or unknown user activity");
    assert!(events_rx.recv_timeout(Duration::from_millis(50)).is_err());

    ready.store(true, Ordering::Release);
    assert_eq!(receive(&events_rx), Event::Started(1));
    ready.store(false, Ordering::Release);
    assert_eq!(receive(&events_rx), Event::Cancelled(1));
    assert_eq!(
        wait_for_status(&root, "waiting")["completed_this_session"],
        0
    );

    ready.store(true, Ordering::Release);
    assert_eq!(receive(&events_rx), Event::Started(2));
    let foreground = Lease::try_acquire(&root.join("foreground.lock"))
        .unwrap()
        .unwrap();
    assert_eq!(receive(&events_rx), Event::Cancelled(2));
    assert_eq!(
        wait_for_status(&root, "waiting")["completed_this_session"],
        0
    );
    assert!(events_rx.recv_timeout(Duration::from_millis(50)).is_err());
    drop(foreground);

    assert_eq!(receive(&events_rx), Event::Started(3));
    runner.join().unwrap().unwrap();
    let status = wait_for_status(&root, "complete");
    assert_eq!(status["completed_this_session"], 1);
    assert_eq!(
        fs::read_to_string(root.join("mock-checkpoint")).unwrap(),
        "3"
    );
    assert!(!shutdown.load(Ordering::Acquire));
}

#[test]
fn mock_idle_scheduler_stops_while_waiting_for_work() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().to_owned();
    let thread_root = root.clone();
    let runner = thread::spawn(move || {
        let mut config = WatchConfig::new(
            &thread_root,
            1,
            IdlePolicy::default(),
            Arc::new(AtomicBool::new(false)),
            "mock work",
        );
        config.probe = Arc::new(|_| panic!("idle probe should not run without work"));
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(5);
        watch(
            config,
            || Ok(Some("waiting for mock work")),
            |_| panic!("mock work should not start"),
        )
    });
    assert_eq!(
        wait_for_status(&root, "waiting")["detail"],
        "waiting for mock work"
    );
    atomic_write(&root.join("stop.request"), b"stop\n").unwrap();
    runner.join().unwrap().unwrap();
    assert_eq!(
        wait_for_status(&root, "stopped")["completed_this_session"],
        0
    );
    assert!(!root.join("stop.request").exists());
}

#[test]
fn mock_pending_work_does_not_count_as_a_completed_round() {
    let directory = tempfile::tempdir().unwrap();
    let mut config = WatchConfig::new(
        directory.path(),
        1,
        IdlePolicy::default(),
        Arc::new(AtomicBool::new(false)),
        "mock work",
    );
    config.probe = Arc::new(|_| mock_snapshot(true));
    config.ddust_lease = || Ok(None);
    config.wait_unit = Duration::from_millis(5);
    config.watchdog_interval = Duration::from_millis(5);
    let mut attempts = 0;
    watch(
        config,
        || Ok(None),
        |_| {
            attempts += 1;
            Ok(if attempts == 1 {
                json!({"pending":true})
            } else {
                json!({"complete":true})
            })
        },
    )
    .unwrap();
    assert_eq!(attempts, 2);
    assert_eq!(
        wait_for_status(directory.path(), "complete")["completed_this_session"],
        1
    );
}

#[test]
fn review_oversized_mock_round_report_is_rejected_before_persistence() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let mut config = WatchConfig::new(
        root,
        1,
        IdlePolicy::default(),
        Arc::new(AtomicBool::new(false)),
        "oversized mock report",
    );
    config.probe = Arc::new(|_| mock_snapshot(true));
    config.ddust_lease = || Ok(None);
    config.wait_unit = Duration::from_millis(5);
    config.watchdog_interval = Duration::from_millis(5);
    let error = watch(
        config,
        || Ok(None),
        |_| Ok(json!({"content":"x".repeat(1_048_576)})),
    )
    .expect_err("oversized report was accepted");
    assert!(
        error.contains("report") && error.contains("limit"),
        "{error}"
    );
    assert!(!root.join("idle-round-reports").exists());
    let status = wait_for_status(root, "error");
    assert_eq!(status["completed_this_session"], 0);
    assert!(
        Lease::try_acquire(&root.join("device.lock"))
            .unwrap()
            .is_some()
    );
    assert!(
        Lease::try_acquire(&root.join("runner.lock"))
            .unwrap()
            .is_some()
    );
}

#[test]
fn review_report_serialization_counts_quotes_and_final_newline_in_the_limit() {
    let exact = serde_json::Value::String("x".repeat(MAX_WATCH_REPORT_BYTES - 3));
    let bytes = serialize_watch_report(&exact).unwrap();
    assert_eq!(bytes.len(), MAX_WATCH_REPORT_BYTES);
    assert_eq!(bytes.last(), Some(&b'\n'));
    let overflow = serde_json::Value::String("x".repeat(MAX_WATCH_REPORT_BYTES - 2));
    assert!(
        serialize_watch_report(&overflow)
            .unwrap_err()
            .contains("byte limit")
    );
}

#[test]
fn review_report_persistence_also_rejects_unbounded_input() {
    let directory = tempfile::tempdir().unwrap();
    let oversized = vec![b'x'; MAX_WATCH_REPORT_BYTES + 1];
    let error = persist_watch_report(directory.path(), &oversized).unwrap_err();
    assert!(error.contains("byte limit"), "{error}");
    assert!(!directory.path().join("idle-round-reports").exists());
}

#[test]
fn review_report_persistence_failure_records_error_and_preserves_existing_file() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let reports = root.join("idle-round-reports");
    fs::write(&reports, b"preserve existing file").unwrap();
    let mut config = WatchConfig::new(
        root,
        1,
        IdlePolicy::default(),
        Arc::new(AtomicBool::new(false)),
        "report failure mock",
    );
    config.probe = Arc::new(|_| mock_snapshot(true));
    config.ddust_lease = || Ok(None);
    config.wait_unit = Duration::from_millis(5);
    config.watchdog_interval = Duration::from_millis(5);
    let error = watch(config, || Ok(None), |_| Ok(json!({"complete":true})))
        .expect_err("report persistence unexpectedly succeeded");
    assert!(
        error.contains("cannot persist completed round report"),
        "{error}"
    );
    let status = wait_for_status(root, "error");
    assert_eq!(status["completed_this_session"], 0);
    assert_eq!(fs::read(&reports).unwrap(), b"preserve existing file");
    assert!(
        Lease::try_acquire(&root.join("device.lock"))
            .unwrap()
            .is_some()
    );
    assert!(
        Lease::try_acquire(&root.join("runner.lock"))
            .unwrap()
            .is_some()
    );
}

#[test]
fn review_cli_hook_delegates_other_thread_during_watchdog_probe_failure() {
    use std::sync::atomic::AtomicUsize;
    const ROLE: &str = "AUDIO_REVIEW_CONCURRENT_PANIC";
    if std::env::var_os(ROLE).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "watch_tests::review_cli_hook_delegates_other_thread_during_watchdog_probe_failure",
                "--nocapture",
            ])
            .env(ROLE, "1")
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
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    std::panic::set_hook(Box::new(move |_| {
        observed.fetch_add(1, Ordering::AcqRel);
    }));
    // Model the binary startup, before any application thread is created.
    install_watchdog_panic_hook();
    let trigger = Arc::new(AtomicBool::new(false));
    let entered = Arc::new(AtomicBool::new(false));
    let other_done = Arc::new(AtomicBool::new(false));
    let other_entered = entered.clone();
    let done = other_done.clone();
    let other = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(4);
        while !other_entered.load(Ordering::Acquire) {
            assert!(Instant::now() < deadline, "watchdog probe did not enter");
            thread::sleep(Duration::from_millis(5));
        }
        assert!(std::panic::catch_unwind(|| panic!("ordinary thread panic")).is_err());
        done.store(true, Ordering::Release);
    });
    let directory = tempfile::tempdir().unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));
    let mut config = WatchConfig::new(
        directory.path(),
        1,
        IdlePolicy::default(),
        shutdown.clone(),
        "panic hook mock",
    );
    let check_trigger = trigger.clone();
    config.probe = Arc::new(move |_| {
        if check_trigger.load(Ordering::Acquire) {
            entered.store(true, Ordering::Release);
            let deadline = Instant::now() + Duration::from_secs(4);
            while !other_done.load(Ordering::Acquire) {
                assert!(Instant::now() < deadline, "ordinary panic did not finish");
                thread::sleep(Duration::from_millis(5));
            }
            panic!("watchdog probe failed");
        }
        mock_snapshot(true)
    });
    config.ddust_lease = || Ok(None);
    config.wait_unit = Duration::from_millis(5);
    config.watchdog_interval = Duration::from_millis(5);
    let error = watch(
        config,
        || Ok(None),
        |cancel| {
            trigger.store(true, Ordering::Release);
            let deadline = Instant::now() + Duration::from_secs(4);
            while !cancel.load(Ordering::Acquire) {
                assert!(
                    Instant::now() < deadline,
                    "probe failure did not cancel work"
                );
                thread::sleep(Duration::from_millis(5));
            }
            shutdown.store(true, Ordering::Release);
            Ok(json!({"complete":false}))
        },
    )
    .unwrap_err();
    assert!(error.contains("idle probe failed"), "{error}");
    other.join().unwrap();
    assert_eq!(
        calls.load(Ordering::Acquire),
        1,
        "other thread diagnostic was lost or probe panic escaped"
    );
}

#[test]
fn review_successful_round_racing_cancellation_retains_report() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let shutdown = Arc::new(AtomicBool::new(false));
    let changed = Arc::new(AtomicBool::new(false));
    let probe_changed = changed.clone();
    let mut config = WatchConfig::new(
        root,
        1,
        IdlePolicy::default(),
        shutdown.clone(),
        "success race mock",
    );
    config.probe = Arc::new(move |_| {
        let mut snapshot = mock_snapshot(true);
        if probe_changed.load(Ordering::Acquire) {
            snapshot.idle_seconds = Some(0.0);
        }
        snapshot
    });
    config.ddust_lease = || Ok(None);
    config.wait_unit = Duration::from_millis(5);
    config.watchdog_interval = Duration::from_millis(5);
    let mut yielded = false;
    watch_with_leases(
        config,
        || Ok(None),
        |cancel, _| {
            assert!(
                std::process::Command::new("/usr/bin/true")
                    .status()
                    .unwrap()
                    .success()
            );
            changed.store(true, Ordering::Release);
            let deadline = Instant::now() + Duration::from_secs(4);
            while !cancel.load(Ordering::Acquire) {
                assert!(Instant::now() < deadline, "watchdog did not cancel");
                thread::sleep(Duration::from_millis(5));
            }
            shutdown.store(true, Ordering::Release);
            Ok(json!({"complete": true, "synthetic": "completed"}))
        },
        || {
            yielded = true;
            Ok(())
        },
    )
    .unwrap();
    assert!(yielded);
    let status = wait_for_status(root, "stopped");
    assert_eq!(status["completed_this_session"], 1);
    let entries: Vec<_> = fs::read_dir(root.join("idle-round-reports"))
        .unwrap()
        .collect();
    assert_eq!(entries.len(), 1);
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(entries[0].as_ref().unwrap().path()).unwrap()).unwrap();
    assert_eq!(report["synthetic"], "completed");
}

#[cfg(unix)]
#[test]
fn review_failed_round_keeps_previously_unconfirmed_report_path() {
    use std::{
        io::{ErrorKind, Read, Write},
        os::{
            fd::AsRawFd,
            unix::{net::UnixStream, process::CommandExt},
        },
        process::{Child, Command, Stdio},
    };
    const ROLE: &str = "AUDIO_390_REPORT_ROLE";
    if std::env::var_os(ROLE).is_some() {
        let root = PathBuf::from(std::env::var_os("AUDIO_390_ROOT").unwrap());
        let fd: i32 = std::env::var("AUDIO_390_OUTPUT_FD")
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(
            unsafe { libc::dup2(fd, libc::STDOUT_FILENO) },
            libc::STDOUT_FILENO
        );
        let mut config = WatchConfig::new(
            &root,
            2,
            IdlePolicy::default(),
            Arc::new(AtomicBool::new(false)),
            "report failure mock",
        );
        config.probe = Arc::new(|_| mock_snapshot(true));
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(5);
        let mut calls = 0;
        let error = watch(
            config,
            || Ok(None),
            |_| {
                calls += 1;
                if calls == 1 {
                    Ok(json!({"complete":false,"synthetic":"saved"}))
                } else {
                    Err("synthetic second-round failure".into())
                }
            },
        )
        .unwrap_err();
        assert!(error.contains("synthetic second-round failure"), "{error}");
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
            Ok(0) => panic!("zero write"),
            Ok(_) => {}
            Err(error) if error.kind() == ErrorKind::WouldBlock => break,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => panic!("fill stdout: {error}"),
        }
    }
    writer.set_nonblocking(false).unwrap();
    let fd = writer.as_raw_fd();
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "watch_tests::review_failed_round_keeps_previously_unconfirmed_report_path",
            "--nocapture",
        ])
        .env(ROLE, "1")
        .env("AUDIO_390_ROOT", root)
        .env("AUDIO_390_OUTPUT_FD", fd.to_string())
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
    let exited = loop {
        if let Some(status) = child.0.try_wait().unwrap() {
            break Some(status);
        }
        if Instant::now() >= deadline {
            break None;
        }
        thread::sleep(Duration::from_millis(5));
    };
    let status: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("status.json")).unwrap()).unwrap();
    let reports: Vec<_> = fs::read_dir(root.join("idle-round-reports"))
        .unwrap()
        .collect();
    assert_eq!(reports.len(), 1);
    let path = reports[0].as_ref().unwrap().path();
    let saved: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    reader.set_nonblocking(true).unwrap();
    let mut buffer = [0_u8; 8192];
    let _ = reader.read(&mut buffer);
    assert!(
        exited.is_some_and(|status| status.success()),
        "scheduler did not exit"
    );
    assert_eq!(status["phase"], "error");
    assert_eq!(status["completed_this_session"], 1);
    assert_eq!(saved["synthetic"], "saved");
    assert!(
        status["detail"]
            .as_str()
            .unwrap()
            .contains(&path.display().to_string()),
        "status lost report path: {status}"
    );
}

#[cfg(unix)]
#[test]
fn review_scheduler_panic_releases_leases_with_blocked_stderr() {
    use std::{
        io::{ErrorKind, Read, Write},
        os::{fd::OwnedFd, unix::net::UnixStream},
        process::{Child, Command, Stdio},
    };
    const ROLE: &str = "AUDIO_390_PANIC_ROLE";
    if std::env::var_os(ROLE).is_some() {
        let root = PathBuf::from(std::env::var_os("AUDIO_390_ROOT").unwrap());
        install_watchdog_panic_hook();
        let mut config = WatchConfig::new(
            &root,
            1,
            IdlePolicy::default(),
            Arc::new(AtomicBool::new(false)),
            "scheduler panic mock",
        );
        config.probe = Arc::new(|_| mock_snapshot(true));
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(5);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            watch_with_leases(
                config,
                || Ok(None),
                |_, _| -> Result<serde_json::Value, String> {
                    fs::write(root.join("panic-entered"), b"ready").unwrap();
                    panic!("synthetic scheduler callback panic")
                },
                || Ok(()),
            )
        }));
        assert!(matches!(outcome, Err(_) | Ok(Err(_))));
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
            Ok(0) => panic!("zero write"),
            Ok(_) => {}
            Err(error) if error.kind() == ErrorKind::WouldBlock => break,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => panic!("fill stderr: {error}"),
        }
    }
    writer.set_nonblocking(false).unwrap();
    let mut child = Reap(
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "watch_tests::review_scheduler_panic_releases_leases_with_blocked_stderr",
                "--nocapture",
            ])
            .env(ROLE, "1")
            .env("AUDIO_390_ROOT", root)
            .stdout(Stdio::null())
            .stderr(Stdio::from(OwnedFd::from(writer)))
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(8);
    while !root.join("panic-entered").exists() {
        assert!(Instant::now() < deadline, "panic callback did not start");
        thread::sleep(Duration::from_millis(5));
    }
    let foreground = Lease::try_acquire(&root.join("foreground.lock"))
        .unwrap()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    let released = loop {
        let device = Lease::try_acquire(&root.join("device.lock")).unwrap();
        let runner = Lease::try_acquire(&root.join("runner.lock")).unwrap();
        if device.is_some() && runner.is_some() {
            break true;
        }
        if Instant::now() >= deadline {
            break false;
        }
        thread::sleep(Duration::from_millis(5));
    };
    reader.set_nonblocking(true).unwrap();
    let mut buffer = [0_u8; 8192];
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        match reader.read(&mut buffer) {
            Ok(_) => {}
            Err(error) if error.kind() == ErrorKind::WouldBlock => {}
            Err(error) => panic!("drain stderr: {error}"),
        }
        if let Some(status) = child.0.try_wait().unwrap() {
            assert!(status.success(), "{status}");
            break;
        }
        assert!(
            Instant::now() < deadline,
            "panic fixture did not exit after stderr drained"
        );
        thread::sleep(Duration::from_millis(5));
    }
    drop(foreground);
    assert!(
        released,
        "scheduler panic hook retained device or runner lease"
    );
}

#[test]
fn review_scheduler_panic_keeps_other_threads_original_hook() {
    use std::sync::atomic::AtomicUsize;
    const ROLE: &str = "AUDIO_390_SCHEDULER_HOOK_ROLE";
    if std::env::var_os(ROLE).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "watch_tests::review_scheduler_panic_keeps_other_threads_original_hook",
                "--nocapture",
            ])
            .env(ROLE, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let count = Arc::new(AtomicUsize::new(0));
    let observed = count.clone();
    std::panic::set_hook(Box::new(move |_| {
        observed.fetch_add(1, Ordering::AcqRel);
    }));
    install_watchdog_panic_hook();
    let temp = tempfile::tempdir().unwrap();
    let mut config = WatchConfig::new(
        temp.path(),
        1,
        IdlePolicy::default(),
        Arc::new(AtomicBool::new(false)),
        "scheduler panic",
    );
    config.probe = Arc::new(|_| mock_snapshot(true));
    config.ddust_lease = || Ok(None);
    config.watchdog_interval = Duration::from_millis(5);
    let error = watch(
        config,
        || Ok(None),
        |_| {
            thread::spawn(|| {
                let _ = std::panic::catch_unwind(|| panic!("other thread"));
            })
            .join()
            .unwrap();
            panic!("scheduler callback");
        },
    )
    .unwrap_err();
    assert!(error.contains("idle round panicked"));
    assert_eq!(
        count.load(Ordering::Acquire),
        1,
        "original hook must run only for the other thread"
    );
}

#[test]
fn review_finished_worker_does_not_wait_for_blocked_probe_to_release_device() {
    const ROLE: &str = "AUDIO_390_FINISHED_PROBE";
    if std::env::var_os(ROLE).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "watch_tests::review_finished_worker_does_not_wait_for_blocked_probe_to_release_device", "--nocapture"])
            .env(ROLE, "1").output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    use std::sync::atomic::AtomicUsize;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().to_owned();
    let thread_root = root.clone();
    let entered = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let completed = Arc::new(AtomicBool::new(false));
    let probe_entered = entered.clone();
    let probe_release = release.clone();
    let work_completed = completed.clone();
    let calls = Arc::new(AtomicUsize::new(0));
    let scheduler = thread::spawn(move || {
        let mut config = WatchConfig::new(
            &thread_root,
            1,
            IdlePolicy::default(),
            Arc::new(AtomicBool::new(false)),
            "finished worker mock",
        );
        config.probe = Arc::new(move |_| {
            if calls.fetch_add(1, Ordering::AcqRel) > 0 {
                probe_entered.store(true, Ordering::Release);
                while !probe_release.load(Ordering::Acquire) {
                    thread::sleep(Duration::from_millis(5));
                }
            }
            mock_snapshot(true)
        });
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(5);
        watch(
            config,
            || Ok(None),
            |_| {
                let deadline = Instant::now() + Duration::from_secs(4);
                while !entered.load(Ordering::Acquire) {
                    assert!(Instant::now() < deadline, "watchdog probe did not block");
                    thread::sleep(Duration::from_millis(5));
                }
                work_completed.store(true, Ordering::Release);
                Ok(json!({"complete":true, "synthetic":"worker-finished"}))
            },
        )
    });
    let deadline = Instant::now() + Duration::from_secs(8);
    while !completed.load(Ordering::Acquire) {
        assert!(Instant::now() < deadline, "mock worker did not finish");
        thread::sleep(Duration::from_millis(5));
    }
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
        thread::sleep(Duration::from_millis(5));
    };
    let mut second_error = None;
    let mut second_started = false;
    if released {
        let other = root.join("other-scheduler");
        fs::create_dir(&other).unwrap();
        let mut config = WatchConfig::new(
            &other,
            1,
            IdlePolicy::default(),
            Arc::new(AtomicBool::new(false)),
            "second mock",
        );
        config.probe = Arc::new(|_| mock_snapshot(true));
        config.ddust_lease = || Ok(None);
        second_error = watch(
            config,
            || Ok(None),
            |_| {
                second_started = true;
                Ok(json!({"complete": true}))
            },
        )
        .err();
    }
    release.store(true, Ordering::Release);
    let _ = scheduler.join().unwrap();
    assert!(
        released,
        "completed worker waited for blocked watchdog before device release"
    );
    assert!(
        !second_started,
        "a new worker started while the previous probe remained blocked"
    );
    assert!(
        second_error
            .unwrap_or_default()
            .contains("previous idle probe is still blocked")
    );
}

#[test]
fn review_independent_healthy_schedulers_can_probe_concurrently() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let first_root = first.path().to_owned();
    let (started, waiting) = mpsc::channel();
    let (release, proceed) = mpsc::channel();
    let other = thread::spawn(move || {
        let mut config = WatchConfig::new(
            &first_root,
            1,
            IdlePolicy::default(),
            Arc::new(AtomicBool::new(false)),
            "first mock",
        );
        config.probe = Arc::new(|_| mock_snapshot(true));
        config.ddust_lease = || Ok(None);
        config.watchdog_interval = Duration::from_millis(10);
        watch(
            config,
            || Ok(None),
            |_| {
                started.send(()).unwrap();
                proceed.recv_timeout(Duration::from_secs(4)).unwrap();
                Ok(json!({"complete": true}))
            },
        )
    });
    waiting.recv_timeout(Duration::from_secs(4)).unwrap();
    let mut config = WatchConfig::new(
        second.path(),
        1,
        IdlePolicy::default(),
        Arc::new(AtomicBool::new(false)),
        "second mock",
    );
    config.probe = Arc::new(|_| mock_snapshot(true));
    config.ddust_lease = || Ok(None);
    config.watchdog_interval = Duration::from_millis(10);
    let result = watch(config, || Ok(None), |_| Ok(json!({"complete": true})));
    release.send(()).unwrap();
    other.join().unwrap().unwrap();
    result.unwrap();
}
