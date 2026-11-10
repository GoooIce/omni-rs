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
