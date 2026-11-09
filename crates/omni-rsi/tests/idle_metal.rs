#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use omni_rsi::chart::{ChartProgram, render};
use std::{
    path::Path,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}

fn status(root: &Path) -> serde_json::Value {
    std::fs::read(root.join("status.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or(serde_json::Value::Null)
}

#[test]
#[ignore = "requires a quiet Apple Silicon Mac, local LFM3 weights and a real GPU"]
fn foreground_reader_cancels_the_running_idle_round_and_receives_the_device() {
    let root = tempfile::tempdir().unwrap();
    let image = root.path().join("chart.png");
    render(&ChartProgram { left: 7, right: 3 })
        .unwrap()
        .save(&image)
        .unwrap();
    let log_path = root.path().join("watch.log");
    let log = std::fs::File::create(&log_path).unwrap();
    let binary = env!("CARGO_BIN_EXE_omni-rsi");
    let mut watcher = ChildGuard(
        Command::new(binary)
            .arg("--state")
            .arg(root.path())
            .args([
                "watch",
                "--rounds",
                "1",
                "--idle-seconds",
                "0",
                "--families",
                "2",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::from(log))
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    while status(root.path())["phase"] != "running" && Instant::now() < deadline {
        assert!(
            watcher.0.try_wait().unwrap().is_none(),
            "watch exited early"
        );
        thread::sleep(Duration::from_millis(100));
    }
    assert_eq!(
        status(root.path())["phase"],
        "running",
        "watch blocked: {}",
        std::fs::read_to_string(&log_path).unwrap_or_default()
    );
    thread::sleep(Duration::from_secs(8));
    assert!(!root.path().join("progress.json").exists());

    let reader_log = root.path().join("reader.log");
    let mut foreground = ChildGuard(
        Command::new(binary)
            .arg("--state")
            .arg(root.path())
            .args(["read", "--image"])
            .arg(&image)
            .args(["--question", "twice"])
            .stdout(Stdio::piped())
            .stderr(Stdio::from(std::fs::File::create(&reader_log).unwrap()))
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(60);
    let exit = loop {
        if let Some(exit) = foreground.0.try_wait().unwrap() {
            break exit;
        }
        assert!(
            Instant::now() < deadline,
            "foreground reader did not finish"
        );
        thread::sleep(Duration::from_millis(100));
    };
    let mut output = String::new();
    std::io::Read::read_to_string(foreground.0.stdout.as_mut().unwrap(), &mut output).unwrap();
    assert!(
        exit.success(),
        "foreground reader failed: {output}; {}",
        std::fs::read_to_string(&reader_log).unwrap_or_default()
    );
    let result: serde_json::Value = serde_json::from_str(&output).unwrap();
    assert_eq!(result["answer"], "Yes");
    assert_eq!(result["adapted"], false);
    assert!(!root.path().join("progress.json").exists());

    unsafe { libc::kill(watcher.0.id() as libc::pid_t, libc::SIGINT) };
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        if let Some(exit) = watcher.0.try_wait().unwrap() {
            assert!(exit.success());
            break;
        }
        assert!(Instant::now() < deadline, "watch did not stop");
        thread::sleep(Duration::from_millis(100));
    }
    assert_eq!(status(root.path())["phase"], "stopped");
}
