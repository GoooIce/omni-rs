use omni_rsi::platform::competing_process_from_ps;

#[test]
fn worker_descendants_do_not_block_their_own_scheduler() {
    let processes = "1 0 /sbin/launchd /sbin/launchd\n10 1 /tmp/omni-rsi omni-rsi\n11 10 /opt/homebrew/bin/uv uv run\n12 11 /usr/bin/python3 /Users/test/ddust/trainer.py\n13 12 /tmp/r2t2 r2t2 transcribe\n";
    assert_eq!(competing_process_from_ps(processes, 10), Some(false));
    let other = format!("{processes}20 1 /usr/bin/python3 /Users/test/ddust/other.py\n");
    assert_eq!(competing_process_from_ps(&other, 10), Some(true));
    let app = format!("{processes}21 1 /tmp/r2t2-makepad r2t2-makepad\n");
    assert_eq!(competing_process_from_ps(&app, 10), Some(true));
    for executable in ["r2t2", "breeze-tts"] {
        let other = format!("{processes}22 1 /tmp/{executable} {executable}\n");
        assert_eq!(competing_process_from_ps(&other, 10), Some(true));
    }
}

#[test]
fn missing_or_corrupt_process_information_does_not_report_idle() {
    assert_eq!(
        competing_process_from_ps("1 0 /sbin/launchd launchd\n", 10),
        None
    );
    assert_eq!(
        competing_process_from_ps("10 1 /tmp/omni-rsi omni-rsi\nnot-a-process\n", 10),
        None
    );
    assert_eq!(
        competing_process_from_ps("10 1 /tmp/omni-rsi omni-rsi\n20 21 x x\n21 20 x x\n", 10),
        None
    );
}
