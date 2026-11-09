use omni_rsi::idle::{IdlePolicy, IdleSnapshot};

fn ready() -> IdleSnapshot {
    IdleSnapshot {
        idle_seconds: Some(301.0),
        ac_power: Some(true),
        low_power: Some(false),
        thermal_nominal: Some(true),
        memory_normal: Some(true),
        free_bytes: Some(20 * 1024 * 1024 * 1024),
        competing_process: Some(false),
    }
}

#[test]
fn idle_work_is_admitted_only_when_every_signal_is_known_and_safe() {
    let policy = IdlePolicy::default();
    assert!(policy.blocker(&ready()).is_none());
    let mut state = ready();
    state.idle_seconds = Some(299.0);
    assert!(policy.blocker(&state).is_some());
    let mut state = ready();
    state.ac_power = None;
    assert!(policy.blocker(&state).is_some());
    let mut state = ready();
    state.low_power = Some(true);
    assert!(policy.blocker(&state).is_some());
    let mut state = ready();
    state.thermal_nominal = Some(false);
    assert!(policy.blocker(&state).is_some());
    let mut state = ready();
    state.memory_normal = Some(false);
    assert!(policy.blocker(&state).is_some());
    let mut state = ready();
    state.free_bytes = Some(1);
    assert!(policy.blocker(&state).is_some());
    let mut state = ready();
    state.competing_process = Some(true);
    assert!(policy.blocker(&state).is_some());
}
