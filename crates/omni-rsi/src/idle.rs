//! Fail-closed admission policy for background evolution.
#[derive(Clone, Debug, Default)]
pub struct IdleSnapshot {
    pub idle_seconds: Option<f64>,
    pub ac_power: Option<bool>,
    pub low_power: Option<bool>,
    pub thermal_nominal: Option<bool>,
    pub memory_normal: Option<bool>,
    pub free_bytes: Option<u64>,
    pub competing_process: Option<bool>,
}
#[derive(Clone, Debug)]
pub struct IdlePolicy {
    pub min_idle_seconds: f64,
    pub min_free_bytes: u64,
}
impl Default for IdlePolicy {
    fn default() -> Self {
        Self {
            min_idle_seconds: 300.0,
            min_free_bytes: 5 * 1024 * 1024 * 1024,
        }
    }
}
impl IdlePolicy {
    /// Unknown, invalid, or adverse signals always block a new background step.
    pub fn blocker(&self, state: &IdleSnapshot) -> Option<&'static str> {
        if !self.min_idle_seconds.is_finite() || self.min_idle_seconds < 0.0 {
            return Some("invalid idle policy");
        }
        if !state
            .idle_seconds
            .is_some_and(|s| s.is_finite() && s >= self.min_idle_seconds)
        {
            return Some("recent or unknown user activity");
        }
        if state.ac_power != Some(true) {
            return Some("AC power unavailable or unknown");
        }
        if state.low_power != Some(false) {
            return Some("low power mode active or unknown");
        }
        if state.thermal_nominal != Some(true) {
            return Some("thermal state is not nominal or unknown");
        }
        if state.memory_normal != Some(true) {
            return Some("memory pressure or unknown state");
        }
        if !state.free_bytes.is_some_and(|n| n >= self.min_free_bytes) {
            return Some("disk space is insufficient or unknown");
        }
        if state.competing_process != Some(false) {
            return Some("another model process is active or unknown");
        }
        None
    }
}
