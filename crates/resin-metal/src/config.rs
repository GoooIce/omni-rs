//! Engine-side experiment switches for the resin runners (#72).
//!
//! One mechanism, one switch: `multi_sequence_batch` lets every decode row of
//! one plan share a single multi-sequence verify forward. It is default off;
//! the config field is the durable switch and `RESIN_MULTI_SEQUENCE_BATCH=1`
//! is the same-binary experimental override the A/B protocol asks for
//! (docs/agent-workflow.md, "Performance measurement"). The header names the
//! switches and the measurement basis once per process when a switch is
//! enabled or `OMNI_AB_HEADER=1` is set.

use std::sync::Once;

/// Experimental engine configuration. Every field defaults off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResinConfig {
    /// Verify all decode rows of a plan (`>= 2`) in one target forward
    /// instead of one forward per row. Greedy DSpark only for now.
    pub multi_sequence_batch: bool,
}

impl ResinConfig {
    /// Defaults plus the experimental environment overrides.
    pub fn from_env() -> Self {
        Self::default().with_env_overrides()
    }

    /// Fold `RESIN_MULTI_SEQUENCE_BATCH=1` into this config. The environment
    /// is an experimental override: when set it wins over an explicit
    /// `false`, so one binary can A/B the switch (same-binary kill switch).
    pub fn with_env_overrides(mut self) -> Self {
        self.multi_sequence_batch |= env_flag("RESIN_MULTI_SEQUENCE_BATCH");
        self
    }
}

/// One header line naming every switch and the measurement basis.
pub fn experiment_header(config: &ResinConfig) -> String {
    format!(
        "resin experiment: multi_sequence_batch={} | basis: host wall clock at the runner seam \
         (prepare/forward/complete); per-row decode cost = forward time / plan rows; \
         no device timestamps",
        on_off(config.multi_sequence_batch)
    )
}

/// Print [`experiment_header`] once per process when a switch is enabled or
/// `OMNI_AB_HEADER=1`; a no-op otherwise.
pub fn print_header(config: &ResinConfig) {
    if !config.multi_sequence_batch && !env_flag("OMNI_AB_HEADER") {
        return;
    }
    static PRINTED: Once = Once::new();
    PRINTED.call_once(|| eprintln!("{}", experiment_header(config)));
}

fn on_off(enabled: bool) -> &'static str {
    if enabled { "on" } else { "off" }
}

/// `NAME=1`; any other value, or an unset variable, is off.
fn env_flag(name: &str) -> bool {
    matches!(std::env::var(name).ok().as_deref(), Some("1"))
}

#[cfg(test)]
mod tests {
    use super::{ResinConfig, experiment_header};

    #[test]
    fn config_defaults_every_switch_off() {
        assert!(!ResinConfig::default().multi_sequence_batch);
    }

    #[test]
    fn header_names_switch_state_and_measurement_basis() {
        let off = experiment_header(&ResinConfig::default());
        assert!(off.contains("multi_sequence_batch=off"), "{off}");
        assert!(off.contains("host wall clock"), "{off}");
        assert!(off.contains("per-row decode cost"), "{off}");
        assert!(off.contains("no device timestamps"), "{off}");

        let on = experiment_header(&ResinConfig {
            multi_sequence_batch: true,
        });
        assert!(on.contains("multi_sequence_batch=on"), "{on}");
    }
}
