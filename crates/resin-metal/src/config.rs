//! Engine-side experiment switches for the resin runners (#72, #73, #74).
//!
//! One mechanism, one switch: `multi_sequence_batch` lets every decode row of
//! one plan share a single multi-sequence verify forward; `tree_batch` stacks
//! a capped best-first tree per co-batched sequence in that same forward;
//! `conversation_slots` lets a request resume a stored conversation
//! checkpoint instead of prefilling the whole prompt. All three are default
//! off; the config field is the durable switch and
//! `RESIN_MULTI_SEQUENCE_BATCH=1` / `RESIN_TREE_BATCH=1` /
//! `RESIN_CONVERSATION_SLOTS=1` are the same-binary experimental overrides the
//! A/B protocol asks for (docs/agent-workflow.md, "Performance measurement").
//! `tree_batch` is the composite (tree + batching) switch, not a master
//! switch for either solo mechanism: it changes only co-batched rows, never
//! the per-row path and never the model-side single-stream tree mode. The
//! header names the switches and the measurement basis once per process when
//! a switch is enabled or `OMNI_AB_HEADER=1` is set.

use std::sync::Once;

/// Experimental engine configuration. Every field defaults off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResinConfig {
    /// Verify all decode rows of a plan (`>= 2`) in one target forward
    /// instead of one forward per row. Greedy DSpark only for now.
    pub multi_sequence_batch: bool,
    /// Stack one capped best-first tree per co-batched sequence (#73): each
    /// member verifies at most 4 rows including the bonus, the plan at most
    /// 8. Also enables the co-batched path itself (it is the composite
    /// mode); solo rows keep the per-row chain.
    pub tree_batch: bool,
    /// Store conversation checkpoints for requests tagged with a slot key and
    /// resume the deepest prefix-matching one (#74). No slot key means the
    /// unchanged full prefill.
    pub conversation_slots: bool,
}

impl ResinConfig {
    /// Defaults plus the experimental environment overrides.
    pub fn from_env() -> Self {
        Self::default().with_env_overrides()
    }

    /// Fold `RESIN_MULTI_SEQUENCE_BATCH=1`, `RESIN_TREE_BATCH=1` and
    /// `RESIN_CONVERSATION_SLOTS=1` into this config. The environment is an
    /// experimental override: when set it wins over an explicit `false`, so
    /// one binary can A/B a switch (same-binary kill switch).
    pub fn with_env_overrides(mut self) -> Self {
        self.multi_sequence_batch |= env_flag("RESIN_MULTI_SEQUENCE_BATCH");
        self.tree_batch |= env_flag("RESIN_TREE_BATCH");
        self.conversation_slots |= env_flag("RESIN_CONVERSATION_SLOTS");
        self
    }

    /// Whether a decode plan with `rows` rows takes the co-batched path. The
    /// stacking switch counts: the composite mode is tree + batching.
    pub fn co_batches(&self, rows: usize) -> bool {
        (self.multi_sequence_batch || self.tree_batch) && rows >= 2
    }
}

/// One header line naming every switch and the measurement basis.
pub fn experiment_header(config: &ResinConfig) -> String {
    format!(
        "resin experiment: multi_sequence_batch={} tree_batch={} conversation_slots={} | \
         tree_batch is the composite tree+batch mode (co-batched rows only: <=4 rows per \
         sequence, <=8 per plan); basis: host wall clock at the runner seam \
         (prepare/forward/complete); per-row decode cost = forward time / plan rows; prefill \
         tokens and checkpoint hits/misses come from the session counters; no device timestamps",
        on_off(config.multi_sequence_batch),
        on_off(config.tree_batch),
        on_off(config.conversation_slots)
    )
}

/// Print [`experiment_header`] once per process when a switch is enabled or
/// `OMNI_AB_HEADER=1`; a no-op otherwise.
pub fn print_header(config: &ResinConfig) {
    if !config.multi_sequence_batch
        && !config.tree_batch
        && !config.conversation_slots
        && !env_flag("OMNI_AB_HEADER")
    {
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
        assert!(!ResinConfig::default().tree_batch);
        assert!(!ResinConfig::default().conversation_slots);
    }

    #[test]
    fn tree_batch_is_a_composite_switch_for_co_batched_plans_only() {
        let tree = ResinConfig {
            tree_batch: true,
            ..ResinConfig::default()
        };
        assert!(tree.co_batches(2));
        assert!(!tree.co_batches(1), "solo rows keep the per-row path");
        let batch = ResinConfig {
            multi_sequence_batch: true,
            ..ResinConfig::default()
        };
        assert!(batch.co_batches(2));
        assert!(!ResinConfig::default().co_batches(2));
    }

    #[test]
    fn header_names_switch_state_and_measurement_basis() {
        let off = experiment_header(&ResinConfig::default());
        assert!(off.contains("multi_sequence_batch=off"), "{off}");
        assert!(off.contains("tree_batch=off"), "{off}");
        assert!(off.contains("conversation_slots=off"), "{off}");
        assert!(off.contains("host wall"), "{off}");
        assert!(off.contains("per-row decode cost"), "{off}");
        assert!(off.contains("checkpoint hits/misses"), "{off}");
        assert!(off.contains("no device timestamps"), "{off}");
        assert!(off.contains("composite tree+batch mode"), "{off}");
        assert!(off.contains("<=4 rows per sequence"), "{off}");

        let on = experiment_header(&ResinConfig {
            multi_sequence_batch: true,
            tree_batch: true,
            conversation_slots: true,
        });
        assert!(on.contains("multi_sequence_batch=on"), "{on}");
        assert!(on.contains("tree_batch=on"), "{on}");
        assert!(on.contains("conversation_slots=on"), "{on}");
    }
}
