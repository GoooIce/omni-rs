//! Engine-side experiment switches for the resin runners (#72, #74, #75,
//! #76).
//!
//! One mechanism, one switch: `multi_sequence_batch` lets every decode row of
//! one plan share a single multi-sequence verify forward;
//! `conversation_slots` lets a request resume a stored conversation
//! checkpoint instead of prefilling the whole prompt; `state_cache` persists
//! those checkpoints to SSD so a later process can restore them (#75);
//! `kv_budget_mb` caps live slot KV plus in-memory checkpoints and turns on
//! the #65 release order (#76). All are default off; the config field is the
//! durable switch and `RESIN_MULTI_SEQUENCE_BATCH=1` /
//! `RESIN_CONVERSATION_SLOTS=1` / `RESIN_STATE_CACHE=1` /
//! `RESIN_KV_BUDGET_MB=<mib>` are the same-binary experimental overrides the
//! A/B protocol asks for (docs/agent-workflow.md, "Performance measurement").
//! The composite tree+batch mode failed #67's gate and was removed (#73); the
//! model-side single-stream tree mode is unchanged and independent. The
//! header names the switches, the KV release order and the measurement basis
//! once per process when a switch is enabled or `OMNI_AB_HEADER=1` is set.

use std::sync::Once;

/// Experimental engine configuration. Every field defaults off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResinConfig {
    /// Verify all decode rows of a plan (`>= 2`) in one target forward
    /// instead of one forward per row. Greedy DSpark only for now.
    pub multi_sequence_batch: bool,
    /// Store conversation checkpoints for requests tagged with a slot key and
    /// resume the deepest prefix-matching one (#74). No slot key means the
    /// unchanged full prefill.
    pub conversation_slots: bool,
    /// Persist conversation checkpoints through
    /// [`crate::state_cache::StateCache`] (#75). Requires
    /// `conversation_slots` and a cache built with a caller-supplied
    /// namespace; without one the session stays memory-only. Off means no
    /// cache reads or writes.
    pub state_cache: bool,
    /// KV memory budget in MiB (#76). `None` keeps the unbounded pre-#76
    /// behavior: no accounting-driven eviction and no checkpoint spills.
    /// When set, the session enforces the #65 release order after checkpoint
    /// insertion and slot creation (see [`crate::session`]); live slots are
    /// never evicted, so a plan whose slots alone exceed the budget still
    /// runs and reports its honest over-budget remainder.
    pub kv_budget_mb: Option<usize>,
}

impl ResinConfig {
    /// Defaults plus the experimental environment overrides.
    pub fn from_env() -> Self {
        Self::default().with_env_overrides()
    }

    /// Fold `RESIN_MULTI_SEQUENCE_BATCH=1`, `RESIN_CONVERSATION_SLOTS=1`,
    /// `RESIN_STATE_CACHE=1` and `RESIN_KV_BUDGET_MB=<mib>` into this config.
    /// The environment is an experimental override: when set it wins over an
    /// explicit `false`, so one binary can A/B a switch (same-binary kill
    /// switch).
    pub fn with_env_overrides(mut self) -> Self {
        self.multi_sequence_batch |= env_flag("RESIN_MULTI_SEQUENCE_BATCH");
        self.conversation_slots |= env_flag("RESIN_CONVERSATION_SLOTS");
        self.state_cache |= env_flag("RESIN_STATE_CACHE");
        if let Some(mb) = std::env::var("RESIN_KV_BUDGET_MB")
            .ok()
            .and_then(|value| parse_budget_mb(&value))
        {
            self.kv_budget_mb = Some(mb);
        }
        self
    }

    /// Whether a decode plan with `rows` rows takes the co-batched path.
    /// Solo rows always keep the per-row path.
    pub fn co_batches(&self, rows: usize) -> bool {
        self.multi_sequence_batch && rows >= 2
    }

    /// The configured budget in bytes (`MiB` is the switch's unit).
    pub fn kv_budget_bytes(&self) -> Option<u64> {
        self.kv_budget_mb.map(|mb| {
            u64::try_from(mb)
                .unwrap_or(u64::MAX)
                .saturating_mul(1 << 20)
        })
    }
}

/// `RESIN_KV_BUDGET_MB` value: a positive mebibyte count; `0`, garbage and an
/// unset variable all mean "no budget".
fn parse_budget_mb(value: &str) -> Option<usize> {
    value.trim().parse::<usize>().ok().filter(|&mb| mb > 0)
}

/// One header line naming every switch, the KV release order and the
/// measurement basis.
pub fn experiment_header(config: &ResinConfig) -> String {
    format!(
        "resin experiment: multi_sequence_batch={} conversation_slots={} \
         state_cache={} kv_budget_mb={} | kv release order (#65): \
         self-drop of the missing key's checkpoints, then idle-slot checkpoints LRU \
         (spill to SSD first when state_cache is on, otherwise drop), then image embedding \
         cache=none (this runner holds none), pending-prefill KV=none (single-chunk prefill, \
         never kept across requests), ANE split=out of scope; live slots are never evicted \
         (over-budget remainder is reported); basis: host wall clock at the runner seam \
         (prepare/forward/complete); per-row decode cost = forward time / plan rows; prefill \
         tokens and checkpoint hits/misses come from the session counters; SSD state-cache \
         writes drain synchronously on the runner thread at the idle boundary (after release / \
         before the next forward); no device timestamps",
        on_off(config.multi_sequence_batch),
        on_off(config.conversation_slots),
        on_off(config.state_cache),
        match config.kv_budget_mb {
            None => "none".to_string(),
            Some(mb) => mb.to_string(),
        }
    )
}

/// Print [`experiment_header`] once per process when a switch or the KV
/// budget is enabled or `OMNI_AB_HEADER=1`; a no-op otherwise.
pub fn print_header(config: &ResinConfig) {
    if !config.multi_sequence_batch
        && !config.conversation_slots
        && !config.state_cache
        && config.kv_budget_mb.is_none()
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
    use super::{ResinConfig, experiment_header, parse_budget_mb};

    #[test]
    fn config_defaults_every_switch_off() {
        assert!(!ResinConfig::default().multi_sequence_batch);
        assert!(!ResinConfig::default().conversation_slots);
        assert!(!ResinConfig::default().state_cache);
        assert_eq!(ResinConfig::default().kv_budget_mb, None);
    }

    #[test]
    fn kv_budget_is_optional_and_zero_means_off() {
        assert_eq!(parse_budget_mb("128"), Some(128));
        assert_eq!(parse_budget_mb(" 64 "), Some(64));
        assert_eq!(parse_budget_mb("0"), None, "0 is not a usable budget");
        assert_eq!(parse_budget_mb("many"), None);
        let budgeted = ResinConfig {
            kv_budget_mb: Some(3),
            ..ResinConfig::default()
        };
        assert_eq!(budgeted.kv_budget_bytes(), Some(3 << 20));
        assert_eq!(ResinConfig::default().kv_budget_bytes(), None);
    }

    #[test]
    fn co_batches_only_with_multi_sequence_batch() {
        let batch = ResinConfig {
            multi_sequence_batch: true,
            ..ResinConfig::default()
        };
        assert!(batch.co_batches(2));
        assert!(!batch.co_batches(1), "solo rows keep the per-row path");
        assert!(!ResinConfig::default().co_batches(2));
    }

    #[test]
    fn header_names_switch_state_and_measurement_basis() {
        let off = experiment_header(&ResinConfig::default());
        assert!(off.contains("multi_sequence_batch=off"), "{off}");
        assert!(!off.contains("tree_batch"), "{off}");
        assert!(off.contains("conversation_slots=off"), "{off}");
        assert!(off.contains("state_cache=off"), "{off}");
        assert!(off.contains("host wall"), "{off}");
        assert!(off.contains("per-row decode cost"), "{off}");
        assert!(off.contains("checkpoint hits/misses"), "{off}");
        assert!(off.contains("idle boundary"), "{off}");
        assert!(off.contains("no device timestamps"), "{off}");
        assert!(off.contains("kv_budget_mb=none"), "{off}");
        assert!(off.contains("self-drop"), "{off}");
        assert!(off.contains("spill to SSD first"), "{off}");
        assert!(off.contains("image embedding cache=none"), "{off}");
        assert!(off.contains("pending-prefill KV=none"), "{off}");
        assert!(off.contains("ANE split=out of scope"), "{off}");

        let on = experiment_header(&ResinConfig {
            multi_sequence_batch: true,
            conversation_slots: true,
            state_cache: true,
            ..ResinConfig::default()
        });
        assert!(on.contains("multi_sequence_batch=on"), "{on}");
        assert!(!on.contains("tree_batch"), "{on}");
        assert!(on.contains("conversation_slots=on"), "{on}");
        assert!(on.contains("state_cache=on"), "{on}");

        let budgeted = experiment_header(&ResinConfig {
            kv_budget_mb: Some(12),
            ..ResinConfig::default()
        });
        assert!(budgeted.contains("kv_budget_mb=12"), "{budgeted}");
    }
}
