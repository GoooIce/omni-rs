//! Every environment kill switch in one place, each read once per process.
//!
//! These switches are the A/B measurement discipline's load-bearing wall
//! (docs/agent-workflow.md, "Performance measurement"): same-binary
//! comparisons depend on them being cached and consistent, so all readers go
//! through here instead of touching `std::env` directly.
//!
//! Knob table (default · scope · purpose):
//!
//! | knob | default | purpose |
//! | --- | --- | --- |
//! | `MINICPM_DSPARK_TRACE` | off | phase + propose-split timings on stderr |
//! | `MINICPM_DSPARK_BLOCK` | adaptive | pin a fixed draft block, `1..=31` |
//! | `MINICPM_DSPARK_LAYERS` | `[3,11,19,27,35]` | draft cross-attention layers, five ascending `0..40` |
//! | `MINICPM_DSPARK_TREE` | off | markov-tree layout (adaptive runs only) |
//! | `MINICPM_DSPARK_MARKOV_QUANT` | off | int8 markov scoring |
//! | `MINICPM_DSPARK_LEDGER` | off | calibrated verify-size ledger (`=1`) |
//! | `MINICPM_DSPARK_LEDGER_PATH` | `target/dspark-ledger/` | ledger JSON path override |
//! | `MINICPM_DSPARK_CHIP` | arch/OS | ledger chip fingerprint override |
//! | `MINICPM_GLUE` | on | fused verify seams (`=0` disables) |
//! | `MINICPM_DRAFT_GLUE` | on | fused draft seams (`=0` disables) |
//! | `OMNI_AB_HEADER` | off | print the one-line A/B header (`=1`) |

/// The draft's cross-attention layers over the 40-layer target tower.
pub(crate) const DEFAULT_LAYERS: [usize; 5] = [3, 11, 19, 27, 35];

/// `MINICPM_DSPARK_TRACE`
pub fn dspark_trace() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| std::env::var_os("MINICPM_DSPARK_TRACE").is_some())
}

/// `MINICPM_DSPARK_BLOCK`
pub(crate) fn dspark_block() -> Option<usize> {
    static VALUE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("MINICPM_DSPARK_BLOCK")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|size| (1..=31).contains(size))
    })
}

/// `MINICPM_DSPARK_LAYERS`
pub fn dspark_layers() -> &'static [usize; 5] {
    static VALUE: std::sync::OnceLock<[usize; 5]> = std::sync::OnceLock::new();
    VALUE.get_or_init(|| {
        let parsed: Option<Vec<usize>> =
            std::env::var("MINICPM_DSPARK_LAYERS")
                .ok()
                .and_then(|value| {
                    let parts: Vec<usize> = value
                        .split(',')
                        .map(|part| part.trim().parse().ok())
                        .collect::<Option<Vec<_>>>()?;
                    (parts.len() == 5
                        && parts.windows(2).all(|pair| pair[0] < pair[1])
                        && parts.first().is_some_and(|&first| first < 40)
                        && parts.last().is_some_and(|&last| last < 40))
                    .then_some(parts)
                });
        parsed.map_or(DEFAULT_LAYERS, |parts| {
            let mut layers = [0usize; 5];
            layers.copy_from_slice(&parts);
            layers
        })
    })
}

/// `MINICPM_DSPARK_TREE`
pub(crate) fn dspark_tree() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        matches!(
            std::env::var("MINICPM_DSPARK_TREE").ok().as_deref(),
            Some("1")
        )
    })
}

/// `MINICPM_DSPARK_MARKOV_QUANT`
pub(crate) fn dspark_markov_quant() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        matches!(
            std::env::var("MINICPM_DSPARK_MARKOV_QUANT").ok().as_deref(),
            Some("1")
        )
    })
}

/// `MINICPM_DSPARK_LEDGER` — the calibrated verify-size ledger. Default off:
/// with the switch off, `dspark::generate` runs the unchanged AIMD ladder and
/// reads or writes no ledger file.
pub fn dspark_ledger() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        matches!(
            std::env::var("MINICPM_DSPARK_LEDGER").ok().as_deref(),
            Some("1")
        )
    })
}

/// `OMNI_AB_HEADER` — the A/B run header without the per-cycle trace.
fn ab_header() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| matches!(std::env::var("OMNI_AB_HEADER").ok().as_deref(), Some("1")))
}

/// The A/B run header: every DSpark switch, the ledger's allowed set and
/// cost basis, the fused-seam switches and the build's git hash when the
/// build provided one. Printed once per process when `MINICPM_DSPARK_TRACE`
/// or `OMNI_AB_HEADER=1` is set; callers invoke it at a run's entry.
pub fn experiment_header() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    if !(dspark_trace() || ab_header()) {
        return;
    }
    ONCE.call_once(|| eprintln!("{}", header_line()));
}

fn header_line() -> String {
    let block = dspark_block().map_or_else(|| "adaptive".to_owned(), |size| size.to_string());
    let allowed = dspark_block()
        .map(|size| vec![size])
        .unwrap_or_else(|| crate::ledger::ALLOWED_SIZES.to_vec());
    let layers = dspark_layers()
        .iter()
        .map(|layer| layer.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "dspark ab-header: trace={} block={} layers={} tree={} markov_quant={} ledger={} ledger_allowed={:?} ledger_basis={} glue={} draft_glue={} git={}",
        dspark_trace(),
        block,
        layers,
        dspark_tree(),
        dspark_markov_quant(),
        dspark_ledger(),
        allowed,
        crate::ledger::COST_BASIS,
        verify_glue(),
        draft_glue(),
        option_env!("OMNI_GIT_HASH").unwrap_or("unknown"),
    )
}

/// `MINICPM_GLUE` — the fused verify seams (norm, residual, split-half RoPE)
/// default ON: the verify forward is launch-bound and the seams removed ~20%
/// of its wall in same-window interleaved A/B (#52), with greedy output
/// byte-identical on both harness prompts. `=0` is the same-binary kill
/// switch; lib unit tests exercise the kernels regardless.
pub(crate) fn verify_glue() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        cfg!(test) || !matches!(std::env::var("MINICPM_GLUE").ok().as_deref(), Some("0"))
    })
}

/// `MINICPM_DRAFT_GLUE` — the draft seams default ON (the draft forward is
/// CPU-dispatch-bound); `=0` is the same-binary kill switch.
pub(crate) fn draft_glue() -> bool {
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        !matches!(
            std::env::var("MINICPM_DRAFT_GLUE").ok().as_deref(),
            Some("0")
        )
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn header_lists_switches_allowed_set_and_basis() {
        let line = super::header_line();
        for field in [
            "trace=",
            "block=",
            "layers=",
            "tree=",
            "markov_quant=",
            "ledger=",
            "ledger_allowed=",
            "ledger_basis=host-wall-clock",
            "glue=",
            "draft_glue=",
            "git=",
        ] {
            assert!(line.contains(field), "missing {field} in {line}");
        }
    }
}
