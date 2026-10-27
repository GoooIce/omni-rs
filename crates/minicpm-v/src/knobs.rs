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
//! | `MINICPM_GLUE` | on | fused verify seams (`=0` disables) |
//! | `MINICPM_DRAFT_GLUE` | on | fused draft seams (`=0` disables) |

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
