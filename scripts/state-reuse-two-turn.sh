#!/usr/bin/env bash
# Two-turn conversation gate (#76, protocol #66): the second prompt is the
# rendered first-turn history plus a new question. PASS requires the suffix +
# question prefill, token-identical output against a full re-prefill, and a
# second-turn TTFT at most half the full re-prefill's.
#
# Requires the local 1B checkpoint; skips (exit 0) when it is absent. Never run
# this concurrently with benchmarks: it loads the model on the CPU.
#
#   scripts/state-reuse-two-turn.sh
#   MODEL_DIR=... scripts/state-reuse-two-turn.sh --tokens 16
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${MODEL_DIR:-$root/models/MiniCPM-V-4.7-1B}"

if [[ ! -f "$model/config.json" ]]; then
    echo "state-reuse: gate=skip reason=checkpoint-missing model=$model"
    exit 0
fi

exec cargo run --quiet --release \
    --manifest-path "$root/crates/resin-metal/Cargo.toml" \
    --no-default-features --features cpu \
    --example state_reuse_ab -- two-turn --model "$model" "$@"
