#!/usr/bin/env bash
# Two-process cold-start gate for the resin SSD state cache (#75, protocol #66).
#
# Process 1 (`cold-write`) runs a fixed 272-token prompt on the 1B checkpoint
# and exits after the cache reports its writes. Process 2 (`cold-read`) is a
# fresh process: it runs the same prompt once without a slot key (full-prefill
# baseline) and once with a slot key, restoring the SSD entry. The gate passes
# when the restored request prefills no prompt token, its TTFT is at most half
# the full prefill's, and both first tokens agree.
#
# Requires the local 1B checkpoint; skips (exit 0) when it is absent. Never run
# this concurrently with benchmarks: it loads the model on the CPU.
#
#   scripts/state-cache-cold-start.sh
#   MODEL_DIR=... RESIN_STATE_CACHE_DIR=... scripts/state-cache-cold-start.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${MODEL_DIR:-$root/models/MiniCPM-V-4.7-1B}"

if [[ ! -f "$model/config.json" ]]; then
    echo "state-cache: gate=skip reason=checkpoint-missing model=$model"
    exit 0
fi

export RESIN_STATE_CACHE=1
export RESIN_STATE_CACHE_DIR="${RESIN_STATE_CACHE_DIR:-$root/artifacts/state/resin-state-cold-start}"
rm -rf "$RESIN_STATE_CACHE_DIR"

run() {
    cargo run --quiet --release \
        --manifest-path "$root/crates/resin-metal/Cargo.toml" \
        --no-default-features --features cpu \
        --example state_cache_cold -- "$@"
}

write_out="$(run cold-write --model "$model")"
printf '%s\n' "$write_out"
read_out="$(run cold-read --model "$model")"
printf '%s\n' "$read_out"

if grep -q 'state-cache: gate=pass' <<<"$read_out"; then
    echo "state-cache: two-process cold-start gate passed"
else
    echo "state-cache: two-process cold-start gate failed" >&2
    exit 1
fi
