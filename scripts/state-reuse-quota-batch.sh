#!/usr/bin/env bash
# Quota-constrained two-request co-batch gate (#76, protocol #66): both arms
# run the #61 load (A decodes, B joins with the shared prefix). PASS requires
# the constrained arm to complete, keep >= 0.9x the unconstrained throughput,
# evict the idle prefix under the quota, and remain able to reuse the evicted
# prefix (memory or SSD reload).
#
# Requires the local 1B checkpoint; skips (exit 0) when it is absent. The
# default quota is sized from the unconstrained arm's observed live/checkpoint
# peaks; --budget-mb pins one. Never run this concurrently with benchmarks.
#
#   scripts/state-reuse-quota-batch.sh
#   MODEL_DIR=... RESIN_STATE_CACHE_DIR=... scripts/state-reuse-quota-batch.sh \
#     --prefix 128 --suffix 16 --tokens 32 --budget-mb 8
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
    --example state_reuse_ab -- quota-batch --model "$model" "$@"
