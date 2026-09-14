#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
args=(
  --seeds "${SEEDS:-5}"
  --output "${OUTPUT:-outputs/validation/synthetic_benchmark.csv}"
)
if [[ -n "${CONFIG:-}" ]]; then
  args+=(--config "$CONFIG")
fi
if [[ -n "${MAX_RUNS:-}" ]]; then
  args+=(--max-runs "$MAX_RUNS")
fi
python -m src.synthetic_benchmark "${args[@]}"
