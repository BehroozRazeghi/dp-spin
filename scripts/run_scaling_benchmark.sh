#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
python -m src.scaling_benchmark \
  --config "${1:-configs/scaling_benchmark.yaml}" \
  --output "${2:-outputs/scaling/scaling_results.json}"
