#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
python -m src.verifier_benchmark --n "${N_CASES:-1000}" --seed "${SEED:-0}" --output "${OUTPUT:-outputs/validation/verifier_benchmark.json}"
