#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
python -m src.external_validity \
  --config "${1:-configs/external_validity_template.yaml}" \
  --output "${2:-outputs/external_validity/results.json}"
