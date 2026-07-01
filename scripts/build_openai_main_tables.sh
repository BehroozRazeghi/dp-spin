#!/usr/bin/env bash
set -euo pipefail

# Build the compact cross-dataset OpenAI tables from saved reports.
# The command is report-only: it neither calls OpenAI nor reruns a DP mechanism.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

REPORTS_ROOT="${REPORTS_ROOT:-outputs/openai_paper}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPORTS_ROOT/main_tables}"
UNIT="${UNIT:-record}"
QUALITATIVE_MAX_CHARS="${QUALITATIVE_MAX_CHARS:-650}"
DATASETS="${DATASETS:-}"

args=(
  --reports-root "$REPORTS_ROOT"
  --output-dir "$OUTPUT_DIR"
  --unit "$UNIT"
  --qualitative-max-chars "$QUALITATIVE_MAX_CHARS"
)
if [[ -n "$DATASETS" ]]; then
  args+=(--datasets $DATASETS)
fi

python -m src.openai_main_tables "${args[@]}"
