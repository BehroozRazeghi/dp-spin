#!/usr/bin/env bash
set -euo pipefail

# Verbalize selected saved DP-SPIN plans with OpenAI.
# Optional plan-conditioned judging and text diagnostics can be enabled. Histogram
# baselines, the URANIA-style baseline, and cross-method manuscript tables are
# excluded; use run_openai_paper.sh for the full comparison workflow.
#
# Required:
#   export OPENAI_API_KEY="..."
#
# Usage:
#   bash scripts/run_openai_dpspin_only.sh
#   DATASET=amazon UNITS="record user" EPS="1 4" bash scripts/run_openai_dpspin_only.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  echo 'Run: export OPENAI_API_KEY="..."' >&2
  exit 1
fi

DATASET="${DATASET:-all}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
OPENAI_TEMPERATURE="${OPENAI_TEMPERATURE:-0.0}"
OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-700}"
MECHS="${MECHS:-gaussian private_topL}"
EPS="${EPS:-1 4}"
SEEDS="${SEEDS:-0}"
UNITS="${UNITS:-}"
M_VALUES="${M_VALUES:-}"
USER_BOUNDS="${USER_BOUNDS:-}"
K_VALUE="${K_VALUE:-3}"
L_VALUE="${L_VALUE:-10}"
JUDGE_BACKEND="${JUDGE_BACKEND:-none}"  # Optional plan-conditioned judge retained for compatibility.
COMPARATIVE_BACKEND="${COMPARATIVE_BACKEND:-none}"
SUMMARY_EMBEDDING_BACKEND="${SUMMARY_EMBEDDING_BACKEND:-tfidf}"
FORCE_REDECODE="${FORCE_REDECODE:-0}"
MAX_RUNS_PER_DATASET="${MAX_RUNS_PER_DATASET:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

args=(
  --dataset "$DATASET"
  --model "$OPENAI_MODEL"
  --temperature "$OPENAI_TEMPERATURE"
  --max-tokens "$OPENAI_MAX_TOKENS"
  --mechanisms $MECHS
  --epsilons $EPS
  --seeds $SEEDS
  --K "$K_VALUE"
  --L "$L_VALUE"
  --judge-backend "$JUDGE_BACKEND"
  --comparative-backend "$COMPARATIVE_BACKEND"
  --summary-embedding-backend "$SUMMARY_EMBEDDING_BACKEND"
)
if [[ -n "$UNITS" ]]; then args+=(--units $UNITS); fi
if [[ -n "$M_VALUES" ]]; then args+=(--m-values $M_VALUES); fi
if [[ -n "$USER_BOUNDS" ]]; then args+=(--user-contribution-bounds $USER_BOUNDS); fi
if [[ -n "$MAX_RUNS_PER_DATASET" ]]; then args+=(--max-runs-per-dataset "$MAX_RUNS_PER_DATASET"); fi
if [[ -n "$OUTPUT_DIR" ]]; then args+=(--output-dir "$OUTPUT_DIR"); fi
if [[ "$FORCE_REDECODE" == "1" ]]; then args+=(--force); fi

echo "Project root: $PROJECT_ROOT"
echo "Dataset: $DATASET"
echo "OpenAI model: $OPENAI_MODEL"
echo "Mechanisms: $MECHS"
echo "Epsilons: $EPS"
echo "Seeds: $SEEDS"
echo "K, L: $K_VALUE, $L_VALUE"
echo "Judge backend: $JUDGE_BACKEND"

python -m src.openai_decode_representatives "${args[@]}"
