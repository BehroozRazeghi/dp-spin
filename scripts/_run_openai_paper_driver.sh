#!/usr/bin/env bash
set -euo pipefail

# Internal argument-mapping driver for run_openai_paper.sh.
# Invoke scripts/run_openai_paper.sh rather than calling this file directly.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATASET="${DATASET:-all}"                       # all, cfpb, amazon, yelp
case "$DATASET" in
  cfpb) DEFAULT_REPORT_SCOPE="cfpb" ;;
  amazon) DEFAULT_REPORT_SCOPE="amazon_All_Beauty" ;;
  yelp) DEFAULT_REPORT_SCOPE="yelp_restaurants" ;;
  all) DEFAULT_REPORT_SCOPE="all_datasets" ;;
  *) DEFAULT_REPORT_SCOPE="$DATASET" ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-outputs/openai_paper}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
OPENAI_TEMPERATURE="${OPENAI_TEMPERATURE:-0.0}"
OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-700}"
MECHS="${MECHS:-gaussian private_topL}"
EPS="${EPS:-1 4}"
SEEDS="${SEEDS:-0 1 2}"
UNITS="${UNITS:-}"
M_VALUES="${M_VALUES:-}"
USER_BOUNDS="${USER_BOUNDS:-}"
N_RECORDS_VALUES="${N_RECORDS_VALUES:-}"
N_USERS_VALUES="${N_USERS_VALUES:-}"
K_VALUE="${K_VALUE:-3}"
L_VALUE="${L_VALUE:-10}"
MAX_DPSPIN_RUNS_PER_DATASET="${MAX_DPSPIN_RUNS_PER_DATASET:-}"
BASELINE_SUBSET_SIZE="${BASELINE_SUBSET_SIZE:-}"
BASELINE_EPSILON="${BASELINE_EPSILON:-1.0}"
BASELINE_SEED="${BASELINE_SEED:-0}"
BASELINE_SEEDS="${BASELINE_SEEDS:-$BASELINE_SEED}"
BASELINE_METHODS="${BASELINE_METHODS:-dp_keyword dp_category}"
COMPARISON_METHODS="${COMPARISON_METHODS:-DP-SPIN private plan|DP keyword histogram|DP category histogram|URANIA-style public keywords}"
COMPARISON_DPSPIN_MECHANISM="${COMPARISON_DPSPIN_MECHANISM:-gaussian}"
COMPARISON_DPSPIN_EPSILON="${COMPARISON_DPSPIN_EPSILON:-1.0}"
QUALITATIVE_METHODS="${QUALITATIVE_METHODS:-$COMPARISON_METHODS}"
JUDGE_METHODS="${JUDGE_METHODS:-}"
URANIA_EPS_TOTAL="${URANIA_EPS_TOTAL:-1.0}"
URANIA_DELTA="${URANIA_DELTA:-1e-6}"
URANIA_CLUSTERS="${URANIA_CLUSTERS:-10}"
URANIA_KEYWORDS="${URANIA_KEYWORDS:-10}"
RUN_URANIA="${RUN_URANIA:-1}"
DPSPIN_JUDGE_BACKEND="${DPSPIN_JUDGE_BACKEND:-none}"
COMPARATIVE_BACKEND="${COMPARATIVE_BACKEND:-none}"
SUMMARY_EMBEDDING_BACKEND="${SUMMARY_EMBEDDING_BACKEND:-tfidf}"
GENERIC_JUDGE_BACKEND="${GENERIC_JUDGE_BACKEND:-openai}"
RUN_DPSPIN="${RUN_DPSPIN:-1}"
RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"
FORCE="${FORCE:-0}"
OPENAI_REQUIRED="${OPENAI_REQUIRED:-1}"
REPORT_ONLY="${REPORT_ONLY:-0}"
QUALITATIVE_MAX_CHARS="${QUALITATIVE_MAX_CHARS:-0}"

if [[ "$REPORT_ONLY" != "1" && "$OPENAI_REQUIRED" == "1" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  echo 'Run: export OPENAI_API_KEY="..."' >&2
  exit 1
fi

IFS='|' read -r -a COMPARISON_METHOD_ARRAY <<< "$COMPARISON_METHODS"
IFS='|' read -r -a QUALITATIVE_METHOD_ARRAY <<< "$QUALITATIVE_METHODS"
if [[ -n "$JUDGE_METHODS" ]]; then
  IFS='|' read -r -a JUDGE_METHOD_ARRAY <<< "$JUDGE_METHODS"
else
  JUDGE_METHOD_ARRAY=()
fi

args=(
  --dataset "$DATASET"
  --output-dir "$OUTPUT_DIR"
  --model "$OPENAI_MODEL"
  --temperature "$OPENAI_TEMPERATURE"
  --max-tokens "$OPENAI_MAX_TOKENS"
  --mechanisms $MECHS
  --epsilons $EPS
  --seeds $SEEDS
  --K "$K_VALUE"
  --L "$L_VALUE"
  --baseline-epsilon "$BASELINE_EPSILON"
  --baseline-seed "$BASELINE_SEED"
  --baseline-seeds $BASELINE_SEEDS
  --baseline-methods $BASELINE_METHODS
  --comparison-methods "${COMPARISON_METHOD_ARRAY[@]}"
  --comparison-dpspin-mechanism "$COMPARISON_DPSPIN_MECHANISM"
  --comparison-dpspin-epsilon "$COMPARISON_DPSPIN_EPSILON"
  --qualitative-methods "${QUALITATIVE_METHOD_ARRAY[@]}"
  --urania-epsilon-total "$URANIA_EPS_TOTAL"
  --urania-delta "$URANIA_DELTA"
  --urania-clusters "$URANIA_CLUSTERS"
  --urania-keywords "$URANIA_KEYWORDS"
  --dpspin-judge-backend "$DPSPIN_JUDGE_BACKEND"
  --comparative-backend "$COMPARATIVE_BACKEND"
  --summary-embedding-backend "$SUMMARY_EMBEDDING_BACKEND"
  --generic-judge-backend "$GENERIC_JUDGE_BACKEND"
  --qualitative-max-chars "$QUALITATIVE_MAX_CHARS"
)

if [[ -n "$UNITS" ]]; then args+=(--units $UNITS); fi
if [[ -n "$M_VALUES" ]]; then args+=(--m-values $M_VALUES); fi
if [[ -n "$USER_BOUNDS" ]]; then args+=(--user-contribution-bounds $USER_BOUNDS); fi
if [[ -n "$N_RECORDS_VALUES" ]]; then args+=(--n-records-values $N_RECORDS_VALUES); fi
if [[ -n "$N_USERS_VALUES" ]]; then args+=(--n-users-values $N_USERS_VALUES); fi
if [[ -n "$MAX_DPSPIN_RUNS_PER_DATASET" ]]; then args+=(--max-dpspin-runs-per-dataset "$MAX_DPSPIN_RUNS_PER_DATASET"); fi
if [[ -n "$BASELINE_SUBSET_SIZE" ]]; then args+=(--baseline-subset-size "$BASELINE_SUBSET_SIZE"); fi
if [[ "$RUN_DPSPIN" == "0" ]]; then args+=(--no-run-dpspin); fi
if [[ "$RUN_BASELINES" == "0" ]]; then args+=(--no-run-baselines); fi
if [[ "$RUN_JUDGE" == "0" ]]; then args+=(--no-run-judge); fi
if [[ "$RUN_URANIA" == "0" ]]; then args+=(--no-run-urania); fi
if [[ "${#JUDGE_METHOD_ARRAY[@]}" -gt 0 ]]; then args+=(--judge-methods "${JUDGE_METHOD_ARRAY[@]}"); fi
if [[ "$OPENAI_REQUIRED" == "0" ]]; then args+=(--no-openai-required); fi
if [[ "$FORCE" == "1" ]]; then args+=(--force); fi
if [[ "$REPORT_ONLY" == "1" ]]; then args+=(--report-only --no-openai-required); fi

if [[ "$(basename "$OUTPUT_DIR")" == "$DEFAULT_REPORT_SCOPE" ]]; then
  REPORT_DIR="$OUTPUT_DIR"
else
  REPORT_DIR="$OUTPUT_DIR/$DEFAULT_REPORT_SCOPE"
fi

echo "Project root: $PROJECT_ROOT"
echo "Dataset: $DATASET"
echo "Output root: $OUTPUT_DIR"
echo "Dataset-scoped report: $REPORT_DIR"
echo "OpenAI model: $OPENAI_MODEL"
echo "Mechanisms: $MECHS"
echo "Epsilons: $EPS"
echo "Seeds: $SEEDS"
echo "K, L: $K_VALUE, $L_VALUE"
echo "Run DP-SPIN decoding: $RUN_DPSPIN"
echo "Run baselines: $RUN_BASELINES"
echo "Baseline methods: $BASELINE_METHODS"
echo "Run URANIA-style baseline: $RUN_URANIA"
echo "Run judge: $RUN_JUDGE"
echo "Legacy DP-SPIN-only judge backend: $DPSPIN_JUDGE_BACKEND"
echo "Generic judge backend: $GENERIC_JUDGE_BACKEND"
echo "Comparison methods: $COMPARISON_METHODS"
echo "Comparison DP-SPIN condition: mechanism=$COMPARISON_DPSPIN_MECHANISM, epsilon=$COMPARISON_DPSPIN_EPSILON"
echo "Qualitative methods: $QUALITATIVE_METHODS"
echo "Report only: $REPORT_ONLY"
echo "Qualitative max chars: $QUALITATIVE_MAX_CHARS"

python -m src.openai_paper_pipeline "${args[@]}"

cat <<EOF

Done. Main outputs:
  $REPORT_DIR/tables/dpspin_openai_runs.csv
  $REPORT_DIR/tables/baseline_runs.csv
  $REPORT_DIR/judgments/judgments.csv
  $REPORT_DIR/latex/tab_openai_dpspin.tex
  $REPORT_DIR/latex/tab_openai_baseline_comparison.tex
  $REPORT_DIR/latex/tab_openai_qualitative_examples.tex
  $REPORT_DIR/latex/qualitative_examples.md
  $REPORT_DIR/README_OPENAI_PAPER_REPORT.md
EOF
