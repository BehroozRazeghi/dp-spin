#!/usr/bin/env bash
set -euo pipefail

# Run the OpenAI verbalization and release-conditioned evaluation workflow on
# completed DP-SPIN runs. The script selects saved releases, generates DP-SPIN
# and baseline summaries, runs the configured judges, and writes CSV, LaTeX,
# Markdown, and JSON reports. It does not rerun a DP-SPIN release mechanism.
#
# Required unless REPORT_ONLY=1 or OPENAI_REQUIRED=0:
#   export OPENAI_API_KEY="..."
#
# Usage:
#   bash scripts/run_openai_paper.sh
#   DATASET=amazon bash scripts/run_openai_paper.sh
#   DATASET=yelp EPS="1" bash scripts/run_openai_paper.sh
#   DATASET=cfpb OPENAI_PROFILE=grid bash scripts/run_openai_paper.sh
#   REPORT_ONLY=1 bash scripts/run_openai_paper.sh
#
# Profiles:
#   representative  selected manuscript conditions (default)
#   grid            expanded epsilon selection

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

OPENAI_PROFILE="${OPENAI_PROFILE:-representative}"  # representative or grid
DATASET="${DATASET:-all}"                            # all, cfpb, amazon, yelp
OUTPUT_DIR="${OUTPUT_DIR:-outputs/openai_paper}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
OPENAI_TEMPERATURE="${OPENAI_TEMPERATURE:-0.0}"
OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-700}"
K_VALUE="${K_VALUE:-3}"
L_VALUE="${L_VALUE:-10}"

case "$OPENAI_PROFILE" in
  representative)
    MECHS="${MECHS:-gaussian private_topL}"
    EPS="${EPS:-1 4}"
    SEEDS="${SEEDS:-0 1 2}"
    ;;
  grid)
    MECHS="${MECHS:-gaussian private_topL}"
    EPS="${EPS:-0.5 1 2 4}"
    SEEDS="${SEEDS:-0 1 2}"
    ;;
  *)
    echo "ERROR: OPENAI_PROFILE must be 'representative' or 'grid'." >&2
    exit 1
    ;;
esac

# Empty filters use the dataset-specific defaults in src.openai_paper_pipeline.
UNITS="${UNITS:-}"
M_VALUES="${M_VALUES:-}"
USER_BOUNDS="${USER_BOUNDS:-}"
MAX_DPSPIN_RUNS_PER_DATASET="${MAX_DPSPIN_RUNS_PER_DATASET:-}"
N_RECORDS_VALUES="${N_RECORDS_VALUES:-}"
N_USERS_VALUES="${N_USERS_VALUES:-}"

# The default comparison uses only the configured DP baselines. Add a non-private
# reference only after checking its input boundary and dataset-use constraints.
BASELINE_SUBSET_SIZE="${BASELINE_SUBSET_SIZE:-}"
BASELINE_EPSILON="${BASELINE_EPSILON:-1.0}"
BASELINE_SEED="${BASELINE_SEED:-0}"
BASELINE_SEEDS="${BASELINE_SEEDS:-$SEEDS}"
BASELINE_METHODS="${BASELINE_METHODS:-dp_keyword dp_category}"
RUN_URANIA="${RUN_URANIA:-1}"
URANIA_EPS_TOTAL="${URANIA_EPS_TOTAL:-1.0}"
URANIA_DELTA="${URANIA_DELTA:-1e-6}"
URANIA_CLUSTERS="${URANIA_CLUSTERS:-10}"
URANIA_KEYWORDS="${URANIA_KEYWORDS:-10}"

# Method lists are pipe-delimited because method names contain spaces.
COMPARISON_METHODS="${COMPARISON_METHODS:-DP-SPIN private plan|DP keyword histogram|DP category histogram|URANIA-style public keywords}"
COMPARISON_DPSPIN_MECHANISM="${COMPARISON_DPSPIN_MECHANISM:-gaussian}"
COMPARISON_DPSPIN_EPSILON="${COMPARISON_DPSPIN_EPSILON:-1.0}"
QUALITATIVE_METHODS="${QUALITATIVE_METHODS:-$COMPARISON_METHODS}"
JUDGE_METHODS="${JUDGE_METHODS:-}"

DPSPIN_JUDGE_BACKEND="${DPSPIN_JUDGE_BACKEND:-none}"      # Compatibility judge for DP-SPIN summaries only.
GENERIC_JUDGE_BACKEND="${GENERIC_JUDGE_BACKEND:-openai}"    # none, heuristic, or openai
COMPARATIVE_BACKEND="${COMPARATIVE_BACKEND:-none}"          # none, heuristic, or openai
SUMMARY_EMBEDDING_BACKEND="${SUMMARY_EMBEDDING_BACKEND:-none}"
QUALITATIVE_MAX_CHARS="${QUALITATIVE_MAX_CHARS:-0}"
RUN_DPSPIN="${RUN_DPSPIN:-1}"
RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_JUDGE="${RUN_JUDGE:-1}"
REPORT_ONLY="${REPORT_ONLY:-0}"
FORCE="${FORCE:-0}"
OPENAI_REQUIRED="${OPENAI_REQUIRED:-1}"

# Rebuild compact cross-dataset tables after the dataset workflow. The table step
# reads saved reports only and makes no OpenAI call.
BUILD_MAIN_TABLES="${BUILD_MAIN_TABLES:-1}"
MAIN_TABLE_UNIT="${MAIN_TABLE_UNIT:-record}"
MAIN_TABLE_MAX_CHARS="${MAIN_TABLE_MAX_CHARS:-650}"
MAIN_TABLE_OUTPUT_DIR="${MAIN_TABLE_OUTPUT_DIR:-$OUTPUT_DIR/main_tables}"

export DATASET OUTPUT_DIR OPENAI_MODEL OPENAI_TEMPERATURE OPENAI_MAX_TOKENS
export MECHS EPS SEEDS UNITS M_VALUES USER_BOUNDS N_RECORDS_VALUES N_USERS_VALUES K_VALUE L_VALUE MAX_DPSPIN_RUNS_PER_DATASET
export BASELINE_SUBSET_SIZE BASELINE_EPSILON BASELINE_SEED BASELINE_SEEDS BASELINE_METHODS RUN_URANIA
export URANIA_EPS_TOTAL URANIA_DELTA URANIA_CLUSTERS URANIA_KEYWORDS
export COMPARISON_METHODS COMPARISON_DPSPIN_MECHANISM COMPARISON_DPSPIN_EPSILON QUALITATIVE_METHODS JUDGE_METHODS
export DPSPIN_JUDGE_BACKEND GENERIC_JUDGE_BACKEND COMPARATIVE_BACKEND SUMMARY_EMBEDDING_BACKEND
export QUALITATIVE_MAX_CHARS RUN_DPSPIN RUN_BASELINES RUN_JUDGE REPORT_ONLY FORCE OPENAI_REQUIRED
export BUILD_MAIN_TABLES MAIN_TABLE_UNIT MAIN_TABLE_MAX_CHARS MAIN_TABLE_OUTPUT_DIR

bash scripts/_run_openai_paper_driver.sh

if [[ "$BUILD_MAIN_TABLES" == "1" ]]; then
  python -m src.openai_main_tables \
    --reports-root "$OUTPUT_DIR" \
    --output-dir "$MAIN_TABLE_OUTPUT_DIR" \
    --unit "$MAIN_TABLE_UNIT" \
    --qualitative-max-chars "$MAIN_TABLE_MAX_CHARS"
fi
