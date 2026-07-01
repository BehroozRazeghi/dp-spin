#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Compatibility wrapper for appending controlled-probe runs to existing Amazon
# or Yelp grids. The primary dataset runners already call src.review_probe.

RUN_AMAZON_PROBE="${RUN_AMAZON_PROBE:-1}"
RUN_YELP_PROBE="${RUN_YELP_PROBE:-1}"
RUN_RECORD="${RUN_RECORD:-1}"
RUN_USER="${RUN_USER:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

PROBE_SUPPORTS="${PROBE_SUPPORTS:-1 2 5 10 25 50 100 200}"
PROBE_EPS="${PROBE_EPS:-0.5 1 2 4}"
PROBE_SEEDS="${PROBE_SEEDS:-0 1 2}"
PROBE_MECHS="${PROBE_MECHS:-gaussian laplace private_topL}"
PROBE_K="${PROBE_K:-3}"
PROBE_L="${PROBE_L:-10}"
PROBE_TEXT="${PROBE_TEXT:-The review contains the synthetic probe phrase BLUE-LANTERN-791.}"

AMAZON_CATEGORY="${AMAZON_CATEGORY:-All_Beauty}"
AMAZON_PROBE_M="${AMAZON_PROBE_M:-46}"
AMAZON_PROBE_BASE_RECORDS="${AMAZON_PROBE_BASE_RECORDS:-2000}"
AMAZON_PROBE_BASE_USERS="${AMAZON_PROBE_BASE_USERS:-1000}"
AMAZON_PROBE_USER_B="${AMAZON_PROBE_USER_B:-2}"

YELP_CATEGORY_FILTER="${YELP_CATEGORY_FILTER:-Restaurants}"
YELP_SLUG="yelp_$(echo "$YELP_CATEGORY_FILTER" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
if [[ -z "$YELP_CATEGORY_FILTER" ]]; then YELP_SLUG="yelp_all"; fi
YELP_PROBE_M="${YELP_PROBE_M:-100}"
YELP_PROBE_BASE_RECORDS="${YELP_PROBE_BASE_RECORDS:-2000}"
YELP_PROBE_BASE_USERS="${YELP_PROBE_BASE_USERS:-1000}"
YELP_PROBE_USER_B="${YELP_PROBE_USER_B:-2}"
YELP_EMBEDDING_FIT_MAX_RECORDS="${YELP_EMBEDDING_FIT_MAX_RECORDS:-50000}"

unit_args=()
if [[ "$RUN_RECORD" == "1" ]]; then unit_args+=(--run-record); fi
if [[ "$RUN_USER" == "1" ]]; then unit_args+=(--run-user); fi
skip_args=()
if [[ "$SKIP_EXISTING" == "1" ]]; then skip_args+=(--skip-existing); fi

if [[ "$RUN_AMAZON_PROBE" == "1" ]]; then
  AMAZON_OUT="outputs/amazon_${AMAZON_CATEGORY}"
  python -m src.review_probe \
    --dataset-tag "amazon_${AMAZON_CATEGORY}" \
    --out "$AMAZON_OUT" \
    --target-records "data/protected_target/amazon_${AMAZON_CATEGORY}_target_records.jsonl" \
    --aux-records "data/auxiliary/amazon_${AMAZON_CATEGORY}_atom_auxiliary_records.jsonl" \
    --base-atoms "data/public_atoms/amazon_${AMAZON_CATEGORY}_atoms_m${AMAZON_PROBE_M}.json" \
    --probe-atoms "data/public_atoms/amazon_${AMAZON_CATEGORY}_atoms_probe_m${AMAZON_PROBE_M}.json" \
    --m "$AMAZON_PROBE_M" --K "$PROBE_K" --L "$PROBE_L" \
    --supports $PROBE_SUPPORTS --epsilons $PROBE_EPS --seeds $PROBE_SEEDS --mechanisms $PROBE_MECHS \
    --record-base-records "$AMAZON_PROBE_BASE_RECORDS" --user-base-users "$AMAZON_PROBE_BASE_USERS" --user-B "$AMAZON_PROBE_USER_B" \
    --probe-text "$PROBE_TEXT" --run-list "$AMAZON_OUT/probe_run_list.txt" \
    "${unit_args[@]}" "${skip_args[@]}"
  if [[ "$RUN_RECORD" == "1" ]]; then python -m src.summarize_runs --runs-dir "$AMAZON_OUT/record/runs" --output-dir "$AMAZON_OUT/summaries/record" --prefix record; fi
  if [[ "$RUN_USER" == "1" ]]; then python -m src.summarize_runs --runs-dir "$AMAZON_OUT/user/runs" --output-dir "$AMAZON_OUT/summaries/user" --prefix user; fi
  python -m src.report --dataset-name "amazon_${AMAZON_CATEGORY}" --out "$AMAZON_OUT" --record-summary "$AMAZON_OUT/summaries/record/record_summary.csv" --user-summary "$AMAZON_OUT/summaries/user/user_summary.csv"
fi

if [[ "$RUN_YELP_PROBE" == "1" ]]; then
  YELP_OUT="outputs/$YELP_SLUG"
  embed_args=()
  if [[ -n "$YELP_EMBEDDING_FIT_MAX_RECORDS" ]]; then embed_args+=(--embedding-fit-max-records "$YELP_EMBEDDING_FIT_MAX_RECORDS"); fi
  python -m src.review_probe \
    --dataset-tag "$YELP_SLUG" \
    --out "$YELP_OUT" \
    --target-records "data/protected_target/${YELP_SLUG}_target_records.jsonl" \
    --aux-records "data/auxiliary/${YELP_SLUG}_atom_auxiliary_records.jsonl" \
    --base-atoms "data/public_atoms/${YELP_SLUG}_atoms_m${YELP_PROBE_M}.json" \
    --probe-atoms "data/public_atoms/${YELP_SLUG}_atoms_probe_m${YELP_PROBE_M}.json" \
    --m "$YELP_PROBE_M" --K "$PROBE_K" --L "$PROBE_L" \
    --supports $PROBE_SUPPORTS --epsilons $PROBE_EPS --seeds $PROBE_SEEDS --mechanisms $PROBE_MECHS \
    --record-base-records "$YELP_PROBE_BASE_RECORDS" --user-base-users "$YELP_PROBE_BASE_USERS" --user-B "$YELP_PROBE_USER_B" \
    --probe-text "$PROBE_TEXT" --run-list "$YELP_OUT/probe_run_list.txt" \
    "${unit_args[@]}" "${skip_args[@]}" "${embed_args[@]}"
  if [[ "$RUN_RECORD" == "1" ]]; then python -m src.summarize_runs --runs-dir "$YELP_OUT/record/runs" --output-dir "$YELP_OUT/summaries/record" --prefix record; fi
  if [[ "$RUN_USER" == "1" ]]; then python -m src.summarize_runs --runs-dir "$YELP_OUT/user/runs" --output-dir "$YELP_OUT/summaries/user" --prefix user; fi
  python -m src.report --dataset-name "$YELP_SLUG" --out "$YELP_OUT" --record-summary "$YELP_OUT/summaries/record/record_summary.csv" --user-summary "$YELP_OUT/summaries/user/user_summary.csv"
fi
