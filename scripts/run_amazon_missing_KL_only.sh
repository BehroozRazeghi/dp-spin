#!/usr/bin/env bash
set -euo pipefail

# Append missing non-probe Amazon K-by-L operating points.
# Completed runs are preserved and skipped. Override the defaults at invocation
# time when a different operating point is required.

CLEAN_RUNS=0 \
SKIP_EXISTING=1 \
RUN_PROBE=0 \
RUN_RECORD="${RUN_RECORD:-1}" \
RUN_USER="${RUN_USER:-1}" \
AMAZON_CATEGORY="${AMAZON_CATEGORY:-All_Beauty}" \
AMAZON_M_VALUES="${AMAZON_M_VALUES:-46}" \
AMAZON_K_VALUES="${AMAZON_K_VALUES:-1 3}" \
AMAZON_L_VALUES="${AMAZON_L_VALUES:-5 10}" \
AMAZON_EPS="${AMAZON_EPS:-1}" \
AMAZON_SEEDS="${AMAZON_SEEDS:-0 1 2}" \
AMAZON_MECHS="${AMAZON_MECHS:-gaussian laplace private_topL}" \
AMAZON_N_RECORDS="${AMAZON_N_RECORDS:-10000}" \
AMAZON_N_USERS="${AMAZON_N_USERS:-5000}" \
AMAZON_USER_BOUNDS="${AMAZON_USER_BOUNDS:-1}" \
bash scripts/run_amazon.sh
