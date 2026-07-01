#!/usr/bin/env bash
set -euo pipefail

# Build the compact manuscript figures from saved DP-SPIN reports.
# No mechanism, decoder, or baseline is rerun.
#
# Usage:
#   bash scripts/build_main_paper_figures.sh
#   MAIN_FIG_K=3 MAIN_FIG_L=10 MAIN_FIG_USER_B=1 bash scripts/build_main_paper_figures.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

REPORTS_ROOT="${REPORTS_ROOT:-outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/main_paper_figures}"
MAIN_FIG_K="${MAIN_FIG_K:-3}"
MAIN_FIG_L="${MAIN_FIG_L:-10}"
MAIN_FIG_USER_B="${MAIN_FIG_USER_B:-1}"

python -m src.main_paper_figures \
  --reports-root "$REPORTS_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --K "$MAIN_FIG_K" \
  --L "$MAIN_FIG_L" \
  --user-B "$MAIN_FIG_USER_B"
