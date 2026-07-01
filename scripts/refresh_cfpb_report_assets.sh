#!/usr/bin/env bash
set -euo pipefail

# Rebuild CFPB reports, organized figures, and manuscript assets from saved runs.
# The script does not rerun a DP mechanism, rebuild atoms, decode summaries,
# call OpenAI, or regenerate baselines.
#
# Usage:
#   bash scripts/refresh_cfpb_report_assets.sh
#   OUT=outputs/cfpb bash scripts/refresh_cfpb_report_assets.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

OUT="${OUT:-outputs/cfpb}"
MAIN="$OUT/main"
HARD="$OUT/hard"
ABLATION="$OUT/ablation"
PROBE="$OUT/probe"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

if [[ ! -d "$OUT" ]]; then
  echo "ERROR: output folder not found: $OUT" >&2
  exit 1
fi

# Rebuild CSV, LaTeX, and metrics_summary.csv files from saved metrics.json files.
# src.plot_results requires output image paths, so it first writes temporary
# all-curve plots. src.cfpb_report_figures then replaces them with the organized
# block-level figures used by the manuscript.
for block in main hard ablation probe; do
  runs="$OUT/$block/runs"
  reports="$OUT/$block/reports"
  plots="$OUT/$block/plots"
  mkdir -p "$reports" "$plots"
  if [[ -d "$runs" ]]; then
    echo "Refreshing report for $block from $runs"
    python -m src.report --runs-dir "$runs" --output-dir "$reports" \
      2>&1 | tee "$LOGS/refresh_${block}_report.log"
    python -m src.plot_results \
      --runs-dir "$runs" \
      --csv "$OUT/$block/metrics_summary.csv" \
      --plot "$plots/utility_vs_epsilon.png" \
      --probe-plot "$plots/probe_admission.png" \
      2>&1 | tee "$LOGS/refresh_${block}_metrics.log"
  else
    echo "Skipping $block: missing $runs"
  fi
done

# Rebuild the organized figures and diagnostics from the saved runs and tables.
python -m src.cfpb_report_figures \
  --out "$OUT" \
  --epsilon "${CFPB_FIG_EPSILON:-0.5}" \
  --K "${CFPB_FIG_K:-3}" \
  --L "${CFPB_FIG_L:-10}" \
  --m "${CFPB_FIG_M:-200}" \
  --write-block-plots \
  2>&1 | tee "$LOGS/refresh_cfpb_report_figures.log"


# Remove legacy figure names that predate the controlled-probe terminology.
rm -f \
  "$OUT/paper_figures/fig6_probe_admission_vs_support.png" \
  "$OUT/paper_figures/fig7_probe_non_private_rank_vs_support.png" \
  "$OUT/paper_figures/fig6_cfpb_probe_admission_vs_support.png" \
  "$OUT/paper_figures/fig7_cfpb_probe_non_private_rank_vs_support.png" \
  "$OUT/paper_figures/probe_admission.png" \
  "$OUT/paper_figures/probe_admission_unused.png" \
  "$OUT/paper_assets/figures/fig6_probe_admission_vs_support.png" \
  "$OUT/paper_assets/figures/fig7_probe_non_private_rank_vs_support.png" \
  "$OUT/paper_assets/figures/fig6_cfpb_probe_admission_vs_support.png" \
  "$OUT/paper_assets/figures/fig7_cfpb_probe_non_private_rank_vs_support.png" \
  "$OUT/paper_assets/figures/probe_admission.png" \
  "$OUT/paper_assets/figures/probe_admission_unused.png"

# Collect the manuscript-ready assets in paper_assets/.
python -m src.paper_assets --out "$OUT" \
  2>&1 | tee "$LOGS/refresh_paper_assets.log"

echo "Refreshed CFPB report assets under $OUT"
echo "Paper figures: $OUT/paper_figures"
echo "Canonical figures: $OUT/paper_assets/figures"
echo "Canonical tables: $OUT/paper_assets/tables"
