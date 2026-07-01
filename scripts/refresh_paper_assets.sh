#!/usr/bin/env bash
set -euo pipefail

# Rebuild manuscript assets from existing CFPB experiment outputs.
# Use this after an interrupted reporting step when the run directories are
# already complete. It regenerates tables, asset directories, and the manifest
# without rerunning experiment grids.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

OUT="${OUT:-outputs/cfpb}"

MAIN="$OUT/main"
HARD="$OUT/hard"
ABLATION="$OUT/ablation"
PROBE="$OUT/probe"

mkdir -p "$OUT/paper_assets"

echo "Refreshing report tables from existing run directories under $OUT"
if [[ -d "$MAIN/runs" ]]; then
  python -m src.report --runs-dir "$MAIN/runs" --output-dir "$MAIN/reports"
else
  echo "Warning: missing $MAIN/runs"
fi

if [[ -d "$HARD/runs" ]]; then
  python -m src.report --runs-dir "$HARD/runs" --output-dir "$HARD/reports"
else
  echo "Warning: missing $HARD/runs"
fi

if [[ -d "$ABLATION/runs" ]]; then
  python -m src.report --runs-dir "$ABLATION/runs" --output-dir "$ABLATION/reports"
else
  echo "Warning: missing $ABLATION/runs"
fi

if [[ -d "$PROBE/runs" ]]; then
  python -m src.report --runs-dir "$PROBE/runs" --output-dir "$PROBE/reports"
else
  echo "Warning: missing $PROBE/runs"
fi

echo "Regenerating URANIA-style paper tables and figures from existing outputs..."
python -m src.urania_report --out "$OUT" --probe-text "${PROBE_TEXT:-BLUE-LANTERN-791}" || echo "Warning: URANIA-style report generation skipped or incomplete"

echo "Collecting canonical paper assets..."
python -m src.paper_assets --out "$OUT"

echo "Done. Use these folders in the paper:"
echo "  $OUT/paper_assets/figures"
echo "  $OUT/paper_assets/tables"
echo "  $OUT/paper_assets/raw_tables"
echo "  $OUT/paper_assets/MANIFEST.md"
