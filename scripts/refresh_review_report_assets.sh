#!/usr/bin/env bash
set -euo pipefail

# Rebuild Amazon and Yelp tables and figures from saved runs.
# Only saved metrics are read; DP-SPIN mechanisms are not rerun.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

AMAZON_CATEGORY="${AMAZON_CATEGORY:-All_Beauty}"
AMAZON_OUT="${AMAZON_OUT:-outputs/amazon_${AMAZON_CATEGORY}}"
YELP_CATEGORY_FILTER="${YELP_CATEGORY_FILTER:-Restaurants}"
if [[ -n "$YELP_CATEGORY_FILTER" ]]; then
  YELP_SLUG="yelp_$(echo "$YELP_CATEGORY_FILTER" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
else
  YELP_SLUG="yelp_all"
fi
YELP_OUT="${YELP_OUT:-outputs/${YELP_SLUG}}"

refresh_one_dataset() {
  local dataset_name="$1"
  local out="$2"
  echo "Refreshing saved-run report assets for ${dataset_name} under ${out}"

  if [[ -d "$out/record/runs" ]]; then
    python -m src.summarize_runs \
      --runs-dir "$out/record/runs" \
      --output-dir "$out/summaries/record" \
      --prefix record
  else
    echo "No record runs found at $out/record/runs"
  fi

  if [[ -d "$out/user/runs" ]]; then
    python -m src.summarize_runs \
      --runs-dir "$out/user/runs" \
      --output-dir "$out/summaries/user" \
      --prefix user
  else
    echo "No user runs found at $out/user/runs"
  fi

  python -m src.report \
    --dataset-name "$dataset_name" \
    --out "$out" \
    --record-summary "$out/summaries/record/record_summary.csv" \
    --user-summary "$out/summaries/user/user_summary.csv"
}

refresh_one_dataset "amazon_${AMAZON_CATEGORY}" "$AMAZON_OUT"
refresh_one_dataset "$YELP_SLUG" "$YELP_OUT"

echo "Done. Updated figures are under:"
echo "  $AMAZON_OUT/plots"
echo "  $YELP_OUT/plots"
