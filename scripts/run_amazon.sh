#!/usr/bin/env bash
set -euo pipefail

# Run the Amazon Reviews 2023 DP-SPIN experiments.
#
# If the selected category file is missing and DOWNLOAD_AMAZON=auto, the runner
# downloads it from the configured McAuley Lab source. The prepared data include
# user_id, so both record-level and user-level releases are supported.
#
# Usage:
#   bash scripts/run_amazon.sh
#   AMAZON_CATEGORY=Digital_Music bash scripts/run_amazon.sh
#   FULL=1 bash scripts/run_amazon.sh
#   DOWNLOAD_AMAZON=0 AMAZON_REVIEWS=/path/to/reviews.jsonl.gz bash scripts/run_amazon.sh
#
# Atom refinement:
#   none         keep the frequent auxiliary phrases
#   deduplicate  merge duplicate auxiliary phrases deterministically (default)
#   llm          select from auxiliary candidates with the configured OpenAI backend

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo "Python: $(which python)"
python --version

python - <<'PY'
missing = []
for pkg in ["numpy", "scipy", "sklearn", "yaml", "pandas", "matplotlib"]:
    try:
        __import__(pkg if pkg != "sklearn" else "sklearn")
    except Exception:
        missing.append(pkg)
if missing:
    print("Missing packages:", missing)
    raise SystemExit(1)
print("All required packages are importable.")
PY

FULL="${FULL:-0}"
REUSE_PREP="${REUSE_PREP:-1}"
FORCE_PREP="${FORCE_PREP:-0}"
RUN_RECORD="${RUN_RECORD:-1}"
RUN_USER="${RUN_USER:-1}"
CLEAN_RUNS="${CLEAN_RUNS:-0}"  # Set to 1 to clear selected run directories before the grid.
SKIP_EXISTING="${SKIP_EXISTING:-0}"  # Preserve and skip completed operating points.
RUN_PROBE="${RUN_PROBE:-1}"          # Run the controlled-probe experiment after the main grid.

AMAZON_CATEGORY="${AMAZON_CATEGORY:-All_Beauty}"
AMAZON_REVIEWS="${AMAZON_REVIEWS:-data/raw/amazon_${AMAZON_CATEGORY}.jsonl.gz}"
DOWNLOAD_AMAZON="${DOWNLOAD_AMAZON:-auto}"  # auto, 1, or 0
AMAZON_DOWNLOAD_SOURCE="${AMAZON_DOWNLOAD_SOURCE:-official}"
AMAZON_USER_FIELD="${AMAZON_USER_FIELD:-}"
AMAZON_TEXT_FIELD="${AMAZON_TEXT_FIELD:-}"
RAW_MAX_ROWS="${RAW_MAX_ROWS:-}"
AMAZON_ATOM_REFINEMENT="${AMAZON_ATOM_REFINEMENT:-deduplicate}"  # none, deduplicate, or llm
AMAZON_ATOM_REFINE_BACKEND="${AMAZON_ATOM_REFINE_BACKEND:-template}"  # template or openai
AMAZON_ATOM_REFINE_MODEL="${AMAZON_ATOM_REFINE_MODEL:-gpt-4o-mini}"
AMAZON_ATOM_REFINE_TOP_CANDIDATES="${AMAZON_ATOM_REFINE_TOP_CANDIDATES:-1000}"

# Controlled-probe settings. The primary runner includes these conditions so the
# main-grid utility results and probe diagnostics use the same prepared data.
AMAZON_PROBE_M="${AMAZON_PROBE_M:-46}"
AMAZON_PROBE_SUPPORTS="${AMAZON_PROBE_SUPPORTS:-1 2 5 10 25 50 100 200}"
AMAZON_PROBE_EPS="${AMAZON_PROBE_EPS:-${AMAZON_EPS:-0.5 1 2 4}}"
AMAZON_PROBE_SEEDS="${AMAZON_PROBE_SEEDS:-${AMAZON_SEEDS:-0 1 2}}"
AMAZON_PROBE_MECHS="${AMAZON_PROBE_MECHS:-${AMAZON_MECHS:-gaussian laplace private_topL}}"
AMAZON_PROBE_K="${AMAZON_PROBE_K:-3}"
AMAZON_PROBE_L="${AMAZON_PROBE_L:-10}"
AMAZON_PROBE_BASE_RECORDS="${AMAZON_PROBE_BASE_RECORDS:-2000}"
AMAZON_PROBE_BASE_USERS="${AMAZON_PROBE_BASE_USERS:-1000}"
AMAZON_PROBE_USER_B="${AMAZON_PROBE_USER_B:-2}"
AMAZON_PROBE_TEXT="${AMAZON_PROBE_TEXT:-The review contains the synthetic probe phrase BLUE-LANTERN-791.}"

RECORDS="data/processed/amazon_${AMAZON_CATEGORY}_reviews_all.jsonl"
AUX_RECORDS="data/auxiliary/amazon_${AMAZON_CATEGORY}_atom_auxiliary_records.jsonl"
TARGET_RECORDS="data/protected_target/amazon_${AMAZON_CATEGORY}_target_records.jsonl"
ATOM_DIR="data/public_atoms"
ATOMS_M46="$ATOM_DIR/amazon_${AMAZON_CATEGORY}_atoms_m46.json"
ATOMS_M50="$ATOM_DIR/amazon_${AMAZON_CATEGORY}_atoms_m50.json"
ATOMS_M100="$ATOM_DIR/amazon_${AMAZON_CATEGORY}_atoms_m100.json"
ATOMS_M200="$ATOM_DIR/amazon_${AMAZON_CATEGORY}_atoms_m200.json"
OUT="outputs/amazon_${AMAZON_CATEGORY}"
mkdir -p data/raw data/processed data/auxiliary data/protected_target "$ATOM_DIR" "$OUT"/record/runs "$OUT"/user/runs "$OUT"/plots "$OUT"/reports "$OUT"/stats "$OUT"/summaries
if [[ "$FORCE_PREP" == "1" ]]; then
  rm -f "$ATOM_DIR/amazon_${AMAZON_CATEGORY}_atoms_m500.json" "$ATOM_DIR/amazon_${AMAZON_CATEGORY}_atoms_m500.json.manifest.json"
fi

if [[ "$DOWNLOAD_AMAZON" == "1" || ( "$DOWNLOAD_AMAZON" == "auto" && ! -f "$AMAZON_REVIEWS" ) ]]; then
  DOWNLOAD_ARGS=(amazon --source "$AMAZON_DOWNLOAD_SOURCE" --category "$AMAZON_CATEGORY" --reviews "$AMAZON_REVIEWS")
  if [[ "$FORCE_PREP" == "1" ]]; then DOWNLOAD_ARGS+=(--force); fi
  python -m src.download_data "${DOWNLOAD_ARGS[@]}"
else
  python -m src.download_data amazon --source manual --reviews "$AMAZON_REVIEWS"
fi

PREP_ARGS=(--input "$AMAZON_REVIEWS" --output "$RECORDS" --category-name "$AMAZON_CATEGORY")
if [[ -n "$RAW_MAX_ROWS" ]]; then PREP_ARGS+=(--max-rows "$RAW_MAX_ROWS"); fi
if [[ -n "$AMAZON_USER_FIELD" ]]; then PREP_ARGS+=(--user-field "$AMAZON_USER_FIELD"); fi
if [[ -n "$AMAZON_TEXT_FIELD" ]]; then PREP_ARGS+=(--text-field "$AMAZON_TEXT_FIELD"); fi

if [[ "$FORCE_PREP" == "1" || "$REUSE_PREP" != "1" || ! -f "$RECORDS" ]]; then
  python -m src.prepare_reviews amazon "${PREP_ARGS[@]}"
else
  echo "Reusing processed Amazon records: $RECORDS"
fi
python -m src.dataset_stats --records "$RECORDS" --user-id-field user_id --dataset-name "amazon_${AMAZON_CATEGORY}" --split-name processed_all --output-dir "$OUT/stats"

# Split by user so the auxiliary and protected target corpora are user-disjoint.
if [[ "$FORCE_PREP" == "1" || "$REUSE_PREP" != "1" || ! -f "$AUX_RECORDS" || ! -f "$TARGET_RECORDS" ]]; then
  python -m src.split_records \
    --input "$RECORDS" \
    --aux-output "$AUX_RECORDS" \
    --target-output "$TARGET_RECORDS" \
    --method hash \
    --aux-fraction 0.50 \
    --seed 1729 \
    --split-unit user \
    --unit-field user_id
else
  echo "Reusing Amazon auxiliary/target split."
fi
python -m src.dataset_stats --records "$AUX_RECORDS" --user-id-field user_id --dataset-name "amazon_${AMAZON_CATEGORY}" --split-name auxiliary --output-dir "$OUT/stats"
python -m src.dataset_stats --records "$TARGET_RECORDS" --user-id-field user_id --dataset-name "amazon_${AMAZON_CATEGORY}" --split-name protected_target --output-dir "$OUT/stats"

# Construct review-theme atoms from the auxiliary split only; protected target
# reviews are never used to name atoms. Under the default All_Beauty filters and
# AMAZON_MIN_DF=25, 46 atoms remain after canonicalization and deduplication.
# The m50, m100, and m200 files are requested-cap diagnostics and may therefore
# contain the same 46 retained atoms. Changing the support threshold or filters
# can change the retained count.
AMAZON_ATOM_CAPS="${AMAZON_ATOM_CAPS:-46 50 100 200}"
for max_atoms in $AMAZON_ATOM_CAPS; do
  out="$ATOM_DIR/amazon_${AMAZON_CATEGORY}_atoms_m${max_atoms}.json"
  if [[ "$FORCE_PREP" == "1" || "$REUSE_PREP" != "1" || ! -f "$out" ]]; then
    python -m src.build_atoms \
      --records "$AUX_RECORDS" \
      --output "$out" \
      --mode text_phrases \
      --ngram-min "${AMAZON_NGRAM_MIN:-2}" \
      --ngram-max "${AMAZON_NGRAM_MAX:-2}" \
      --min-df "${AMAZON_MIN_DF:-25}" \
      --max-df "${AMAZON_MAX_DF:-1.0}" \
      --max-atoms "$max_atoms" \
      --source-role auxiliary \
      --refinement "${AMAZON_ATOM_REFINEMENT}" \
      --refine-backend "${AMAZON_ATOM_REFINE_BACKEND}" \
      --refine-model "${AMAZON_ATOM_REFINE_MODEL}" \
      --refine-top-candidates "${AMAZON_ATOM_REFINE_TOP_CANDIDATES}" \
      --review-domain amazon
  fi
done

if [[ "$FULL" == "1" ]]; then
  SEEDS=(0 1 2 3 4)
  EPS=(0.5 1 2 4 8)
  N_RECORDS=(5000 10000)
  N_USERS=(1000 2000 5000)
  USER_BOUNDS=(1 2 5)
  M_VALUES=(46)
  K_VALUES=(1 3 5)
  L_VALUES=(5 10 20)
else
  SEEDS=(0 1 2)
  EPS=(0.5 1 2 4)
  N_RECORDS=(2000 5000)
  N_USERS=(1000 2000)
  USER_BOUNDS=(1 2)
  M_VALUES=(46)
  K_VALUES=(1 3)
  L_VALUES=(5 10)
fi
# Space-delimited environment variables can restrict the grid. For example,
# AMAZON_N_USERS="1000 5000" and AMAZON_USER_BOUNDS="1 2" select user counts and
# clipping bounds. Leave AMAZON_M_VALUES unset for the manuscript m=46 setting;
# set it explicitly only for requested-cap ablations.
if [[ -n "${AMAZON_N_RECORDS:-}" ]]; then read -r -a N_RECORDS <<< "$AMAZON_N_RECORDS"; fi
if [[ -n "${AMAZON_N_USERS:-}" ]]; then read -r -a N_USERS <<< "$AMAZON_N_USERS"; fi
if [[ -n "${AMAZON_USER_BOUNDS:-}" ]]; then read -r -a USER_BOUNDS <<< "$AMAZON_USER_BOUNDS"; fi
if [[ -n "${AMAZON_M_VALUES:-}" ]]; then read -r -a M_VALUES <<< "$AMAZON_M_VALUES"; fi
if [[ -n "${AMAZON_K_VALUES:-}" ]]; then read -r -a K_VALUES <<< "$AMAZON_K_VALUES"; fi
if [[ -n "${AMAZON_L_VALUES:-}" ]]; then read -r -a L_VALUES <<< "$AMAZON_L_VALUES"; fi
MECHS=(gaussian laplace private_topL)
if [[ -n "${AMAZON_SEEDS:-}" ]]; then read -r -a SEEDS <<< "$AMAZON_SEEDS"; fi
if [[ -n "${AMAZON_EPS:-}" ]]; then read -r -a EPS <<< "$AMAZON_EPS"; fi
if [[ -n "${AMAZON_MECHS:-}" ]]; then read -r -a MECHS <<< "$AMAZON_MECHS"; fi

# The main grid uses the deterministic template decoder. Use
# scripts/run_openai_dpspin_only.sh or scripts/run_openai_paper.sh to verbalize
# selected saved plans with OpenAI.
DECODER_BACKEND="${DECODER_BACKEND:-template}"
DECODER_MODEL="${DECODER_MODEL:-gpt-4o-mini}"
DECODER_TEMPERATURE="${DECODER_TEMPERATURE:-0.0}"
CLEAN_REFERENCE_DECODER_BACKEND="${CLEAN_REFERENCE_DECODER_BACKEND:-template}"
CLEAN_REFERENCE_DECODER_MODEL="${CLEAN_REFERENCE_DECODER_MODEL:-$DECODER_MODEL}"
SUMMARY_COMPARATIVE_EVAL_BACKEND="${SUMMARY_COMPARATIVE_EVAL_BACKEND:-none}"
SUMMARY_EMBEDDING_BACKEND="${SUMMARY_EMBEDDING_BACKEND:-none}"
GRID_DECODER_ARGS=(
  --decoder-backend "$DECODER_BACKEND"
  --decoder-model "$DECODER_MODEL"
  --decoder-temperature "$DECODER_TEMPERATURE"
  --clean-reference-decoder-backend "$CLEAN_REFERENCE_DECODER_BACKEND"
  --clean-reference-decoder-model "$CLEAN_REFERENCE_DECODER_MODEL"
  --summary-comparative-eval-backend "$SUMMARY_COMPARATIVE_EVAL_BACKEND"
  --summary-embedding-backend "$SUMMARY_EMBEDDING_BACKEND"
)

echo "Amazon grid: m=${M_VALUES[*]}, K=${K_VALUES[*]}, L=${L_VALUES[*]}, eps=${EPS[*]}, seeds=${SEEDS[*]}, mechanisms=${MECHS[*]}"
RUN_GRID_EXTRA_ARGS=()
if [[ "$SKIP_EXISTING" == "1" ]]; then RUN_GRID_EXTRA_ARGS+=(--skip-existing); fi

if [[ "$CLEAN_RUNS" == "1" ]]; then
  echo "CLEAN_RUNS=1: removing selected run directories under $OUT before execution."
  if [[ "$RUN_RECORD" == "1" ]]; then rm -rf "$OUT/record/runs"; mkdir -p "$OUT/record/runs"; fi
  if [[ "$RUN_USER" == "1" ]]; then rm -rf "$OUT/user/runs"; mkdir -p "$OUT/user/runs"; fi
fi

if [[ "$RUN_RECORD" == "1" ]]; then
  python -m src.run_grid \
    --config configs/amazon_record_template.yaml \
    --epsilons "${EPS[@]}" \
    --seeds "${SEEDS[@]}" \
    --mechanisms "${MECHS[@]}" \
    --m-values "${M_VALUES[@]}" \
    --K-values "${K_VALUES[@]}" \
    --L-values "${L_VALUES[@]}" \
    --n-records-values "${N_RECORDS[@]}" \
    --records "$TARGET_RECORDS" \
    --embedding-fit-records "$AUX_RECORDS" \
    --atoms-template "$ATOM_DIR/amazon_${AMAZON_CATEGORY}_atoms_m{m}.json" \
    "${GRID_DECODER_ARGS[@]}" \
    "${RUN_GRID_EXTRA_ARGS[@]}" \
    --output-dir "$OUT/record/runs"
fi

if [[ "$RUN_USER" == "1" ]]; then
  python -m src.run_grid \
    --config configs/amazon_user_template.yaml \
    --epsilons "${EPS[@]}" \
    --seeds "${SEEDS[@]}" \
    --mechanisms "${MECHS[@]}" \
    --m-values "${M_VALUES[@]}" \
    --K-values "${K_VALUES[@]}" \
    --L-values "${L_VALUES[@]}" \
    --n-users-values "${N_USERS[@]}" \
    --user-contribution-bounds "${USER_BOUNDS[@]}" \
    --records "$TARGET_RECORDS" \
    --embedding-fit-records "$AUX_RECORDS" \
    --atoms-template "$ATOM_DIR/amazon_${AMAZON_CATEGORY}_atoms_m{m}.json" \
    "${GRID_DECODER_ARGS[@]}" \
    "${RUN_GRID_EXTRA_ARGS[@]}" \
    --output-dir "$OUT/user/runs"
fi

if [[ "$RUN_PROBE" == "1" ]]; then
  PROBE_UNIT_ARGS=()
  if [[ "$RUN_RECORD" == "1" ]]; then PROBE_UNIT_ARGS+=(--run-record); fi
  if [[ "$RUN_USER" == "1" ]]; then PROBE_UNIT_ARGS+=(--run-user); fi
  PROBE_SKIP_ARGS=()
  if [[ "$SKIP_EXISTING" == "1" ]]; then PROBE_SKIP_ARGS+=(--skip-existing); fi
  EMBED_LIMIT_ARGS=()
  if [[ -n "${AMAZON_EMBEDDING_FIT_MAX_RECORDS:-}" ]]; then
    EMBED_LIMIT_ARGS+=(--embedding-fit-max-records "$AMAZON_EMBEDDING_FIT_MAX_RECORDS")
  fi
  python -m src.review_probe \
    --dataset-tag "amazon_${AMAZON_CATEGORY}" \
    --out "$OUT" \
    --target-records "$TARGET_RECORDS" \
    --aux-records "$AUX_RECORDS" \
    --base-atoms "$ATOM_DIR/amazon_${AMAZON_CATEGORY}_atoms_m${AMAZON_PROBE_M}.json" \
    --probe-atoms "$ATOM_DIR/amazon_${AMAZON_CATEGORY}_atoms_probe_m${AMAZON_PROBE_M}.json" \
    --m "$AMAZON_PROBE_M" \
    --K "$AMAZON_PROBE_K" \
    --L "$AMAZON_PROBE_L" \
    --supports $AMAZON_PROBE_SUPPORTS \
    --epsilons $AMAZON_PROBE_EPS \
    --seeds $AMAZON_PROBE_SEEDS \
    --mechanisms $AMAZON_PROBE_MECHS \
    --record-base-records "$AMAZON_PROBE_BASE_RECORDS" \
    --user-base-users "$AMAZON_PROBE_BASE_USERS" \
    --user-B "$AMAZON_PROBE_USER_B" \
    --probe-text "$AMAZON_PROBE_TEXT" \
    --run-list "$OUT/probe_run_list.txt" \
    "${PROBE_UNIT_ARGS[@]}" \
    "${PROBE_SKIP_ARGS[@]}" \
    "${EMBED_LIMIT_ARGS[@]}"
fi

if [[ "$RUN_RECORD" == "1" ]]; then
  python -m src.summarize_runs --runs-dir "$OUT/record/runs" --output-dir "$OUT/summaries/record" --prefix record || true
fi
if [[ "$RUN_USER" == "1" ]]; then
  python -m src.summarize_runs --runs-dir "$OUT/user/runs" --output-dir "$OUT/summaries/user" --prefix user || true
fi


# Build grouped CSV and LaTeX tables and the dataset figures.
python -m src.report \
  --dataset-name "amazon_${AMAZON_CATEGORY}" \
  --out "$OUT" \
  --record-summary "$OUT/summaries/record/record_summary.csv" \
  --user-summary "$OUT/summaries/user/user_summary.csv" || true

echo "Amazon runs complete. Outputs under $OUT."
