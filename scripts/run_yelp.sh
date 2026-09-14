#!/usr/bin/env bash
set -euo pipefail

# Run the Yelp Open Dataset DP-SPIN experiments.
#
# Yelp data remain subject to Yelp/Kaggle terms. Kaggle download requires a
# configured Kaggle CLI and prior acceptance of those terms. The default corpus
# is the Restaurants subset obtained by joining reviews with the business file.
# Set YELP_CATEGORY_FILTER="" to use all business categories.
#
# Usage:
#   DOWNLOAD_YELP=1 YELP_SOURCE=kaggle bash scripts/run_yelp.sh
#   bash scripts/run_yelp.sh
#   FULL=1 bash scripts/run_yelp.sh
#   RUN_RECORD=1 RUN_USER=1 bash scripts/run_yelp.sh
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
SKIP_FULL_DATASET_STATS="${SKIP_FULL_DATASET_STATS:-1}"  # Skip the full-corpus pass to limit memory use.
YELP_ONE_RUN_PER_PROCESS="${YELP_ONE_RUN_PER_PROCESS:-1}"  # Isolate each grid point in a fresh process.
YELP_STREAM_SPLIT="${YELP_STREAM_SPLIT:-1}"  # Stream the split for the full Yelp JSONL file.

DOWNLOAD_YELP="${DOWNLOAD_YELP:-0}"
YELP_SOURCE="${YELP_SOURCE:-manual}"  # manual or kaggle
YELP_RAW_DIR="${YELP_RAW_DIR:-data/raw/yelp}"
YELP_REVIEWS="${YELP_REVIEWS:-$YELP_RAW_DIR/yelp_academic_dataset_review.json}"
YELP_BUSINESSES="${YELP_BUSINESSES:-$YELP_RAW_DIR/yelp_academic_dataset_business.json}"
RAW_MAX_ROWS="${RAW_MAX_ROWS:-}"
YELP_CATEGORY_FILTER="${YELP_CATEGORY_FILTER:-Restaurants}"
YELP_ATOM_REFINEMENT="${YELP_ATOM_REFINEMENT:-deduplicate}"  # none, deduplicate, or llm
YELP_ATOM_REFINE_BACKEND="${YELP_ATOM_REFINE_BACKEND:-template}"  # template or openai
YELP_ATOM_REFINE_MODEL="${YELP_ATOM_REFINE_MODEL:-gpt-4o-mini}"
YELP_ATOM_REFINE_TOP_CANDIDATES="${YELP_ATOM_REFINE_TOP_CANDIDATES:-1000}"

# Controlled-probe settings. The primary runner includes these conditions so the
# main-grid utility results and probe diagnostics use the same prepared data.
YELP_PROBE_M="${YELP_PROBE_M:-100}"
YELP_PROBE_SUPPORTS="${YELP_PROBE_SUPPORTS:-1 2 5 10 25 50 100 200}"
YELP_PROBE_EPS="${YELP_PROBE_EPS:-${YELP_EPS:-0.5 1 2 4}}"
YELP_PROBE_SEEDS="${YELP_PROBE_SEEDS:-${YELP_SEEDS:-0 1 2}}"
YELP_PROBE_MECHS="${YELP_PROBE_MECHS:-${YELP_MECHS:-gaussian laplace private_topL}}"
YELP_PROBE_K="${YELP_PROBE_K:-3}"
YELP_PROBE_L="${YELP_PROBE_L:-10}"
YELP_PROBE_BASE_RECORDS="${YELP_PROBE_BASE_RECORDS:-2000}"
YELP_PROBE_BASE_USERS="${YELP_PROBE_BASE_USERS:-1000}"
YELP_PROBE_USER_B="${YELP_PROBE_USER_B:-2}"
YELP_PROBE_TEXT="${YELP_PROBE_TEXT:-The review contains the synthetic probe phrase BLUE-LANTERN-791.}"

if [[ -n "$YELP_CATEGORY_FILTER" ]]; then
  # Use a distinct slug so filtered and unfiltered artifacts cannot be mixed.
  YELP_SLUG="yelp_$(echo "$YELP_CATEGORY_FILTER" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//')"
else
  YELP_SLUG="yelp_all"
fi

RECORDS="data/processed/${YELP_SLUG}_reviews.jsonl"
AUX_RECORDS="data/auxiliary/${YELP_SLUG}_atom_auxiliary_records.jsonl"
TARGET_RECORDS="data/protected_target/${YELP_SLUG}_target_records.jsonl"
ATOM_DIR="data/public_atoms"
ATOMS_M50="$ATOM_DIR/${YELP_SLUG}_atoms_m50.json"
ATOMS_M100="$ATOM_DIR/${YELP_SLUG}_atoms_m100.json"
ATOMS_M200="$ATOM_DIR/${YELP_SLUG}_atoms_m200.json"
OUT="outputs/${YELP_SLUG}"
mkdir -p data/raw "$YELP_RAW_DIR" data/processed data/auxiliary data/protected_target "$ATOM_DIR" "$OUT"/record/runs "$OUT"/user/runs "$OUT"/plots "$OUT"/reports "$OUT"/stats "$OUT"/summaries

echo "Yelp category filter: ${YELP_CATEGORY_FILTER:-<none>}"
echo "Yelp experiment slug: $YELP_SLUG"

if [[ "$DOWNLOAD_YELP" == "1" ]]; then
  DOWNLOAD_ARGS=(yelp --source "$YELP_SOURCE" --raw-dir "$YELP_RAW_DIR")
  if [[ "$FORCE_PREP" == "1" ]]; then DOWNLOAD_ARGS+=(--force); fi
  python -m src.download_data "${DOWNLOAD_ARGS[@]}"
else
  # The business file is optional only when no category filter is requested.
  if [[ -f "$YELP_BUSINESSES" ]]; then
    python -m src.download_data yelp --source manual --reviews "$YELP_REVIEWS" --businesses "$YELP_BUSINESSES"
  else
    python -m src.download_data yelp --source manual --reviews "$YELP_REVIEWS" --businesses ""
  fi
fi

PREP_ARGS=(--reviews "$YELP_REVIEWS" --output "$RECORDS")
if [[ -f "$YELP_BUSINESSES" ]]; then
  PREP_ARGS+=(--businesses "$YELP_BUSINESSES")
else
  if [[ -n "$YELP_CATEGORY_FILTER" ]]; then
    echo "ERROR: YELP_CATEGORY_FILTER=$YELP_CATEGORY_FILTER requires YELP_BUSINESSES=$YELP_BUSINESSES" >&2
    exit 1
  fi
  echo "NOTE: Yelp business file not found. This is acceptable only when YELP_CATEGORY_FILTER is empty."
fi
if [[ -n "$YELP_CATEGORY_FILTER" ]]; then
  PREP_ARGS+=(--business-category-filter "$YELP_CATEGORY_FILTER")
fi
if [[ -n "$RAW_MAX_ROWS" ]]; then
  PREP_ARGS+=(--max-rows "$RAW_MAX_ROWS")
fi

if [[ "$FORCE_PREP" == "1" || "$REUSE_PREP" != "1" || ! -f "$RECORDS" ]]; then
  python -m src.prepare_reviews yelp "${PREP_ARGS[@]}"
else
  echo "Reusing processed Yelp records: $RECORDS"
fi
if [[ "$SKIP_FULL_DATASET_STATS" != "1" ]]; then
  python -m src.dataset_stats --records "$RECORDS" --user-id-field user_id --dataset-name "$YELP_SLUG" --split-name processed_all --output "$OUT/stats/processed_all_stats.json"
else
  echo "Skipping full Yelp processed_all dataset statistics (SKIP_FULL_DATASET_STATS=1)."
fi

# Split by user so the auxiliary and protected target corpora are user-disjoint.
if [[ "$FORCE_PREP" == "1" || "$REUSE_PREP" != "1" || ! -f "$AUX_RECORDS" || ! -f "$TARGET_RECORDS" ]]; then
  SPLIT_ARGS=(
    --input "$RECORDS"
    --aux-output "$AUX_RECORDS"
    --target-output "$TARGET_RECORDS"
    --method hash
    --aux-fraction 0.50
    --seed 1729
    --split-unit user
    --unit-field user_id
  )

  if [[ "$YELP_STREAM_SPLIT" == "1" ]]; then
    SPLIT_ARGS+=(--streaming)
  fi

  echo "Splitting Yelp auxiliary/target records. Streaming: $YELP_STREAM_SPLIT"
  python -m src.split_records "${SPLIT_ARGS[@]}"
else
  echo "Reusing Yelp auxiliary/target split."
fi
python -m src.dataset_stats --records "$AUX_RECORDS" --user-id-field user_id --dataset-name "$YELP_SLUG" --split-name auxiliary --output "$OUT/stats/auxiliary_stats.json"
python -m src.dataset_stats --records "$TARGET_RECORDS" --user-id-field user_id --dataset-name "$YELP_SLUG" --split-name protected_target --output "$OUT/stats/protected_target_stats.json"

# Construct review-theme atoms from the user-disjoint auxiliary split only;
# protected target reviews are never used to name atoms.
for spec in "50:$ATOMS_M50" "100:$ATOMS_M100" "200:$ATOMS_M200"; do
  max_atoms="${spec%%:*}"
  out="${spec#*:}"
  if [[ "$FORCE_PREP" == "1" || "$REUSE_PREP" != "1" || ! -f "$out" ]]; then
    python -m src.build_atoms \
      --records "$AUX_RECORDS" \
      --output "$out" \
      --mode text_phrases \
      --ngram-min "${YELP_NGRAM_MIN:-2}" \
      --ngram-max "${YELP_NGRAM_MAX:-2}" \
      --min-df "${YELP_MIN_DF:-2}" \
      --max-df "${YELP_MAX_DF:-1.0}" \
      --max-atoms "$max_atoms" \
      --source-role auxiliary \
      --refinement "${YELP_ATOM_REFINEMENT}" \
      --refine-backend "${YELP_ATOM_REFINE_BACKEND}" \
      --refine-model "${YELP_ATOM_REFINE_MODEL}" \
      --refine-top-candidates "${YELP_ATOM_REFINE_TOP_CANDIDATES}" \
      --review-domain yelp
  fi
done

if [[ "$FULL" == "1" ]]; then
  SEEDS=(0 1 2 3 4)
  EPS=(0.5 1 2 4 8)
  N_RECORDS=(5000 10000)
  N_USERS=(2000 5000)
  USER_BOUNDS=(1 2 5)
  M_VALUES=(50 100)
  K_VALUES=(1 3 5)
  L_VALUES=(5 10 20)
else
  SEEDS=(0 1 2)
  EPS=(0.5 1 2 4)
  N_RECORDS=(2000)
  N_USERS=(1000 2000)
  USER_BOUNDS=(1 2)
  M_VALUES=(50 100)
  K_VALUES=(1 3)
  L_VALUES=(5 10)
fi
# Space-delimited environment variables can restrict the grid. For example,
# YELP_N_USERS="1000 5000" and YELP_USER_BOUNDS="1 2" select user counts and
# clipping bounds; YELP_K_VALUES and YELP_L_VALUES select assignment sparsities
# and released plan sizes.
if [[ -n "${YELP_N_RECORDS:-}" ]]; then read -r -a N_RECORDS <<< "$YELP_N_RECORDS"; fi
if [[ -n "${YELP_N_USERS:-}" ]]; then read -r -a N_USERS <<< "$YELP_N_USERS"; fi
if [[ -n "${YELP_USER_BOUNDS:-}" ]]; then read -r -a USER_BOUNDS <<< "$YELP_USER_BOUNDS"; fi
if [[ -n "${YELP_M_VALUES:-}" ]]; then read -r -a M_VALUES <<< "$YELP_M_VALUES"; fi
if [[ -n "${YELP_K_VALUES:-}" ]]; then read -r -a K_VALUES <<< "$YELP_K_VALUES"; fi
if [[ -n "${YELP_L_VALUES:-}" ]]; then read -r -a L_VALUES <<< "$YELP_L_VALUES"; fi
MECHS=(gaussian laplace private_topL)
if [[ -n "${YELP_SEEDS:-}" ]]; then read -r -a SEEDS <<< "$YELP_SEEDS"; fi
if [[ -n "${YELP_EPS:-}" ]]; then read -r -a EPS <<< "$YELP_EPS"; fi
if [[ -n "${YELP_MECHS:-}" ]]; then read -r -a MECHS <<< "$YELP_MECHS"; fi

# Map legacy spellings to the current private_topL identifier. K denotes
# assignment sparsity; L denotes the number of admitted atoms.
for i in "${!MECHS[@]}"; do
  case "${MECHS[$i]}" in
    private_topk|private_topK|private_topl|private_top_l|private_top-l|sparse_private_topL)
      MECHS[$i]="private_topL"
      ;;
  esac
done

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
# Fit the TF-IDF assignment map on a deterministic auxiliary subset to bound
# memory and repeated preprocessing cost. Set YELP_EMBEDDING_FIT_MAX_RECORDS=""
# to fit on the full auxiliary split.
YELP_EMBEDDING_FIT_MAX_RECORDS="${YELP_EMBEDDING_FIT_MAX_RECORDS:-50000}"
YELP_EMBEDDING_FIT_SELECTION="${YELP_EMBEDDING_FIT_SELECTION:-hash}"
YELP_EMBEDDING_FIT_SEED="${YELP_EMBEDDING_FIT_SEED:-1729}"
GRID_DECODER_ARGS=(
  --decoder-backend "$DECODER_BACKEND"
  --decoder-model "$DECODER_MODEL"
  --decoder-temperature "$DECODER_TEMPERATURE"
  --clean-reference-decoder-backend "$CLEAN_REFERENCE_DECODER_BACKEND"
  --clean-reference-decoder-model "$CLEAN_REFERENCE_DECODER_MODEL"
  --summary-comparative-eval-backend "$SUMMARY_COMPARATIVE_EVAL_BACKEND"
  --summary-embedding-backend "$SUMMARY_EMBEDDING_BACKEND"
)
if [[ -n "$YELP_EMBEDDING_FIT_MAX_RECORDS" ]]; then
  GRID_DECODER_ARGS+=(
    --embedding-fit-max-records "$YELP_EMBEDDING_FIT_MAX_RECORDS"
    --embedding-fit-selection "$YELP_EMBEDDING_FIT_SELECTION"
    --embedding-fit-seed "$YELP_EMBEDDING_FIT_SEED"
  )
fi

echo "Yelp grid: m=${M_VALUES[*]}, K=${K_VALUES[*]}, L=${L_VALUES[*]}, eps=${EPS[*]}, seeds=${SEEDS[*]}, mechanisms=${MECHS[*]}"
echo "Yelp embedding fit max records: ${YELP_EMBEDDING_FIT_MAX_RECORDS:-all}"
RUN_GRID_EXTRA_ARGS=()
if [[ "$SKIP_EXISTING" == "1" ]]; then RUN_GRID_EXTRA_ARGS+=(--skip-existing); fi

if [[ "$CLEAN_RUNS" == "1" ]]; then
  echo "CLEAN_RUNS=1: removing selected run directories under $OUT before execution."
  if [[ "$RUN_RECORD" == "1" ]]; then rm -rf "$OUT/record/runs"; mkdir -p "$OUT/record/runs"; fi
  if [[ "$RUN_USER" == "1" ]]; then rm -rf "$OUT/user/runs"; mkdir -p "$OUT/user/runs"; fi
fi

run_yelp_record_grid() {
  if [[ "$YELP_ONE_RUN_PER_PROCESS" == "1" ]]; then
    for n in "${N_RECORDS[@]}"; do
      for m in "${M_VALUES[@]}"; do
        for K in "${K_VALUES[@]}"; do
          for L in "${L_VALUES[@]}"; do
            for eps in "${EPS[@]}"; do
              for seed in "${SEEDS[@]}"; do
                for mech in "${MECHS[@]}"; do
                  echo "[Yelp record] n=$n m=$m K=$K L=$L eps=$eps seed=$seed mech=$mech"
                  python -m src.run_grid \
                    --config configs/yelp_record_template.yaml \
                    --epsilons "$eps" \
                    --seeds "$seed" \
                    --mechanisms "$mech" \
                    --m-values "$m" \
                    --K-values "$K" \
                    --L-values "$L" \
                    --n-records-values "$n" \
                    --records "$TARGET_RECORDS" \
                    --embedding-fit-records "$AUX_RECORDS" \
                    --atoms-template "$ATOM_DIR/${YELP_SLUG}_atoms_m{m}.json" \
                    "${GRID_DECODER_ARGS[@]}" \
                    "${RUN_GRID_EXTRA_ARGS[@]}" \
                    --output-dir "$OUT/record/runs"
                done
              done
            done
          done
        done
      done
    done
  else
    python -m src.run_grid \
      --config configs/yelp_record_template.yaml \
      --epsilons "${EPS[@]}" \
      --seeds "${SEEDS[@]}" \
      --mechanisms "${MECHS[@]}" \
      --m-values "${M_VALUES[@]}" \
      --K-values "${K_VALUES[@]}" \
      --L-values "${L_VALUES[@]}" \
      --n-records-values "${N_RECORDS[@]}" \
      --records "$TARGET_RECORDS" \
      --embedding-fit-records "$AUX_RECORDS" \
      --atoms-template "$ATOM_DIR/${YELP_SLUG}_atoms_m{m}.json" \
      "${GRID_DECODER_ARGS[@]}" \
      "${RUN_GRID_EXTRA_ARGS[@]}" \
      --output-dir "$OUT/record/runs"
  fi
}

run_yelp_user_grid() {
  if [[ "$YELP_ONE_RUN_PER_PROCESS" == "1" ]]; then
    for u in "${N_USERS[@]}"; do
      for B in "${USER_BOUNDS[@]}"; do
        for m in "${M_VALUES[@]}"; do
          for K in "${K_VALUES[@]}"; do
            for L in "${L_VALUES[@]}"; do
              for eps in "${EPS[@]}"; do
                for seed in "${SEEDS[@]}"; do
                  for mech in "${MECHS[@]}"; do
                    echo "[Yelp user] u=$u B=$B m=$m K=$K L=$L eps=$eps seed=$seed mech=$mech"
                    python -m src.run_grid \
                      --config configs/yelp_user_template.yaml \
                      --epsilons "$eps" \
                      --seeds "$seed" \
                      --mechanisms "$mech" \
                      --m-values "$m" \
                      --K-values "$K" \
                      --L-values "$L" \
                      --n-users-values "$u" \
                      --user-contribution-bounds "$B" \
                      --records "$TARGET_RECORDS" \
                      --embedding-fit-records "$AUX_RECORDS" \
                      --atoms-template "$ATOM_DIR/${YELP_SLUG}_atoms_m{m}.json" \
                      "${GRID_DECODER_ARGS[@]}" \
                      "${RUN_GRID_EXTRA_ARGS[@]}" \
                      --output-dir "$OUT/user/runs"
                  done
                done
              done
            done
          done
        done
      done
    done
  else
    python -m src.run_grid \
      --config configs/yelp_user_template.yaml \
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
      --atoms-template "$ATOM_DIR/${YELP_SLUG}_atoms_m{m}.json" \
      "${GRID_DECODER_ARGS[@]}" \
      "${RUN_GRID_EXTRA_ARGS[@]}" \
      --output-dir "$OUT/user/runs"
  fi
}

if [[ "$RUN_RECORD" == "1" ]]; then
  run_yelp_record_grid
fi

if [[ "$RUN_USER" == "1" ]]; then
  run_yelp_user_grid
fi

if [[ "$RUN_PROBE" == "1" ]]; then
  PROBE_UNIT_ARGS=()
  if [[ "$RUN_RECORD" == "1" ]]; then PROBE_UNIT_ARGS+=(--run-record); fi
  if [[ "$RUN_USER" == "1" ]]; then PROBE_UNIT_ARGS+=(--run-user); fi
  PROBE_SKIP_ARGS=()
  if [[ "$SKIP_EXISTING" == "1" ]]; then PROBE_SKIP_ARGS+=(--skip-existing); fi
  EMBED_LIMIT_ARGS=()
  if [[ -n "$YELP_EMBEDDING_FIT_MAX_RECORDS" ]]; then
    EMBED_LIMIT_ARGS+=(--embedding-fit-max-records "$YELP_EMBEDDING_FIT_MAX_RECORDS" --embedding-fit-selection "$YELP_EMBEDDING_FIT_SELECTION" --embedding-fit-seed "$YELP_EMBEDDING_FIT_SEED")
  fi
  python -m src.review_probe \
    --dataset-tag "$YELP_SLUG" \
    --out "$OUT" \
    --target-records "$TARGET_RECORDS" \
    --aux-records "$AUX_RECORDS" \
    --base-atoms "$ATOM_DIR/${YELP_SLUG}_atoms_m${YELP_PROBE_M}.json" \
    --probe-atoms "$ATOM_DIR/${YELP_SLUG}_atoms_probe_m${YELP_PROBE_M}.json" \
    --m "$YELP_PROBE_M" \
    --K "$YELP_PROBE_K" \
    --L "$YELP_PROBE_L" \
    --supports $YELP_PROBE_SUPPORTS \
    --epsilons $YELP_PROBE_EPS \
    --seeds $YELP_PROBE_SEEDS \
    --mechanisms $YELP_PROBE_MECHS \
    --record-base-records "$YELP_PROBE_BASE_RECORDS" \
    --user-base-users "$YELP_PROBE_BASE_USERS" \
    --user-B "$YELP_PROBE_USER_B" \
    --probe-text "$YELP_PROBE_TEXT" \
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
  --dataset-name "$YELP_SLUG" \
  --out "$OUT" \
  --record-summary "$OUT/summaries/record/record_summary.csv" \
  --user-summary "$OUT/summaries/user/user_summary.csv" || true

echo "Yelp runs complete. Outputs under $OUT."
