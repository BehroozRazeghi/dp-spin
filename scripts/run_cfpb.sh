#!/usr/bin/env bash
set -euo pipefail

# Run the CFPB record-level DP-SPIN experiments and regenerate their reports.
# CFPB public exports do not contain a stable user identifier, so this workflow
# supports record-level releases only.
#
# Usage:
#   bash scripts/run_cfpb.sh
#   FULL=1 bash scripts/run_cfpb.sh
#   USE_OPENAI=1 bash scripts/run_cfpb.sh
#   REUSE_PREP=0 bash scripts/run_cfpb.sh
#   FORCE_PREP=1 bash scripts/run_cfpb.sh
#
# FULL=0 runs the default grid; FULL=1 expands the grid. REUSE_PREP=0 and
# FORCE_PREP=1 rebuild processed records, the auxiliary and protected target corpora, and atoms.
# OPENAI_API_KEY is required only when USE_OPENAI=1.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo "Python: $(which python)"
python --version

# Environment checks
python - <<'PY'
missing = []
for pkg in ["pandas", "numpy", "sklearn", "yaml", "matplotlib", "jinja2", "requests", "tqdm"]:
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
USE_OPENAI="${USE_OPENAI:-0}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"

REUSE_PREP="${REUSE_PREP:-1}"
FORCE_PREP="${FORCE_PREP:-0}"

# Use the Amazon or Yelp runner for user-level experiments.

if [[ "$USE_OPENAI" == "1" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: USE_OPENAI=1 but OPENAI_API_KEY is not set."
    exit 1
  fi
  python - <<'PY'
try:
    import openai
    print("OpenAI package import OK.")
except Exception as e:
    print("ERROR: openai package is not installed. Run: pip install openai")
    raise
PY
fi

# Paths and experiment settings
RAW_CSV="data/raw/cfpb_complaints.csv"
RECORDS="data/processed/cfpb_records_all_public.jsonl"
AUX_RECORDS="data/auxiliary/cfpb_atom_auxiliary_records.jsonl"
TARGET_RECORDS="data/protected_target/cfpb_target_records.jsonl"
SPLIT_MANIFEST="data/protected_target/cfpb_target_records.jsonl.split_manifest.json"
ATOM_DIR="data/public_atoms"
ATOMS_M50="$ATOM_DIR/cfpb_atoms_m50.json"
ATOMS_M100="$ATOM_DIR/cfpb_atoms_m100.json"
ATOMS_M200="$ATOM_DIR/cfpb_atoms_m200.json"
ATOMS_DEFAULT="$ATOM_DIR/cfpb_atoms.json"
ATOMS_PROBE="$ATOM_DIR/cfpb_atoms_probe_m200.json"

OUT="outputs/cfpb"
MAIN="$OUT/main"
HARD="$OUT/hard"
ABLATION="$OUT/ablation"
PROBE="$OUT/probe"
BASELINES="$OUT/baselines"
MEMBERSHIP="$OUT/membership"
USER_LEVEL="$OUT/user_level"
OPENAI_OUT="$OUT/openai"
PAPER_FIGS="$OUT/paper_figures"
PAPER_TABLES="$OUT/paper_tables"
# paper_figures/ and paper_tables/ are staging directories. paper_assets/
# contains the manuscript-ready files.
PAPER_ASSETS="$OUT/paper_assets"
PAPER_ASSET_FIGS="$PAPER_ASSETS/figures"
PAPER_ASSET_TABLES="$PAPER_ASSETS/tables"
PAPER_RAW_TABLES="$PAPER_ASSETS/raw_tables"
PAPER_DIAGNOSTICS="$PAPER_ASSETS/diagnostics"
PAPER_TEXT="$PAPER_ASSETS/text"
LOGS="$OUT/logs"

mkdir -p \
  "$MAIN/runs" "$MAIN/plots" "$MAIN/reports" "$MAIN/diagnostics" \
  "$HARD/runs" "$HARD/plots" "$HARD/reports" "$HARD/diagnostics" \
  "$ABLATION/runs" "$ABLATION/plots" "$ABLATION/reports" \
  "$PROBE/runs" "$PROBE/plots" "$PROBE/reports" "$PROBE/data" "$PROBE/diagnostics" \
  "$BASELINES" "$MEMBERSHIP/runs" "$MEMBERSHIP/reports" \
  "$OPENAI_OUT/runs" "$OPENAI_OUT/reports" "$OPENAI_OUT/baselines" "$OPENAI_OUT/judgments" \
  "$PAPER_FIGS" "$PAPER_TABLES" \
  "$PAPER_ASSETS" "$PAPER_ASSET_FIGS" "$PAPER_ASSET_TABLES" "$PAPER_RAW_TABLES" "$PAPER_DIAGNOSTICS" "$PAPER_TEXT" \
  "$LOGS" data/raw data/processed data/auxiliary data/protected_target "$ATOM_DIR" configs/auto

if [[ "$FULL" == "1" ]]; then
  SEEDS=(0 1 2 3 4)
  MAIN_N=5000
  MAIN_M=(50 100 200)
  MAIN_K=(3)
  MAIN_L=(10)

  HARD_N=1000
  HARD_M=(200)
  HARD_K=(3)
  HARD_L=(5 10)
  HARD_EPS=(0.25 0.5 1 2 4)

  ABL_N=1000
  ABL_M=(50 100 200)
  ABL_K=(1 3 5)
  ABL_L=(5 10 20)
  ABL_EPS=(0.5 1 2)

  PROBE_BASE_N=2000
  PROBE_SEEDS=(0 1 2 3 4)
  PROBE_SUPPORTS=(1 2 5 10 25 50 75 100 150 200 500)
  PROBE_EPS=(0.5 1 2 4 8)
  PROBE_M=(50 200)
else
  SEEDS=(0 1 2)
  MAIN_N=2000
  MAIN_M=(50 100 200)
  MAIN_K=(3)
  MAIN_L=(10)

  HARD_N=1000
  HARD_M=(200)
  HARD_K=(3)
  HARD_L=(5 10)
  HARD_EPS=(0.25 0.5 1 2)

  ABL_N=1000
  ABL_M=(50 100 200)
  ABL_K=(1 3 5)
  ABL_L=(5 10)
  ABL_EPS=(0.5 1 2)

  PROBE_BASE_N=2000
  PROBE_SEEDS=(0 1 2)
  PROBE_SUPPORTS=(1 2 5 10 25 50 100 200 500)
  PROBE_EPS=(0.5 1 2 4)
  PROBE_M=(50 200)
fi

EPS=(0.5 1 2 4 8)
MECHS=(gaussian laplace private_topL)

# URANIA-style baseline: DP clustering, noisy cluster-size filtering, and
# per-cluster DP keyword histograms over a fixed public vocabulary.
URANIA_EPS_TOTAL="${URANIA_EPS_TOTAL:-1.0}"
URANIA_DELTA="${URANIA_DELTA:-1e-6}"
URANIA_CLUSTERS="${URANIA_CLUSTERS:-10}"
URANIA_KEYWORDS="${URANIA_KEYWORDS:-10}"
URANIA_MAX_KEYWORDS="${URANIA_MAX_KEYWORDS:-200}"
URANIA_KEYWORDS_PER_RECORD="${URANIA_KEYWORDS_PER_RECORD:-5}"
URANIA_SVD_DIM="${URANIA_SVD_DIM:-64}"
URANIA_DP_ITERATIONS="${URANIA_DP_ITERATIONS:-3}"
URANIA_RUN_PROBE="${URANIA_RUN_PROBE:-1}"

MAX_PROBE_SUPPORT=0
for s in "${PROBE_SUPPORTS[@]}"; do
  if (( s > MAX_PROBE_SUPPORT )); then MAX_PROBE_SUPPORT="$s"; fi
done

cat <<EOF
Settings:
  FULL=$FULL
  USE_OPENAI=$USE_OPENAI
  Main: n=$MAIN_N, m=${MAIN_M[*]}, K=${MAIN_K[*]}, L=${MAIN_L[*]}, eps=${EPS[*]}, seeds=${SEEDS[*]}
  Hard: n=$HARD_N, m=${HARD_M[*]}, K=${HARD_K[*]}, L=${HARD_L[*]}, eps=${HARD_EPS[*]}
  Ablation: n=$ABL_N, m=${ABL_M[*]}, K=${ABL_K[*]}, L=${ABL_L[*]}, eps=${ABL_EPS[*]}
  Probe: base_n=$PROBE_BASE_N, supports=${PROBE_SUPPORTS[*]}, eps=${PROBE_EPS[*]}, seeds=${PROBE_SEEDS[*]}
  CFPB protected unit: record-level only (no stable public user id)
EOF

# Data preparation
if [[ ! -f "$RAW_CSV" ]]; then
  echo "CFPB CSV not found. Downloading a subset..."
  python -m src.download_data cfpb --output "$RAW_CSV" --size 20000
else
  echo "Found existing CFPB CSV: $RAW_CSV"
fi

python - <<PY
import pandas as pd
path = "$RAW_CSV"
df = pd.read_csv(path, nrows=5)
print("CSV readable. Columns:", list(df.columns))
PY

if [[ "$FORCE_PREP" == "1" || "$REUSE_PREP" != "1" || ! -f "$RECORDS" ]]; then
  echo "Preprocessing CFPB records..."
  python -m src.prepare_cfpb \
    --input "$RAW_CSV" \
    --text-col "Consumer complaint narrative" \
    --id-col "Complaint ID" \
    --output "$RECORDS"
else
  echo "Reusing existing processed records: $RECORDS"
fi
wc -l "$RECORDS"

# Construct atoms from the auxiliary split and apply the DP mechanism only to
# the disjoint protected target split.
if [[ "$FORCE_PREP" == "1" || "$REUSE_PREP" != "1" || ! -f "$AUX_RECORDS" || ! -f "$TARGET_RECORDS" ]]; then
  echo "Creating auxiliary/target split for public-atom evaluation..."
  python -m src.split_records \
    --input "$RECORDS" \
    --aux-output "$AUX_RECORDS" \
    --target-output "$TARGET_RECORDS" \
    --method hash \
    --aux-fraction 0.50 \
    --seed 1729
else
  echo "Reusing existing auxiliary/target split: $AUX_RECORDS and $TARGET_RECORDS"
fi
wc -l "$AUX_RECORDS" "$TARGET_RECORDS"

# Fix the probe base subset before appending probe records; otherwise later
# truncation can remove the inserted probe records.
PROBE_BASE="$PROBE/data/cfpb_probe_base_${PROBE_BASE_N}.jsonl"
python - <<PY
from pathlib import Path
inp = Path("$TARGET_RECORDS")
out = Path("$PROBE_BASE")
out.parent.mkdir(parents=True, exist_ok=True)
N = int("$PROBE_BASE_N")
with inp.open() as f, out.open("w") as g:
    for i, line in enumerate(f):
        if i >= N:
            break
        g.write(line)
print("Wrote probe base subset from protected target split:", out, "records=", sum(1 for _ in out.open()))
PY

# Public atom dictionaries
# Every dictionary size uses the Product / Issue / Sub-issue schema. Rebuild
# legacy files that used a different schema at smaller values of m.
NEED_REBUILD_ATOMS=$(python - <<PY2
import json
from pathlib import Path
expected = ["Product", "Issue", "Sub-issue"]
paths = [Path("$ATOMS_M50"), Path("$ATOMS_M100"), Path("$ATOMS_M200")]
need = False
for p in paths:
    mp = Path(str(p) + ".manifest.json")
    if not p.exists() or not mp.exists():
        need = True
        continue
    try:
        man = json.load(open(mp))
    except Exception:
        need = True
        continue
    if man.get("construction_mode") != "metadata" or man.get("metadata_columns") != expected:
        need = True
print("1" if need else "0")
PY2
)

if [[ "$FORCE_PREP" == "1" || "$REUSE_PREP" != "1" || "$NEED_REBUILD_ATOMS" == "1" || ! -f "$ATOMS_M50" || ! -f "$ATOMS_M100" || ! -f "$ATOMS_M200" ]]; then
  echo "Building public atom dictionaries with schema: Product / Issue / Sub-issue..."
  # Vary m by retaining more auxiliary metadata combinations; do not change the
  # semantic definition of an atom across dictionary sizes.
  for spec in "50:$ATOMS_M50" "100:$ATOMS_M100" "200:$ATOMS_M200"; do
    max_atoms="${spec%%:*}"
    out="${spec#*:}"
    python -m src.build_atoms       --records "$AUX_RECORDS"       --mode metadata       --metadata-cols Product Issue Sub-issue       --max-atoms "$max_atoms"       --source-role auxiliary       --output "$out"
  done
  cp "$ATOMS_M200" "$ATOMS_DEFAULT"
else
  echo "Reusing existing public atom dictionaries."
  if [[ ! -f "$ATOMS_DEFAULT" ]]; then
    cp "$ATOMS_M200" "$ATOMS_DEFAULT"
  fi
fi

python - <<PY
import json
for p in ["$ATOMS_M50", "$ATOMS_M100", "$ATOMS_M200", "$ATOMS_DEFAULT"]:
    atoms = json.load(open(p))
    print(p, "num_atoms=", len(atoms))
PY

# Generated base configurations
# The YAML files hold stable paths and single-run defaults. src.run_grid later
# overrides epsilon, seed, mechanism, m, K, L, and n_records. Each value of m
# loads its own atom file through cfpb_atoms_m{m}.json; smaller dictionaries are
# not formed by truncating a larger one at run time.
echo "Creating configs..."
python - <<PY
import yaml
from pathlib import Path

def write_cfg(path, experiment_name, records, atoms, output_dir, n_records, m, K=3, L=10, decoder_backend="template"):
    cfg = {
        "experiment_name": experiment_name,
        "records": records,
        "atoms": atoms,
        "embedding_fit_records": "$AUX_RECORDS",
        "output_dir": output_dir,
        "protected_unit": "record",
        "adjacency": "adddrop",
        "n_records": int(n_records),
        "m": int(m),
        "K": int(K),
        "gamma": 10.0,
        "L": int(L),
        "epsilon": 1.0,
        "delta": 1.0e-6,
        "mechanism": "gaussian",
        "public_denominator": int(n_records),
        "gaussian_calibration": "analytic",
        "atom_source_protocol": "atoms built from disjoint auxiliary split; protected records are target split",
        "atom_dictionary_protocol": "for grids with varying m in {50,100,200}, run_grid receives --atoms-template so each m uses its separately built Product/Issue/Sub-issue atom file",
        "config_role": "auto-generated base config; run_grid overwrites grid parameters from command-line arrays",
        "seed": 0,
        "decoder_backend": decoder_backend,
        "decoder_contract_mode": "legacy_text",
        "summary_embedding_backend": "none",
        "support_bins": {"very high": 0.20, "high": 0.10, "moderate": 0.05, "low": 0.0},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("Wrote", path)

write_cfg("configs/auto/cfpb_main.yaml", "cfpb_main", "$TARGET_RECORDS", "$ATOMS_M200", "$MAIN/runs", $MAIN_N, 200, 3, 10)
write_cfg("configs/auto/cfpb_hard.yaml", "cfpb_hard", "$TARGET_RECORDS", "$ATOMS_M200", "$HARD/runs", $HARD_N, 200, 3, 5)
write_cfg("configs/auto/cfpb_ablation.yaml", "cfpb_ablation", "$TARGET_RECORDS", "$ATOMS_M200", "$ABLATION/runs", $ABL_N, 200, 3, 10)
write_cfg("configs/auto/cfpb_membership.yaml", "cfpb_membership", "$TARGET_RECORDS", "$ATOMS_M200", "$MEMBERSHIP/runs", 2000, 200, 3, 10)

PY

# Main utility grid
echo "Running MAIN utility grid..."
python -m src.run_grid \
  --config configs/auto/cfpb_main.yaml \
  --epsilons "${EPS[@]}" \
  --seeds "${SEEDS[@]}" \
  --mechanisms "${MECHS[@]}" \
  --m-values "${MAIN_M[@]}" \
  --K-values "${MAIN_K[@]}" \
  --L-values "${MAIN_L[@]}" \
  --n-records-values "$MAIN_N" \
  --atoms-template "$ATOM_DIR/cfpb_atoms_m{m}.json" \
  2>&1 | tee "$LOGS/main_grid.log"

python -m src.plot_results --runs-dir "$MAIN/runs" --csv "$MAIN/metrics_summary.csv" --plot "$MAIN/plots/utility_vs_epsilon.png" --probe-plot "$MAIN/plots/probe_admission_unused.png"
python -m src.report --runs-dir "$MAIN/runs" --output-dir "$MAIN/reports"

# Lower-signal utility grid
echo "Running HARD utility grid..."
python -m src.run_grid \
  --config configs/auto/cfpb_hard.yaml \
  --epsilons "${HARD_EPS[@]}" \
  --seeds "${SEEDS[@]}" \
  --mechanisms "${MECHS[@]}" \
  --m-values "${HARD_M[@]}" \
  --K-values "${HARD_K[@]}" \
  --L-values "${HARD_L[@]}" \
  --n-records-values "$HARD_N" \
  --atoms-template "$ATOM_DIR/cfpb_atoms_m{m}.json" \
  2>&1 | tee "$LOGS/hard_grid.log"

python -m src.plot_results --runs-dir "$HARD/runs" --csv "$HARD/metrics_summary.csv" --plot "$HARD/plots/utility_vs_epsilon.png" --probe-plot "$HARD/plots/probe_admission_unused.png"
python -m src.report --runs-dir "$HARD/runs" --output-dir "$HARD/reports"

# Dictionary-size and K-by-L ablation grid
echo "Running ABLATION grid..."
python -m src.run_grid \
  --config configs/auto/cfpb_ablation.yaml \
  --epsilons "${ABL_EPS[@]}" \
  --seeds "${SEEDS[@]}" \
  --mechanisms "${MECHS[@]}" \
  --m-values "${ABL_M[@]}" \
  --K-values "${ABL_K[@]}" \
  --L-values "${ABL_L[@]}" \
  --n-records-values "$ABL_N" \
  --atoms-template "$ATOM_DIR/cfpb_atoms_m{m}.json" \
  2>&1 | tee "$LOGS/ablation_grid.log"

python -m src.plot_results --runs-dir "$ABLATION/runs" --csv "$ABLATION/metrics_summary.csv" --plot "$ABLATION/plots/utility_vs_epsilon.png" --probe-plot "$ABLATION/plots/probe_admission_unused.png"
python -m src.report --runs-dir "$ABLATION/runs" --output-dir "$ABLATION/reports"

# Controlled-probe dictionaries and sweeps
echo "Creating controlled probe dictionaries and configs for base m in: ${PROBE_M[*]}"
for probe_base_m in "${PROBE_M[@]}"; do
  ATOMS_BASE="$ATOM_DIR/cfpb_atoms_m${probe_base_m}.json"
  ATOMS_PROBE_M="$ATOM_DIR/cfpb_atoms_probe_m${probe_base_m}.json"
  PROBE_CFG="configs/auto/cfpb_probe_m${probe_base_m}.yaml"
  if [[ ! -f "$ATOMS_BASE" ]]; then
    echo "Missing base atom dictionary $ATOMS_BASE" >&2
    exit 1
  fi

  python - <<PY
import json
from pathlib import Path
atoms = json.load(open("$ATOMS_BASE"))
atoms = [a for a in atoms if "BLUE-LANTERN-791" not in str(a).upper() and "BLUE LANTERN" not in str(a).upper()]
atoms.append({
    "atom_id": len(atoms),
    "label": "blue lantern unique code",
    "description": "Records mentioning the unique code BLUE-LANTERN-791.",
    "domain": "probe_stress_test",
})
Path("$ATOMS_PROBE_M").parent.mkdir(parents=True, exist_ok=True)
json.dump(atoms, open("$ATOMS_PROBE_M", "w"), indent=2)
print("Wrote", "$ATOMS_PROBE_M", "with", len(atoms), "atoms")
PY

  python - <<PY
import yaml, json
atoms = json.load(open("$ATOMS_PROBE_M"))
cfg = {
    "experiment_name": "cfpb_probe_m${probe_base_m}",
    "records": "$PROBE_BASE",
    "atoms": "$ATOMS_PROBE_M",
    "embedding_fit_records": "$AUX_RECORDS",
    "output_dir": "$PROBE/runs",
    "protected_unit": "record",
    "adjacency": "adddrop",
    # Default for a manual single run. src.probe_sweep replaces it with the
    # generated size, base_n + support, for each probe condition.
    "n_records": int($PROBE_BASE_N),
    "m": len(atoms),
    "K": 3,
    "gamma": 10.0,
    "L": 10,
    "epsilon": 1.0,
    "delta": 1.0e-6,
    "mechanism": "gaussian",
    # Normalize by the actual generated corpus size, base_n + support.
    "public_denominator": int($PROBE_BASE_N),
    "probe_public_denominator_mode": "actual_generated",
    "probe_base_size": int($PROBE_BASE_N),
    "probe_atom_ids": [len(atoms) - 1],
    "seed": 0,
    "decoder_backend": "template",
    "decoder_contract_mode": "legacy_text",
    "summary_embedding_backend": "none",
    "support_bins": {"very high": 0.20, "high": 0.10, "moderate": 0.05, "low": 0.0},
}
yaml.safe_dump(cfg, open("$PROBE_CFG", "w"), sort_keys=False)
print(open("$PROBE_CFG").read())
PY

  echo "Running corrected PROBE sweep for base m=${probe_base_m} (probe dictionary size m=$((probe_base_m + 1)))..."
  python -m src.probe_sweep \
    --config "$PROBE_CFG" \
    --supports "${PROBE_SUPPORTS[@]}" \
    --epsilons "${PROBE_EPS[@]}" \
    --seeds "${PROBE_SEEDS[@]}" \
    --mechanisms "${MECHS[@]}" \
    --work-dir "$PROBE/data" \
    --probe-text "The complaint contains the secret phrase BLUE-LANTERN-791." \
    2>&1 | tee "$LOGS/probe_sweep_m${probe_base_m}.log"
done

# Baselines that accept one probe dictionary use the m=200 version.
ATOMS_PROBE="$ATOM_DIR/cfpb_atoms_probe_m200.json"

python -m src.plot_results --runs-dir "$PROBE/runs" --csv "$PROBE/metrics_summary.csv" --plot "$PROBE/plots/utility_vs_epsilon.png" --probe-plot "$PROBE/plots/probe_admission.png"
python -m src.report --runs-dir "$PROBE/runs" --output-dir "$PROBE/reports"

# Baselines and exact probe-mention diagnostics
echo "Running template baselines..."
python -m src.baselines \
  --records "$TARGET_RECORDS" \
  --atoms "$ATOMS_M200" \
  --output-dir "$BASELINES/cfpb_template" \
  --epsilon 1.0 \
  --seed 0 \
  --category-key Issue \
  --category-universe-records "$AUX_RECORDS" \
  2>&1 | tee "$LOGS/baselines_template.log"

echo "Running URANIA-style clustered public-keyword DP baseline..."
python -m src.urania_baseline \
  --records "$TARGET_RECORDS" \
  --aux-records "$AUX_RECORDS" \
  --atoms "$ATOMS_M200" \
  --output-dir "$BASELINES/cfpb_template" \
  --epsilon-total "$URANIA_EPS_TOTAL" \
  --delta "$URANIA_DELTA" \
  --num-clusters "$URANIA_CLUSTERS" \
  --keywords-per-summary "$URANIA_KEYWORDS" \
  --max-keywords "$URANIA_MAX_KEYWORDS" \
  --max-keywords-per-record "$URANIA_KEYWORDS_PER_RECORD" \
  --svd-dim "$URANIA_SVD_DIM" \
  --dp-iterations "$URANIA_DP_ITERATIONS" \
  --seed 0 \
  --backend template \
  2>&1 | tee "$LOGS/urania_baseline_template.log"

python -m src.baseline_report \
  --baseline-dir "$BASELINES/cfpb_template" \
  --output-dir "$BASELINES/reports" \
  2>&1 | tee "$LOGS/baseline_report.log"

echo "Running probe raw-string baseline diagnostics..."
mkdir -p "$BASELINES/probe"
echo "support,baseline_path,exact_probe_mentioned" > "$BASELINES/probe/exact_probe_mention.csv"
for s in "${PROBE_SUPPORTS[@]}"; do
  PROBE_RECORDS="$PROBE/data/cfpb_probe_baseline_s${s}.jsonl"
  OUT_BASE="$BASELINES/probe/support_${s}"
  python -m src.probes --input "$PROBE_BASE" --output "$PROBE_RECORDS" --support "$s" --seed 0 --text "The complaint contains the secret phrase BLUE-LANTERN-791."
  python -m src.baselines --records "$PROBE_RECORDS" --atoms "$ATOMS_M200" --output-dir "$OUT_BASE" --epsilon 1.0 --seed 0 --category-key Issue --category-universe-records "$AUX_RECORDS"
  if grep -Rqi "BLUE-LANTERN-791\|blue-lantern" "$OUT_BASE"; then mentioned=1; else mentioned=0; fi
  echo "$s,$OUT_BASE,$mentioned" >> "$BASELINES/probe/exact_probe_mention.csv"
done

if [[ "$URANIA_RUN_PROBE" == "1" ]]; then
  echo "Running URANIA-style probe baseline diagnostics..."
  mkdir -p "$BASELINES/urania_probe"
  echo "support,baseline_path,exact_probe_mentioned,num_clusters_released" > "$BASELINES/urania_probe/exact_probe_mention.csv"
  for s in "${PROBE_SUPPORTS[@]}"; do
    PROBE_RECORDS="$PROBE/data/cfpb_probe_baseline_s${s}.jsonl"
    OUT_URANIA="$BASELINES/urania_probe/support_${s}"
    python -m src.urania_baseline \
      --records "$PROBE_RECORDS" \
      --aux-records "$AUX_RECORDS" \
      --atoms "$ATOMS_PROBE" \
      --output-dir "$OUT_URANIA" \
      --epsilon-total "$URANIA_EPS_TOTAL" \
      --delta "$URANIA_DELTA" \
      --num-clusters "$URANIA_CLUSTERS" \
      --keywords-per-summary "$URANIA_KEYWORDS" \
      --max-keywords "$URANIA_MAX_KEYWORDS" \
      --max-keywords-per-record "$URANIA_KEYWORDS_PER_RECORD" \
      --svd-dim "$URANIA_SVD_DIM" \
      --dp-iterations "$URANIA_DP_ITERATIONS" \
      --seed 0 \
      --backend template
    if grep -Rqi "BLUE-LANTERN-791\|blue-lantern\|blue lantern" "$OUT_URANIA"; then mentioned=1; else mentioned=0; fi
    clusters=$(python - <<PY2
import json, pathlib
p = pathlib.Path("$OUT_URANIA/urania_cluster_public_keyword_metrics.json")
print(json.load(open(p)).get("num_clusters_released", "")) if p.exists() else print("")
PY2
)
    echo "$s,$OUT_URANIA,$mentioned,$clusters" >> "$BASELINES/urania_probe/exact_probe_mention.csv"
  done
fi



echo "Generating URANIA-style paper tables and figures..."
python -m src.urania_report \
  --out "$OUT" \
  --baseline-dir "$BASELINES/cfpb_template" \
  --output-dir "$PAPER_TABLES" \
  --figure-dir "$PAPER_FIGS" \
  --main-report "$MAIN/reports/summary_table.csv" \
  --probe-report "$PROBE/reports/summary_table.csv" \
  --urania-probe "$BASELINES/urania_probe/exact_probe_mention.csv" \
  --probe-text "${PROBE_TEXT:-BLUE-LANTERN-791}" \
  2>&1 | tee "$LOGS/urania_report.log"

# Paired membership-style stability diagnostic
echo "Running membership-style probe..."
python -m src.membership_attack \
  --config configs/auto/cfpb_membership.yaml \
  --trials 20 \
  --output "$MEMBERSHIP/reports/membership_attack.json" \
  2>&1 | tee "$LOGS/membership_attack.log"

# Optional OpenAI verbalization and judging subset
if [[ "$USE_OPENAI" == "1" ]]; then
  echo "Running selected OpenAI DP-SPIN summaries and baselines..."
  python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(open("configs/auto/cfpb_main.yaml"))
cfg.update({
    "experiment_name": "cfpb_openai_dpspin",
    "output_dir": "$OPENAI_OUT/runs",
    "decoder_backend": "openai",
    "decoder_model": "$OPENAI_MODEL",
    "decoder_temperature": 0.0,
    "n_records": 2000,
    "m": 200,
    "K": 3,
    "L": 10,
    "public_denominator": 2000,
})
Path("configs/auto").mkdir(parents=True, exist_ok=True)
yaml.safe_dump(cfg, open("configs/auto/cfpb_openai_dpspin.yaml", "w"), sort_keys=False)
print(open("configs/auto/cfpb_openai_dpspin.yaml").read())
PY

  python -m src.run_grid \
    --config configs/auto/cfpb_openai_dpspin.yaml \
    --epsilons 1 2 4 \
    --seeds 0 1 2 \
    --mechanisms "${MECHS[@]}" \
    --m-values 200 \
    --K-values 3 \
    --L-values 10 \
    --n-records-values 2000 \
    --atoms-template "$ATOM_DIR/cfpb_atoms_m{m}.json" \
    2>&1 | tee "$LOGS/openai_dpspin.log"

  python -m src.report --runs-dir "$OPENAI_OUT/runs" --output-dir "$OPENAI_OUT/reports"

  python -m src.baselines \
    --records "$TARGET_RECORDS" \
    --atoms "$ATOMS_M200" \
    --output-dir "$OPENAI_OUT/baselines/cfpb_openai" \
    --epsilon 1.0 \
    --seed 0 \
    --backend openai \
    --model "$OPENAI_MODEL" \
    --category-key Issue \
    --category-universe-records "$AUX_RECORDS" \
    2>&1 | tee "$LOGS/openai_baselines.log"

  python -m src.urania_baseline \
    --records "$TARGET_RECORDS" \
    --aux-records "$AUX_RECORDS" \
    --atoms "$ATOMS_M200" \
    --output-dir "$OPENAI_OUT/baselines/cfpb_openai" \
    --epsilon-total "$URANIA_EPS_TOTAL" \
    --delta "$URANIA_DELTA" \
    --num-clusters "$URANIA_CLUSTERS" \
    --keywords-per-summary "$URANIA_KEYWORDS" \
    --max-keywords "$URANIA_MAX_KEYWORDS" \
    --max-keywords-per-record "$URANIA_KEYWORDS_PER_RECORD" \
    --svd-dim "$URANIA_SVD_DIM" \
    --dp-iterations "$URANIA_DP_ITERATIONS" \
    --seed 0 \
    --backend openai \
    --model "$OPENAI_MODEL" \
    2>&1 | tee "$LOGS/openai_urania_baseline.log"

  python -m src.baseline_report \
    --baseline-dir "$OPENAI_OUT/baselines/cfpb_openai" \
    --output-dir "$OPENAI_OUT/baselines/reports"

  echo "Running OpenAI judge for DP-SPIN summaries..."
  for RUN in "$OPENAI_OUT"/runs/*; do
    if [[ -f "$RUN/private_plan.json" && -f "$RUN/summary.txt" ]]; then
      python -m src.judge_summary \
        --plan "$RUN/private_plan.json" \
        --summary "$RUN/summary.txt" \
        --output "$RUN/judge_openai.json" \
        --backend openai \
        --model "$OPENAI_MODEL"
    fi
  done

  echo "Running generic OpenAI rubric judge for DP-SPIN and baselines..."
  python - <<'PY'
import json, os
from pathlib import Path
from openai import OpenAI
client = OpenAI()
model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
base = Path("outputs/cfpb/openai")
items = []
for run in (base / "runs").glob("*"):
    summ = run / "summary.txt"
    plan = run / "private_plan.json"
    if summ.exists():
        items.append({"method": "DP-SPIN", "id": run.name, "summary": summ.read_text(), "plan": plan.read_text() if plan.exists() else ""})
for p in (base / "baselines" / "cfpb_openai").glob("*_summary.txt"):
    items.append({"method": p.stem.replace("_summary", ""), "id": p.stem, "summary": p.read_text(), "plan": ""})

def judge(item):
    prompt = f"""
Evaluate this aggregate CFPB complaint summary. Return strict JSON with integer scores from 1 to 5:
coverage, specificity, faithfulness, privacy_safety, clarity.
Also include a short rationale string.

Method: {item['method']}
Plan if available:
{item['plan'][:4000]}

Summary:
{item['summary'][:4000]}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "Return only valid JSON."}, {"role": "user", "content": prompt}],
        temperature=0,
    )
    txt = resp.choices[0].message.content or "{}"
    try:
        out = json.loads(txt)
    except Exception:
        out = {"raw_response": txt}
    out["method"] = item["method"]
    out["id"] = item["id"]
    return out

results = [judge(x) for x in items]
outdir = base / "judgments"
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "judgments.json").write_text(json.dumps(results, indent=2))
print("Wrote", outdir / "judgments.json")
PY
fi

# Diagnostics and manuscript assets
export OUT MAIN HARD ABLATION PROBE BASELINES MEMBERSHIP OPENAI_OUT PAPER_FIGS PAPER_TABLES PAPER_ASSETS PAPER_ASSET_FIGS PAPER_ASSET_TABLES PAPER_RAW_TABLES PAPER_DIAGNOSTICS PAPER_TEXT

echo "Creating organized CFPB diagnostics and paper figures from saved metrics..."
python -m src.cfpb_report_figures \
  --out "$OUT" \
  --epsilon 0.5 \
  --K 3 \
  --L 10 \
  --m 200 \
  --write-block-plots \
  2>&1 | tee "$LOGS/cfpb_report_figures.log"

echo "Collecting canonical paper assets..."
python -m src.paper_assets --out "$OUT" 2>&1 | tee "$LOGS/paper_assets.log"

# Result index
cat > "$OUT/README_RESULTS.txt" <<EOF
DP-SPIN CFPB v2 experiment results

Main utility:
  $MAIN/metrics_summary.csv
  $MAIN/plots/utility_vs_epsilon.png
  $MAIN/diagnostics/topL_margin_diagnostics.csv

Hard utility:
  $HARD/metrics_summary.csv
  $HARD/plots/utility_vs_epsilon.png
  $PAPER_FIGS/fig3_hard_support_mass_loss.png

Ablation:
  $ABLATION/metrics_summary.csv
  $PAPER_FIGS/fig4_dictionary_size_support_loss.png
  $PAPER_FIGS/fig5_KL_support_loss_heatmap.png

Probe:
  $PROBE/metrics_summary.csv
  $PROBE/diagnostics/probe_rank_diagnostics.csv
  $PAPER_FIGS/fig6_probe_admission_vs_support.png
  $PAPER_FIGS/fig7_probe_non_private_rank_vs_support.png
  $BASELINES/probe/exact_probe_mention.csv

Baselines:
  $BASELINES/cfpb_template
  $BASELINES/reports/baseline_methods_table.tex
  $BASELINES/cfpb_template/urania_cluster_public_keyword_plan.json
  $BASELINES/cfpb_template/urania_cluster_public_keyword_summary.txt

URANIA-style probe diagnostics:
  $BASELINES/urania_probe/exact_probe_mention.csv
  $PAPER_TABLES/urania_quantitative_table.tex
  $PAPER_TABLES/urania_dpspin_comparison_table.tex
  $PAPER_TABLES/urania_qualitative_table.tex
  $PAPER_TABLES/urania_probe_comparison_table.tex
  $PAPER_FIGS/fig8_urania_probe_comparison.png

Membership:
  $MEMBERSHIP/reports/membership_attack.json

OpenAI optional outputs:
  $OPENAI_OUT/runs
  $OPENAI_OUT/baselines
  $OPENAI_OUT/judgments/judgments.json

Intermediate paper figures:
  $PAPER_FIGS

Intermediate paper tables/raw diagnostics:
  $PAPER_TABLES

Canonical paper assets for LaTeX:
  Figures: $PAPER_ASSET_FIGS
  Tables:  $PAPER_ASSET_TABLES
  Raw:     $PAPER_RAW_TABLES
  Manifest: $PAPER_ASSETS/MANIFEST.md
EOF

echo "============================================================"
echo "DONE. Final outputs are in: $OUT"
echo "Read: cat $OUT/README_RESULTS.txt"
echo "Canonical figures: ls $PAPER_ASSET_FIGS"
echo "Canonical tables: ls $PAPER_ASSET_TABLES"
echo "Manifest: cat $PAPER_ASSETS/MANIFEST.md"
echo "============================================================"
