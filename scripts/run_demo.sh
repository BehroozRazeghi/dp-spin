#!/usr/bin/env bash
set -euo pipefail

# Run the synthetic DP-SPIN demonstration without external datasets or APIs.
# The script creates record-level and user-level runs, a controlled-probe sweep,
# baseline outputs, plots, and reports under outputs/demo/.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

rm -rf outputs/demo outputs/baselines/demo

python -m src.generate_demo_data \
  --records data/processed/demo_records.jsonl \
  --atoms data/public_atoms/demo_atoms.json \
  --n 800 \
  --n-users 160 \
  --seed 0

python -m src.run_grid \
  --config configs/demo_record.yaml \
  --epsilons 0.5 1 2 4 8 \
  --seeds 0 1 2 \
  --mechanisms gaussian laplace private_topL \
  --m-values 15 \
  --K-values 3 \
  --L-values 5

python -m src.run_grid \
  --config configs/demo_user.yaml \
  --epsilons 0.5 1 2 4 \
  --seeds 0 1 \
  --mechanisms gaussian laplace private_topL \
  --m-values 15 \
  --K-values 3 \
  --L-values 5 \
  --n-users-values 160

python -m src.probe_sweep \
  --config configs/demo_record.yaml \
  --supports 1 5 20 \
  --epsilons 1 2 \
  --seeds 0 1 \
  --probe-text "The support ticket mentions the unique code HYPERION-X17." \
  --work-dir outputs/demo/probe_data

python -m src.baselines \
  --records data/processed/demo_records.jsonl \
  --atoms data/public_atoms/demo_atoms.json \
  --output-dir outputs/baselines/demo \
  --epsilon 1.0 \
  --seed 0

python -m src.plot_results \
  --runs-dir outputs/demo/record/runs \
  --csv outputs/demo/record/metrics_summary.csv \
  --plot outputs/demo/record/plots/utility_vs_epsilon.png \
  --probe-plot outputs/demo/record/plots/probe_admission.png

python -m src.report \
  --runs-dir outputs/demo/record/runs \
  --output-dir outputs/demo/record/reports

python -m src.plot_results \
  --runs-dir outputs/demo/user/runs \
  --csv outputs/demo/user/metrics_summary.csv \
  --plot outputs/demo/user/plots/utility_vs_epsilon.png \
  --probe-plot outputs/demo/user/plots/probe_admission.png

python -m src.report \
  --runs-dir outputs/demo/user/runs \
  --output-dir outputs/demo/user/reports

echo "Synthetic demonstration complete. Outputs are under outputs/demo/."
