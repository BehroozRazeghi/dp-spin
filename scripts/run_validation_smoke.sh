#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
python -m unittest discover -s tests -v
python -m src.generate_demo_data \
  --records data/processed/demo_records.jsonl \
  --atoms data/public_atoms/demo_atoms.json \
  --n 800 --n-users 160 --seed 0
python -m src.run_experiment \
  --config configs/validation_demo.yaml \
  --n-records 200 --public-denominator 200 \
  --epsilon 1.0 --seed 1
python -m src.verifier_benchmark --n 1000 --seed 0 --output outputs/validation/verifier_benchmark.json
python -m src.synthetic_benchmark --seeds 3 --output outputs/validation/synthetic_benchmark.csv
