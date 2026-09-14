"""Machine-readable map from theoretical statements to experimental variables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .utils import write_json

THEOREM_EXPERIMENT_MAP: list[dict[str, Any]] = [
    {
        "statement_id": "record_adddrop_sensitivity",
        "manuscript_object": "Lemma: record-level add/drop sensitivity",
        "tested_variables": ["K", "record_l1_mass_bound"],
        "experiment": "property tests on adjacent assignment matrices",
        "observable": "max l1/l2 sketch difference",
        "expected_relation": "l1 <= 1 and l2 <= 1",
    },
    {
        "statement_id": "user_clipping_distortion",
        "manuscript_object": "Proposition: exact radial l1 clipping distortion",
        "tested_variables": ["B", "records_per_user", "user_mass_distribution"],
        "experiment": "controlled user-level benchmark",
        "observable": "aggregate clipping l1 error",
        "expected_relation": "equals sum_u (||U_u||_1-B)_+",
    },
    {
        "statement_id": "full_sketch_noise",
        "manuscript_object": "Gaussian/Laplace coordinate and top-L utility bounds",
        "tested_variables": ["epsilon", "delta", "m", "L", "boundary_gap", "n"],
        "experiment": "controlled ground-truth benchmark",
        "observable": "coordinate error, overlap@L, support-mass loss",
        "expected_relation": "error decreases with epsilon and support recovery improves with boundary gap",
    },
    {
        "statement_id": "sequential_selection_regret",
        "manuscript_object": "Theorem: sequential exponential-mechanism support regret",
        "tested_variables": ["m", "L", "epsilon_selection", "allocation", "boundary_gap", "beta"],
        "experiment": "sequential-admission synthetic sweep",
        "observable": "selected-support regret and exact top-L recovery",
        "expected_relation": "empirical regret <= sum alpha_t at stated coverage; exact recovery above sufficient gap",
    },
    {
        "statement_id": "semantic_abstention",
        "manuscript_object": "Abstaining semantic assignment model",
        "tested_variables": ["embedding_family", "similarity_threshold", "misspecification_fraction", "transfer_type"],
        "experiment": "external-validity and misspecification stress tests",
        "observable": "coverage, selective risk, false assignment on uncovered records",
        "expected_relation": "abstention lowers false assignment under misspecification at a measurable coverage cost",
    },
    {
        "statement_id": "finite_claim_verifier",
        "manuscript_object": "Structured-claim support guarantee",
        "tested_variables": ["claim_type", "admission_status", "rank", "support_bin", "comparison_margin"],
        "experiment": "balanced schema-level verifier benchmark",
        "observable": "false-accept and false-reject rates",
        "expected_relation": "zero false accepts for enumerated finite-schema violations in deterministic tests",
    },
    {
        "statement_id": "release_scaling",
        "manuscript_object": "Computational complexity discussion",
        "tested_variables": ["m", "L", "mechanism"],
        "experiment": "runtime and tracemalloc scaling benchmark",
        "observable": "wall-clock time and peak Python-tracked memory",
        "expected_relation": "full sketch scales approximately linearly in m; transparent sequential admission scales approximately with L*m",
    },
]


def write_csv(path: str | Path) -> None:
    rows = THEOREM_EXPERIMENT_MAP
    fieldnames = ["statement_id", "manuscript_object", "tested_variables", "experiment", "observable", "expected_relation"]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "tested_variables": "; ".join(row["tested_variables"])})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="outputs/theory_experiment_map.json")
    parser.add_argument("--csv", default="outputs/theory_experiment_map.csv")
    args = parser.parse_args()
    write_json({"rows": THEOREM_EXPERIMENT_MAP}, args.json)
    write_csv(args.csv)
    print(json.dumps({"json": args.json, "csv": args.csv, "n_rows": len(THEOREM_EXPERIMENT_MAP)}, indent=2))


if __name__ == "__main__":
    main()
