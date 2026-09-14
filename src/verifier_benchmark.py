"""Synthetic labeled benchmark for the structured claim verifier."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .claims import verify_structured_claims
from .utils import write_json, write_jsonl


def example_plan(m: int = 8) -> dict:
    atoms = []
    for i in range(m):
        atoms.append({
            "atom_id": i,
            "label": f"Theme {i}",
            "rank": i + 1,
            "raw_noisy_mass": float(m - i),
            "noisy_mass": float(m - i),
            "support_bin": "high" if i < 2 else "moderate" if i < 5 else "low",
        })
    return {"selected_atoms": atoms}


def generate_cases(n: int = 1000, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    plan = example_plan()
    cases = []
    violation_types = ["unadmitted_atom", "wrong_bin", "wrong_rank", "wrong_comparison", "invalid_relation", "invalid_schema"]
    for i in range(n):
        valid = i % 2 == 0
        if valid:
            kind = ["theme", "support", "rank", "comparison"][i % 4]
            a = int(rng.integers(0, 8))
            if kind == "theme": claim = {"claim_type": "theme", "atom_id": a}
            elif kind == "support": claim = {"claim_type": "support", "atom_id": a, "bin": plan["selected_atoms"][a]["support_bin"]}
            elif kind == "rank": claim = {"claim_type": "rank", "atom_id": a, "rank": a + 1}
            else:
                l, r = sorted(rng.choice(8, size=2, replace=False))
                claim = {"claim_type": "comparison", "left_atom_id": int(l), "right_atom_id": int(r), "relation": "higher_support_than"}
            label = "valid"
        else:
            label = violation_types[(i // 2) % len(violation_types)]
            if label == "unadmitted_atom": claim = {"claim_type": "theme", "atom_id": 999}
            elif label == "wrong_bin": claim = {"claim_type": "support", "atom_id": 0, "bin": "low"}
            elif label == "wrong_rank": claim = {"claim_type": "rank", "atom_id": 0, "rank": 2}
            elif label == "wrong_comparison": claim = {"claim_type": "comparison", "left_atom_id": 7, "right_atom_id": 0, "relation": "higher_support_than"}
            elif label == "invalid_relation": claim = {"claim_type": "comparison", "left_atom_id": 0, "right_atom_id": 1, "relation": "dominates"}
            else: claim = {"claim_type": "causal", "text": "unsupported"}
        cases.append({"case_id": i, "expected_valid": valid, "violation_type": label, "claims": [claim]})
    return cases


def evaluate(cases: list[dict], plan: dict | None = None) -> dict:
    plan = example_plan() if plan is None else plan
    counts = defaultdict(lambda: {"n": 0, "accepted": 0})
    false_accept = false_reject = valid_n = invalid_n = 0
    rows = []
    for case in cases:
        result = verify_structured_claims(case["claims"], plan)
        accepted = bool(result["accept"])
        expected = bool(case["expected_valid"])
        key = str(case["violation_type"])
        counts[key]["n"] += 1
        counts[key]["accepted"] += int(accepted)
        valid_n += int(expected); invalid_n += int(not expected)
        false_accept += int((not expected) and accepted)
        false_reject += int(expected and not accepted)
        rows.append({**case, "accepted": accepted, "violations": result["violations"]})
    return {
        "n_cases": len(cases),
        "valid_cases": valid_n,
        "invalid_cases": invalid_n,
        "false_accept_count": false_accept,
        "false_reject_count": false_reject,
        "false_accept_rate": false_accept / max(1, invalid_n),
        "false_reject_rate": false_reject / max(1, valid_n),
        "by_type": {k: {**v, "accept_rate": v["accepted"] / max(1, v["n"])} for k, v in sorted(counts.items())},
        "scope": "Finite structured claim language only; does not assess semantic correctness relative to raw records.",
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/verifier_benchmark.json")
    args = parser.parse_args()
    result = evaluate(generate_cases(args.n, args.seed))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_json({k: v for k, v in result.items() if k != "cases"}, args.output)
    write_jsonl(result["cases"], str(Path(args.output).with_suffix(".cases.jsonl")))
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
