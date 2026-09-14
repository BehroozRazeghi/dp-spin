"""Controlled ground-truth benchmark for semantic assignment, clipping, and DP release.

The benchmark creates an oracle semantic contribution matrix, a misspecified
assignment matrix, repeated records per user, deterministic radial clipping, and
a DP release.  It therefore separates assignment/dictionary error, clipping
error, and DP release error against known ground truth.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .dp_mechanisms import release, topL
from .utility_decomposition import radial_l1_clipping_distortion, user_level_decomposition
from .utils import load_yaml, write_json


def generate_semantic_data(
    n: int = 1000,
    m: int = 50,
    K: int = 3,
    L: int = 10,
    gap: float = 0.05,
    users: int = 250,
    misspecification: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Generate oracle and misspecified per-record semantic contributions."""
    if not (1 <= K <= m and 1 <= L < m):
        raise ValueError("require 1 <= K <= m and 1 <= L < m")
    if n < 1 or users < 1:
        raise ValueError("n and users must be positive")
    if gap < 0 or not 0 <= misspecification <= 1:
        raise ValueError("gap must be nonnegative and misspecification must be in [0,1]")
    rng = np.random.default_rng(seed)

    # Decreasing support profile with an explicitly controlled boundary gap.
    base = np.linspace(1.0, 0.1, m)
    boundary_lower = max(1e-6, float(base[L]))
    base[L - 1] = boundary_lower + float(gap)
    # Preserve monotonicity above and below the boundary.
    for j in range(L - 2, -1, -1):
        base[j] = max(base[j], base[j + 1] + 1e-6)
    for j in range(L + 1, m):
        base[j] = min(base[j], base[j - 1] - 1e-6)
        base[j] = max(base[j], 1e-6)
    probs = base / base.sum()

    oracle = np.zeros((n, m), dtype=float)
    assigned = np.zeros((n, m), dtype=float)
    user_ids = rng.integers(0, users, size=n)
    for i in range(n):
        idx = rng.choice(m, size=min(K, m), replace=False, p=probs)
        weights = rng.dirichlet(np.ones(len(idx)))
        oracle[i, idx] = weights
        if rng.random() >= misspecification:
            assigned[i, idx] = weights
        else:
            wrong = rng.choice(m, size=len(idx), replace=False)
            assigned[i, wrong] = weights
    return oracle, assigned, user_ids, {
        "requested_boundary_gap": float(gap),
        "sampling_weight_gap": float(base[L - 1] - base[L]),
    }


def _clip_users(V: np.ndarray, user_ids: np.ndarray, B: float) -> tuple[np.ndarray, np.ndarray]:
    users = sorted(set(map(int, user_ids)))
    U0 = np.vstack([V[user_ids == user].sum(axis=0) for user in users])
    mass = U0.sum(axis=1)
    scale = np.minimum(1.0, B / np.maximum(mass, 1e-12))
    return U0, U0 * scale[:, None]


def one_run(
    n: int = 1000,
    m: int = 50,
    K: int = 3,
    L: int = 10,
    epsilon: float = 2.0,
    delta: float = 1e-6,
    B: float = 2.0,
    gap: float = 0.05,
    misspecification: float = 0.1,
    mechanism: str = "gaussian",
    seed: int = 0,
    allocation: str = "uniform",
    n_users: int | None = None,
) -> dict[str, Any]:
    n_users = int(n_users if n_users is not None else max(2, n // 4))
    oracle, assigned, user_ids, generation = generate_semantic_data(
        n, m, K, L, gap, n_users, misspecification, seed
    )
    oracle_U0, _ = _clip_users(oracle, user_ids, B)
    assigned_U0, assigned_U = _clip_users(assigned, user_ids, B)
    C_oracle = oracle_U0.sum(axis=0)
    C_unclipped = assigned_U0.sum(axis=0)
    C_clipped = assigned_U.sum(axis=0)

    released_raw, info = release(
        C_clipped,
        mechanism,
        epsilon,
        delta,
        adjacency="adddrop",
        L=L,
        contribution_bound=B,
        rng=np.random.default_rng(seed + 1_000_003),
        selection_budget_allocation=allocation,
    )
    # Sparse private admission uses -inf internally for unselected coordinates.
    # For support selection retain -inf; for vector error, the released sparse
    # plan is embedded in R^m with zero at unselected coordinates.
    support_view = np.where(np.isfinite(released_raw), released_raw, -np.inf)
    released_dense = np.where(np.isfinite(released_raw), released_raw, 0.0)
    oracle_top = set(topL(C_oracle, L).tolist())
    released_top = set(topL(support_view, L).tolist())
    assigned_top = set(topL(C_clipped, L).tolist())
    decomp = user_level_decomposition(C_oracle, C_unclipped, C_clipped, released_dense, ord=1)
    clipping = radial_l1_clipping_distortion(assigned_U0, B)
    ordered = np.sort(C_clipped)[::-1]
    actual_gap = float(ordered[L - 1] - ordered[L]) if L < len(ordered) else None

    return {
        "n": n,
        "n_users": n_users,
        "m": m,
        "K": K,
        "L": L,
        "epsilon": epsilon,
        "delta": delta,
        "B": B,
        "requested_gap": gap,
        "realized_clipped_sketch_boundary_gap": actual_gap,
        "sampling_weight_gap": generation["sampling_weight_gap"],
        "misspecification": misspecification,
        "mechanism": mechanism,
        "allocation": allocation,
        "seed": seed,
        "oracle_to_released_topL_overlap": len(oracle_top & released_top) / max(1, L),
        "assignment_to_released_topL_overlap": len(assigned_top & released_top) / max(1, L),
        "assignment_l1_error": decomp["assignment_dictionary_error"],
        "clipping_l1_error": decomp["clipping_error"],
        "dp_l1_error": decomp["dp_release_error"],
        "total_l1_error": decomp["total_oracle_to_release_error"],
        "triangle_upper_bound": decomp["triangle_upper_bound"],
        "triangle_inequality_verified": decomp["triangle_inequality_verified"],
        "fraction_users_clipped": clipping["fraction_users_clipped"],
        "total_mass_removed": clipping["total_mass_removed"],
        "clipping_formula_verified": clipping["formula_verified"],
        "mechanism_info": info,
    }


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _default_grid() -> dict[str, list[Any]]:
    return {
        "n": [1000],
        "m": [50],
        "K": [3],
        "L": [10],
        "epsilon": [0.5, 1.0, 2.0, 4.0, 8.0],
        "B": [2.0],
        "gap": [0.05],
        "misspecification": [0.1],
        "mechanisms": ["gaussian", "laplace", "private_topL"],
        "selection_budget_allocations": ["theory_optimal"],
    }


def iter_grid(config: dict[str, Any], seeds: int) -> Iterable[dict[str, Any]]:
    grid = _default_grid()
    grid.update({key: _as_list(value) for key, value in config.items() if key in grid})
    for values in itertools.product(
        grid["n"], grid["m"], grid["K"], grid["L"], grid["epsilon"],
        grid["B"], grid["gap"], grid["misspecification"], grid["mechanisms"],
    ):
        n, m, K, L, epsilon, B, gap, misspecification, mechanism = values
        allocations = grid["selection_budget_allocations"] if mechanism == "private_topL" else ["not_applicable"]
        for allocation in allocations:
            for seed in range(seeds):
                yield {
                    "n": int(n), "m": int(m), "K": int(K), "L": int(L),
                    "epsilon": float(epsilon), "B": float(B), "gap": float(gap),
                    "misspecification": float(misspecification), "mechanism": str(mechanism),
                    "allocation": "uniform" if allocation == "not_applicable" else str(allocation),
                    "seed": seed,
                    "delta": float(config.get("delta", 1e-6)),
                    "n_users": config.get("n_users"),
                }


def run_grid(
    output: str,
    seeds: int = 5,
    config: dict[str, Any] | None = None,
    max_runs: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run_index, kwargs in enumerate(iter_grid(config or {}, seeds)):
        if max_runs is not None and run_index >= max_runs:
            break
        rows.append(one_run(**kwargs))
    if not rows:
        raise ValueError("synthetic grid produced no runs")

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [{key: value for key, value in row.items() if key != "mechanism_info"} for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)

    summary: dict[str, Any] = {"n_runs": len(flat), "groups": {}}
    group_keys = sorted({(row["mechanism"], row["allocation"]) for row in flat})
    for mechanism, allocation in group_keys:
        subset = [row for row in flat if row["mechanism"] == mechanism and row["allocation"] == allocation]
        name = f"{mechanism}:{allocation}"
        summary["groups"][name] = {
            "runs": len(subset),
            "mean_oracle_topL_overlap": float(np.mean([row["oracle_to_released_topL_overlap"] for row in subset])),
            "mean_assignment_topL_overlap": float(np.mean([row["assignment_to_released_topL_overlap"] for row in subset])),
            "mean_assignment_l1_error": float(np.mean([row["assignment_l1_error"] for row in subset])),
            "mean_clipping_l1_error": float(np.mean([row["clipping_l1_error"] for row in subset])),
            "mean_dp_l1_error": float(np.mean([row["dp_l1_error"] for row in subset])),
            "all_triangle_inequalities_verified": bool(all(row["triangle_inequality_verified"] for row in subset)),
            "all_clipping_formulas_verified": bool(all(row["clipping_formula_verified"] for row in subset)),
        }
    write_json(summary, path.with_suffix(".summary.json"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/synthetic_benchmark.csv")
    parser.add_argument("--config")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()
    config = load_yaml(args.config) if args.config else {}
    print(json.dumps(run_grid(args.output, args.seeds, config, args.max_runs), indent=2))


if __name__ == "__main__":
    main()
