"""Runtime and memory scaling benchmark for DP-SPIN plan release.

The benchmark isolates release-time cost from semantic assignment.  It compares
full-sketch Gaussian/Laplace release with sequential private atom admission as
the public dictionary size ``m`` grows.  Results include wall-clock time,
tracemalloc peak memory, output size, and explicit asymptotic-operation labels.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from .dp_mechanisms import normalize_mechanism_name, release, topL
from .utils import load_yaml, write_json


def synthetic_sketch(m: int, total_mass: float, exponent: float, seed: int) -> np.ndarray:
    if m < 1:
        raise ValueError("m must be positive")
    if total_mass <= 0:
        raise ValueError("total_mass must be positive")
    ranks = np.arange(1, m + 1, dtype=float)
    values = ranks ** (-float(exponent))
    values /= values.sum()
    rng = np.random.default_rng(seed)
    # A public random permutation prevents implementation artifacts tied to
    # sorted input coordinates while preserving the controlled mass profile.
    values = values[rng.permutation(m)]
    return values * float(total_mass)


def _complexity_label(mechanism: str, m: int, L: int) -> dict[str, Any]:
    mech = normalize_mechanism_name(mechanism)
    if mech in {"gaussian", "laplace"}:
        return {
            "time_complexity": "O(m log m) in the current deterministic full-sort Top-L implementation",
            "working_memory_complexity": "O(m)",
            "released_coordinate_count": int(min(L, m)),
            "internally_perturbed_coordinate_count": int(m),
        }
    return {
        "time_complexity": "O(L m) for transparent sequential exponential-mechanism admission",
        "working_memory_complexity": "O(m)",
        "released_coordinate_count": int(min(L, m)),
        "internally_perturbed_coordinate_count": int(min(L, m)),
    }


def benchmark_one(
    C: np.ndarray,
    *,
    mechanism: str,
    epsilon: float,
    delta: float,
    adjacency: str,
    L: int,
    value_epsilon_fraction: float,
    repeats: int,
    warmup: int,
    seed: int,
    selection_budget_allocation: str,
) -> dict[str, Any]:
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    mech = normalize_mechanism_name(mechanism)
    kwargs = dict(
        mechanism=mech,
        epsilon=float(epsilon),
        delta=float(delta),
        adjacency=str(adjacency),
        L=int(L),
        value_epsilon_fraction=float(value_epsilon_fraction),
        value_mechanism="gaussian",
        gaussian_calibration="analytic",
        contribution_bound=1.0,
        selection_budget_allocation=str(selection_budget_allocation),
        selection_bound_beta=0.05,
    )
    for i in range(warmup):
        release(C, rng=np.random.default_rng(seed + 10_000 + i), **kwargs)

    times: list[float] = []
    peaks: list[int] = []
    output_bytes: list[int] = []
    metadata: dict[str, Any] | None = None
    for i in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        released, metadata = release(C, rng=np.random.default_rng(seed + i), **kwargs)
        if mech in {"gaussian", "laplace"}:
            admitted = topL(released, L)
        else:
            admitted = np.flatnonzero(np.isfinite(released)).astype(int)
        admitted_values = np.asarray(released[admitted], dtype=np.float64)
        # Materialize the minimal numeric plan payload: int64 atom indices plus
        # float64 released values. Public labels/descriptions are excluded.
        payload_bytes = int(admitted.astype(np.int64).nbytes + admitted_values.nbytes)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(float(elapsed))
        peaks.append(int(peak))
        output_bytes.append(payload_bytes)

    q = np.quantile(np.asarray(times), [0.05, 0.5, 0.95])
    return {
        "mechanism": mech,
        "m": int(C.size),
        "L": int(min(L, C.size)),
        "L_configured": int(L),
        "repeats": int(repeats),
        "runtime_seconds_mean": float(statistics.mean(times)),
        "runtime_seconds_median": float(q[1]),
        "runtime_seconds_q05": float(q[0]),
        "runtime_seconds_q95": float(q[2]),
        "runtime_seconds_stdev": float(statistics.stdev(times)) if len(times) > 1 else 0.0,
        "peak_python_bytes_mean": float(statistics.mean(peaks)),
        "peak_python_bytes_max": int(max(peaks)),
        "released_numeric_plan_bytes_mean": float(statistics.mean(output_bytes)),
        "internal_return_array_bytes": int(np.asarray(C).nbytes),
        "mechanism_metadata": metadata,
        **_complexity_label(mech, int(C.size), int(L)),
    }


def _loglog_slope(rows: list[dict[str, Any]], field: str) -> float | None:
    usable = [row for row in rows if float(row.get(field, 0.0)) > 0 and int(row["m"]) > 0]
    if len(usable) < 2:
        return None
    x = np.log(np.asarray([int(row["m"]) for row in usable], dtype=float))
    y = np.log(np.asarray([float(row[field]) for row in usable], dtype=float))
    return float(np.polyfit(x, y, 1)[0])


def run_scaling_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    m_values = sorted(set(int(x) for x in config.get("m_values", [50, 100, 200, 500, 1000, 5000, 10000])))
    mechanisms = [normalize_mechanism_name(str(x)) for x in config.get("mechanisms", ["gaussian", "laplace", "private_topL"])]
    if "L_values" in config:
        L_values = sorted(set(int(x) for x in config["L_values"]))
    else:
        L_values = [int(config.get("L", 10))]
    if any(m < 1 for m in m_values) or any(L < 1 for L in L_values):
        raise ValueError("m_values and L_values must be positive")
    repeats = int(config.get("repeats", 20))
    warmup = int(config.get("warmup", 3))
    seed = int(config.get("seed", 0))
    rows = []
    for L in L_values:
        for m in m_values:
            C = synthetic_sketch(
                m,
                total_mass=float(config.get("total_mass", 5000.0)),
                exponent=float(config.get("zipf_exponent", 1.1)),
                seed=seed + 1000 * L + m,
            )
            for mechanism in mechanisms:
                rows.append(
                    benchmark_one(
                        C,
                        mechanism=mechanism,
                        epsilon=float(config.get("epsilon", 2.0)),
                        delta=float(config.get("delta", 1e-6)),
                        adjacency=str(config.get("adjacency", "adddrop")),
                        L=L,
                        value_epsilon_fraction=float(config.get("value_epsilon_fraction", 0.5)),
                        repeats=repeats,
                        warmup=warmup,
                        seed=seed + 100_000 * (mechanisms.index(mechanism) + 1) + 1000 * L + m,
                        selection_budget_allocation=str(config.get("selection_budget_allocation", "theory_optimal")),
                    )
                )
    scaling_summary: list[dict[str, Any]] = []
    for L in L_values:
        for mechanism in mechanisms:
            subset = [row for row in rows if row["mechanism"] == mechanism and int(row["L_configured"]) == int(L)]
            scaling_summary.append({
                "mechanism": mechanism,
                "configured_L": int(L),
                "runtime_loglog_slope_vs_m": _loglog_slope(subset, "runtime_seconds_median"),
                "python_peak_memory_loglog_slope_vs_m": _loglog_slope(subset, "peak_python_bytes_mean"),
                "n_dictionary_sizes": len(subset),
                "interpretation": "descriptive implementation scaling exponent, not an asymptotic proof",
            })
    return {
        "experiment": "plan_release_runtime_memory_scaling",
        "configuration": config,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "numpy_version": np.__version__,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "rows": rows,
        "empirical_scaling_summary": scaling_summary,
        "measurement_notes": {
            "runtime": "wall-clock release plus atom-admission/Top-L materialization measured with time.perf_counter after warmup",
            "memory": "peak Python-tracked allocation measured by tracemalloc; native BLAS/runtime and operating-system allocations may not be included",
            "payload": "minimal numeric plan payload counts int64 admitted indices and float64 released values; public labels/descriptions are excluded",
            "scope": "release-time benchmark only; semantic assignment and decoder latency are excluded",
            "topL_implementation": "full-sketch mechanisms use deterministic full sorting; reported O(m log m) matches the current code",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_scaling_benchmark(load_yaml(args.config))
    write_json(result, args.output)
    print(json.dumps({"output": args.output, "n_rows": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
