"""Repeated paired runs, bootstrap intervals, and variance decomposition.

The principal entry point is :func:`run_paired_repetitions`.  All methods are
validated against the same protected sample, public dictionary, privacy
parameters, contribution bound, plan size, and public denominator.  Data-sample
and DP-mechanism randomness are varied independently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .matched_protocol import require_matched_configs
from .run_experiment import run
from .utils import load_yaml, read_json, write_json


def bootstrap_ci(values: Iterable[float], confidence: float = 0.95, draws: int = 5000, seed: int = 0) -> dict[str, Any]:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": None, "median": None, "lower": None, "upper": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(x, size=(draws, x.size), replace=True), axis=1)
    alpha = (1 - confidence) / 2
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1 - alpha)),
        "n": int(x.size),
        "confidence": float(confidence),
        "bootstrap_draws": int(draws),
    }


def paired_bootstrap_difference(
    left: Iterable[float],
    right: Iterable[float],
    *,
    confidence: float = 0.95,
    draws: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap the paired mean difference ``left - right``."""
    a = np.asarray(list(left), dtype=float)
    b = np.asarray(list(right), dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    out = bootstrap_ci(d, confidence=confidence, draws=draws, seed=seed)
    out["direction"] = "left_minus_right"
    out["fraction_positive"] = float(np.mean(d > 0)) if d.size else None
    out["fraction_negative"] = float(np.mean(d < 0)) if d.size else None
    return out


def _extract_metrics(run_dir: Path, metrics: tuple[str, ...]) -> dict[str, float | None]:
    path = run_dir / "offline_evaluation" / "metrics.json"
    if not path.exists():
        path = run_dir / "metrics.json"
    payload = read_json(path)
    sketch = payload.get("sketch", payload)
    return {metric: sketch.get(metric) for metric in metrics}


def _variance_decomposition(records: list[dict[str, Any]], metric: str, method: str) -> dict[str, float]:
    subset = [row for row in records if row["method"] == method and row.get(metric) is not None]
    data_trials = sorted({int(row["data_trial"]) for row in subset})
    by_data = []
    within = []
    for data_trial in data_trials:
        values = np.asarray([float(row[metric]) for row in subset if int(row["data_trial"]) == data_trial], dtype=float)
        if values.size:
            by_data.append(float(values.mean()))
        if values.size > 1:
            within.append(float(values.var(ddof=1)))
    return {
        "between_target_sample_variance": float(np.var(by_data, ddof=1)) if len(by_data) > 1 else 0.0,
        "mean_within_sample_dp_variance": float(np.mean(within)) if within else 0.0,
    }


def run_paired_repetitions(
    configs: list[dict[str, Any]],
    method_names: list[str] | None = None,
    *,
    data_trials: int = 10,
    mechanism_trials: int = 20,
    metrics: tuple[str, ...] = ("overlap_at_L", "mass_retained", "support_mass_loss"),
    bootstrap_draws: int = 5000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    """Run a paired repeated-trial design over multiple methods.

    Each method sees the same data seed at a given ``data_trial``.  The same
    mechanism-trial index is used to construct method-specific mechanism seeds,
    making the pairing explicit without assuming that different random
    mechanisms consume identical random streams.
    """
    if len(configs) < 2:
        raise ValueError("at least two method configurations are required")
    if data_trials < 1 or mechanism_trials < 1:
        raise ValueError("data_trials and mechanism_trials must be positive")
    require_matched_configs(configs)
    names = method_names or [str(cfg.get("method_name", cfg.get("mechanism", f"method_{i}"))) for i, cfg in enumerate(configs)]
    if len(names) != len(configs) or len(set(names)) != len(names):
        raise ValueError("method_names must be unique and match the number of configurations")

    base_data_seed = int(configs[0].get("data_seed", configs[0].get("seed", 0)))
    base_mechanism_seed = int(configs[0].get("mechanism_seed", 100003))
    records: list[dict[str, Any]] = []

    for data_trial in range(data_trials):
        data_seed = base_data_seed + data_trial
        for mechanism_trial in range(mechanism_trials):
            for method_index, (name, base_cfg) in enumerate(zip(names, configs)):
                cfg = dict(base_cfg)
                cfg["data_seed"] = data_seed
                cfg["mechanism_seed"] = base_mechanism_seed + mechanism_trial + method_index * 1_000_003
                cfg["decoder_seed"] = int(base_cfg.get("decoder_seed", 200003)) + mechanism_trial + method_index * 1_000_003
                cfg["judge_seed"] = int(base_cfg.get("judge_seed", 300007)) + mechanism_trial + method_index * 1_000_003
                cfg["seed"] = data_trial * 100_000 + mechanism_trial
                root = Path(str(base_cfg.get("output_dir", "outputs/repeated")))
                cfg["output_dir"] = str(root / "paired" / name / f"data_{data_trial:03d}")
                run_dir = run(cfg)
                metric_values = _extract_metrics(Path(run_dir), metrics)
                records.append(
                    {
                        "method": name,
                        "data_trial": data_trial,
                        "mechanism_trial": mechanism_trial,
                        "data_seed": data_seed,
                        "mechanism_seed": cfg["mechanism_seed"],
                        "run_dir": str(run_dir),
                        **metric_values,
                    }
                )

    method_summary: dict[str, Any] = {}
    for method_index, name in enumerate(names):
        subset = [row for row in records if row["method"] == name]
        method_summary[name] = {}
        for metric_index, metric in enumerate(metrics):
            values = [row[metric] for row in subset if row.get(metric) is not None]
            summary = bootstrap_ci(values, draws=bootstrap_draws, seed=bootstrap_seed + 1000 * method_index + metric_index)
            summary.update(_variance_decomposition(records, metric, name))
            method_summary[name][metric] = summary

    paired_differences: dict[str, Any] = {}
    index = {
        (row["method"], int(row["data_trial"]), int(row["mechanism_trial"])): row
        for row in records
    }
    for left_index, left in enumerate(names):
        for right_index in range(left_index + 1, len(names)):
            right = names[right_index]
            pair_name = f"{left}__minus__{right}"
            paired_differences[pair_name] = {}
            for metric_index, metric in enumerate(metrics):
                left_values = []
                right_values = []
                for data_trial in range(data_trials):
                    for mechanism_trial in range(mechanism_trials):
                        lrow = index[(left, data_trial, mechanism_trial)]
                        rrow = index[(right, data_trial, mechanism_trial)]
                        if lrow.get(metric) is None or rrow.get(metric) is None:
                            continue
                        left_values.append(float(lrow[metric]))
                        right_values.append(float(rrow[metric]))
                paired_differences[pair_name][metric] = paired_bootstrap_difference(
                    left_values,
                    right_values,
                    draws=bootstrap_draws,
                    seed=bootstrap_seed + 10_000 * left_index + 100 * right_index + metric_index,
                )

    return {
        "design": {
            "data_trials": data_trials,
            "mechanism_trials": mechanism_trials,
            "paired_target_samples": True,
            "methods": names,
            "metrics": list(metrics),
            "bootstrap_draws": bootstrap_draws,
        },
        "method_summary": method_summary,
        "paired_differences": paired_differences,
        "records": records,
    }


def run_repetitions(
    base_cfg: dict[str, Any],
    data_trials: int = 10,
    mechanism_trials: int = 20,
    metrics: tuple[str, ...] = ("overlap_at_L", "mass_retained"),
) -> dict[str, Any]:
    """Backward-compatible single-method repeated-run helper."""
    records = []
    for data_trial in range(data_trials):
        data_seed = int(base_cfg.get("data_seed", 0)) + data_trial
        for mechanism_trial in range(mechanism_trials):
            cfg = dict(base_cfg)
            cfg["data_seed"] = data_seed
            cfg["mechanism_seed"] = int(base_cfg.get("mechanism_seed", 100003)) + mechanism_trial
            cfg["seed"] = data_trial * 100_000 + mechanism_trial
            cfg["output_dir"] = str(Path(base_cfg.get("output_dir", "outputs/repeated")) / f"d{data_trial}")
            run_dir = run(cfg)
            result = _extract_metrics(Path(run_dir), metrics)
            records.append({"data_trial": data_trial, "mechanism_trial": mechanism_trial, **result})
    summary = {metric: bootstrap_ci([row[metric] for row in records if row.get(metric) is not None], seed=17) for metric in metrics}
    for metric in metrics:
        method_records = [{"method": "single", **row} for row in records]
        summary[metric].update(_variance_decomposition(method_records, metric, "single"))
    return {"design": {"data_trials": data_trials, "mechanism_trials": mechanism_trials}, "summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--method-names", nargs="+")
    parser.add_argument("--data-trials", type=int, default=10)
    parser.add_argument("--mechanism-trials", type=int, default=20)
    parser.add_argument("--output", default="outputs/repeated_summary.json")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    configs = [load_yaml(path) for path in args.configs]
    if len(configs) == 1:
        result = run_repetitions(configs[0], args.data_trials, args.mechanism_trials)
        display = result["summary"]
    else:
        result = run_paired_repetitions(
            configs,
            args.method_names,
            data_trials=args.data_trials,
            mechanism_trials=args.mechanism_trials,
            bootstrap_draws=args.bootstrap_draws,
        )
        display = result["method_summary"]
    write_json(result, args.output)
    print(json.dumps(display, indent=2))


if __name__ == "__main__":
    main()
