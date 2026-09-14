"""Blinded common-ground human evaluation for aggregate summaries.

This module prepares anonymized evaluation tasks and aggregates completed
annotations.  It never sends protected records to a decoder.  References used
for evaluation are offline artifacts and must remain outside ``release_bundle``.

Input summary manifest JSONL fields
-----------------------------------
``dataset``, ``target_sample_id``, ``method``, ``summary`` and optionally
``reference``.  Rows sharing ``dataset`` and ``target_sample_id`` are compared
under the same common-ground reference.

Completed annotation JSONL fields
---------------------------------
``task_id``, ``annotator_id``, and integer scores from 1 to 5 for
``factual_support``, ``important_theme_coverage``, ``usefulness``, and
``clarity``.  ``unsupported_additions`` is scored from 1 (none) to 5 (many).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import cohen_kappa_score

from .utils import read_json, read_jsonl, write_json, write_jsonl

SCORE_FIELDS = (
    "factual_support",
    "important_theme_coverage",
    "unsupported_additions",
    "usefulness",
    "clarity",
)


def _stable_task_id(dataset: str, target_sample_id: str, method: str, salt: str) -> str:
    raw = f"{salt}|{dataset}|{target_sample_id}|{method}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def create_blinded_batch(
    rows: Iterable[dict[str, Any]],
    output_tasks: str | Path,
    output_key: str | Path,
    *,
    seed: int = 0,
    include_reference: bool = True,
) -> dict[str, Any]:
    """Create randomized blinded tasks and a separate offline method key."""
    rows = list(rows)
    if not rows:
        raise ValueError("summary manifest is empty")
    rng = np.random.default_rng(seed)
    salt = hashlib.sha256(f"dpspin-human-eval-{seed}".encode()).hexdigest()[:16]
    order = rng.permutation(len(rows))
    tasks: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for display_index, source_index in enumerate(order):
        row = rows[int(source_index)]
        required = ("dataset", "target_sample_id", "method", "summary")
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(f"manifest row {source_index} missing {missing}")
        task_id = _stable_task_id(
            str(row["dataset"]), str(row["target_sample_id"]), str(row["method"]), salt
        )
        task = {
            "task_id": task_id,
            "dataset": str(row["dataset"]),
            "target_sample_id": str(row["target_sample_id"]),
            "candidate_summary": str(row["summary"]),
            "reference": str(row.get("reference", "")) if include_reference else "",
            "instructions": {
                "scale": "1 (lowest) to 5 (highest), except unsupported_additions where 1 means none and 5 means many",
                "fields": list(SCORE_FIELDS),
                "do_not_infer_method": True,
            },
            "annotator_id": "",
            **{field: None for field in SCORE_FIELDS},
            "notes": "",
        }
        tasks.append(task)
        key.append(
            {
                "task_id": task_id,
                "display_index": display_index,
                "dataset": str(row["dataset"]),
                "target_sample_id": str(row["target_sample_id"]),
                "method": str(row["method"]),
                "source_metadata": row.get("metadata", {}),
            }
        )
    write_jsonl(tasks, output_tasks)
    write_jsonl(key, output_key)
    return {
        "n_tasks": len(tasks),
        "seed": seed,
        "tasks_path": str(output_tasks),
        "offline_key_path": str(output_key),
        "warning": "Keep the method key hidden from annotators and outside deployment release artifacts.",
    }


def _bootstrap_ci(values: list[float], *, seed: int = 0, draws: int = 5000) -> dict[str, float | int | None]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {"n": 0, "mean": None, "median": None, "ci95_lower": None, "ci95_upper": None}
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(draws, x.size), replace=True).mean(axis=1)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
    }


def _validate_score(value: Any, field: str) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer from 1 to 5") from exc
    if score < 1 or score > 5:
        raise ValueError(f"{field} must be an integer from 1 to 5")
    return score


def _pairwise_weighted_kappa(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    by_task: dict[str, dict[str, int]] = defaultdict(dict)
    for row in rows:
        by_task[str(row["task_id"])][str(row["annotator_id"])] = _validate_score(row[field], field)
    annotators = sorted({ann for task in by_task.values() for ann in task})
    pair_values: list[dict[str, Any]] = []
    for i, a in enumerate(annotators):
        for b in annotators[i + 1 :]:
            paired = [(task[a], task[b]) for task in by_task.values() if a in task and b in task]
            if len(paired) < 2:
                continue
            left, right = zip(*paired)
            if len(set(left)) == 1 and len(set(right)) == 1:
                kappa = 1.0 if left[0] == right[0] else 0.0
            else:
                kappa = float(cohen_kappa_score(left, right, weights="quadratic", labels=[1, 2, 3, 4, 5]))
            pair_values.append({"annotator_a": a, "annotator_b": b, "n_common": len(paired), "quadratic_weighted_kappa": kappa})
    finite = [p["quadratic_weighted_kappa"] for p in pair_values if np.isfinite(p["quadratic_weighted_kappa"])]
    return {
        "mean_pairwise_quadratic_weighted_kappa": float(np.mean(finite)) if finite else None,
        "pairs": pair_values,
    }


def evaluate_annotations(
    annotations: Iterable[dict[str, Any]],
    method_key: Iterable[dict[str, Any]],
    *,
    minimum_annotators_per_task: int = 2,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Aggregate blinded annotations by method with uncertainty and agreement."""
    annotations = list(annotations)
    key_rows = list(method_key)
    method_by_task = {str(row["task_id"]): str(row["method"]) for row in key_rows}
    if not annotations:
        raise ValueError("annotations are empty")
    grouped_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        task_id = str(row.get("task_id", ""))
        if task_id not in method_by_task:
            raise ValueError(f"unknown task_id={task_id!r}")
        if not str(row.get("annotator_id", "")).strip():
            raise ValueError(f"task_id={task_id} has an empty annotator_id")
        normalized = dict(row)
        for field in SCORE_FIELDS:
            normalized[field] = _validate_score(row.get(field), field)
        grouped_task[task_id].append(normalized)

    retained = {task_id: group for task_id, group in grouped_task.items() if len({r['annotator_id'] for r in group}) >= minimum_annotators_per_task}
    if not retained:
        raise ValueError("no tasks meet minimum_annotators_per_task")

    task_means: list[dict[str, Any]] = []
    for task_id, group in retained.items():
        row = {"task_id": task_id, "method": method_by_task[task_id], "n_annotators": len(group)}
        for field in SCORE_FIELDS:
            row[field] = float(np.mean([r[field] for r in group]))
        task_means.append(row)

    methods = sorted({row["method"] for row in task_means})
    method_summary: dict[str, Any] = {}
    for method_index, method in enumerate(methods):
        subset = [row for row in task_means if row["method"] == method]
        method_summary[method] = {
            field: _bootstrap_ci([row[field] for row in subset], seed=bootstrap_seed + 100 * method_index + field_index)
            for field_index, field in enumerate(SCORE_FIELDS)
        }
        method_summary[method]["n_tasks"] = len(subset)

    all_retained_rows = [row for group in retained.values() for row in group]
    agreement = {field: _pairwise_weighted_kappa(all_retained_rows, field) for field in SCORE_FIELDS}
    return {
        "design": {
            "minimum_annotators_per_task": minimum_annotators_per_task,
            "n_annotation_rows": len(annotations),
            "n_tasks_retained": len(retained),
            "n_tasks_dropped": len(grouped_task) - len(retained),
            "score_fields": list(SCORE_FIELDS),
            "unsupported_additions_direction": "lower is better",
        },
        "method_summary": method_summary,
        "agreement": agreement,
        "task_means": task_means,
        "interpretation": "Common-ground human evaluation; release-conditioned fidelity should be reported separately.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--manifest", required=True)
    create.add_argument("--tasks", required=True)
    create.add_argument("--key", required=True)
    create.add_argument("--seed", type=int, default=0)
    create.add_argument("--omit-reference", action="store_true")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--annotations", required=True)
    evaluate.add_argument("--key", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--minimum-annotators", type=int, default=2)
    evaluate.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    if args.command == "create":
        report = create_blinded_batch(
            read_jsonl(args.manifest), args.tasks, args.key,
            seed=args.seed, include_reference=not args.omit_reference,
        )
        print(json.dumps(report, indent=2))
    else:
        report = evaluate_annotations(
            read_jsonl(args.annotations), read_jsonl(args.key),
            minimum_annotators_per_task=args.minimum_annotators,
            bootstrap_seed=args.bootstrap_seed,
        )
        write_json(report, args.output)
        print(json.dumps(report["design"], indent=2))


if __name__ == "__main__":
    main()
