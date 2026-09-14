"""Human/weak-supervision evaluation for semantic assignment and abstention.

Annotation JSONL schema (one row per annotator-record pair):
``record_id``, ``annotator_id``, ``relevant_atom_ids`` (list), and ``covered``
(boolean indicating whether the public dictionary contains an adequate atom).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .utils import read_json, read_jsonl, write_json, write_jsonl


def create_annotation_batch(records: list[dict[str, Any]], atoms: list[dict[str, Any]], output: str | Path, limit: int = 500) -> None:
    public_atoms = [{"atom_id": int(a.get("atom_id", i)), "label": a.get("label"), "description": a.get("description", "")} for i, a in enumerate(atoms)]
    rows = []
    for i, rec in enumerate(records[:limit]):
        rows.append({
            "record_id": str(rec.get("id", rec.get("record_id", i))),
            "text": str(rec.get("text", rec.get("review_text", ""))),
            "public_atoms": public_atoms,
            "annotator_id": "",
            "relevant_atom_ids": [],
            "covered": None,
            "notes": "",
        })
    write_jsonl(rows, output)


def aggregate_annotations(rows: list[dict[str, Any]]) -> tuple[dict[str, set[int]], dict[str, bool], dict]:
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_record[str(row["record_id"])].append(row)
    relevant: dict[str, set[int]] = {}
    covered: dict[str, bool] = {}
    exact_agreements = []
    covered_agreements = []
    jaccards = []
    for rid, group in by_record.items():
        sets = [set(map(int, g.get("relevant_atom_ids", []))) for g in group]
        universe = set().union(*sets) if sets else set()
        votes = {a: sum(a in s for s in sets) for a in universe}
        threshold = max(1, int(np.ceil(len(sets) / 2)))
        relevant[rid] = {a for a, v in votes.items() if v >= threshold}
        cv = [bool(g.get("covered")) for g in group if g.get("covered") is not None]
        covered[rid] = bool(sum(cv) >= np.ceil(len(cv) / 2)) if cv else bool(relevant[rid])
        if len(group) >= 2:
            exact_agreements.append(float(all(s == sets[0] for s in sets[1:])))
            if cv:
                covered_agreements.append(float(all(x == cv[0] for x in cv[1:])))
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    union = sets[i] | sets[j]
                    jaccards.append(1.0 if not union else len(sets[i] & sets[j]) / len(union))
    agreement = {
        "records": len(by_record),
        "exact_label_set_agreement": float(np.mean(exact_agreements)) if exact_agreements else None,
        "covered_binary_agreement": float(np.mean(covered_agreements)) if covered_agreements else None,
        "pairwise_jaccard": float(np.mean(jaccards)) if jaccards else None,
    }
    return relevant, covered, agreement



def _assignment_outcomes(
    predictions: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return one evaluation row per annotated record.

    Keeping record-level outcomes explicit enables nonparametric confidence
    intervals and paired comparisons without treating repeated annotations as
    independent observations.
    """
    gold, covered, agreement = aggregate_annotations(annotations)
    pred = {str(r["record_id"]): list(map(int, r.get("predicted_atom_ids", []))) for r in predictions}
    ids = sorted(set(gold) & set(pred))
    rows: list[dict[str, Any]] = []
    for rid in ids:
        p = pred[rid]
        g = gold[rid]
        is_covered = bool(covered[rid])
        abstained = not p
        top1_correct = bool(p) and p[0] in g
        rows.append({
            "record_id": rid,
            "top1_correct": float(top1_correct),
            "topK_recall": 1.0 if not g else float(len(set(p) & g) / len(g)),
            "assigned": float(not abstained),
            "abstained": float(abstained),
            "covered": float(is_covered),
            "uncovered": float(not is_covered),
            "abstention_correct": float(abstained == (not is_covered)),
            "assigned_top1_correct": float(top1_correct) if p else np.nan,
            "covered_assigned_top1_correct": float(top1_correct) if p and is_covered else np.nan,
            "false_assignment_uncovered": float(bool(p)) if not is_covered else np.nan,
            "uncovered_detection_tp": float(abstained and not is_covered),
            "uncovered_detection_fp": float(abstained and is_covered),
            "uncovered_detection_fn": float((not abstained) and (not is_covered)),
        })
    return rows, agreement


def _nanmean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else None


def _metrics_from_outcomes(rows: list[dict[str, Any]], agreement: dict[str, Any] | None = None) -> dict[str, Any]:
    if not rows:
        return {"n_evaluated": 0}
    def arr(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=float)

    tp = float(arr("uncovered_detection_tp").sum())
    fp = float(arr("uncovered_detection_fp").sum())
    fn = float(arr("uncovered_detection_fn").sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    selective_accuracy = _nanmean(arr("assigned_top1_correct"))
    covered_selective_accuracy = _nanmean(arr("covered_assigned_top1_correct"))
    return {
        "n_evaluated": len(rows),
        "top1_precision": float(arr("top1_correct").mean()),
        "topK_recall": float(arr("topK_recall").mean()),
        "coverage_rate": float(arr("assigned").mean()),
        "abstention_rate": float(arr("abstained").mean()),
        "selective_top1_accuracy": selective_accuracy,
        "selective_risk": None if selective_accuracy is None else float(1.0 - selective_accuracy),
        "covered_selective_top1_accuracy": covered_selective_accuracy,
        "false_assignment_rate_on_uncovered": _nanmean(arr("false_assignment_uncovered")),
        "uncovered_ground_truth_rate": float(arr("uncovered").mean()),
        "abstention_accuracy": float(arr("abstention_correct").mean()),
        "uncovered_detection_precision": float(precision),
        "uncovered_detection_recall": float(recall),
        "uncovered_detection_f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "annotator_agreement": agreement,
        "interpretation": "These metrics evaluate semantic assignment/dictionary coverage, separately from DP plan fidelity.",
    }


def bootstrap_assignment_metrics(
    predictions: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Record-level nonparametric bootstrap intervals for assignment metrics."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    rows, agreement = _assignment_outcomes(predictions, annotations)
    if not rows:
        return {"n_bootstrap": n_bootstrap, "intervals": {}}
    base = _metrics_from_outcomes(rows, agreement)
    metric_names = [
        "top1_precision", "topK_recall", "coverage_rate", "selective_top1_accuracy",
        "selective_risk", "false_assignment_rate_on_uncovered",
        "uncovered_detection_f1", "abstention_accuracy",
    ]
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    n = len(rows)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        metrics = _metrics_from_outcomes([rows[int(i)] for i in indices])
        for name in metric_names:
            value = metrics.get(name)
            if value is not None and np.isfinite(float(value)):
                samples[name].append(float(value))
    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, Any] = {}
    for name in metric_names:
        values = np.asarray(samples[name], dtype=float)
        if not values.size:
            intervals[name] = None
            continue
        lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
        intervals[name] = {
            "estimate": base.get(name),
            "lower": float(lo),
            "upper": float(hi),
            "bootstrap_standard_error": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }
    return {
        "method": "record-level nonparametric percentile bootstrap",
        "n_records": len(rows),
        "n_bootstrap": int(n_bootstrap),
        "confidence": float(confidence),
        "seed": int(seed),
        "intervals": intervals,
    }


def paired_bootstrap_metric_difference(
    predictions_a: list[dict[str, Any]],
    predictions_b: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    *,
    metric: str,
    n_bootstrap: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired record bootstrap for metric(A)-metric(B)."""
    rows_a, _ = _assignment_outcomes(predictions_a, annotations)
    rows_b, _ = _assignment_outcomes(predictions_b, annotations)
    map_a = {row["record_id"]: row for row in rows_a}
    map_b = {row["record_id"]: row for row in rows_b}
    ids = sorted(set(map_a) & set(map_b))
    if not ids:
        raise ValueError("paired comparison has no shared annotated records")
    base_a = _metrics_from_outcomes([map_a[rid] for rid in ids]).get(metric)
    base_b = _metrics_from_outcomes([map_b[rid] for rid in ids]).get(metric)
    if base_a is None or base_b is None:
        raise ValueError(f"metric {metric!r} is undefined for the paired sample")
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    n = len(ids)
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, n, size=n)
        sample_ids = [ids[int(i)] for i in sampled]
        ma = _metrics_from_outcomes([map_a[rid] for rid in sample_ids]).get(metric)
        mb = _metrics_from_outcomes([map_b[rid] for rid in sample_ids]).get(metric)
        if ma is not None and mb is not None:
            diffs.append(float(ma) - float(mb))
    values = np.asarray(diffs, dtype=float)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return {
        "metric": metric,
        "difference_definition": "metric_A_minus_metric_B",
        "estimate": float(base_a) - float(base_b),
        "lower": float(lo),
        "upper": float(hi),
        "n_records": n,
        "n_bootstrap": int(n_bootstrap),
        "confidence": float(confidence),
        "seed": int(seed),
    }

def paired_bootstrap_metric_difference_two_annotations(
    predictions_a: list[dict[str, Any]],
    annotations_a: list[dict[str, Any]],
    predictions_b: list[dict[str, Any]],
    annotations_b: list[dict[str, Any]],
    *,
    metric: str,
    n_bootstrap: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired record bootstrap when the two dictionaries use compatible but distinct annotations.

    Target record identifiers are paired, while each condition is scored against
    the annotation set defined for its own public dictionary. This is useful for
    source-versus-matched dictionary transfer when atom identifiers differ. The
    comparison is meaningful only when both annotation protocols use the same
    semantic adequacy rubric on the same target records.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    rows_a, _ = _assignment_outcomes(predictions_a, annotations_a)
    rows_b, _ = _assignment_outcomes(predictions_b, annotations_b)
    map_a = {row["record_id"]: row for row in rows_a}
    map_b = {row["record_id"]: row for row in rows_b}
    ids = sorted(set(map_a) & set(map_b))
    if not ids:
        raise ValueError("paired comparison has no shared annotated target records")
    base_a = _metrics_from_outcomes([map_a[rid] for rid in ids]).get(metric)
    base_b = _metrics_from_outcomes([map_b[rid] for rid in ids]).get(metric)
    if base_a is None or base_b is None:
        raise ValueError(f"metric {metric!r} is undefined for the paired sample")
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    n = len(ids)
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, n, size=n)
        sample_ids = [ids[int(i)] for i in sampled]
        ma = _metrics_from_outcomes([map_a[rid] for rid in sample_ids]).get(metric)
        mb = _metrics_from_outcomes([map_b[rid] for rid in sample_ids]).get(metric)
        if ma is not None and mb is not None:
            diffs.append(float(ma) - float(mb))
    if not diffs:
        raise ValueError(f"metric {metric!r} was undefined in every bootstrap replicate")
    values = np.asarray(diffs, dtype=float)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return {
        "metric": metric,
        "difference_definition": "metric_A_minus_metric_B",
        "annotation_protocol": "condition-specific annotations paired by target record_id",
        "estimate": float(base_a) - float(base_b),
        "lower": float(lo),
        "upper": float(hi),
        "n_records": n,
        "n_bootstrap": int(n_bootstrap),
        "confidence": float(confidence),
        "seed": int(seed),
    }


def evaluate_assignments(predictions: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> dict:
    """Evaluate predicted ordered atom IDs and abstention against annotations."""
    rows, agreement = _assignment_outcomes(predictions, annotations)
    return _metrics_from_outcomes(rows, agreement)


def selective_decision_cost(
    predictions: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    *,
    error_cost: float = 1.0,
    abstention_cost: float = 0.25,
) -> dict[str, Any]:
    """Evaluate a transparent assignment-versus-abstention cost model.

    A correct top-1 assignment has zero cost, an incorrect or uncovered forced
    assignment has ``error_cost``, and abstention has ``abstention_cost``.
    Costs are analysis parameters and must be reported rather than hidden.
    """
    if error_cost < 0 or abstention_cost < 0:
        raise ValueError("costs must be nonnegative")
    gold, covered, _ = aggregate_annotations(annotations)
    pred = {str(r["record_id"]): list(map(int, r.get("predicted_atom_ids", []))) for r in predictions}
    ids = sorted(set(gold) & set(pred))
    costs: list[float] = []
    n_correct = n_incorrect = n_abstained = 0
    for rid in ids:
        p = pred[rid]
        if not p:
            costs.append(float(abstention_cost))
            n_abstained += 1
        elif covered[rid] and p[0] in gold[rid]:
            costs.append(0.0)
            n_correct += 1
        else:
            costs.append(float(error_cost))
            n_incorrect += 1
    return {
        "n": len(ids),
        "mean_decision_cost": float(np.mean(costs)) if costs else None,
        "total_decision_cost": float(np.sum(costs)) if costs else 0.0,
        "error_cost": float(error_cost),
        "abstention_cost": float(abstention_cost),
        "n_correct": int(n_correct),
        "n_incorrect": int(n_incorrect),
        "n_abstained": int(n_abstained),
    }


def calibrate_threshold(examples: list[dict[str, Any]], thresholds: list[float]) -> dict:
    """Select an auxiliary-data threshold maximizing covered/uncovered balanced accuracy."""
    scored = []
    y = np.asarray([bool(x["covered"]) for x in examples])
    scores = np.asarray([float(x["max_similarity"]) for x in examples])
    for tau in thresholds:
        pred = scores >= tau
        tpr = np.mean(pred[y]) if np.any(y) else 0.0
        tnr = np.mean(~pred[~y]) if np.any(~y) else 0.0
        scored.append({"threshold": float(tau), "balanced_accuracy": float((tpr + tnr) / 2), "covered_recall": float(tpr), "uncovered_recall": float(tnr)})
    best = max(scored, key=lambda r: (r["balanced_accuracy"], -r["threshold"]))
    return {"selected_threshold": best["threshold"], "criterion": "auxiliary balanced accuracy", "grid": scored}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create a record/atom annotation batch")
    create.add_argument("--records", required=True)
    create.add_argument("--atoms", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--limit", type=int, default=500)

    evaluate = sub.add_parser("evaluate", help="Evaluate assignment predictions against completed annotations")
    evaluate.add_argument("--annotations", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)

    calibrate = sub.add_parser("calibrate", help="Select an abstention threshold on auxiliary annotations")
    calibrate.add_argument("--examples", required=True, help="JSONL with covered and max_similarity fields")
    calibrate.add_argument("--thresholds", nargs="+", type=float, required=True)
    calibrate.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "create":
        create_annotation_batch(read_jsonl(args.records), read_json(args.atoms), args.output, args.limit)
    elif args.command == "evaluate":
        result = evaluate_assignments(read_jsonl(args.predictions), read_jsonl(args.annotations))
        write_json(result, args.output)
    else:
        result = calibrate_threshold(read_jsonl(args.examples), args.thresholds)
        write_json(result, args.output)


if __name__ == "__main__":
    main()
