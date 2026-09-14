"""External-validity experiments for DP-SPIN semantic assignment.

This module addresses three questions that plan-fidelity experiments cannot
answer:

1. Do conclusions hold across at least two fixed public embedding families?
2. Does a dictionary constructed in one domain or time period transfer to a
   different target domain/time period?
3. Under dictionary misspecification, when is abstention preferable to forced
   assignment?

All fitted assignment representations use only atom text and optional public or
disjoint auxiliary text.  Protected target records are never used to fit the
representation.  Human or weak-supervision annotations are evaluation-only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import sparse

from .assign import topK_sparse_assignments
from .assignment_eval import (
    aggregate_annotations,
    bootstrap_assignment_metrics,
    evaluate_assignments,
    paired_bootstrap_metric_difference,
    paired_bootstrap_metric_difference_two_annotations,
    selective_decision_cost,
)
from .embed import SUPPORTED_EMBEDDING_FAMILIES
from .utils import load_yaml, read_json, read_jsonl, text_field, write_json, write_jsonl


def _record_id(record: dict[str, Any], index: int) -> str:
    for key in ("id", "record_id", "complaint_id", "review_id"):
        if record.get(key) is not None:
            return str(record[key])
    payload = f"{index}|{text_field(record)[:500]}".encode("utf-8", errors="ignore")
    return "hash:" + hashlib.sha256(payload).hexdigest()[:20]


def _prediction_rows(
    records: list[dict[str, Any]],
    assignments: sparse.csr_matrix,
    atoms: list[dict[str, Any]],
    record_diagnostics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if assignments.shape != (len(records), len(atoms)):
        raise ValueError("assignment matrix shape mismatch")
    if len(record_diagnostics) != len(records):
        raise ValueError("record diagnostics length mismatch")
    rows: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        row = assignments.getrow(i)
        order = np.argsort(-row.data, kind="stable") if row.nnz else np.asarray([], dtype=int)
        indices = row.indices[order].astype(int).tolist() if row.nnz else []
        weights = row.data[order].astype(float).tolist() if row.nnz else []
        detail = record_diagnostics[i]
        rows.append(
            {
                "record_id": _record_id(record, i),
                "predicted_atom_ids": [int(atoms[j].get("atom_id", j)) for j in indices],
                "predicted_weights": weights,
                "abstained": not indices,
                "row_mass": float(sum(weights)),
                "max_similarity": float(detail["max_similarity"]),
                "best_atom_id_before_threshold": detail.get("best_atom_id"),
            }
        )
    return rows


def _public_fit_texts(path: str | None, max_records: int | None = None) -> list[str] | None:
    if not path:
        return None
    rows = read_jsonl(path)
    if max_records is not None:
        rows = rows[: int(max_records)]
    return [text_field(row) for row in rows]


def run_assignment_condition(
    *,
    records: list[dict[str, Any]],
    atoms: list[dict[str, Any]],
    annotations: list[dict[str, Any]] | None,
    public_fit_texts: list[str] | None,
    condition: dict[str, Any],
) -> dict[str, Any]:
    """Run one fixed assignment condition and evaluate it when annotations exist."""
    C, V, diagnostics, record_diagnostics = topK_sparse_assignments(
        records,
        atoms,
        K=int(condition.get("K", 3)),
        gamma=float(condition.get("gamma", 10.0)),
        public_fit_texts=public_fit_texts,
        allow_private_fit=False,
        similarity_threshold=(
            None if condition.get("similarity_threshold") is None
            else float(condition["similarity_threshold"])
        ),
        abstention_mode=str(condition.get("abstention_mode", "zero")),
        other_atom_id=condition.get("other_atom_id"),
        other_atom_label=condition.get("other_atom_label"),
        mass_mode=str(condition.get("mass_mode", "unit")),
        require_positive_similarity=bool(condition.get("require_positive_similarity", True)),
        return_diagnostics=True,
        return_record_diagnostics=True,
        embedding_family=str(condition.get("embedding_family", "tfidf_word")),
        embedding_min_df=int(condition.get("embedding_min_df", 1)),
        embedding_max_features=(
            None if condition.get("embedding_max_features") is None
            else int(condition.get("embedding_max_features", 50000))
        ),
        embedding_word_ngram_range=tuple(condition.get("embedding_word_ngram_range", [1, 2])),
        embedding_char_ngram_range=tuple(condition.get("embedding_char_ngram_range", [3, 5])),
        embedding_model_name=str(condition.get(
            "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )),
        embedding_model_revision=condition.get("embedding_model_revision"),
        embedding_model_batch_size=int(condition.get("embedding_model_batch_size", 64)),
        embedding_model_device=condition.get("embedding_model_device"),
        embedding_model_local_files_only=bool(
            condition.get("embedding_model_local_files_only", False)
        ),
        embedding_model_trust_remote_code=bool(
            condition.get("embedding_model_trust_remote_code", False)
        ),
        show_progress=bool(condition.get("show_progress", False)),
    )
    predictions = _prediction_rows(records, V, atoms, record_diagnostics)
    evaluation = evaluate_assignments(predictions, annotations) if annotations else None
    confidence_intervals = None
    decision_cost = None
    if annotations:
        n_bootstrap = int(condition.get("bootstrap_replicates", 0))
        if n_bootstrap > 0:
            confidence_intervals = bootstrap_assignment_metrics(
                predictions,
                annotations,
                n_bootstrap=n_bootstrap,
                seed=int(condition.get("bootstrap_seed", 0)),
                confidence=float(condition.get("bootstrap_confidence", 0.95)),
            )
        decision_cost = selective_decision_cost(
            predictions,
            annotations,
            error_cost=float(condition.get("error_cost", 1.0)),
            abstention_cost=float(condition.get("abstention_cost", 0.25)),
        )
    return {
        "condition": copy.deepcopy(condition),
        "assignment_diagnostics": diagnostics,
        "evaluation": evaluation,
        "confidence_intervals": confidence_intervals,
        "selective_decision_cost": decision_cost,
        "predictions": predictions,
        "clean_sketch": C.astype(float).tolist(),
    }


def evaluate_embedding_families(spec: dict[str, Any]) -> dict[str, Any]:
    """Evaluate heterogeneous fixed representations on the same records.

    ``embedding_families`` accepts either family-name strings or dictionaries
    with a display ``name`` plus family-specific parameters.  This permits a
    lexical model and a fixed pretrained sentence encoder to be compared under
    an otherwise identical assignment protocol.
    """
    records = read_jsonl(spec["records"])
    atoms = read_json(spec["atoms"])
    annotations = read_jsonl(spec["annotations"]) if spec.get("annotations") else None
    public_fit = _public_fit_texts(spec.get("public_fit_records"), spec.get("public_fit_max_records"))
    base = dict(spec.get("assignment", {}))
    raw_families = list(spec.get("embedding_families", ["tfidf_word", "tfidf_char"]))
    family_specs: list[dict[str, Any]] = []
    for index, item in enumerate(raw_families):
        if isinstance(item, str):
            family_specs.append({"name": item, "embedding_family": item})
        elif isinstance(item, dict):
            payload = dict(item)
            family = str(payload.get("embedding_family", payload.get("family", "")))
            if not family:
                raise ValueError(f"embedding_families[{index}] lacks embedding_family")
            payload["embedding_family"] = family
            payload.setdefault("name", family)
            family_specs.append(payload)
        else:
            raise TypeError("embedding_families entries must be strings or mappings")
    unknown = sorted(
        {str(item["embedding_family"]) for item in family_specs}
        - set(SUPPORTED_EMBEDDING_FAMILIES)
    )
    if unknown:
        raise ValueError(f"unsupported embedding families: {unknown}")
    names = [str(item["name"]) for item in family_specs]
    if len(set(names)) != len(names):
        raise ValueError("embedding-family condition names must be unique")

    results: dict[str, Any] = {}
    for item in family_specs:
        name = str(item["name"])
        overrides = {k: v for k, v in item.items() if k != "name"}
        condition = {**base, **overrides}
        results[name] = run_assignment_condition(
            records=records,
            atoms=atoms,
            annotations=annotations,
            public_fit_texts=public_fit,
            condition=condition,
        )

    paired: list[dict[str, Any]] = []
    if annotations and len(names) >= 2:
        metrics = list(spec.get(
            "paired_metrics",
            ["top1_precision", "topK_recall", "coverage_rate", "selective_risk"],
        ))
        n_bootstrap = int(spec.get("paired_bootstrap_replicates", base.get("bootstrap_replicates", 2000)))
        confidence = float(spec.get("paired_bootstrap_confidence", 0.95))
        seed = int(spec.get("paired_bootstrap_seed", 0))
        reference_name = str(spec.get("paired_reference", names[0]))
        if reference_name not in results:
            raise ValueError("paired_reference is not an embedding-family condition name")
        for name in names:
            if name == reference_name:
                continue
            for metric in metrics:
                try:
                    comparison = paired_bootstrap_metric_difference(
                        results[name]["predictions"],
                        results[reference_name]["predictions"],
                        annotations,
                        metric=str(metric),
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                        confidence=confidence,
                    )
                except ValueError as exc:
                    comparison = {"metric": str(metric), "undefined_reason": str(exc)}
                comparison.update({"condition_A": name, "condition_B": reference_name})
                paired.append(comparison)
    return {
        "experiment": "fixed_embedding_family_external_validity",
        "records": spec["records"],
        "atoms": spec["atoms"],
        "annotations": spec.get("annotations"),
        "family_conditions": family_specs,
        "results": results,
        "paired_differences": paired,
        "claim_scope": (
            "Variation across fixed public embedding families evaluates representation sensitivity; "
            "it does not change the differential-privacy accounting. A fixed pretrained encoder "
            "performs no fitting on project records."
        ),
    }


def evaluate_dictionary_transfer(spec: dict[str, Any]) -> dict[str, Any]:
    """Evaluate dictionaries on a target domain or later time split.

    A transfer task supplies target records and annotations made against the
    source dictionary.  It may additionally supply a matched target-domain/time
    dictionary and its own annotations.  The matched condition is evaluated on
    the identical target records, providing an explicit reference for the loss
    attributable to dictionary transfer rather than target-sample difficulty.
    """
    tasks = spec.get("transfer_tasks", [])
    if not tasks:
        raise ValueError("transfer_tasks must be nonempty")
    output: list[dict[str, Any]] = []
    for task in tasks:
        required = {
            "name", "source_domain", "target_domain", "source_period", "target_period",
            "target_records", "source_atoms",
        }
        missing = sorted(required - set(task))
        if missing:
            raise ValueError(f"transfer task {task.get('name', '<unnamed>')} missing {missing}")
        records = read_jsonl(task["target_records"])
        atoms = read_json(task["source_atoms"])
        annotations = read_jsonl(task["target_annotations"]) if task.get("target_annotations") else None
        public_fit = _public_fit_texts(task.get("source_public_fit_records"), task.get("public_fit_max_records"))
        matched_atoms = read_json(task["matched_target_atoms"]) if task.get("matched_target_atoms") else None
        matched_annotations = (
            read_jsonl(task["matched_target_annotations"])
            if task.get("matched_target_annotations") else annotations
        )
        matched_public_fit = _public_fit_texts(
            task.get("matched_target_public_fit_records"), task.get("public_fit_max_records")
        )
        if matched_atoms is not None and matched_annotations is None:
            raise ValueError("matched_target_atoms requires matched_target_annotations or target_annotations")

        conditions = task.get("conditions") or [spec.get("assignment", {})]
        condition_results = []
        for i, condition in enumerate(conditions):
            merged = {**spec.get("assignment", {}), **condition}
            result = run_assignment_condition(
                records=records,
                atoms=atoms,
                annotations=annotations,
                public_fit_texts=public_fit,
                condition=merged,
            )
            result["condition_id"] = str(condition.get("name", f"condition_{i}"))
            matched_result = None
            metric_gaps = None
            if matched_atoms is not None:
                matched_result = run_assignment_condition(
                    records=records,
                    atoms=matched_atoms,
                    annotations=matched_annotations,
                    public_fit_texts=matched_public_fit,
                    condition=merged,
                )
                metric_gaps = {}
                for metric in [
                    "top1_precision", "topK_recall", "coverage_rate", "selective_risk",
                    "false_assignment_rate_on_uncovered",
                ]:
                    source_value = (result.get("evaluation") or {}).get(metric)
                    matched_value = (matched_result.get("evaluation") or {}).get(metric)
                    metric_gaps[metric] = (
                        None if source_value is None or matched_value is None
                        else float(source_value) - float(matched_value)
                    )
                if annotations is not None and matched_annotations is not None:
                    paired = []
                    n_bootstrap = int(merged.get("bootstrap_replicates", 2000))
                    if n_bootstrap > 0:
                        for metric in ["top1_precision", "topK_recall", "coverage_rate", "selective_risk"]:
                            try:
                                paired.append(paired_bootstrap_metric_difference_two_annotations(
                                    result["predictions"],
                                    annotations,
                                    matched_result["predictions"],
                                    matched_annotations,
                                    metric=metric,
                                    n_bootstrap=n_bootstrap,
                                    seed=int(merged.get("bootstrap_seed", 0)),
                                    confidence=float(merged.get("bootstrap_confidence", 0.95)),
                                ))
                            except ValueError as exc:
                                paired.append({"metric": metric, "undefined_reason": str(exc)})
                    metric_gaps["paired_bootstrap_source_minus_matched"] = paired
            condition_results.append({
                "condition_id": result["condition_id"],
                "source_dictionary_result": result,
                "matched_target_dictionary_result": matched_result,
                "source_minus_matched_metric_gaps": metric_gaps,
            })
        output.append(
            {
                "name": task["name"],
                "source_domain": task["source_domain"],
                "target_domain": task["target_domain"],
                "source_period": task["source_period"],
                "target_period": task["target_period"],
                "transfer_type": (
                    "in_domain_time_transfer"
                    if task["source_domain"] == task["target_domain"] and task["source_period"] != task["target_period"]
                    else "cross_domain_transfer"
                    if task["source_domain"] != task["target_domain"]
                    else "in_domain_same_period"
                ),
                "target_records": task["target_records"],
                "source_atoms": task["source_atoms"],
                "target_annotations_against_source_dictionary": task.get("target_annotations"),
                "matched_target_atoms": task.get("matched_target_atoms"),
                "matched_target_annotations": task.get("matched_target_annotations"),
                "results": condition_results,
            }
        )
    return {
        "experiment": "dictionary_transfer_external_validity",
        "tasks": output,
        "interpretation": (
            "Transfer metrics quantify semantic validity and coverage under distribution shift. "
            "When supplied, the matched target dictionary is evaluated on identical target records, "
            "so source-minus-matched gaps isolate dictionary transfer more directly. Plan fidelity "
            "remains conditional on whichever fixed assignment map is used."
        ),
    }


def _atom_frequency_from_annotations(annotations: list[dict[str, Any]]) -> Counter[int]:
    gold, _, _ = aggregate_annotations(annotations)
    counts: Counter[int] = Counter()
    for atom_ids in gold.values():
        counts.update(atom_ids)
    return counts


def make_misspecified_dictionary(
    atoms: list[dict[str, Any]],
    *,
    fraction: float,
    mode: str,
    seed: int,
    annotations: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create a controlled public-dictionary misspecification.

    ``drop_random`` removes a public random subset. ``drop_frequent_gold``
    removes the most frequently annotated atoms and is an adversarial stress
    test, not a deployable dictionary-construction rule. ``shuffle_descriptions``
    preserves labels and IDs but permutes public descriptions among selected
    atoms, testing representation dependence on descriptive text.
    """
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0, 1)")
    if mode not in {"drop_random", "drop_frequent_gold", "shuffle_descriptions"}:
        raise ValueError("unknown misspecification mode")
    rng = np.random.default_rng(seed)
    atoms_copy = [copy.deepcopy(atom) for atom in atoms]
    n_change = int(round(fraction * len(atoms_copy)))
    if fraction > 0 and n_change == 0:
        n_change = 1
    if n_change == 0:
        return atoms_copy, {"fraction": fraction, "mode": mode, "changed_atom_ids": []}

    atom_ids = [int(atom.get("atom_id", i)) for i, atom in enumerate(atoms_copy)]
    if mode == "drop_random":
        changed = set(map(int, rng.choice(atom_ids, size=n_change, replace=False)))
        modified = [atom for atom, aid in zip(atoms_copy, atom_ids) if aid not in changed]
    elif mode == "drop_frequent_gold":
        if not annotations:
            raise ValueError("drop_frequent_gold requires annotations")
        frequency = _atom_frequency_from_annotations(annotations)
        ranked = sorted(atom_ids, key=lambda aid: (-frequency.get(aid, 0), aid))
        changed = set(ranked[:n_change])
        modified = [atom for atom, aid in zip(atoms_copy, atom_ids) if aid not in changed]
    else:
        selected_positions = sorted(map(int, rng.choice(len(atoms_copy), size=n_change, replace=False)))
        descriptions = [str(atoms_copy[i].get("description", "")) for i in selected_positions]
        if len(descriptions) > 1:
            descriptions = descriptions[1:] + descriptions[:1]
        else:
            descriptions = ["deliberately misspecified public description unrelated to the original atom"]
        for pos, description in zip(selected_positions, descriptions):
            atoms_copy[pos]["description"] = description
        changed = {atom_ids[i] for i in selected_positions}
        modified = atoms_copy

    if not modified:
        raise ValueError("misspecification removed every atom")
    return modified, {
        "fraction": float(fraction),
        "mode": mode,
        "changed_atom_ids": sorted(map(int, changed)),
        "n_original_atoms": len(atoms),
        "n_modified_atoms": len(modified),
        "seed": int(seed),
    }


def _bootstrap_mean_interval(values: list[float], *, seed: int, n_bootstrap: int = 2000) -> dict[str, Any] | None:
    finite = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=float)
    if not finite.size:
        return None
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        draws[i] = float(rng.choice(finite, size=finite.size, replace=True).mean())
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(finite.mean()),
        "lower_95": float(lo),
        "upper_95": float(hi),
        "n_independent_seeds": int(finite.size),
        "bootstrap_replicates": int(n_bootstrap),
    }


def evaluate_misspecification(spec: dict[str, Any]) -> dict[str, Any]:
    """Stress dictionary misspecification without target-tuned abstention.

    The full risk--coverage curve is reported on target annotations.  When a
    separate annotated auxiliary calibration set is supplied, operating
    thresholds are selected only on that set and then evaluated unchanged on
    the target records.  Without such a set, no threshold is called selected or
    recommended.
    """
    records = read_jsonl(spec["records"])
    atoms = read_json(spec["atoms"])
    annotations = read_jsonl(spec["annotations"])
    public_fit = _public_fit_texts(spec.get("public_fit_records"), spec.get("public_fit_max_records"))
    calibration_records = read_jsonl(spec["calibration_records"]) if spec.get("calibration_records") else None
    calibration_annotations = (
        read_jsonl(spec["calibration_annotations"]) if spec.get("calibration_annotations") else None
    )
    if (calibration_records is None) != (calibration_annotations is None):
        raise ValueError("calibration_records and calibration_annotations must be supplied together")
    fractions = [float(x) for x in spec.get("fractions", [0.0, 0.1, 0.3, 0.5])]
    modes = list(spec.get("misspecification_modes", ["drop_random", "drop_frequent_gold"]))
    seeds = [int(x) for x in spec.get("seeds", [0, 1, 2, 3, 4])]
    thresholds = sorted(set(float(x) for x in spec.get("abstention_thresholds", [0.0, 0.05, 0.1, 0.2])))
    base = dict(spec.get("assignment", {}))
    error_cost = float(base.get("error_cost", 1.0))
    decision_mass_mode = str(spec.get("decision_mass_mode", "unit"))
    if decision_mass_mode not in {"unit", "confidence"}:
        raise ValueError("decision_mass_mode must be unit or confidence")
    abstention_costs = [
        float(x) for x in spec.get("abstention_costs", [base.get("abstention_cost", 0.25)])
    ]
    if any(cost < 0 for cost in abstention_costs):
        raise ValueError("abstention costs must be nonnegative")
    rows: list[dict[str, Any]] = []

    for mode in modes:
        for fraction in fractions:
            for seed in seeds:
                modified, modification = make_misspecified_dictionary(
                    atoms,
                    fraction=fraction,
                    mode=mode,
                    seed=seed,
                    annotations=annotations,
                )
                conditions: list[dict[str, Any]] = [
                    {
                        **base,
                        "name": "forced_assignment",
                        "abstention_mode": "disabled",
                        "similarity_threshold": None,
                        "mass_mode": decision_mass_mode,
                    }
                ]
                for threshold in thresholds:
                    conditions.append(
                        {
                            **base,
                            "name": f"abstain_tau_{threshold:g}",
                            "abstention_mode": "zero",
                            "similarity_threshold": threshold,
                            "mass_mode": decision_mass_mode,
                        }
                    )

                calibration_payloads: dict[str, dict[str, Any]] = {}
                if calibration_records is not None and calibration_annotations is not None:
                    for condition in conditions:
                        cal = run_assignment_condition(
                            records=calibration_records,
                            atoms=modified,
                            annotations=calibration_annotations,
                            public_fit_texts=public_fit,
                            condition={**condition, "bootstrap_replicates": 0},
                        )
                        calibration_payloads[str(condition["name"])] = cal

                selected_by_cost: dict[str, str | None] = {}
                for abstention_cost in abstention_costs:
                    key = f"{abstention_cost:g}"
                    if not calibration_payloads:
                        selected_by_cost[key] = None
                        continue
                    candidates: list[tuple[float, float, float, str]] = []
                    for condition in conditions:
                        name = str(condition["name"])
                        cal = calibration_payloads[name]
                        cost = selective_decision_cost(
                            cal["predictions"],
                            calibration_annotations,
                            error_cost=error_cost,
                            abstention_cost=abstention_cost,
                        )["mean_decision_cost"]
                        coverage = float((cal["evaluation"] or {}).get("coverage_rate") or 0.0)
                        threshold = condition.get("similarity_threshold")
                        threshold_sort = -1.0 if threshold is None else float(threshold)
                        candidates.append((float(cost), -coverage, threshold_sort, name))
                    selected_by_cost[key] = min(candidates)[3]

                condition_results = []
                for condition in conditions:
                    result = run_assignment_condition(
                        records=records,
                        atoms=modified,
                        annotations=annotations,
                        public_fit_texts=public_fit,
                        condition=condition,
                    )
                    name = str(condition["name"])
                    cost_grid: dict[str, Any] = {}
                    for abstention_cost in abstention_costs:
                        key = f"{abstention_cost:g}"
                        target_cost = selective_decision_cost(
                            result["predictions"],
                            annotations,
                            error_cost=error_cost,
                            abstention_cost=abstention_cost,
                        )
                        target_cost["selected_by_auxiliary_calibration"] = selected_by_cost[key] == name
                        if calibration_payloads:
                            target_cost["auxiliary_calibration_cost"] = selective_decision_cost(
                                calibration_payloads[name]["predictions"],
                                calibration_annotations,
                                error_cost=error_cost,
                                abstention_cost=abstention_cost,
                            )["mean_decision_cost"]
                        else:
                            target_cost["auxiliary_calibration_cost"] = None
                        cost_grid[key] = target_cost
                    condition_results.append(
                        {
                            "condition": condition,
                            "assignment_diagnostics": result["assignment_diagnostics"],
                            "evaluation": result["evaluation"] or {},
                            "confidence_intervals": result.get("confidence_intervals"),
                            "decision_cost_grid": cost_grid,
                        }
                    )

                forced_item = next(item for item in condition_results if item["condition"]["name"] == "forced_assignment")
                for item in condition_results:
                    for cost_key, target_cost in item["decision_cost_grid"].items():
                        forced_cost = forced_item["decision_cost_grid"][cost_key]["mean_decision_cost"]
                        item_cost = target_cost["mean_decision_cost"]
                        target_cost["cost_difference_vs_forced"] = float(item_cost) - float(forced_cost)
                        target_cost["abstention_preferred_to_forced"] = bool(
                            item["condition"].get("abstention_mode") != "disabled"
                            and float(item_cost) < float(forced_cost)
                        )
                rows.append(
                    {
                        "modification": modification,
                        "calibration_available": bool(calibration_payloads),
                        "selected_condition_by_abstention_cost": selected_by_cost,
                        "conditions": condition_results,
                    }
                )

    summary: list[dict[str, Any]] = []
    for mode in modes:
        for fraction in fractions:
            subset = [
                row for row in rows
                if row["modification"]["mode"] == mode
                and float(row["modification"]["fraction"]) == float(fraction)
            ]
            condition_names = sorted({item["condition"]["name"] for row in subset for item in row["conditions"]})
            for condition_name in condition_names:
                items = [
                    next(item for item in row["conditions"] if item["condition"]["name"] == condition_name)
                    for row in subset
                ]
                for abstention_cost in abstention_costs:
                    key = f"{abstention_cost:g}"
                    summary.append(
                        {
                            "mode": mode,
                            "fraction": float(fraction),
                            "condition": condition_name,
                            "abstention_cost": float(abstention_cost),
                            "coverage": _bootstrap_mean_interval(
                                [item["evaluation"].get("coverage_rate") for item in items],
                                seed=10_000 + int(round(100 * fraction)) + len(summary),
                            ),
                            "selective_risk": _bootstrap_mean_interval(
                                [item["evaluation"].get("selective_risk") for item in items],
                                seed=20_000 + int(round(100 * fraction)) + len(summary),
                            ),
                            "false_assignment_on_uncovered": _bootstrap_mean_interval(
                                [item["evaluation"].get("false_assignment_rate_on_uncovered") for item in items],
                                seed=30_000 + int(round(100 * fraction)) + len(summary),
                            ),
                            "decision_cost": _bootstrap_mean_interval(
                                [item["decision_cost_grid"][key].get("mean_decision_cost") for item in items],
                                seed=40_000 + int(round(100 * fraction)) + len(summary),
                            ),
                            "preferred_to_forced_fraction": float(np.mean([
                                item["decision_cost_grid"][key]["abstention_preferred_to_forced"]
                                for item in items
                            ])),
                            "selected_by_auxiliary_calibration_fraction": float(np.mean([
                                item["decision_cost_grid"][key]["selected_by_auxiliary_calibration"]
                                for item in items
                            ])),
                        }
                    )

    return {
        "experiment": "dictionary_misspecification_and_abstention",
        "records": spec["records"],
        "atoms": spec["atoms"],
        "annotations": spec["annotations"],
        "calibration_records": spec.get("calibration_records"),
        "calibration_annotations": spec.get("calibration_annotations"),
        "threshold_selection_policy": (
            "auxiliary annotated calibration only" if calibration_records is not None
            else "none; target risk-coverage grid is descriptive and not used to select an operating threshold"
        ),
        "error_cost": error_cost,
        "decision_mass_mode": decision_mass_mode,
        "abstention_costs": abstention_costs,
        "rows": rows,
        "summary_across_independent_misspecification_seeds": summary,
        "interpretation": (
            "Abstention is declared preferable only under the explicitly reported cost model. "
            "Coverage-selective-risk curves are reported independently. Any chosen operating "
            "threshold is selected on separate auxiliary annotations, never on the protected target labels."
        ),
    }


def run_external_validity(config: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {"schema_version": "external_validity_v2"}
    if config.get("embedding_family_experiment"):
        output["embedding_family_experiment"] = evaluate_embedding_families(config["embedding_family_experiment"])
    if config.get("dictionary_transfer_experiment"):
        output["dictionary_transfer_experiment"] = evaluate_dictionary_transfer(config["dictionary_transfer_experiment"])
    if config.get("misspecification_experiment"):
        output["misspecification_experiment"] = evaluate_misspecification(config["misspecification_experiment"])
    if len(output) == 1:
        raise ValueError("configuration did not enable any external-validity experiment")
    return output


def _write_predictions(result: dict[str, Any], output_dir: Path) -> None:
    """Write large prediction rows separately and replace them with paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    family = result.get("embedding_family_experiment", {}).get("results", {})
    for name, payload in family.items():
        rows = payload.pop("predictions", None)
        if rows is not None:
            path = output_dir / f"embedding_{name}_predictions.jsonl"
            write_jsonl(rows, path)
            payload["predictions_path"] = str(path)
    for task in result.get("dictionary_transfer_experiment", {}).get("tasks", []):
        safe_task = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in task["name"])
        for item in task.get("results", []):
            condition_id = item["condition_id"]
            for key, suffix in [
                ("source_dictionary_result", "source"),
                ("matched_target_dictionary_result", "matched"),
            ]:
                payload = item.get(key)
                if not payload:
                    continue
                rows = payload.pop("predictions", None)
                if rows is not None:
                    path = output_dir / f"transfer_{safe_task}_{condition_id}_{suffix}_predictions.jsonl"
                    write_jsonl(rows, path)
                    payload["predictions_path"] = str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifacts-dir", default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    result = run_external_validity(config)
    artifacts = Path(args.artifacts_dir or (str(Path(args.output).with_suffix("")) + "_artifacts"))
    _write_predictions(result, artifacts)
    write_json(result, args.output)
    print(json.dumps({"output": args.output, "artifacts_dir": str(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
