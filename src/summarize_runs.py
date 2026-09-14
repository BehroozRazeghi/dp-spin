"""Aggregate completed run directories into CSV and JSON summaries.

The module reads saved configurations, metrics, verifier outputs, and optional comparative evaluations. 
It groups repeated conditions without accessing raw protected records or rerunning any release mechanism.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .utils import ensure_dir, write_json


def _parse_float_token(token: str) -> float:
    """Parse run-name float tokens where decimal points were slugified to underscores.

    Examples: ``0_5`` -> 0.5, ``1_0`` -> 1.0, ``2`` -> 2.0.
    """
    return float(str(token).replace("_", "."))


def _parse_run_name(name: str) -> dict[str, Any]:
    """Parse encoded run parameters from a DP-SPIN output-directory name.

    Run names are slugified, so the user contribution bound may appear as
    either ``_B1_0_`` in pre-slug text or ``_b1_0_`` on disk.  This parser is
    case-insensitive for the clipping-bound token so downstream summaries do not
    silently mix distinct ``B`` values.
    """
    out: dict[str, Any] = {}
    patterns = {
        "n_records": r"_n([0-9]+)_",
        "n_users": r"_u([0-9]+)_",
        "user_contribution_bound": r"_[bB]([0-9]+(?:_[0-9]+)?)_",
        "m": r"_m([0-9]+)_",
        "K": r"_k([0-9]+)_",
        "L": r"_l([0-9]+)_",
        "epsilon": r"_eps([0-9]+(?:_[0-9]+)?)_",
        "seed": r"_seed([0-9]+)$",
        "probe_support": r"support([0-9]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, name)
        if not m:
            continue
        val = m.group(1)
        if key in {"epsilon", "user_contribution_bound"}:
            out[key] = _parse_float_token(val)
        else:
            out[key] = int(val)
    for mech in ["private_topL", "gaussian", "laplace"]:
        if f"_{mech}_" in name:
            out["mechanism"] = mech
            break
    return out


def _normalize_comparative_choice(choice: object) -> str | None:
    """Normalize comparative-evaluator choices to DP, NP, or tie."""
    if choice is None:
        return None
    raw = str(choice).strip()
    mapping = {
        "A": "DP",
        "B": "NP",
        "DP": "DP",
        "NP": "NP",
        "non-private": "NP",
        "non_private": "NP",
        "nonprivate": "NP",
        "tie": "tie",
    }
    return mapping.get(raw, mapping.get(raw.lower(), raw))


def _dp_preferred_score(choice: object) -> float | None:
    normalized = _normalize_comparative_choice(choice)
    if normalized == "DP":
        return 1.0
    if normalized == "tie":
        return 0.5
    if normalized == "NP":
        return 0.0
    return None


def _load_row(metrics_path: Path) -> dict[str, Any]:
    run_dir = metrics_path.parent
    run = run_dir.name
    data = json.load(open(metrics_path, encoding="utf-8"))
    sketch = data.get("sketch", data)
    cfg: dict[str, Any] = {}
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(open(cfg_path, encoding="utf-8")) or {}
        except Exception:
            cfg = {}
    parsed = _parse_run_name(run)
    clipping = data.get("assignment_and_clipping", {}) or {}
    selection = data.get("protected_unit_selection", {}) or {}
    protected_unit = cfg.get("protected_unit") or parsed.get("protected_unit")
    text_eval = data.get("text_eval", {}) or {}
    lexical = text_eval.get("lexical", {}) or {}
    embedding = text_eval.get("embedding", {}) or {}
    comparative = text_eval.get("comparative", {}) or {}
    verifier = data.get("verifier", {}) or {}
    reverified_path = run_dir / "verifier_reverified.json"
    if reverified_path.exists():
        try:
            verifier = json.load(open(reverified_path, encoding="utf-8")) or verifier
        except Exception:
            pass
    regeneration = verifier.get("regeneration", {}) or {}
    return {
        "run": run,
        "path": str(run_dir),
        "protected_unit": protected_unit,
        "n_records": cfg.get("n_records_effective", selection.get("n_records_selected", parsed.get("n_records"))),
        "n_users": cfg.get("n_users_effective", selection.get("n_users_selected", parsed.get("n_users"))),
        "n_protected_units": cfg.get("n_protected_units_effective"),
        "user_contribution_bound": cfg.get("user_contribution_bound", parsed.get("user_contribution_bound")),
        "m": cfg.get("m", parsed.get("m")),
        "K": cfg.get("K", parsed.get("K")),
        "L": cfg.get("L", parsed.get("L")),
        "epsilon": cfg.get("epsilon_effective", cfg.get("epsilon", parsed.get("epsilon"))),
        "epsilon_configured": cfg.get("epsilon_configured", cfg.get("epsilon", parsed.get("epsilon"))),
        "delta": cfg.get("delta_effective", cfg.get("delta")),
        "delta_configured": cfg.get("delta_configured", cfg.get("delta")),
        "delta_effective": cfg.get("delta_effective", cfg.get("delta")),
        "adjacency": cfg.get("adjacency"),
        "gaussian_calibration": cfg.get("gaussian_calibration"),
        "value_epsilon_fraction": cfg.get("value_epsilon_fraction"),
        "value_mechanism": cfg.get("value_mechanism"),
        "decoder_backend": cfg.get("decoder_backend"),
        "decoder_contract_mode": cfg.get("decoder_contract_mode"),
        "clean_reference_decoder_backend": cfg.get("clean_reference_decoder_backend"),
        "summary_embedding_backend_configured": cfg.get("summary_embedding_backend"),
        "mechanism": cfg.get("mechanism", parsed.get("mechanism")),
        "seed": cfg.get("seed", parsed.get("seed")),
        "overlap_at_L": sketch.get("overlap_at_L"),
        "mass_retained_fraction": sketch.get("mass_retained_fraction"),
        "support_mass_loss": sketch.get("support_mass_loss"),
        "num_total_coordinates": sketch.get("num_total_coordinates"),
        "num_released_coordinates": sketch.get("num_released_coordinates"),
        "is_sparse_release": sketch.get("is_sparse_release"),
        "UAR": verifier.get("UAR"),
        "UCR": verifier.get("UCR"),
        "verifier_accept": verifier.get("accept"),
        "comparison_count": verifier.get("comparison_count"),
        "supported_comparison_count": verifier.get("supported_comparison_count"),
        "unsupported_comparison_count": verifier.get("unsupported_comparison_count"),
        "attempts_used": regeneration.get("attempts_used"),
        "accepted_before_fallback": regeneration.get("accepted_before_fallback"),
        "fallback_used": regeneration.get("fallback_used"),
        "first_attempt_accept": (
            None
            if not isinstance(regeneration.get("accepted_before_fallback"), bool)
            else bool(regeneration.get("accepted_before_fallback")) and int(regeneration.get("attempts_used", 0) or 0) == 1
        ),
        # User-level clipping diagnostics, present only for user-level runs.
        "n_users_clipped": clipping.get("n_users_clipped"),
        "fraction_users_clipped": clipping.get("fraction_users_clipped"),
        "max_unclipped_user_mass": clipping.get("max_unclipped_user_mass"),
        "max_clipped_user_mass": clipping.get("max_clipped_user_mass"),
        "mean_records_per_user": clipping.get("mean_records_per_user"),
        "users_with_multiple_records": clipping.get("users_with_multiple_records"),
        # Text-quality metrics against the non-private top-L reference summary.
        "text_token_jaccard": lexical.get("token_jaccard"),
        "text_bigram_jaccard": lexical.get("bigram_jaccard"),
        "text_trigram_jaccard": lexical.get("trigram_jaccard"),
        "text_keyphrase_jaccard": lexical.get("keyphrase_jaccard"),
        "text_tfidf_cosine": lexical.get("tfidf_cosine"),
        "summary_embedding_cosine": embedding.get("summary_embedding_cosine"),
        "summary_embedding_backend": embedding.get("summary_embedding_backend"),
        "summary_embedding_model": embedding.get("summary_embedding_model"),
        "comparative_eval_enabled": comparative.get("enabled"),
        "comparative_eval_backend": comparative.get("backend"),
        "comparative_choice": _normalize_comparative_choice(comparative.get("choice")),
        "comparative_raw_choice": comparative.get("raw_choice"),
        "comparative_score_dp_summary": comparative.get("score_dp_summary", comparative.get("score_A_dp_summary")),
        "comparative_score_np_reference_summary": comparative.get("score_np_reference_summary", comparative.get("score_B_clean_summary")),
        "comparative_dp_preferred": _dp_preferred_score(comparative.get("choice")),
        # Controlled probe-atom metrics.
        "probe_support": cfg.get("probe_support", parsed.get("probe_support")),
        "probe_base_records": cfg.get("probe_base_records"),
        "probe_generated_records": cfg.get("probe_generated_records"),
        "probe_plan_plan_admission_rate": (data.get("probe_plan", {}) or {}).get("plan_admission_rate"),
        "probe_plan_num_plan_admitted": (data.get("probe_plan", {}) or {}).get("num_plan_admitted"),
        "probe_summary_mention_rate": (data.get("probe_summary", data.get("probe", {})) or {}).get("mention_rate"),
        "probe_summary_num_mentioned": (data.get("probe_summary", data.get("probe", {})) or {}).get("num_mentioned"),
    }

def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    cols = [
        "run", "protected_unit", "n_records", "n_users", "n_protected_units", "user_contribution_bound",
        "mechanism", "m", "K", "L", "probe_support", "epsilon", "epsilon_configured", "delta", "delta_configured", "delta_effective", "adjacency",
        "gaussian_calibration", "value_epsilon_fraction", "value_mechanism",
        "decoder_backend", "decoder_contract_mode", "clean_reference_decoder_backend",
        "summary_embedding_backend_configured", "seed",
        "overlap_at_L", "mass_retained_fraction", "support_mass_loss",
        "num_total_coordinates", "num_released_coordinates", "is_sparse_release",
        "UAR", "UCR", "verifier_accept", "comparison_count",
        "supported_comparison_count", "unsupported_comparison_count",
        "attempts_used", "first_attempt_accept", "accepted_before_fallback", "fallback_used",
        "n_users_clipped", "fraction_users_clipped", "max_unclipped_user_mass",
        "max_clipped_user_mass", "mean_records_per_user", "users_with_multiple_records",
        "text_token_jaccard", "text_bigram_jaccard", "text_trigram_jaccard",
        "text_keyphrase_jaccard", "text_tfidf_cosine", "summary_embedding_cosine",
        "summary_embedding_backend", "summary_embedding_model",
        "comparative_eval_enabled", "comparative_eval_backend", "comparative_choice",
        "comparative_score_dp_summary", "comparative_score_np_reference_summary", "comparative_dp_preferred",
        "probe_base_records", "probe_generated_records",
        "probe_plan_plan_admission_rate", "probe_plan_num_plan_admitted",
        "probe_summary_mention_rate", "probe_summary_num_mentioned",
        "path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def _summary_group_key(row: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Return the grouping key for scientific summaries.

    For user-level runs, the number of selected records varies across seeds
    because each seed selects a fixed number of users and then includes all
    records belonging to those users.  Therefore user-level summaries must not
    group by effective n_records.  Instead, we group by n_users and contribution
    bound B, and report n_records as a distribution inside the group.
    """
    protected_unit = row.get("protected_unit")
    base: list[tuple[str, Any]] = [("protected_unit", protected_unit)]
    if protected_unit == "user":
        base.extend([
            ("n_users", row.get("n_users")),
            ("user_contribution_bound", row.get("user_contribution_bound")),
        ])
    else:
        base.append(("n_records", row.get("n_records")))
    # Controlled probe sweeps must be grouped by probe support when present.
    if row.get("probe_support") is not None:
        base.append(("probe_support", row.get("probe_support")))
    base.extend([
        ("mechanism", row.get("mechanism")),
        ("m", row.get("m")),
        ("K", row.get("K")),
        ("L", row.get("L")),
        ("epsilon", row.get("epsilon")),
        ("delta", row.get("delta")),
        ("adjacency", row.get("adjacency")),
        ("gaussian_calibration", row.get("gaussian_calibration")),
        ("value_epsilon_fraction", row.get("value_epsilon_fraction")),
        ("value_mechanism", row.get("value_mechanism")),
        ("decoder_backend", row.get("decoder_backend")),
        ("decoder_contract_mode", row.get("decoder_contract_mode")),
        ("clean_reference_decoder_backend", row.get("clean_reference_decoder_backend")),
        ("summary_embedding_backend_configured", row.get("summary_embedding_backend_configured")),
    ])
    return tuple(base)


def _numeric_summary(item: dict[str, Any], rs: list[dict[str, Any]], field: str, prefix: str | None = None) -> None:
    vals = [r.get(field) for r in rs if isinstance(r.get(field), (int, float))]
    if not vals:
        return
    name = prefix or field
    item[f"{name}_mean"] = mean(vals)
    item[f"{name}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
    item[f"{name}_min"] = min(vals)
    item[f"{name}_max"] = max(vals)


def _write_summary(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    ensure_dir(path.parent)
    groups: dict[tuple[tuple[str, Any], ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[_summary_group_key(r)].append(r)
    out: list[dict[str, Any]] = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for pair in kv[0] for x in pair)):
        item = {k: v for k, v in key}
        item["num_runs"] = len(rs)
        seeds = sorted(r.get("seed") for r in rs if r.get("seed") is not None)
        if seeds:
            item["seeds"] = " ".join(str(s) for s in seeds)
            item["num_seeds"] = len(set(seeds))
        # Always summarize effective sample sizes.  For user-level runs, this is
        # the correct way to expose seed-to-seed record-count variation without
        # splitting the group.
        for sample_field in ["n_records", "n_users", "n_protected_units"]:
            _numeric_summary(item, rs, sample_field)
        for metric in [
            "overlap_at_L", "mass_retained_fraction", "support_mass_loss",
            "UAR", "UCR", "comparison_count", "supported_comparison_count",
            "unsupported_comparison_count", "attempts_used",
            "n_users_clipped", "fraction_users_clipped",
            "max_unclipped_user_mass", "max_clipped_user_mass",
            "mean_records_per_user", "users_with_multiple_records",
            "text_token_jaccard", "text_bigram_jaccard", "text_trigram_jaccard",
            "text_keyphrase_jaccard", "text_tfidf_cosine", "summary_embedding_cosine",
            "comparative_score_dp_summary", "comparative_score_np_reference_summary", "comparative_dp_preferred",
            "probe_plan_plan_admission_rate", "probe_summary_mention_rate",
        ]:
            _numeric_summary(item, rs, metric)
        acc = [bool(r.get("verifier_accept")) for r in rs if r.get("verifier_accept") is not None]
        if acc:
            item["verifier_accept_rate"] = sum(acc) / len(acc)
        for field in ["first_attempt_accept", "accepted_before_fallback", "fallback_used"]:
            vals = [bool(r.get(field)) for r in rs if r.get(field) is not None]
            if vals:
                item[f"{field}_rate"] = sum(vals) / len(vals)
        out.append(item)
    preferred = [
        "protected_unit", "n_records", "n_users", "user_contribution_bound",
        "mechanism", "m", "K", "L", "probe_support", "epsilon", "epsilon_configured", "delta", "delta_configured", "delta_effective", "adjacency",
        "gaussian_calibration", "value_epsilon_fraction", "value_mechanism",
        "decoder_backend", "decoder_contract_mode", "clean_reference_decoder_backend",
        "summary_embedding_backend_configured",
        "num_runs", "num_seeds", "seeds",
        "n_records_mean", "n_records_min", "n_records_max", "n_records_std",
        "n_users_mean", "n_users_min", "n_users_max", "n_users_std",
        "overlap_at_L_mean", "overlap_at_L_std", "overlap_at_L_min", "overlap_at_L_max",
        "mass_retained_fraction_mean", "mass_retained_fraction_std", "mass_retained_fraction_min", "mass_retained_fraction_max",
        "support_mass_loss_mean", "support_mass_loss_std", "support_mass_loss_min", "support_mass_loss_max",
        "verifier_accept_rate", "UAR_mean", "UCR_mean",
        "comparison_count_mean", "supported_comparison_count_mean", "unsupported_comparison_count_mean",
        "attempts_used_mean", "first_attempt_accept_rate", "accepted_before_fallback_rate", "fallback_used_rate",
        "text_keyphrase_jaccard_mean", "text_token_jaccard_mean", "text_bigram_jaccard_mean",
        "text_tfidf_cosine_mean", "summary_embedding_cosine_mean", "comparative_dp_preferred_mean",
        "probe_plan_plan_admission_rate_mean", "probe_summary_mention_rate_mean",
        "fraction_users_clipped_mean", "n_users_clipped_mean", "users_with_multiple_records_mean",
    ]
    all_cols = sorted({k for r in out for k in r.keys()})
    cols = [c for c in preferred if c in all_cols] + [c for c in all_cols if c not in preferred]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow(r)
    return out

def main() -> None:
    """Aggregate completed run directories from the command line."""
    p = argparse.ArgumentParser(description="Summarize DP-SPIN run metrics without mixing different sample sizes or user bounds.")
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prefix", default="metrics")
    args = p.parse_args()
    metrics = sorted(Path(args.runs_dir).glob("*/metrics.json"))
    rows = [_load_row(p) for p in metrics]
    outdir = ensure_dir(args.output_dir)
    _write_rows(rows, outdir / f"{args.prefix}_runs.csv")
    summary = _write_summary(rows, outdir / f"{args.prefix}_summary.csv")
    write_json({"num_runs": len(rows), "groups": summary}, outdir / f"{args.prefix}_summary.json")
    print(f"Wrote {len(rows)} run rows and {len(summary)} grouped rows to {outdir}")


if __name__ == "__main__":
    main()
