"""Validation helpers for matched cross-method experimental comparisons."""
from __future__ import annotations

MATCH_KEYS = (
    "records", "atoms", "data_seed", "n_records", "n_users", "protected_unit",
    "epsilon", "delta", "adjacency", "public_denominator", "L", "K",
    "user_contribution_bound", "assignment_similarity_threshold",
)


def validate_matched_configs(configs: list[dict]) -> dict:
    if len(configs) < 2:
        raise ValueError("at least two method configurations are required")
    reference = configs[0]
    mismatches = []
    for i, cfg in enumerate(configs[1:], start=1):
        for key in MATCH_KEYS:
            if reference.get(key) != cfg.get(key):
                mismatches.append({"config_index": i, "field": key, "reference": reference.get(key), "value": cfg.get(key)})
    return {"matched": not mismatches, "mismatches": mismatches, "checked_fields": list(MATCH_KEYS)}


def require_matched_configs(configs: list[dict]) -> None:
    report = validate_matched_configs(configs)
    if not report["matched"]:
        raise ValueError(f"unmatched baseline protocol: {report['mismatches']}")
