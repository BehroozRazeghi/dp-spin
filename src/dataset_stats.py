"""Compute dataset diagnostics for record-level and user-level protocols.

The reported quantities include record counts, protected-user counts, text-length statistics, and optional metadata frequencies. 
This code is not a DP mechanism.
Run it on public or auxiliary data, or treat any statistics computed from a protected target split as non-private evaluation artifacts.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .utils import read_jsonl, write_json, ensure_dir


def _lookup(record: dict[str, Any], field: str) -> Any:
    if field in record:
        return record[field]
    meta = record.get("metadata") or {}
    if isinstance(meta, dict) and field in meta:
        return meta[field]
    if "." in field:
        cur: Any = record
        for part in field.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return None
        return cur
    return None


def _record_text(record: dict[str, Any]) -> str:
    value = record.get("text")
    if value is None:
        value = _lookup(record, "text")
    return str(value or "")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return float(values[idx])


def _split_categories(value: Any) -> list[str]:
    if value in {None, ""}:
        return []
    if isinstance(value, list):
        raw = []
        for x in value:
            raw.extend(_split_categories(x))
        return raw
    # Yelp preprocessing stores categories separated by " | "; raw Yelp and
    # other datasets may use commas.  Support both.
    return [x.strip() for x in re.split(r"\s*[|,]\s*", str(value)) if x.strip()]


def _write_key_value_csv(obj: dict[str, Any], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                continue
            writer.writerow([k, v])


def _write_counter_csv(counter: Counter[str], path: str | Path, limit: int = 100) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["value", "count"])
        for k, v in counter.most_common(limit):
            writer.writerow([k, int(v)])


def stats(
    records_path: str,
    output: str | None = None,
    user_id_field: str = "user_id",
    category_field: str = "metadata.categories",
    dataset_name: str | None = None,
    split_name: str | None = None,
    output_dir: str | None = None,
    top_k: int = 50,
) -> dict[str, Any]:
    """Compute record, user, text-length, and optional metadata-frequency diagnostics."""
    records = read_jsonl(records_path)
    n = len(records)

    users: list[str] = []
    missing = 0
    for r in records:
        uid = _lookup(r, user_id_field)
        if uid in {None, ""}:
            missing += 1
        else:
            users.append(str(uid))
    user_counts = Counter(users)
    counts = list(user_counts.values())

    texts = [_record_text(r) for r in records]
    char_lengths = [len(t) for t in texts]
    token_lengths = [len(t.split()) for t in texts]

    category_counter: Counter[str] = Counter()
    category_records = 0
    primary_counter: Counter[str] = Counter()
    for r in records:
        cats = _lookup(r, category_field)
        if cats in {None, ""}:
            cats = _lookup(r, "categories") or _lookup(r, "business_categories")
        split = _split_categories(cats)
        if split:
            category_records += 1
            category_counter.update(split)
        primary = _lookup(r, "primary_category") or _lookup(r, "business_primary_category")
        if primary not in {None, ""}:
            primary_counter[str(primary)] += 1

    out: dict[str, Any] = {
        "dataset_name": dataset_name or Path(records_path).stem,
        "split_name": split_name or "unspecified",
        "records_path": records_path,
        "records": n,
        "record_level_dp_applicable": True,
        "user_id_field": user_id_field,
        "records_with_user_id": len(users),
        "records_missing_user_id": missing,
        "unique_users": len(user_counts),
        "mean_records_per_user": float(sum(counts) / len(counts)) if counts else 0.0,
        "median_records_per_user": _quantile([float(x) for x in counts], 0.5),
        "p90_records_per_user": _quantile([float(x) for x in counts], 0.9),
        "p95_records_per_user": _quantile([float(x) for x in counts], 0.95),
        "p99_records_per_user": _quantile([float(x) for x in counts], 0.99),
        "max_records_per_user": int(max(counts)) if counts else 0,
        "users_with_at_least_2_records": int(sum(1 for x in counts if x >= 2)),
        "users_with_at_least_3_records": int(sum(1 for x in counts if x >= 3)),
        "users_with_at_least_5_records": int(sum(1 for x in counts if x >= 5)),
        "users_with_at_least_10_records": int(sum(1 for x in counts if x >= 10)),
        "users_with_at_least_20_records": int(sum(1 for x in counts if x >= 20)),
        "fraction_users_with_at_least_2_records": float(sum(1 for x in counts if x >= 2) / max(1, len(counts))),
        "fraction_users_with_at_least_5_records": float(sum(1 for x in counts if x >= 5) / max(1, len(counts))),
        "user_level_dp_applicable": bool(len(user_counts) > 0 and missing == 0),
        "user_level_dp_repeated_contribution_present": bool(any(x >= 2 for x in counts)),
        "mean_text_chars": float(sum(char_lengths) / n) if n else 0.0,
        "median_text_chars": _quantile([float(x) for x in char_lengths], 0.5),
        "p90_text_chars": _quantile([float(x) for x in char_lengths], 0.9),
        "mean_text_tokens": float(sum(token_lengths) / n) if n else 0.0,
        "median_text_tokens": _quantile([float(x) for x in token_lengths], 0.5),
        "p90_text_tokens": _quantile([float(x) for x in token_lengths], 0.9),
        "category_field": category_field,
        "records_with_categories": category_records,
        "unique_categories": len(category_counter),
        "top_categories": dict(category_counter.most_common(top_k)),
        "top_primary_categories": dict(primary_counter.most_common(top_k)),
        "note": (
            "User-level DP is meaningful when records contain stable user identifiers. "
            "Repeated contributions make the distinction from record-level DP empirically visible; "
            "sparse-user datasets remain valid but should be described as sparse-user settings."
        ),
    }

    if output:
        write_json(out, output)
        base = Path(output)
        _write_key_value_csv(out, base.with_suffix(".csv"))
        if category_counter:
            _write_counter_csv(category_counter, base.with_name(base.stem + "_top_categories.csv"), top_k)
        if primary_counter:
            _write_counter_csv(primary_counter, base.with_name(base.stem + "_top_primary_categories.csv"), top_k)

    if output_dir:
        ensure_dir(output_dir)
        stem = f"{out['dataset_name']}_{out['split_name']}_stats"
        write_json(out, Path(output_dir) / f"{stem}.json")
        _write_key_value_csv(out, Path(output_dir) / f"{stem}.csv")
        if category_counter:
            _write_counter_csv(category_counter, Path(output_dir) / f"{stem}_top_categories.csv", top_k)
        if primary_counter:
            _write_counter_csv(primary_counter, Path(output_dir) / f"{stem}_top_primary_categories.csv", top_k)

    return out


def main() -> None:
    """Compute dataset diagnostics from the command line."""
    p = argparse.ArgumentParser(description="Compute record/user/category diagnostics for DP-SPIN datasets.")
    p.add_argument("--records", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--split-name", default=None)
    p.add_argument("--user-id-field", default="user_id")
    p.add_argument("--category-field", default="metadata.categories")
    p.add_argument("--top-k", type=int, default=50)
    args = p.parse_args()
    out = stats(
        args.records,
        output=args.output,
        user_id_field=args.user_id_field,
        category_field=args.category_field,
        dataset_name=args.dataset_name,
        split_name=args.split_name,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    for k, v in out.items():
        if isinstance(v, (dict, list)):
            print(f"{k}: <{type(v).__name__}, len={len(v)}>")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
