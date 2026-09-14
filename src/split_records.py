"""Create disjoint auxiliary and protected target splits.

The auxiliary split is used to name decoder-visible atoms and fit the fixed text representation. 
The protected target split is used only through bounded semantic assignment, aggregation, and differentially private semantic-plan release.
Record-level or user-level hash splitting and date-based splitting are supported; the streaming path preserves the same split rules for large JSONL corpora.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .utils import read_jsonl, write_jsonl, write_json, ensure_dir

DATE_KEYS = ["Date received", "date_received", "Date sent to company", "created_date"]


def _stream_jsonl(path: str):
    """Yield JSON objects from a JSONL file without loading the full file."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e


def _write_jsonl_record(handle, record: dict[str, Any]) -> None:
    """Write one JSONL record."""
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_date(value: Any) -> datetime | None:
    """Parse common date formats used by CFPB exports and API outputs.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    # CFPB exports often use YYYY-MM-DD; API variants may include timestamps.
    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _record_date(record: dict[str, Any]) -> datetime | None:
    """Extract the first usable record date from known metadata date keys.
    """
    meta = record.get("metadata", {}) or {}
    for k in DATE_KEYS:
        if k in meta:
            d = _parse_date(meta[k])
            if d is not None:
                return d
    return None


def _lookup_field(record: dict[str, Any], field: str) -> Any:
    """Lookup a top-level, metadata, or dotted-path field."""
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


def _hash_value(value: Any, seed: int) -> float:
    """Map a value and seed to a deterministic number in [0, 1)."""
    h = hashlib.sha256(f"{seed}:{value}".encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:16], 16) / float(16 ** 16)


def _split_key(record: dict[str, Any], seed: int, split_unit: str, unit_field: str | None) -> str:
    """Return the unit key used for hash splitting.

    For user-level experiments, splitting by user keeps all records from the same
    contributor on the same side of the auxiliary/target boundary.
    """
    if split_unit == "user":
        if not unit_field:
            unit_field = "user_id"
        value = _lookup_field(record, unit_field)
        if value in {None, ""}:
            raise KeyError(f"Missing split unit field {unit_field!r} for user-disjoint split")
        return "user:" + str(value)
    rid = record.get("id", "")
    return "record:" + str(rid)


def _hash_unit(record: dict[str, Any], seed: int, split_unit: str = "record", unit_field: str | None = None) -> float:
    """Map a record or user split key and seed to a deterministic number in [0, 1)."""
    return _hash_value(_split_key(record, seed, split_unit, unit_field), seed)


def split_records(
    input_path: str,
    aux_output: str,
    target_output: str,
    method: str = "hash",
    aux_fraction: float = 0.5,
    aux_before: str | None = None,
    seed: int = 0,
    max_aux: int | None = None,
    max_target: int | None = None,
    split_unit: str = "record",
    unit_field: str | None = None,
    streaming: bool = False,
) -> None:
    """Write disjoint auxiliary and target splits plus a split manifest.

    The default path uses the in-memory implementation.
    Pass ``streaming=True`` to process one JSONL record at a time. Streaming
    is intended for very large corpora such as Yelp and preserves the same
    deterministic hash/date split rules.
    """
    if streaming:
        _split_records_streaming(
            input_path=input_path,
            aux_output=aux_output,
            target_output=target_output,
            method=method,
            aux_fraction=aux_fraction,
            aux_before=aux_before,
            seed=seed,
            max_aux=max_aux,
            max_target=max_target,
            split_unit=split_unit,
            unit_field=unit_field,
        )
        return

    records = read_jsonl(input_path)
    aux: list[dict[str, Any]] = []
    target: list[dict[str, Any]] = []

    if method == "date":
        # Date splits are preferred when they match the deployment story: older
        # public/auxiliary data define atoms; newer held-out records are treated
        # as the protected target corpus.
        if aux_before is None:
            raise ValueError("--aux-before is required when --method date")
        cutoff = _parse_date(aux_before)
        if cutoff is None:
            raise ValueError("Could not parse --aux-before; expected YYYY-MM-DD")
        missing_dates = 0
        for r in records:
            d = _record_date(r)
            if d is None:
                missing_dates += 1
                # Missing dates go to target to avoid silently contaminating the
                # auxiliary atom source with unknown temporal status.
                target.append(r)
            elif d < cutoff:
                aux.append(r)
            else:
                target.append(r)
    elif method == "hash":
        # Hash splitting gives reproducible disjoint splits when no meaningful
        # temporal boundary is available.
        if not (0.0 < aux_fraction < 1.0):
            raise ValueError("--aux-fraction must be in (0, 1)")
        missing_dates = None
        for r in records:
            (aux if _hash_unit(r, seed, split_unit, unit_field) < aux_fraction else target).append(r)
    else:
        raise ValueError("method must be 'hash' or 'date'")

    if max_aux is not None:
        aux = aux[: int(max_aux)]
    if max_target is not None:
        target = target[: int(max_target)]

    write_jsonl(aux, aux_output)
    write_jsonl(target, target_output)

    manifest = {
        "input": input_path,
        "aux_output": aux_output,
        "target_output": target_output,
        "method": method,
        "aux_fraction": aux_fraction if method == "hash" else None,
        "aux_before": aux_before if method == "date" else None,
        "seed": seed,
        "num_input_records": len(records),
        "num_auxiliary_records": len(aux),
        "num_target_records": len(target),
        "max_aux": max_aux,
        "max_target": max_target,
        "split_unit": split_unit,
        "unit_field": unit_field,
        "missing_dates_assigned_to_target": missing_dates,
        "streaming_split": False,
        "privacy_note": (
            "Use aux_output only for public/auxiliary atom construction. Use "
            "target_output as the protected target corpus for DP sketch release."
        ),
    }
    manifest_path = Path(target_output).with_suffix(Path(target_output).suffix + ".split_manifest.json")
    ensure_dir(manifest_path.parent)
    write_json(manifest, manifest_path)
    print(f"Wrote auxiliary records: {aux_output} ({len(aux)})")
    print(f"Wrote target records: {target_output} ({len(target)})")
    print(f"Wrote split manifest: {manifest_path}")


def _split_records_streaming(
    input_path: str,
    aux_output: str,
    target_output: str,
    method: str = "hash",
    aux_fraction: float = 0.5,
    aux_before: str | None = None,
    seed: int = 0,
    max_aux: int | None = None,
    max_target: int | None = None,
    split_unit: str = "record",
    unit_field: str | None = None,
) -> None:
    """Streaming implementation of the auxiliary/target split.

    This path is equivalent to the in-memory split for deterministic hash and
    date rules, but writes each record immediately instead of storing the full
    input corpus in memory. It is used for Yelp-scale JSONL files.
    """
    if method == "hash" and not (0.0 < aux_fraction < 1.0):
        raise ValueError("--aux-fraction must be in (0, 1)")
    if method == "date":
        if aux_before is None:
            raise ValueError("--aux-before is required when --method date")
        cutoff = _parse_date(aux_before)
        if cutoff is None:
            raise ValueError("Could not parse --aux-before; expected YYYY-MM-DD")
    elif method != "hash":
        raise ValueError("method must be 'hash' or 'date'")

    ensure_dir(Path(aux_output).parent)
    ensure_dir(Path(target_output).parent)

    num_input = 0
    num_aux = 0
    num_target = 0
    missing_dates = 0 if method == "date" else None

    with open(aux_output, "w", encoding="utf-8") as aux_f, open(target_output, "w", encoding="utf-8") as target_f:
        for r in _stream_jsonl(input_path):
            num_input += 1
            if method == "date":
                d = _record_date(r)
                if d is None:
                    missing_dates = int(missing_dates or 0) + 1
                    to_aux = False
                else:
                    to_aux = d < cutoff  # type: ignore[operator]
            else:
                to_aux = _hash_unit(r, seed, split_unit, unit_field) < aux_fraction

            if to_aux:
                if max_aux is None or num_aux < int(max_aux):
                    _write_jsonl_record(aux_f, r)
                    num_aux += 1
            else:
                if max_target is None or num_target < int(max_target):
                    _write_jsonl_record(target_f, r)
                    num_target += 1

    manifest = {
        "input": input_path,
        "aux_output": aux_output,
        "target_output": target_output,
        "method": method,
        "aux_fraction": aux_fraction if method == "hash" else None,
        "aux_before": aux_before if method == "date" else None,
        "seed": seed,
        "num_input_records": num_input,
        "num_auxiliary_records": num_aux,
        "num_target_records": num_target,
        "max_aux": max_aux,
        "max_target": max_target,
        "split_unit": split_unit,
        "unit_field": unit_field,
        "missing_dates_assigned_to_target": missing_dates,
        "streaming_split": True,
        "privacy_note": (
            "Use aux_output only for public/auxiliary atom construction. Use "
            "target_output as the protected target corpus for DP sketch release."
        ),
    }
    manifest_path = Path(target_output).with_suffix(Path(target_output).suffix + ".split_manifest.json")
    ensure_dir(manifest_path.parent)
    write_json(manifest, manifest_path)
    print(f"Wrote auxiliary records: {aux_output} ({num_aux})")
    print(f"Wrote target records: {target_output} ({num_target})")
    print(f"Wrote split manifest: {manifest_path}")


def main() -> None:
    """Command-line entry point for auxiliary/target splitting.
    """
    p = argparse.ArgumentParser(description="Split records into an auxiliary atom-build split and a protected target split.")
    p.add_argument("--input", required=True)
    p.add_argument("--aux-output", required=True)
    p.add_argument("--target-output", required=True)
    p.add_argument("--method", choices=["hash", "date"], default="hash")
    p.add_argument("--aux-fraction", type=float, default=0.5)
    p.add_argument("--aux-before", default=None, help="YYYY-MM-DD cutoff for date split; records before this date go to auxiliary.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-aux", type=int, default=None)
    p.add_argument("--max-target", type=int, default=None)
    p.add_argument("--split-unit", choices=["record", "user"], default="record", help="Hash split by record id or by stable user id. Use user for user-level DP datasets.")
    p.add_argument("--unit-field", default=None, help="Field used when --split-unit user, e.g. user_id or metadata.user_id.")
    p.add_argument("--streaming", action="store_true", help="Stream JSONL records one at a time instead of loading the full corpus into memory. Use for Yelp-scale corpora.")
    args = p.parse_args()
    split_records(
        args.input,
        args.aux_output,
        args.target_output,
        args.method,
        args.aux_fraction,
        args.aux_before,
        args.seed,
        args.max_aux,
        args.max_target,
        args.split_unit,
        args.unit_field,
        args.streaming,
    )


if __name__ == "__main__":
    main()
