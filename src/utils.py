"""Shared configuration, serialization, naming, and file helpers.

The functions centralize deterministic random seeding, YAML/JSON/JSONL input and output, run-directory naming, and protected-record text access. Privacy-sensitive modules still define and enforce their own data-access boundaries.
"""
from __future__ import annotations

import json
import os
import re
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Set Python and NumPy random seeds for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory path if needed and return it as a Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Any, path: str | Path) -> None:
    """Write a YAML configuration file, creating parent directories as needed.
    """
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def read_json(path: str | Path) -> Any:
    """Read and parse a JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Write JSON with UTF-8 encoding and stable indentation.
    """
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)




def iter_jsonl(path: str | Path):
    """Yield newline-delimited JSON records without materializing the file.

    For large corpora such as Yelp, loading the whole
    JSONL file into a Python list can exhaust memory before the requested public
    protocol subset is selected.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_jsonl(path: str | Path) -> int:
    """Count nonempty JSONL records without parsing the whole file."""
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON records.

    For very large files, prefer :func:`iter_jsonl` plus an explicit streaming
    selection routine so the caller does not materialize millions of records.
    """
    return list(iter_jsonl(path))


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write newline-delimited JSON records.
    """
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def slugify(text: str, max_len: int = 80) -> str:
    """Convert free text into a filesystem-safe run-name component.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "run"


def run_name(cfg: dict[str, Any]) -> str:
    """Construct the deterministic output-directory name for one normalized configuration."""
    name = cfg.get("experiment_name", "experiment")
    protected_unit = cfg.get('protected_unit', 'record')
    sample_part = (
        f"u{cfg.get('n_users_effective')}"
        if protected_unit == 'user' and cfg.get('n_users_effective') is not None
        else f"n{cfg.get('n_records_effective', cfg.get('n_records', 'all'))}"
    )
    parts = [
        slugify(name),
        f"unit{protected_unit}",
        sample_part,
    ]
    if protected_unit == 'user':
        parts.append(f"B{cfg.get('user_contribution_bound', cfg.get('user_clip_B', 'na'))}")
    parts += [
        f"m{cfg.get('m', 'all')}",
        f"K{cfg.get('K')}",
        f"L{cfg.get('L')}",
        str(cfg.get('mechanism', 'mechanism')),
        f"eps{cfg.get('epsilon_effective', cfg.get('epsilon'))}",
        f"deleff{cfg.get('delta_effective', cfg.get('delta', 'na'))}",
        str(cfg.get('adjacency', 'adj')),
        str(cfg.get('decoder_backend', 'template')),
        f"seed{cfg.get('seed')}",
    ]
    return "_".join(slugify(str(p), max_len=40) for p in parts)


def text_field(record: dict[str, Any]) -> str:
    """Return the protected record text field, raising an error if absent.
    """
    if "text" not in record:
        raise KeyError("Each record must contain a 'text' field.")
    return str(record["text"])
