"""Run cross-dataset OpenAI verbalization and release-conditioned evaluation.

The pipeline operates on completed DP-SPIN releases and selected baseline artifacts. It can:

1. verbalize released DP-SPIN semantic plans without exposing protected records;
2. generate explicitly labeled baseline summaries, including non-private
   references when requested;
3. evaluate summaries relative to the object released to each decoder; and
4. write dataset-scoped CSV, LaTeX, Markdown, and qualitative-example outputs.

The pipeline does not rerun the DP-SPIN privacy mechanism. 
Raw-record, heuristic-redaction, and target-derived clustering baselines are non-private and must not be presented as satisfying the DP-SPIN decoder boundary.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .baselines import run_baselines
from .openai_decode_representatives import (
    DEFAULT_DATASETS,
    _decode_one,
    select_runs,
)
from .summary_eval import compute_summary_text_evaluation
from .urania_baseline import run_urania_public_keyword_baseline
from .utils import ensure_dir, read_json, read_jsonl, text_field, write_json


@dataclass(frozen=True)
class DatasetSpec:
    """Describe dataset inputs and output locations for the OpenAI evaluation pipeline."""
    key: str
    dataset_name: str
    display_name: str
    output_root: str
    target_records: str
    auxiliary_records: str
    atoms: str
    category_key: str | None
    default_units: tuple[str, ...]
    default_m: int
    default_B: float | None
    baseline_subset_size: int


DATASET_SPECS: dict[str, DatasetSpec] = {
    "cfpb": DatasetSpec(
        key="cfpb",
        dataset_name="cfpb",
        display_name="CFPB consumer complaints",
        output_root="outputs/cfpb",
        target_records="data/protected_target/cfpb_target_records.jsonl",
        auxiliary_records="data/auxiliary/cfpb_atom_auxiliary_records.jsonl",
        atoms="data/public_atoms/cfpb_atoms_m200.json",
        category_key="Issue",
        default_units=("record",),
        default_m=200,
        default_B=None,
        baseline_subset_size=2000,
    ),
    "amazon": DatasetSpec(
        key="amazon",
        dataset_name="amazon_All_Beauty",
        display_name="Amazon All Beauty reviews",
        output_root="outputs/amazon_All_Beauty",
        target_records="data/protected_target/amazon_All_Beauty_target_records.jsonl",
        auxiliary_records="data/auxiliary/amazon_All_Beauty_atom_auxiliary_records.jsonl",
        atoms="data/public_atoms/amazon_All_Beauty_atoms_m46.json",
        # Rating bins define a fixed public metadata universe for review data.
        category_key="rating_bin",
        default_units=("record", "user"),
        default_m=46,
        default_B=1.0,
        baseline_subset_size=2000,
    ),
    "yelp": DatasetSpec(
        key="yelp",
        dataset_name="yelp_restaurants",
        display_name="Yelp Restaurants reviews",
        output_root="outputs/yelp_restaurants",
        target_records="data/protected_target/yelp_restaurants_target_records.jsonl",
        auxiliary_records="data/auxiliary/yelp_restaurants_atom_auxiliary_records.jsonl",
        atoms="data/public_atoms/yelp_restaurants_atoms_m100.json",
        # Business category is nearly fixed by the Restaurants filter; rating bins
        # are more informative for a category-histogram baseline.
        category_key="rating_bin",
        default_units=("record", "user"),
        default_m=100,
        default_B=1.0,
        baseline_subset_size=2000,
    ),
}

MECHANISM_DISPLAY = {
    "gaussian": r"Gaussian sketch + top-$L$",
    "laplace": r"Laplace sketch + top-$L$",
    "private_topL": r"Private top-$L$ admission + Gaussian values",
}

BASELINE_SUMMARY_FILES = {
    "OpenAI raw records": "raw_summary.txt",
    "OpenAI heuristic redaction": "heuristic_redacted_summary.txt",
    "OpenAI non-private clustering": "non_private_clustering_summary.txt",
    "DP keyword histogram": "dp_keyword_summary.txt",
    "DP category histogram": "dp_category_summary.txt",
    "URANIA-style public keywords": "urania_cluster_public_keyword_summary.txt",
}

BASELINE_METHOD_TO_KEY = {
    "OpenAI raw records": "raw",
    "OpenAI heuristic redaction": "redacted",
    "OpenAI non-private clustering": "non_private_clustering",
    "DP keyword histogram": "dp_keyword",
    "DP category histogram": "dp_category",
    "URANIA-style public keywords": "urania",
}

DEFAULT_PAPER_METHODS = [
    "DP-SPIN private plan",
    "DP keyword histogram",
    "DP category histogram",
    "URANIA-style public keywords",
]

GENERIC_SCORE_COLS = ["coverage", "specificity", "insightfulness", "faithfulness", "privacy_safety", "clarity"]
DPSPIN_SCORE_COLS = ["coverage", "faithfulness", "usefulness", "clarity"]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _short(text: str, n: int = 900) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def _display_text_for_item(item: dict[str, Any], text: str) -> str:
    """Return generated summary text for qualitative tables.
    
    For older URANIA-style outputs, cluster identifiers, exact noisy sizes, and
    method boilerplate are removed from the displayed prose. The complete released
    cluster object remains stored separately.
    """
    method = str(item.get("method") or "")
    if method != "URANIA-style public keywords":
        return text

    src = text or ""
    # Newer URANIA summaries already contain aggregate prose. Only remove
    # legacy/debug renderings that expose cluster IDs, noisy sizes, XML tags, or
    # method boilerplate.
    legacy_pattern = r"Cluster\s+\d+|noisy size|URANIA-style released cluster summaries|<\s*topic\s*>|<\s*description\s*>|public-keyword baseline returns"
    if not re.search(legacy_pattern, src, flags=re.IGNORECASE):
        return src.strip()
    # Remove wrapper sentences added by older renderers; the table row already
    # identifies the method.
    src = re.sub(
        r"The URANIA-style public-keyword baseline returns retained cluster-level themes from released public keywords\.\s*",
        "",
        src,
        flags=re.IGNORECASE,
    )
    src = re.sub(
        r"The representative retained themes are:\s*",
        "",
        src,
        flags=re.IGNORECASE,
    )
    src = re.sub(r"URANIA-style released cluster summaries:\s*", "", src, flags=re.IGNORECASE)

    # Older outputs may contain one long paragraph with explicit cluster IDs and
    # noisy sizes. Split at those markers before cleaning.
    segments = re.split(r"\s*-\s*Cluster\s+\d+\s*\((?:released\s+)?noisy size [^)]+\):\s*", src, flags=re.IGNORECASE)
    if len(segments) <= 1:
        segments = [z for z in re.split(r"\n\s*-\s*", src) if z.strip()]

    themes: list[str] = []
    for segment in segments:
        line = re.sub(r"\s+", " ", str(segment).strip())
        if not line:
            continue
        line = re.sub(r"\bThe noisy cluster size is [0-9.]+\.?", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\bThe noisy size is [0-9.]+\.?", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\bnoisy cluster size is [0-9.]+\.?", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\breleased noisy size [0-9.]+\.?", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+", " ", line).strip(" -")
        if not line:
            continue
        topic_match = re.search(r"Topic:\s*([^.]*)\s*Description:\s*(.*)", line, flags=re.IGNORECASE)
        if topic_match:
            topic = topic_match.group(1).strip()
            desc = topic_match.group(2).strip()
            if topic and desc:
                themes.append(f"- {topic}: {desc}")
            elif topic:
                themes.append(f"- {topic}")
        else:
            themes.append("- " + line)

    # Remove accidental duplicate lines while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for theme in themes:
        key = theme.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(theme)
    deduped = deduped[:3]
    if not deduped:
        return src.strip()
    return "Retained cluster-level themes include:\n" + "\n".join(deduped)

def _latex_escape(value: Any) -> str:
    s = "" if value is None else str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _latex_escape_preserving_math(value: Any) -> str:
    r"""Escape LaTeX table text while preserving supported inline math.

    The generated report uses a few intentional LaTeX snippets in table cells,
    such as ``Top-$L$ atoms`` and ``$\pm$``. The ordinary escape routine is
    still needed for arbitrary text, but escaping these snippets produces invalid
    displays such as ``Top-\$L\$`` or ``$\textbackslash{}pm\$``. This helper
    escapes only text outside balanced single-dollar math spans.
    """
    s = "" if value is None else str(value)
    if "$" not in s:
        return _latex_escape(s)
    parts = re.split(r"(\$[^$]*\$)", s)
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if len(part) >= 2 and part.startswith("$") and part.endswith("$"):
            out.append(part)
        else:
            out.append(_latex_escape(part))
    return "".join(out)


def _fmt_mean_std(mean: Any, std: Any = None, count: int | None = None) -> str:
    try:
        if pd.isna(mean):
            return "--"
        mean_f = float(mean)
    except Exception:
        return "--"
    if std is None or pd.isna(std) or (count is not None and count <= 1):
        return f"{mean_f:.2f}"
    return f"{mean_f:.2f} $\\pm$ {float(std):.2f}"


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def _safe_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None


def _effective_protected_units(row: dict[str, Any] | pd.Series) -> int | None:
    """Return the effective number of protected units for a run row."""
    unit = str(row.get("protected_unit") or "").lower()
    if unit == "user":
        return _safe_int(row.get("n_users_effective") or row.get("n_users"))
    return _safe_int(row.get("n_records_effective") or row.get("n_records") or row.get("subset_records"))


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    try:
        return read_json(path) if path.exists() else {}
    except Exception:
        return {}


def _count_jsonl_if_exists(path: str | Path | None) -> int | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return sum(1 for line in p.open("r", encoding="utf-8") if line.strip())
    except Exception:
        return None


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "dataset", "display_name", "protected_unit", "method", "method_family", "mechanism", "mechanism_display",
        "epsilon", "seed", "m", "K", "L", "category_key", "num_clusters_requested", "keywords_per_summary",
        "user_contribution_bound", "n_records_effective", "n_users_effective", "baseline_subset_records", "summary_path",
        "plan_path", "source_run", "output_run",
    ]
    cols = sorted({k for r in rows for k in r})
    cols = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)




def _normalize_method_filter(methods: Iterable[str] | None) -> set[str] | None:
    if methods is None:
        return None
    out = {str(m).strip() for m in methods if str(m).strip()}
    if not out or "all" in out:
        return None
    return out


def _item_method_name(item: dict[str, Any] | pd.Series) -> str:
    method = str(item.get("method") or "")
    family = str(item.get("method_family") or "")
    if method == "DP-SPIN private plan" or family == "DP-SPIN":
        return "DP-SPIN private plan"
    if method == "Non-private-plan reference" or family == "non_private_reference":
        return "Non-private-plan reference"
    return method


def _filter_items_by_method(items: list[dict[str, Any]], methods: Iterable[str] | None) -> list[dict[str, Any]]:
    allowed = _normalize_method_filter(methods)
    if allowed is None:
        return list(items)
    return [item for item in items if _item_method_name(item) in allowed]


def _filter_df_by_method(df: pd.DataFrame, methods: Iterable[str] | None) -> pd.DataFrame:
    allowed = _normalize_method_filter(methods)
    if allowed is None or df.empty or "method" not in df.columns:
        return df
    return df[df.apply(lambda row: _item_method_name(row) in allowed, axis=1)].copy()


def _sample_jsonl(input_path: Path, output_path: Path, n: int, seed: int) -> int:
    ensure_dir(output_path.parent)
    records = input_path.read_text(encoding="utf-8").splitlines()
    records = [line for line in records if line.strip()]
    if n <= 0 or len(records) <= n:
        selected = records
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(records), size=n, replace=False))
        selected = [records[int(i)] for i in idx]
    output_path.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
    return len(selected)


def _dataset_openai_dir(spec: DatasetSpec, report_name: str) -> Path:
    return Path(spec.output_root) / report_name


# -----------------------------------------------------------------------------
# Dataset-scoped report helpers.
# These helpers keep single-dataset reports from overwriting each other and make
# generated LaTeX captions/labels unambiguous.
# -----------------------------------------------------------------------------

def _latex_label_slug(value: Any) -> str:
    """Return a stable lower-case slug suitable for LaTeX labels."""
    text = str(value or "report").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "report"


def _dataset_scope_name(specs: list[DatasetSpec], dataset_arg: str) -> str:
    """Directory name for this report scope."""
    if dataset_arg != "all" and len(specs) == 1:
        return specs[0].dataset_name
    if len(specs) == 1:
        return specs[0].dataset_name
    return "all_datasets"


def _dataset_caption_from_names(names: list[str]) -> str:
    """Format the selected dataset names for a table caption."""
    unique = [n for i, n in enumerate(names) if n and n not in names[:i]]
    if not unique:
        return "selected datasets"
    slug_to_display = {spec.dataset_name: spec.display_name for spec in DATASET_SPECS.values()}
    display = [slug_to_display.get(n, n.replace("_", " ")) for n in unique]
    if len(display) == 1:
        return display[0]
    return "selected datasets"


def _dataset_caption_name(obj: Any) -> str:
    """Dataset phrase for a DataFrame or a list of row dictionaries/items."""
    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return "selected datasets"
        if "display_name" in obj.columns:
            vals = [str(x) for x in obj["display_name"].dropna().unique().tolist() if str(x)]
            if len(vals) == 1:
                return vals[0]
            if len(vals) > 1:
                return "selected datasets"
        if "dataset" in obj.columns:
            vals = [str(x) for x in obj["dataset"].dropna().unique().tolist() if str(x)]
            return _dataset_caption_from_names(vals)
        return "selected datasets"
    if isinstance(obj, list):
        vals = []
        display_vals = []
        for item in obj:
            if isinstance(item, dict):
                ds = str(item.get("dataset") or "")
                dn = str(item.get("display_name") or "")
                if ds and ds not in vals:
                    vals.append(ds)
                if dn and dn not in display_vals:
                    display_vals.append(dn)
        if len(display_vals) == 1:
            return display_vals[0]
        if len(display_vals) > 1:
            return "selected datasets"
        return _dataset_caption_from_names(vals)
    return "selected datasets"


def _dataset_scope_from_df(df: pd.DataFrame) -> tuple[str, str, str]:
    """Return (dataset_slug, caption_phrase, latex_label_suffix)."""
    phrase = _dataset_caption_name(df)
    if isinstance(df, pd.DataFrame) and not df.empty and "dataset" in df.columns:
        vals = [str(x) for x in df["dataset"].dropna().unique().tolist() if str(x)]
        slug = vals[0] if len(vals) == 1 else "all_datasets"
    else:
        slug = "all_datasets"
    return slug, phrase, _latex_label_slug(slug if slug != "all_datasets" else phrase)


def _write_dataset_named_copy(path: Path, dataset_slug: str) -> None:
    """Write an additional dataset-suffixed copy of a generated report file."""
    if not path.exists():
        return
    suffix = path.suffix
    stem = path.stem
    dataset_slug = str(dataset_slug or "dataset")
    named = path.with_name(f"{stem}_{dataset_slug}{suffix}")
    if named == path:
        return
    named.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def _write_named_latex(path: Path, dataset_slug: str, base_name: str, content: str) -> None:
    """Write the canonical LaTeX output file for one report table."""
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def _decode_dpspin_for_dataset(
    spec: DatasetSpec,
    out_dir: Path,
    mechanisms: list[str],
    epsilons: list[float],
    seeds: list[int],
    units: list[str] | None,
    m_values: list[int] | None,
    user_bounds: list[float] | None,
    n_records_values: list[int] | None,
    n_users_values: list[int] | None,
    K: int,
    L: int,
    model: str,
    temperature: float,
    max_tokens: int,
    force: bool,
    judge_backend: str,
    comparative_backend: str,
    summary_embedding_backend: str,
    max_runs: int | None,
) -> list[dict[str, Any]]:
    selected = select_runs(
        spec.key,
        mechanisms=mechanisms,
        epsilons=epsilons,
        seeds=seeds,
        units=units or list(spec.default_units),
        m_values=m_values or [spec.default_m],
        K=K,
        L=L,
        B_values=user_bounds if user_bounds is not None else ([spec.default_B] if spec.default_B is not None else None),
        n_records_values=n_records_values,
        n_users_values=n_users_values,
        max_runs=max_runs,
    )
    rows: list[dict[str, Any]] = []
    dpspin_root = out_dir / "dpspin"
    for run, _cfg in selected:
        print(f"[DP-SPIN] {spec.dataset_name}: decoding {run}")
        row = _decode_one(
            run,
            dpspin_root,
            dataset=spec.dataset_name,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            force=force,
            judge_backend=judge_backend,
            comparative_backend=comparative_backend,
            summary_embedding_backend=summary_embedding_backend,
        )
        row["display_name"] = spec.display_name
        row["method"] = "DP-SPIN"
        row["mechanism_display"] = MECHANISM_DISPLAY.get(str(row.get("mechanism")), str(row.get("mechanism")))
        row["output_run"] = row.get("openai_run")
        rows.append(row)
    _write_csv(rows, out_dir / "tables" / "dpspin_openai_runs.csv")
    return rows




def _baseline_outputs_current(marker: Path, baseline_methods: Iterable[str], backend: str, model: str) -> bool:
    if not marker.exists():
        return False
    try:
        meta = read_json(marker)
    except Exception:
        return False
    done = set(str(x) for x in meta.get("baseline_methods", []))
    requested = set(str(x) for x in baseline_methods)
    return (
        requested.issubset(done)
        and str(meta.get("backend")) == str(backend)
        and str(meta.get("model")) == str(model)
    )


def _run_baselines_for_dataset(
    spec: DatasetSpec,
    out_dir: Path,
    model: str,
    epsilon: float,
    seed: int,
    subset_size: int,
    force: bool,
    urania_epsilon_total: float,
    urania_delta: float,
    urania_clusters: int,
    urania_keywords: int,
    baseline_methods: list[str],
    run_urania: bool,
) -> dict[str, Any]:
    baseline_root = ensure_dir(out_dir / "baselines")
    data_dir = ensure_dir(baseline_root / "data")
    subset_path = data_dir / f"{spec.dataset_name}_baseline_subset_{subset_size}_seed{seed}.jsonl"
    if force or not subset_path.exists():
        n_written = _sample_jsonl(Path(spec.target_records), subset_path, subset_size, seed)
    else:
        n_written = sum(1 for _ in subset_path.open("r", encoding="utf-8"))

    out = ensure_dir(baseline_root / spec.dataset_name / f"seed{seed}")
    marker = out / "baseline_metadata.json"
    if force or not _baseline_outputs_current(marker, baseline_methods, backend="openai", model=model):
        print(f"[Baselines] {spec.dataset_name}: running OpenAI baselines on {n_written} records")
        run_baselines(
            records_path=str(subset_path),
            atoms_path=spec.atoms,
            output_dir=str(out),
            epsilon=epsilon,
            seed=seed,
            backend="openai",
            model=model,
            category_key=spec.category_key,
            adjacency="adddrop",
            category_universe_records=spec.auxiliary_records if spec.category_key else None,
            category_universe_json=None,
            max_keywords=200,
            baseline_methods=baseline_methods,
        )
    else:
        print(f"[Baselines] {spec.dataset_name}: baseline outputs already exist")

    urania_summary = out / "urania_cluster_public_keyword_summary.txt"
    if run_urania and (force or not urania_summary.exists()):
        print(f"[URANIA-style] {spec.dataset_name}: running public-keyword baseline")
        run_urania_public_keyword_baseline(
            records_path=str(subset_path),
            aux_records_path=spec.auxiliary_records,
            atoms_path=spec.atoms,
            output_dir=str(out),
            epsilon_total=urania_epsilon_total,
            delta=urania_delta,
            num_clusters=urania_clusters,
            keywords_per_summary=urania_keywords,
            max_keywords=200,
            max_keywords_per_record=5,
            sample_per_cluster=50,
            extra_public_keywords=[],
            cluster_size_threshold=None,
            svd_dim=64,
            dp_iterations=3,
            seed=seed,
            backend="openai",
            model=model,
            n_records=None,
            cluster_fraction=0.45,
            size_fraction=0.15,
            embedding_mode="public_tfidf_svd",
            cluster_backend="repository_noisy_lloyd",
            keyword_selector="lexical",
            histogram_mechanism="discrete_laplace",
        )
    elif run_urania:
        print(f"[URANIA-style] {spec.dataset_name}: output already exists")
    else:
        print(f"[URANIA-style] {spec.dataset_name}: skipped")

    return {
        "dataset": spec.dataset_name,
        "display_name": spec.display_name,
        "subset_path": str(subset_path),
        "subset_records": n_written,
        "baseline_dir": str(out),
        "seed": int(seed),
        "baseline_epsilon": float(epsilon),
        "urania_epsilon_total": float(urania_epsilon_total),
    }


def _baseline_rows(spec: DatasetSpec, baseline_dir: Path, allowed_methods: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Collect baseline summary rows and the privacy parameters used for each release.

    The DP keyword/category baselines and the URANIA-style public-keyword baseline
    have their own privacy parameters. These parameters are stored in the
    baseline metadata files and should not be left as missing values in the
    manuscript tables or judge logs.
    """
    rows: list[dict[str, Any]] = []
    allowed = _normalize_method_filter(allowed_methods)
    baseline_meta = _read_json_if_exists(baseline_dir / "baseline_metadata.json")
    urania_meta = _read_json_if_exists(baseline_dir / "urania_cluster_public_keyword_metrics.json")
    subset_records = _safe_int(baseline_meta.get("subset_records"))
    if subset_records is None:
        subset_records = _count_jsonl_if_exists(baseline_meta.get("records_path"))

    # Read plan metadata needed for the table configuration columns.
    dp_keyword_plan = _read_json_if_exists(baseline_dir / "dp_keyword_plan.json")
    dp_category_plan = _read_json_if_exists(baseline_dir / "dp_category_plan.json")
    urania_plan = _read_json_if_exists(baseline_dir / "urania_cluster_public_keyword_plan.json")
    method_meta = {
        "DP keyword histogram": {
            "mechanism": "dp_keyword_histogram",
            "mechanism_display": "DP keyword histogram",
            "epsilon": baseline_meta.get("epsilon"),
            "delta": baseline_meta.get("delta"),
            "protected_unit": "record",
            "n_records_effective": subset_records,
            # Number of noisy keyword coordinates displayed to the decoder.  In
            # the current baseline implementation this is top-10 unless changed
            # in src.baselines.  Do not confuse this with the full fixed public
            # keyword vocabulary size.
            "L": dp_keyword_plan.get("L") or baseline_meta.get("L"),
            "vocabulary_size": baseline_meta.get("max_keywords"),
            "plan_path": str(baseline_dir / "dp_keyword_plan.json") if (baseline_dir / "dp_keyword_plan.json").exists() else None,
        },
        "DP category histogram": {
            "mechanism": "dp_category_histogram",
            "mechanism_display": "DP category histogram",
            "epsilon": baseline_meta.get("epsilon"),
            "delta": baseline_meta.get("delta"),
            "protected_unit": "record",
            "n_records_effective": subset_records,
            # Number of noisy category coordinates displayed to the decoder.
            "L": dp_category_plan.get("L") or baseline_meta.get("L"),
            "category_key": baseline_meta.get("category_key"),
            "category_universe_size": (dp_category_plan.get("mechanism_info") or {}).get("num_categories"),
            "plan_path": str(baseline_dir / "dp_category_plan.json") if (baseline_dir / "dp_category_plan.json").exists() else None,
        },
        "URANIA-style public keywords": {
            "mechanism": "urania_public_keywords",
            "mechanism_display": "URANIA-style public keywords",
            "epsilon": urania_meta.get("epsilon_total"),
            "delta": urania_meta.get("delta_total"),
            "protected_unit": "record",
            "n_records_effective": _safe_int(urania_meta.get("num_records")) or subset_records,
            "num_clusters_requested": (urania_plan.get("parameters") or {}).get("num_clusters_requested") or urania_meta.get("num_clusters_requested"),
            "keywords_per_summary": (urania_plan.get("parameters") or {}).get("keywords_per_summary") or urania_meta.get("keywords_per_summary"),
            "plan_path": str(baseline_dir / "urania_cluster_public_keyword_plan.json") if (baseline_dir / "urania_cluster_public_keyword_plan.json").exists() else None,
        },
    }
    for method, fname in BASELINE_SUMMARY_FILES.items():
        if allowed is not None and method not in allowed:
            continue
        p = baseline_dir / fname
        if p.exists():
            meta = method_meta.get(method, {})
            row = {
                "dataset": spec.dataset_name,
                "display_name": spec.display_name,
                "method": method,
                "summary_path": str(p),
                "baseline_dir": str(baseline_dir),
                "seed": baseline_meta.get("seed"),
                "baseline_subset_records": subset_records,
                "is_private_output": method.startswith("DP ") or "URANIA" in method,
            }
            row.update({k: v for k, v in meta.items() if v is not None})
            rows.append(row)
    return rows



def _row_dataset_spec(row: dict[str, Any], specs_by_dataset: dict[str, DatasetSpec]) -> DatasetSpec | None:
    dataset = str(row.get("dataset") or "")
    return specs_by_dataset.get(dataset)


def _row_preference_score(row: dict[str, Any], specs_by_dataset: dict[str, DatasetSpec]) -> tuple:
    spec = _row_dataset_spec(row, specs_by_dataset)
    target_n = spec.baseline_subset_size if spec is not None else None
    n = _effective_protected_units(row)
    run = str(row.get("source_run") or row.get("run") or row.get("output_run") or "").lower()
    if target_n is not None and n is not None:
        n_score = abs(int(n) - int(target_n))
    elif n is not None:
        n_score = int(n)
    else:
        n_score = 10**12
    v2_penalty = 1 if "v2" in run else 0
    deleff_penalty = 1 if "deleff" in run else 0
    probe_penalty = 1 if "probe" in run else 0
    return (probe_penalty, n_score, v2_penalty, deleff_penalty, run)


def _dedupe_dpspin_rows(
    rows: list[dict[str, Any]],
    specs: list[DatasetSpec],
    *,
    keep_record_counts: bool,
    keep_user_counts: bool,
) -> list[dict[str, Any]]:
    """Collapse stale duplicate DP-SPIN OpenAI rows.

    Saved output directories may contain several historical folders with the
    same logical setting.  Unless the caller explicitly asks for several
    protected-unit counts, we keep one representative row per
    dataset/unit/mechanism/epsilon/seed/K/L/m/B.
    """
    specs_by_dataset = {spec.dataset_name: spec for spec in specs}
    best: dict[tuple, tuple[tuple, dict[str, Any]]] = {}
    for row in rows:
        unit = str(row.get("protected_unit") or "")
        key = [
            row.get("dataset"),
            unit,
            row.get("mechanism"),
            _safe_float(row.get("epsilon")),
            _safe_int(row.get("seed")),
            _safe_int(row.get("m")),
            _safe_int(row.get("K")),
            _safe_int(row.get("L")),
        ]
        if unit == "user":
            key.append(_safe_float(row.get("user_contribution_bound")))
            if keep_user_counts:
                key.append(_effective_protected_units(row))
        elif keep_record_counts:
            key.append(_effective_protected_units(row))
        key_t = tuple(key)
        score = _row_preference_score(row, specs_by_dataset)
        old = best.get(key_t)
        if old is None or score < old[0]:
            best[key_t] = (score, row)
    out = [row for _score, row in best.values()]
    out.sort(key=lambda r: (
        str(r.get("dataset") or ""),
        str(r.get("protected_unit") or ""),
        str(r.get("mechanism") or ""),
        _safe_float(r.get("epsilon")) or 0,
        _safe_int(r.get("seed")) or 0,
        _effective_protected_units(r) or 0,
    ))
    return out


def _openai_judge(prompt: str, model: str) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only strict valid JSON. Do not wrap in markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    txt = resp.choices[0].message.content or "{}"
    try:
        return json.loads(txt)
    except Exception:
        return {"raw_response": txt, "parse_error": True}


def _heuristic_generic_judge(summary: str, has_plan: bool) -> dict[str, Any]:
    """Compute the deterministic release-conditioned fallback scores."""
    summary = summary or ""
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", summary)
    length = len(words)
    lower = summary.lower()
    bullet_count = len(re.findall(r"(?:^|\n)\s*[-*]", summary))
    coverage = 3 if length < 60 else 4 if length < 260 else 3
    semantic_terms = ["problem", "issue", "concern", "theme", "report", "credit", "debt", "service", "quality", "price", "delivery", "food", "skin"]
    specificity = 2 + int(any(tok in lower for tok in semantic_terms)) + int(bullet_count >= 2)
    insightfulness = 2 + int("keyword" not in lower) + int(any(tok in lower for tok in ["incorrect", "investigation", "collection", "account", "review", "rating", "restaurant"]))
    faithfulness = 4 if has_plan else 3
    privacy_safety = 5 - int(bool(re.search(r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b|[\w.\-]+@[\w.\-]+", summary)))
    clarity = 4 if 35 <= length <= 260 else 3
    return {
        "coverage": max(1, min(5, coverage)),
        "specificity": max(1, min(5, specificity)),
        "insightfulness": max(1, min(5, insightfulness)),
        "faithfulness": max(1, min(5, faithfulness)),
        "privacy_safety": max(1, min(5, privacy_safety)),
        "clarity": max(1, min(5, clarity)),
        "rationale": "Heuristic fallback judge based on summary length, semantic terms, and simple privacy-pattern checks.",
    }

def _released_object_label(method: str) -> str:
    return {
        "DP-SPIN private plan": "released differentially private semantic plan",
        "DP keyword histogram": "released differentially private keyword histogram over a fixed public vocabulary",
        "DP category histogram": "released differentially private category histogram over a fixed public category universe",
        "URANIA-style public keywords": "released URANIA-style public-keyword cluster object",
        "Non-private-plan reference": "non-private reference plan used only for evaluation",
        "OpenAI raw records": "non-private raw-record summary input",
        "OpenAI heuristic redaction": "heuristically redacted non-private record input",
        "OpenAI non-private clustering": "non-private target-derived cluster input",
    }.get(method, "released object")


def _judge_item(item: dict[str, Any], backend: str, model: str) -> dict[str, Any]:
    """Judge one generated summary against its own released object.

    The judge is release-conditioned: it sees the generated summary, the method
    name, and the exact released object made available to that decoder. It does
    not see raw protected records. The resulting scores are secondary
    presentation measures and should not be used as formal privacy or utility
    guarantees.
    """
    summary = _read_text(Path(item["summary_path"]))
    plan_text = _read_text(Path(item.get("plan_path", ""))) if item.get("plan_path") else ""
    method = str(item.get("method") or "")
    decoder_input = _method_decoder_input(method)
    release_label = _released_object_label(method)
    if backend == "none":
        out: dict[str, Any] = {}
    elif backend == "heuristic":
        out = _heuristic_generic_judge(summary, bool(plan_text))
    else:
        prompt = f"""
You are evaluating a generated aggregate insight summary.

You must judge the summary only with respect to the released object shown below.
Do not assume access to the protected corpus. Do not reward or penalize a method
for information that is absent from its released object. Do not treat this as a
formal differential-privacy audit.

Return strict valid JSON with integer scores from 1 to 5 for exactly these keys:
- coverage: Does the summary cover the main high-salience entries in the released object?
- specificity: Does the summary give concrete theme-level detail permitted by the released object, rather than only generic wording or a list of isolated keywords?
- insightfulness: Does the summary synthesize the released entries into useful collection-level insight, beyond merely repeating labels or counts?
- faithfulness: Does the summary avoid adding themes, examples, causes, entities, comparisons, or record-level details unsupported by the released object?
- privacy_safety: Does the generated text avoid individual-level, identifying, rare-record, raw-example, or quote-like details? This is a text-quality score, not a formal DP guarantee.
- clarity: Is the summary readable, concise, and organized?

Scoring calibration:
1. A summary that faithfully repeats a short keyword histogram may receive high faithfulness and privacy_safety, but should receive high specificity or insightfulness only if it turns released keywords into clearly supported aggregate themes.
2. A public category histogram can be highly useful when the public categories are themselves semantically informative. Do not automatically penalize it for being a histogram.
3. A DP-SPIN semantic plan can receive high specificity or insightfulness only when the generated summary uses the admitted atom labels, descriptions, and released bins coherently.
4. Penalize any method for unsupported causal claims, invented examples, exact private details, or unsupported comparisons.
5. Return only valid JSON with the six integer score keys and a short rationale string.

Dataset: {item.get('display_name')}
Method: {method}
Decoder input: {decoder_input}
Released object type: {release_label}
Mechanism: {item.get('mechanism')}
Epsilon: {item.get('epsilon')}
Protected unit: {item.get('protected_unit')}
Protected units: {_effective_protected_units(item)}

Released object:
{plan_text[:9000]}

Generated summary:
{summary[:7000]}
"""
        out = _openai_judge(prompt, model=model)
    # Preserve parsed scores and attach metadata used by report tables.
    out.update({
        "dataset": item.get("dataset"),
        "display_name": item.get("display_name"),
        "method": item.get("method"),
        "method_family": item.get("method_family"),
        "mechanism": item.get("mechanism"),
        "mechanism_display": item.get("mechanism_display"),
        "epsilon": item.get("epsilon"),
        "seed": item.get("seed"),
        "m": item.get("m"),
        "K": item.get("K"),
        "L": item.get("L"),
        "category_key": item.get("category_key"),
        "num_clusters_requested": item.get("num_clusters_requested"),
        "keywords_per_summary": item.get("keywords_per_summary"),
        "protected_unit": item.get("protected_unit"),
        "n_records_effective": item.get("n_records_effective"),
        "n_users_effective": item.get("n_users_effective"),
        "user_contribution_bound": item.get("user_contribution_bound"),
        "baseline_subset_records": item.get("baseline_subset_records"),
        "summary_path": item.get("summary_path"),
        "plan_path": item.get("plan_path"),
        "decoder_input": decoder_input,
        "release_object_type": release_label,
    })
    return out



def _filter_comparison_judge_items(
    items: list[dict[str, Any]],
    *,
    dpspin_mechanism: str | None,
    dpspin_epsilon: float | None,
) -> list[dict[str, Any]]:
    """Restrict generic comparison judging to the matched DP-SPIN condition.

    The DP-SPIN mechanism-by-epsilon table is built from all decoded DP-SPIN
    representatives.  The cross-method baseline-comparison table should not
    aggregate all those settings against a single baseline run.  This filter
    keeps only the specified DP-SPIN representative condition for the generic
    judge while leaving all selected baseline methods unchanged.
    """
    out: list[dict[str, Any]] = []
    for item in items:
        method = str(item.get("method") or "")
        family = str(item.get("method_family") or "")
        if method == "DP-SPIN private plan" or family == "DP-SPIN":
            if dpspin_mechanism and str(item.get("mechanism")) != str(dpspin_mechanism):
                continue
            if dpspin_epsilon is not None:
                eps = _safe_float(item.get("epsilon"))
                if eps is None or abs(eps - float(dpspin_epsilon)) > 1e-9:
                    continue
        out.append(item)
    return out


def _judge_all(items: list[dict[str, Any]], out_dir: Path, backend: str, model: str, force: bool) -> pd.DataFrame:
    ensure_dir(out_dir)
    json_path = out_dir / "judgments.json"
    csv_path = out_dir / "judgments.csv"
    if not force and csv_path.exists():
        return pd.read_csv(csv_path)
    results = []
    for item in items:
        if not item.get("summary_path") or not Path(str(item["summary_path"])).exists():
            continue
        print(f"[Judge] {item.get('dataset')} / {item.get('method')} / {item.get('mechanism')} / eps={item.get('epsilon')}")
        results.append(_judge_item(item, backend=backend, model=model))
    write_json(results, json_path)
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    if not df.empty:
        numeric = [c for c in GENERIC_SCORE_COLS if c in df.columns]
        if numeric:
            df.groupby(["dataset", "method", "mechanism", "epsilon"], dropna=False)[numeric].agg(["mean", "std", "count"]).to_csv(out_dir / "judgments_aggregate.csv")
    return df


def _collect_items(dpspin_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect presentation-oriented summaries for generic judging and qualitative tables.

    DP-SPIN rows point to the display summary. Audit summaries are preserved in
    row metadata and are used for verifier measures, not presentation display.
    """
    items: list[dict[str, Any]] = []
    for row in dpspin_rows:
        openai_run = Path(str(row.get("openai_run", "")))
        display_path = row.get("summary_display_path") or row.get("summary_path") or str(openai_run / "summary_openai_display.txt")
        audit_path = row.get("summary_audit_path") or str(openai_run / "summary_openai_audit.txt")
        np_display_path = row.get("non_private_reference_summary_display_path") or row.get("non_private_reference_summary_path") or str(openai_run / "non_private_reference_summary_openai_display.txt")
        items.append({
            "dataset": row.get("dataset"),
            "display_name": row.get("display_name"),
            "protected_unit": row.get("protected_unit"),
            "method": "DP-SPIN private plan",
            "method_family": "DP-SPIN",
            "mechanism": row.get("mechanism"),
            "mechanism_display": row.get("mechanism_display"),
            "epsilon": row.get("epsilon"),
            "seed": row.get("seed"),
            "m": row.get("m"),
            "K": row.get("K"),
            "L": row.get("L"),
            "n_records_effective": row.get("n_records_effective"),
            "n_users_effective": row.get("n_users_effective"),
            "user_contribution_bound": row.get("user_contribution_bound"),
            "summary_path": display_path,
            "summary_display_path": display_path,
            "summary_audit_path": audit_path,
            "plan_path": row.get("plan_path") or str(openai_run / "private_plan.json"),
            "non_private_reference_summary_path": np_display_path,
            "non_private_reference_plan_path": row.get("non_private_reference_plan_path") or str(openai_run / "non_private_reference_plan.json"),
            "is_private_output": True,
        })
        # Include non-private-plan reference for qualitative comparison but do not call
        # it DP-SPIN output. It is an offline non-private reference generated
        # from the non-private top-L plan.
        if np_display_path and Path(str(np_display_path)).exists():
            items.append({
                "dataset": row.get("dataset"),
                "display_name": row.get("display_name"),
                "protected_unit": row.get("protected_unit"),
                "method": "Non-private-plan reference",
                "method_family": "non_private_reference",
                "mechanism": row.get("mechanism"),
                "mechanism_display": row.get("mechanism_display"),
                "epsilon": row.get("epsilon"),
                "seed": row.get("seed"),
                "m": row.get("m"),
                "K": row.get("K"),
                "L": row.get("L"),
                "n_records_effective": row.get("n_records_effective"),
                "n_users_effective": row.get("n_users_effective"),
                "user_contribution_bound": row.get("user_contribution_bound"),
                "summary_path": np_display_path,
                "summary_display_path": np_display_path,
                "plan_path": row.get("non_private_reference_plan_path") or str(openai_run / "non_private_reference_plan.json"),
                "is_private_output": False,
            })
    for row in baseline_rows:
        row = dict(row)
        row.setdefault("method_family", "baseline")
        row.setdefault("is_private_output", row.get("method", "").startswith("DP ") or "URANIA" in str(row.get("method", "")))
        items.append(row)
    return items

def _maybe_drop_single_dataset_column(tab: pd.DataFrame) -> pd.DataFrame:
    if "Dataset" in tab.columns and tab["Dataset"].nunique(dropna=True) <= 1:
        return tab.drop(columns=["Dataset"])
    return tab


def _make_dpspin_table(df: pd.DataFrame, path: Path) -> None:
    """Write the DP-SPIN release-conditioned OpenAI evaluation table."""
    ensure_dir(path.parent)
    # Accept either the full judgments dataframe or a pre-filtered DP-SPIN one.
    if not df.empty and "method" in df.columns:
        df = df[df["method"].astype(str).eq("DP-SPIN private plan")].copy()
    dataset_slug, dataset_phrase, label_suffix = _dataset_scope_from_df(df)
    if df.empty:
        content = "% No DP-SPIN OpenAI judgments found.\n"
        _write_named_latex(path, dataset_slug, "tab_openai_dpspin", content)
        return

    numeric = [c for c in GENERIC_SCORE_COLS if c in df.columns]
    rows = []
    group_cols = [c for c in ["dataset", "protected_unit", "mechanism", "epsilon", "n_records_effective", "n_users_effective", "user_contribution_bound", "m", "K", "L"] if c in df.columns]
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        effective_units = sorted({x for x in (_effective_protected_units(rr) for _, rr in g.iterrows()) if x is not None})
        first = g.iloc[0]
        row = {
            "Dataset": key_map.get("dataset"),
            "Unit": key_map.get("protected_unit"),
            "Mechanism": MECHANISM_DISPLAY.get(str(key_map.get("mechanism")), str(key_map.get("mechanism", ""))),
            "Configuration": _row_configuration(first),
            r"$\varepsilon$": key_map.get("epsilon"),
            "Protected units": effective_units[0] if len(effective_units) == 1 else ",".join(map(str, effective_units)),
            r"$N$": len(g),
        }
        for col in numeric:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if col == "privacy_safety":
                label = "Judge privacy safety"
            else:
                label = col.capitalize().replace("_", " ")
            row[label] = _fmt_mean_std(vals.mean(), vals.std(), len(vals)) if len(vals) else "--"
        rows.append(row)

    tab = pd.DataFrame(rows)
    if tab.empty:
        content = "% Empty DP-SPIN OpenAI table.\n"
        _write_named_latex(path, dataset_slug, "tab_openai_dpspin", content)
        return
    tab = _maybe_drop_single_dataset_column(tab)
    cols = list(tab.columns)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Release-conditioned OpenAI evaluation of representative \texttt{{DP-SPIN}} summaries on {_latex_escape(dataset_phrase)}. The decoder receives only the released differentially private semantic plan and public decoding instructions. Scores are assigned on the same 1--5 rubric used for the baseline-comparison table.}}",
        rf"\label{{tab:openai-dpspin-{label_suffix}}}",
        r"\small",
        r"\resizebox{0.99\textwidth}{!}{%",
        r"\begin{tabular}{" + "l" * len(cols) + r"}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for _, rr in tab.iterrows():
        lines.append(" & ".join(_latex_escape_preserving_math(rr[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    _write_named_latex(path, dataset_slug, "tab_openai_dpspin", "\n".join(lines))


def _method_decoder_input(method: str) -> str:
    return {
        "DP-SPIN private plan": "DP semantic plan",
        "DP keyword histogram": "DP keyword histogram",
        "DP category histogram": "DP category histogram",
        "URANIA-style public keywords": "DP public keywords",
        "Non-private-plan reference": "Non-private plan",
        "OpenAI raw records": "Raw target records",
        "OpenAI heuristic redaction": "Heuristically redacted records",
        "OpenAI non-private clustering": "Non-private clusters",
    }.get(method, "Generated summary")



def _format_optional_number(name: str, value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if str(value).strip() in {"", "nan", "None"}:
        return None
    v_float = _safe_float(value)
    if v_float is not None and abs(v_float - round(v_float)) < 1e-9:
        return f"{name}={int(round(v_float))}"
    if v_float is not None:
        return f"{name}={v_float:g}"
    return f"{name}={value}"


def _row_configuration(row: dict[str, Any] | pd.Series) -> str:
    """Format the release configuration shown in comparison tables."""
    method = str(row.get("method") or "")
    family = str(row.get("method_family") or "")
    mech = str(row.get("mechanism") or "")

    if method in {"DP-SPIN private plan", "DP-SPIN"} or family == "DP-SPIN":
        parts = []
        unit = str(row.get("protected_unit") or "")
        if unit and unit not in {"nan", "None"}:
            parts.append(unit)
        display = str(row.get("mechanism_display") or MECHANISM_DISPLAY.get(mech, mech))
        if display and display not in {"nan", "None"}:
            parts.append(display)
        for name, key in [("m", "m"), ("K", "K"), ("L", "L")]:
            z = _format_optional_number(name, row.get(key))
            if z:
                parts.append(z)
        if unit == "user":
            z = _format_optional_number("B", row.get("user_contribution_bound"))
            if z:
                parts.append(z)
        return "; ".join(parts) if parts else "DP semantic plan"

    if method == "DP keyword histogram":
        parts = ["fixed public keyword vocabulary"]
        vocab = _format_optional_number("vocab", row.get("vocabulary_size"))
        if vocab:
            parts.append(vocab)
        parts.append("Laplace histogram")
        L = _safe_int(row.get("L"))
        parts.append(f"released top-{L} keywords" if L is not None else "released top-$L$ keywords")
        return "; ".join(parts)

    if method == "DP category histogram":
        cat = row.get("category_key") or "public category field"
        parts = [f"fixed {cat} universe"]
        C = _format_optional_number("categories", row.get("category_universe_size"))
        if C:
            parts.append(C)
        parts.append("Laplace histogram")
        L = _safe_int(row.get("L"))
        parts.append(f"released top-{L} categories" if L is not None else "released top-$L$ categories")
        return "; ".join(parts)

    if method == "URANIA-style public keywords":
        C = _format_optional_number("clusters", row.get("num_clusters_requested"))
        k = _format_optional_number("keywords", row.get("keywords_per_summary"))
        parts = ["DP clustering", "DP public-keyword histograms"]
        if C:
            parts.append(C)
        if k:
            parts.append(k)
        return "; ".join(parts)

    return str(row.get("mechanism_display") or row.get("mechanism") or "")



def _filter_judgments_for_comparison(
    df: pd.DataFrame,
    *,
    dpspin_mechanism: str | None,
    dpspin_epsilon: float | None,
) -> pd.DataFrame:
    """Filter stored judgment rows to the matched DP-SPIN comparison condition."""
    if df.empty or "method" not in df.columns:
        return df
    out = df.copy()
    is_dpspin = out["method"].astype(str).eq("DP-SPIN private plan")
    keep = ~is_dpspin
    if dpspin_mechanism is not None and "mechanism" in out.columns:
        keep = keep | (is_dpspin & out["mechanism"].astype(str).eq(str(dpspin_mechanism)))
    else:
        keep = keep | is_dpspin
    out = out[keep].copy()

    if dpspin_epsilon is not None and "epsilon" in out.columns:
        eps = pd.to_numeric(out["epsilon"], errors="coerce")
        is_dpspin = out["method"].astype(str).eq("DP-SPIN private plan")
        out = out[(~is_dpspin) | (eps.sub(float(dpspin_epsilon)).abs() <= 1e-9)].copy()
    return out


def _make_baseline_table(df: pd.DataFrame, path: Path, methods: Iterable[str] | None = None) -> None:
    """Write the release-conditioned OpenAI baseline comparison table."""
    ensure_dir(path.parent)
    df = _filter_df_by_method(df, methods)
    dataset_slug, dataset_phrase, label_suffix = _dataset_scope_from_df(df)
    if df.empty:
        content = "% No OpenAI judgments found for the selected methods.\n"
        _write_named_latex(path, dataset_slug, "tab_openai_baseline_comparison", content)
        return

    numeric = [c for c in GENERIC_SCORE_COLS if c in df.columns]
    rows = []
    # Include configuration in the grouping for DP-SPIN, because mechanism and
    # m/K/L are part of the reported condition.  Baselines are grouped by their
    # method and privacy budget.
    group_cols = [c for c in [
        "dataset", "method", "protected_unit", "mechanism", "epsilon",
        "n_records_effective", "n_users_effective", "user_contribution_bound", "m", "K", "L",
        "category_key", "num_clusters_requested", "keywords_per_summary",
    ] if c in df.columns]
    include_epsilon = "epsilon" in group_cols
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        dataset = key_map.get("dataset")
        method = str(key_map.get("method"))
        effective_units = sorted({x for x in (_effective_protected_units(rr) for _, rr in g.iterrows()) if x is not None})
        # Use the first row only for descriptive fields; scores are aggregated over g.
        first = g.iloc[0]
        row = {
            "Dataset": dataset,
            "Method": method,
            "Release object": _method_decoder_input(method),
            "Configuration": _row_configuration(first),
        }
        if include_epsilon:
            row[r"$\varepsilon$"] = key_map.get("epsilon")
        if effective_units:
            row["Protected units"] = effective_units[0] if len(effective_units) == 1 else ",".join(map(str, effective_units))
        row[r"$N$"] = len(g)
        for col in numeric:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if col == "privacy_safety":
                label = "Judge privacy safety"
            else:
                label = col.capitalize().replace("_", " ")
            row[label] = _fmt_mean_std(vals.mean(), vals.std(), len(vals)) if len(vals) else "--"
        rows.append(row)

    tab = pd.DataFrame(rows)
    if tab.empty:
        content = "% Empty OpenAI comparison table.\n"
        _write_named_latex(path, dataset_slug, "tab_openai_baseline_comparison", content)
        return

    method_order = {m: i for i, m in enumerate(DEFAULT_PAPER_METHODS + [
        "Non-private-plan reference",
        "OpenAI heuristic redaction",
        "OpenAI non-private clustering",
        "OpenAI raw records",
    ])}
    tab["sort_key"] = tab["Method"].astype(str).map(lambda x: method_order.get(x, 999))
    tab = tab.sort_values(["Dataset", "sort_key", "Method"]).drop(columns=["sort_key"])
    tab = _maybe_drop_single_dataset_column(tab)
    cols = list(tab.columns)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Release-conditioned OpenAI evaluation of representative generated summaries on {_latex_escape(dataset_phrase)}. Each summary is evaluated only with respect to the object released to its decoder. Scores are assigned on a 1--5 scale and report verbalization quality. The judge privacy-safety column evaluates only the generated text and is not a formal differential-privacy guarantee.}}",
        rf"\label{{tab:openai-baseline-comparison-{label_suffix}}}",
        r"\small",
        r"\resizebox{0.99\textwidth}{!}{%",
        r"\begin{tabular}{" + "l" * len(cols) + r"}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for _, rr in tab.iterrows():
        lines.append(" & ".join(_latex_escape_preserving_math(rr[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    _write_named_latex(path, dataset_slug, "tab_openai_baseline_comparison", "\n".join(lines))

def _make_private_non_private_eval_table(df: pd.DataFrame, path: Path) -> None:
    """Write the table comparing DP-SPIN display summaries with offline non-private references."""
    ensure_dir(path.parent)
    dataset_caption = _dataset_caption_name(df)
    label_suffix = _latex_label_slug(dataset_caption)
    if df.empty:
        path.write_text("% No DP-SPIN OpenAI rows found.\n", encoding="utf-8")
        return
    metric_specs = [
        ("text_token_jaccard", "$J_{\\mathrm{tok}}$"),
        ("text_bigram_jaccard", "$J_2$"),
        ("text_keyphrase_jaccard", "$J_{\\mathrm{kp}}$"),
        ("text_tfidf_cosine", "$\\mathsf{Cos}_{\\mathrm{tfidf}}$"),
        ("embedding_summary_embedding_cosine", "$\\mathsf{Cos}_{\\mathrm{emb}}$"),
    ]
    rows: list[dict[str, Any]] = []
    # Group by the full DP-SPIN release condition.  Grouping only by mechanism
    # and epsilon can merge runs with different m, K, L, protected-unit counts,
    # or user clipping bounds, which makes the configuration column misleading.
    group_cols = [c for c in [
        "dataset", "protected_unit", "mechanism", "epsilon",
        "n_records_effective", "n_users_effective", "user_contribution_bound",
        "m", "K", "L",
    ] if c in df.columns]
    if not group_cols:
        group_cols = ["mechanism", "epsilon"]
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        effective_units = sorted({x for x in (_effective_protected_units(rr) for _, rr in g.iterrows()) if x is not None})
        first = g.iloc[0]
        row = {
            "Dataset": key_map.get("dataset", ""),
            "Unit": key_map.get("protected_unit", ""),
            "Mechanism": MECHANISM_DISPLAY.get(str(key_map.get("mechanism")), str(key_map.get("mechanism", ""))),
            "Configuration": _row_configuration({**first.to_dict(), "method": "DP-SPIN private plan"}),
            r"$\varepsilon$": key_map.get("epsilon", ""),
            "Protected units": effective_units[0] if len(effective_units) == 1 else ",".join(map(str, effective_units)),
            "Runs": len(g),
        }
        for col, label in metric_specs:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce")
                row[label] = _fmt_mean_std(vals.mean(), vals.std(), len(vals.dropna()))
        rows.append(row)
    tab = pd.DataFrame(rows)
    if tab.empty:
        path.write_text("% Empty private-non-private evaluation table.\n", encoding="utf-8")
        return
    # Remove dataset column for single-dataset reports; the caption already names it.
    if tab["Dataset"].nunique(dropna=True) <= 1:
        tab = tab.drop(columns=["Dataset"])
    cols = list(tab.columns)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Similarity between OpenAI summaries generated from \texttt{{DP-SPIN}} private plans and summaries generated from the corresponding non-private top-$L$ plans on {_latex_escape(dataset_caption)}. Metrics compare generated texts and do not evaluate the underlying semantic sketches.}}",
        rf"\label{{tab:openai-private-non-private-{label_suffix}}}",
        r"\small",
        r"\resizebox{0.99\textwidth}{!}{%",
        r"\begin{tabular}{" + "l" * len(cols) + r"}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for _, rr in tab.iterrows():
        lines.append(" & ".join(_latex_escape_preserving_math(rr[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")

def _standardize_display_text_for_latex(text: str) -> str:
    """Normalize display formatting without changing the generated claims.

    The function does not summarize,
    paraphrase, remove noisy values, remove atom identifiers, or alter claims.
    It only converts common generator markup into LaTeX-safe display forms.
    """
    s = text or ""
    # Normalize line endings and excessive whitespace while preserving paragraph breaks.
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Convert XML-like tags emitted by the URANIA-style baseline into readable labels.
    s = re.sub(r"<\s*topic\s*>", "Topic: ", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*/\s*topic\s*>", ". ", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*description\s*>", "Description: ", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*/\s*description\s*>", " ", s, flags=re.IGNORECASE)
    # Markdown bold becomes a LaTeX command after escaping is handled manually below.
    # Protect bold spans with temporary markers before escaping.
    bold_spans: list[str] = []
    def repl_bold(m: re.Match[str]) -> str:
        bold_spans.append(m.group(1))
        return f"@@BOLD{len(bold_spans)-1}@@"
    s = re.sub(r"\*\*([^*]+)\*\*", repl_bold, s)
    # Collapse long whitespace but keep enough separation for readability in a table cell.
    s = re.sub(r"\s+", " ", s).strip()
    escaped = _latex_escape(s)
    for i, span in enumerate(bold_spans):
        escaped = escaped.replace(f"@@BOLD{i}@@", r"\textbf{" + _latex_escape(span) + "}")
    return escaped


def _method_label_for_item(item: dict[str, Any]) -> str:
    """Format a method label that includes the decoder-visible release configuration."""
    method = str(item.get("method") or "")
    family = str(item.get("method_family") or "")

    def add_eps_and_n(parts: list[str]) -> list[str]:
        eps = _safe_float(item.get("epsilon"))
        n = _effective_protected_units(item)
        if eps is not None:
            parts.append(rf"$\varepsilon={eps:g}$")
        if n is not None:
            unit = str(item.get("protected_unit") or "record")
            parts.append(rf"$u={n}$" if unit == "user" else rf"$n={n}$")
        return parts

    if family == "non_private_reference" or method == "Non-private-plan reference":
        parts = [_row_configuration({**item, "method": "DP-SPIN private plan"})]
        parts = add_eps_and_n(parts)
        return "Non-private-plan reference" + (" (" + "; ".join([p for p in parts if p]) + ")" if parts else "")

    if family == "DP-SPIN" or method == "DP-SPIN private plan":
        parts = [_row_configuration({**item, "method": "DP-SPIN private plan"})]
        parts = add_eps_and_n(parts)
        return "DP-SPIN private plan" + (" (" + "; ".join([p for p in parts if p]) + ")" if parts else "")

    if method in BASELINE_SUMMARY_FILES:
        parts = [_row_configuration(item)]
        parts = add_eps_and_n(parts)
        return method + (" (" + "; ".join([p for p in parts if p]) + ")" if parts else "")

    return method


def _make_qualitative_tables(
    items: list[dict[str, Any]],
    root: Path,
    qualitative_max_chars: int = 0,
    qualitative_methods: Iterable[str] | None = None,
) -> None:
    """Write one canonical qualitative table and one Markdown file."""
    ensure_dir(root)
    items = _filter_items_by_method(items, qualitative_methods)
    dataset_caption = _dataset_caption_name(items)
    dataset_values: list[str] = []
    for item in items:
        ds = str(item.get("dataset") or "")
        if ds and ds not in dataset_values:
            dataset_values.append(ds)
    single_dataset = len(dataset_values) == 1
    label_suffix = _latex_label_slug(dataset_caption)

    preferred_eps = [1.0, 2.0, 4.0, 0.5]
    def sort_key(item: dict[str, Any]) -> tuple:
        method = str(item.get("method") or "")
        family = str(item.get("method_family") or "")
        eps = _safe_float(item.get("epsilon"))
        eps_rank = preferred_eps.index(eps) if eps in preferred_eps else len(preferred_eps)
        unit_rank = 0 if str(item.get("protected_unit") or "") == "record" else 1
        mech_rank = 0 if str(item.get("mechanism") or "") == "gaussian" else 1
        method_rank = {
            "DP-SPIN": 0,
            "DP-SPIN private plan": 0,
            "non_private_reference": 1,
            "Non-private-plan reference": 1,
            "OpenAI raw records": 2,
            "OpenAI heuristic redaction": 3,
            "OpenAI non-private clustering": 4,
            "DP keyword histogram": 5,
            "DP category histogram": 6,
            "URANIA-style public keywords": 7,
        }.get(family, {"OpenAI raw records": 2, "OpenAI heuristic redaction": 3, "OpenAI non-private clustering": 4, "DP keyword histogram": 5, "DP category histogram": 6, "URANIA-style public keywords": 7}.get(method, 9))
        return (str(item.get("dataset") or ""), method_rank, eps_rank, unit_rank, mech_rank)

    # Keep one representative example per dataset/method family. For DP-SPIN and
    # non-private reference, this pairs the private and non-private summaries from the same
    # preferred representative run.
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in sorted(items, key=sort_key):
        dataset = str(item.get("dataset") or "")
        family = str(item.get("method_family") or item.get("method") or "")
        method = str(item.get("method") or "")
        if family == "DP-SPIN" or method == "DP-SPIN private plan":
            key_family = "DP-SPIN"
        elif family == "non_private_reference" or method == "Non-private-plan reference":
            key_family = "non_private_reference"
        else:
            # Baseline rows all have method_family="baseline"; use the method
            # name so DP keyword, DP category, and URANIA-style baselines each
            # appear in qualitative examples.
            key_family = method or family
        key = (dataset, key_family)
        p = Path(str(item.get("summary_path", "")))
        if key in seen or not p.exists():
            continue
        seen.add(key)
        selected.append(item)

    def build_tex(raw: bool) -> str:
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            rf"\caption{{Representative OpenAI-generated aggregate summaries on {_latex_escape(dataset_caption)}. Each row reports the generated summary for one selected \texttt{{DP-SPIN}} or baseline release object.}}",
            rf"\label{{tab:openai-qualitative-{label_suffix}}}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{1.04}",
            r"\resizebox{0.99\textwidth}{!}{%",
        ]
        if single_dataset:
            lines += [
                r"\begin{tabular}{p{0.27\textwidth}p{0.68\textwidth}}",
                r"\toprule",
                r"Method and release configuration & Generated summary \\",
                r"\midrule",
            ]
        else:
            lines += [
                r"\begin{tabular}{p{0.13\textwidth}p{0.20\textwidth}p{0.62\textwidth}}",
                r"\toprule",
                r"Dataset & Method & Generated summary \\",
                r"\midrule",
            ]
        for item in selected:
            display = str(item.get("display_name") or item.get("dataset") or "")
            method_label = _method_label_for_item(item)
            text = _read_text(Path(str(item.get("summary_path", ""))))
            if not raw:
                text = _display_text_for_item(item, text)
            if qualitative_max_chars and qualitative_max_chars > 0:
                text = _short(text, qualitative_max_chars)
            summary = _latex_escape(text) if raw else _standardize_display_text_for_latex(text)
            if single_dataset:
                lines.append(_latex_escape_preserving_math(method_label) + " & " + summary + r" \\")
            else:
                lines.append(_latex_escape(display) + " & " + _latex_escape_preserving_math(method_label) + " & " + summary + r" \\")
            lines.append(r"\addlinespace[1pt]")
        lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
        return "\n".join(lines)

    def build_md(raw: bool) -> str:
        kind = "raw" if raw else "display"
        lines = [
            f"# Representative OpenAI-generated aggregate summaries ({kind}): {dataset_caption}",
            "",
            "DP-SPIN summaries are generated from released DP plans only. Baseline summaries are labeled with their release object and configuration.",
            "",
        ]
        for item in selected:
            display = str(item.get("display_name") or item.get("dataset") or "")
            method_label = _method_label_for_item(item)
            text = _read_text(Path(str(item.get("summary_path", ""))))
            if not raw:
                text = _display_text_for_item(item, text)
                text = re.sub(r"<\s*topic\s*>", "Topic: ", text, flags=re.IGNORECASE)
                text = re.sub(r"<\s*/\s*topic\s*>", ". ", text, flags=re.IGNORECASE)
                text = re.sub(r"<\s*description\s*>", "Description: ", text, flags=re.IGNORECASE)
                text = re.sub(r"<\s*/\s*description\s*>", " ", text, flags=re.IGNORECASE)
            if qualitative_max_chars and qualitative_max_chars > 0:
                text = _short(text, qualitative_max_chars)
            lines += [f"## {display}: {method_label}", "", text.strip(), ""]
        return "\n".join(lines)

    for stale in [
        "tab_openai_qualitative_examples_raw.tex",
        "tab_openai_qualitative_examples_display.tex",
        "qualitative_examples_raw.md",
        "qualitative_examples_display.md",
    ]:
        p = root / stale
        if p.exists():
            p.unlink()
    for pattern in [
        "tab_openai_qualitative_examples_*_cfpb.tex",
        "tab_openai_qualitative_examples_*_amazon_All_Beauty.tex",
        "tab_openai_qualitative_examples_*_yelp_restaurants.tex",
        "qualitative_examples*_cfpb.md",
        "qualitative_examples*_amazon_All_Beauty.md",
        "qualitative_examples*_yelp_restaurants.md",
    ]:
        for stale in root.glob(pattern):
            stale.unlink()

    (root / "tab_openai_qualitative_examples.tex").write_text(build_tex(raw=False), encoding="utf-8")
    (root / "qualitative_examples.md").write_text(build_md(raw=False), encoding="utf-8")

def _write_readme(
    root: Path,
    specs: list[DatasetSpec],
    args: argparse.Namespace,
    *,
    dpspin_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    judged_items: list[dict[str, Any]],
) -> None:
    """Write a dataset-specific README for the OpenAI paper report."""
    dpspin_df = pd.DataFrame(dpspin_rows)
    baseline_df = pd.DataFrame(baseline_rows)
    judged_df = pd.DataFrame(judged_items)

    dataset_names = ", ".join(spec.display_name for spec in specs)
    lines = [
        f"# OpenAI paper report: {dataset_names}",
        "",
        "This directory contains representative OpenAI verbalization and OpenAI-judge results.",
        "The pipeline does not rerun the DP mechanisms; it uses completed DP-SPIN runs and",
        "separately generated DP baseline release objects.",
        "",
        "## Included methods",
        "",
        "- `DP-SPIN private plan`: the decoder receives only the released DP semantic plan.",
        "- `DP keyword histogram`: the decoder receives only a released DP keyword histogram.",
        "- `DP category histogram`: the decoder receives only a released DP category histogram.",
        "- `URANIA-style public keywords`: the decoder receives released public keywords and DP cluster-level statistics from the URANIA-style baseline.",
        "",
        "## Files",
        "",
        "- `tables/dpspin_openai_runs.csv`: all decoded DP-SPIN representative runs.",
        "- `tables/baseline_runs.csv`: generated baseline summary files and privacy metadata.",
        "- `tables/all_summary_items.csv`: all candidate summaries collected for reporting.",
        "- `tables/judged_summary_items.csv`: all selected summaries sent to the release-conditioned OpenAI judge.",
        "- `judgments/judgments.csv`: release-conditioned OpenAI rubric judgments for the selected summaries.",
        "- `latex/tab_openai_dpspin.tex`: DP-SPIN representative verbalization table.",
        "- `latex/tab_openai_private_non_private_eval.tex`: DP-SPIN private-plan versus non-private-plan text similarity table.",
        "- `latex/tab_openai_baseline_comparison.tex`: matched comparison of DP-SPIN and DP baselines.",
        "- `latex/tab_openai_qualitative_examples.tex`: paper-facing generated insight examples.",
        "- `latex/qualitative_examples.md`: Markdown copy of the generated examples.",
        "",
        "## Run configuration",
        "",
        f"- OpenAI model: `{args.model}`",
        f"- DP-SPIN mechanisms decoded: `{args.mechanisms}`",
        f"- DP-SPIN epsilons decoded: `{args.epsilons}`",
        f"- DP-SPIN seeds decoded: `{args.seeds}`",
        f"- DP-SPIN units: `{args.units}`",
        f"- DP-SPIN n-record filters: `{args.n_records_values}`",
        f"- DP-SPIN n-user filters: `{args.n_users_values}`",
        f"- baseline methods generated: `{args.baseline_methods}`",
        f"- baseline epsilon: `{args.baseline_epsilon}`",
        f"- baseline seeds: `{args.baseline_seeds if args.baseline_seeds is not None else [args.baseline_seed]}`",
        f"- URANIA-style baseline enabled: `{args.run_urania}`",
        f"- comparison methods: `{args.comparison_methods}`",
        f"- qualitative methods: `{args.qualitative_methods}`",
        f"- matched DP-SPIN comparison condition: mechanism=`{args.comparison_dpspin_mechanism}`, epsilon=`{args.comparison_dpspin_epsilon}`",
        "",
        "## Row counts",
        "",
        f"- DP-SPIN decoded rows: `{len(dpspin_df)}`",
        f"- baseline rows: `{len(baseline_df)}`",
        f"- judged rows in comparison table: `{len(judged_df)}`",
        "",
    ]
    if not dpspin_df.empty:
        lines += ["### DP-SPIN decoded rows", ""]
        cols = [c for c in ["dataset", "protected_unit", "mechanism", "epsilon", "seed", "n_records_effective", "n_users_effective"] if c in dpspin_df.columns]
        counts = dpspin_df[cols].fillna("").astype(str)
        for _, rr in counts.iterrows():
            lines.append("- " + ", ".join(f"{c}={rr[c]}" for c in cols if rr[c] not in {"", "nan", "None"}))
        lines.append("")
    if not baseline_df.empty:
        lines += ["### Baseline rows", ""]
        cols = [c for c in ["dataset", "method", "mechanism", "epsilon", "seed", "n_records_effective"] if c in baseline_df.columns]
        for _, rr in baseline_df[cols].fillna("").astype(str).iterrows():
            lines.append("- " + ", ".join(f"{c}={rr[c]}" for c in cols if rr[c] not in {"", "nan", "None"}))
        lines.append("")
    lines += [
        "## Interpretation note",
        "",
        "The paper-facing tables use one release-conditioned judge for all methods.",
        "The judge sees each generated summary and the object released to that decoder; it does not see raw protected records.",
        "The scores are secondary presentation measures. They do not replace the formal differential-privacy accounting or the plan-level utility metrics.",
        "Similar judge scores across methods usually mean that the generated texts are readable and faithful to their own released objects.",
        "They do not imply that the underlying release objects have the same semantic coverage, privacy boundary, or mechanism utility.",
        "The baseline-comparison table reports one matched representative DP-SPIN condition; the DP-SPIN-only table reports all decoded representative DP-SPIN conditions.",
    ]
    (root / "README_OPENAI_PAPER_REPORT.md").write_text("\n".join(lines), encoding="utf-8")



def run_pipeline(args: argparse.Namespace) -> None:
    """Run the selected cross-dataset verbalization and release-conditioned evaluations."""
    if args.openai_required and not os.environ.get("OPENAI_API_KEY") and not args.report_only:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running OpenAI decoding/baselines/judge.")

    selected_keys = list(DATASET_SPECS) if args.dataset == "all" else [args.dataset]
    specs = [DATASET_SPECS[k] for k in selected_keys]
    scope_name = _dataset_scope_name(specs, args.dataset)
    base_output_dir = Path(args.output_dir)
    root = ensure_dir(base_output_dir / scope_name)
    ensure_dir(root / "tables")
    ensure_dir(root / "latex")
    ensure_dir(root / "judgments")
    ensure_dir(root / "logs")

    # Remove stale LaTeX files from older pipeline versions.  The current
    # pipeline writes one canonical file per table because the directory is
    # already dataset-scoped.
    for stale in list((root / "latex").glob("*_cfpb.tex")) + list((root / "latex").glob("*_amazon_All_Beauty.tex")) + list((root / "latex").glob("*_yelp_restaurants.tex")):
        try:
            stale.unlink()
        except OSError:
            pass
    for stale_name in [
        "tab_openai_qualitative_examples_raw.tex",
        "tab_openai_qualitative_examples_display.tex",
        "qualitative_examples_raw.md",
        "qualitative_examples_display.md",
    ]:
        stale = root / "latex" / stale_name
        if stale.exists():
            stale.unlink()
    for pattern in [
        "tab_openai_*_cfpb.tex",
        "tab_openai_*_amazon_All_Beauty.tex",
        "tab_openai_*_yelp_restaurants.tex",
        "qualitative_examples*_cfpb.md",
        "qualitative_examples*_amazon_All_Beauty.md",
        "qualitative_examples*_yelp_restaurants.md",
    ]:
        for stale in (root / "latex").glob(pattern):
            try:
                stale.unlink()
            except OSError:
                pass

    all_dpspin_rows: list[dict[str, Any]] = []
    all_baseline_rows: list[dict[str, Any]] = []
    baseline_meta: list[dict[str, Any]] = []
    judgments = pd.DataFrame()

    comparison_methods = list(args.comparison_methods or DEFAULT_PAPER_METHODS)
    qualitative_methods = list(args.qualitative_methods or comparison_methods)
    judge_methods = list(args.judge_methods or sorted(set(comparison_methods).union(qualitative_methods)))
    baseline_row_methods = [m for m in comparison_methods + qualitative_methods + judge_methods if m in BASELINE_SUMMARY_FILES]
    baseline_methods = list(args.baseline_methods)

    if args.report_only:
        dpspin_csv = root / "tables" / "dpspin_openai_runs.csv"
        baseline_csv = root / "tables" / "baseline_runs.csv"
        judgments_csv = root / "judgments" / "judgments.csv"
        if dpspin_csv.exists() and dpspin_csv.stat().st_size > 0:
            all_dpspin_rows = pd.read_csv(dpspin_csv).to_dict(orient="records")
        if baseline_csv.exists() and baseline_csv.stat().st_size > 0:
            all_baseline_rows = pd.read_csv(baseline_csv).to_dict(orient="records")
        if judgments_csv.exists() and judgments_csv.stat().st_size > 0:
            judgments = pd.read_csv(judgments_csv)
        print(f"[Report-only] Loaded {len(all_dpspin_rows)} DP-SPIN rows and {len(all_baseline_rows)} baseline rows from {root}")
    else:
        if args.run_dpspin:
            for spec in specs:
                rows = _decode_dpspin_for_dataset(
                    spec=spec,
                    out_dir=root / "datasets" / spec.dataset_name,
                    mechanisms=args.mechanisms,
                    epsilons=args.epsilons,
                    seeds=args.seeds,
                    units=args.units,
                    m_values=args.m_values,
                    user_bounds=args.user_contribution_bounds,
                    n_records_values=args.n_records_values,
                    n_users_values=args.n_users_values,
                    K=args.K,
                    L=args.L,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    force=args.force,
                    judge_backend=args.dpspin_judge_backend,
                    comparative_backend=args.comparative_backend,
                    summary_embedding_backend=args.summary_embedding_backend,
                    max_runs=args.max_dpspin_runs_per_dataset,
                )
                all_dpspin_rows.extend(rows)

        if args.run_baselines:
            baseline_seeds = list(args.baseline_seeds) if args.baseline_seeds is not None else [int(args.baseline_seed)]
            for spec in specs:
                subset_size = args.baseline_subset_size or spec.baseline_subset_size
                for baseline_seed in baseline_seeds:
                    meta = _run_baselines_for_dataset(
                        spec=spec,
                        out_dir=root / "datasets" / spec.dataset_name,
                        model=args.model,
                        epsilon=args.baseline_epsilon,
                        seed=int(baseline_seed),
                        subset_size=subset_size,
                        force=args.force,
                        urania_epsilon_total=args.urania_epsilon_total,
                        urania_delta=args.urania_delta,
                        urania_clusters=args.urania_clusters,
                        urania_keywords=args.urania_keywords,
                        baseline_methods=baseline_methods,
                        run_urania=args.run_urania,
                    )
                    baseline_meta.append(meta)
                    all_baseline_rows.extend(_baseline_rows(spec, Path(meta["baseline_dir"]), allowed_methods=baseline_row_methods or None))

        all_dpspin_rows = _dedupe_dpspin_rows(
            all_dpspin_rows,
            specs,
            keep_record_counts=args.n_records_values is not None,
            keep_user_counts=args.n_users_values is not None,
        )
        _write_csv(all_dpspin_rows, root / "tables" / "dpspin_openai_runs.csv")
        _write_csv(all_baseline_rows, root / "tables" / "baseline_runs.csv")
        pd.DataFrame(baseline_meta).to_csv(root / "tables" / "baseline_subsets.csv", index=False)

    all_dpspin_rows = _dedupe_dpspin_rows(
        all_dpspin_rows,
        specs,
        keep_record_counts=args.n_records_values is not None,
        keep_user_counts=args.n_users_values is not None,
    )
    _write_csv(all_dpspin_rows, root / "tables" / "dpspin_openai_runs.csv")
    items = _collect_items(all_dpspin_rows, all_baseline_rows)
    _write_csv(items, root / "tables" / "all_summary_items.csv")
    judge_items = _filter_items_by_method(items, judge_methods)
    _write_csv(judge_items, root / "tables" / "judged_summary_items.csv")
    comparison_judge_items = _filter_comparison_judge_items(
        judge_items,
        dpspin_mechanism=args.comparison_dpspin_mechanism,
        dpspin_epsilon=args.comparison_dpspin_epsilon,
    )
    _write_csv(comparison_judge_items, root / "tables" / "comparison_judged_summary_items.csv")

    if args.run_judge and not args.report_only:
        judgments = _judge_all(judge_items, root / "judgments", backend=args.generic_judge_backend, model=args.model, force=args.force)
    elif judgments.empty and (root / "judgments" / "judgments.csv").exists():
        judgments = pd.read_csv(root / "judgments" / "judgments.csv")

    # In report-only mode, older judgments.csv files may contain stale rows from
    # previous broader selections.  Keep only rows corresponding to the current
    # judged_summary_items.csv selection when summary paths are available.
    if not judgments.empty and "summary_path" in judgments.columns:
        allowed_paths = {str(item.get("summary_path")) for item in judge_items if item.get("summary_path")}
        if allowed_paths:
            judgments = judgments[judgments["summary_path"].astype(str).isin(allowed_paths)].copy()

    dpspin_df = pd.DataFrame(all_dpspin_rows)
    dpspin_df.to_csv(root / "latex" / "dpspin_openai_runs_collected.csv", index=False)
    judgments.to_csv(root / "latex" / "generic_openai_judgments_collected.csv", index=False)
    _make_dpspin_table(judgments, root / "latex" / "tab_openai_dpspin.tex")
    _make_private_non_private_eval_table(dpspin_df, root / "latex" / "tab_openai_private_non_private_eval.tex")
    comparison_judgments = _filter_judgments_for_comparison(
        judgments,
        dpspin_mechanism=args.comparison_dpspin_mechanism,
        dpspin_epsilon=args.comparison_dpspin_epsilon,
    )
    _make_baseline_table(comparison_judgments, root / "latex" / "tab_openai_baseline_comparison.tex", methods=comparison_methods)
    _make_qualitative_tables(items, root / "latex", qualitative_max_chars=args.qualitative_max_chars, qualitative_methods=qualitative_methods)
    results_text = r"""
\paragraph{Representative OpenAI verbalization.}
For DP-SPIN, the OpenAI decoder receives only the released differentially private semantic plan and public instructions. It does not receive raw protected records. The audit decoder produces marker-preserving summaries for verifier measures. The display decoder produces presentation-oriented summaries from the same released plan and is used for qualitative examples and text-level comparison with the non-private-plan reference. Raw, redacted, and non-private clustering baselines are utility baselines and do not satisfy the same decoder privacy boundary. The OpenAI rubric judgments are secondary qualitative measures; the primary mechanism-utility results remain the plan-level recovery and mass-retention metrics.
""".strip()
    (root / "latex" / "openai_results_text.tex").write_text(results_text + "\n", encoding="utf-8")

    index = {
        "output_dir": str(root),
        "datasets": [spec.__dict__ for spec in specs],
        "dpspin_rows": all_dpspin_rows,
        "baseline_rows": all_baseline_rows,
        "baseline_meta": baseline_meta,
        "judgment_rows": judgments.to_dict(orient="records") if not judgments.empty else [],
        "report_only": bool(args.report_only),
        "comparison_methods": comparison_methods,
        "qualitative_methods": qualitative_methods,
        "judge_methods": judge_methods,
        "baseline_methods": baseline_methods,
        "run_urania": bool(args.run_urania),
        "comparison_dpspin_mechanism": args.comparison_dpspin_mechanism,
        "comparison_dpspin_epsilon": args.comparison_dpspin_epsilon,
        "summary_modes": {
            "audit": "verifier-oriented, marker-preserving, exact noisy masses allowed",
            "display": "presentation-oriented, plan-only, no atom IDs or exact noisy masses requested",
        },
    }
    write_json(index, root / "openai_paper_report_index.json")
    _write_readme(root, specs, args, dpspin_rows=all_dpspin_rows, baseline_rows=all_baseline_rows, judged_items=comparison_judge_items)
    print(f"Wrote OpenAI paper-report pipeline outputs to {root}")

def _float_list(xs: Iterable[str]) -> list[float]:
    return [float(x) for x in xs]


def main() -> None:
    """Run the OpenAI evaluation pipeline from the command line."""
    p = argparse.ArgumentParser(description="Run cross-dataset OpenAI representative decoding, baselines, judging, and reporting.")
    p.add_argument("--dataset", choices=["all", *DATASET_SPECS.keys()], default="all")
    p.add_argument("--output-dir", default="outputs/openai_paper_report")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--mechanisms", nargs="+", default=["gaussian", "private_topL"])
    p.add_argument("--epsilons", nargs="+", type=float, default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--units", nargs="+", choices=["record", "user"], default=None)
    p.add_argument("--m-values", nargs="+", type=int, default=None)
    p.add_argument("--user-contribution-bounds", nargs="+", type=float, default=None)
    p.add_argument("--n-records-values", nargs="+", type=int, default=None)
    p.add_argument("--n-users-values", nargs="+", type=int, default=None)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--L", type=int, default=10)
    p.add_argument("--max-dpspin-runs-per-dataset", type=int, default=None)
    p.add_argument("--baseline-subset-size", type=int, default=None)
    p.add_argument("--baseline-methods", nargs="+", choices=["raw", "redacted", "non_private_clustering", "dp_keyword", "dp_category"], default=["dp_keyword", "dp_category"], help="Baseline methods generated by src.baselines. Default keeps paper runs focused on DP histogram baselines.")
    p.add_argument("--comparison-methods", nargs="+", default=DEFAULT_PAPER_METHODS, help="Methods included in the OpenAI comparison table. Use 'all' to include all judged methods.")
    p.add_argument("--comparison-dpspin-mechanism", default="gaussian", help="DP-SPIN mechanism used in the cross-method OpenAI comparison table.")
    p.add_argument("--comparison-dpspin-epsilon", type=float, default=1.0, help="DP-SPIN epsilon used in the cross-method OpenAI comparison table.")
    p.add_argument("--qualitative-methods", nargs="+", default=DEFAULT_PAPER_METHODS, help="Methods included in qualitative generated-summary tables. Use 'all' to include all generated methods.")
    p.add_argument("--judge-methods", nargs="+", default=None, help="Methods sent to the generic judge. Default is the union of comparison and qualitative methods.")
    p.add_argument("--baseline-epsilon", type=float, default=1.0)
    p.add_argument("--baseline-seed", type=int, default=0)
    p.add_argument("--baseline-seeds", nargs="+", type=int, default=None, help="Baseline seeds. Defaults to --baseline-seed when omitted.")
    p.add_argument("--urania-epsilon-total", type=float, default=1.0)
    p.add_argument("--urania-delta", type=float, default=1e-6)
    p.add_argument("--urania-clusters", type=int, default=10)
    p.add_argument("--urania-keywords", type=int, default=10)
    p.add_argument("--run-urania", action=argparse.BooleanOptionalAction, default=True, help="Generate and include the URANIA-style public-keyword baseline.")
    p.add_argument("--dpspin-judge-backend", choices=["none", "heuristic", "openai"], default="none", help="Optional legacy DP-SPIN-only plan-conditioned judge. The paper tables use --generic-judge-backend.")
    p.add_argument("--comparative-backend", choices=["none", "heuristic", "openai"], default="none")
    p.add_argument("--summary-embedding-backend", choices=["tfidf", "auto", "sentence_transformer"], default="tfidf")
    p.add_argument("--generic-judge-backend", choices=["none", "heuristic", "openai"], default="openai")
    p.add_argument("--run-dpspin", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-baselines", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run-judge", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--openai-required", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--report-only", action="store_true", help="Regenerate CSV-derived report files without OpenAI calls.")
    p.add_argument("--qualitative-max-chars", type=int, default=0, help="Optional qualitative table character limit; 0 disables truncation.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
