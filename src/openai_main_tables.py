"""Build cross-dataset OpenAI tables from saved report directories.

No OpenAI API call is made. The code reads dataset-scoped outputs created by :mod:`src.openai_paper_pipeline` and writes the selected CSV and LaTeX tables
used in the manuscript. Dataset-specific reports retain the complete set of conditions.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DATASET_DIRS = {
    "cfpb": "cfpb",
    "amazon": "amazon_All_Beauty",
    "yelp": "yelp_restaurants",
}
DATASET_DISPLAY = {
    "cfpb": "CFPB",
    "amazon_All_Beauty": "Amazon",
    "yelp_restaurants": "Yelp",
}
DATASET_LONG_DISPLAY = {
    "cfpb": "CFPB consumer complaints",
    "amazon_All_Beauty": "Amazon All Beauty reviews",
    "yelp_restaurants": "Yelp restaurant reviews",
}
METHOD_ORDER = {
    "DP-SPIN private plan": 0,
    "DP keyword histogram": 1,
    "DP category histogram": 2,
    "URANIA-style public keywords": 3,
}
METHOD_DISPLAY = {
    "DP-SPIN private plan": "DP-SPIN",
    "DP keyword histogram": "DP keyword histogram",
    "DP category histogram": "DP category histogram",
    "URANIA-style public keywords": "URANIA-style public keywords",
}
MECHANISM_DISPLAY = {
    "gaussian": r"Gaussian sketch + top-$L$",
    "laplace": r"Laplace sketch + top-$L$",
    "private_topL": r"Private top-$L$ admission + Gaussian values",
    "dp_keyword_histogram": "DP keyword histogram",
    "dp_category_histogram": "DP category histogram",
    "urania_public_keywords": "URANIA-style public keywords",
}
SCORE_COLUMNS = ["coverage", "specificity", "insightfulness", "faithfulness", "privacy_safety", "clarity"]
MAIN_SCORE_COLUMNS = ["coverage", "insightfulness", "faithfulness", "privacy_safety"]


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
    s = "" if value is None else str(value)
    if "$" not in s:
        return _latex_escape(s)
    parts = re.split(r"(\$[^$]*\$)", s)
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("$") and part.endswith("$"):
            out.append(part)
        else:
            out.append(_latex_escape(part))
    return "".join(out)


def _safe_int(x: Any) -> int | None:
    try:
        if pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None


def _safe_float(x: Any) -> float | None:
    try:
        if pd.isna(x):
            return None
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def _fmt_number(x: Any) -> str:
    if x is None:
        return "--"
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(xf):
        return "--"
    if abs(xf - round(xf)) < 1e-9:
        return str(int(round(xf)))
    return f"{xf:g}"


def _fmt_eps(x: Any) -> str:
    return _fmt_number(x)


def _fmt_mean_std(values: pd.Series) -> str:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return "--"
    mean = vals.mean()
    if len(vals) <= 1:
        return f"{mean:.2f}"
    std = vals.std()
    if pd.isna(std):
        return f"{mean:.2f}"
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def _protected_units(row: pd.Series | dict[str, Any]) -> int | None:
    unit = str(row.get("protected_unit") or "")
    if unit == "user":
        return _safe_int(row.get("n_users_effective") or row.get("n_users"))
    return _safe_int(row.get("n_records_effective") or row.get("n_records") or row.get("baseline_subset_records"))


def _method_sort(method: str) -> int:
    return METHOD_ORDER.get(method, 99)


def _dataset_sort(dataset: str) -> int:
    order = {"cfpb": 0, "amazon_All_Beauty": 1, "yelp_restaurants": 2}
    return order.get(dataset, 99)


def _release_config(row: pd.Series | dict[str, Any], *, compact: bool = True) -> str:
    method = str(row.get("method") or "")
    mechanism = str(row.get("mechanism") or "")
    unit = str(row.get("protected_unit") or "record")
    m = _safe_int(row.get("m"))
    K = _safe_int(row.get("K"))
    L = _safe_int(row.get("L"))
    vocab = _safe_int(row.get("vocabulary_size"))
    cats = _safe_int(row.get("category_universe_size"))
    cat_key = row.get("category_key")
    clusters = _safe_int(row.get("num_clusters_requested"))
    keywords = _safe_int(row.get("keywords_per_summary"))
    B = _safe_float(row.get("user_contribution_bound"))

    if method == "DP-SPIN private plan":
        mech = MECHANISM_DISPLAY.get(mechanism, mechanism)
        parts = [unit, mech]
        if m is not None:
            parts.append(f"m={m}")
        if K is not None:
            parts.append(f"K={K}")
        if L is not None:
            parts.append(f"L={L}")
        if unit == "user" and B is not None:
            parts.append(f"B={_fmt_number(B)}")
        return "; ".join(parts)
    if method == "DP keyword histogram":
        parts = ["fixed public keywords", "Laplace histogram"]
        if vocab is not None:
            parts.append(f"vocab={vocab}")
        if L is not None:
            parts.append(f"top-{L}")
        return "; ".join(parts)
    if method == "DP category histogram":
        label = f"fixed {cat_key} universe" if cat_key not in (None, "", float("nan")) else "fixed category universe"
        parts = [label, "Laplace histogram"]
        if cats is not None:
            parts.append(f"categories={cats}")
        if L is not None:
            parts.append(f"top-{L}")
        return "; ".join(parts)
    if method == "URANIA-style public keywords":
        parts = ["DP clustering", "DP public-keyword histograms"]
        if clusters is not None:
            parts.append(f"clusters={clusters}")
        if keywords is not None:
            parts.append(f"keywords={keywords}")
        return "; ".join(parts)
    return MECHANISM_DISPLAY.get(mechanism, mechanism or method)


def _resolve_summary_path(path_value: Any, reports_root: Path) -> Path:
    """Resolve a summary path recorded by the OpenAI reporting pipeline.
    
    When an archived report contains a project-relative ``outputs/openai_paper``
    path, the function maps that prefix to the supplied reports root if the original
    working-directory path is unavailable.
    """
    p = Path(str(path_value or ""))
    if p.exists():
        return p
    text = str(path_value or "")
    marker = "outputs/openai_paper/"
    if marker in text:
        rel = text.split(marker, 1)[1]
        q = reports_root / rel
        if q.exists():
            return q
    # Fallback for paths that start directly from a dataset directory.
    parts = p.parts
    for i, part in enumerate(parts):
        if part in {"cfpb", "amazon_All_Beauty", "yelp_restaurants"}:
            q = reports_root.joinpath(*parts[i:])
            if q.exists():
                return q
    return p


def _read_dataset_report(report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    judgments_path = report_dir / "judgments" / "judgments.csv"
    comp_items_path = report_dir / "tables" / "comparison_judged_summary_items.csv"
    dpspin_path = report_dir / "tables" / "dpspin_openai_runs.csv"
    judgments = pd.read_csv(judgments_path) if judgments_path.exists() and judgments_path.stat().st_size else pd.DataFrame()
    comp_items = pd.read_csv(comp_items_path) if comp_items_path.exists() and comp_items_path.stat().st_size else pd.DataFrame()
    dpspin = pd.read_csv(dpspin_path) if dpspin_path.exists() and dpspin_path.stat().st_size else pd.DataFrame()
    return judgments, comp_items, dpspin


def _filter_to_comparison(judgments: pd.DataFrame, comp_items: pd.DataFrame, *, unit: str = "record") -> pd.DataFrame:
    if judgments.empty:
        return judgments
    df = judgments.copy()
    if not comp_items.empty and "summary_path" in comp_items.columns and "summary_path" in df.columns:
        allowed = set(comp_items["summary_path"].dropna().astype(str))
        if allowed:
            df = df[df["summary_path"].astype(str).isin(allowed)].copy()
    if unit and "protected_unit" in df.columns:
        # Baselines are record-level. For a main-body cross-method table, keep
        # record-level DP-SPIN so the comparison is not mixed with user-level rows.
        df = df[df["protected_unit"].astype(str).eq(unit)].copy()
    df = df[df["method"].astype(str).isin(METHOD_ORDER)].copy()
    return df


def collect_reports(reports_root: Path, dataset_dirs: Iterable[str] | None = None, *, unit: str = "record") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset-scoped OpenAI report tables from the selected directories."""
    rows = []
    dpspin_rows = []
    dirs = list(dataset_dirs or DATASET_DIRS.values())
    for d in dirs:
        report_dir = reports_root / d
        if not report_dir.exists():
            continue
        judgments, comp_items, dpspin = _read_dataset_report(report_dir)
        comparison = _filter_to_comparison(judgments, comp_items, unit=unit)
        if not comparison.empty:
            rows.append(comparison)
        if not dpspin.empty:
            if unit and "protected_unit" in dpspin.columns:
                dpspin = dpspin[dpspin["protected_unit"].astype(str).eq(unit)].copy()
            dpspin_rows.append(dpspin)
    all_comp = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    all_dpspin = pd.concat(dpspin_rows, ignore_index=True) if dpspin_rows else pd.DataFrame()
    return all_comp, all_dpspin


def _aggregate_scores(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    # Fill display/config fields before grouping.
    df = df.copy()
    df["Dataset"] = df["dataset"].astype(str).map(DATASET_DISPLAY).fillna(df["dataset"].astype(str))
    df["Method"] = df["method"].astype(str).map(METHOD_DISPLAY).fillna(df["method"].astype(str))
    df["Release configuration"] = df.apply(_release_config, axis=1)
    df["Protected units"] = df.apply(_protected_units, axis=1)

    group_cols = ["dataset", "Dataset", "method", "Method", "Release configuration", "epsilon", "Protected units"]
    records = []
    for keys, g in df.groupby(group_cols, dropna=False):
        key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rec = {
            "dataset_sort": _dataset_sort(str(key_map["dataset"])),
            "method_sort": _method_sort(str(key_map["method"])),
            "Dataset": key_map["Dataset"],
            "Method": key_map["Method"],
            "Release configuration": key_map["Release configuration"],
            r"$\varepsilon$": _fmt_eps(key_map["epsilon"]),
            "Protected units": _fmt_number(key_map["Protected units"]),
            r"$N$": len(g),
        }
        label_map = {
            "coverage": "Coverage",
            "specificity": "Specificity",
            "insightfulness": "Insightfulness",
            "faithfulness": "Faithfulness",
            "privacy_safety": "Text Safety",
            "clarity": "Clarity",
        }
        for col in score_cols:
            if col in g.columns:
                rec[label_map[col]] = _fmt_mean_std(g[col])
        records.append(rec)
    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(["dataset_sort", "method_sort", r"$\varepsilon$"]).drop(columns=["dataset_sort", "method_sort"])
    return out


def _latex_table(tab: pd.DataFrame, *, caption: str, label: str, cols: list[str], fontsize: str = "scriptsize") -> str:
    if tab.empty:
        return f"% No rows available for {label}.\n"
    tab = tab[[c for c in cols if c in tab.columns]].copy()
    align = "l" * len(tab.columns)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\{fontsize}",
        r"\resizebox{0.99\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(tab.columns) + r" \\",
        r"\midrule",
    ]
    for _, rr in tab.iterrows():
        cells = []
        for c in tab.columns:
            if c in {r"$\varepsilon$", "Coverage", "Specificity", "Insightfulness", "Faithfulness", "Text Safety", "Clarity"}:
                cells.append(_latex_escape_preserving_math(rr[c]))
            else:
                cells.append(_latex_escape_preserving_math(rr[c]))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    return "\n".join(lines)


def build_main_comparison_table(comparison: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Aggregate release-conditioned scores for DP-SPIN and DP baselines."""
    tab = _aggregate_scores(comparison, MAIN_SCORE_COLUMNS)
    caption = (
        r"Representative OpenAI-judged aggregate-summary quality across datasets. "
        r"Rows use the record-level representative release objects selected for the paper: "
        r"the matched \texttt{DP-SPIN} condition, DP keyword and category histograms, "
        r"and the URANIA-style public-keyword baseline. Protected-unit counts are reported per row "
        r"because representative saved DP-SPIN runs and baseline subsets may use different sizes. "
        r"Scores are assigned on a 1--5 scale "
        r"by the same release-conditioned OpenAI judge. The safety score evaluates the generated "
        r"text and is not a formal differential-privacy guarantee."
    )
    cols = ["Dataset", "Method", "Release configuration", r"$\varepsilon$", "Protected units", r"$N$", "Coverage", "Insightfulness", "Faithfulness", "Text Safety"]
    tex = _latex_table(tab, caption=caption, label="tab:openai-main-comparison", cols=cols, fontsize="scriptsize")
    return tab, tex


def build_main_dpspin_table(dpspin_or_judgments: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Aggregate the selected DP-SPIN verbalization scores across datasets."""
    df = dpspin_or_judgments.copy()
    if "method" in df.columns:
        df = df[df["method"].astype(str).eq("DP-SPIN private plan")].copy()
    tab = _aggregate_scores(df, ["coverage", "insightfulness", "faithfulness", "clarity"])
    caption = (
        r"Representative \texttt{DP-SPIN} OpenAI verbalization results across datasets. "
        r"The decoder receives only the released differentially private semantic plan and public "
        r"decoding instructions. The configuration column gives the protected unit, release "
        r"mechanism, atom dictionary size, assignment sparsity, and released plan size."
    )
    cols = ["Dataset", "Method", "Release configuration", r"$\varepsilon$", "Protected units", r"$N$", "Coverage", "Insightfulness", "Faithfulness", "Clarity"]
    tex = _latex_table(tab, caption=caption, label="tab:openai-main-dpspin", cols=cols, fontsize="scriptsize")
    return tab, tex


def _standardize_text(text: str, max_chars: int) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if max_chars and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _select_dpspin_qualitative_examples(reports_root: Path, dataset_dirs: Iterable[str], *, max_chars: int) -> pd.DataFrame:
    rows = []
    for d in dataset_dirs:
        report_dir = reports_root / d
        all_items = report_dir / "tables" / "all_summary_items.csv"
        if not all_items.exists() or not all_items.stat().st_size:
            continue
        items = pd.read_csv(all_items)
        if items.empty:
            continue
        # Prefer record-level Gaussian eps=1 seed=0, then fall back deterministically.
        cand = items[items["method"].astype(str).eq("DP-SPIN private plan")].copy()
        if "protected_unit" in cand.columns:
            cand = cand[cand["protected_unit"].astype(str).eq("record")].copy()
        for col, val in [("mechanism", "gaussian"), ("epsilon", 1.0), ("seed", 0)]:
            if col in cand.columns and not cand.empty:
                tmp = cand[cand[col].astype(str).eq(str(val))] if col != "epsilon" else cand[pd.to_numeric(cand[col], errors="coerce").eq(float(val))]
                if not tmp.empty:
                    cand = tmp
        if cand.empty:
            continue
        cand = cand.sort_values([c for c in ["dataset", "protected_unit", "mechanism", "epsilon", "seed"] if c in cand.columns])
        row = cand.iloc[0]
        summary_path = _resolve_summary_path(row.get("summary_path"), reports_root)
        if not summary_path.exists():
            continue
        dataset = str(row.get("dataset") or d)
        rows.append({
            "Dataset": DATASET_DISPLAY.get(dataset, dataset),
            "Configuration": _release_config(row),
            "Generated summary": _standardize_text(summary_path.read_text(encoding="utf-8"), max_chars=max_chars),
        })
    return pd.DataFrame(rows)


def build_main_qualitative_table(reports_root: Path, dataset_dirs: Iterable[str], *, max_chars: int = 650) -> tuple[pd.DataFrame, str]:
    """Collect representative generated summaries for the qualitative table."""
    tab = _select_dpspin_qualitative_examples(reports_root, dataset_dirs, max_chars=max_chars)
    caption = (
        r"Representative \texttt{DP-SPIN} OpenAI-generated aggregate insights across datasets. "
        r"Each summary is generated from a released differentially private semantic plan only. "
        r"Text is shortened for display. Full method-wise qualitative examples are reported in the appendix."
    )
    if tab.empty:
        return tab, "% No qualitative examples found.\n"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:openai-main-qualitative}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\resizebox{0.99\textwidth}{!}{%",
        r"\begin{tabular}{p{0.11\textwidth}p{0.26\textwidth}p{0.58\textwidth}}",
        r"\toprule",
        r"Dataset & Release configuration & Generated summary \\",
        r"\midrule",
    ]
    for _, rr in tab.iterrows():
        lines.append(
            _latex_escape_preserving_math(rr["Dataset"]) + " & " +
            _latex_escape_preserving_math(rr["Configuration"]) + " & " +
            _latex_escape(rr["Generated summary"]) + r" \\")
        lines.append(r"\addlinespace[1pt]")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    return tab, "\n".join(lines)


def build_tables(reports_root: Path, output_dir: Path, datasets: list[str] | None, *, unit: str, qualitative_max_chars: int) -> None:
    """Write the selected cross-dataset OpenAI CSV and LaTeX tables."""
    dataset_dirs = [DATASET_DIRS.get(d, d) for d in (datasets or list(DATASET_DIRS))]
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison, all_dpspin = collect_reports(reports_root, dataset_dirs, unit=unit)
    comparison.to_csv(output_dir / "main_comparison_rows.csv", index=False)
    all_dpspin.to_csv(output_dir / "main_dpspin_rows.csv", index=False)

    main_tab, main_tex = build_main_comparison_table(comparison)
    main_tab.to_csv(output_dir / "tab_openai_main_comparison.csv", index=False)
    (output_dir / "tab_openai_main_comparison.tex").write_text(main_tex, encoding="utf-8")

    # Use judgments for DP-SPIN scores when available because the common judge
    # is the release-conditioned judge used in the manuscript. This keeps the DP-SPIN table consistent
    # with the cross-method comparison table.
    dpspin_judgments = comparison[comparison.get("method", pd.Series(dtype=str)).astype(str).eq("DP-SPIN private plan")].copy() if not comparison.empty else pd.DataFrame()
    dpspin_source = dpspin_judgments if not dpspin_judgments.empty else all_dpspin
    dpspin_tab, dpspin_tex = build_main_dpspin_table(dpspin_source)
    dpspin_tab.to_csv(output_dir / "tab_openai_main_dpspin.csv", index=False)
    (output_dir / "tab_openai_main_dpspin.tex").write_text(dpspin_tex, encoding="utf-8")

    qual_tab, qual_tex = build_main_qualitative_table(reports_root, dataset_dirs, max_chars=qualitative_max_chars)
    qual_tab.to_csv(output_dir / "tab_openai_main_qualitative.csv", index=False)
    (output_dir / "tab_openai_main_qualitative_examples.tex").write_text(qual_tex, encoding="utf-8")

    readme = [
        "# OpenAI main-paper tables",
        "",
        "These tables are built from the dataset-scoped OpenAI reports under the reports root.",
        "They do not run OpenAI calls or rerun any DP mechanisms.",
        "",
        "Generated files:",
        "",
        "- `tab_openai_main_comparison.tex`: compact cross-dataset comparison of DP-SPIN and DP baselines.",
        "- `tab_openai_main_dpspin.tex`: compact cross-dataset DP-SPIN verbalization table.",
        "- `tab_openai_main_qualitative_examples.tex`: one DP-SPIN generated example per dataset.",
        "",
        f"Protected-unit filter used for cross-method comparison: `{unit}`.",
        "The record-level filter is the default because the DP histogram and URANIA-style baselines are record-level release objects.",
        "Full per-dataset tables remain in each dataset appendix directory.",
    ]
    (output_dir / "README_OPENAI_MAIN_TABLES.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"Wrote OpenAI main-paper tables to {output_dir}")


def main() -> None:
    """Build cross-dataset OpenAI tables from the command line."""
    p = argparse.ArgumentParser(description="Build compact main-paper OpenAI tables from existing dataset-scoped reports.")
    p.add_argument("--reports-root", default="outputs/openai_paper", help="Root containing cfpb, amazon_All_Beauty, and yelp_restaurants report directories.")
    p.add_argument("--output-dir", default=None, help="Output directory. Defaults to <reports-root>/main_tables.")
    p.add_argument("--datasets", nargs="+", default=None, help="Dataset keys or report directory names. Default: cfpb amazon yelp.")
    p.add_argument("--unit", default="record", choices=["record", "user", ""], help="Protected unit to use in the cross-method comparison table. Default: record.")
    p.add_argument("--qualitative-max-chars", type=int, default=650, help="Maximum displayed characters per qualitative example. Use 0 for no truncation.")
    args = p.parse_args()
    reports_root = Path(args.reports_root)
    output_dir = Path(args.output_dir) if args.output_dir else reports_root / "main_tables"
    build_tables(reports_root, output_dir, args.datasets, unit=args.unit, qualitative_max_chars=args.qualitative_max_chars)


if __name__ == "__main__":
    main()
