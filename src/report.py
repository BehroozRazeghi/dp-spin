"""Generate CSV reports, LaTeX tables, and figures from saved run metrics.

The reporting code does not execute a privacy mechanism or read raw protected text. 
It aggregates artifacts written by completed DP-SPIN runs. 
Generated LaTeX files contain complete ``table`` or ``table*`` environments with captions and labels and can be collected by :mod:`src.paper_assets`.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt


PAPER_PLOT_RC = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply_paper_plot_style() -> None:
    """Apply consistent paper-style matplotlib settings."""
    plt.rcParams.update(PAPER_PLOT_RC)


MECHANISM_LABELS = {
    "gaussian": r"Gaussian sketch + Top-$L$",
    "laplace": r"Laplace sketch + Top-$L$",
    "private_topL": r"Private Top-$L$ admission + Gaussian values",
}

# Mechanism is encoded by line style and marker; dictionary/user setting is
# encoded by color.  This avoids reusing the same visual channel for both
# mechanism and experimental condition.
MECHANISM_LINE_STYLES = {
    "gaussian": "-",
    "laplace": "--",
    "private_topL": "-.",
}
MECHANISM_MARKERS = {
    "gaussian": "o",
    "laplace": "s",
    "private_topL": "^",
}
CONDITION_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def _canonical_mechanism_key(mechanism: object) -> str:
    """Normalize saved mechanism keys without changing assignment sparsity K."""
    key = str(mechanism).strip()
    aliases = {
        "private_topk": "private_topL",
        "private_topK": "private_topL",
        "private_top_l": "private_topL",
        "private_top-l": "private_topL",
        "sparse_private_topL": "private_topL",
    }
    return aliases.get(key, key)


def mechanism_display_name(mechanism: object) -> str:
    """Return a formal mechanism name for legends and tables."""
    key = _canonical_mechanism_key(mechanism)
    return MECHANISM_LABELS.get(key, str(mechanism))


def add_paper_grid() -> None:
    """Add a light major grid for paper figures."""
    plt.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5, alpha=0.35)

from .plot_results import collect_metrics


METRIC_COLS = [
    "sketch_overlap_at_L",
    "sketch_mass_retained_fraction",
    "sketch_support_mass_loss",
    "verifier_UAR",
    "verifier_UCR",
    "verifier_accept",
    "text_token_jaccard",
    "text_keyphrase_jaccard",
    "summary_embedding_cosine",
    "comparative_dp_preferred",
    "probe_summary_mention_rate",
    "probe_plan_plan_admission_rate",
]

GROUP_COLS_CANDIDATES = [
    "experiment_name",
    "mechanism",
    "epsilon",
    "delta",
    "m",
    "K",
    "L",
    "n_records_effective",
    "n_users_effective",
    "user_contribution_bound",
    "probe_support",
]

DISPLAY_NAMES = {
    "experiment_name": "Experiment",
    "mechanism": "Mechanism",
    "epsilon": r"$\varepsilon$",
    "delta": r"$\delta_{\mathrm{eff}}$",
    "m": r"$m$",
    "K": r"$K$",
    "L": r"$L$",
    "n_records_effective": r"$n$",
    "n_users_effective": r"$u$",
    "user_contribution_bound": r"$B$",
    "probe_support": "Probe support $s$",
    "sketch_overlap_at_L": r"$\mathsf{Overlap}@L$",
    "sketch_mass_retained_fraction": "$\\mathsf{MassRetained}$",
    "sketch_support_mass_loss": "$\\mathsf{{Loss}}_L$",
    "verifier_UAR": "$\\mathsf{UAR}$",
    "verifier_UCR": "$\\mathsf{UCR}$",
    "verifier_accept": "$\\mathsf{AcceptRate}$",
    "text_token_jaccard": "$J_{\\mathrm{tok}}$",
    "text_keyphrase_jaccard": "$J_{\\mathrm{kp}}$",
    "summary_embedding_cosine": "$\\mathsf{Cos}_{\\mathrm{emb}}$",
    "comparative_dp_preferred": "$\\mathsf{DPPreferred}$",
    "probe_summary_mention_rate": "$\\mathsf{Mention}_{\\mathrm{text}}$",
    "probe_plan_plan_admission_rate": "$\\mathsf{Adm}_{\\mathrm{plan}}$",
    "num_seeds": "Runs",
}

TEXT_COLUMNS = {"experiment_name", "mechanism"}
METRIC_LIKE_COLUMNS = set(METRIC_COLS + ["num_seeds"])


def latex_escape(value: Any) -> str:
    """Escape non-math text for inclusion in LaTeX tables.

    Metric cells are handled separately because they may contain math
    markup such as ``$\\pm$``.  Configuration values and experiment names are
    escaped here to avoid LaTeX failures caused by underscores in run names.
    """
    if value is None:
        return ""
    s = str(value)
    replacements = {
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
    return "".join(replacements.get(ch, ch) for ch in s)


def sanitize_label(label: str) -> str:
    """Return a LaTeX-safe label containing only common label characters."""
    label = label.replace("_", "-").lower()
    label = re.sub(r"[^a-z0-9:.-]+", "-", label)
    label = re.sub(r"-+", "-", label).strip("-")
    return label or "tab:dpspin-results"


def is_missing(value: Any) -> bool:
    """Return True for values that should be printed as ``--``."""
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _is_uninformative_value(value: Any) -> bool:
    """Return True for table values that carry no experimental information."""
    if is_missing(value):
        return True
    if isinstance(value, str):
        return value.strip() in {"", "--", "N/A", "nan", "None"}
    return False


def _column_has_informative_value(df: pd.DataFrame, column: str) -> bool:
    """Return True if ``column`` contains at least one informative value."""
    if column not in df.columns:
        return False
    return any(not _is_uninformative_value(v) for v in df[column].tolist())


def _drop_uninformative_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are entirely missing or placeholder values.

    This prevents experiment-family tables from showing irrelevant columns such
    as probe support or probe admission in non-probe reports.
    """
    if df.empty:
        return df.copy()
    keep = [c for c in df.columns if _column_has_informative_value(df, c)]
    return df[keep].copy() if keep else df.copy()


def format_cell(value: Any, *, contains_latex: bool = False, column: str | None = None) -> str:
    """Format one scalar cell for CSV-derived LaTeX tables."""
    if is_missing(value):
        return "--"
    if column == "mechanism":
        return mechanism_display_name(value)
    if contains_latex:
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return latex_escape(value)


def mean_pm_std(group: pd.DataFrame, col: str) -> str:
    """Format a grouped metric as ``mean ± std`` in LaTeX math notation."""
    if col not in group.columns or group[col].dropna().empty:
        return "--"
    mean = group[col].mean()
    std = group[col].std(ddof=0)
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def infer_caption_and_label(output_dir: Path, table_kind: str) -> tuple[str, str]:
    """Derive a factual caption and stable LaTeX label from the report path.
    
    ``table_kind`` distinguishes the full summary table from the selected manuscript
    table. Captions describe the displayed quantities without adding interpretive
    claims.
    """
    name = str(output_dir).lower()

    if "probe" in name:
        caption = (
            "Controlled probe-atom results on the CFPB benchmark. Probe admission "
            "and mention rates are reported across probe supports and privacy "
            "levels; plan probe admission denotes admission of the public probe "
            "atom into the released differentially private plan."
        )
        label = "tab:cfpb-probe"
    elif "ablation" in name:
        caption = (
            "Ablation results on the CFPB benchmark. The table reports mean "
            "$\\pm$ standard deviation across runs for dictionary size $m$, "
            "assignment sparsity $K$, and plan size $L$."
        )
        label = "tab:cfpb-ablation"
    elif "hard" in name:
        caption = (
            "Hard-setting CFPB utility results under reduced sample size and a "
            "large semantic dictionary. The table reports mean $\\pm$ standard "
            "deviation across runs."
        )
        label = "tab:cfpb-hard"
    elif "main" in name:
        caption = (
            "Main CFPB utility results for the released DP semantic plan. The "
            "table reports mean $\\pm$ standard deviation across runs."
        )
        label = "tab:cfpb-main"
    else:
        caption = (
            "Aggregate DP-SPIN experiment results. The table reports mean "
            "$\\pm$ standard deviation across runs."
        )
        label = "tab:dpspin-results"

    if table_kind == "paper":
        label += "-compact"
        caption = caption.replace("The table reports", "This compact table reports")

    return caption, sanitize_label(label)


def write_latex_table(
    table: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    *,
    compact: bool,
) -> None:
    """Write a complete LaTeX ``table`` or ``table*`` environment.
    
    The output includes centering, caption, label, size command, and a
    ``booktabs``-style tabular. The consuming document should load ``booktabs``.
    """
    if table.empty:
        path.write_text("% Empty table: no metrics were found.\n", encoding="utf-8")
        return

    columns = list(table.columns)
    display_columns = [DISPLAY_NAMES.get(c, latex_escape(c)) for c in columns]
    alignment = "".join("l" if c in TEXT_COLUMNS else "c" for c in columns)
    env = "table" if compact else "table*"
    size = r"\small" if compact else r"\scriptsize"

    lines = [
        "% Auto-generated by src.report. Do not edit manually.",
        "% Requires: \\usepackage{booktabs,graphicx}",
        rf"\begin{{{env}}}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{sanitize_label(label)}}}",
        size,
        r"\resizebox{0.99\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(display_columns) + r" \\",
        r"\midrule",
    ]

    for _, row in table.iterrows():
        values = [
            format_cell(row[col], contains_latex=(col in METRIC_LIKE_COLUMNS), column=col)
            for col in columns
        ]
        lines.append(" & ".join(values) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            rf"\end{{{env}}}",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Group raw run metrics into a mean/std summary table."""
    group_cols = [c for c in GROUP_COLS_CANDIDATES if _column_has_informative_value(df, c)]
    if not group_cols:
        group_cols = ["mechanism", "epsilon"] if {"mechanism", "epsilon"}.issubset(df.columns) else []

    rows: list[dict[str, Any]] = []
    iterator = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]

    for keys, group in iterator:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row: dict[str, Any] = dict(zip(group_cols, keys))
        for col in METRIC_COLS:
            if col in group.columns:
                row[col] = mean_pm_std(group, col)
        row["num_seeds"] = len(group)
        rows.append(row)

    return pd.DataFrame(rows)


def make_compact_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Select the columns used in the main-text summary table."""
    preferred = [
        "mechanism",
        "epsilon",
        "delta",
        "m",
        "K",
        "L",
        "n_records_effective",
        "n_users_effective",
        "user_contribution_bound",
        "probe_support",
        "sketch_overlap_at_L",
        "sketch_mass_retained_fraction",
        "sketch_support_mass_loss",
        "text_keyphrase_jaccard",
        "summary_embedding_cosine",
        "verifier_UAR",
        "verifier_UCR",
        "verifier_accept",
        "probe_plan_plan_admission_rate",
        "probe_summary_mention_rate",
        "num_seeds",
    ]
    compact = _drop_uninformative_columns(summary)
    keep = [c for c in preferred if c in compact.columns]
    return compact[keep].copy() if keep else compact.copy()



# ---------------------------------------------------------------------------
# Dataset-level assets from grouped summaries
# ---------------------------------------------------------------------------
# These functions generate dataset-level tables and figures for
# Amazon and Yelp from grouped summary CSV files produced by src.summarize_runs.
# They live in src.report so reporting has one public entry point rather than a
# separate dataset_assets.py utility.


def _read_grouped_summary(path: str | None) -> pd.DataFrame:
    """Read a grouped summary CSV, returning an empty dataframe if absent."""
    if not path:
        return pd.DataFrame()
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _non_probe_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows from ordinary runs, excluding controlled probe-atom conditions.

    Probe sweeps change the sample size and dictionary size as the
    injected support changes.  They should not enter ordinary plan-utility
    heatmaps or representative tables, which are meant to summarize the main
    non-probe experimental grid.
    """
    if df.empty or "probe_support" not in df.columns:
        return df.copy()
    return df[df["probe_support"].isna()].copy()


def _probe_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return controlled probe-atom rows, if present."""
    if df.empty or "probe_support" not in df.columns:
        return pd.DataFrame(columns=df.columns)
    return df[df["probe_support"].notna()].copy()


def _safe_file_stem(value: object) -> str:
    """Return a filesystem/label-safe stem."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return s or "value"


def _format_dataset_asset_cell(value: Any, *, column: str | None = None) -> str:
    """Format a scalar cell for dataset-level LaTeX tables."""
    if is_missing(value):
        return "--"
    if column == "mechanism":
        return mechanism_display_name(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return latex_escape(value)


def _dataset_asset_columns(df: pd.DataFrame, protected_unit: str) -> list[str]:
    """Choose the columns retained in grouped dataset result tables."""
    base = [
        "protected_unit",
        "n_records",
        "n_records_mean",
        "n_records_min",
        "n_records_max",
        "n_users",
        "user_contribution_bound",
        "mechanism",
        "m",
        "K",
        "L",
        "epsilon",
        "delta",
        "num_runs",
        "num_seeds",
        "overlap_at_L_mean",
        "overlap_at_L_std",
        "mass_retained_fraction_mean",
        "mass_retained_fraction_std",
        "support_mass_loss_mean",
        "support_mass_loss_std",
        "probe_support",
        "probe_plan_plan_admission_rate_mean",
        "probe_summary_mention_rate_mean",
        "verifier_accept_rate",
        "UAR_mean",
        "UCR_mean",
        "text_keyphrase_jaccard_mean",
        "text_token_jaccard_mean",
        "text_bigram_jaccard_mean",
        "text_tfidf_cosine_mean",
        "summary_embedding_cosine_mean",
        "comparative_dp_preferred_mean",
    ]
    if protected_unit == "record":
        base = [c for c in base if c not in {"n_users", "user_contribution_bound"}]
    return [c for c in base if _column_has_informative_value(df, c)]



def _dataset_table_caption(dataset_name: str, protected_unit: str, *, representative: bool) -> str:
    """Return a short formal caption for dataset-level tables."""
    level = "record-level" if protected_unit == "record" else "user-level"
    if representative:
        return (
            f"Representative {level} results on {dataset_name}. The representative table "
            "uses the preferred operating point when available and reports the selected "
            "assignment sparsity $K$ and plan size $L$."
        )
    return (
        f"Grouped {level} results on {dataset_name}. The table reports all grouped "
        "dictionary sizes, assignment sparsities $K$, released plan sizes $L$, "
        "mechanisms, privacy budgets, and clipping bounds when applicable."
    )


def _preferred_subset(df: pd.DataFrame, column: str, preferred: list[float | int]) -> pd.Series:
    """Return a boolean mask selecting preferred values, with nearest-value fallback."""
    if column not in df.columns or df[column].dropna().empty:
        return pd.Series([True] * len(df), index=df.index)
    vals = pd.to_numeric(df[column], errors="coerce")
    available = sorted(set(float(v) for v in vals.dropna()))
    if not available:
        return pd.Series([True] * len(df), index=df.index)
    chosen = [float(v) for v in preferred if float(v) in available]
    if not chosen:
        target = float(preferred[0])
        chosen = [min(available, key=lambda v: abs(v - target))]
    return vals.isin(chosen)


def _select_largest_available(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Select rows at the largest observed numeric value of ``column``.

    Representative tables should not contain multiple hidden sample-size
    conditions.  Full grouped tables keep all sample sizes; the representative table selects
    one sample size and prints it explicitly.
    """
    if column not in df.columns or df[column].dropna().empty:
        return df
    vals = pd.to_numeric(df[column], errors="coerce")
    if vals.dropna().empty:
        return df
    return df.loc[vals == vals.max()].copy()


def _make_representative_dataset_table(df: pd.DataFrame, protected_unit: str) -> pd.DataFrame:
    """Select one explicit set of conditions for the representative dataset table.
    
    The full grouped table retains every condition. The representative table selects
    one dictionary and assignment setting, one visible protected-unit count, and the
    requested privacy budgets. Missing preferred values are replaced by the nearest
    observed values and remain explicit in the output columns.
    """
    if df.empty:
        return df.copy()
    mask = pd.Series([True] * len(df), index=df.index)
    mask &= _preferred_subset(df, "m", [100])
    mask &= _preferred_subset(df, "K", [3])
    mask &= _preferred_subset(df, "L", [10])
    mask &= _preferred_subset(df, "epsilon", [1, 2])
    if protected_unit == "user":
        mask &= _preferred_subset(df, "user_contribution_bound", [1, 2])
    out = df.loc[mask].copy()
    if out.empty:
        out = df.copy()

    # Select one sample-size condition for the representative table. Without this step,
    # record tables may contain separate n-record settings that look identical
    # after column selection, and user tables may contain separate
    # n-user settings that differ only in a hidden column.
    if protected_unit == "record":
        out = _select_largest_available(out, "n_records")
    elif protected_unit == "user":
        out = _select_largest_available(out, "n_users")

    sort_cols = [c for c in ["mechanism", "user_contribution_bound", "m", "epsilon"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)
    return out


def _dataset_representative_columns(df: pd.DataFrame, protected_unit: str) -> list[str]:
    """Choose the columns shown in representative main-text tables."""
    cols = [
        "mechanism",
        "epsilon",
        "m",
        "K",
        "L",
        "probe_support",
        "n_records",
        "n_users",
        "user_contribution_bound",
        "num_runs",
        "overlap_at_L_mean",
        "mass_retained_fraction_mean",
        "support_mass_loss_mean",
        "probe_plan_plan_admission_rate_mean",
        "probe_summary_mention_rate_mean",
        "text_keyphrase_jaccard_mean",
        "summary_embedding_cosine_mean",
        "verifier_accept_rate",
    ]
    if protected_unit == "record":
        cols = [c for c in cols if c not in {"n_users", "user_contribution_bound"}]
    else:
        cols = [c for c in cols if c != "n_records"]
    return [c for c in cols if _column_has_informative_value(df, c)]


def _write_dataset_latex_table(
    df: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
    protected_unit: str,
    columns: list[str] | None = None,
) -> None:
    """Write a complete LaTeX table from a grouped summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("% Empty table: no grouped runs found.\n", encoding="utf-8")
        return

    cols = columns or _dataset_asset_columns(df, protected_unit)
    if not cols:
        path.write_text("% Empty table: no known grouped-summary columns found.\n", encoding="utf-8")
        return

    display = {
        "protected_unit": "Unit",
        "n_records": r"$n$",
        "n_records_mean": r"$n$ mean",
        "n_records_min": r"$n$ min",
        "n_records_max": r"$n$ max",
        "n_users": r"$u$",
        "user_contribution_bound": r"$B$",
        "mechanism": "Mechanism",
        "m": r"$m$",
        "K": r"$K$",
        "L": r"$L$",
        "epsilon": r"$\varepsilon$",
        "delta": r"$\delta_{\mathrm{eff}}$",
        "delta_effective": r"$\delta_{\mathrm{eff}}$",
        "delta_configured": r"$\delta_{\mathrm{cfg}}$",
        "num_runs": "Runs",
        "num_seeds": "Seeds",
        "overlap_at_L_mean": r"$\mathsf{Overlap}@L$",
        "overlap_at_L_std": "$\\mathsf{Overlap}@L$ std",
        "mass_retained_fraction_mean": "$\\mathsf{MassRetained}$",
        "mass_retained_fraction_std": "$\\mathsf{MassRetained}$ std",
        "support_mass_loss_mean": "$\\mathsf{{Loss}}_L$",
        "support_mass_loss_std": "$\\mathsf{{Loss}}_L$ std",
        "verifier_accept_rate": "$\\mathsf{AcceptRate}$",
        "UAR_mean": "$\\mathsf{UAR}$",
        "UCR_mean": "$\\mathsf{UCR}$",
        "text_keyphrase_jaccard_mean": "$J_{\\mathrm{kp}}$",
        "text_token_jaccard_mean": "$J_{\\mathrm{tok}}$",
        "text_bigram_jaccard_mean": "$J_2$",
        "text_tfidf_cosine_mean": "$\\mathsf{Cos}_{\\mathrm{tfidf}}$",
        "summary_embedding_cosine_mean": "$\\mathsf{Cos}_{\\mathrm{emb}}$",
        "comparative_dp_preferred_mean": "$\\mathsf{DPPreferred}$",
    }

    alignment = "l" + "c" * (len(cols) - 1)
    lines = [
        "% Auto-generated by src.report. Do not edit manually.",
        "% Requires: \\usepackage{booktabs,graphicx}",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{sanitize_label(label)}}}",
        r"\scriptsize",
        r"\resizebox{0.99\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(display.get(c, latex_escape(c)) for c in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df[cols].iterrows():
        lines.append(" & ".join(_format_dataset_asset_cell(row[c], column=c) for c in cols) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _clean_group_value(value: object) -> object:
    """Normalize pandas/numpy values for stable grouping and labels."""
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _format_condition_label(col: str, value: object) -> str:
    """Format non-mechanism condition fields for concise plot legends."""
    value = _clean_group_value(value)
    if col == "user_contribution_bound":
        return fr"$B={float(value):g}$" if isinstance(value, (int, float)) else fr"$B={value}$"
    if col == "m":
        return fr"$m={int(value)}$" if isinstance(value, (int, float)) else fr"$m={value}$"
    if col == "K":
        return fr"$K={int(value)}$" if isinstance(value, (int, float)) else fr"$K={value}$"
    if col == "L":
        return fr"$L={int(value)}$" if isinstance(value, (int, float)) else fr"$L={value}$"
    return f"{col}={value}"


def _condition_sort_key(key: object) -> tuple:
    """Sort condition tuples numerically when possible."""
    if not isinstance(key, tuple):
        key = (key,)
    out = []
    for value in key:
        value = _clean_group_value(value)
        if value is None:
            out.append((1, ""))
        elif isinstance(value, (int, float)):
            out.append((0, float(value)))
        else:
            out.append((0, str(value)))
    return tuple(out)


def _inside_legend_location(metric: str) -> str:
    """Choose an inside legend location that hides as little data as possible."""
    if "loss" in metric or "error" in metric:
        return "upper center"
    return "lower center"


def _has_multiple_values(df: pd.DataFrame, column: str) -> bool:
    """Return True when a grouped-summary column has more than one observed value."""
    return column in df.columns and df[column].dropna().nunique() > 1


def _active_plot_condition_cols(df: pd.DataFrame, protected_unit: str) -> list[str]:
    """Return non-mechanism condition columns that must be visible in plot legends.

    We avoid averaging over hidden experimental conditions. Dictionary size is
    shown when present, user-level clipping bound is shown for user-level runs,
    and assignment sparsity ``K`` or plan size ``L`` is shown whenever the input
    summary contains more than one value. This makes Amazon/Yelp K--L ablations
    explicit in plot legends instead of silently averaging them.
    """
    cols: list[str] = []
    if "m" in df.columns:
        cols.append("m")
    if protected_unit == "user" and "user_contribution_bound" in df.columns:
        cols.append("user_contribution_bound")
    for col in ["K", "L"]:
        if _has_multiple_values(df, col):
            cols.append(col)
    return cols


def _plot_grouped_metric(
    df: pd.DataFrame,
    path: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
    protected_unit: str,
) -> None:
    """Plot a grouped-summary metric against epsilon.

    The same visual encoding is used across dataset-level plots:
    the mechanism determines line style and marker, while the non-mechanism
    condition determines color. The condition includes the atom dictionary size
    ``m``, the user-level clipping bound ``B`` when present, and ``K`` or ``L``
    whenever the input summary contains multiple assignment sparsities or plan
    sizes. Thus K--L ablations are not averaged silently in the figures.
    """
    required = {metric, "epsilon", "mechanism"}
    if df.empty or not required.issubset(df.columns):
        return

    condition_cols = _active_plot_condition_cols(df, protected_unit)
    group_cols = ["mechanism", *condition_cols, "epsilon"]

    d = df.dropna(subset=[metric, "epsilon"]).copy()
    if d.empty:
        return
    d = d.groupby(group_cols, dropna=False)[metric].mean().reset_index()

    label_cols = [c for c in group_cols if c != "epsilon"]

    # Assign colors to experimental conditions, not to mechanisms. This gives
    # the same color semantics to all metrics for the same protected-unit type.
    condition_keys = []
    for key, _ in d.groupby(condition_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        condition_keys.append(tuple(_clean_group_value(v) for v in key))
    condition_keys = sorted(set(condition_keys), key=_condition_sort_key)
    color_map = {
        key: CONDITION_COLORS[i % len(CONDITION_COLORS)]
        for i, key in enumerate(condition_keys)
    }

    apply_paper_plot_style()
    plt.figure(figsize=(6.4, 4.1))

    grouped = list(d.groupby(label_cols, dropna=False))
    def full_key_sort(item: tuple[object, pd.DataFrame]) -> tuple:
        key, _ = item
        if not isinstance(key, tuple):
            key = (key,)
        key_map = dict(zip(label_cols, key))
        mech_order = {"gaussian": 0, "laplace": 1, "private_topL": 2}
        cond_key = tuple(_clean_group_value(key_map[c]) for c in condition_cols)
        return (_condition_sort_key(cond_key), mech_order.get(str(key_map.get("mechanism")).lower(), 99))

    for key, sub in sorted(grouped, key=full_key_sort):
        if not isinstance(key, tuple):
            key = (key,)
        key_map = {col: _clean_group_value(value) for col, value in zip(label_cols, key)}
        mech_key = str(key_map.get("mechanism", "")).lower()
        cond_key = tuple(key_map[c] for c in condition_cols)
        label_parts = [mechanism_display_name(mech_key)]
        label_parts.extend(_format_condition_label(c, key_map[c]) for c in condition_cols)
        sub = sub.sort_values("epsilon")
        plt.plot(
            sub["epsilon"],
            sub[metric],
            color=color_map.get(cond_key),
            marker=MECHANISM_MARKERS.get(mech_key, "o"),
            linestyle=MECHANISM_LINE_STYLES.get(mech_key, "-"),
            linewidth=1.8,
            markersize=4.2,
            label=", ".join(label_parts),
        )

    plt.xlabel(r"Privacy budget $\varepsilon$")
    plt.ylabel(ylabel)
    if (
        "fraction" in metric
        or "overlap" in metric
        or "accept" in metric
        or metric in {"UAR_mean", "UCR_mean"}
    ):
        plt.ylim(-0.02, 1.05)
    add_paper_grid()
    n_handles = len(grouped)
    ncol = 2 if n_handles > 8 else 1
    legend = plt.legend(
        frameon=True,
        fancybox=False,
        framealpha=0.82,
        edgecolor="0.85",
        fontsize=6.8 if n_handles > 10 else 7.2,
        ncol=ncol,
        loc=_inside_legend_location(metric),
    )
    if legend is not None:
        legend.get_frame().set_linewidth(0.5)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def _ordered_values(df: pd.DataFrame, column: str) -> list[Any]:
    """Return observed values in numeric order when possible."""
    if column not in df.columns:
        return []
    vals = [v for v in df[column].dropna().unique().tolist()]
    return sorted(vals, key=lambda x: float(x) if isinstance(x, (int, float)) else str(x))


def _condition_color_map(keys: list[tuple[Any, ...]]) -> dict[tuple[Any, ...], str]:
    """Assign stable colors to non-mechanism plotting conditions."""
    ordered = sorted(set(keys), key=_condition_sort_key)
    return {k: CONDITION_COLORS[i % len(CONDITION_COLORS)] for i, k in enumerate(ordered)}


def _mechanism_handles() -> list[Any]:
    """Legend handles for the release mechanisms."""
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0], [0],
            color="0.25",
            linestyle=MECHANISM_LINE_STYLES[m],
            marker=MECHANISM_MARKERS[m],
            linewidth=1.7,
            label=MECHANISM_LABELS[m],
        )
        for m in ["gaussian", "laplace", "private_topL"]
    ]


def _condition_label(cols: list[str], key: tuple[Any, ...]) -> str:
    """Format one non-mechanism condition key for a legend."""
    if not cols:
        return "all"
    parts = []
    for col, value in zip(cols, key):
        parts.append(_format_condition_label(col, value))
    return ", ".join(parts)


def _condition_handles(cols: list[str], color_map: dict[tuple[Any, ...], str]) -> list[Any]:
    """Legend handles for non-mechanism plotting conditions."""
    from matplotlib.lines import Line2D

    handles = []
    for key in sorted(color_map, key=_condition_sort_key):
        handles.append(
            Line2D(
                [0], [0],
                color=color_map[key],
                linestyle="-",
                linewidth=2.0,
                label=_condition_label(cols, key),
            )
        )
    return handles


def _metric_specs_grouped(df: pd.DataFrame) -> list[tuple[str, str, bool]]:
    """Metrics available in grouped Amazon/Yelp summaries."""
    specs = [
        ("overlap_at_L_mean", r"$\mathsf{Overlap}@L$", True),
        ("mass_retained_fraction_mean", "$\\mathsf{MassRetained}$", True),
        ("support_mass_loss_mean", "$\\mathsf{{Loss}}_L$", False),
        ("verifier_accept_rate", "$\\mathsf{AcceptRate}$", True),
        ("UAR_mean", "$\\mathsf{UAR}$", True),
        ("UCR_mean", "$\\mathsf{UCR}$", True),
        ("text_keyphrase_jaccard_mean", "$J_{\\mathrm{kp}}$", True),
        ("summary_embedding_cosine_mean", "$\\mathsf{Cos}_{\\mathrm{emb}}$", True),
    ]
    return [(m, y, frac) for m, y, frac in specs if m in df.columns and df[m].notna().any()]


def _preferred_filter(df: pd.DataFrame, protected_unit: str, *, K: int = 3, L: int = 10) -> pd.DataFrame:
    """Select the main paper operating point with nearest-value fallback."""
    if df.empty:
        return df.copy()
    out = df.copy()
    for col, target in [("K", K), ("L", L)]:
        if col in out.columns and out[col].notna().any():
            mask = _preferred_subset(out, col, [target])
            cand = out.loc[mask].copy()
            if not cand.empty:
                out = cand
    if protected_unit == "record":
        out = _select_largest_available(out, "n_records")
    elif protected_unit == "user":
        out = _select_largest_available(out, "n_users")
    return out


def _plot_condition_cols(df: pd.DataFrame, protected_unit: str) -> list[str]:
    """Return visible non-mechanism conditions for organized dataset plots."""
    cols: list[str] = []
    if "m" in df.columns and df["m"].notna().any():
        cols.append("m")
    if protected_unit == "user" and "user_contribution_bound" in df.columns and df["user_contribution_bound"].notna().any():
        cols.append("user_contribution_bound")
    return cols


def _safe_metric_stem(metric: str) -> str:
    """Turn a grouped metric name into a short filename stem."""
    stem = metric
    for suffix in ["_mean", "_rate"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    replacements = {
        "overlap_at_L": "overlap_at_L",
        "mass_retained_fraction": "mass_retained_fraction",
        "support_mass_loss": "support_mass_loss",
        "probe_plan_plan_admission_rate": "probe_plan_admission",
        "probe_summary_mention_rate": "probe_summary_mention",
        "verifier_accept": "verifier_accept",
        "UAR": "unsupported_atom_rate",
        "UCR": "unsupported_comparison_rate",
        "text_keyphrase_jaccard": "text_keyphrase_jaccard",
        "summary_embedding_cosine": "summary_embedding_cosine",
    }
    return replacements.get(stem, _safe_file_stem(stem))


def _write_configuration_coverage(df: pd.DataFrame, path: Path) -> None:
    """Write the observed plotting grid so missing endpoints are explicit."""
    if df.empty:
        return
    cols = [c for c in [
        "protected_unit", "n_records", "n_users", "user_contribution_bound",
        "mechanism", "m", "K", "L", "epsilon", "delta", "num_runs", "num_seeds",
    ] if c in df.columns]
    if not cols:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df[cols].sort_values([c for c in cols if c in df.columns]).to_csv(path, index=False)


def _plot_metric_grid_by_KL_grouped(
    df: pd.DataFrame,
    path: Path,
    *,
    metric: str,
    ylabel: str,
    protected_unit: str,
    y_is_fraction: bool,
) -> None:
    """Plot epsilon curves in a K-by-L grid, with visible non-mechanism conditions.

    This is the Amazon/Yelp analogue of the organized CFPB ablation figures.
    The mechanism controls line style and marker. Dictionary size and, for
    user-level runs, the clipping bound control color. No K/L condition is
    averaged inside a panel.
    """
    required = {"mechanism", "epsilon", "K", "L", metric}
    if df.empty or not required.issubset(df.columns):
        return
    d = df.dropna(subset=["epsilon", "K", "L", metric]).copy()
    if d.empty:
        return

    K_vals = _ordered_values(d, "K")
    L_vals = _ordered_values(d, "L")
    condition_cols = _plot_condition_cols(d, protected_unit)
    group_cols = ["mechanism", *condition_cols, "epsilon"]

    condition_keys: list[tuple[Any, ...]] = []
    if condition_cols:
        for key, _ in d.groupby(condition_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            condition_keys.append(tuple(_clean_group_value(v) for v in key))
    else:
        condition_keys = [("all",)]
    color_map = _condition_color_map(condition_keys)

    apply_paper_plot_style()
    fig, axes = plt.subplots(
        len(K_vals),
        len(L_vals),
        figsize=(4.35 * len(L_vals), 3.15 * len(K_vals)),
        sharex=True,
        sharey=True,
    )
    import numpy as _np

    axes_arr = _np.array(axes, dtype=object).reshape(len(K_vals), len(L_vals))
    for i, K_val in enumerate(K_vals):
        for j, L_val in enumerate(L_vals):
            ax = axes_arr[i, j]
            sub = d[(d["K"] == K_val) & (d["L"] == L_val)]
            if sub.empty:
                ax.axis("off")
                continue
            g = sub.groupby(group_cols, dropna=False)[metric].mean().reset_index()
            label_cols = [c for c in group_cols if c != "epsilon"]
            for key, ss in sorted(g.groupby(label_cols, dropna=False), key=lambda item: _condition_sort_key(item[0])):
                if not isinstance(key, tuple):
                    key = (key,)
                key_map = dict(zip(label_cols, key))
                mech = _canonical_mechanism_key(key_map.get("mechanism"))
                cond_key = tuple(_clean_group_value(key_map[c]) for c in condition_cols) if condition_cols else ("all",)
                ss = ss.sort_values("epsilon")
                ax.plot(
                    ss["epsilon"],
                    ss[metric],
                    color=color_map.get(cond_key, "0.25"),
                    linestyle=MECHANISM_LINE_STYLES.get(mech, "-"),
                    marker=MECHANISM_MARKERS.get(mech, "o"),
                    linewidth=1.45,
                    markersize=3.6,
                )
            ax.set_title(fr"$K={_format_dataset_asset_cell(K_val)},\ L={_format_dataset_asset_cell(L_val)}$", fontsize=9)
            if i == len(K_vals) - 1:
                ax.set_xlabel(r"Privacy budget $\varepsilon$")
            if j == 0:
                ax.set_ylabel(ylabel, labelpad=8)
            if y_is_fraction:
                ax.set_ylim(-0.02, 1.05)
            ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5, alpha=0.35)

    note_cols = [c for c in ["n_records", "n_users"] if c in d.columns and d[c].notna().any()]
    if note_cols:
        notes = []
        for col in note_cols:
            vals = sorted(d[col].dropna().unique(), key=lambda x: float(x) if isinstance(x, (int, float)) else str(x))
            if len(vals) == 1:
                label = r"n" if col == "n_records" else r"u"
                notes.append(fr"${label}={_format_dataset_asset_cell(vals[0])}$")
        if notes:
            fig.suptitle("; ".join(notes), y=1.01, fontsize=9)
    fig.legend(handles=_mechanism_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False)
    if condition_cols:
        fig.legend(
            handles=_condition_handles(condition_cols, color_map),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=min(len(color_map), 4),
            frameon=False,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_vs_m_grid_by_KL_grouped(
    df: pd.DataFrame,
    path: Path,
    *,
    metric: str,
    ylabel: str,
    protected_unit: str,
    y_is_fraction: bool,
    preferred_epsilon: float = 1.0,
) -> None:
    """Plot dictionary-size effects at one epsilon, using the same K-by-L grid."""
    required = {"mechanism", "epsilon", "m", "K", "L", metric}
    if df.empty or not required.issubset(df.columns) or df["m"].dropna().nunique() <= 1:
        return
    d = df.copy()
    d = d.loc[_preferred_subset(d, "epsilon", [preferred_epsilon])].copy()
    if protected_unit == "record":
        d = _select_largest_available(d, "n_records")
    elif protected_unit == "user":
        d = _select_largest_available(d, "n_users")
    d = d.dropna(subset=["m", "K", "L", metric]).copy()
    if d.empty:
        return
    K_vals = _ordered_values(d, "K")
    L_vals = _ordered_values(d, "L")
    condition_cols = ["user_contribution_bound"] if protected_unit == "user" and _has_multiple_values(d, "user_contribution_bound") else []
    condition_keys: list[tuple[Any, ...]] = []
    if condition_cols:
        for key, _ in d.groupby(condition_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            condition_keys.append(tuple(_clean_group_value(v) for v in key))
    else:
        condition_keys = [("all",)]
    color_map = _condition_color_map(condition_keys)

    apply_paper_plot_style()
    fig, axes = plt.subplots(len(K_vals), len(L_vals), figsize=(4.35 * len(L_vals), 3.15 * len(K_vals)), sharex=True, sharey=True)
    import numpy as _np

    axes_arr = _np.array(axes, dtype=object).reshape(len(K_vals), len(L_vals))
    for i, K_val in enumerate(K_vals):
        for j, L_val in enumerate(L_vals):
            ax = axes_arr[i, j]
            sub = d[(d["K"] == K_val) & (d["L"] == L_val)]
            if sub.empty:
                ax.axis("off")
                continue
            group_cols = ["mechanism", *condition_cols, "m"]
            g = sub.groupby(group_cols, dropna=False)[metric].mean().reset_index()
            label_cols = [c for c in group_cols if c != "m"]
            for key, ss in sorted(g.groupby(label_cols, dropna=False), key=lambda item: _condition_sort_key(item[0])):
                if not isinstance(key, tuple):
                    key = (key,)
                key_map = dict(zip(label_cols, key))
                mech = _canonical_mechanism_key(key_map.get("mechanism"))
                cond_key = tuple(_clean_group_value(key_map[c]) for c in condition_cols) if condition_cols else ("all",)
                ss = ss.sort_values("m")
                ax.plot(
                    ss["m"],
                    ss[metric],
                    color=color_map.get(cond_key, "0.25"),
                    linestyle=MECHANISM_LINE_STYLES.get(mech, "-"),
                    marker=MECHANISM_MARKERS.get(mech, "o"),
                    linewidth=1.45,
                    markersize=3.6,
                )
            ax.set_title(fr"$K={_format_dataset_asset_cell(K_val)},\ L={_format_dataset_asset_cell(L_val)}$", fontsize=9)
            if i == len(K_vals) - 1:
                ax.set_xlabel(r"Atom dictionary size $m$")
            if j == 0:
                ax.set_ylabel(ylabel, labelpad=8)
            if y_is_fraction:
                ax.set_ylim(-0.02, 1.05)
            ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5, alpha=0.35)
    eps_vals = _ordered_values(d, "epsilon")
    if eps_vals:
        fig.suptitle(fr"$\varepsilon={float(eps_vals[0]):g}$", y=1.01, fontsize=9)
    fig.legend(handles=_mechanism_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False)
    if condition_cols:
        fig.legend(handles=_condition_handles(condition_cols, color_map), loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=min(len(color_map), 4), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_vs_B_grid_by_KL_grouped(
    df: pd.DataFrame,
    path: Path,
    *,
    metric: str,
    ylabel: str,
    y_is_fraction: bool,
    preferred_epsilon: float = 1.0,
) -> None:
    """Plot user-level clipping-bound effects at one epsilon in a K-by-L grid."""
    required = {"mechanism", "epsilon", "user_contribution_bound", "K", "L", metric}
    if df.empty or not required.issubset(df.columns) or df["user_contribution_bound"].dropna().nunique() <= 1:
        return
    d = df.loc[_preferred_subset(df, "epsilon", [preferred_epsilon])].copy()
    d = _select_largest_available(d, "n_users")
    # If dictionary size varies, use the nearest preferred main dictionary size.
    if "m" in d.columns and d["m"].notna().any():
        d = d.loc[_preferred_subset(d, "m", [100])].copy()
    d = d.dropna(subset=["user_contribution_bound", "K", "L", metric]).copy()
    if d.empty:
        return
    K_vals = _ordered_values(d, "K")
    L_vals = _ordered_values(d, "L")
    apply_paper_plot_style()
    fig, axes = plt.subplots(len(K_vals), len(L_vals), figsize=(4.35 * len(L_vals), 3.15 * len(K_vals)), sharex=True, sharey=True)
    import numpy as _np

    axes_arr = _np.array(axes, dtype=object).reshape(len(K_vals), len(L_vals))
    for i, K_val in enumerate(K_vals):
        for j, L_val in enumerate(L_vals):
            ax = axes_arr[i, j]
            sub = d[(d["K"] == K_val) & (d["L"] == L_val)]
            if sub.empty:
                ax.axis("off")
                continue
            g = sub.groupby(["mechanism", "user_contribution_bound"], dropna=False)[metric].mean().reset_index()
            for mech, ss in sorted(g.groupby("mechanism", dropna=False), key=lambda item: _condition_sort_key(item[0])):
                mech = _canonical_mechanism_key(mech)
                ss = ss.sort_values("user_contribution_bound")
                ax.plot(
                    ss["user_contribution_bound"],
                    ss[metric],
                    color="0.25",
                    linestyle=MECHANISM_LINE_STYLES.get(mech, "-"),
                    marker=MECHANISM_MARKERS.get(mech, "o"),
                    linewidth=1.45,
                    markersize=3.6,
                )
            ax.set_title(fr"$K={_format_dataset_asset_cell(K_val)},\ L={_format_dataset_asset_cell(L_val)}$", fontsize=9)
            if i == len(K_vals) - 1:
                ax.set_xlabel(r"User contribution bound $B$")
            if j == 0:
                ax.set_ylabel(ylabel, labelpad=8)
            if y_is_fraction:
                ax.set_ylim(-0.02, 1.05)
            ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5, alpha=0.35)
    note_parts = []
    for col, latex_name in [("epsilon", r"\varepsilon"), ("m", "m"), ("n_users", "u")]:
        if col in d.columns and d[col].notna().any():
            vals = _ordered_values(d, col)
            if len(vals) == 1:
                note_parts.append(fr"${latex_name}={_format_dataset_asset_cell(vals[0])}$")
    if note_parts:
        fig.suptitle("; ".join(note_parts), y=1.01, fontsize=9)
    fig.legend(handles=_mechanism_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_two_panel_utility_grouped(
    df: pd.DataFrame,
    path: Path,
    *,
    protected_unit: str,
) -> None:
    """Plot the two plan-utility measures at the selected main-text operating point."""
    required = {"mechanism", "epsilon", "overlap_at_L_mean", "mass_retained_fraction_mean"}
    if df.empty or not required.issubset(df.columns):
        return
    d = _preferred_filter(df, protected_unit)
    d = d.dropna(subset=["epsilon", "overlap_at_L_mean", "mass_retained_fraction_mean"]).copy()
    if d.empty:
        return
    condition_cols = _plot_condition_cols(d, protected_unit)
    group_cols = ["mechanism", *condition_cols, "epsilon"]
    g = d.groupby(group_cols, dropna=False)[["overlap_at_L_mean", "mass_retained_fraction_mean"]].mean().reset_index()

    condition_keys: list[tuple[Any, ...]] = []
    if condition_cols:
        for key, _ in g.groupby(condition_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            condition_keys.append(tuple(_clean_group_value(v) for v in key))
    else:
        condition_keys = [("all",)]
    color_map = _condition_color_map(condition_keys)

    apply_paper_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharex=True, sharey=True)
    panels = [
        (axes[0], "overlap_at_L_mean", r"$\mathsf{Overlap}@L$"),
        (axes[1], "mass_retained_fraction_mean", "$\\mathsf{MassRetained}$"),
    ]
    label_cols = [c for c in group_cols if c != "epsilon"]
    for ax, metric, title in panels:
        for key, ss in sorted(g.groupby(label_cols, dropna=False), key=lambda item: _condition_sort_key(item[0])):
            if not isinstance(key, tuple):
                key = (key,)
            key_map = dict(zip(label_cols, key))
            mech = _canonical_mechanism_key(key_map.get("mechanism"))
            cond_key = tuple(_clean_group_value(key_map[c]) for c in condition_cols) if condition_cols else ("all",)
            ss = ss.sort_values("epsilon")
            ax.plot(
                ss["epsilon"],
                ss[metric],
                color=color_map.get(cond_key, "0.25"),
                linestyle=MECHANISM_LINE_STYLES.get(mech, "-"),
                marker=MECHANISM_MARKERS.get(mech, "o"),
                linewidth=1.8,
                markersize=4.2,
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"Privacy budget $\varepsilon$")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5, alpha=0.35)
    axes[0].set_ylabel("Plan-recovery score")
    note_parts = []
    for col, latex_name in [("K", "K"), ("L", "L"), ("n_records", "n"), ("n_users", "u")]:
        if col in d.columns and d[col].notna().any():
            vals = _ordered_values(d, col)
            if len(vals) == 1:
                note_parts.append(fr"${latex_name}={_format_dataset_asset_cell(vals[0])}$")
    if note_parts:
        fig.suptitle("; ".join(note_parts), y=1.02, fontsize=9)
    fig.legend(handles=_mechanism_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False)
    if condition_cols:
        fig.legend(handles=_condition_handles(condition_cols, color_map), loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=min(len(color_map), 4), frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _remove_legacy_dataset_plots(plots: Path, protected_unit: str) -> None:
    """Remove older all-curves-in-one-panel dataset plots to avoid confusion."""
    legacy = [
        f"{protected_unit}_overlap_vs_epsilon.png",
        f"{protected_unit}_mass_retained_vs_epsilon.png",
        f"{protected_unit}_support_mass_loss_vs_epsilon.png",
        f"{protected_unit}_verifier_accept_vs_epsilon.png",
        f"{protected_unit}_text_keyphrase_jaccard_vs_epsilon.png",
        f"{protected_unit}_summary_embedding_cosine_vs_epsilon.png",
        f"{protected_unit}_probe_plan_admission_vs_support.png",
        f"{protected_unit}_probe_summary_mention_vs_support.png",
        f"{protected_unit}_probe_admission_vs_support.png",
        f"{protected_unit}_probe_mention_vs_support.png",
    ]
    for name in legacy:
        path = plots / name
        if path.exists():
            path.unlink()



def _select_preferred_numeric(df: pd.DataFrame, column: str, preferred: float | int) -> pd.DataFrame:
    """Select a preferred numeric value, falling back to the nearest observed value."""
    if df.empty or column not in df.columns or df[column].dropna().empty:
        return df
    vals = pd.to_numeric(df[column], errors="coerce")
    available = sorted(set(float(v) for v in vals.dropna()))
    if not available:
        return df
    target = float(preferred)
    chosen = min(available, key=lambda v: abs(v - target))
    return df.loc[vals == chosen].copy()


def _plot_support_loss_heatmap_grouped(
    df: pd.DataFrame,
    path: Path,
    *,
    protected_unit: str,
    preferred_epsilon: float = 1.0,
    preferred_m: int = 100,
    preferred_B: float = 1.0,
) -> None:
    r"""Plot a K/L heatmap of $\mathsf{Loss}_L$ for Amazon/Yelp summaries."""
    metric = "support_mass_loss_mean"
    required = {"mechanism", "K", "L", metric}
    if df.empty or not required.issubset(df.columns):
        return
    d = df.copy()
    d = _select_preferred_numeric(d, "epsilon", preferred_epsilon)
    d = _select_preferred_numeric(d, "m", preferred_m)
    if protected_unit == "user":
        d = _select_preferred_numeric(d, "user_contribution_bound", preferred_B)
        d = _select_largest_available(d, "n_users")
    else:
        d = _select_largest_available(d, "n_records")
    d = d.dropna(subset=["K", "L", metric]).copy()
    if d.empty:
        return

    mechs = [m for m in ["gaussian", "laplace", "private_topL"] if m in set(d["mechanism"].dropna())]
    if not mechs:
        mechs = sorted(d["mechanism"].dropna().unique(), key=str)
    if not mechs:
        return
    vmin = float(d[metric].min())
    vmax = float(d[metric].max())
    apply_paper_plot_style()
    fig, axes = plt.subplots(1, len(mechs), figsize=(4.2 * len(mechs), 3.5), sharex=True, sharey=True)
    if len(mechs) == 1:
        axes = [axes]
    im = None
    for ax, mech in zip(axes, mechs):
        ss = d[d["mechanism"] == mech]
        if ss.empty:
            ax.axis("off")
            continue
        g = ss.groupby(["K", "L"], dropna=False)[metric].mean().reset_index()
        pivot = g.pivot(index="K", columns="L", values=metric).sort_index().sort_index(axis=1)
        if pivot.empty:
            ax.axis("off")
            continue
        im = ax.imshow(pivot.values, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(mechanism_display_name(mech), fontsize=9)
        ax.set_xticks(range(len(pivot.columns)), [str(int(c)) if float(c).is_integer() else str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [str(int(c)) if float(c).is_integer() else str(c) for c in pivot.index])
        ax.set_xlabel(r"Released plan size $L$")
    axes[0].set_ylabel(r"Assignment sparsity $K$")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.86, label=r"$\mathsf{Loss}_L$")
    note_parts = []
    for col, latex_name in [("epsilon", r"\varepsilon"), ("m", "m"), ("user_contribution_bound", "B"), ("n_records", "n"), ("n_users", "u")]:
        if col in d.columns and d[col].notna().any() and d[col].dropna().nunique() == 1:
            value = d[col].dropna().iloc[0]
            note_parts.append(fr"${latex_name}={_format_dataset_asset_cell(value)}$")
    if note_parts:
        fig.suptitle("; ".join(note_parts), y=1.03, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _plot_probe_support_grouped(
    df: pd.DataFrame,
    path: Path,
    *,
    metric: str,
    ylabel: str,
    protected_unit: str,
) -> None:
    """Plot controlled probe-atom admission or mention as a function of support.

    Probe experiments change ``n_records`` or ``n_users`` as the
    injected support changes. The plotting path must not select the
    largest observed sample-size column; doing so would keep only the largest
    probe support.  Instead, it fixes stable operating-point fields when they
    vary, and then averages over seeds at each support level.
    """
    required = {"mechanism", "epsilon", "probe_support", metric}
    if df.empty or not required.issubset(df.columns):
        return
    d = df.dropna(subset=["epsilon", "probe_support", metric]).copy()
    if d.empty:
        return

    # Keep the same operating-point convention as the CFPB probe figures.
    # These selections affect only non-support configuration columns.
    d = _select_preferred_numeric(d, "K", 3)
    d = _select_preferred_numeric(d, "L", 10)
    if protected_unit == "user":
        d = _select_preferred_numeric(d, "user_contribution_bound", 2.0)

    # If several dictionary sizes are present, facet by dictionary size for
    # record-level plots and by clipping bound for user-level plots.  In the
    # common Amazon/Yelp probe-only setting there is a single facet.
    facet_col = None
    if protected_unit == "user" and _has_multiple_values(d, "user_contribution_bound"):
        facet_col = "user_contribution_bound"
    elif _has_multiple_values(d, "m"):
        facet_col = "m"
    facets = _ordered_values(d, facet_col) if facet_col else [None]

    eps_values = _ordered_values(d, "epsilon")
    color_map = _condition_color_map([(e,) for e in eps_values])

    apply_paper_plot_style()
    fig, axes = plt.subplots(1, len(facets), figsize=(5.1 * len(facets), 3.65), sharex=True, sharey=True)
    if len(facets) == 1:
        axes = [axes]

    for ax, facet in zip(axes, facets):
        sub = d if facet is None else d[d[facet_col] == facet]
        g = (
            sub.groupby(["mechanism", "epsilon", "probe_support"], dropna=False)[metric]
            .mean()
            .reset_index()
        )
        for (mech, eps), ss in sorted(
            g.groupby(["mechanism", "epsilon"], dropna=False),
            key=lambda item: (_condition_sort_key(item[0][1]), str(item[0][0])),
        ):
            mech_key = _canonical_mechanism_key(mech)
            ss = ss.sort_values("probe_support")
            ax.plot(
                ss["probe_support"],
                ss[metric],
                color=color_map.get((eps,), "0.25"),
                linestyle=MECHANISM_LINE_STYLES.get(mech_key, "-"),
                marker=MECHANISM_MARKERS.get(mech_key, "o"),
                linewidth=1.55,
                markersize=3.6,
            )
        ax.set_xlabel(r"Probe support $s$")
        ax.set_ylim(-0.02, 1.05)
        if facet_col is not None:
            ax.set_title(_format_condition_label(facet_col, facet), fontsize=10)
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5, alpha=0.35)

    axes[0].set_ylabel(ylabel)

    note_parts = []
    for col, latex_name in [("K", "K"), ("L", "L"), ("m", "m")]:
        if col in d.columns and d[col].notna().any() and d[col].dropna().nunique() == 1:
            value = d[col].dropna().iloc[0]
            note_parts.append(fr"${latex_name}={_format_dataset_asset_cell(value)}$")
    if protected_unit == "user" and "user_contribution_bound" in d.columns and d["user_contribution_bound"].notna().any() and d["user_contribution_bound"].dropna().nunique() == 1:
        value = d["user_contribution_bound"].dropna().iloc[0]
        note_parts.append(fr"$B={_format_dataset_asset_cell(value)}$")
    if note_parts:
        fig.suptitle("; ".join(note_parts), y=1.02, fontsize=9)

    fig.legend(handles=_mechanism_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False)

    from matplotlib.lines import Line2D

    eps_handles = [
        Line2D([0], [0], color=color_map[(eps,)], linewidth=2.0, label=fr"$\varepsilon={float(eps):g}$")
        for eps in eps_values
    ]
    if eps_handles:
        fig.legend(handles=eps_handles, loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=min(len(eps_handles), 5), frameon=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)

def _write_dataset_plot_readme(plots: Path, dataset_name: str, protected_unit: str) -> None:
    """Describe the organized dataset-level figure layout."""
    text = f"""# {dataset_name} {protected_unit}-level plot organization

These figures are regenerated from saved grouped summaries. They do not rerun any differentially private mechanism.

- `*_two_panel_utility_vs_epsilon_by_condition.png` is the paper-facing utility figure at the main operating point, with mechanism encoded by line style and non-mechanism conditions encoded by color.
- `*_vs_epsilon_grid_by_K_L_colored_by_condition.png` keeps all saved K/L curves but places each K,L pair in its own panel. This replaces the older all-curves-in-one-panel plots.
- `*_vs_m_grid_by_K_L_at_epsilon.png` is written when multiple dictionary sizes are available.
- `*_vs_B_grid_by_K_L_at_epsilon_m.png` is written for user-level runs when multiple clipping bounds are available.
- `*_support_mass_loss_heatmap_K_by_L_fixed_epsilon_m*.png` reports $\\mathsf{{Loss}}_L$ as a K/L heatmap at one public operating point.
- `*_probe_*_vs_support.png` is written when saved controlled probe-atom metrics are present.

Mechanism labels describe the release construction: Gaussian full-sketch release, Laplace full-sketch release, and sparse private Top-L atom admission with Gaussian selected-value release.
"""
    plots.mkdir(parents=True, exist_ok=True)
    (plots / f"README_{protected_unit.upper()}_PLOTS.md").write_text(text, encoding="utf-8")


def _write_organized_dataset_plots(
    df: pd.DataFrame,
    plots: Path,
    *,
    dataset_name: str,
    protected_unit: str,
) -> None:
    """Write CFPB-style organized figures for Amazon/Yelp grouped summaries."""
    if df.empty:
        return
    plots.mkdir(parents=True, exist_ok=True)
    _remove_legacy_dataset_plots(plots, protected_unit)

    general = _non_probe_rows(df)
    probes = _probe_rows(df)

    # Ordinary utility figures are built only from non-probe rows.  Otherwise,
    # controlled probe sweeps can create one-cell heatmaps because probe support
    # changes n_records/n_users and adds one probe atom to the dictionary.
    if not general.empty:
        _write_configuration_coverage(general, plots / f"{protected_unit}_configuration_coverage.csv")

        _plot_two_panel_utility_grouped(
            general,
            plots / f"{protected_unit}_two_panel_utility_vs_epsilon_by_condition.png",
            protected_unit=protected_unit,
        )
        heatmap_name = (
            f"{protected_unit}_support_mass_loss_heatmap_K_by_L_fixed_epsilon_m_B.png"
            if protected_unit == "user"
            else f"{protected_unit}_support_mass_loss_heatmap_K_by_L_fixed_epsilon_m.png"
        )
        _plot_support_loss_heatmap_grouped(
            general,
            plots / heatmap_name,
            protected_unit=protected_unit,
        )

    if not probes.empty and "probe_plan_plan_admission_rate_mean" in probes.columns:
        _plot_probe_support_grouped(
            probes,
            plots / f"{protected_unit}_probe_plan_admission_vs_support.png",
            metric="probe_plan_plan_admission_rate_mean",
            ylabel=r"$\mathsf{Adm}_{\mathrm{plan}}$",
            protected_unit=protected_unit,
        )
    if not probes.empty and "probe_summary_mention_rate_mean" in probes.columns:
        _plot_probe_support_grouped(
            probes,
            plots / f"{protected_unit}_probe_summary_mention_vs_support.png",
            metric="probe_summary_mention_rate_mean",
            ylabel=r"$\mathsf{Mention}_{\mathrm{text}}$",
            protected_unit=protected_unit,
        )

    for metric, ylabel, is_frac in _metric_specs_grouped(general):
        if metric.startswith("probe_"):
            continue
        stem = _safe_metric_stem(metric)
        _plot_metric_grid_by_KL_grouped(
            general,
            plots / f"{protected_unit}_{stem}_vs_epsilon_grid_by_K_L_colored_by_condition.png",
            metric=metric,
            ylabel=ylabel,
            protected_unit=protected_unit,
            y_is_fraction=is_frac,
        )
        _plot_metric_vs_m_grid_by_KL_grouped(
            general,
            plots / f"{protected_unit}_{stem}_vs_m_grid_by_K_L_at_epsilon.png",
            metric=metric,
            ylabel=ylabel,
            protected_unit=protected_unit,
            y_is_fraction=is_frac,
        )
        if protected_unit == "user":
            _plot_metric_vs_B_grid_by_KL_grouped(
                general,
                plots / f"{protected_unit}_{stem}_vs_B_grid_by_K_L_at_epsilon_m.png",
                metric=metric,
                ylabel=ylabel,
                y_is_fraction=is_frac,
            )

    _write_dataset_plot_readme(plots, dataset_name, protected_unit)

def _write_dataset_assets_for_unit(
    df: pd.DataFrame,
    out: Path,
    *,
    dataset_name: str,
    protected_unit: str,
) -> None:
    """Write CSV, LaTeX, and plots for one protected-unit summary."""
    if df.empty:
        return

    tables = out / "tables"
    plots = out / "plots"
    tables.mkdir(parents=True, exist_ok=True)

    df.to_csv(tables / f"{protected_unit}_summary_table.csv", index=False)
    _write_dataset_latex_table(
        df,
        tables / f"{protected_unit}_summary_table.tex",
        caption=_dataset_table_caption(dataset_name, protected_unit, representative=False),
        label=f"tab:{_safe_file_stem(dataset_name).lower()}-{protected_unit}-summary",
        protected_unit=protected_unit,
    )

    representative_source = _non_probe_rows(df)
    if representative_source.empty:
        representative_source = df
    representative = _make_representative_dataset_table(representative_source, protected_unit)
    representative.to_csv(tables / f"{protected_unit}_representative_table.csv", index=False)
    _write_dataset_latex_table(
        representative,
        tables / f"{protected_unit}_representative_table.tex",
        caption=_dataset_table_caption(dataset_name, protected_unit, representative=True),
        label=f"tab:{_safe_file_stem(dataset_name).lower()}-{protected_unit}-representative",
        protected_unit=protected_unit,
        columns=_dataset_representative_columns(representative, protected_unit),
    )

    _write_organized_dataset_plots(
        df,
        plots,
        dataset_name=dataset_name,
        protected_unit=protected_unit,
    )


def make_dataset_assets(
    *,
    dataset_name: str,
    out: str,
    record_summary: str | None = None,
    user_summary: str | None = None,
) -> None:
    """Generate dataset-level tables and figures from grouped summaries."""
    out_path = Path(out)
    record = _read_grouped_summary(record_summary)
    user = _read_grouped_summary(user_summary)
    _write_dataset_assets_for_unit(record, out_path, dataset_name=dataset_name, protected_unit="record")
    _write_dataset_assets_for_unit(user, out_path, dataset_name=dataset_name, protected_unit="user")
    print(f"Wrote dataset tables/plots under {out_path / 'tables'} and {out_path / 'plots'}")


def make_reports(runs_dir: str, output_dir: str) -> None:
    """Aggregate one experiment family into CSV and full LaTeX tables."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = collect_metrics(runs_dir)
    df.to_csv(out / "all_metrics.csv", index=False)

    if df.empty:
        (out / "README_REPORT.md").write_text(
            f"# Empty report\n\nNo metrics were found in `{runs_dir}`.\n",
            encoding="utf-8",
        )
        print(f"No metrics found in {runs_dir}")
        return

    summary = _drop_uninformative_columns(build_summary_table(df))
    summary.to_csv(out / "summary_table.csv", index=False)

    caption, label = infer_caption_and_label(out, "summary")
    write_latex_table(summary, out / "summary_table.tex", caption, label, compact=False)

    compact_table = make_compact_table(summary)
    compact_table.to_csv(out / "paper_table.csv", index=False)

    caption, label = infer_caption_and_label(out, "paper")
    write_latex_table(compact_table, out / "paper_table.tex", caption, label, compact=True)

    (out / "README_REPORT.md").write_text(
        "# DP-SPIN report outputs\n\n"
        f"Source runs: `{runs_dir}`\n\n"
        "- `all_metrics.csv`: one row per run.\n"
        "- `summary_table.csv`: grouped mean/std table.\n"
        "- `summary_table.tex`: full LaTeX table environment for appendix use.\n"
        "- `paper_table.csv`: compact grouped table.\n"
        "- `paper_table.tex`: compact LaTeX table environment for paper use.\n\n"
        "The `.tex` files are complete table environments with captions and labels.\n",
        encoding="utf-8",
    )

    print(f"Wrote reports to {out}")


def main() -> None:
    """Command-line entry point for report and dataset-asset generation."""
    parser = argparse.ArgumentParser(description="Create DP-SPIN CSV, LaTeX reports, and dataset-level assets.")

    # Existing report mode: aggregate raw run directories.
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--output-dir", default="outputs/reports")

    # Dataset-asset mode: consume grouped summaries from src.summarize_runs.
    parser.add_argument("--dataset-name", help="Dataset name for dataset-level tables/figures.")
    parser.add_argument("--out", help="Dataset output directory for tables/plots.")
    parser.add_argument("--record-summary", help="Grouped record-level summary CSV from src.summarize_runs.")
    parser.add_argument("--user-summary", help="Grouped user-level summary CSV from src.summarize_runs.")
    args = parser.parse_args()

    if args.dataset_name or args.out or args.record_summary or args.user_summary:
        if not args.dataset_name or not args.out:
            parser.error("dataset-asset mode requires --dataset-name and --out")
        make_dataset_assets(
            dataset_name=args.dataset_name,
            out=args.out,
            record_summary=args.record_summary,
            user_summary=args.user_summary,
        )
    else:
        make_reports(args.runs_dir, args.output_dir)


if __name__ == "__main__":
    main()
