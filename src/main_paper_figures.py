"""Generate main-text figures from saved cross-dataset reports.

This report-only module reads existing CFPB, Amazon, and Yelp summary tables. 
It does not rerun privacy mechanisms, rebuild atom dictionaries, decode summaries, or call external APIs. 
Full parameter grids remain in the dataset-specific and appendix reports.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PAPER_RC = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
}

MECHANISM_ORDER = {"gaussian": 0, "laplace": 1, "private_topL": 2}
MECHANISM_LABELS = {
    "gaussian": r"Gaussian sketch",
    "laplace": r"Laplace sketch",
    "private_topL": r"Private top-$L$ admission",
}
MECHANISM_LINE_STYLES = {"gaussian": "-", "laplace": "--", "private_topL": ":"}
MECHANISM_MARKERS = {"gaussian": "o", "laplace": "s", "private_topL": "^"}

# Okabe--Ito-style colors for colorblind-safe dataset encoding.
DATASET_COLORS = {
    "cfpb": "#0072B2",    # blue
    "amazon": "#E69F00",  # orange
    "yelp": "#009E73",    # green
}
MECHANISM_COLORS = {
    "gaussian": "#000000",
    "laplace": "#666666",
    "private_topL": "#333333",
}


@dataclass(frozen=True)
class DatasetSpec:
    """Describe the saved report locations and preferred atom count for one dataset."""
    key: str
    display_name: str
    preferred_m: int
    output_root: Path
    record_summary: Path | None = None
    user_summary: Path | None = None
    cfpb_main_metrics: Path | None = None
    cfpb_probe_metrics: Path | None = None


def dataset_specs(reports_root: Path) -> list[DatasetSpec]:
    """Return the CFPB, Amazon, and Yelp report specifications."""
    return [
        DatasetSpec(
            key="cfpb",
            display_name="CFPB",
            preferred_m=200,
            output_root=reports_root / "cfpb",
            cfpb_main_metrics=reports_root / "cfpb" / "main" / "metrics_summary.csv",
            cfpb_probe_metrics=reports_root / "cfpb" / "probe" / "metrics_summary.csv",
        ),
        DatasetSpec(
            key="amazon",
            display_name="Amazon",
            preferred_m=46,
            output_root=reports_root / "amazon_All_Beauty",
            record_summary=reports_root / "amazon_All_Beauty" / "summaries" / "record" / "record_summary.csv",
            user_summary=reports_root / "amazon_All_Beauty" / "summaries" / "user" / "user_summary.csv",
        ),
        DatasetSpec(
            key="yelp",
            display_name="Yelp",
            preferred_m=100,
            output_root=reports_root / "yelp_restaurants",
            record_summary=reports_root / "yelp_restaurants" / "summaries" / "record" / "record_summary.csv",
            user_summary=reports_root / "yelp_restaurants" / "summaries" / "user" / "user_summary.csv",
        ),
    ]


def apply_style() -> None:
    """Apply the shared matplotlib settings used by main-text figures."""
    plt.rcParams.update(PAPER_RC)


def ensure_dir(path: Path) -> Path:
    """Create a directory and return its path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def canonical_mechanism(x: Any) -> str:
    """Normalize saved aliases for a plan-release mechanism."""
    raw = str(x).strip()
    aliases = {
        "private_topk": "private_topL",
        "private_topK": "private_topL",
        "private_top_l": "private_topL",
        "private_top-l": "private_topL",
        "sparse_private_topL": "private_topL",
        "private_topl": "private_topL",
    }
    return aliases.get(raw, raw)


def safe_float(x: Any) -> float | None:
    """Convert a value to a finite float or return ``None``."""
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return None
        y = float(x)
        return y if math.isfinite(y) else None
    except Exception:
        return None


def safe_int(x: Any) -> int | None:
    """Convert a finite numeric value to the nearest integer."""
    y = safe_float(x)
    if y is None:
        return None
    return int(round(y))


def fmt_num(x: Any) -> str:
    """Format a numeric configuration value for labels and captions."""
    y = safe_float(x)
    if y is None:
        return str(x)
    if abs(y - round(y)) < 1e-9:
        return str(int(round(y)))
    return f"{y:g}"


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _rename_if_present(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    rename = {k: v for k, v in mapping.items() if k in df.columns and v not in df.columns}
    return df.rename(columns=rename)


def _normalize_review_summary(df: pd.DataFrame, *, spec: DatasetSpec, unit: str) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["dataset"] = spec.key
    d["dataset_display"] = spec.display_name
    d["protected_unit"] = unit
    d = _rename_if_present(d, {
        "overlap_at_L_mean": "overlap_at_L",
        "mass_retained_fraction_mean": "mass_retained_fraction",
        "support_mass_loss_mean": "support_mass_loss",
        "probe_plan_plan_admission_rate_mean": "probe_plan_plan_admission_rate",
        "probe_summary_mention_rate_mean": "probe_summary_mention_rate",
    })
    if "mechanism" in d.columns:
        d["mechanism"] = d["mechanism"].map(canonical_mechanism)
    return d


def _normalize_cfpb_metrics(df: pd.DataFrame, *, spec: DatasetSpec, probe: bool = False) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["dataset"] = spec.key
    d["dataset_display"] = spec.display_name
    d["protected_unit"] = "record"
    d = _rename_if_present(d, {
        "sketch_overlap_at_L": "overlap_at_L",
        "sketch_mass_retained_fraction": "mass_retained_fraction",
        "sketch_support_mass_loss": "support_mass_loss",
        "n_records_effective": "n_records",
        "n_users_effective": "n_users",
    })
    if "mechanism" in d.columns:
        d["mechanism"] = d["mechanism"].map(canonical_mechanism)
    if probe and "probe_support" not in d.columns:
        # Leave empty; some older metrics summaries do not include probe support.
        d["probe_support"] = np.nan
    return d


def load_all_results(reports_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ordinary and controlled-probe metrics into a common column schema."""
    ordinary: list[pd.DataFrame] = []
    probe: list[pd.DataFrame] = []
    for spec in dataset_specs(reports_root):
        if spec.key == "cfpb":
            main_path = _first_existing([
                spec.cfpb_main_metrics or Path("__missing__"),
                spec.output_root / "main" / "reports" / "summary_table.csv",
            ])
            probe_path = _first_existing([
                spec.cfpb_probe_metrics or Path("__missing__"),
                spec.output_root / "probe" / "reports" / "summary_table.csv",
            ])
            ordinary.append(_normalize_cfpb_metrics(_read_csv(main_path), spec=spec, probe=False))
            probe.append(_normalize_cfpb_metrics(_read_csv(probe_path), spec=spec, probe=True))
        else:
            ordinary.append(_normalize_review_summary(_read_csv(spec.record_summary), spec=spec, unit="record"))
            ordinary.append(_normalize_review_summary(_read_csv(spec.user_summary), spec=spec, unit="user"))
            # Controlled-probe review rows are stored in the same grouped summary files.
            probe.append(_normalize_review_summary(_read_csv(spec.record_summary), spec=spec, unit="record"))
            probe.append(_normalize_review_summary(_read_csv(spec.user_summary), spec=spec, unit="user"))
    ordinary_df = pd.concat([x for x in ordinary if not x.empty], ignore_index=True) if any(not x.empty for x in ordinary) else pd.DataFrame()
    probe_df = pd.concat([x for x in probe if not x.empty], ignore_index=True) if any(not x.empty for x in probe) else pd.DataFrame()
    return ordinary_df, probe_df


def _drop_probe_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "probe_support" not in df.columns:
        return df.copy()
    s = pd.to_numeric(df["probe_support"], errors="coerce")
    return df[s.isna()].copy()


def _keep_probe_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "probe_support" not in df.columns:
        return pd.DataFrame()
    s = pd.to_numeric(df["probe_support"], errors="coerce")
    return df[s.notna()].copy()


def _select_nearest_numeric(df: pd.DataFrame, column: str, preferred: float | int | None) -> pd.DataFrame:
    if df.empty or preferred is None or column not in df.columns:
        return df
    vals = pd.to_numeric(df[column], errors="coerce")
    available = sorted(set(float(v) for v in vals.dropna()))
    if not available:
        return df
    chosen = min(available, key=lambda v: abs(v - float(preferred)))
    return df[vals.eq(chosen)].copy()


def _select_best_protected_count(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Select one protected-unit count with broad mechanism and privacy-budget coverage."""
    col = "n_users" if unit == "user" else "n_records"
    if df.empty or col not in df.columns or df[col].dropna().empty:
        return df
    vals = pd.to_numeric(df[col], errors="coerce")
    candidates = []
    for val in sorted(set(float(v) for v in vals.dropna())):
        sub = df[vals.eq(val)]
        coverage = sub[[c for c in ["mechanism", "epsilon", "seed"] if c in sub.columns]].drop_duplicates().shape[0]
        mechanisms = sub["mechanism"].dropna().nunique() if "mechanism" in sub.columns else 0
        eps = sub["epsilon"].dropna().nunique() if "epsilon" in sub.columns else 0
        candidates.append((coverage, mechanisms, eps, val))
    if not candidates:
        return df
    # Prefer broad coverage; break ties by larger sample/user count.
    chosen = max(candidates, key=lambda z: (z[0], z[1], z[2], z[3]))[3]
    return df[vals.eq(chosen)].copy()


def _select_main_operating_point(df: pd.DataFrame, *, spec: DatasetSpec, unit: str, K: int, L: int, B: float | None = None) -> pd.DataFrame:
    d = df.copy()
    d = _drop_probe_rows(d)
    if "dataset" in d.columns:
        d = d[d["dataset"].eq(spec.key)].copy()
    if "protected_unit" in d.columns:
        d = d[d["protected_unit"].eq(unit)].copy()
    d = _select_nearest_numeric(d, "K", K)
    d = _select_nearest_numeric(d, "L", L)
    d = _select_nearest_numeric(d, "m", spec.preferred_m)
    if unit == "user":
        d = _select_nearest_numeric(d, "user_contribution_bound", B)
    d = _select_best_protected_count(d, unit)
    return d


def _metric_present(df: pd.DataFrame, metric: str) -> bool:
    return metric in df.columns and pd.to_numeric(df[metric], errors="coerce").notna().any()


def _aggregate_metric(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    d = df.copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d["epsilon"] = pd.to_numeric(d["epsilon"], errors="coerce") if "epsilon" in d.columns else np.nan
    d = d.dropna(subset=[metric])
    if d.empty:
        return d
    return d.groupby(group_cols, dropna=False)[metric].mean().reset_index()


def _mechanism_handles(color: str = "0.25") -> list[Line2D]:
    handles: list[Line2D] = []
    for mech in ["gaussian", "laplace", "private_topL"]:
        handles.append(Line2D(
            [0], [0], color=color, linestyle=MECHANISM_LINE_STYLES[mech],
            marker=MECHANISM_MARKERS[mech], linewidth=1.8, markersize=4.4,
            label=MECHANISM_LABELS[mech]
        ))
    return handles


def _dataset_handles(dataset_keys: Iterable[str]) -> list[Line2D]:
    present = set(str(k) for k in dataset_keys)
    handles: list[Line2D] = []
    for key in ["cfpb", "amazon", "yelp"]:
        if key in present:
            label = {"cfpb": "CFPB", "amazon": "Amazon", "yelp": "Yelp"}[key]
            handles.append(Line2D([0], [0], color=DATASET_COLORS[key], linewidth=2.2, label=label))
    return handles


def _format_axes(ax: plt.Axes, *, ylim01: bool = False) -> None:
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.28)
    ax.tick_params(axis="both", which="major", length=3.0, width=0.75)
    if ylim01:
        ax.set_ylim(-0.02, 1.04)
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])


def _add_bottom_legends(
    fig: plt.Figure,
    *,
    dataset_keys: Iterable[str] | None = None,
    mechanism: bool = True,
    y_dataset: float = 0.085,
    y_mechanism: float = 0.015,
) -> None:
    if dataset_keys is not None:
        handles = _dataset_handles(dataset_keys)
        if handles:
            fig.legend(
                handles=handles, loc="lower center", bbox_to_anchor=(0.5, y_dataset),
                ncol=len(handles), frameon=False, handlelength=2.8, columnspacing=1.4,
                borderaxespad=0.0,
            )
    if mechanism:
        fig.legend(
            handles=_mechanism_handles(), loc="lower center", bbox_to_anchor=(0.5, y_mechanism),
            ncol=3, frameon=False, handlelength=2.8, columnspacing=1.4,
            borderaxespad=0.0,
        )


def _finish_panel_figure(
    fig: plt.Figure,
    *,
    xlabel: str,
    ylabel: str,
    bottom: float = 0.30,
    top: float = 0.86,
    left: float = 0.085,
    right: float = 0.995,
    wspace: float = 0.22,
) -> None:
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace)
    fig.supxlabel(xlabel, y=bottom - 0.12, fontsize=PAPER_RC["axes.labelsize"])
    fig.supylabel(ylabel, x=0.012, fontsize=PAPER_RC["axes.labelsize"])


def _savefig(fig: plt.Figure, path: Path) -> None:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if path.suffix.lower() == ".png":
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _fixed_note(df: pd.DataFrame, unit: str) -> str:
    parts = []
    for col, name in [("K", "K"), ("L", "L"), ("m", "m"), ("user_contribution_bound", "B")]:
        if col in df.columns and df[col].notna().any():
            vals = sorted(set(pd.to_numeric(df[col], errors="coerce").dropna()))
            if len(vals) == 1 and not (col == "user_contribution_bound" and unit != "user"):
                parts.append(fr"${name}={fmt_num(vals[0])}$")
    count_col = "n_users" if unit == "user" else "n_records"
    count_name = "u" if unit == "user" else "n"
    if count_col in df.columns and df[count_col].notna().any():
        vals = sorted(set(pd.to_numeric(df[count_col], errors="coerce").dropna()))
        if len(vals) == 1:
            parts.append(fr"${count_name}={fmt_num(vals[0])}$")
    return "; ".join(parts)


def plot_record_plan_utility(df: pd.DataFrame, out: Path, *, K: int, L: int) -> bool:
    """Plot record-level overlap and retained mass across datasets."""
    specs = dataset_specs(Path("outputs"))
    rows = []
    for spec in specs:
        d = _select_main_operating_point(df, spec=spec, unit="record", K=K, L=L)
        if not d.empty:
            rows.append(d)
    if not rows:
        return False
    d = pd.concat(rows, ignore_index=True)
    metrics = [("overlap_at_L", r"$\mathsf{Overlap}@L$"), ("mass_retained_fraction", r"$\mathsf{MassRetained}$")]
    if not all(_metric_present(d, m) for m, _ in metrics):
        return False

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), sharex=True, sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        g = _aggregate_metric(d, ["dataset", "mechanism", "epsilon"], metric)
        for (ds, mech), ss in sorted(
            g.groupby(["dataset", "mechanism"], dropna=False),
            key=lambda item: (str(item[0][0]), MECHANISM_ORDER.get(str(item[0][1]), 99)),
        ):
            mech = canonical_mechanism(mech)
            ss = ss.sort_values("epsilon")
            ax.plot(
                ss["epsilon"], ss[metric],
                color=DATASET_COLORS.get(str(ds), "0.25"),
                linestyle=MECHANISM_LINE_STYLES.get(mech, "-"),
                marker=MECHANISM_MARKERS.get(mech, "o"),
                linewidth=1.7, markersize=3.8,
            )
        ax.set_title(title, pad=5)
        _format_axes(ax, ylim01=True)
    _finish_panel_figure(
        fig,
        xlabel=r"Privacy budget $\varepsilon$",
        ylabel="Record-level plan-recovery score",
        bottom=0.34,
        top=0.88,
        left=0.085,
        wspace=0.20,
    )
    _add_bottom_legends(fig, dataset_keys=d["dataset"].dropna().unique(), y_dataset=0.085, y_mechanism=0.018)
    _savefig(fig, out)
    return True



def plot_record_support_loss(df: pd.DataFrame, out: Path, *, K: int, L: int) -> bool:
    """Plot record-level support-mass loss across datasets."""
    specs = dataset_specs(Path("outputs"))
    rows = []
    for spec in specs:
        d = _select_main_operating_point(df, spec=spec, unit="record", K=K, L=L)
        if not d.empty:
            rows.append(d)
    if not rows:
        return False
    d = pd.concat(rows, ignore_index=True)
    metric = "support_mass_loss"
    if not _metric_present(d, metric):
        return False

    ds_order = [k for k in ["cfpb", "amazon", "yelp"] if k in set(d["dataset"].dropna())]
    if not ds_order:
        return False

    apply_style()
    fig, axes = plt.subplots(1, len(ds_order), figsize=(7.25, 2.95), sharex=True, sharey=False)
    if len(ds_order) == 1:
        axes = [axes]
    g = _aggregate_metric(d, ["dataset", "mechanism", "epsilon"], metric)
    for ax, ds in zip(axes, ds_order):
        sub = g[g["dataset"].eq(ds)].copy()
        for mech, ss in sorted(sub.groupby("mechanism", dropna=False), key=lambda item: MECHANISM_ORDER.get(str(item[0]), 99)):
            mech = canonical_mechanism(mech)
            ss = ss.sort_values("epsilon")
            ax.plot(
                ss["epsilon"], ss[metric],
                color="0.25",
                linestyle=MECHANISM_LINE_STYLES.get(mech, "-"),
                marker=MECHANISM_MARKERS.get(mech, "o"),
                linewidth=1.7, markersize=3.8,
            )
        ax.set_title({"cfpb": "CFPB", "amazon": "Amazon", "yelp": "Yelp"}.get(ds, str(ds)), pad=5)
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.8)
        ax.set_ylim(bottom=0)
        _format_axes(ax, ylim01=False)
        ax.grid(True, which="both", linestyle="--", linewidth=0.45, alpha=0.28)
    _finish_panel_figure(
        fig,
        xlabel=r"Privacy budget $\varepsilon$",
        ylabel=r"Support-mass loss $\mathsf{Loss}_L$",
        bottom=0.27,
        top=0.86,
        left=0.075,
        wspace=0.25,
    )
    fig.legend(
        handles=_mechanism_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.015),
        ncol=3, frameon=False, handlelength=2.8, columnspacing=1.4, borderaxespad=0.0,
    )
    _savefig(fig, out)
    return True



def plot_user_plan_utility(df: pd.DataFrame, out: Path, *, K: int, L: int, B: float) -> bool:
    """Plot user-level overlap and retained mass for review datasets."""
    rows = []
    for spec in dataset_specs(Path("outputs")):
        if spec.key == "cfpb":
            continue
        d = _select_main_operating_point(df, spec=spec, unit="user", K=K, L=L, B=B)
        if not d.empty:
            rows.append(d)
    if not rows:
        return False
    d = pd.concat(rows, ignore_index=True)
    metrics = [("overlap_at_L", r"$\mathsf{Overlap}@L$"), ("mass_retained_fraction", r"$\mathsf{MassRetained}$")]
    if not all(_metric_present(d, m) for m, _ in metrics):
        return False

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), sharex=True, sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        g = _aggregate_metric(d, ["dataset", "mechanism", "epsilon"], metric)
        for (ds, mech), ss in sorted(
            g.groupby(["dataset", "mechanism"], dropna=False),
            key=lambda item: (str(item[0][0]), MECHANISM_ORDER.get(str(item[0][1]), 99)),
        ):
            mech = canonical_mechanism(mech)
            ss = ss.sort_values("epsilon")
            ax.plot(
                ss["epsilon"], ss[metric],
                color=DATASET_COLORS.get(str(ds), "0.25"),
                linestyle=MECHANISM_LINE_STYLES.get(mech, "-"),
                marker=MECHANISM_MARKERS.get(mech, "o"),
                linewidth=1.7, markersize=3.8,
            )
        ax.set_title(title, pad=5)
        _format_axes(ax, ylim01=True)
    _finish_panel_figure(
        fig,
        xlabel=r"Privacy budget $\varepsilon$",
        ylabel="User-level plan-recovery score",
        bottom=0.34,
        top=0.88,
        left=0.085,
        wspace=0.20,
    )
    _add_bottom_legends(fig, dataset_keys=d["dataset"].dropna().unique(), y_dataset=0.085, y_mechanism=0.018)
    _savefig(fig, out)
    return True



def plot_record_probe(probe_df: pd.DataFrame, out: Path, *, K: int, L: int) -> bool:
    """Plot record-level probe admission and summary mention rates."""
    d0 = _keep_probe_rows(probe_df)
    rows = []
    for spec in dataset_specs(Path("outputs")):
        d = d0.copy()
        if d.empty:
            continue
        d = d[d.get("dataset", pd.Series(dtype=str)).eq(spec.key)].copy()
        if "protected_unit" in d.columns:
            d = d[d["protected_unit"].eq("record")].copy()
        d = _select_nearest_numeric(d, "K", K)
        d = _select_nearest_numeric(d, "L", L)
        d = _select_nearest_numeric(d, "m", spec.preferred_m + (1 if spec.key in {"amazon", "yelp"} else 0))
        if not d.empty:
            rows.append(d)
    if not rows:
        return False
    d = pd.concat(rows, ignore_index=True)
    metrics = [("probe_plan_plan_admission_rate", r"$\mathsf{Adm}_{\mathrm{plan}}$"), ("probe_summary_mention_rate", r"$\mathsf{Mention}_{\mathrm{text}}$")]
    if not all(_metric_present(d, m) for m, _ in metrics):
        return False

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), sharex=True, sharey=True)
    chosen_eps_by_metric: list[float] = []
    for ax, (metric, title) in zip(axes, metrics):
        dd = d.copy()
        dd["probe_support"] = pd.to_numeric(dd["probe_support"], errors="coerce")
        g = _aggregate_metric(dd, ["dataset", "mechanism", "epsilon", "probe_support"], metric)
        if "epsilon" in g.columns and g["epsilon"].notna().any():
            eps_vals = sorted(set(float(x) for x in g["epsilon"].dropna()))
            chosen_eps = min(eps_vals, key=lambda x: abs(x - 1.0))
            chosen_eps_by_metric.append(chosen_eps)
            g = g[pd.to_numeric(g["epsilon"], errors="coerce").eq(chosen_eps)].copy()
        for (ds, mech), ss in sorted(
            g.groupby(["dataset", "mechanism"], dropna=False),
            key=lambda item: (str(item[0][0]), MECHANISM_ORDER.get(str(item[0][1]), 99)),
        ):
            mech = canonical_mechanism(mech)
            ss = ss.sort_values("probe_support")
            ax.plot(
                ss["probe_support"], ss[metric],
                color=DATASET_COLORS.get(str(ds), "0.25"),
                linestyle=MECHANISM_LINE_STYLES.get(mech, "-"),
                marker=MECHANISM_MARKERS.get(mech, "o"),
                linewidth=1.7, markersize=3.8,
            )
        ax.set_title(title, pad=5)
        _format_axes(ax, ylim01=True)
    _finish_panel_figure(
        fig,
        xlabel=r"Injected probe support $s$",
        ylabel="Rate",
        bottom=0.34,
        top=0.88,
        left=0.085,
        wspace=0.20,
    )
    _add_bottom_legends(fig, dataset_keys=d["dataset"].dropna().unique(), y_dataset=0.085, y_mechanism=0.018)
    _savefig(fig, out)
    return True



def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the generated-figure manifest as CSV."""
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_latex_fragment(figures: list[dict[str, Any]], path: Path) -> None:
    """Write LaTeX figure environments for generated assets."""
    ensure_dir(path.parent)
    lines: list[str] = []
    for fig in figures:
        width = fig.get("width", "0.92\\linewidth")
        lines.extend([
            r"\begin{figure}[t]",
            r"    \centering",
            rf"    \includegraphics[width={width}]{{{fig['latex_path']}}}",
            rf"    \caption{{{fig['caption']}}}",
            rf"    \label{{{fig['label']}}}",
            r"\end{figure}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def build_main_figures(
    reports_root: Path,
    output_dir: Path,
    *,
    K: int = 3,
    L: int = 10,
    user_B: float = 1.0,
) -> None:
    """Build the selected cross-dataset figure bundle from saved reports."""
    ordinary, probe = load_all_results(reports_root)
    fig_dir = ensure_dir(output_dir / "figures")
    latex_dir = ensure_dir(output_dir / "latex")
    manifest: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []

    specs = [
        {
            "name": "fig_main_record_plan_utility_vs_epsilon.png",
            "builder": lambda p: plot_record_plan_utility(ordinary, p, K=K, L=L),
            "caption": (r"Record-level plan utility at the main operating point. "
                        r"The figure reports $\mathsf{Overlap}@L$ and $\mathsf{MassRetained}$ as the privacy budget $\varepsilon$ varies, "
                        r"with assignment sparsity $K=3$ and released plan size $L=10$. Curves compare the release mechanisms; colors indicate the dataset."),
            "label": "fig:main-record-plan-utility",
        },
        {
            "name": "fig_main_record_support_loss_vs_epsilon.png",
            "builder": lambda p: plot_record_support_loss(ordinary, p, K=K, L=L),
            "caption": (r"Record-level support-mass loss at the main operating point. "
                        r"The figure reports $\mathsf{Loss}_L$ as the privacy budget $\varepsilon$ varies, with $K=3$ and $L=10$. "
                        r"Lower values indicate less non-private semantic mass lost by the released atom set. The vertical axis uses a symmetric-log scale."),
            "label": "fig:main-record-support-loss",
        },
        {
            "name": "fig_main_user_plan_utility_vs_epsilon.png",
            "builder": lambda p: plot_user_plan_utility(ordinary, p, K=K, L=L, B=user_B),
            "caption": (r"User-level plan utility for Amazon and Yelp at the main operating point. "
                        r"The figure reports $\mathsf{Overlap}@L$ and $\mathsf{MassRetained}$ as $\varepsilon$ varies, with $K=3$, $L=10$, and user contribution bound $B=1$."),
            "label": "fig:main-user-plan-utility",
        },
        {
            "name": "fig_main_controlled_probe_record.png",
            "builder": lambda p: plot_record_probe(probe, p, K=K, L=L),
            "caption": (r"Controlled probe-atom behavior in the record-level setting. "
                        r"The figure reports plan-admission rate $\mathsf{Adm}_{\mathrm{plan}}$ and summary mention rate $\mathsf{Mention}_{\mathrm{text}}$ as the injected probe support $s$ varies, with $K=3$ and $L=10$."),
            "label": "fig:main-controlled-probe-record",
        },
    ]

    for item in specs:
        path = fig_dir / item["name"]
        ok = item["builder"](path)
        manifest.append({"figure": item["name"], "written": bool(ok), "label": item["label"], "caption": item["caption"]})
        if ok:
            figures.append({
                "latex_path": f"figures/{Path(item['name']).with_suffix('.pdf').name}",
                "caption": item["caption"],
                "label": item["label"],
            })

    write_manifest(manifest, output_dir / "main_paper_figures_manifest.csv")
    write_latex_fragment(figures, latex_dir / "main_paper_figures.tex")
    readme = [
        "# Main-paper figure bundle",
        "",
        "This folder contains compact figures derived from the saved CFPB, Amazon, and Yelp reports.",
        "No DP mechanisms, atom construction, decoding, or OpenAI calls are rerun.",
        "",
        "Selection policy:",
        f"- assignment sparsity K = {K}",
        f"- released plan size L = {L}",
        "- preferred atom dictionary sizes: CFPB m=200, Amazon m=46, Yelp m=100",
        f"- user-level figure uses B = {user_B}",
        "- ordinary utility figures exclude controlled-probe rows",
        "- controlled-probe figure uses record-level probe runs and the epsilon nearest to 1.0",
        "",
        "Files:",
    ]
    for fig in figures:
        readme.append(f"- `figures/{Path(fig['latex_path']).name}`: {fig['label']}")
    (output_dir / "README_MAIN_PAPER_FIGURES.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'README_MAIN_PAPER_FIGURES.md'}")


def main() -> None:
    """Build the selected cross-dataset figures from the command line."""
    p = argparse.ArgumentParser(description="Build compact main-paper figures from saved DP-SPIN report CSV files.")
    p.add_argument("--reports-root", default="outputs", help="Repository output root containing cfpb/, amazon_All_Beauty/, and yelp_restaurants/.")
    p.add_argument("--output-dir", default="outputs/main_paper_figures", help="Output directory for the compact figure bundle.")
    p.add_argument("--K", type=int, default=3, help="Assignment sparsity to select, with nearest-value fallback.")
    p.add_argument("--L", type=int, default=10, help="Released plan size to select, with nearest-value fallback.")
    p.add_argument("--user-B", type=float, default=1.0, help="User contribution bound for the user-level representative figure.")
    args = p.parse_args()
    build_main_figures(Path(args.reports_root), Path(args.output_dir), K=args.K, L=args.L, user_B=args.user_B)


if __name__ == "__main__":
    main()
