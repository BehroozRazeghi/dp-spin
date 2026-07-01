"""Generate CFPB publication figures from saved DP-SPIN artifacts.

The module reads completed-run metrics and configuration files. 
It does not recompute semantic assignments, rerun a DP mechanism, or decode summaries.
Figures are organized so that each panel varies a limited set of experimental parameters while keeping the remaining conditions explicit.

Notation follows the paper: ``K`` is assignment sparsity, ``L`` is released plan size, and ``private_topL`` denotes sparse private atom admission followed by
noisy selected-value release. 
The deprecated saved key ``private_topk`` is normalized only as a mechanism name; it is not interpreted as the parameter
``K``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


LEGACY_PROBE_FIGURE_NAMES = {
    "fig6_probe_admission_vs_support.png",
    "fig7_probe_non_private_rank_vs_support.png",
    "fig6_cfpb_probe_admission_vs_support.png",
    "fig7_cfpb_probe_non_private_rank_vs_support.png",
    "probe_admission.png",
    "probe_admission_unused.png",
}


def remove_legacy_probe_figures(*directories: Path) -> None:
    """Delete superseded controlled-probe figure files before writing the current set."""
    for directory in directories:
        for name in LEGACY_PROBE_FIGURE_NAMES:
            path = directory / name
            if path.exists():
                path.unlink()


PAPER_RC = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

MECHANISM_ORDER = {"gaussian": 0, "laplace": 1, "private_topL": 2}
MECHANISM_LABELS = {
    "gaussian": r"Gaussian sketch + Top-$L$",
    "laplace": r"Laplace sketch + Top-$L$",
    "private_topL": r"Private Top-$L$ admission + Gaussian values",
}
MECHANISM_STYLES = {"gaussian": "-", "laplace": "--", "private_topL": "-."}
MECHANISM_MARKERS = {"gaussian": "o", "laplace": "s", "private_topL": "^"}
CONDITION_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def apply_style() -> None:
    """Apply the shared matplotlib settings used by CFPB figures."""
    plt.rcParams.update(PAPER_RC)


def canonical_mechanism(value: object) -> str:
    """Normalize mechanism keys without touching the separate parameter K."""
    s = str(value).strip()
    aliases = {
        "private_topk": "private_topL",
        "private_topK": "private_topL",
        "private_top_l": "private_topL",
        "private_top-l": "private_topL",
        "sparse_private_topL": "private_topL",
    }
    return aliases.get(s, s)


def mechanism_label(value: object) -> str:
    """Return the display label for a saved mechanism key."""
    return MECHANISM_LABELS.get(canonical_mechanism(value), str(value))


def fmt_num(value: object) -> str:
    """Format a configuration value for plot annotations and legends."""
    try:
        f = float(value)
        return str(int(f)) if f.is_integer() else f"{f:g}"
    except Exception:
        return str(value)


def fixed_note_label(col: str) -> str:
    """Return the mathematical label used for a configuration column."""
    labels = {
        "epsilon": r"\varepsilon",
        "m": "m",
        "K": "K",
        "L": "L",
        "n_records_effective": r"n_{\mathrm{eff}}",
        "n_records": "n",
        "probe_support": "s",
    }
    return labels.get(col, col.replace("_", r"\_"))


def has_varied(df: pd.DataFrame, col: str) -> bool:
    """Return whether a dataframe column contains more than one observed value."""
    return col in df.columns and df[col].dropna().nunique() > 1


def fixed_note(df: pd.DataFrame, cols: Iterable[str]) -> str:
    """Describe the observed values of selected fixed-condition columns."""
    notes: list[str] = []
    for col in cols:
        if col in df.columns and df[col].notna().any():
            vals = sorted(pd.unique(df[col].dropna()), key=lambda x: float(x) if isinstance(x, (int, float, np.number)) else str(x))
            vals_s = [fmt_num(v) for v in vals]
            if len(vals_s) == 1:
                notes.append(fr"${fixed_note_label(col)}={vals_s[0]}$")
            elif len(vals_s) <= 5:
                notes.append(fr"${fixed_note_label(col)}\in\{{{','.join(vals_s)}\}}$")
            else:
                notes.append(fr"multiple ${fixed_note_label(col)}$")
    return "; ".join(notes)


def add_grid(ax: plt.Axes) -> None:
    """Add the shared major-grid formatting to an axis."""
    ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5, alpha=0.35)


def ensure_dir(path: Path) -> None:
    """Create a directory and any missing parents."""
    path.mkdir(parents=True, exist_ok=True)


def remove_legacy_all_curve_plots(plot_dir: Path) -> None:
    """Remove legacy all-curves CFPB plots that are superseded by organized grids."""
    for name in [
        "utility_vs_epsilon.png",
        "probe_admission.png",
        "probe_admission_unused.png",
    ]:
        path = plot_dir / name
        if path.exists():
            path.unlink()
            print(f"Removed legacy all-curves plot {path}")


def savefig(fig: plt.Figure, path: Path) -> None:
    """Save a figure with the configured resolution and close it."""
    ensure_dir(path.parent)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def read_metrics(path: Path) -> pd.DataFrame:
    """Read a metrics CSV and normalize saved mechanism aliases."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "mechanism" in df.columns:
        df["mechanism"] = df["mechanism"].map(canonical_mechanism)
    return df


def select_fixed(df: pd.DataFrame, requested: dict[str, float | int | str]) -> pd.DataFrame:
    """Select requested fixed values, falling back to nearest observed values."""
    out = df.copy()
    for col, target in requested.items():
        if col not in out.columns or out[col].dropna().empty:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        if isinstance(target, str):
            cand = out[out[col].astype(str) == target]
        elif vals.notna().any():
            available = sorted(set(float(v) for v in vals.dropna()))
            chosen = min(available, key=lambda v: abs(v - float(target)))
            cand = out[vals == chosen]
        else:
            cand = out[out[col] == target]
        if not cand.empty:
            out = cand.copy()
    return out


def mechanism_handles() -> list[Line2D]:
    """Build legend handles for the three plan-release mechanisms."""
    return [
        Line2D([0], [0], color="0.2", linestyle=MECHANISM_STYLES[m], marker=MECHANISM_MARKERS[m], label=MECHANISM_LABELS[m])
        for m in ["gaussian", "laplace", "private_topL"]
    ]


def value_color_map(values: Iterable[object]) -> dict[object, str]:
    """Assign deterministic plotting colors to observed condition values."""
    ordered = sorted(set(values), key=lambda x: float(x) if isinstance(x, (int, float, np.number)) else str(x))
    return {v: CONDITION_COLORS[i % len(CONDITION_COLORS)] for i, v in enumerate(ordered)}


def value_handles(values: Iterable[object], cmap: dict[object, str], prefix: str) -> list[Line2D]:
    """Build legend handles for condition values and their assigned colors."""
    ordered = sorted(set(values), key=lambda x: float(x) if isinstance(x, (int, float, np.number)) else str(x))
    return [Line2D([0], [0], color=cmap[v], linestyle="-", linewidth=2, label=fr"${prefix}={fmt_num(v)}$") for v in ordered]


# ---------------------------------------------------------------------------
# Diagnostics from saved run directories
# ---------------------------------------------------------------------------


def write_topL_margin_diagnostics(runs_dir: Path, output_csv: Path) -> pd.DataFrame:
    """Compute and save the non-private mass margin at the top-``L`` boundary."""
    rows: list[dict[str, Any]] = []
    for run in sorted(runs_dir.glob("*")):
        c_path = run / "clean_sketch.npy"
        cfg_path = run / "config.json"
        if not (c_path.exists() and cfg_path.exists()):
            continue
        try:
            cfg = json.load(open(cfg_path, encoding="utf-8"))
            if canonical_mechanism(cfg.get("mechanism")) != "gaussian":
                continue
            eps = float(cfg.get("epsilon", math.nan))
            if abs(eps - 0.5) > 1e-9:
                continue
            if int(cfg.get("seed", -1)) != 0:
                continue
            C = np.load(c_path)
            L = int(cfg.get("L", 10))
            vals = np.sort(C)[::-1]
            if len(vals) <= L:
                continue
            rows.append({
                "run": run.name,
                "experiment_name": cfg.get("experiment_name"),
                "m": cfg.get("m"),
                "K": cfg.get("K"),
                "L": L,
                "epsilon": eps,
                "mechanism": canonical_mechanism(cfg.get("mechanism")),
                "C_L": float(vals[L - 1]),
                "C_L_plus_1": float(vals[L]),
                "margin": float(vals[L - 1] - vals[L]),
                "top1": float(vals[0]),
                "top5": float(vals[min(4, len(vals) - 1)]),
            })
        except Exception as exc:
            rows.append({"run": run.name, "error": str(exc)})
    df = pd.DataFrame(rows)
    ensure_dir(output_csv.parent)
    if not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"Wrote {output_csv}")
    return df


def infer_support_from_name(name: str) -> int | None:
    """Extract controlled-probe support from a run-directory name."""
    m = re.search(r"support(\d+)", name)
    return int(m.group(1)) if m else None


def write_probe_rank_diagnostics(runs_dir: Path, output_csv: Path) -> pd.DataFrame:
    """Compute and save the non-private rank and mass of each configured probe atom."""
    rows: list[dict[str, Any]] = []
    for run in sorted(runs_dir.glob("*")):
        c_path = run / "clean_sketch.npy"
        cfg_path = run / "config.json"
        if not (c_path.exists() and cfg_path.exists()):
            continue
        try:
            cfg = json.load(open(cfg_path, encoding="utf-8"))
            raw_ids = cfg.get("probe_atom_ids") or ([] if cfg.get("probe_atom_id") is None else [cfg.get("probe_atom_id")])
            probe_ids = [int(x) for x in raw_ids]
            if not probe_ids:
                continue
            C = np.load(c_path)
            j = int(probe_ids[0])
            order = (-C).argsort().tolist()
            rows.append({
                "run": run.name,
                "support": cfg.get("probe_support", infer_support_from_name(run.name)),
                "epsilon": cfg.get("epsilon"),
                "seed": cfg.get("seed"),
                "mechanism": canonical_mechanism(cfg.get("mechanism")),
                "m": cfg.get("m"),
                "K": cfg.get("K"),
                "L": cfg.get("L"),
                "probe_id": j,
                "probe_non_private_mass": float(C[j]) if 0 <= j < len(C) else math.nan,
                "probe_non_private_rank": order.index(j) + 1 if j in order else math.nan,
            })
        except Exception as exc:
            rows.append({"run": run.name, "error": str(exc)})
    df = pd.DataFrame(rows)
    ensure_dir(output_csv.parent)
    if not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"Wrote {output_csv}")
    return df


# ---------------------------------------------------------------------------
# Focused figures
# ---------------------------------------------------------------------------


def plot_main_utility(df: pd.DataFrame, path: Path) -> None:
    """Plot record-level overlap and retained-mass utility against privacy budget."""
    required = {"mechanism", "m", "epsilon", "sketch_overlap_at_L", "sketch_mass_retained_fraction"}
    if df.empty or not required.issubset(df.columns):
        return
    d = df.dropna(subset=["epsilon", "m"]).copy()
    g = d.groupby(["mechanism", "m", "epsilon"], dropna=False)[["sketch_overlap_at_L", "sketch_mass_retained_fraction"]].mean().reset_index()
    m_values = sorted(g["m"].dropna().unique(), key=lambda x: float(x))
    cmap = value_color_map(m_values)

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharex=True, sharey=True)
    panels = [
        (axes[0], "sketch_overlap_at_L", r"$\mathsf{Overlap}@L$"),
        (axes[1], "sketch_mass_retained_fraction", "$\\mathsf{MassRetained}$"),
    ]
    for ax, metric, title in panels:
        for (mech, m_val), ss in sorted(g.groupby(["mechanism", "m"], dropna=False), key=lambda item: (MECHANISM_ORDER.get(item[0][0], 99), float(item[0][1]))):
            ss = ss.sort_values("epsilon")
            ax.plot(
                ss["epsilon"], ss[metric],
                color=cmap.get(m_val),
                linestyle=MECHANISM_STYLES.get(mech, "-"),
                marker=MECHANISM_MARKERS.get(mech, "o"),
                linewidth=1.8, markersize=4.2,
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"Privacy budget $\varepsilon$")
        ax.set_ylim(-0.02, 1.05)
        add_grid(ax)
    axes[0].set_ylabel("Plan-recovery score")
    note = fixed_note(d, ["K", "L", "n_records_effective"])
    if note:
        fig.suptitle(note, y=1.02, fontsize=9)
    fig.legend(handles=mechanism_handles(), loc="lower center", bbox_to_anchor=(0.50, -0.07), ncol=3, frameon=False)
    fig.legend(handles=value_handles(m_values, cmap, "m"), loc="lower center", bbox_to_anchor=(0.50, -0.18), ncol=len(m_values), frameon=False)
    savefig(fig, path)


def plot_hard_support_loss(df: pd.DataFrame, path: Path) -> None:
    """Plot support-mass loss for the hard CFPB experiment block."""
    required = {"mechanism", "epsilon", "L", "sketch_support_mass_loss"}
    if df.empty or not required.issubset(df.columns):
        return
    d = df.dropna(subset=["epsilon", "L"]).copy()
    g = d.groupby(["mechanism", "L", "epsilon"], dropna=False)["sketch_support_mass_loss"].mean().reset_index()
    L_values = sorted(g["L"].dropna().unique(), key=lambda x: float(x))
    n = len(L_values)
    apply_style()
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.7), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, L_val in zip(axes, L_values):
        sub = g[g["L"] == L_val]
        for mech, ss in sorted(sub.groupby("mechanism"), key=lambda item: MECHANISM_ORDER.get(item[0], 99)):
            ss = ss.sort_values("epsilon")
            ax.plot(
                ss["epsilon"], ss["sketch_support_mass_loss"],
                color="0.25",
                linestyle=MECHANISM_STYLES.get(mech, "-"),
                marker=MECHANISM_MARKERS.get(mech, "o"),
                linewidth=1.8, markersize=4.2,
            )
        ax.set_title(fr"Released plan size $L={fmt_num(L_val)}$", fontsize=10)
        ax.set_xlabel(r"Privacy budget $\varepsilon$")
        add_grid(ax)
    axes[0].set_ylabel("$\\mathsf{Loss}_L$")
    note = fixed_note(d, ["m", "K", "n_records_effective"])
    if note:
        fig.suptitle(note, y=1.02, fontsize=9)
    fig.legend(handles=mechanism_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    savefig(fig, path)


def plot_dictionary_size_tradeoff(df: pd.DataFrame, path: Path, *, epsilon: float = 0.5, K: int = 3, L: int = 10) -> None:
    """Plot plan utility as a function of atom dictionary size."""
    required = {"mechanism", "m", "sketch_support_mass_loss", "sketch_overlap_at_L", "sketch_mass_retained_fraction"}
    if df.empty or not required.issubset(df.columns):
        return
    d = select_fixed(df, {"epsilon": epsilon, "K": K, "L": L})
    if d.empty:
        return
    metrics = [
        ("sketch_support_mass_loss", "$\\mathsf{Loss}_L$"),
        ("sketch_overlap_at_L", r"$\mathsf{Overlap}@L$"),
        ("sketch_mass_retained_fraction", "$\\mathsf{MassRetained}$"),
    ]
    g = d.groupby(["mechanism", "m"], dropna=False)[[m for m, _ in metrics]].mean().reset_index()
    # For a fair dictionary-size trade-off, keep only dictionary sizes that are
    # observed for every displayed mechanism at this fixed operating point.
    # This prevents, for example, Gaussian/private-Top-L curves continuing to
    # m=500 when the matching Laplace run at the same epsilon/K/L is absent.
    shared_m = _shared_m_values(g) if "m" in g.columns else []
    if shared_m:
        g = g[g["m"].isin(shared_m)].copy()
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6), sharex=True)
    for ax, (metric, title) in zip(axes, metrics):
        for mech, ss in sorted(g.groupby("mechanism"), key=lambda item: MECHANISM_ORDER.get(item[0], 99)):
            ss = ss.sort_values("m")
            ax.plot(
                ss["m"], ss[metric],
                color="0.25",
                linestyle=MECHANISM_STYLES.get(mech, "-"),
                marker=MECHANISM_MARKERS.get(mech, "o"),
                linewidth=1.8, markersize=4.2,
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"Atom dictionary size $m$")
        add_grid(ax)
        if metric != "sketch_support_mass_loss":
            ax.set_ylim(-0.02, 1.05)
    axes[0].set_ylabel("Mean score")
    fig.suptitle(fixed_note(d, ["epsilon", "K", "L", "n_records_effective"]), y=1.02, fontsize=9)
    fig.legend(handles=mechanism_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    savefig(fig, path)


def plot_KL_heatmaps(df: pd.DataFrame, path: Path, *, epsilon: float = 0.5, m: int = 200) -> None:
    """Plot assignment-sparsity and released-plan-size ablation heatmaps."""
    required = {"mechanism", "K", "L", "sketch_support_mass_loss"}
    if df.empty or not required.issubset(df.columns):
        return
    d = select_fixed(df, {"epsilon": epsilon, "m": m})
    if d.empty:
        return
    mechs = [m for m in ["gaussian", "laplace", "private_topL"] if m in set(d["mechanism"].dropna())]
    if not mechs:
        mechs = sorted(d["mechanism"].dropna().unique())
    apply_style()
    fig, axes = plt.subplots(1, len(mechs), figsize=(4.1 * len(mechs), 3.6), sharex=True, sharey=True)
    if len(mechs) == 1:
        axes = [axes]
    vmin = d["sketch_support_mass_loss"].min()
    vmax = d["sketch_support_mass_loss"].max()
    im = None
    for ax, mech in zip(axes, mechs):
        ss = d[d["mechanism"] == mech]
        g = ss.groupby(["K", "L"], dropna=False)["sketch_support_mass_loss"].mean().reset_index()
        pivot = g.pivot(index="K", columns="L", values="sketch_support_mass_loss").sort_index().sort_index(axis=1)
        im = ax.imshow(pivot.values, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(mechanism_label(mech), fontsize=9)
        ax.set_xticks(range(len(pivot.columns)), [fmt_num(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [fmt_num(c) for c in pivot.index])
        ax.set_xlabel(r"Released plan size $L$")
    axes[0].set_ylabel(r"Assignment sparsity $K$")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.88, label="$\\mathsf{Loss}_L$")
    fig.suptitle(fixed_note(d, ["epsilon", "m", "n_records_effective"]), y=1.03, fontsize=9)
    savefig(fig, path)


def filter_probe_report_m(df: pd.DataFrame, preferred_m: int = 201) -> pd.DataFrame:
    """Restrict saved CFPB probe reports to one dictionary size when requested.
    
    If the preferred value is unavailable, the function selects the largest observed
    size so that a single condition is displayed rather than silently averaging
    across dictionary sizes.
    """
    if df.empty or "m" not in df.columns or df["m"].dropna().empty:
        return df
    vals = pd.to_numeric(df["m"], errors="coerce")
    if (vals == float(preferred_m)).any():
        return df[vals == float(preferred_m)].copy()
    available = sorted(set(float(v) for v in vals.dropna()))
    if not available:
        return df
    fallback = available[-1]
    return df[vals == fallback].copy()


def plot_probe_admission(df: pd.DataFrame, path: Path) -> None:
    """Plot controlled probe-atom admission rate against injected support."""
    metric = "probe_plan_plan_admission_rate" if "probe_plan_plan_admission_rate" in df.columns else "probe_summary_mention_rate" if "probe_summary_mention_rate" in df.columns else None
    required = {"mechanism", "epsilon", "probe_support"}
    if df.empty or metric is None or not required.issubset(df.columns):
        return
    d = df.dropna(subset=["probe_support", "epsilon"]).copy()
    d = filter_probe_report_m(d)
    facet_col = "m" if has_varied(d, "m") else None
    facets = sorted(d[facet_col].dropna().unique(), key=lambda x: float(x)) if facet_col else [None]
    eps_values = sorted(d["epsilon"].dropna().unique(), key=lambda x: float(x))
    eps_cmap = value_color_map(eps_values)
    apply_style()
    fig, axes = plt.subplots(1, len(facets), figsize=(6.0 * len(facets), 3.9), sharex=True, sharey=True)
    if len(facets) == 1:
        axes = [axes]
    for ax, facet in zip(axes, facets):
        sub = d if facet is None else d[d[facet_col] == facet]
        g = sub.groupby(["mechanism", "epsilon", "probe_support"], dropna=False)[metric].mean().reset_index()
        for (mech, eps), ss in sorted(g.groupby(["mechanism", "epsilon"], dropna=False), key=lambda item: (MECHANISM_ORDER.get(item[0][0], 99), float(item[0][1]))):
            ss = ss.sort_values("probe_support")
            ax.plot(
                ss["probe_support"], ss[metric],
                color=eps_cmap.get(eps),
                linestyle=MECHANISM_STYLES.get(mech, "-"),
                marker=MECHANISM_MARKERS.get(mech, "o"),
                linewidth=1.6, markersize=3.8,
            )
        ax.set_xlabel("Probe support $s$")
        ax.set_ylim(-0.02, 1.02)
        if facet is not None:
            ax.set_title(fr"Probe-atom dictionary $m={fmt_num(facet)}$", fontsize=10)
        add_grid(ax)
    axes[0].set_ylabel("$\\mathsf{Adm}_{\\mathrm{plan}}$" if "plan" in metric else "$\\mathsf{Mention}_{\\mathrm{text}}$")
    fig.legend(handles=mechanism_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False)
    fig.legend(handles=value_handles(eps_values, eps_cmap, r"\varepsilon"), loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=min(len(eps_values), 5), frameon=False)
    savefig(fig, path)


def plot_probe_rank(rank_df: pd.DataFrame, path: Path) -> None:
    """Plot the non-private probe-atom rank against injected support."""
    required = {"support", "probe_non_private_rank", "m", "L"}
    if rank_df.empty or not required.issubset(rank_df.columns):
        return
    d = rank_df.dropna(subset=["support", "probe_non_private_rank"]).copy()
    if d.empty:
        return
    facet_col = "m" if has_varied(d, "m") else None
    facets = sorted(d[facet_col].dropna().unique(), key=lambda x: float(x)) if facet_col else [None]
    apply_style()
    fig, axes = plt.subplots(1, len(facets), figsize=(5.3 * len(facets), 3.8), sharex=True, sharey=True)
    if len(facets) == 1:
        axes = [axes]
    for ax, facet in zip(axes, facets):
        sub = d if facet is None else d[d[facet_col] == facet]
        g = sub.groupby("support", dropna=False)["probe_non_private_rank"].mean().reset_index().sort_values("support")
        ax.plot(g["support"], g["probe_non_private_rank"], marker="o", linewidth=1.8, color="0.25")
        L_values = sorted(sub["L"].dropna().unique(), key=lambda x: float(x))
        for L_val in L_values:
            ax.axhline(float(L_val), linestyle="--", linewidth=1, color="0.45")
        ax.invert_yaxis()
        ax.set_xlabel("Probe support $s$")
        if facet is not None:
            ax.set_title(fr"Probe-atom dictionary $m={fmt_num(facet)}$", fontsize=10)
        add_grid(ax)
    axes[0].set_ylabel("$\\mathsf{Rank}_{\\mathrm{NP}}$")
    note = fixed_note(d, ["K", "L"])
    if note:
        fig.suptitle(note, y=1.02, fontsize=9)
    savefig(fig, path)


def copy_if_exists(src: Path, dst: Path) -> None:
    """Copy an artifact when the source file exists."""
    if src.exists():
        ensure_dir(dst.parent)
        dst.write_bytes(src.read_bytes())
        print(f"Copied {src} -> {dst}")


# ---------------------------------------------------------------------------
# Block-level organized plot bundles
# ---------------------------------------------------------------------------


def write_coverage_table(df: pd.DataFrame, path: Path, *, fixed: dict[str, float | int | str] | None = None) -> pd.DataFrame:
    """Write observed configuration coverage so missing curve endpoints are explicit."""
    if df.empty:
        return pd.DataFrame()
    d = select_fixed(df, fixed or {}) if fixed else df.copy()
    cols = [c for c in ["mechanism", "m", "K", "L", "epsilon", "n_records_effective", "probe_support"] if c in d.columns]
    metric_cols = [
        c for c in [
            "sketch_overlap_at_L",
            "sketch_mass_retained_fraction",
            "sketch_support_mass_loss",
            "probe_plan_plan_admission_rate",
            "probe_summary_mention_rate",
        ] if c in d.columns
    ]
    if not cols:
        return pd.DataFrame()
    rows = []
    for keys, ss in d.groupby(cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(cols, keys))
        row["n_rows"] = len(ss)
        for metric in metric_cols:
            row[f"{metric}_nonmissing"] = int(ss[metric].notna().sum())
            row[f"{metric}_mean"] = float(ss[metric].mean()) if ss[metric].notna().any() else math.nan
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(cols)
    ensure_dir(path.parent)
    out.to_csv(path, index=False)
    print(f"Wrote {path}")
    return out


def _metric_specs(df: pd.DataFrame) -> list[tuple[str, str, bool]]:
    specs = [
        ("sketch_overlap_at_L", r"$\mathsf{Overlap}@L$", True),
        ("sketch_mass_retained_fraction", "$\\mathsf{MassRetained}$", True),
        ("sketch_support_mass_loss", "$\\mathsf{Loss}_L$", False),
        ("probe_plan_plan_admission_rate", "$\\mathsf{Adm}_{\\mathrm{plan}}$", True),
        ("probe_summary_mention_rate", "$\\mathsf{Mention}_{\\mathrm{text}}$", True),
    ]
    return [spec for spec in specs if spec[0] in df.columns]


def _ordered_values(df: pd.DataFrame, col: str) -> list[Any]:
    if col not in df.columns:
        return []
    vals = [v for v in pd.unique(df[col].dropna())]
    return sorted(vals, key=lambda x: float(x) if isinstance(x, (int, float, np.number)) else str(x))


def _shared_m_values(g: pd.DataFrame) -> list[Any]:
    """Dictionary sizes present for every displayed mechanism at a fixed setting."""
    if g.empty or "m" not in g.columns or "mechanism" not in g.columns:
        return []
    m_sets = []
    for _, ss in g.groupby("mechanism", dropna=False):
        vals = set(ss["m"].dropna().tolist())
        if vals:
            m_sets.append(vals)
    if not m_sets:
        return []
    shared = set.intersection(*m_sets)
    return sorted(shared, key=lambda x: float(x) if isinstance(x, (int, float, np.number)) else str(x))


def plot_metric_vs_epsilon_faceted(
    df: pd.DataFrame,
    path: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
    facet_col: str | None = None,
    color_col: str | None = None,
    fixed: dict[str, float | int | str] | None = None,
    y_is_fraction: bool = False,
) -> None:
    """Plot one metric against privacy budget in a faceted condition grid."""
    required = {"mechanism", "epsilon", metric}
    if df.empty or not required.issubset(df.columns):
        return
    d = select_fixed(df, fixed or {}) if fixed else df.copy()
    d = d.dropna(subset=["epsilon", metric]).copy()
    if d.empty:
        return
    if facet_col and (facet_col not in d.columns or d[facet_col].dropna().nunique() <= 1):
        facet_col = None
    if color_col and (color_col not in d.columns or d[color_col].dropna().nunique() <= 1):
        color_col = None
    facets = _ordered_values(d, facet_col) if facet_col else [None]
    color_values = _ordered_values(d, color_col) if color_col else ["all"]
    cmap = value_color_map(color_values)

    apply_style()
    fig, axes = plt.subplots(1, len(facets), figsize=(4.7 * len(facets), 3.8), sharex=True, sharey=True)
    if len(facets) == 1:
        axes = [axes]
    for ax, facet in zip(axes, facets):
        sub = d if facet is None else d[d[facet_col] == facet]
        group_cols = ["mechanism", "epsilon"] + ([color_col] if color_col else [])
        g = sub.groupby(group_cols, dropna=False)[metric].mean().reset_index()
        if color_col:
            groups = g.groupby(["mechanism", color_col], dropna=False)
            sort_key = lambda item: (MECHANISM_ORDER.get(item[0][0], 99), float(item[0][1]) if isinstance(item[0][1], (int, float, np.number)) else str(item[0][1]))
        else:
            groups = g.groupby("mechanism", dropna=False)
            sort_key = lambda item: MECHANISM_ORDER.get(item[0], 99)
        for key, ss in sorted(groups, key=sort_key):
            mech = key[0] if isinstance(key, tuple) else key
            color_value = key[1] if color_col and isinstance(key, tuple) else "all"
            ss = ss.sort_values("epsilon")
            ax.plot(
                ss["epsilon"], ss[metric],
                color=cmap.get(color_value, "0.25"),
                linestyle=MECHANISM_STYLES.get(mech, "-"),
                marker=MECHANISM_MARKERS.get(mech, "o"),
                linewidth=1.7, markersize=4.0,
            )
        panel_title = title if facet is None else fr"{title}; ${facet_col}={fmt_num(facet)}$"
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlabel(r"Privacy budget $\varepsilon$")
        if y_is_fraction:
            ax.set_ylim(-0.02, 1.05)
        add_grid(ax)
    axes[0].set_ylabel(ylabel)
    fig.legend(handles=mechanism_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    if color_col:
        fig.legend(handles=value_handles(color_values, cmap, color_col), loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=min(len(color_values), 6), frameon=False)
    note_cols = [c for c in ["m", "K", "L", "n_records_effective"] if c in d.columns and c not in {facet_col, color_col}]
    note = fixed_note(d, note_cols)
    if note:
        fig.suptitle(note, y=1.02, fontsize=9)
    savefig(fig, path)


def plot_metric_grid_by_KL(
    df: pd.DataFrame,
    path: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
    fixed: dict[str, float | int | str] | None = None,
    color_col: str = "m",
    y_is_fraction: bool = False,
) -> None:
    """All ablation curves, grouped as a K-by-L panel grid."""
    required = {"mechanism", "epsilon", "K", "L", metric}
    if df.empty or not required.issubset(df.columns):
        return
    d = select_fixed(df, fixed or {}) if fixed else df.copy()
    d = d.dropna(subset=["epsilon", "K", "L", metric]).copy()
    if d.empty:
        return
    K_vals = _ordered_values(d, "K")
    L_vals = _ordered_values(d, "L")
    color_values = _ordered_values(d, color_col) if color_col in d.columns and d[color_col].dropna().nunique() > 1 else ["all"]
    use_color_col = color_col if color_values != ["all"] else None
    cmap = value_color_map(color_values)

    apply_style()
    fig, axes = plt.subplots(len(K_vals), len(L_vals), figsize=(4.3 * len(L_vals), 3.15 * len(K_vals)), sharex=True, sharey=True)
    axes_arr = np.array(axes, dtype=object).reshape(len(K_vals), len(L_vals))
    for i, K_val in enumerate(K_vals):
        for j, L_val in enumerate(L_vals):
            ax = axes_arr[i, j]
            sub = d[(d["K"] == K_val) & (d["L"] == L_val)]
            if sub.empty:
                ax.axis("off")
                continue
            group_cols = ["mechanism", "epsilon"] + ([use_color_col] if use_color_col else [])
            g = sub.groupby(group_cols, dropna=False)[metric].mean().reset_index()
            if use_color_col:
                groups = g.groupby(["mechanism", use_color_col], dropna=False)
                sort_key = lambda item: (MECHANISM_ORDER.get(item[0][0], 99), float(item[0][1]) if isinstance(item[0][1], (int, float, np.number)) else str(item[0][1]))
            else:
                groups = g.groupby("mechanism", dropna=False)
                sort_key = lambda item: MECHANISM_ORDER.get(item[0], 99)
            for key, ss in sorted(groups, key=sort_key):
                mech = key[0] if isinstance(key, tuple) else key
                color_value = key[1] if use_color_col and isinstance(key, tuple) else "all"
                ss = ss.sort_values("epsilon")
                ax.plot(
                    ss["epsilon"], ss[metric],
                    color=cmap.get(color_value, "0.25"),
                    linestyle=MECHANISM_STYLES.get(mech, "-"),
                    marker=MECHANISM_MARKERS.get(mech, "o"),
                    linewidth=1.4, markersize=3.5,
                )
            ax.set_title(fr"$K={fmt_num(K_val)},\ L={fmt_num(L_val)}$", fontsize=9)
            if i == len(K_vals) - 1:
                ax.set_xlabel(r"Privacy budget $\varepsilon$")
            if j == 0:
                ax.set_ylabel(ylabel, labelpad=8)
            if y_is_fraction:
                ax.set_ylim(-0.02, 1.05)
            add_grid(ax)
    fig.suptitle(title + (("; " + fixed_note(d, ["n_records_effective"])) if fixed_note(d, ["n_records_effective"]) else ""), y=1.01, fontsize=10)
    fig.legend(handles=mechanism_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False)
    if use_color_col:
        fig.legend(handles=value_handles(color_values, cmap, use_color_col), loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=min(len(color_values), 6), frameon=False)
    savefig(fig, path)


def plot_metric_vs_m_grid_by_KL(
    df: pd.DataFrame,
    path: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
    epsilon: float,
    y_is_fraction: bool = False,
) -> None:
    """Dictionary-size ablation, grouped by K and L at one epsilon."""
    required = {"mechanism", "epsilon", "m", "K", "L", metric}
    if df.empty or not required.issubset(df.columns):
        return
    d = select_fixed(df, {"epsilon": epsilon})
    d = d.dropna(subset=["m", "K", "L", metric]).copy()
    if d.empty:
        return
    K_vals = _ordered_values(d, "K")
    L_vals = _ordered_values(d, "L")
    apply_style()
    fig, axes = plt.subplots(len(K_vals), len(L_vals), figsize=(4.3 * len(L_vals), 3.15 * len(K_vals)), sharex=True, sharey=True)
    axes_arr = np.array(axes, dtype=object).reshape(len(K_vals), len(L_vals))
    for i, K_val in enumerate(K_vals):
        for j, L_val in enumerate(L_vals):
            ax = axes_arr[i, j]
            sub = d[(d["K"] == K_val) & (d["L"] == L_val)]
            if sub.empty:
                ax.axis("off")
                continue
            g = sub.groupby(["mechanism", "m"], dropna=False)[metric].mean().reset_index()
            shared_m = _shared_m_values(g)
            if shared_m:
                g = g[g["m"].isin(shared_m)]
            for mech, ss in sorted(g.groupby("mechanism", dropna=False), key=lambda item: MECHANISM_ORDER.get(item[0], 99)):
                ss = ss.sort_values("m")
                ax.plot(
                    ss["m"], ss[metric],
                    color="0.25",
                    linestyle=MECHANISM_STYLES.get(mech, "-"),
                    marker=MECHANISM_MARKERS.get(mech, "o"),
                    linewidth=1.5, markersize=3.7,
                )
            ax.set_title(fr"$K={fmt_num(K_val)},\ L={fmt_num(L_val)}$", fontsize=9)
            if i == len(K_vals) - 1:
                ax.set_xlabel(r"Atom dictionary size $m$")
            if j == 0:
                ax.set_ylabel(ylabel, labelpad=8)
            if y_is_fraction:
                ax.set_ylim(-0.02, 1.05)
            add_grid(ax)
    fig.suptitle(title + fr"; $\varepsilon={fmt_num(select_fixed(df, {'epsilon': epsilon})['epsilon'].dropna().iloc[0])}$", y=1.01, fontsize=10)
    fig.legend(handles=mechanism_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False)
    savefig(fig, path)


def plot_probe_support_grid(df: pd.DataFrame, path: Path, *, metric: str, ylabel: str) -> None:
    """Plot probe outcomes against support in a faceted condition grid."""
    required = {"mechanism", "epsilon", "probe_support", metric}
    if df.empty or not required.issubset(df.columns):
        return
    d = df.dropna(subset=["epsilon", "probe_support", metric]).copy()
    d = filter_probe_report_m(d)
    if d.empty:
        return
    facet_col = "m" if has_varied(d, "m") else None
    facets = _ordered_values(d, facet_col) if facet_col else [None]
    eps_values = _ordered_values(d, "epsilon")
    eps_cmap = value_color_map(eps_values)
    apply_style()
    fig, axes = plt.subplots(1, len(facets), figsize=(5.3 * len(facets), 3.7), sharex=True, sharey=True)
    if len(facets) == 1:
        axes = [axes]
    for ax, facet in zip(axes, facets):
        sub = d if facet is None else d[d[facet_col] == facet]
        g = sub.groupby(["mechanism", "epsilon", "probe_support"], dropna=False)[metric].mean().reset_index()
        for (mech, eps), ss in sorted(g.groupby(["mechanism", "epsilon"], dropna=False), key=lambda item: (MECHANISM_ORDER.get(item[0][0], 99), float(item[0][1]))):
            ss = ss.sort_values("probe_support")
            ax.plot(
                ss["probe_support"], ss[metric],
                color=eps_cmap.get(eps),
                linestyle=MECHANISM_STYLES.get(mech, "-"),
                marker=MECHANISM_MARKERS.get(mech, "o"),
                linewidth=1.5, markersize=3.5,
            )
        ax.set_xlabel("Probe support $s$")
        ax.set_ylim(-0.02, 1.05)
        if facet is not None:
            ax.set_title(fr"$m={fmt_num(facet)}$", fontsize=10)
        add_grid(ax)
    axes[0].set_ylabel(ylabel)
    fig.legend(handles=mechanism_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    fig.legend(handles=value_handles(eps_values, eps_cmap, r"\varepsilon"), loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=min(len(eps_values), 6), frameon=False)
    savefig(fig, path)


def write_block_organized_plots(out: Path, *, preferred_epsilon: float = 0.5, preferred_K: int = 3, preferred_L: int = 10, preferred_m: int = 200) -> None:
    """Create organized CFPB plot bundles directly in each block's plots folder.

    No extra `focused/` folder is created.  Files are named by the experimental
    question they answer: epsilon utility, dictionary-size ablation, K/L
    ablation, and controlled probe-atom diagnostics.
    """
    main_df = read_metrics(out / "main" / "metrics_summary.csv")
    hard_df = read_metrics(out / "hard" / "metrics_summary.csv")
    ablation_df = read_metrics(out / "ablation" / "metrics_summary.csv")
    probe_df = read_metrics(out / "probe" / "metrics_summary.csv")

    # Avoid leaving the old, unreadable all-curves figures beside the organized
    # figures.  metrics_summary.csv is still regenerated by src.plot_results.
    for block in ["main", "hard", "ablation", "probe"]:
        remove_legacy_all_curve_plots(out / block / "plots")

    for block in ["main", "hard", "ablation", "probe"]:
        old = out / block / "plots" / "focused"
        if old.exists():
            import shutil
            shutil.rmtree(old)
            print(f"Removed old folder {old}")

    # MAIN: effect of epsilon and dictionary size m under the primary K,L setting.
    main_plots = out / "main" / "plots"
    write_coverage_table(main_df, main_plots / "configuration_coverage.csv")
    for metric, ylabel, is_frac in _metric_specs(main_df):
        if metric.startswith("probe_"):
            continue
        plot_metric_vs_epsilon_faceted(
            main_df,
            main_plots / f"main_{metric}_vs_epsilon_by_m.png",
            metric=metric,
            ylabel=ylabel,
            title="Main CFPB: epsilon utility trade-off",
            facet_col="m",
            color_col=None,
            y_is_fraction=is_frac,
        )
    plot_main_utility(main_df, main_plots / "main_two_panel_utility_vs_epsilon_by_m.png")

    # HARD/STRESS-TEST: preserve all curves, but group them exactly like the ablation
    # plots.  In the default sweep this becomes a one-row K-by-L grid; if future
    # hard sweeps vary K or m, the same plotting code shows those conditions too.
    hard_plots = out / "hard" / "plots"
    write_coverage_table(hard_df, hard_plots / "configuration_coverage.csv")
    for metric, ylabel, is_frac in _metric_specs(hard_df):
        if metric.startswith("probe_"):
            continue
        plot_metric_grid_by_KL(
            hard_df,
            hard_plots / f"hard_{metric}_vs_epsilon_grid_by_K_L_colored_by_m.png",
            metric=metric,
            ylabel=ylabel,
            title=r"CFPB stress test: $\varepsilon$ effect by $K,L$",
            color_col="m",
            y_is_fraction=is_frac,
        )

    # ABLATION: all ablation curves are preserved but grouped by the parameter under study.
    ablation_plots = out / "ablation" / "plots"
    write_coverage_table(ablation_df, ablation_plots / "configuration_coverage.csv")
    write_coverage_table(ablation_df, ablation_plots / "dictionary_size_coverage_at_fixed_operating_point.csv", fixed={"epsilon": preferred_epsilon})
    for metric, ylabel, is_frac in _metric_specs(ablation_df):
        if metric.startswith("probe_"):
            continue
        plot_metric_grid_by_KL(
            ablation_df,
            ablation_plots / f"ablation_{metric}_vs_epsilon_grid_by_K_L_colored_by_m.png",
            metric=metric,
            ylabel=ylabel,
            title=r"CFPB ablation: $\varepsilon$ effect by $K,L$",
            color_col="m",
            y_is_fraction=is_frac,
        )
        plot_metric_vs_m_grid_by_KL(
            ablation_df,
            ablation_plots / f"ablation_{metric}_vs_m_grid_by_K_L_at_epsilon.png",
            metric=metric,
            ylabel=ylabel,
            title=r"CFPB ablation: dictionary-size effect by $K,L$",
            epsilon=preferred_epsilon,
            y_is_fraction=is_frac,
        )
    plot_KL_heatmaps(
        ablation_df,
        ablation_plots / "ablation_support_mass_loss_heatmap_K_by_L_fixed_epsilon_m.png",
        epsilon=preferred_epsilon,
        m=preferred_m,
    )

    # PROBE: injected support is the x-axis; m is a facet when it varies.
    probe_plots = out / "probe" / "plots"
    write_coverage_table(probe_df, probe_plots / "configuration_coverage.csv")
    rank_df = write_probe_rank_diagnostics(out / "probe" / "runs", out / "probe" / "diagnostics" / "probe_rank_diagnostics.csv")
    if "probe_plan_plan_admission_rate" in probe_df.columns:
        plot_probe_support_grid(probe_df, probe_plots / "probe_plan_admission_vs_support_by_m.png", metric="probe_plan_plan_admission_rate", ylabel="$\\mathsf{Adm}_{\\mathrm{plan}}$")
    if "probe_summary_mention_rate" in probe_df.columns:
        plot_probe_support_grid(probe_df, probe_plots / "probe_summary_mention_vs_support_by_m.png", metric="probe_summary_mention_rate", ylabel="$\\mathsf{Mention}_{\\mathrm{text}}$")
    plot_probe_rank(rank_df, probe_plots / "probe_non_private_rank_vs_support_by_m.png")

    readme = """# CFPB plot organization

These plots are regenerated from saved metrics and run artifacts only. They do not rerun DP mechanisms.

## Main block
Primary CFPB utility setting. The plots vary privacy budget epsilon and group by dictionary size m. K is assignment sparsity and L is released plan size.

## Hard / stress-test block
Stress-test setting with smaller protected sample size and lower epsilon values than the main block. The plots use the same K-by-L grid organization as the ablation plots, so all saved hard curves are retained without one oversized legend.

## Ablation block
Parameter-study setting. The `*_vs_epsilon_grid_by_K_L_colored_by_m.png` plots keep all saved curves but organize them into a K-by-L grid. The `*_vs_m_grid_by_K_L_at_epsilon.png` plots study dictionary-size effects at one epsilon. Missing endpoints are not imputed; see `configuration_coverage.csv`.

## Probe block
Controlled probe-atom diagnostics. Admission and mention plots use probe support on the x-axis. The non-private-rank plot shows whether the probe atom becomes salient enough to enter a Top-$L$ release.

Mechanism-name note: old saved key `private_topk` is treated only as an alias for the sparse `private_topL` mechanism in labels. The parameter K remains assignment sparsity and the parameter L remains released plan size.
"""
    for folder in [main_plots, hard_plots, ablation_plots, probe_plots]:
        ensure_dir(folder)
        (folder / "README_PLOTS.md").write_text(readme, encoding="utf-8")
    print("Organized block-level CFPB plots written directly under each */plots folder.")


# Backward-compatible name for old scripts.
def write_block_focused_plots(out: Path, *, preferred_epsilon: float = 0.5, preferred_K: int = 3, preferred_L: int = 10, preferred_m: int = 200) -> None:
    """Generate the focused plot set for each CFPB experiment block."""
    write_block_organized_plots(out, preferred_epsilon=preferred_epsilon, preferred_K=preferred_K, preferred_L=preferred_L, preferred_m=preferred_m)

def generate_cfpb_report_assets(out: Path, *, preferred_epsilon: float = 0.5, preferred_K: int = 3, preferred_L: int = 10, preferred_m: int = 200, write_block_plots: bool = False) -> None:
    """Generate CFPB figures, diagnostics, and copied report assets."""
    main = out / "main"
    hard = out / "hard"
    ablation = out / "ablation"
    probe = out / "probe"
    baselines = out / "baselines"
    membership = out / "membership"
    openai_out = out / "openai"
    paper_figs = out / "paper_figures"
    paper_tables = out / "paper_tables"
    ensure_dir(paper_figs)
    ensure_dir(paper_tables)
    remove_legacy_probe_figures(paper_figs, out / "paper_assets" / "figures")

    # Diagnostics saved beside the relevant experiment family and mirrored into
    # paper_tables for auditability.
    margin = write_topL_margin_diagnostics(main / "runs", main / "diagnostics" / "topL_margin_diagnostics.csv")
    if not margin.empty:
        copy_if_exists(main / "diagnostics" / "topL_margin_diagnostics.csv", paper_tables / "topL_margin_diagnostics.csv")
    rank_df = write_probe_rank_diagnostics(probe / "runs", probe / "diagnostics" / "probe_rank_diagnostics.csv")
    if not rank_df.empty:
        copy_if_exists(probe / "diagnostics" / "probe_rank_diagnostics.csv", paper_tables / "probe_rank_diagnostics.csv")

    main_df = read_metrics(main / "metrics_summary.csv")
    hard_df = read_metrics(hard / "metrics_summary.csv")
    ablation_df = read_metrics(ablation / "metrics_summary.csv")
    probe_df = read_metrics(probe / "metrics_summary.csv")

    plot_main_utility(main_df, paper_figs / "fig2_main_utility_vs_epsilon.png")
    plot_hard_support_loss(hard_df, paper_figs / "fig3_hard_support_mass_loss.png")
    plot_dictionary_size_tradeoff(ablation_df, paper_figs / "fig4_dictionary_size_support_loss.png", epsilon=preferred_epsilon, K=preferred_K, L=preferred_L)
    plot_KL_heatmaps(ablation_df, paper_figs / "fig5_KL_support_loss_heatmap.png", epsilon=preferred_epsilon, m=preferred_m)
    plot_probe_admission(probe_df, paper_figs / "fig6_probe_admission_vs_support.png")
    plot_probe_rank(rank_df, paper_figs / "fig7_probe_non_private_rank_vs_support.png")

    if write_block_plots:
        write_block_organized_plots(out, preferred_epsilon=preferred_epsilon, preferred_K=preferred_K, preferred_L=preferred_L, preferred_m=preferred_m)

    # Preserve CSV/TEX/JSON report artifacts in the staging folder.  The
    # canonical paper_assets folder is populated afterwards by src.paper_assets.
    copies = [
        (main / "reports" / "summary_table.csv", paper_tables / "main_summary_table.csv"),
        (main / "reports" / "summary_table.tex", paper_tables / "main_summary_table.tex"),
        (main / "reports" / "paper_table.tex", paper_tables / "main_paper_table.tex"),
        (hard / "reports" / "summary_table.csv", paper_tables / "hard_summary_table.csv"),
        (hard / "reports" / "summary_table.tex", paper_tables / "hard_summary_table.tex"),
        (ablation / "reports" / "summary_table.csv", paper_tables / "ablation_summary_table.csv"),
        (ablation / "reports" / "summary_table.tex", paper_tables / "ablation_summary_table.tex"),
        (probe / "reports" / "summary_table.csv", paper_tables / "probe_summary_table.csv"),
        (probe / "reports" / "summary_table.tex", paper_tables / "probe_summary_table.tex"),
        (baselines / "probe" / "exact_probe_mention.csv", paper_tables / "baseline_exact_probe_mention.csv"),
        (baselines / "urania_probe" / "exact_probe_mention.csv", paper_tables / "urania_probe_exact_mention.csv"),
        (baselines / "reports" / "baseline_methods.csv", paper_tables / "baseline_methods.csv"),
        (baselines / "reports" / "baseline_methods_table.tex", paper_tables / "baseline_methods_table.tex"),
        (baselines / "cfpb_template" / "urania_cluster_public_keyword_metrics.json", paper_tables / "urania_cluster_public_keyword_metrics.json"),
        (membership / "reports" / "membership_attack.json", paper_tables / "membership_attack.json"),
        (openai_out / "judgments" / "judgments.json", paper_tables / "openai_judgments.json"),
    ]
    for src, dst in copies:
        copy_if_exists(src, dst)

    manifest = [
        "# CFPB report figures",
        "",
        "Generated by `python -m src.cfpb_report_figures` from already saved metrics and run artifacts.",
        "",
        "## Figure logic",
        "- `fig2_main_utility_vs_epsilon.png`: main utility only; varies epsilon, mechanism, and dictionary size m; K and L are shown as fixed-note metadata.",
        "- `fig3_hard_support_mass_loss.png`: CFPB stress-test setting; separate panels fix L so each panel shows only epsilon-vs-loss by mechanism.",
        "- `fig4_dictionary_size_support_loss.png`: dictionary-size trade-off after fixing epsilon, K, and L (nearest available values are used if exact values are absent).",
        "- `fig5_KL_support_loss_heatmap.png`: K/L trade-off heatmaps with epsilon and m fixed.",
        "- `fig6_probe_admission_vs_support.png`: controlled probe-support measure for plan admission (or summary mention if plan admission is absent); it derives m from each run and facets by m when m varies.",
        "- `fig7_probe_non_private_rank_vs_support.png`: non-private probe-atom rank measure; this shows when the probe becomes salient enough to enter a Top-$L$ release.",
        "",
        "Mechanism-name note: deprecated saved key `private_topk` is treated only as an alias for the sparse `private_topL` mechanism in plot labels; the model parameter `K` remains assignment sparsity.",
    ]
    (paper_figs / "README_FIGURES.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"CFPB paper figures written to {paper_figs}")


def main() -> None:
    """Generate CFPB report figures from the command line."""
    parser = argparse.ArgumentParser(description="Regenerate focused CFPB report figures from saved metrics.")
    parser.add_argument("--out", default="outputs/cfpb", help="Root output directory containing main/hard/ablation/probe subfolders.")
    parser.add_argument("--epsilon", type=float, default=0.5, help="Preferred epsilon for focused ablation figures; nearest available value is used.")
    parser.add_argument("--K", type=int, default=3, help="Preferred assignment sparsity K for dictionary-size figure; nearest available value is used.")
    parser.add_argument("--L", type=int, default=10, help="Preferred released plan size L for dictionary-size figure; nearest available value is used.")
    parser.add_argument("--m", type=int, default=200, help="Preferred dictionary size m for K/L heatmaps; nearest available value is used.")
    parser.add_argument("--write-block-plots", action="store_true", help="Also write organized block-level figures directly into main/hard/ablation/probe plots folders.")
    args = parser.parse_args()
    generate_cfpb_report_assets(Path(args.out), preferred_epsilon=args.epsilon, preferred_K=args.K, preferred_L=args.L, preferred_m=args.m, write_block_plots=args.write_block_plots)


if __name__ == "__main__":
    main()
