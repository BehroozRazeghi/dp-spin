"""Collect saved metrics and plot plan-utility and probe diagnostics.

The module reads run-level metrics rather than protected records. 
It plots semantic-plan recovery and controlled probe-atom outcomes across privacy budgets, seeds, mechanisms, and explicit experimental conditions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd
import re
import matplotlib.pyplot as plt


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


def apply_paper_style() -> None:
    """Apply consistent paper-style matplotlib settings."""
    plt.rcParams.update(PAPER_RC)


MECHANISM_LABELS = {
    "gaussian": r"Gaussian sketch + Top-$L$",
    "laplace": r"Laplace sketch + Top-$L$",
    "private_topL": r"Private Top-$L$ admission + Gaussian values",
}


def _canonical_mechanism_key(mechanism: object) -> str:
    """Normalize saved mechanism keys without changing the model parameter K.

    Older CFPB result folders may contain the deprecated mechanism string
    ``private_topk``.  That string referred to the sparse private Top-L
    admission mechanism, not to the assignment-sparsity parameter K.
    """
    key = str(mechanism).strip()
    aliases = {
        "private_topk": "private_topL",
        "private_topK": "private_topL",
        "private_top_l": "private_topL",
        "private_top-l": "private_topL",
        "sparse_private_topL": "private_topL",
    }
    return aliases.get(key, key)


def mechanism_label(mechanism: object) -> str:
    """Formal mechanism label for plot legends.

    The labels describe the DP release construction rather than only the noise
    distribution. Dense Gaussian/Laplace mechanisms perturb the full sketch and
    then select displayed atoms by post-processing. The sparse mechanism admits
    atoms with the exponential mechanism and releases Gaussian-noised selected
    values.
    """
    key = _canonical_mechanism_key(mechanism)
    return MECHANISM_LABELS.get(key, str(mechanism))


def add_metric_grid() -> None:
    """Add a light grid suitable for line plots."""
    plt.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5, alpha=0.35)


def epsilon_label(epsilon: object) -> str:
    """Return a LaTeX legend fragment for a privacy-budget value."""
    try:
        return fr"$\varepsilon={float(epsilon):g}$"
    except Exception:
        return fr"$\varepsilon={epsilon}$"


def infer_probe_support(name: str):
    """Infer the intended controlled-probe support from a run path when possible.

    Run names use the pattern ``probe_support<s>`` when support is encoded in the directory name.
    """
    m = re.search(r"support(\d+)", str(name))
    return int(m.group(1)) if m else None

def flatten(prefix: str, obj: dict, out: dict) -> None:
    """Flatten nested metric dictionaries into a single-level record.
    """
    for k, v in obj.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            flatten(key, v, out)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v


def collect_metrics(runs_dir: str) -> pd.DataFrame:
    """Collect metrics from all run directories into a dataframe.
    """
    rows = []
    for p in Path(runs_dir).glob("*/metrics.json"):
        run_dir = p.parent
        row = {}

        try:
            metrics = json.load(open(p, encoding="utf-8"))
            flatten("", metrics, row)
            # Flat aliases for nested text-evaluation metrics used in report tables.
            # The raw metrics JSON stores these under text_eval/lexical and
            # text_eval/embedding; reports use concise stable column names.
            alias_map = {
                "text_eval_lexical_token_jaccard": "text_token_jaccard",
                "text_eval_lexical_bigram_jaccard": "text_bigram_jaccard",
                "text_eval_lexical_trigram_jaccard": "text_trigram_jaccard",
                "text_eval_lexical_keyphrase_jaccard": "text_keyphrase_jaccard",
                "text_eval_lexical_tfidf_cosine": "text_tfidf_cosine",
                "text_eval_embedding_summary_embedding_cosine": "summary_embedding_cosine",
                "text_eval_comparative_dp_preferred": "comparative_dp_preferred",
            }
            for src_key, dst_key in alias_map.items():
                if src_key in row and dst_key not in row:
                    row[dst_key] = row[src_key]
            row["run"] = run_dir.name
            inferred_support = infer_probe_support(run_dir.name)
            if inferred_support is not None:
                row["probe_support"] = inferred_support

            cfgp = run_dir / "config.json"
            if cfgp.exists():
                cfg = json.load(open(cfgp, encoding="utf-8"))
                for k in [
                    "experiment_name",
                    "epsilon",
                    "delta",
                    "delta_effective",
                    "delta_configured",
                    "mechanism",
                    "seed",
                    "m",
                    "K",
                    "L",
                    "n_records",
                    "n_records_effective",
                    "n_users_effective",
                    "user_contribution_bound",
                    "probe_support",
                ]:
                    if k in cfg:
                        row[k] = _canonical_mechanism_key(cfg[k]) if k == "mechanism" else cfg[k]
                if "delta_effective" in cfg:
                    row["delta"] = cfg["delta_effective"]
                if "epsilon_effective" in cfg:
                    row["epsilon"] = cfg["epsilon_effective"]

        except Exception as e:
            row["run"] = run_dir.name
            row["error"] = str(e)

        rows.append(row)

    return pd.DataFrame(rows)


def _range_note(df: pd.DataFrame, cols: list[str]) -> str:
    """Describe whether selected configuration fields are fixed or vary across rows."""
    notes = []
    for col in cols:
        if col in df.columns and df[col].notna().any():
            vals = sorted(pd.unique(df[col].dropna()))
            vals_fmt = [str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() else str(v) for v in vals]
            if len(vals_fmt) == 1:
                notes.append(fr"${col}={vals_fmt[0]}$")
            elif len(vals_fmt) <= 4:
                notes.append(fr"mean over ${col}\in\{{{','.join(vals_fmt)}\}}$")
            else:
                notes.append(fr"mean over ${col}$")
    return "; ".join(notes)


def _has_multiple_values(df: pd.DataFrame, column: str) -> bool:
    """Return True when a column has more than one observed value."""
    return column in df.columns and df[column].dropna().nunique() > 1


def _format_condition_label(col: str, value: object) -> str:
    """Format a non-mechanism condition field for plot legends."""
    try:
        if pd.isna(value):
            return f"{col}=NA"
    except Exception:
        pass
    if col in {"m", "K", "L"}:
        try:
            return fr"${col}={int(float(value))}$"
        except Exception:
            return fr"${col}={value}$"
    return f"{col}={value}"


def _active_condition_cols(df: pd.DataFrame) -> list[str]:
    """Return condition columns that should be visible in utility plot legends."""
    cols: list[str] = []
    if "m" in df.columns:
        cols.append("m")
    for col in ["K", "L"]:
        if _has_multiple_values(df, col):
            cols.append(col)
    return cols


def plot_utility(df: pd.DataFrame, output: str) -> None:
    """Plot the two plan-recovery metrics as a function of epsilon.

    Overlap@L is the fraction of non-private top-L atoms recovered in the
    released plan. $\\mathsf{MassRetained}$ is the non-private top-L
    semantic mass captured by the released atom set, divided by the non-private
    top-L mass.
    If the input summary contains multiple dictionary sizes, assignment
    sparsities, or plan sizes, these conditions are included explicitly in
    the curve labels rather than averaged silently.
    """
    if df.empty or 'epsilon' not in df:
        return
    condition_cols = _active_condition_cols(df)
    group_cols = ['mechanism', *condition_cols, 'epsilon']
    dfg = df.groupby(group_cols, dropna=False)[['sketch_overlap_at_L','sketch_mass_retained_fraction']].mean().reset_index()
    label_cols = [c for c in group_cols if c != 'epsilon']
    apply_paper_style()
    plt.figure(figsize=(6.8, 4.2))
    for key, sub in dfg.groupby(label_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_map = dict(zip(label_cols, key))
        mech = key_map.get('mechanism')
        sub = sub.sort_values('epsilon')
        name_parts = [mechanism_label(mech)]
        name_parts.extend(_format_condition_label(c, key_map[c]) for c in condition_cols)
        name = ', '.join(name_parts)
        plt.plot(sub['epsilon'], sub['sketch_overlap_at_L'], marker='o', linestyle='-', label=f'{name}: ' + r'$\mathsf{Overlap}@L$')
        plt.plot(sub['epsilon'], sub['sketch_mass_retained_fraction'], marker='s', linestyle='--', label=f'{name}: ' + r'$\mathsf{MassRetained}$')
    plt.xlabel(r'Privacy budget $\varepsilon$')
    plt.ylabel('Mean plan-recovery score')
    plt.ylim(0,1.05)
    add_metric_grid()
    plt.legend(frameon=False, ncol=2, fontsize=6.8 if len(dfg) > 10 else 8)
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Wrote {output}")


def plot_probe(df: pd.DataFrame, output: str) -> None:
    """Plot controlled probe admission or summary-mention metrics as a function of support."""
    metric = None
    if 'probe_plan_plan_admission_rate' in df.columns:
        metric = 'probe_plan_plan_admission_rate'
    elif 'probe_summary_mention_rate' in df.columns:
        metric = 'probe_summary_mention_rate'

    if df.empty or metric is None:
        return

    if 'probe_support' in df.columns and df['probe_support'].notna().any():
        dfg = (
            df.dropna(subset=['probe_support'])
              .groupby(['mechanism', 'epsilon', 'probe_support'], dropna=False)[metric]
              .mean()
              .reset_index()
        )
        apply_paper_style()
        plt.figure(figsize=(6.2, 4.0))
        for (mech, eps), sub in dfg.groupby(['mechanism', 'epsilon']):
            sub = sub.sort_values('probe_support')
            plt.plot(
                sub['probe_support'],
                sub[metric],
                marker='o',
                label=f'{mechanism_label(mech)}, {epsilon_label(eps)}'
            )
        plt.xlabel(r'Probe support $s$')
    else:
        dfg = df.groupby(['mechanism', 'epsilon'], dropna=False)[metric].mean().reset_index()
        apply_paper_style()
        plt.figure(figsize=(6.2, 4.0))
        for mech, sub in dfg.groupby('mechanism'):
            sub = sub.sort_values('epsilon')
            plt.plot(sub['epsilon'], sub[metric], marker='o', label=mechanism_label(mech))
        plt.xlabel(r'Privacy budget $\varepsilon$')

    plt.ylabel(r'$\mathsf{Adm}_{\mathrm{plan}}$' if 'plan' in metric else r'$\mathsf{Mention}_{\mathrm{text}}$')
    plt.ylim(-0.02, 1.02)
    add_metric_grid()
    plt.legend(frameon=False)
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Wrote {output}")


def main() -> None:
    """Command-line entry point for metric collection and plotting.
    """
    p = argparse.ArgumentParser(description='Collect metrics and make plots.')
    p.add_argument('--runs-dir', default='outputs/runs')
    p.add_argument('--csv', default='outputs/metrics_summary.csv')
    p.add_argument('--plot', default='outputs/plots/utility_vs_epsilon.png')
    p.add_argument('--probe-plot', default='outputs/plots/probe_admission.png')
    args = p.parse_args()
    df = collect_metrics(args.runs_dir)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    print(f"Wrote {args.csv}")
    plot_utility(df, args.plot)
    plot_probe(df, args.probe_plot)

if __name__ == '__main__':
    main()
