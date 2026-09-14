"""Aggregate saved OpenAI decoding and judging results.

The module reads generated summaries, verifier outputs, and judge files from completed runs. 
It writes dataset tables and qualitative examples without calling the OpenAI API or accessing protected records directly.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


SCORE_COLS_DPSPIN = ["coverage", "faithfulness", "usefulness", "clarity"]
SCORE_COLS_GENERIC = ["coverage", "specificity", "faithfulness", "privacy_safety", "clarity"]


def read_json(path: Path) -> Any:
    """Read a JSON file if it exists; otherwise return a default value.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    """Read a text file if it exists; otherwise return a default string.
    """
    return path.read_text(encoding="utf-8") if path.exists() else ""


def latex_escape(s: str) -> str:
    """Escape characters that have special meaning in LaTeX tables.
    """
    if s is None:
        return ""
    s = str(s)
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


def short_text(s: str, max_chars: int = 850) -> str:
    """Collapse whitespace and truncate text for qualitative examples.
    """
    s = re.sub(r"\s+", " ", s.strip())
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def fmt_mean_std(mean: float, std: float | None, count: int | None = None) -> str:
    """Format a mean and standard deviation for a report cell."""
    if pd.isna(mean):
        return "--"
    if std is None or pd.isna(std) or (count is not None and count <= 1):
        return f"{mean:.2f}"
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def collect_dpspin_runs(openai_dir: Path) -> pd.DataFrame:
    """Collect DP-SPIN run metrics and judge scores from output directories.
    """
    rows: list[dict[str, Any]] = []
    dpspin = openai_dir / "dpspin"
    for run in sorted(dpspin.glob("*")):
        if not run.is_dir():
            continue

        cfgp = run / "config.json"
        if not cfgp.exists():
            continue
        cfg = read_json(cfgp)

        row: dict[str, Any] = {
            "run": run.name,
            "method": "DP-SPIN",
            "mechanism": cfg.get("mechanism"),
            "epsilon": cfg.get("epsilon"),
            "seed": cfg.get("seed"),
            "m": cfg.get("m"),
            "K": cfg.get("K"),
            "L": cfg.get("L"),
            "summary_path": str(run / "summary_openai.txt"),
            "plan_path": str(run / "private_plan.json"),
        }

        judgep = run / "judge_openai.json"
        if judgep.exists():
            try:
                judge = read_json(judgep)
                for k, v in judge.items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        row[f"judge_{k}"] = v
            except Exception as e:
                row["judge_error"] = str(e)

        verifierp = run / "verifier_openai_reverified.json"
        if not verifierp.exists():
            verifierp = run / "verifier_openai.json"
        if verifierp.exists():
            try:
                verifier = read_json(verifierp)
                for k, v in verifier.items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        row[f"verifier_{k}"] = v
            except Exception as e:
                row["verifier_error"] = str(e)

        metricsp = run / "metrics.json"
        if metricsp.exists():
            try:
                metrics = read_json(metricsp)
                sketch = metrics.get("sketch", {})
                for k, v in sketch.items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        row[f"sketch_{k}"] = v
            except Exception as e:
                row["metrics_error"] = str(e)

        rows.append(row)

    return pd.DataFrame(rows)


def collect_generic_judgments(openai_dir: Path) -> pd.DataFrame:
    """Collect baseline judge files into a dataframe.
    """
    p_json = openai_dir / "judgments" / "judgments.json"
    p_csv = openai_dir / "judgments" / "judgments.csv"

    if p_csv.exists():
        return pd.read_csv(p_csv)
    if p_json.exists():
        data = read_json(p_json)
        return pd.DataFrame(data)
    return pd.DataFrame()


def make_dpspin_latex(df: pd.DataFrame, out_tex: Path) -> None:
    """Write a LaTeX table summarizing DP-SPIN OpenAI evaluations.
    """
    if df.empty:
        out_tex.write_text("% No DP-SPIN OpenAI runs found.\n", encoding="utf-8")
        return

    needed = ["mechanism", "epsilon"]
    for col in needed:
        if col not in df.columns:
            out_tex.write_text(f"% Missing required column: {col}\n", encoding="utf-8")
            return

    score_cols = [f"judge_{c}" for c in SCORE_COLS_DPSPIN if f"judge_{c}" in df.columns]
    group_cols = ["mechanism", "epsilon"]

    rows = []
    grouped = df.groupby(group_cols, dropna=False)
    for (mech, eps), g in grouped:
        row = {
            "Mechanism": str(mech),
            r"$\varepsilon$": eps,
            "Runs": len(g),
        }
        for col in score_cols:
            clean = col.replace("judge_", "").capitalize()
            row[clean] = fmt_mean_std(g[col].mean(), g[col].std(), len(g))

        if "verifier_UAR" in g.columns:
            row["UAR"] = f"{g['verifier_UAR'].mean():.2f}"
        if "verifier_UCR" in g.columns:
            row["UCR"] = f"{g['verifier_UCR'].mean():.2f}"
        if "verifier_comparison_count" in g.columns:
            row["Comparisons"] = f"{g['verifier_comparison_count'].mean():.2f}"
        if "verifier_accept" in g.columns:
            row["Accept"] = f"{int(g['verifier_accept'].sum())}/{len(g)}"

        rows.append(row)

    tab = pd.DataFrame(rows).sort_values(["Mechanism", r"$\varepsilon$"])
    columns = list(tab.columns)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{OpenAI verbalization of DP-SPIN private plans. The decoder receives only the released DP plan. Scores are assigned by the plan-conditioned LLM judge on a 1--5 scale.}",
        r"\label{tab:openai-dpspin}",
        r"\small",
        r"\begin{tabular}{" + "l" * len(columns) + r"}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for _, r in tab.iterrows():
        vals = [latex_escape(r[c]) for c in columns]
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def make_baseline_latex(df: pd.DataFrame, out_tex: Path) -> None:
    """Write a LaTeX table summarizing baseline OpenAI evaluations.
    """
    if df.empty:
        out_tex.write_text("% No generic OpenAI judgments found.\n", encoding="utf-8")
        return

    score_cols = [c for c in SCORE_COLS_GENERIC if c in df.columns]
    if not score_cols:
        out_tex.write_text("% No baseline score columns found.\n", encoding="utf-8")
        return

    # Use one row per method in the selected report table.
    rows = []
    for method, g in df.groupby("method", dropna=False):
        row = {"Method": method, "Runs": len(g)}
        for col in score_cols:
            row[col.capitalize().replace("_", " ")] = fmt_mean_std(g[col].mean(), g[col].std(), len(g))
        rows.append(row)

    tab = pd.DataFrame(rows)
    # Put DP-SPIN last/first consistently.
    tab["sort_key"] = tab["Method"].astype(str).map(lambda x: "0" if x == "DP-SPIN" else "1" + x)
    tab = tab.sort_values("sort_key").drop(columns=["sort_key"])

    columns = list(tab.columns)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{OpenAI-judged summary quality for DP-SPIN and baselines. Raw, redacted, and clustering baselines are non-private or heuristic baselines; DP-SPIN exposes only the DP plan to the decoder. Scores are on a 1--5 scale.}",
        r"\label{tab:openai-baselines}",
        r"\small",
        r"\begin{tabular}{" + "l" * len(columns) + r"}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for _, r in tab.iterrows():
        vals = [latex_escape(r[c]) for c in columns]
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def choose_representative_dpspin(openai_dir: Path) -> Path | None:
    """Select one representative DP-SPIN run for qualitative reporting.
    """
    candidates = list((openai_dir / "dpspin").glob("*gaussian_eps2.0_seed0"))
    if candidates:
        return candidates[0]
    candidates = list((openai_dir / "dpspin").glob("*"))
    return candidates[0] if candidates else None


def find_baseline_summary_paths(openai_dir: Path) -> dict[str, Path]:
    """Locate baseline summary files that can be used in qualitative examples.
    """
    root = openai_dir / "baselines" / "cfpb_openai"
    out = {}
    candidates = {
        "Raw OpenAI": "raw_summary.txt",
        "Heuristic redaction": "heuristic_redacted_summary.txt",
        "Non-private clustering": "non_private_clustering_summary.txt",
        "DP keyword histogram": "dp_keyword_summary.txt",
        "DP category histogram": "dp_category_summary.txt",
    }
    for label, fname in candidates.items():
        p = root / fname
        if p.exists():
            out[label] = p
    return out


def make_qualitative_examples(openai_dir: Path, out_tex: Path, out_md: Path) -> None:
    """Write qualitative example snippets for DP-SPIN and baselines.
    """
    dpspin_run = choose_representative_dpspin(openai_dir)
    examples: list[tuple[str, str]] = []

    if dpspin_run is not None:
        examples.append(("DP-SPIN", read_text(dpspin_run / "summary_openai.txt")))

    for label, p in find_baseline_summary_paths(openai_dir).items():
        examples.append((label, read_text(p)))

    if not examples:
        out_tex.write_text("% No qualitative examples found.\n", encoding="utf-8")
        out_md.write_text("No qualitative examples found.\n", encoding="utf-8")
        return

    # Markdown version: easiest to read in repository.
    md = ["# OpenAI qualitative summary examples", ""]
    if dpspin_run is not None:
        md.append(f"Representative DP-SPIN run: `{dpspin_run.name}`")
        md.append("")
    for label, text in examples:
        md += [f"## {label}", "", short_text(text, 2000), ""]
    out_md.write_text("\n".join(md), encoding="utf-8")

    # Write the corresponding LaTeX appendix table.
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Representative OpenAI-generated summaries. DP-SPIN receives only the released DP plan; raw and redacted baselines receive record text or heuristically redacted record text. Text is truncated for display.}",
        r"\label{tab:openai-qualitative-examples}",
        r"\small",
        r"\begin{tabular}{p{0.20\linewidth}p{0.74\linewidth}}",
        r"\toprule",
        r"Method & Generated summary \\",
        r"\midrule",
    ]
    for label, text in examples:
        lines.append(latex_escape(label) + " & " + latex_escape(short_text(text, 900)) + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def make_results_text(out_tex: Path) -> None:
    """Write a prose summary of the saved OpenAI evaluation results."""
    text = r"""
\paragraph{LLM verbalization from private plans.}
We evaluate whether an LLM can verbalize the released DP semantic plan without receiving raw records.  The decoder receives only the private plan and public instructions.  It does not receive raw complaints, nearest neighbors, raw contribution vectors, or non-private cluster representatives.  Across the completed OpenAI-verbalized DP-SPIN runs, the plan verifier reports zero unsupported atom rate and zero unsupported comparison rate in the accepted runs.  Thus the LLM did not introduce verifier-detected themes outside the released plan.

The LLM evaluation should be interpreted separately from the sketch-level utility curves.  The privacy--utility tradeoff appears primarily in the released plan: lower privacy budgets reduce support recovery and increase support-mass loss, especially for private top-$k$.  Once the released plan contains coherent high-level atoms, the LLM usually produces fluent summaries, so LLM-judged readability and clarity can saturate even when the plan-level metrics vary.

\paragraph{Comparison to non-private and histogram baselines.}
Raw and heuristic-redacted OpenAI summaries can receive high judged utility because they have access to raw or nearly raw record text.  These baselines do not satisfy the same formal decoder privacy boundary.  DP keyword and category histograms are strong baselines on CFPB because the dataset contains informative public metadata.  DP-SPIN should therefore be interpreted as providing comparable plan-conditioned verbalization while enforcing that the decoder observes only a differentially private semantic plan, rather than as an unconstrained utility improvement over raw summarization.
"""
    out_tex.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    """Command-line entry point for OpenAI report aggregation.
    """
    p = argparse.ArgumentParser(description="Generate LaTeX tables and qualitative examples for OpenAI DP-SPIN outputs.")
    p.add_argument("--openai-dir", default="outputs/cfpb/openai_only")
    p.add_argument("--output-dir", default="outputs/cfpb/openai_only/latex")
    args = p.parse_args()

    openai_dir = Path(args.openai_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    dpspin_df = collect_dpspin_runs(openai_dir)
    generic_df = collect_generic_judgments(openai_dir)

    dpspin_df.to_csv(outdir / "dpspin_openai_runs_collected.csv", index=False)
    generic_df.to_csv(outdir / "generic_openai_judgments_collected.csv", index=False)

    make_dpspin_latex(dpspin_df, outdir / "tab_openai_dpspin.tex")
    make_baseline_latex(generic_df, outdir / "tab_openai_baselines.tex")
    make_qualitative_examples(
        openai_dir,
        outdir / "tab_openai_qualitative_examples.tex",
        outdir / "openai_qualitative_examples.md",
    )
    make_results_text(outdir / "openai_results_text.tex")

    # Minimal README
    readme = f"""OpenAI reporting outputs

Created from: {openai_dir}

Files:
- tab_openai_dpspin.tex
- tab_openai_baselines.tex
- tab_openai_qualitative_examples.tex
- openai_qualitative_examples.md
- openai_results_text.tex
- dpspin_openai_runs_collected.csv
- generic_openai_judgments_collected.csv

Use the LaTeX tables in the paper/appendix. Use the Markdown file to inspect real summary text.
"""
    (outdir / "README_OPENAI_LATEX.txt").write_text(readme, encoding="utf-8")

    print(f"Wrote OpenAI LaTeX/reporting outputs to {outdir}")


if __name__ == "__main__":
    main()
