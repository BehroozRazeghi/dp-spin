"""Compare saved DP-SPIN and URANIA-style release artifacts.

The reports keep final generated summaries, decoder-visible release objects, and controlled-probe outcomes as separate quantities. Plan-level DP-SPIN metrics are
not directly equated with URANIA cluster-keyword metrics. Summary-quality scores are included only when the corresponding external judge output exists; lexical and TF-IDF proxy measures are labeled separately.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import read_json, write_json

URANIA_PLAN = "urania_cluster_public_keyword_plan.json"
URANIA_METRICS = "urania_cluster_public_keyword_metrics.json"
URANIA_SUMMARY = "urania_cluster_public_keyword_summary.txt"
URANIA_VOCAB = "urania_public_keyword_vocabulary.json"
DEFAULT_PROBE_TEXT = "BLUE-LANTERN-791"

STOPWORDS = {
    "the", "and", "with", "that", "this", "from", "have", "has", "had", "was", "were", "for", "about", "into",
    "they", "their", "there", "then", "when", "what", "which", "you", "your", "are", "is", "to", "of", "in", "on",
    "a", "an", "as", "by", "or", "be", "it", "its", "at", "not", "only", "records", "record", "complaint",
    "complaints", "consumer", "summary", "baseline", "cluster", "clusters", "theme", "themes", "issue", "issues",
}

METHOD_ORDER = [
    "Raw summary",
    "Heuristic-redacted summary",
    "Non-private clustering",
    "DP keyword histogram",
    "DP category histogram",
    "URANIA-style public-keyword baseline",
    "DP-SPIN Gaussian sketch + Top-L atoms",
    "DP-SPIN exponential-mechanism Top-L atoms + Gaussian values",
]
DP_METHOD_ORDER = [
    "DP keyword histogram",
    "DP category histogram",
    "URANIA-style public-keyword baseline",
    "DP-SPIN Gaussian sketch + Top-L atoms",
    "DP-SPIN exponential-mechanism Top-L atoms + Gaussian values",
]


@dataclass
class MethodArtifact:
    """Describe one saved method summary, release object, and privacy classification."""
    method: str
    group: str
    summary_path: Path | None
    plan_path: Path | None
    metrics_path: Path | None
    decoder_input: str
    privacy_boundary: str
    released_object: str
    epsilon: Any = ""
    delta: Any = ""
    verifier_compatible: str = ""

    @property
    def summary_text(self) -> str:
        return _read_text(self.summary_path) if self.summary_path is not None else ""

    @property
    def plan(self) -> dict[str, Any]:
        return _safe_read_json(self.plan_path) if self.plan_path is not None else {}

    @property
    def metrics(self) -> dict[str, Any]:
        return _safe_read_json(self.metrics_path) if self.metrics_path is not None else {}


# ---------------------------------------------------------------------------
# basic I/O and formatting
# ---------------------------------------------------------------------------


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _safe_read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        obj = read_json(path)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _safe_read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


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
    for old, new in repl.items():
        s = s.replace(old, new)
    return s


def _fmt_num(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    try:
        x = float(value)
    except Exception:
        return str(value)
    if math.isnan(x):
        return ""
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 10:
        return f"{x:.2f}"
    return f"{x:.{digits}g}"


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        x = float(value)
        return None if math.isnan(x) else x
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    return float(m.group(0)) if m else None


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text or "") if w.lower() not in STOPWORDS]


def _top_terms(text: str, k: int = 40) -> list[str]:
    return [w for w, _ in Counter(_tokens(text)).most_common(k)]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    aa, bb = set(a), set(b)
    if not aa and not bb:
        return 0.0
    return len(aa & bb) / max(1, len(aa | bb))


def _excerpt(text: str, max_words: int = 55) -> str:
    words = re.sub(r"\s+", " ", text or "").strip().split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")


def _write_latex_table(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    columns: list[str],
    headers: list[str],
    widths: list[str] | None = None,
    table_star: bool = True,
    font_size: str = r"\scriptsize",
) -> None:
    env = "table*" if table_star else "table"
    colspec = "l" * len(columns) if widths is None else " ".join(f"p{{{w}}}" for w in widths)
    lines = [
        "% Auto-generated by src.urania_report. Requires booktabs.",
        rf"\begin{{{env}}}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        font_size,
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(_latex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_latex_escape(row.get(c, "")) for c in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# artifact discovery
# ---------------------------------------------------------------------------


def _find_first(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _prefer_baseline_dir(out_root: Path, fallback: Path) -> Path:
    candidates = [
        out_root / "openai" / "baselines" / "cfpb_openai",
        out_root / "openai_all" / "baselines" / "cfpb_openai",
        out_root / "openai_only" / "baselines" / "cfpb_openai",
        fallback,
    ]
    for c in candidates:
        if (c / "raw_summary.txt").exists() or (c / URANIA_SUMMARY).exists():
            return c
    return fallback


def _candidate_dpspin_runs(out_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    patterns = [
        "openai/runs/**/summary.txt",
        "openai_all/runs/**/summary.txt",
        "openai_only/runs/**/summary.txt",
        "openai/runs/**/summary_openai.txt",
        "main/runs/**/summary.txt",
    ]
    out: list[tuple[Path, dict[str, Any]]] = []
    for pattern in patterns:
        for summ in sorted(out_root.glob(pattern)):
            run_dir = summ.parent
            cfg = _safe_read_json(_find_first([run_dir / "config.json", run_dir / "run_config.json"]))
            plan = _safe_read_json(run_dir / "private_plan.json")
            mech = cfg.get("mechanism") or plan.get("mechanism") or (plan.get("mechanism_info") or {}).get("mechanism") or ""
            eps = cfg.get("epsilon") or (plan.get("mechanism_info") or {}).get("epsilon")
            out.append((summ, {"run_dir": run_dir, "mechanism": str(mech), "epsilon": eps, "config": cfg, "plan": plan}))
    return out


def _choose_dpspin_run(out_root: Path, mechanism: str, target_epsilon: float) -> tuple[Path | None, Path | None, dict[str, Any]]:
    aliases = {
        "gaussian": {"gaussian", "full_gaussian", "full-vector-gaussian"},
        "private_topL": {"private_topL", "private_top_l", "private_top-l", "private_topl", "sparse_private_topL"},
    }[mechanism]
    runs = []
    for summ, meta in _candidate_dpspin_runs(out_root):
        mech = str(meta.get("mechanism", "")).lower()
        if mech not in aliases:
            continue
        eps = _as_float(meta.get("epsilon"))
        dist = abs((eps if eps is not None else target_epsilon) - target_epsilon)
        prefer_openai = 0 if "/openai" in str(summ) else 1
        runs.append((prefer_openai, dist, str(summ), summ, meta))
    if not runs:
        return None, None, {}
    runs.sort()
    summ = runs[0][3]
    meta = runs[0][4]
    return summ, meta["run_dir"] / "private_plan.json", meta


def _collect_methods(out_root: Path, baseline_dir: Path, epsilon: float, delta: float) -> list[MethodArtifact]:
    methods: list[MethodArtifact] = [
        MethodArtifact("Raw summary", "non_private", baseline_dir / "raw_summary.txt", None, None, "Sampled raw target records", "Non-private utility reference", "Raw record text", "", "", "N/A"),
        MethodArtifact("Heuristic-redacted summary", "non_private", baseline_dir / "heuristic_redacted_summary.txt", None, None, "Pattern-redacted target records", "Non-private heuristic reference", "Redacted record text", "", "", "N/A"),
        MethodArtifact("Non-private clustering", "non_private", baseline_dir / "non_private_clustering_summary.txt", baseline_dir / "non_private_clustering_plan.json", None, "Target-derived cluster labels", "Non-private clustering reference", "Target-derived clusters", "", "", "N/A"),
        MethodArtifact("DP keyword histogram", "dp_baseline", baseline_dir / "dp_keyword_summary.txt", baseline_dir / "dp_keyword_plan.json", baseline_dir / "baseline_metadata.json", "Noisy histogram over fixed public keywords", "DP under fixed public keyword universe", "Public keywords with noisy counts", "1", "0", "No"),
        MethodArtifact("DP category histogram", "dp_baseline", baseline_dir / "dp_category_summary.txt", baseline_dir / "dp_category_plan.json", baseline_dir / "baseline_metadata.json", "Noisy histogram over fixed public categories", "DP under fixed public category universe", "Public categories with noisy counts", "1", "0", "No"),
        MethodArtifact("URANIA-style public-keyword baseline", "dp_baseline", baseline_dir / URANIA_SUMMARY, baseline_dir / URANIA_PLAN, baseline_dir / URANIA_METRICS, "Released cluster keywords only", "DP under fixed public keyword vocabulary and budget accounting", "Retained clusters with released public keywords", epsilon, delta, "No"),
    ]
    for name, mech in [("DP-SPIN Gaussian sketch + Top-L atoms", "gaussian"), ("DP-SPIN exponential-mechanism Top-L atoms + Gaussian values", "private_topL")]:
        summ, plan, meta = _choose_dpspin_run(out_root, mech, epsilon)
        methods.append(MethodArtifact(
            method=name,
            group="dpspin",
            summary_path=summ,
            plan_path=plan,
            metrics_path=None,
            decoder_input="Released DP semantic plan only",
            privacy_boundary="Decoder receives only admitted public atoms and noisy values or bins",
            released_object="Admitted public atoms with noisy masses or bins",
            epsilon=meta.get("epsilon", epsilon),
            delta=delta if mech == "gaussian" else "composed",
            verifier_compatible="Yes",
        ))
    return methods


# ---------------------------------------------------------------------------
# released object extraction
# ---------------------------------------------------------------------------


def _unique_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        s = str(item).strip()
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _selected_labels_from_plan(plan: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for key in ["selected_atoms", "atoms", "admitted_atoms"]:
        arr = plan.get(key)
        if isinstance(arr, list):
            for item in arr:
                if isinstance(item, dict):
                    lab = item.get("label") or item.get("name") or item.get("description")
                    if lab:
                        labels.append(str(lab))
                elif isinstance(item, str):
                    labels.append(item)
    return _unique_keep_order(labels)


def _clusters_from_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    clusters = plan.get("clusters", [])
    return clusters if isinstance(clusters, list) else []


def _keyword_strings_from_obj(obj: Any, parent_key: str = "") -> list[str]:
    key_l = parent_key.lower()
    fragments = ("keyword", "keywords", "term", "terms", "top_words", "selected_words", "selected_terms", "released_words", "released_terms")
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_keyword_strings_from_obj(v, str(k)))
    elif isinstance(obj, list):
        if any(f in key_l for f in fragments):
            for item in obj:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    for ck in ["keyword", "term", "label", "word", "text"]:
                        if isinstance(item.get(ck), str):
                            out.append(item[ck])
        else:
            for item in obj:
                out.extend(_keyword_strings_from_obj(item, parent_key))
    elif isinstance(obj, str) and any(f in key_l for f in fragments):
        out.append(obj)
    return out


def released_keywords_from_plan(plan: dict[str, Any]) -> list[str]:
    """Extract the released public keywords from a saved URANIA-style plan."""
    return _unique_keep_order(_keyword_strings_from_obj(plan))


def _released_terms(method: MethodArtifact) -> list[str]:
    if "URANIA" in method.method:
        return released_keywords_from_plan(method.plan)
    return _selected_labels_from_plan(method.plan)


def _numeric_values_released(method: MethodArtifact) -> str:
    plan = method.plan
    if not plan:
        return "No"
    if "URANIA" in method.method:
        clusters = _clusters_from_plan(plan)
        has_size = any("noisy_size" in c for c in clusters)
        has_kw = any(isinstance(c.get("keyword_noisy_counts"), dict) and c.get("keyword_noisy_counts") for c in clusters)
        vals = []
        if has_size:
            vals.append("noisy cluster sizes")
        if has_kw:
            vals.append("noisy keyword counts")
        return ", ".join(vals) if vals else "keyword ranks"
    atoms = plan.get("selected_atoms", [])
    if isinstance(atoms, list) and any(isinstance(a, dict) and ("noisy_mass" in a or "prevalence_bin" in a) for a in atoms):
        return "noisy masses or bins"
    return "No"


def _released_item_count(method: MethodArtifact) -> int | str:
    if not method.plan:
        return ""
    if "URANIA" in method.method:
        return len(_clusters_from_plan(method.plan))
    labels = _selected_labels_from_plan(method.plan)
    return len(labels) if labels else ""


def _released_label_count(method: MethodArtifact) -> int | str:
    terms = _released_terms(method)
    return len(terms) if terms else ""


# ---------------------------------------------------------------------------
# LLM judgment loading
# ---------------------------------------------------------------------------


def _canonical_from_judgment(method: str, ident: str) -> str:
    m = (method or "").lower().replace("-", "_")
    i = (ident or "").lower().replace("-", "_")
    text = f"{m} {i}"
    if "raw" in text and "redacted" not in text:
        return "Raw summary"
    if "heuristic" in text or "redacted" in text:
        return "Heuristic-redacted summary"
    if "non_private" in text or "clustering" in text:
        return "Non-private clustering"
    if "category" in text:
        return "DP category histogram"
    if "keyword" in text and "urania" not in text:
        return "DP keyword histogram"
    if "urania" in text:
        return "URANIA-style public-keyword baseline"
    if "private_top" in text or "topL" in text or "top_l" in text:
        return "DP-SPIN exponential-mechanism Top-L atoms + Gaussian values"
    if "gaussian" in text:
        return "DP-SPIN Gaussian sketch + Top-L atoms"
    if "dpspin" in text or "dp_spin" in text:
        return "DP-SPIN Gaussian sketch + Top-L atoms"
    return method


def _load_llm_judgments(out_root: Path) -> pd.DataFrame:
    candidates = [
        out_root / "openai" / "judgments" / "judgments.json",
        out_root / "openai_all" / "judgments" / "judgments.json",
        out_root / "openai_only" / "judgments" / "judgments.json",
        out_root / "paper_tables" / "openai_judgments.csv",
    ]
    rows: list[dict[str, Any]] = []
    for p in candidates:
        if not p.exists():
            continue
        try:
            if p.suffix == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                data = obj if isinstance(obj, list) else obj.get("judgments", [])
                for r in data:
                    if isinstance(r, dict):
                        rows.append(r)
            else:
                rows.extend(pd.read_csv(p).to_dict("records"))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    clean = []
    for r in rows:
        method = _canonical_from_judgment(str(r.get("method", "")), str(r.get("id", r.get("run", ""))))
        vals = {}
        for key in ["coverage", "specificity", "faithfulness", "privacy_safety", "clarity"]:
            vals[key] = _as_float(r.get(key))
        if all(v is None for v in vals.values()):
            continue
        vals["method"] = method
        vals["id"] = str(r.get("id", ""))
        clean.append(vals)
    return pd.DataFrame(clean)


def _aggregate_judgments(jdf: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if jdf.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for method, sub in jdf.groupby("method"):
        rec: dict[str, Any] = {}
        numeric_for_mean = []
        for key in ["coverage", "specificity", "faithfulness", "privacy_safety", "clarity"]:
            vals = pd.to_numeric(sub[key], errors="coerce").dropna()
            if vals.empty:
                rec[key] = "not run"
            else:
                rec[key] = _fmt_num(vals.mean(), 2)
                numeric_for_mean.append(vals.mean())
        rec["mean_score"] = _fmt_num(sum(numeric_for_mean) / len(numeric_for_mean), 2) if numeric_for_mean else "not run"
        rec["n_judgments"] = int(len(sub))
        out[str(method)] = rec
    return out


# ---------------------------------------------------------------------------
# table builders
# ---------------------------------------------------------------------------


def build_summary_quality_table(methods: list[MethodArtifact], out_root: Path, output_dir: Path) -> pd.DataFrame:
    """Build the summary-quality comparison table from available judge outputs."""
    judgments = _aggregate_judgments(_load_llm_judgments(out_root))
    rows = []
    for method in METHOD_ORDER:
        m = next((z for z in methods if z.method == method), None)
        if m is None:
            continue
        j = judgments.get(method, {})
        row = {
            "method": method,
            "coverage": j.get("coverage", "not run"),
            "specificity": j.get("specificity", "not run"),
            "faithfulness_to_released_object": j.get("faithfulness", "not run" if m.group != "non_private" else "N/A"),
            "privacy_safety": "Non-DP" if m.group == "non_private" else j.get("privacy_safety", "not run"),
            "clarity": j.get("clarity", "not run"),
            "mean_score": j.get("mean_score", "not run"),
            "summary_length": _word_count(m.summary_text),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "summary_quality_comparison_table.csv", index=False)
    _write_latex_table(
        df,
        output_dir / "summary_quality_comparison_table.tex",
        "Summary-quality comparison across DP-SPIN and baselines. Scores are reported only when an external LLM judge was run. Non-private references are marked Non-DP in the privacy-safety column.",
        "tab:summary-quality-comparison",
        ["method", "coverage", "specificity", "faithfulness_to_released_object", "privacy_safety", "clarity", "mean_score", "summary_length"],
        ["Method", "Coverage", "Specificity", "Faithfulness", "Privacy safety", "Clarity", "Mean", "Words"],
        widths=["0.22\\linewidth", "0.07\\linewidth", "0.07\\linewidth", "0.09\\linewidth", "0.09\\linewidth", "0.06\\linewidth", "0.06\\linewidth", "0.06\\linewidth"],
    )
    return df


def build_released_object_table(methods: list[MethodArtifact], output_dir: Path) -> pd.DataFrame:
    """Build a table describing each decoder-visible release object."""
    rows = []
    for m in methods:
        if m.method not in DP_METHOD_ORDER:
            continue
        rows.append({
            "method": m.method,
            "released_object": m.released_object,
            "decoder_input": m.decoder_input,
            "epsilon": _fmt_num(m.epsilon),
            "delta": _fmt_num(m.delta),
            "released_items": _released_item_count(m),
            "released_labels": _released_label_count(m),
            "numeric_values_released": _numeric_values_released(m),
            "verifier_compatible": m.verifier_compatible,
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "released_object_comparison_table.csv", index=False)
    _write_latex_table(
        df,
        output_dir / "released_object_comparison_table.tex",
        "Released-object comparison under the matched public-vocabulary target-split protocol. The table reports what is passed to the decoder before final language generation.",
        "tab:released-object-comparison",
        ["method", "released_object", "decoder_input", "epsilon", "delta", "released_items", "released_labels", "numeric_values_released", "verifier_compatible"],
        ["Method", "Released object", "Decoder input", "epsilon", "delta", "Items", "Labels", "Numeric values", "Verifier"],
        widths=["0.15\\linewidth", "0.20\\linewidth", "0.17\\linewidth", "0.045\\linewidth", "0.055\\linewidth", "0.05\\linewidth", "0.05\\linewidth", "0.16\\linewidth", "0.06\\linewidth"],
    )
    return df


def _tfidf_similarity_to_reference(reference: str, text: str) -> float:
    if not reference or not text:
        return 0.0
    try:
        X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1).fit_transform([reference, text])
        return float(cosine_similarity(X[0], X[1])[0, 0])
    except Exception:
        return 0.0


def _preference_from_judgments(method: str, quality_df: pd.DataFrame) -> str:
    if quality_df.empty or "mean_score" not in quality_df.columns:
        return "not run"
    def val_for(m: str) -> float | None:
        vals = quality_df.loc[quality_df["method"] == m, "mean_score"]
        if vals.empty:
            return None
        return _as_float(vals.iloc[0])
    v = val_for(method)
    g = val_for("DP-SPIN Gaussian sketch + Top-L atoms")
    t = val_for("DP-SPIN exponential-mechanism Top-L atoms + Gaussian values")
    base = max([x for x in [g, t] if x is not None], default=None)
    if v is None or base is None:
        return "not run"
    if abs(v - base) < 0.25:
        return "tie by LLM judge"
    return "method preferred by LLM judge" if v > base else "DP-SPIN preferred by LLM judge"


def build_reference_similarity_table(methods: list[MethodArtifact], quality_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Build lexical and TF-IDF proxy comparisons with the selected reference summary."""
    reference_text = next((m.summary_text for m in methods if m.method == "Raw summary" and m.summary_text), "")
    if not reference_text:
        reference_text = next((m.summary_text for m in methods if m.method == "Non-private clustering" and m.summary_text), "")
    ref_terms = set(_top_terms(reference_text, 40))
    rows = []
    for method in DP_METHOD_ORDER:
        m = next((z for z in methods if z.method == method), None)
        if m is None:
            continue
        text = m.summary_text
        released_terms = set(_tokens(" ".join(_released_terms(m))))
        rows.append({
            "method": method,
            "lexical_overlap_with_reference": _fmt_num(_jaccard(_top_terms(text, 40), ref_terms), 3),
            "atom_or_keyword_overlap_with_reference": _fmt_num(_jaccard(released_terms, ref_terms) if released_terms else 0.0, 3),
            "embedding_similarity_to_reference_summary": _fmt_num(_tfidf_similarity_to_reference(reference_text, text), 3),
            "llm_preference_vs_dpspin": _preference_from_judgments(method, quality_df),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "reference_summary_similarity_table.csv", index=False)
    _write_latex_table(
        df,
        output_dir / "reference_summary_similarity_table.tex",
        "Reference-summary similarity for DP methods. The reference is the non-private raw summary when available. The embedding similarity column uses TF-IDF n-gram cosine unless an external semantic-embedding evaluator is supplied separately. Pairwise preference is reported only when LLM judgments are available.",
        "tab:reference-summary-similarity",
        ["method", "lexical_overlap_with_reference", "atom_or_keyword_overlap_with_reference", "embedding_similarity_to_reference_summary", "llm_preference_vs_dpspin"],
        ["Method", "Lexical overlap", "Atom or keyword overlap", "Embedding similarity", "Preference vs DP-SPIN"],
        widths=["0.20\\linewidth", "0.13\\linewidth", "0.16\\linewidth", "0.16\\linewidth", "0.25\\linewidth"],
    )
    return df


# ---------------------------------------------------------------------------
# Controlled probe-atom diagnostics
# ---------------------------------------------------------------------------


def _normalize_token_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _probe_matches_string(candidate: Any, probe_text: str) -> bool:
    raw = str(probe_text or DEFAULT_PROBE_TEXT).lower().strip()
    norm = _normalize_token_text(raw)
    parts = norm.split()
    s = str(candidate or "").lower()
    sn = _normalize_token_text(s)
    compact = sn.replace(" ", "")
    return bool(raw and raw in s) or bool(norm and norm in sn) or bool(parts and "".join(parts) in compact)


def probe_keyword_released(plan: dict[str, Any], probe_text: str = DEFAULT_PROBE_TEXT) -> tuple[str, list[str]]:
    """Check whether a probe keyword appears in a released cluster-keyword object."""
    kws = released_keywords_from_plan(plan)
    matches = [kw for kw in kws if _probe_matches_string(kw, probe_text)]
    return ("1" if matches else "0"), matches


def _urania_vocab_contains_probe(base: Path, probe_text: str) -> str:
    obj = _safe_read_json(base / URANIA_VOCAB)
    vocab = obj.get("public_keywords", []) if isinstance(obj, dict) else []
    if not vocab:
        return "unknown"
    return "1" if any(_probe_matches_string(v, probe_text) for v in vocab) else "0"


def _urania_probe_rows(urania_probe_csv: Path, probe_text: str) -> pd.DataFrame:
    raw = _safe_read_csv(urania_probe_csv)
    if raw.empty:
        return pd.DataFrame()
    rows = []
    for _, r in raw.iterrows():
        support = int(r.get("support"))
        base = Path(str(r.get("baseline_path", "")))
        plan = _safe_read_json(base / URANIA_PLAN)
        summary = _read_text(base / URANIA_SUMMARY)
        in_vocab = _urania_vocab_contains_probe(base, probe_text)
        if not plan:
            released, matches = "N/A", []
        elif in_vocab == "0":
            released, matches = "not comparable", []
        else:
            released, matches = probe_keyword_released(plan, probe_text)
        rows.append({
            "support": support,
            "urania_probe_in_keyword_universe": in_vocab,
            "urania_probe_keyword_released": released,
            "urania_matching_keywords": "; ".join(matches),
            "urania_final_text_exact_mention": "1" if _probe_matches_string(summary, probe_text) else str(int(r.get("exact_probe_mentioned", 0) or 0)),
        })
    return pd.DataFrame(rows).sort_values("support")


def _dpspin_probe_rows(probe_report: Path, epsilon: float) -> pd.DataFrame:
    df = _safe_read_csv(probe_report)
    if df.empty:
        return pd.DataFrame()
    if "epsilon" in df.columns:
        eps = df["epsilon"].apply(_as_float)
        valid = eps.dropna()
        if not valid.empty:
            closest = valid.iloc[(valid - epsilon).abs().argsort().iloc[0]]
            df = df[(eps - closest).abs() < 1e-9]
    if "m" in df.columns:
        mvals = pd.to_numeric(df["m"], errors="coerce")
        if mvals.notna().any():
            df = df[mvals == mvals.max()]
    if "L" in df.columns:
        Lvals = pd.to_numeric(df["L"], errors="coerce")
        if Lvals.notna().any():
            df = df[Lvals == 10]
    rows = []
    for _, r in df.iterrows():
        support = int(r.get("probe_support", r.get("support", 0)))
        mech = str(r.get("mechanism", ""))
        admission_raw = r.get("probe_plan_plan_admission_rate", r.get("probe_summary_mention_rate", ""))
        final_raw = r.get("probe_summary_mention_rate", r.get("probe_final_text_exact_mention", r.get("exact_probe_mentioned", "N/A")))
        admission = _as_float(admission_raw)
        final_exact = _as_float(final_raw)
        rows.append({
            "support": support,
            "mechanism": mech,
            "plan_admission": "" if admission is None else admission,
            "final_exact": "N/A" if final_exact is None else final_exact,
        })
    return pd.DataFrame(rows)


def build_probe_diagnostic_table(probe_report: Path, urania_probe_csv: Path, output_dir: Path, figure_dir: Path, epsilon: float, probe_text: str) -> pd.DataFrame:
    """Build controlled-probe release and final-text diagnostic tables and figures."""
    u = _urania_probe_rows(urania_probe_csv, probe_text)
    d = _dpspin_probe_rows(probe_report, epsilon)
    rows: dict[int, dict[str, Any]] = {}
    for _, r in u.iterrows() if not u.empty else []:
        s = int(r["support"])
        rows.setdefault(s, {"support": s})
        rows[s]["urania_probe_in_keyword_universe"] = r.get("urania_probe_in_keyword_universe", "")
        rows[s]["urania_probe_keyword_released"] = r.get("urania_probe_keyword_released", "")
        rows[s]["urania_final_text_exact_mention"] = r.get("urania_final_text_exact_mention", "")
    for _, r in d.iterrows() if not d.empty else []:
        s = int(r["support"])
        rows.setdefault(s, {"support": s})
        mech = str(r.get("mechanism", ""))
        if mech == "gaussian":
            rows[s]["dpspin_gaussian_probe_atom_admitted"] = r.get("plan_admission", "")
            rows[s]["dpspin_final_text_exact_mention"] = r.get("final_exact", "N/A")
        elif mech == "laplace":
            rows[s]["dpspin_laplace_probe_atom_admitted"] = r.get("plan_admission", "")
        elif "private" in mech:
            rows[s]["dpspin_private_topL_probe_atom_admitted"] = r.get("plan_admission", "")
    df = pd.DataFrame([rows[k] for k in sorted(rows)]) if rows else pd.DataFrame(columns=["support"])
    for name in ["probe_released_object_diagnostic_table.csv", "urania_released_object_probe_table.csv", "urania_probe_comparison_table.csv"]:
        df.to_csv(output_dir / name, index=False)
    cols = [c for c in [
        "support",
        "dpspin_gaussian_probe_atom_admitted",
        "dpspin_laplace_probe_atom_admitted",
        "dpspin_private_topL_probe_atom_admitted",
        "urania_probe_in_keyword_universe",
        "urania_probe_keyword_released",
        "dpspin_final_text_exact_mention",
        "urania_final_text_exact_mention",
    ] if c in df.columns]
    headers = {
        "support": "Support",
        "dpspin_gaussian_probe_atom_admitted": "DP-SPIN Gaussian probe atom admitted",
        "dpspin_laplace_probe_atom_admitted": "DP-SPIN Laplace probe atom admitted",
        "dpspin_private_topL_probe_atom_admitted": "DP-SPIN exponential-mechanism Top-L probe atom admitted",
        "urania_probe_in_keyword_universe": "URANIA probe string in vocabulary",
        "urania_probe_keyword_released": "URANIA keyword released",
        "dpspin_final_text_exact_mention": "DP-SPIN final text mention",
        "urania_final_text_exact_mention": "URANIA final text mention",
    }
    _write_latex_table(
        df[cols] if cols else df,
        output_dir / "probe_released_object_diagnostic_table.tex",
        "Controlled probe-atom released-object comparison. DP-SPIN plan admission means admission of the public probe atom into the released semantic plan. URANIA released-keyword admission means release of the probe keyword in a per-cluster DP keyword histogram. The URANIA released-keyword column is comparable only when the probe string is present in the fixed public keyword vocabulary.",
        "tab:probe-released-object-comparison",
        cols if cols else list(df.columns),
        [headers.get(c, c) for c in (cols if cols else list(df.columns))],
    )
    for alias in ["urania_released_object_probe_table.tex", "urania_probe_comparison_table.tex"]:
        (output_dir / alias).write_text((output_dir / "probe_released_object_diagnostic_table.tex").read_text(encoding="utf-8"), encoding="utf-8")

    if not df.empty:
        figure_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7.2, 4.2))
        x = pd.to_numeric(df["support"], errors="coerce")
        for col, label in [
            ("dpspin_gaussian_probe_atom_admitted", r"DP-SPIN Gaussian probe atom admitted"),
            ("dpspin_laplace_probe_atom_admitted", r"DP-SPIN Laplace probe atom admitted"),
            ("dpspin_private_topL_probe_atom_admitted", r"DP-SPIN sparse exponential-mechanism probe atom admitted"),
            ("urania_probe_keyword_released", r"URANIA keyword released"),
        ]:
            if col in df.columns:
                y = pd.to_numeric(df[col], errors="coerce")
                if y.notna().any():
                    plt.plot(x, y, marker="o", label=label)
        plt.xlabel("Probe support $s$")
        plt.ylabel("Released-object admission rate")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, which="major", axis="both", linestyle="--", linewidth=0.5, alpha=0.35)
        plt.legend(frameon=False)
        plt.tight_layout()
        for name in ["fig8_released_object_probe_comparison.png", "fig8_urania_probe_comparison.png"]:
            plt.savefig(figure_dir / name, dpi=240)
        plt.close()
    return df


# ---------------------------------------------------------------------------
# legacy and qualitative helpers
# ---------------------------------------------------------------------------


def build_urania_quantitative_table(baseline_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Build the quantitative table from saved URANIA-style baseline metrics."""
    plan = _safe_read_json(baseline_dir / URANIA_PLAN)
    metrics = _safe_read_json(baseline_dir / URANIA_METRICS)
    clusters = _clusters_from_plan(plan)
    keywords = released_keywords_from_plan(plan)
    row = {
        "method": "URANIA-style public-keyword baseline",
        "epsilon": _fmt_num(metrics.get("epsilon_total")),
        "delta": _fmt_num(metrics.get("delta_total")),
        "records": _fmt_num(metrics.get("num_records"), 0),
        "public_keywords": _fmt_num(metrics.get("num_public_keywords"), 0),
        "released_clusters": _fmt_num(metrics.get("num_clusters_released", len(clusters)), 0),
        "records_matched": _fmt_num(metrics.get("num_records_with_public_keyword_match"), 0),
        "mean_noisy_cluster_size": _fmt_num(metrics.get("mean_noisy_released_cluster_size"), 2),
        "released_keywords": _fmt_num(len(keywords), 0),
        "summary_words": _fmt_num(_word_count(_read_text(baseline_dir / URANIA_SUMMARY)), 0),
    }
    df = pd.DataFrame([row])
    df.to_csv(output_dir / "urania_quantitative_table.csv", index=False)
    _write_latex_table(df, output_dir / "urania_quantitative_table.tex", "Mechanism diagnostics for the URANIA-style public-keyword baseline. This is not a utility comparison table.", "tab:urania-quantitative", list(df.columns), [c.replace("_", " ") for c in df.columns])
    return df


def build_qualitative_table(methods: list[MethodArtifact], output_dir: Path) -> pd.DataFrame:
    """Build the qualitative summary and release-object comparison table."""
    rows = []
    for m in methods:
        if m.summary_text:
            rows.append({"method": m.method, "decoder_input": m.decoder_input, "privacy_boundary": m.privacy_boundary, "summary_excerpt": _excerpt(m.summary_text, 55)})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "urania_qualitative_table.csv", index=False)
    _write_latex_table(df, output_dir / "urania_qualitative_table.tex", "Representative summary excerpts with decoder inputs and privacy boundaries.", "tab:urania-qualitative", ["method", "decoder_input", "privacy_boundary", "summary_excerpt"], ["Method", "Decoder input", "Privacy boundary", "Summary excerpt"], widths=["0.14\\linewidth", "0.20\\linewidth", "0.22\\linewidth", "0.36\\linewidth"])
    return df


def build_legacy_method_comparison(released_df: pd.DataFrame, output_dir: Path) -> None:
    """Write the compatibility method-comparison table used by earlier reports."""
    released_df.to_csv(output_dir / "urania_dpspin_comparison_table.csv", index=False)
    _write_latex_table(released_df, output_dir / "urania_dpspin_comparison_table.tex", "Released-object comparison between DP-SPIN and DP baselines.", "tab:urania-dpspin-results", list(released_df.columns), [c.replace("_", " ") for c in released_df.columns])


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def create_urania_reports(
    out_root: Path,
    baseline_dir: Path | None = None,
    output_dir: Path | None = None,
    figure_dir: Path | None = None,
    probe_report: Path | None = None,
    urania_probe_csv: Path | None = None,
    probe_text: str = DEFAULT_PROBE_TEXT,
) -> None:
    """Generate all DP-SPIN and URANIA-style comparison report artifacts."""
    baseline_dir = _prefer_baseline_dir(out_root, baseline_dir or out_root / "baselines" / "cfpb_template")
    output_dir = output_dir or out_root / "paper_tables"
    figure_dir = figure_dir or out_root / "paper_figures"
    probe_report = probe_report or out_root / "probe" / "reports" / "summary_table.csv"
    urania_probe_csv = urania_probe_csv or out_root / "baselines" / "urania_probe" / "exact_probe_mention.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics = _safe_read_json(baseline_dir / URANIA_METRICS)
    epsilon = float(metrics.get("epsilon_total", 1.0) or 1.0)
    delta = float(metrics.get("delta_total", 1e-6) or 1e-6)
    methods = _collect_methods(out_root, baseline_dir, epsilon, delta)

    quality_df = build_summary_quality_table(methods, out_root, output_dir)
    released_df = build_released_object_table(methods, output_dir)
    reference_df = build_reference_similarity_table(methods, quality_df, output_dir)
    probe_df = build_probe_diagnostic_table(probe_report, urania_probe_csv, output_dir, figure_dir, epsilon, probe_text)
    build_urania_quantitative_table(baseline_dir, output_dir)
    build_qualitative_table(methods, output_dir)
    build_legacy_method_comparison(released_df, output_dir)

    manifest = {
        "out_root": str(out_root),
        "baseline_dir_used": str(baseline_dir),
        "external_llm_judgments_found": not _load_llm_judgments(out_root).empty,
        "generated_tables": [
            "summary_quality_comparison_table.tex",
            "released_object_comparison_table.tex",
            "reference_summary_similarity_table.tex",
            "probe_released_object_diagnostic_table.tex",
            "urania_quantitative_table.tex",
            "urania_qualitative_table.tex",
            "urania_dpspin_comparison_table.tex",
        ],
        "generated_figures": ["fig8_released_object_probe_comparison.png"],
        "missing_summaries": [m.method for m in methods if not m.summary_text],
        "notes": [
            "Summary-quality scores are not filled unless external LLM judgments are present.",
            "Reference-summary similarity uses local lexical and TF-IDF metrics and should be described as a proxy unless a semantic evaluator is added.",
            "URANIA probe-keyword admission is comparable only when the probe string is present in the fixed public keyword vocabulary.",
        ],
    }
    write_json(manifest, output_dir / "urania_report_manifest.json")
    print(f"Wrote comparison tables to {output_dir}")
    print(f"Wrote comparison figures to {figure_dir}")


def main() -> None:
    """Generate DP-SPIN and URANIA-style comparison reports from the command line."""
    p = argparse.ArgumentParser(description="Create paper-facing DP-SPIN and URANIA comparison tables from existing outputs.")
    p.add_argument("--out", default="outputs/cfpb")
    p.add_argument("--baseline-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--figure-dir", default=None)
    p.add_argument("--main-report", default=None, help="Accepted for compatibility. Not used.")
    p.add_argument("--probe-report", default=None)
    p.add_argument("--urania-probe", default=None)
    p.add_argument("--probe-text", default=DEFAULT_PROBE_TEXT)
    args = p.parse_args()
    create_urania_reports(
        out_root=Path(args.out),
        baseline_dir=Path(args.baseline_dir) if args.baseline_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        figure_dir=Path(args.figure_dir) if args.figure_dir else None,
        probe_report=Path(args.probe_report) if args.probe_report else None,
        urania_probe_csv=Path(args.urania_probe) if args.urania_probe else None,
        probe_text=args.probe_text,
    )


if __name__ == "__main__":
    main()
