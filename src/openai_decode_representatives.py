"""Decode selected completed runs with an OpenAI backend.

The module selects representative saved DP-SPIN runs, verbalizes each released semantic plan in audit and display modes, applies the public verifier, and writes
qualitative and text-evaluation artifacts. It does not rerun semantic assignment or a DP release mechanism.

For DP-SPIN summaries, the OpenAI prompt contains only the released plan and public instructions. 
Non-private reference plans and summaries are generated separately for offline evaluation and are not supplied to the DP decoder.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .decode import decode_plan
from .judge_summary import heuristic_judge, llm_judge
from .summary_eval import build_clean_reference_plan, compute_summary_text_evaluation
from .utils import ensure_dir, read_json, write_json
from .verify import verify_summary


DEFAULT_DATASETS = {
    "cfpb": {
        "dataset_name": "cfpb",
        "display_name": "CFPB consumer complaints",
        "run_dirs": ["outputs/cfpb/main/runs"],
        "output_dir": "outputs/cfpb/openai_representative",
        "units": ["record"],
        "target_m": 200,
        "target_n": 2000,
        "target_B": None,
    },
    "amazon": {
        "dataset_name": "amazon_All_Beauty",
        "display_name": "Amazon All Beauty reviews",
        "run_dirs": [
            "outputs/amazon_All_Beauty/record/runs",
            "outputs/amazon_All_Beauty/user/runs",
        ],
        "output_dir": "outputs/amazon_All_Beauty/openai_representative",
        "units": ["record", "user"],
        "target_m": 46,
        "target_n": None,
        "target_B": 1.0,
    },
    "yelp": {
        "dataset_name": "yelp_restaurants",
        "display_name": "Yelp Restaurants reviews",
        "run_dirs": [
            "outputs/yelp_restaurants/record/runs",
            "outputs/yelp_restaurants/user/runs",
        ],
        "output_dir": "outputs/yelp_restaurants/openai_representative",
        "units": ["record", "user"],
        "target_m": 100,
        "target_n": None,
        "target_B": 1.0,
    },
}

DEFAULT_MECHANISMS = ["gaussian", "laplace", "private_topL"]
DEFAULT_EPSILONS = [1.0, 2.0]
DEFAULT_SEEDS = [0]
DEFAULT_L = 10
DEFAULT_K = 3

COPY_FILES = [
    "private_plan.json",
    "clean_reference_plan.json",
    "atoms_used.json",
    "config.json",
    "config.yaml",
    "metrics.json",
    "clean_sketch.npy",
]


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


def _cfg_n_records(cfg: dict[str, Any]) -> int | None:
    return _safe_int(cfg.get("n_records") or cfg.get("n_records_effective"))


def _cfg_n_users(cfg: dict[str, Any]) -> int | None:
    return _safe_int(cfg.get("n_users") or cfg.get("n_users_effective"))


def _run_unit(cfg: dict[str, Any], run_path: Path) -> str:
    unit = str(cfg.get("protected_unit") or "").strip().lower()
    if unit in {"record", "user"}:
        return unit
    parts = run_path.parts
    if "user" in parts:
        return "user"
    return "record"


def _config(run_dir: Path) -> dict[str, Any] | None:
    p = run_dir / "config.json"
    if p.exists():
        try:
            return read_json(p)
        except Exception:
            return None
    return None


def _iter_runs(run_dirs: Iterable[str | Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for rd in run_dirs:
        root = Path(rd)
        if not root.exists():
            continue
        for metrics in sorted(root.glob("*/metrics.json")):
            run = metrics.parent
            if not (run / "private_plan.json").exists():
                continue
            cfg = _config(run)
            if cfg is not None:
                yield run, cfg


def _matches(
    run: Path,
    cfg: dict[str, Any],
    units: set[str],
    mechanisms: set[str],
    epsilons: set[float],
    seeds: set[int],
    m_values: set[int] | None,
    K: int | None,
    L: int | None,
    B_values: set[float] | None,
    n_records_values: set[int] | None = None,
    n_users_values: set[int] | None = None,
) -> bool:
    unit = _run_unit(cfg, run)
    if unit not in units:
        return False
    if str(cfg.get("mechanism")) not in mechanisms:
        return False
    eps = _safe_float(cfg.get("epsilon"))
    if eps is None or not any(abs(eps - e) < 1e-9 for e in epsilons):
        return False
    seed = _safe_int(cfg.get("seed"))
    if seed is None or seed not in seeds:
        return False
    if m_values is not None:
        m = _safe_int(cfg.get("m"))
        if m is None or m not in m_values:
            return False
    if K is not None and _safe_int(cfg.get("K")) != K:
        return False
    if L is not None and _safe_int(cfg.get("L")) != L:
        return False
    if unit == "user" and B_values is not None:
        B = _safe_float(cfg.get("user_contribution_bound"))
        if B is None or not any(abs(B - b) < 1e-9 for b in B_values):
            return False
    if unit == "record" and n_records_values is not None:
        n = _cfg_n_records(cfg)
        if n is None or n not in n_records_values:
            return False
    if unit == "user" and n_users_values is not None:
        u = _cfg_n_users(cfg)
        if u is None or u not in n_users_values:
            return False
    return True


def _score_run(cfg: dict[str, Any], target_m: int | None, target_B: float | None) -> tuple:
    """Score candidate runs for deterministic representative-condition selection."""
    m = _safe_int(cfg.get("m"))
    B = _safe_float(cfg.get("user_contribution_bound"))
    eps = _safe_float(cfg.get("epsilon"))
    seed = _safe_int(cfg.get("seed"))
    return (
        abs((m or 10**9) - target_m) if target_m is not None else 0,
        abs((B or 10**9) - target_B) if target_B is not None and str(cfg.get("protected_unit")) == "user" else 0,
        abs((eps or 10**9) - 2.0),
        seed if seed is not None else 10**9,
    )



def _run_preference_score(run: Path, cfg: dict[str, Any], spec: dict[str, Any]) -> tuple:
    """Rank duplicate completed runs for one logical condition.
    
    The ordering prefers the requested protected-unit count, then current directory
    naming over legacy rerun names. Ties are resolved deterministically.
    """
    name = run.name.lower()
    n = _cfg_n_records(cfg)
    u = _cfg_n_users(cfg)
    target_n = _safe_int(spec.get("target_n"))
    target_u = _safe_int(spec.get("target_u"))

    if target_n is not None and n is not None:
        n_score = abs(n - target_n)
    elif n is not None:
        n_score = n
    else:
        n_score = 10**12

    if target_u is not None and u is not None:
        u_score = abs(u - target_u)
    elif u is not None:
        u_score = u
    else:
        u_score = 10**12

    # Prefer original main folders over later ad-hoc variants when all formal
    # parameters are tied.  If only a legacy variant exists, it is still selected.
    v2_penalty = 1 if "v2" in name else 0
    deleff_penalty = 1 if "deleff" in name else 0
    probe_penalty = 1 if "probe" in name else 0
    return (probe_penalty, n_score, u_score, v2_penalty, deleff_penalty, str(run))


def _deduplicate_candidates(
    candidates: list[tuple[Path, dict[str, Any]]],
    spec: dict[str, Any],
    *,
    keep_record_counts: bool,
    keep_user_counts: bool,
) -> list[tuple[Path, dict[str, Any]]]:
    """Keep one completed run per logical representative condition.
    
    When explicit protected-unit filters are supplied, the count remains part of the
    condition key so multiple sizes can be selected. Otherwise duplicate saved runs
    are collapsed deterministically.
    """
    best: dict[tuple, tuple[tuple, Path, dict[str, Any]]] = {}
    for run, cfg in candidates:
        unit = _run_unit(cfg, run)
        key = [
            unit,
            str(cfg.get("mechanism")),
            _safe_float(cfg.get("epsilon")),
            _safe_int(cfg.get("seed")),
            _safe_int(cfg.get("m")),
            _safe_int(cfg.get("K")),
            _safe_int(cfg.get("L")),
        ]
        if unit == "user":
            key.append(_safe_float(cfg.get("user_contribution_bound")))
            if keep_user_counts:
                key.append(_cfg_n_users(cfg))
        elif keep_record_counts:
            key.append(_cfg_n_records(cfg))
        key_t = tuple(key)
        score = _run_preference_score(run, cfg, spec)
        old = best.get(key_t)
        if old is None or score < old[0]:
            best[key_t] = (score, run, cfg)

    selected = [(run, cfg) for _score, run, cfg in best.values()]
    selected.sort(
        key=lambda rc: (
            _run_unit(rc[1], rc[0]),
            str(rc[1].get("mechanism")),
            _safe_float(rc[1].get("epsilon")) or 0,
            _safe_int(rc[1].get("seed")) or 0,
            _cfg_n_records(rc[1]) or 0,
            _cfg_n_users(rc[1]) or 0,
            str(rc[0]),
        )
    )
    return selected


def select_runs(
    dataset: str,
    mechanisms: list[str],
    epsilons: list[float],
    seeds: list[int],
    units: list[str] | None = None,
    m_values: list[int] | None = None,
    K: int | None = DEFAULT_K,
    L: int | None = DEFAULT_L,
    B_values: list[float] | None = None,
    n_records_values: list[int] | None = None,
    n_users_values: list[int] | None = None,
    max_runs: int | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    """Select completed runs that match the requested representative conditions."""
    spec = DEFAULT_DATASETS[dataset]
    unit_set = set(units or spec["units"])
    mech_set = set(mechanisms)
    eps_set = {float(e) for e in epsilons}
    seed_set = {int(s) for s in seeds}
    m_set = {int(m) for m in m_values} if m_values else ({int(spec["target_m"])} if spec.get("target_m") else None)
    B_set = {float(b) for b in B_values} if B_values else ({float(spec["target_B"])} if spec.get("target_B") is not None else None)
    n_records_set = {int(n) for n in n_records_values} if n_records_values else None
    n_users_set = {int(u) for u in n_users_values} if n_users_values else None
    candidates = []
    for run, cfg in _iter_runs(spec["run_dirs"]):
        if _matches(run, cfg, unit_set, mech_set, eps_set, seed_set, m_set, K, L, B_set, n_records_set, n_users_set):
            candidates.append((run, cfg))

    candidates = _deduplicate_candidates(
        candidates,
        spec,
        keep_record_counts=n_records_set is not None,
        keep_user_counts=n_users_set is not None,
    )
    if max_runs is not None:
        candidates = candidates[: int(max_runs)]
    return candidates


def _ensure_clean_reference_plan(src_run: Path, dst_run: Path, cfg: dict[str, Any], private_plan: dict[str, Any]) -> dict[str, Any]:
    dst_plan = dst_run / "clean_reference_plan.json"
    if dst_plan.exists():
        return read_json(dst_plan)
    src_plan = src_run / "clean_reference_plan.json"
    if src_plan.exists():
        shutil.copy2(src_plan, dst_plan)
        return read_json(dst_plan)
    clean_sketch = src_run / "clean_sketch.npy"
    atoms_path = src_run / "atoms_used.json"
    if not clean_sketch.exists() or not atoms_path.exists():
        raise FileNotFoundError(
            f"Cannot build non-private reference plan for {src_run}; missing clean_sketch.npy or atoms_used.json"
        )
    C = np.load(clean_sketch)
    atoms = read_json(atoms_path)
    L = int(cfg.get("L", private_plan.get("L", len(private_plan.get("selected_atoms", [])))))
    public_denominator = private_plan.get("public_denominator", cfg.get("public_denominator"))
    bins = cfg.get("support_bins")
    clean_plan = build_clean_reference_plan(C, atoms, L=L, public_denominator=public_denominator, bins=bins)
    write_json(clean_plan, dst_plan)
    return clean_plan


def _copy_run(src_run: Path, dst_run: Path) -> None:
    ensure_dir(dst_run)
    for name in COPY_FILES:
        src = src_run / name
        if src.exists():
            shutil.copy2(src, dst_run / name)


def _decode_one(
    src_run: Path,
    out_root: Path,
    dataset: str,
    model: str,
    temperature: float,
    max_tokens: int,
    force: bool,
    judge_backend: str,
    comparative_backend: str,
    summary_embedding_backend: str,
) -> dict[str, Any]:
    """Decode one saved DP-SPIN release in audit and display modes.
    
    The audit summary retains canonical atom markers and released numeric values for
    verifier checks. The display summary omits those implementation details and is
    used for qualitative examples and text-level comparison.
    """
    cfg = read_json(src_run / "config.json")
    unit = _run_unit(cfg, src_run)
    dst_run = ensure_dir(out_root / unit / src_run.name)
    _copy_run(src_run, dst_run)

    private_plan = read_json(dst_run / "private_plan.json")
    clean_plan = _ensure_clean_reference_plan(src_run, dst_run, cfg, private_plan)
    atoms = read_json(dst_run / "atoms_used.json") if (dst_run / "atoms_used.json").exists() else []

    # ------------------------------------------------------------------
    # DP-SPIN private-plan decoding: audit mode.
    # Backward-compatible copies summary_openai.txt and decoder_prompt_openai.txt
    # point to the audit version because the verifier expects atom markers.
    # ------------------------------------------------------------------
    dp_audit_summary_path = dst_run / "summary_openai_audit.txt"
    dp_audit_prompt_path = dst_run / "decoder_prompt_openai_audit.txt"
    if force or not dp_audit_summary_path.exists():
        dp_audit_summary, dp_audit_prompt = decode_plan(
            private_plan,
            backend="openai",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            mode="audit",
        )
        dp_audit_summary_path.write_text(dp_audit_summary, encoding="utf-8")
        dp_audit_prompt_path.write_text(dp_audit_prompt, encoding="utf-8")
    else:
        dp_audit_summary = dp_audit_summary_path.read_text(encoding="utf-8")
    shutil.copy2(dp_audit_summary_path, dst_run / "summary_openai.txt")
    shutil.copy2(dp_audit_prompt_path, dst_run / "decoder_prompt_openai.txt")

    # ------------------------------------------------------------------
    # DP-SPIN private-plan decoding: display mode.
    # This version is used in paper qualitative examples and text metrics.
    # ------------------------------------------------------------------
    dp_display_summary_path = dst_run / "summary_openai_display.txt"
    dp_display_prompt_path = dst_run / "decoder_prompt_openai_display.txt"
    if force or not dp_display_summary_path.exists():
        dp_display_summary, dp_display_prompt = decode_plan(
            private_plan,
            backend="openai",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            mode="display",
        )
        dp_display_summary_path.write_text(dp_display_summary, encoding="utf-8")
        dp_display_prompt_path.write_text(dp_display_prompt, encoding="utf-8")
    else:
        dp_display_summary = dp_display_summary_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Non-private-reference plan decoding in both modes. The reference is
    # evaluation-only; it is not part of the DP release.
    # ------------------------------------------------------------------
    clean_audit_summary_path = dst_run / "clean_reference_summary_openai_audit.txt"
    clean_audit_prompt_path = dst_run / "clean_reference_decoder_prompt_openai_audit.txt"
    if force or not clean_audit_summary_path.exists():
        clean_audit_summary, clean_audit_prompt = decode_plan(
            clean_plan,
            backend="openai",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            mode="audit",
        )
        clean_audit_summary_path.write_text(clean_audit_summary, encoding="utf-8")
        clean_audit_prompt_path.write_text(clean_audit_prompt, encoding="utf-8")
    else:
        clean_audit_summary = clean_audit_summary_path.read_text(encoding="utf-8")
    shutil.copy2(clean_audit_summary_path, dst_run / "clean_reference_summary_openai.txt")
    shutil.copy2(clean_audit_prompt_path, dst_run / "clean_reference_decoder_prompt_openai.txt")

    clean_display_summary_path = dst_run / "clean_reference_summary_openai_display.txt"
    clean_display_prompt_path = dst_run / "clean_reference_decoder_prompt_openai_display.txt"
    if force or not clean_display_summary_path.exists():
        clean_display_summary, clean_display_prompt = decode_plan(
            clean_plan,
            backend="openai",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            mode="display",
        )
        clean_display_summary_path.write_text(clean_display_summary, encoding="utf-8")
        clean_display_prompt_path.write_text(clean_display_prompt, encoding="utf-8")
    else:
        clean_display_summary = clean_display_summary_path.read_text(encoding="utf-8")

    # Verifier is run on the audit summary because it contains canonical markers.
    verifier = verify_summary(
        dp_audit_summary,
        private_plan,
        atoms,
        strict_markers=bool(cfg.get("verifier_strict_markers", True)),
        comparison_margin=float(cfg.get("verifier_comparison_margin", 0.0)),
        reject_raw_details=bool(cfg.get("verifier_reject_raw_details", True)),
        reject_unsupported_numbers=bool(cfg.get("verifier_reject_unsupported_numbers", True)),
    )
    write_json(verifier, dst_run / "verifier_openai.json")

    # Optional legacy DP-SPIN-only judge. The paper pipeline uses the common
    # release-conditioned judge in src.openai_paper_pipeline instead.  We keep
    # this optional output under an explicit filename so it is not confused with
    # the cross-method OpenAI judgments.
    judge = None
    if judge_backend != "none":
        judge = llm_judge(dp_display_summary, private_plan, model=model) if judge_backend == "openai" else heuristic_judge(dp_display_summary, private_plan)
        judge_name = "judge_plan_conditioned_openai.json" if judge_backend == "openai" else "judge_plan_conditioned_heuristic.json"
        write_json(judge, dst_run / judge_name)

    # Text metrics compare display-mode DP summaries with offline non-private references.
    text_eval = compute_summary_text_evaluation(
        dp_display_summary,
        clean_display_summary,
        clean_plan,
        embedding_backend=summary_embedding_backend,
        comparative_backend=comparative_backend,
        comparative_model=model,
    )
    write_json(text_eval, dst_run / "summary_text_eval_openai_display.json")
    # Backward-compatible filename points to display-based evaluation.
    write_json(text_eval, dst_run / "summary_text_eval_openai.json")

    row: dict[str, Any] = {
        "dataset": dataset,
        "protected_unit": unit,
        "source_run": str(src_run),
        "openai_run": str(dst_run),
        "run": src_run.name,
        "model": model,
        "mechanism": cfg.get("mechanism"),
        "epsilon": cfg.get("epsilon"),
        "delta": cfg.get("delta"),
        "seed": cfg.get("seed"),
        "m": cfg.get("m"),
        "K": cfg.get("K"),
        "L": cfg.get("L"),
        "n_records_effective": cfg.get("n_records_effective"),
        "n_users_effective": cfg.get("n_users_effective"),
        "user_contribution_bound": cfg.get("user_contribution_bound"),
        "verifier_accept": verifier.get("accept"),
        "UAR": verifier.get("UAR"),
        "UCR": verifier.get("UCR"),
        "comparison_count": verifier.get("comparison_count"),
        "supported_comparison_count": verifier.get("supported_comparison_count"),
        "unsupported_comparison_count": verifier.get("unsupported_comparison_count"),
        "summary_path": str(dp_display_summary_path),
        "summary_display_path": str(dp_display_summary_path),
        "summary_audit_path": str(dp_audit_summary_path),
        "clean_reference_summary_path": str(clean_display_summary_path),
        "clean_reference_summary_display_path": str(clean_display_summary_path),
        "clean_reference_summary_audit_path": str(clean_audit_summary_path),
        "plan_path": str(dst_run / "private_plan.json"),
        "clean_reference_plan_path": str(dst_run / "clean_reference_plan.json"),
        "verifier_summary_path": str(dp_audit_summary_path),
    }
    if judge:
        for key in ["coverage", "faithfulness", "usefulness", "clarity"]:
            if key in judge:
                row[f"judge_{key}"] = judge[key]
    for key, val in (text_eval.get("lexical") or {}).items():
        row[f"text_{key}"] = val
    for key, val in (text_eval.get("embedding") or {}).items():
        row[f"embedding_{key}"] = val
    comp = text_eval.get("comparative") or {}
    if comp.get("enabled"):
        row["comparative_choice"] = comp.get("choice")
        row["comparative_backend"] = comp.get("backend")
    return row

def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = sorted({k for r in rows for k in r})
    preferred = [
        "dataset", "protected_unit", "mechanism", "epsilon", "seed", "m", "K", "L",
        "user_contribution_bound", "n_records_effective", "n_users_effective",
        "verifier_accept", "UAR", "UCR", "comparison_count",
        "supported_comparison_count", "unsupported_comparison_count",
        "summary_path", "clean_reference_summary_path",
    ]
    cols = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _short(s: str, n: int = 1200) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def _write_markdown(rows: list[dict[str, Any]], out_root: Path) -> None:
    lines = [
        "# OpenAI representative DP-SPIN summaries",
        "",
        "The OpenAI decoder receives only the released DP plan and public instructions. It does not receive raw protected records.",
        "",
    ]
    for r in rows:
        p = Path(str(r.get("summary_path", "")))
        cp = Path(str(r.get("clean_reference_summary_path", "")))
        lines.extend([
            f"## {r.get('dataset')} / {r.get('protected_unit')} / {r.get('mechanism')} / eps={r.get('epsilon')} / seed={r.get('seed')}",
            "",
            f"Run: `{r.get('run')}`",
            "",
            "### DP-SPIN OpenAI summary",
            "",
            _short(p.read_text(encoding="utf-8") if p.exists() else ""),
            "",
            "### Non-private-plan OpenAI reference summary",
            "",
            _short(cp.read_text(encoding="utf-8") if cp.exists() else ""),
            "",
        ])
    (out_root / "qualitative_examples.md").write_text("\n".join(lines), encoding="utf-8")


def run_openai_decoding(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Decode selected saved releases and write verifier and text-evaluation artifacts."""
    datasets = list(DEFAULT_DATASETS) if args.dataset == "all" else [args.dataset]
    rows: list[dict[str, Any]] = []
    for ds in datasets:
        spec = DEFAULT_DATASETS[ds]
        out_root = Path(args.output_dir) if args.output_dir else Path(spec["output_dir"])
        selected = select_runs(
            ds,
            mechanisms=args.mechanisms,
            epsilons=args.epsilons,
            seeds=args.seeds,
            units=args.units,
            m_values=args.m_values,
            K=args.K,
            L=args.L,
            B_values=args.user_contribution_bounds,
            max_runs=args.max_runs_per_dataset,
        )
        if not selected:
            print(f"No matching runs found for dataset={ds}.")
            continue
        print(f"Selected {len(selected)} run(s) for {ds}.")
        for run, cfg in selected:
            print(f"Decoding {ds}: {run}")
            rows.append(_decode_one(
                run,
                out_root,
                dataset=spec["dataset_name"],
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                force=args.force,
                judge_backend=args.judge_backend,
                comparative_backend=args.comparative_backend,
                summary_embedding_backend=args.summary_embedding_backend,
            ))
        _write_csv(rows, out_root / "tables" / "openai_representative_runs.csv")
        _write_markdown(rows, out_root)
        write_json({"rows": rows}, out_root / "openai_representative_index.json")
    return rows


def main() -> None:
    """Run representative OpenAI decoding from the command line."""
    p = argparse.ArgumentParser(description="Decode representative existing DP-SPIN runs with OpenAI without rerunning DP mechanisms.")
    p.add_argument("--dataset", choices=["all", *DEFAULT_DATASETS.keys()], default="all")
    p.add_argument("--output-dir", default=None, help="Override output directory. For --dataset all, per-dataset defaults are used unless this is set.")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--mechanisms", nargs="+", default=DEFAULT_MECHANISMS)
    p.add_argument("--epsilons", nargs="+", type=float, default=DEFAULT_EPSILONS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--units", nargs="+", choices=["record", "user"], default=None)
    p.add_argument("--m-values", nargs="+", type=int, default=None)
    p.add_argument("--K", type=int, default=DEFAULT_K)
    p.add_argument("--L", type=int, default=DEFAULT_L)
    p.add_argument("--user-contribution-bounds", nargs="+", type=float, default=None)
    p.add_argument("--max-runs-per-dataset", type=int, default=None)
    p.add_argument("--judge-backend", choices=["none", "heuristic", "openai"], default="none", help="Optional legacy plan-conditioned DP-SPIN judge; paper comparisons use src.openai_paper_pipeline.")
    p.add_argument("--comparative-backend", choices=["none", "heuristic", "openai"], default="none")
    p.add_argument("--summary-embedding-backend", choices=["none", "tfidf", "auto", "sentence_transformer"], default="tfidf")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    rows = run_openai_decoding(args)
    print(f"Wrote OpenAI representative artifacts for {len(rows)} run(s).")


if __name__ == "__main__":
    main()
