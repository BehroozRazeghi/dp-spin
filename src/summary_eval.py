"""Evaluate generated text against an offline non-private reference plan.

The lexical, embedding, and optional comparative metrics are evaluation artifacts. 
The non-private reference plan and its verbalization are created inside the trusted evaluation workflow and are never supplied to the DP decoder or included in the released DP-SPIN output.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .decode import decode_plan
from .plan import make_private_plan
from .utils import read_json, write_json
from .llm_client import maybe_llm

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
_STOP = set(ENGLISH_STOP_WORDS) | {
    "atom", "atoms", "record", "records", "review", "reviews", "text", "theme", "themes",
    "summary", "plan", "aggregate", "aggregates", "differentially", "private", "released",
}


def _tokens(text: str) -> list[str]:
    """Return normalized content tokens for lexical comparison."""
    toks = [m.group(0).lower().strip("'-_") for m in _WORD_RE.finditer(text or "")]
    return [t for t in toks if len(t) >= 2 and t not in _STOP]


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _candidate_keyphrases(text: str, max_n: int = 3) -> set[str]:
    """Extract deterministic lexical key phrases from a generated summary.
    
    This dependency-free extractor is a proxy used for local text comparison; it is
    not the keyphrase-selection procedure used by the URANIA-style baseline.
    """
    toks = _tokens(text)
    phrases: set[str] = set()
    for n in range(1, max_n + 1):
        for ng in _ngrams(toks, n):
            if all(t in _STOP for t in ng):
                continue
            # Avoid isolated numeric-like or too-generic one-word phrases.
            if n == 1 and (len(ng[0]) < 4 or ng[0] in _STOP):
                continue
            phrases.add(" ".join(ng))
    return phrases


def _jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return float(len(a & b) / max(1, len(union)))


def _tfidf_cosine(a: str, b: str) -> float:
    try:
        vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform([a or "", b or ""])
        if X.shape[1] == 0:
            return 0.0
        return float(cosine_similarity(X[0], X[1])[0, 0])
    except Exception:
        return float("nan")


def _sentence_transformer_cosine(a: str, b: str, model_name: str) -> tuple[float, str, str | None]:
    """Compute sentence-embedding cosine when sentence-transformers is installed."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer(model_name)
        E = model.encode([a or "", b or ""], normalize_embeddings=True)
        return float(np.dot(E[0], E[1])), "sentence_transformer", model_name
    except Exception as exc:
        return float("nan"), "sentence_transformer_unavailable", str(exc)[:300]


def lexical_content_metrics(dp_summary: str, clean_summary: str) -> dict[str, Any]:
    """Compute deterministic lexical preservation metrics."""
    dp_tokens = _tokens(dp_summary)
    clean_tokens = _tokens(clean_summary)
    dp_token_set = set(dp_tokens)
    clean_token_set = set(clean_tokens)
    out: dict[str, Any] = {
        "token_jaccard": _jaccard(dp_token_set, clean_token_set),
        "bigram_jaccard": _jaccard(_ngrams(dp_tokens, 2), _ngrams(clean_tokens, 2)),
        "trigram_jaccard": _jaccard(_ngrams(dp_tokens, 3), _ngrams(clean_tokens, 3)),
        "keyphrase_jaccard": _jaccard(_candidate_keyphrases(dp_summary), _candidate_keyphrases(clean_summary)),
        "tfidf_cosine": _tfidf_cosine(dp_summary, clean_summary),
        "dp_summary_num_tokens": len(dp_tokens),
        "clean_summary_num_tokens": len(clean_tokens),
    }
    return out


def embedding_similarity_metrics(
    dp_summary: str,
    clean_summary: str,
    backend: str = "tfidf",
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> dict[str, Any]:
    """Compute summary-level semantic similarity.

    ``backend='sentence_transformer'`` uses a SentenceTransformer if available.
    ``backend='tfidf'`` is dependency-light and deterministic.  ``backend='auto'``
    attempts SentenceTransformer first and falls back to TF-IDF.
    """
    backend = str(backend or "tfidf").lower()
    if backend in {"sentence_transformer", "sentence-transformers", "sbert", "auto"}:
        val, used, err = _sentence_transformer_cosine(dp_summary, clean_summary, model_name)
        if math.isfinite(val):
            return {"summary_embedding_cosine": val, "summary_embedding_backend": used, "summary_embedding_model": model_name}
        if backend != "auto":
            return {"summary_embedding_cosine": val, "summary_embedding_backend": used, "summary_embedding_model": model_name, "summary_embedding_error": err}
    # default/fallback
    return {
        "summary_embedding_cosine": _tfidf_cosine(dp_summary, clean_summary),
        "summary_embedding_backend": "tfidf_fallback" if backend == "auto" else "tfidf",
        "summary_embedding_model": "sklearn_tfidf_1_2gram",
    }


def _atom_lines(plan: dict) -> list[str]:
    lines = []
    for a in plan.get("selected_atoms", []):
        lines.append(
            f"[atom:{a.get('atom_id')}] label={a.get('label')} | "
            f"description={a.get('description')} | mass={float(a.get('noisy_mass', 0.0)):.3f} | "
            f"bin={a.get('prevalence_bin')}"
        )
    return lines


def llm_comparative_prompt(clean_plan: dict, dp_summary: str, clean_summary: str) -> str:
    """Build a prompt that compares summaries using only the non-private aggregate plan."""
    return "\n".join([
        "You are evaluating two aggregate summaries against a non-private aggregate semantic plan.",
        "Use only the non-private aggregate plan as the reference. Do not assume access to raw records.",
        "Decide which summary better captures the non-private aggregate plan.",
        "Criteria: coverage of non-private admitted atoms, faithfulness to the non-private reference plan, clarity, and concision.",
        "Return only valid JSON with keys: choice, score_DP, score_NP, rationale.",
        "choice must be one of: DP, NP, tie.",
        "score_DP and score_NP must be numbers from 1 to 5.",
        "",
        "Non-private aggregate plan:",
        *(_atom_lines(clean_plan)),
        "",
        "DP summary:",
        dp_summary,
        "",
        "Non-private reference summary:",
        clean_summary,
    ])


def heuristic_comparative_eval(clean_plan: dict, dp_summary: str, clean_summary: str) -> dict[str, Any]:
    """Compute the deterministic fallback comparison from public atom-label coverage."""
    labels = [str(a.get("label", "")).lower() for a in clean_plan.get("selected_atoms", [])]
    labels = [l for l in labels if l]

    def cov(text: str) -> float:
        low = text.lower()
        return sum(1 for l in labels if l in low) / max(1, len(labels))

    dp_cov = cov(dp_summary)
    clean_cov = cov(clean_summary)
    # Convert coverage to a 1--5 utility-like score.
    dp_score = 1.0 + 4.0 * dp_cov
    clean_score = 1.0 + 4.0 * clean_cov
    if abs(dp_score - clean_score) < 0.25:
        choice = "tie"
    else:
        choice = "DP" if dp_score > clean_score else "NP"
    return {
        "enabled": True,
        "backend": "heuristic_non_private_plan_comparison",
        "choice": choice,
        "score_dp_summary": float(dp_score),
        "score_np_reference_summary": float(clean_score),
        "score_A_dp_summary": float(dp_score),
        "score_B_clean_summary": float(clean_score),
        "dp_non_private_plan_atom_label_coverage": float(dp_cov),
        "np_non_private_plan_atom_label_coverage": float(clean_cov),
        "rationale": "Heuristic comparison based on non-private-reference-plan atom-label coverage; no raw records used.",
    }


def llm_comparative_eval(
    clean_plan: dict,
    dp_summary: str,
    clean_summary: str,
    backend: str = "none",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> dict[str, Any]:
    """Run optional LLM comparative evaluation against the non-private aggregate plan."""
    backend = str(backend or "none").lower()
    if backend in {"none", "off", "disabled", "false"}:
        return {"enabled": False, "backend": backend, "reason": "disabled"}
    if backend in {"heuristic", "local"}:
        return heuristic_comparative_eval(clean_plan, dp_summary, clean_summary)
    prompt = llm_comparative_prompt(clean_plan, dp_summary, clean_summary)
    try:
        text = maybe_llm(prompt, backend=backend, model=model, temperature=temperature, max_tokens=max_tokens)
        if text is None:
            return {"enabled": False, "backend": backend, "reason": "no_llm_response"}
        m = re.search(r"\{.*\}", text, re.S)
        parsed = json.loads(m.group(0) if m else text)
        raw_choice = str(parsed.get("choice", "")).strip()
        choice_map = {"A": "DP", "B": "NP", "dp": "DP", "np": "NP", "non-private": "NP", "non_private": "NP", "tie": "tie"}
        choice = choice_map.get(raw_choice, choice_map.get(raw_choice.lower(), raw_choice))
        score_dp = parsed.get("score_DP", parsed.get("score_A"))
        score_np = parsed.get("score_NP", parsed.get("score_B"))
        return {
            "enabled": True,
            "backend": backend,
            "model": model,
            "choice": choice,
            "raw_choice": raw_choice,
            "score_dp_summary": score_dp,
            "score_np_reference_summary": score_np,
            "score_A_dp_summary": score_dp,
            "score_B_clean_summary": score_np,
            "rationale": parsed.get("rationale"),
            "raw_response": text,
            "prompt_reference": "non_private_aggregate_plan_only_no_raw_records",
        }
    except Exception as exc:
        return {"enabled": False, "backend": backend, "model": model, "error": str(exc)[:500]}


def build_clean_reference_plan(
    C: np.ndarray,
    atoms: list[dict[str, Any]],
    L: int,
    public_denominator: int | None,
    bins: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Construct an offline non-private top-``L`` reference plan for evaluation."""
    plan = make_private_plan(C, atoms, L=L, public_denominator=public_denominator, bins=bins)
    plan["reference_type"] = "non_private_top_L_plan"
    plan["evaluation_only"] = True
    plan["privacy_note"] = "This non-private reference plan is for offline evaluation only and is not released by the differentially private mechanism."
    return plan


def compute_summary_text_evaluation(
    dp_summary: str,
    clean_summary: str,
    clean_plan: dict,
    embedding_backend: str = "tfidf",
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    comparative_backend: str = "none",
    comparative_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Compute lexical, embedding, and optional comparative text evaluation."""
    lexical = lexical_content_metrics(dp_summary, clean_summary)
    emb = embedding_similarity_metrics(dp_summary, clean_summary, backend=embedding_backend, model_name=embedding_model)
    comp = llm_comparative_eval(
        clean_plan,
        dp_summary,
        clean_summary,
        backend=comparative_backend,
        model=comparative_model,
    )
    return {
        "reference": "non_private_top_L_semantic_plan_and_non_private_reference_summary",
        "evaluation_only": True,
        "uses_raw_records": False,
        "lexical": lexical,
        "embedding": emb,
        "comparative": comp,
    }


def evaluate_run_dir(run_dir: str | Path, overwrite: bool = True) -> dict[str, Any]:
    """Backfill text evaluation for an existing run directory.

    The run must contain clean_sketch.npy, atoms_used.json, config.json,
    private_plan.json, and summary.txt.  The function writes non-private reference
    artifacts and updates metrics.json with a ``text_eval`` block.
    """
    rd = Path(run_dir)
    metrics_path = rd / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    cfg = read_json(rd / "config.json") if (rd / "config.json").exists() else {}
    atoms = read_json(rd / "atoms_used.json")
    C = np.load(rd / "clean_sketch.npy")
    dp_summary = (rd / "summary.txt").read_text(encoding="utf-8")
    private_plan = read_json(rd / "private_plan.json")
    L = int(cfg.get("L", private_plan.get("L", len(private_plan.get("selected_atoms", [])))))
    public_denominator = private_plan.get("public_denominator", cfg.get("public_denominator"))
    bins = cfg.get("prevalence_bins")
    clean_plan_path = rd / "clean_reference_plan.json"
    clean_summary_path = rd / "clean_reference_summary.txt"
    if clean_plan_path.exists() and not overwrite:
        clean_plan = read_json(clean_plan_path)
    else:
        clean_plan = build_clean_reference_plan(C, atoms, L=L, public_denominator=public_denominator, bins=bins)
        write_json(clean_plan, clean_plan_path)
    if clean_summary_path.exists() and not overwrite:
        clean_summary = clean_summary_path.read_text(encoding="utf-8")
    else:
        clean_summary, clean_prompt = decode_plan(
            clean_plan,
            backend=str(cfg.get("clean_reference_decoder_backend", "template")),
            model=str(cfg.get("clean_reference_decoder_model", cfg.get("decoder_model", "gpt-4o-mini"))),
            temperature=float(cfg.get("clean_reference_decoder_temperature", 0.0)),
            max_tokens=int(cfg.get("clean_reference_decoder_max_tokens", cfg.get("decoder_max_tokens", 700))),
        )
        clean_summary_path.write_text(clean_summary, encoding="utf-8")
        (rd / "clean_reference_decoder_prompt.txt").write_text(clean_prompt, encoding="utf-8")
    text_eval = compute_summary_text_evaluation(
        dp_summary,
        clean_summary,
        clean_plan,
        embedding_backend=str(cfg.get("summary_embedding_backend", "tfidf")),
        embedding_model=str(cfg.get("summary_embedding_model", "sentence-transformers/all-mpnet-base-v2")),
        comparative_backend=str(cfg.get("summary_comparative_eval_backend", "none")),
        comparative_model=str(cfg.get("summary_comparative_eval_model", cfg.get("decoder_model", "gpt-4o-mini"))),
    )
    write_json(text_eval, rd / "summary_text_eval.json")
    metrics = read_json(metrics_path)
    metrics["text_eval"] = text_eval
    write_json(metrics, metrics_path)
    return text_eval


def main() -> None:
    """Evaluate generated text for one completed run from the command line."""
    p = argparse.ArgumentParser(description="Compute DP-vs-non-private-reference summary text metrics for existing runs.")
    p.add_argument("--run-dir", default=None, help="Single run directory to evaluate.")
    p.add_argument("--runs-dir", default=None, help="Directory containing run subdirectories.")
    p.add_argument("--no-overwrite", action="store_true")
    args = p.parse_args()
    if not args.run_dir and not args.runs_dir:
        raise SystemExit("Provide --run-dir or --runs-dir")
    targets = []
    if args.run_dir:
        targets.append(Path(args.run_dir))
    if args.runs_dir:
        targets.extend(sorted(p.parent for p in Path(args.runs_dir).glob("*/metrics.json")))
    count = 0
    for rd in targets:
        evaluate_run_dir(rd, overwrite=not args.no_overwrite)
        count += 1
    print(f"Wrote text-evaluation artifacts for {count} run(s).")


if __name__ == "__main__":
    main()
