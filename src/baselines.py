"""Generate non-private references and fixed-universe DP histogram baselines.

The raw-record, heuristic-redaction, and target-derived clustering methods are non-private utility references. 
The keyword and category histogram methods are DP baselines only when their bin universes are fixed from public data or an auxiliary split before the protected target split is processed.

No decoder-visible keyword or category label may be discovered non-privately from the protected target records in a run reported as differentially private.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans

from .utils import read_jsonl, read_json, write_json, ensure_dir, text_field
from .dp_mechanisms import laplace_release, topL, sensitivity
from .decode import template_decode, decode_plan
from .embed import TfidfEmbedder



BASELINE_METHOD_CHOICES = (
    "raw",
    "redacted",
    "non_private_clustering",
    "dp_keyword",
    "dp_category",
)


def _normalize_methods(methods: Iterable[str] | None) -> set[str]:
    if methods is None:
        return set(BASELINE_METHOD_CHOICES)
    out = {str(m).strip() for m in methods if str(m).strip()}
    invalid = sorted(out.difference(BASELINE_METHOD_CHOICES))
    if invalid:
        raise ValueError(f"Unknown baseline method(s): {invalid}. Valid choices are {BASELINE_METHOD_CHOICES}.")
    return out

STOPWORDS = {
    "the", "and", "with", "that", "this", "from", "have", "was", "were", "for",
    "about", "into", "they", "their", "there", "then", "when", "what", "which",
    "you", "your", "records", "record", "associated", "complaint", "complaints", "consumer",
}


def _clean_string(value: object) -> str:
    """Normalize metadata or label text for public-vocabulary construction."""
    s = str(value).strip()
    return "" if s.lower() in {"", "nan", "none", "null"} else s


def _terms(texts: list[str], k: int = 12) -> list[str]:
    """Extract frequent non-stopword terms for the deterministic template baseline."""
    words = Counter()
    for t in texts:
        for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", t.lower()):
            if w not in STOPWORDS:
                words[w] += 1
    return [w for w, _ in words.most_common(k)]


def _choose_records(records: list[dict], max_records: int, seed: int) -> list[dict]:
    """Select a seeded, bounded subset for record-text baselines.
    
    The subset size limits prompt length and API cost. Seeded random sampling avoids
    dependence on source-file ordering. The resulting raw and redacted baselines
    remain non-private.
    """
    if max_records <= 0 or len(records) <= max_records:
        return list(records)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(records), size=max_records, replace=False))
    return [records[int(i)] for i in idx]


def _record_baseline_prompt(texts: list[str], redacted: bool = False) -> str:
    """Build the OpenAI prompt for non-private record-text baselines."""
    header = (
        "You are given a bounded sample of heuristically redacted records."
        if redacted else
        "You are given a bounded sample of records."
    )
    record_header = "Redacted records:" if redacted else "Records:"
    return "\n".join([
        header,
        "",
        "Write a concise aggregate summary of recurring themes in the records.",
        "",
        "Rules:",
        "1. Summarize only aggregate themes.",
        "2. Do not quote records.",
        "3. Do not include names, identifiers, addresses, dates, or other record-specific details.",
        "4. Do not describe individual users, reviewers, businesses, or organizations.",
        "5. Use one concise paragraph, optionally followed by at most three short bullets.",
        "",
        record_header,
        "\n---\n".join(texts),
    ])


def raw_summary(
    records: list[dict],
    backend: str = "template",
    model: str = "gpt-4o-mini",
    max_records: int = 80,
    seed: int = 0,
) -> str:
    """Summarize sampled target records as an explicitly non-private reference."""
    selected = _choose_records(records, max_records=max_records, seed=seed)
    texts = [text_field(r) for r in selected]
    if backend == "openai":
        from .llm_client import openai_chat

        prompt = _record_baseline_prompt(texts, redacted=False)
        return openai_chat(prompt, model=model, temperature=0.0, max_tokens=800)
    return (
        "Raw baseline summary. Frequent terms in the sampled raw records include: "
        + ", ".join(_terms(texts, 12))
        + "."
    )


def redact_text(t: str) -> str:
    """Apply pattern-based redaction for selected identifier formats.
    
    This heuristic is not a privacy mechanism. It can miss sensitive strings and
    remove non-sensitive text.
    """
    t = re.sub(r"\b[\w\.-]+@[\w\.-]+\b", "[EMAIL]", t)
    t = re.sub(r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b", "[SSN]", t)
    t = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", t)
    t = re.sub(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", "[NAME]", t)
    t = re.sub(r"\b[A-Z0-9]{4,}[-][A-Z0-9-]{2,}\b", "[CODE]", t)
    return t


def heuristic_redacted_summary(
    records: list[dict],
    backend: str = "template",
    model: str = "gpt-4o-mini",
    max_records: int = 80,
    seed: int = 0,
) -> str:
    """Run the raw-summary baseline after heuristic string redaction."""
    selected = _choose_records(records, max_records=max_records, seed=seed)
    redacted = [{**r, "text": redact_text(text_field(r))} for r in selected]
    if backend == "openai":
        from .llm_client import openai_chat

        prompt = _record_baseline_prompt([text_field(r) for r in redacted], redacted=True)
        return openai_chat(prompt, model=model, temperature=0.0, max_tokens=800)
    return raw_summary(
        redacted,
        backend=backend,
        model=model,
        max_records=max_records,
        seed=seed,
    ).replace("Raw baseline", "Heuristic-redacted baseline")


def non_private_clustering_summary(
    records: list[dict],
    n_clusters: int = 10,
    backend: str = "template",
    model: str = "gpt-4o-mini",
) -> tuple[dict, str]:
    """Cluster target records and decode cluster labels as a non-private baseline."""
    if not records:
        plan = {"selected_atoms": [], "L": 0, "baseline": "non_private_clustering"}
        return plan, template_decode(plan, mode="display")

    texts = [text_field(r) for r in records]
    embedder = TfidfEmbedder.fit(texts)
    X = embedder.transform(texts)
    k = min(int(n_clusters), len(records))
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)

    atoms = []
    for c in range(k):
        idx = np.where(km.labels_ == c)[0]
        terms = _terms([texts[i] for i in idx[:200]], 5)
        label = " / ".join(terms[:3]) if terms else f"cluster {c}"
        atoms.append(
            {
                "atom_id": c,
                "label": label,
                "description": f"Non-private cluster with {len(idx)} records.",
                "noisy_mass": float(len(idx)),
                "prevalence_bin": None,
            }
        )
    plan = {
        "selected_atoms": sorted(atoms, key=lambda a: -a["noisy_mass"]),
        "L": k,
        "baseline": "non_private_clustering",
    }
    summary, _ = decode_plan(plan, backend=backend, model=model, mode="display")
    return plan, summary


def _atom_component_map(atom: dict) -> dict[str, str]:
    """Recover public metadata components stored in or implied by an atom entry."""
    meta = atom.get("metadata")
    if isinstance(meta, dict) and meta:
        return {str(k): _clean_string(v) for k, v in meta.items() if _clean_string(v)}

    components = atom.get("components")
    if isinstance(components, dict) and components:
        return {str(k): _clean_string(v) for k, v in components.items() if _clean_string(v)}

    columns = atom.get("atom_source_columns") or atom.get("metadata_columns") or []
    label = _clean_string(atom.get("label", ""))
    parts = [_clean_string(p) for p in label.split(" / ")]
    parts = [p for p in parts if p]
    if columns and len(parts) == len(columns):
        return {str(c): p for c, p in zip(columns, parts)}
    return {}


def public_categories_from_atoms(atoms: list[dict], category_key: str) -> list[str]:
    """Extract a fixed public category universe from atom metadata/components."""
    cats = set()
    for atom in atoms:
        comp = _atom_component_map(atom)
        val = _clean_string(comp.get(category_key, ""))
        if val:
            cats.add(val)
    return sorted(cats)


def public_categories_from_records(records: Iterable[dict], category_key: str) -> list[str]:
    """Extract public/auxiliary category values from non-target records."""
    cats = set()
    for r in records:
        val = _clean_string((r.get("metadata") or {}).get(category_key, ""))
        if val:
            cats.add(val)
    return sorted(cats)


def load_category_universe(
    atoms: list[dict],
    category_key: str,
    category_universe_records: str | None = None,
    category_universe_json: str | None = None,
) -> list[str]:
    """Load the fixed public category universe for the DP category baseline."""
    if category_universe_json:
        obj = read_json(category_universe_json)
        if isinstance(obj, dict):
            obj = obj.get(category_key, obj.get("categories", []))
        cats = sorted({_clean_string(c) for c in obj if _clean_string(c)})
        if cats:
            return cats

    if category_universe_records:
        cats = public_categories_from_records(read_jsonl(category_universe_records), category_key)
        if cats:
            return cats

    cats = public_categories_from_atoms(atoms, category_key)
    if cats:
        return cats

    raise ValueError(
        f"Could not construct a public category universe for {category_key!r}. "
        "Pass --category-universe-records using an auxiliary/public split or "
        "--category-universe-json."
    )


def keyword_vocabulary_from_atoms(atoms: list[dict], max_keywords: int = 200) -> list[str]:
    """Build a fixed public keyword vocabulary from public atom labels/descriptions.

    The vocabulary is public because it is derived only from the public atom file.
    We use single-word keywords rather than full atom labels; full labels such as
    ``Product / Issue / Sub-issue`` rarely occur verbatim in complaint narratives
    and lead to a nearly empty histogram.
    """
    counts = Counter()
    for atom in atoms:
        text = f"{atom.get('label', '')} {atom.get('description', '')}"
        for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", text.lower()):
            if w not in STOPWORDS:
                counts[w] += 1
    return [w for w, _ in counts.most_common(max_keywords)]


def _keyword_count(text: str, keyword: str) -> int:
    """Count exact case-insensitive keyword occurrences using word boundaries."""
    kw = re.escape(keyword.lower())
    return len(re.findall(rf"(?<![a-zA-Z]){kw}(?![a-zA-Z])", text.lower()))


def dp_keyword_histogram(
    records: list[dict],
    vocabulary: list[str],
    epsilon: float,
    rng=None,
    adjacency: str = "adddrop",
    L: int = 10,
    backend: str = "template",
    model: str = "gpt-4o-mini",
) -> tuple[dict, str]:
    """Release a noisy histogram over a fixed public keyword vocabulary.

    To keep l1 sensitivity equal to one under add/drop adjacency, each record
    contributes to at most one keyword bin: the public keyword with the largest
    occurrence count in that record, with deterministic public tie-breaking.
    """
    rng = np.random.default_rng() if rng is None else rng
    vocabulary = [_clean_string(v).lower() for v in vocabulary if _clean_string(v)]
    counts = np.zeros(len(vocabulary), dtype=float)

    for r in records:
        txt = text_field(r).lower()
        best_j = None
        best_count = 0
        for j, kw in enumerate(vocabulary):
            c = _keyword_count(txt, kw)
            if c > best_count:
                best_j = j
                best_count = c
        if best_j is not None:
            counts[best_j] += 1.0

    sens = sensitivity(adjacency)
    noisy, info = laplace_release(counts, epsilon=epsilon, l1_sensitivity=sens["l1"], rng=rng)
    info.update(
        {
            "adjacency": adjacency,
            "histogram_universe": "fixed_public_keywords",
            "per_record_contribution": "one_best_matching_keyword",
        }
    )
    idx = topL(noisy, min(int(L), len(vocabulary)))
    plan = {
        "vocabulary_size": int(len(vocabulary)),
        "selected_atoms": [
            {
                "atom_id": int(j),
                "index": int(j),
                "label": vocabulary[int(j)],
                "description": "DP keyword count over a fixed public vocabulary.",
                "noisy_mass": float(max(0.0, noisy[int(j)])),
                "prevalence_bin": None,
            }
            for j in idx
        ],
        "mechanism_info": info,
        "L": int(len(idx)),
        "baseline": "dp_keyword",
    }
    summary, _ = decode_plan(plan, backend=backend, model=model, mode="display")
    return plan, summary


def dp_category_histogram(
    records: list[dict],
    category_key: str,
    epsilon: float,
    categories: list[str],
    rng=None,
    L: int = 10,
    adjacency: str = "adddrop",
    backend: str = "template",
    model: str = "gpt-4o-mini",
) -> tuple[dict, str]:
    """Release a noisy histogram over a fixed public metadata category universe."""
    rng = np.random.default_rng() if rng is None else rng
    cats = sorted({_clean_string(c) for c in categories if _clean_string(c)})
    if not cats:
        raise ValueError("The DP category baseline requires a nonempty public category universe.")

    index = {c: i for i, c in enumerate(cats)}
    counts = np.zeros(len(cats), dtype=float)
    ignored = 0
    for r in records:
        c = _clean_string((r.get("metadata") or {}).get(category_key, ""))
        if c in index:
            counts[index[c]] += 1.0
        else:
            ignored += 1

    sens = sensitivity(adjacency)
    noisy, info = laplace_release(counts, epsilon=epsilon, l1_sensitivity=sens["l1"], rng=rng)
    info.update(
        {
            "adjacency": adjacency,
            "histogram_universe": "fixed_public_or_auxiliary_categories",
            "category_key": category_key,
            "num_categories": len(cats),
            "num_records_ignored_outside_public_universe": int(ignored),
        }
    )

    idx = topL(noisy, min(int(L), len(cats)))
    plan = {
        "category_universe_size": int(len(cats)),
        "selected_atoms": [
            {
                "atom_id": int(j),
                "index": int(j),
                "label": cats[int(j)],
                "description": f"DP category count for public category field {category_key}.",
                "noisy_mass": float(max(0.0, noisy[int(j)])),
                "prevalence_bin": None,
            }
            for j in idx
        ],
        "mechanism_info": info,
        "L": int(len(idx)),
        "baseline": "dp_category",
    }
    summary, _ = decode_plan(plan, backend=backend, model=model, mode="display")
    return plan, summary


def run_baselines(
    records_path: str,
    atoms_path: str,
    output_dir: str,
    epsilon: float = 1.0,
    seed: int = 0,
    backend: str = "template",
    model: str = "gpt-4o-mini",
    category_key: str | None = None,
    adjacency: str = "adddrop",
    category_universe_records: str | None = None,
    category_universe_json: str | None = None,
    max_keywords: int = 200,
    baseline_methods: Iterable[str] | None = None,
) -> None:
    """Generate the requested baseline artifacts in one output directory.
    
    ``baseline_methods`` selects the methods to run and prevents unnecessary API
    calls when only differentially private baselines are needed.
    """
    methods = _normalize_methods(baseline_methods)
    records = read_jsonl(records_path)
    atoms = read_json(atoms_path)
    outdir = ensure_dir(output_dir)

    if "raw" in methods:
        (outdir / "raw_summary.txt").write_text(
            raw_summary(records, backend=backend, model=model, seed=seed), encoding="utf-8"
        )

    if "redacted" in methods:
        (outdir / "heuristic_redacted_summary.txt").write_text(
            heuristic_redacted_summary(records, backend=backend, model=model, seed=seed),
            encoding="utf-8",
        )

    if "non_private_clustering" in methods:
        plan, summary = non_private_clustering_summary(records, backend=backend, model=model)
        write_json(plan, outdir / "non_private_clustering_plan.json")
        (outdir / "non_private_clustering_summary.txt").write_text(summary, encoding="utf-8")

    if "dp_keyword" in methods:
        vocab = keyword_vocabulary_from_atoms(atoms, max_keywords=max_keywords)
        plan, summary = dp_keyword_histogram(
            records,
            vocab,
            epsilon,
            rng=np.random.default_rng(seed),
            adjacency=adjacency,
            L=10,
            backend=backend,
            model=model,
        )
        write_json(plan, outdir / "dp_keyword_plan.json")
        (outdir / "dp_keyword_summary.txt").write_text(summary, encoding="utf-8")

    if category_key and "dp_category" in methods:
        categories = load_category_universe(
            atoms,
            category_key,
            category_universe_records=category_universe_records,
            category_universe_json=category_universe_json,
        )
        plan, summary = dp_category_histogram(
            records,
            category_key,
            epsilon,
            categories,
            rng=np.random.default_rng(seed),
            adjacency=adjacency,
            L=10,
            backend=backend,
            model=model,
        )
        write_json(plan, outdir / "dp_category_plan.json")
        (outdir / "dp_category_summary.txt").write_text(summary, encoding="utf-8")

    metadata = {
        "records_path": records_path,
        "atoms_path": atoms_path,
        "subset_records": len(records),
        "epsilon": epsilon,
        "delta": 0.0,
        "seed": seed,
        "L": 10,
        "backend": backend,
        "model": model,
        "category_key": category_key,
        "adjacency": adjacency,
        "category_universe_records": category_universe_records,
        "category_universe_json": category_universe_json,
        "max_keywords": max_keywords,
        "baseline_methods": sorted(methods),
        "privacy_note": (
            "raw, redacted, and clustering baselines are non-private; DP keyword "
            "and category histograms are private only with respect to their fixed "
            "public vocabularies."
        ),
    }
    write_json(metadata, outdir / "baseline_metadata.json")
    print(f"Wrote baselines to {outdir}")


def main() -> None:
    """Command-line entry point for baseline generation."""
    p = argparse.ArgumentParser(description="Run raw, heuristic, clustering, and DP histogram baselines.")
    p.add_argument("--records", required=True)
    p.add_argument("--atoms", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["template", "openai"], default="template")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--category-key", default=None)
    p.add_argument("--category-universe-records", default=None, help="Public/auxiliary JSONL used only to define the category label universe.")
    p.add_argument("--category-universe-json", default=None, help="JSON list or dictionary containing a fixed public category universe.")
    p.add_argument("--max-keywords", type=int, default=200)
    p.add_argument("--methods", nargs="+", choices=list(BASELINE_METHOD_CHOICES), default=None, help="Baselines to generate. Default: all baselines.")
    p.add_argument("--adjacency", choices=["adddrop", "replacement"], default="adddrop")
    args = p.parse_args()
    run_baselines(
        args.records,
        args.atoms,
        args.output_dir,
        args.epsilon,
        args.seed,
        args.backend,
        args.model,
        args.category_key,
        args.adjacency,
        args.category_universe_records,
        args.category_universe_json,
        args.max_keywords,
        args.methods,
    )


if __name__ == "__main__":
    main()
