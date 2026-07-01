"""Run the repository URANIA-style public-keyword baseline.

The implementation follows the comparison structure used in the paper: private clustering, noisy cluster-size filtering, differentially private keyword histograms over a fixed public vocabulary, and verbalization from the released cluster-keyword object. The default representation and keyword universe are fitted from public or auxiliary data.

This code is not an exact reproduction of the original URANIA implementation.
Repository-local clustering and prompting components replace unavailable external components, and the saved metadata records those differences. 
Enabling a target-derived or privately constructed keyword universe requires separate privacy accounting.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .baselines import keyword_vocabulary_from_atoms
from .decode import decode_plan
from .dp_mechanisms import gaussian_release, laplace_release, topL
from .embed import TfidfEmbedder, atom_text
from .llm_client import maybe_llm
from .utils import ensure_dir, read_json, read_jsonl, text_field, write_json

EmbeddingMode = Literal["public_tfidf_svd", "target_tfidf_svd"]
KeywordSelector = Literal["lexical", "tfidf", "openai"]
ClusterBackend = Literal["repository_noisy_lloyd", "google_dp"]
SummaryBackend = Literal["template", "openai"]


@dataclass
class UraniaBudget:
    """Privacy budget decomposition used by URANIA.

    URANIA's framework gives (epsilon_cluster + epsilon_hist + epsilon_size, delta_cluster)-DP for the released summaries, conditional on
    the keyword set and embedding computation being either public/trusted or
    separately accounted for.
    """

    epsilon_total: float
    delta_total: float
    epsilon_cluster: float
    epsilon_size: float
    epsilon_hist: float
    delta_cluster: float

    @classmethod
    def from_total(
        cls,
        epsilon_total: float,
        delta_total: float,
        cluster_fraction: float = 0.45,
        size_fraction: float = 0.15,
    ) -> "UraniaBudget":
        if epsilon_total <= 0 or not np.isfinite(epsilon_total):
            raise ValueError("epsilon_total must be positive and finite")
        if not (0 <= delta_total < 1):
            raise ValueError("delta_total must be in [0,1)")
        if not (0.0 < cluster_fraction < 1.0):
            raise ValueError("cluster_fraction must be in (0,1)")
        if not (0.0 <= size_fraction < 1.0):
            raise ValueError("size_fraction must be in [0,1)")
        if cluster_fraction + size_fraction >= 1.0:
            raise ValueError("cluster_fraction + size_fraction must be < 1")
        eps_cluster = epsilon_total * cluster_fraction
        eps_size = epsilon_total * size_fraction
        eps_hist = epsilon_total - eps_cluster - eps_size
        return cls(
            epsilon_total=float(epsilon_total),
            delta_total=float(delta_total),
            epsilon_cluster=float(eps_cluster),
            epsilon_size=float(eps_size),
            epsilon_hist=float(eps_hist),
            delta_cluster=float(delta_total),
        )


def _sample_records(records: list[dict], n_records: int | None, seed: int) -> list[dict]:
    """Return at most ``n_records`` records using deterministic random sampling."""
    if n_records is None or n_records <= 0 or len(records) <= n_records:
        return list(records)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(records), size=int(n_records), replace=False))
    return [records[int(i)] for i in idx]


def _safe_json_loads(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_summary_field(record: dict) -> str:
    """Use a precomputed summary/facet field when available; else raw text.

    URANIA embeds an LLM-produced summary/facet representation.  For reproducible
    local experiments, users can provide records containing a ``summary`` or
    ``Summary`` field; otherwise we fall back to the record text.  This fallback
    is explicitly recorded in metadata.
    """
    for key in ("summary", "Summary", "llm_summary", "conversation_summary"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return text_field(record)


def _fit_embedding(
    atoms: list[dict],
    aux_records: list[dict],
    target_records: list[dict],
    svd_dim: int,
    embedding_mode: EmbeddingMode,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit/transform embeddings for the URANIA-style clustering step.

    The default ``public_tfidf_svd`` mode fits the representation only on atom
    text and auxiliary records, which gives a fair fixed-representation protocol
    for comparison with DP-SPIN.  The ``target_tfidf_svd`` mode is closer to a
    trusted-curator embedding pipeline but should be treated as an internal
    feature extraction step and not as a public representation.
    """
    atom_texts = [atom_text(a) for a in atoms]
    aux_texts = [_extract_summary_field(r) for r in aux_records]
    target_texts = [_extract_summary_field(r) for r in target_records]

    if embedding_mode == "public_tfidf_svd":
        fit_texts = atom_texts + aux_texts
        target_records_used_for_fit = False
        fit_source = "public atom text plus public/auxiliary records"
    elif embedding_mode == "target_tfidf_svd":
        fit_texts = target_texts
        target_records_used_for_fit = True
        fit_source = "protected target records inside trusted curator"
    else:
        raise ValueError("embedding_mode must be public_tfidf_svd or target_tfidf_svd")
    if not fit_texts:
        raise ValueError("Need at least one text to fit the embedding representation")

    embedder = TfidfEmbedder.fit(fit_texts, max_features=50000)
    X_aux_sparse = embedder.transform(aux_texts if aux_texts else atom_texts)
    X_target_sparse = embedder.transform(target_texts)
    n_features = X_target_sparse.shape[1]
    if n_features > 2:
        dim = int(min(max(2, svd_dim), n_features - 1))
    else:
        dim = n_features

    if dim >= 2 and n_features > dim:
        fit_sparse = embedder.transform(fit_texts)
        svd = TruncatedSVD(n_components=dim, random_state=0)
        svd.fit(fit_sparse)
        X_aux = svd.transform(X_aux_sparse)
        X_target = svd.transform(X_target_sparse)
        projection = "tfidf_truncated_svd"
    else:
        X_aux = X_aux_sparse.toarray() if sparse.issparse(X_aux_sparse) else np.asarray(X_aux_sparse)
        X_target = X_target_sparse.toarray() if sparse.issparse(X_target_sparse) else np.asarray(X_target_sparse)
        projection = "tfidf_dense"

    X_aux = normalize(np.asarray(X_aux, dtype=float), norm="l2", axis=1, copy=False)
    X_target = normalize(np.asarray(X_target, dtype=float), norm="l2", axis=1, copy=False)
    meta = {
        "embedding_representation": projection,
        "embedding_mode": embedding_mode,
        "fit_source": fit_source,
        "target_records_used_for_fit": target_records_used_for_fit,
        "dimension": int(X_target.shape[1]),
        "num_aux_fit_texts": int(len(aux_texts)),
        "num_atom_texts": int(len(atom_texts)),
        "uses_precomputed_summary_fields_when_available": True,
    }
    return X_target, X_aux, meta


def _public_initial_centers(X_aux: np.ndarray, num_clusters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = int(max(1, num_clusters))
    d = int(X_aux.shape[1])
    if X_aux.shape[0] >= k:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X_aux)
        centers = np.asarray(km.cluster_centers_, dtype=float)
    elif X_aux.shape[0] > 0:
        choices = rng.choice(X_aux.shape[0], size=k, replace=True)
        centers = np.asarray(X_aux[choices], dtype=float)
    else:
        centers = rng.normal(size=(k, d))
    return normalize(centers, norm="l2", axis=1, copy=False)


def _assign_to_centers(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1)[None, :]
    dist = x2 + c2 - 2.0 * X @ centers.T
    return np.argmin(dist, axis=1).astype(int)


def dp_noisy_lloyd_centers(
    X: np.ndarray,
    public_init_centers: np.ndarray,
    epsilon_cluster: float,
    delta_cluster: float,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Repository-local DP clustering substitute for URANIA's DP-KMeans.

    Each l2-clipped record contributes to exactly one cluster sum and one cluster
    count in each iteration.  We release noisy sums and noisy counts, then update
    centers as post-processing.  Privacy is by basic composition over iterations
    and over the sum/count releases.
    """
    if epsilon_cluster <= 0:
        raise ValueError("epsilon_cluster must be positive")
    if delta_cluster <= 0:
        raise ValueError("delta_cluster must be positive for Gaussian DP clustering")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    rng = np.random.default_rng(seed)
    centers = public_init_centers.copy()
    k, d = centers.shape
    eps_iter = epsilon_cluster / iterations
    delta_iter = delta_cluster / iterations
    eps_sums = 0.75 * eps_iter
    eps_counts = eps_iter - eps_sums
    delta_sums = 0.75 * delta_iter
    delta_counts = delta_iter - delta_sums
    history: list[dict[str, Any]] = []

    for it in range(iterations):
        labels = _assign_to_centers(X, centers)
        sums = np.zeros((k, d), dtype=float)
        counts = np.zeros(k, dtype=float)
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                sums[j] = X[mask].sum(axis=0)
                counts[j] = float(np.sum(mask))

        noisy_sums_flat, sum_info = gaussian_release(
            sums.reshape(-1), epsilon=eps_sums, delta=delta_sums,
            l2_sensitivity=1.0, rng=rng, calibration="analytic"
        )
        noisy_counts, count_info = gaussian_release(
            counts, epsilon=eps_counts, delta=delta_counts,
            l2_sensitivity=1.0, rng=rng, calibration="analytic"
        )
        noisy_sums = noisy_sums_flat.reshape(k, d)
        new_centers = centers.copy()
        for j in range(k):
            denom = max(float(noisy_counts[j]), 1.0)
            cand = noisy_sums[j] / denom
            if np.linalg.norm(cand) > 1e-12:
                new_centers[j] = cand
        centers = normalize(new_centers, norm="l2", axis=1, copy=False)
        history.append({
            "iteration": it + 1,
            "epsilon_sums": eps_sums,
            "epsilon_counts": eps_counts,
            "delta_sums": delta_sums,
            "delta_counts": delta_counts,
            "sum_sigma": sum_info.get("sigma"),
            "count_sigma": count_info.get("sigma"),
            "num_nonempty_true_clusters": int(np.sum(counts > 0)),
            "min_noisy_count": float(np.min(noisy_counts)) if len(noisy_counts) else None,
            "max_noisy_count": float(np.max(noisy_counts)) if len(noisy_counts) else None,
        })

    info = {
        "clustering_mechanism": "repository_local_noisy_lloyd_gaussian",
        "matches_urania_backend_exactly": False,
        "urania_backend_note": "URANIA uses Google DP-KMeans; this is a local DP substitute with the same DP role.",
        "epsilon_cluster": float(epsilon_cluster),
        "delta_cluster": float(delta_cluster),
        "iterations": int(iterations),
        "per_iteration_composition": "basic",
        "embedding_l2_clip": 1.0,
        "history": history,
    }
    return centers, info


def _dp_cluster_centers(
    X: np.ndarray,
    X_aux: np.ndarray,
    num_clusters: int,
    epsilon_cluster: float,
    delta_cluster: float,
    iterations: int,
    seed: int,
    backend: ClusterBackend,
) -> tuple[np.ndarray, dict[str, Any]]:
    if backend == "repository_noisy_lloyd":
        init = _public_initial_centers(X_aux, num_clusters=num_clusters, seed=seed)
        return dp_noisy_lloyd_centers(X, init, epsilon_cluster, delta_cluster, iterations, seed)
    if backend == "google_dp":
        raise RuntimeError(
            "google_dp backend is not bundled with this repository. Install and wire the Google "
            "differential-privacy clustering library, or use --cluster-backend repository_noisy_lloyd."
        )
    raise ValueError("unknown cluster backend")


# ---------------------------------------------------------------------------
# Discrete Laplace / PHR utilities
# ---------------------------------------------------------------------------


def _sample_discrete_laplace(shape: tuple[int, ...] | int, epsilon_per_unit: float, rng: np.random.Generator) -> np.ndarray:
    """Sample symmetric discrete Laplace with P(z) proportional to exp(-eps |z|)."""
    if epsilon_per_unit <= 0:
        raise ValueError("epsilon_per_unit must be positive")
    alpha = math.exp(-float(epsilon_per_unit))
    size = shape
    u = rng.random(size=size)
    out = np.zeros(size, dtype=float)
    p0 = (1.0 - alpha) / (1.0 + alpha)
    nonzero = u >= p0
    if np.any(nonzero):
        # conditional magnitude distribution: P(M=m | M>0)=(1-alpha) alpha^(m-1), m>=1
        mags = rng.geometric(p=1.0 - alpha, size=int(np.sum(nonzero))).astype(float)
        signs = rng.choice(np.array([-1.0, 1.0]), size=int(np.sum(nonzero)))
        out[nonzero] = signs * mags
    return out


def private_histogram_release(
    counts: np.ndarray,
    epsilon: float,
    max_bins_per_record: int,
    rng: np.random.Generator,
    mechanism: Literal["discrete_laplace", "continuous_laplace"] = "discrete_laplace",
) -> tuple[np.ndarray, dict[str, Any]]:
    """URANIA PHR_k,epsilon for bounded histograms."""
    if max_bins_per_record <= 0:
        raise ValueError("max_bins_per_record must be positive")
    if mechanism == "discrete_laplace":
        eps_unit = epsilon / float(max_bins_per_record)
        noise = _sample_discrete_laplace(counts.shape, eps_unit, rng)
        return counts.astype(float) + noise, {
            "mechanism": "PHR_discrete_laplace",
            "epsilon": float(epsilon),
            "delta": 0.0,
            "max_bins_per_record": int(max_bins_per_record),
            "epsilon_per_unit": float(eps_unit),
        }
    if mechanism == "continuous_laplace":
        return laplace_release(counts.astype(float), epsilon=epsilon, l1_sensitivity=float(max_bins_per_record), rng=rng)
    raise ValueError("unknown histogram mechanism")


# ---------------------------------------------------------------------------
# Keyword selection
# ---------------------------------------------------------------------------


def _keyword_regex_count(text: str, keyword: str) -> int:
    low = text.lower()
    kw = keyword.lower().strip()
    if not kw:
        return 0
    pattern = rf"(?<![a-zA-Z0-9]){re.escape(kw)}(?![a-zA-Z0-9])"
    return len(re.findall(pattern, low))


def _record_keywords_lexical(text: str, vocabulary: list[str], max_keywords_per_record: int) -> list[int]:
    scored: list[tuple[int, int]] = []
    for j, kw in enumerate(vocabulary):
        c = _keyword_regex_count(text, kw)
        if c > 0:
            scored.append((-c, j))
    scored.sort()
    return [j for _, j in scored[: max(0, int(max_keywords_per_record))]]


def _record_keywords_tfidf(text: str, vocabulary: list[str], max_keywords_per_record: int) -> list[int]:
    """Select public keywords by local TF-IDF cosine similarity to one record."""
    if not vocabulary or max_keywords_per_record <= 0:
        return []
    docs = [text] + vocabulary
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)
    try:
        X = vec.fit_transform(docs)
    except ValueError:
        return []
    rec = X[0]
    kws = X[1:]
    sims = (kws @ rec.T).toarray().reshape(-1)
    positive = np.flatnonzero(sims > 0)
    if len(positive) == 0:
        return []
    order = positive[np.argsort(-sims[positive])]
    return [int(j) for j in order[:max_keywords_per_record]]


def _record_keywords_openai(text: str, vocabulary: list[str], max_keywords_per_record: int, model: str) -> list[int]:
    """URANIA-style LLMSelectKeywords from a fixed keyword set.

    The prompt restricts selection to the supplied
    public vocabulary.  Returned strings are matched back to exact vocabulary
    entries; unknown strings are ignored.
    """
    if not vocabulary or max_keywords_per_record <= 0:
        return []
    vocab_text = ", ".join(vocabulary)
    prompt = (
        "Below is a summary or text record. Based on it, select the most relevant "
        f"{max_keywords_per_record} keywords from the given set of keywords.\n\n"
        f"Summary: {text}\n\nAvailable Keywords: {vocab_text}\n\n"
        "Return a JSON object of the form {\"keywords\": [\"keyword1\", ...]}. "
        "Use only exact keywords from the available set. If none are relevant, return {\"keywords\": []}."
    )
    resp = maybe_llm(prompt, backend="openai", model=model, temperature=0.0, max_tokens=250)
    if not resp:
        return []
    parsed = _safe_json_loads(resp.strip())
    if isinstance(parsed, dict):
        raw = parsed.get("keywords", [])
    elif isinstance(parsed, list):
        raw = parsed
    else:
        raw = re.findall(r"['\"]([^'\"]+)['\"]", resp)
    vocab_map = {kw.lower(): i for i, kw in enumerate(vocabulary)}
    out: list[int] = []
    for item in raw:
        key = str(item).strip().lower()
        if key in vocab_map and vocab_map[key] not in out:
            out.append(vocab_map[key])
        if len(out) >= max_keywords_per_record:
            break
    return out


def _record_keyword_indices(
    text: str,
    vocabulary: list[str],
    max_keywords_per_record: int,
    selector: KeywordSelector,
    model: str,
) -> list[int]:
    if selector == "lexical":
        return _record_keywords_lexical(text, vocabulary, max_keywords_per_record)
    if selector == "tfidf":
        return _record_keywords_tfidf(text, vocabulary, max_keywords_per_record)
    if selector == "openai":
        return _record_keywords_openai(text, vocabulary, max_keywords_per_record, model)
    raise ValueError("unknown keyword selector")


def _summarize_cluster_template(cluster: dict[str, Any]) -> str:
    kws = cluster.get("keywords", [])
    if not kws:
        return "No positive released keywords were available for this retained cluster."
    head = ", ".join(kws[:5])
    tail = ", ".join(kws[5:10])
    if tail:
        return f"Topic: {head}. Description: This retained cluster is characterized by the released public keywords {head}, with additional related keywords {tail}."
    return f"Topic: {head}. Description: This retained cluster is characterized by the released public keywords {head}."


def _summarize_cluster_openai(cluster: dict[str, Any], model: str) -> tuple[str, str]:
    kws = cluster.get("keywords", [])
    noisy_size = float(cluster.get("noisy_size", 0.0))
    prompt = (
        "You are given a retained cluster from a differentially private public-keyword baseline. "
        "The decoder receives only selected public keywords and a released noisy cluster size.\n\n"
        "Write a concise topic label and one cautious aggregate description using only the provided keywords. "
        "The noisy size may be used only to know that the cluster was retained and ordered by released size; "
        "do not repeat the numeric noisy size.\n\n"
        "Rules:\n"
        "1. Use only the provided public keywords and the fact that the cluster was retained.\n"
        "2. Do not infer events, actions, causes, consent, legal status, demographics, companies, products, or user intent unless those words appear in the keywords.\n"
        "3. Do not invent examples, names, organizations, dates, quotes, locations, or record-specific details.\n"
        "4. Do not claim to have read raw records.\n"
        "5. Do not include XML tags, Markdown headings, cluster IDs, or numeric noisy sizes.\n"
        "6. Use the format `Topic: <keyword-based topic>` followed by `Description: <one cautious sentence>`.\n\n"
        f"Released noisy size, used only for retention and ordering:\n{noisy_size:.1f}\n\n"
        "Selected public keywords:\n" + ", ".join(kws) + "\n"
    )
    resp = maybe_llm(prompt, backend="openai", model=model, temperature=0.0, max_tokens=220)
    return (resp or _summarize_cluster_template(cluster), prompt)

def _clean_cluster_summary_for_display(text: str) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    # Remove boilerplate introduced by older prompts.
    s = re.sub(r"\bThe noisy cluster size is [0-9.]+\.?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bThe noisy size is [0-9.]+\.?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bnoisy cluster size is [0-9.]+\.?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" -")
    return s


def _render_all_summaries(clusters: list[dict[str, Any]], max_clusters: int = 4) -> str:
    """Render the generated aggregate summary from retained released clusters."""
    if not clusters:
        return "No cluster passed the released noisy-size threshold."
    lines = ["Retained cluster-level themes include:"]
    for c in clusters[:max_clusters]:
        summary = _clean_cluster_summary_for_display(c.get("summary", ""))
        if summary:
            lines.append(f"- {summary}")
    return "\n".join(lines)

def _render_released_cluster_object(clusters: list[dict[str, Any]]) -> str:
    """Render the complete released cluster object for inspection and archival."""
    if not clusters:
        return "No retained clusters."
    lines = ["URANIA-style retained clusters with released noisy sizes:"]
    for c in clusters:
        lines.append(f"- Cluster {c['cluster_id']} (released noisy size {c['noisy_size']:.1f}): {c.get('summary', '')}")
    return "\n".join(lines)



def _summarize_retained_clusters_openai(clusters: list[dict[str, Any]], model: str, max_clusters: int = 4) -> tuple[str, str]:
    """Verbalize the retained released clusters with an OpenAI backend.
    
    The prompt contains only released public keywords and previously generated
    cluster summaries. Cluster identifiers and exact noisy sizes remain in the audit
    artifact but are omitted from the qualitative prose.
    """
    if not clusters:
        return "No cluster passed the released noisy-size threshold.", ""
    entries = []
    for c in clusters[:max_clusters]:
        kws = ", ".join(str(k) for k in c.get("keywords", [])[:10])
        summary = _clean_cluster_summary_for_display(c.get("summary", ""))
        entries.append(f"Retained cluster keywords: {kws}\nCluster-level summary: {summary}")
    prompt = (
        "You are given retained cluster-level release entries from a differentially private public-keyword baseline. "
        "Write a concise aggregate insight summary using only the provided public keywords and cluster-level summaries.\n\n"
        "Rules:\n"
        "1. Do not mention cluster IDs, numeric noisy sizes, or implementation details.\n"
        "2. Do not claim to have read raw records.\n"
        "3. Do not invent examples, names, organizations, dates, quotes, causes, legal status, product identities, or user intent.\n"
        "4. If the released keywords are broad or ambiguous, use cautious language.\n"
        "5. Write one concise paragraph followed by at most three short bullets.\n\n"
        "Released retained cluster entries:\n"
        + "\n---\n".join(entries)
    )
    resp = maybe_llm(prompt, backend="openai", model=model, temperature=0.0, max_tokens=350)
    return (resp or _render_all_summaries(clusters, max_clusters=max_clusters), prompt)


def _make_public_vocabulary(
    atoms: list[dict],
    max_keywords: int,
    extra_public_keywords: list[str] | None,
) -> list[str]:
    vocabulary = keyword_vocabulary_from_atoms(atoms, max_keywords=max_keywords)
    seen = {v.lower() for v in vocabulary}
    for kw in extra_public_keywords or []:
        kw = str(kw).strip()
        if kw and kw.lower() not in seen:
            vocabulary.append(kw)
            seen.add(kw.lower())
    if not vocabulary:
        raise ValueError("The public keyword vocabulary is empty; check the atom file.")
    return vocabulary


def run_urania_public_keyword_baseline(
    records_path: str,
    aux_records_path: str,
    atoms_path: str,
    output_dir: str,
    epsilon_total: float = 1.0,
    delta: float = 1e-6,
    num_clusters: int = 10,
    keywords_per_summary: int = 10,
    max_keywords: int = 200,
    max_keywords_per_record: int = 5,
    #
    # Runtime cap for keyword-histogram construction. Set to None to use
    # all records assigned to each cluster; the CLI also treats values <= 0 as no cap.
    sample_per_cluster: int | None = 50,
    extra_public_keywords: list[str] | None = None,
    cluster_size_threshold: float | None = None,
    svd_dim: int = 64,
    dp_iterations: int = 3,
    seed: int = 0,
    backend: SummaryBackend = "template",
    model: str = "gpt-4o-mini",
    n_records: int | None = None,
    cluster_fraction: float = 0.45,
    size_fraction: float = 0.15,
    embedding_mode: EmbeddingMode = "public_tfidf_svd",
    cluster_backend: ClusterBackend = "repository_noisy_lloyd",
    keyword_selector: KeywordSelector = "lexical",
    histogram_mechanism: Literal["discrete_laplace", "continuous_laplace"] = "discrete_laplace",
) -> None:
    """Run the URANIA-style DP clustered public-keyword comparison.
    
    The default configuration implements the matched public-vocabulary protocol used
    in this repository. Report it as a URANIA-style baseline rather than an exact
    reproduction of the original experiments.
    """
    rng = np.random.default_rng(seed)
    outdir = ensure_dir(output_dir)
    records_all = read_jsonl(records_path)
    records = _sample_records(records_all, n_records=n_records, seed=seed)
    aux_records = read_jsonl(aux_records_path)
    atoms = read_json(atoms_path)
    vocabulary = _make_public_vocabulary(atoms, max_keywords=max_keywords, extra_public_keywords=extra_public_keywords)

    budget = UraniaBudget.from_total(
        epsilon_total=epsilon_total,
        delta_total=delta,
        cluster_fraction=cluster_fraction,
        size_fraction=size_fraction,
    )

    X, X_aux, embed_meta = _fit_embedding(atoms, aux_records, records, svd_dim=svd_dim, embedding_mode=embedding_mode)
    k = min(int(num_clusters), max(1, len(records)))
    centers, cluster_info = _dp_cluster_centers(
        X, X_aux, num_clusters=k, epsilon_cluster=budget.epsilon_cluster,
        delta_cluster=budget.delta_cluster, iterations=dp_iterations, seed=seed,
        backend=cluster_backend,
    )
    labels = _assign_to_centers(X, centers)

    true_sizes = np.bincount(labels, minlength=k).astype(float)
    noisy_sizes, size_info = private_histogram_release(
        true_sizes, epsilon=budget.epsilon_size, max_bins_per_record=1, rng=rng,
        mechanism=histogram_mechanism,
    )
    if cluster_size_threshold is None:
        cluster_size_threshold = max(10.0, 0.25 * len(records) / max(1, k))

    keyword_counts = np.zeros((k, len(vocabulary)), dtype=float)
    per_record_keyword_counts: list[int] = []
    sampled_record_ids_by_cluster: dict[int, list[int]] = {}

    for c in range(k):
        cluster_indices = np.flatnonzero(labels == c)
        # Runtime cap used in the reported URANIA-style baseline.  Set
        # sample_per_cluster=None to use all records assigned to each cluster.
        if sample_per_cluster is None or sample_per_cluster <= 0 or len(cluster_indices) <= sample_per_cluster:
            sampled = cluster_indices
        else:
            sampled = np.sort(rng.choice(cluster_indices, size=int(sample_per_cluster), replace=False))
        sampled_record_ids_by_cluster[int(c)] = [int(i) for i in sampled]
        for i in sampled:
            kw_idx = _record_keyword_indices(
                _extract_summary_field(records[int(i)]), vocabulary, max_keywords_per_record,
                selector=keyword_selector, model=model,
            )
            per_record_keyword_counts.append(len(kw_idx))
            for j in kw_idx:
                keyword_counts[c, j] += 1.0

    noisy_kw_flat, hist_info = private_histogram_release(
        keyword_counts.reshape(-1), epsilon=budget.epsilon_hist,
        max_bins_per_record=int(max_keywords_per_record), rng=rng,
        mechanism=histogram_mechanism,
    )
    noisy_kw = noisy_kw_flat.reshape(k, len(vocabulary))

    clusters: list[dict[str, Any]] = []
    cluster_prompts: dict[str, str] = {}
    for c in range(k):
        if noisy_sizes[c] < float(cluster_size_threshold):
            continue
        if true_sizes[c] <= 0:
            # URANIA Algorithm 1 assigns random keywords to empty clusters that
            # pass the noisy threshold.
            rand_idx = rng.choice(len(vocabulary), size=min(keywords_per_summary, len(vocabulary)), replace=False)
            idx = np.asarray(rand_idx, dtype=int)
        else:
            idx = topL(noisy_kw[c], min(int(keywords_per_summary), len(vocabulary)))
        keywords = [vocabulary[int(j)] for j in idx if true_sizes[c] <= 0 or noisy_kw[c, int(j)] > 0]
        cluster = {
            "cluster_id": int(c),
            "true_size_for_evaluation_only": float(true_sizes[c]),
            "noisy_size": float(max(0.0, noisy_sizes[c])),
            "size_threshold": float(cluster_size_threshold),
            "keywords": keywords,
            "keyword_noisy_counts": {vocabulary[int(j)]: float(noisy_kw[c, int(j)]) for j in idx},
            "num_sampled_records_for_keywords": int(len(sampled_record_ids_by_cluster.get(int(c), []))),
        }
        if backend == "openai":
            summary, prompt = _summarize_cluster_openai(cluster, model=model)
            cluster_prompts[str(c)] = prompt
        else:
            summary = _summarize_cluster_template(cluster)
        cluster["summary"] = summary
        clusters.append(cluster)
    clusters.sort(key=lambda z: -z["noisy_size"])

    if backend == "openai":
        released_summary_text, final_summary_prompt = _summarize_retained_clusters_openai(clusters, model=model)
    else:
        released_summary_text = _render_all_summaries(clusters)
        final_summary_prompt = ""
    released_cluster_object_text = _render_released_cluster_object(clusters)

    plan = {
        "baseline": "restricted_urania_style_public_keyword",
        "description": (
            "URANIA-style public-keyword DP baseline: DP clustering, DP cluster-size thresholding, "
            "per-cluster DP keyword histograms over a fixed public keyword vocabulary, and "
            "summary generation from released keywords only."
        ),
        "clusters": clusters,
        "selected_atoms": [
            {
                "rank": i + 1,
                "atom_id": int(c["cluster_id"]),
                "index": int(c["cluster_id"]),
                "label": f"URANIA-style cluster {c['cluster_id']}: " + ", ".join(c.get("keywords", [])[:5]),
                "description": "Released DP keyword cluster over a fixed public vocabulary.",
                "noisy_mass": float(c["noisy_size"]),
                "prevalence_bin": None,
            }
            for i, c in enumerate(clusters)
        ],
        "privacy": {
            **asdict(budget),
            "privacy_statement": (
                "By basic composition, the released centers, noisy cluster-size tests, keyword histograms, "
                "and summaries are (epsilon_total, delta_total)-DP under record add/drop adjacency, "
                "conditional on the fixed public keyword universe and the stated embedding/keyword-selection protocol."
            ),
        },
        "parameters": {
            "num_records": int(len(records)),
            "num_clusters_requested": int(num_clusters),
            "num_clusters_effective": int(k),
            "keywords_per_summary": int(keywords_per_summary),
            "max_keywords": int(max_keywords),
            "max_keywords_per_record": int(max_keywords_per_record),
            "sample_per_cluster": None if sample_per_cluster is None else int(sample_per_cluster),
            "extra_public_keywords": list(extra_public_keywords or []),
            "cluster_size_threshold": float(cluster_size_threshold),
            "svd_dim": int(svd_dim),
            "dp_iterations": int(dp_iterations),
            "seed": int(seed),
            "summary_backend": backend,
            "model": model,
            "keyword_selector": keyword_selector,
            "cluster_backend": cluster_backend,
            "histogram_mechanism": histogram_mechanism,
        },
        "embedding": embed_meta,
        "mechanism_info": {
            "clustering": cluster_info,
            "size_release": size_info,
            "keyword_histogram_release": hist_info,
        },
        "scope_note": (
            "This baseline matches URANIA's mechanism structure but uses repository-local clustering and keyword selection. "
            "For fair comparison with DP-SPIN, the default uses a fixed public vocabulary and public/auxiliary representation."
        ),
    }

    metrics = {
        "baseline": "restricted_urania_style_public_keyword",
        "num_records": int(len(records)),
        "num_public_keywords": int(len(vocabulary)),
        "num_clusters_requested": int(num_clusters),
        "num_clusters_released": int(len(clusters)),
        "mean_true_cluster_size": float(np.mean(true_sizes)) if len(true_sizes) else 0.0,
        "mean_noisy_released_cluster_size": float(np.mean([c["noisy_size"] for c in clusters])) if clusters else 0.0,
        "num_records_sampled_for_keyword_histograms": int(sum(len(v) for v in sampled_record_ids_by_cluster.values())),
        "num_sampled_records_with_public_keyword_match": int(sum(1 for z in per_record_keyword_counts if z > 0)),
        "mean_keywords_per_sampled_record_before_clipping": float(np.mean(per_record_keyword_counts)) if per_record_keyword_counts else 0.0,
        "epsilon_total": float(epsilon_total),
        "delta_total": float(delta),
        "epsilon_cluster": float(budget.epsilon_cluster),
        "epsilon_size": float(budget.epsilon_size),
        "epsilon_hist": float(budget.epsilon_hist),
        "cluster_size_threshold": float(cluster_size_threshold),
        "embedding_mode": embedding_mode,
        "cluster_backend": cluster_backend,
        "keyword_selector": keyword_selector,
        "histogram_mechanism": histogram_mechanism,
        "is_full_urania_reproduction": False,
    }

    # Keep legacy filenames for compatibility with existing reporting scripts.
    write_json(plan, outdir / "urania_cluster_public_keyword_plan.json")
    write_json(metrics, outdir / "urania_cluster_public_keyword_metrics.json")
    write_json({"public_keywords": vocabulary}, outdir / "urania_public_keyword_vocabulary.json")
    write_json(cluster_prompts, outdir / "urania_cluster_public_keyword_prompts.json")
    (outdir / "urania_cluster_public_keyword_summary.txt").write_text(released_summary_text, encoding="utf-8")
    (outdir / "urania_cluster_public_keyword_released_object.txt").write_text(released_cluster_object_text, encoding="utf-8")
    (outdir / "urania_cluster_public_keyword_final_prompt.txt").write_text(final_summary_prompt, encoding="utf-8")
    (outdir / "urania_cluster_public_keyword_prompt.txt").write_text(
        "Cluster summaries generated from released public keywords; released noisy sizes are used for retention and ordering, but are not repeated in the paper-facing summary.", encoding="utf-8"
    )
    print(f"Wrote URANIA-style public-keyword baseline to {outdir}")


def main() -> None:
    """Run the URANIA-style public-keyword baseline from the command line."""
    p = argparse.ArgumentParser(description="Run a URANIA-style DP clustered keyword baseline.")
    p.add_argument("--records", required=True, help="Protected target JSONL records.")
    p.add_argument("--aux-records", required=True, help="Public/auxiliary JSONL records for fitting representation only.")
    p.add_argument("--atoms", required=True, help="Public atom file used to define the keyword vocabulary.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epsilon-total", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument("--cluster-fraction", type=float, default=0.45)
    p.add_argument("--size-fraction", type=float, default=0.15)
    p.add_argument("--num-clusters", type=int, default=10)
    p.add_argument("--keywords-per-summary", type=int, default=10)
    p.add_argument("--max-keywords", type=int, default=200)
    p.add_argument("--max-keywords-per-record", type=int, default=5)
    p.add_argument("--sample-per-cluster", type=int, default=50,
                   help="Runtime cap on records used per cluster for keyword histograms; use 0 or a negative value to include all assigned records.")
    p.add_argument("--extra-public-keyword", action="append", default=[], help="Additional public keyword to append to the fixed keyword universe. Repeatable.")
    p.add_argument("--extra-public-keywords-file", default=None, help="Optional text file with one additional public keyword per line.")
    p.add_argument("--cluster-size-threshold", type=float, default=None)
    p.add_argument("--svd-dim", type=int, default=64)
    p.add_argument("--dp-iterations", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["template", "openai"], default="template")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--n-records", type=int, default=None)
    p.add_argument("--embedding-mode", choices=["public_tfidf_svd", "target_tfidf_svd"], default="public_tfidf_svd")
    p.add_argument("--cluster-backend", choices=["repository_noisy_lloyd", "google_dp"], default="repository_noisy_lloyd")
    p.add_argument("--keyword-selector", choices=["lexical", "tfidf", "openai"], default="lexical")
    p.add_argument("--histogram-mechanism", choices=["discrete_laplace", "continuous_laplace"], default="discrete_laplace")
    args = p.parse_args()

    extra_keywords = list(args.extra_public_keyword or [])
    if args.extra_public_keywords_file:
        extra_path = Path(args.extra_public_keywords_file)
        if extra_path.exists():
            extra_keywords.extend([line.strip() for line in extra_path.read_text(encoding="utf-8").splitlines() if line.strip()])

    run_urania_public_keyword_baseline(
        records_path=args.records,
        aux_records_path=args.aux_records,
        atoms_path=args.atoms,
        output_dir=args.output_dir,
        epsilon_total=args.epsilon_total,
        delta=args.delta,
        num_clusters=args.num_clusters,
        keywords_per_summary=args.keywords_per_summary,
        max_keywords=args.max_keywords,
        max_keywords_per_record=args.max_keywords_per_record,
        sample_per_cluster=None if args.sample_per_cluster <= 0 else args.sample_per_cluster,
        extra_public_keywords=extra_keywords,
        cluster_size_threshold=args.cluster_size_threshold,
        svd_dim=args.svd_dim,
        dp_iterations=args.dp_iterations,
        seed=args.seed,
        backend=args.backend,
        model=args.model,
        n_records=args.n_records,
        cluster_fraction=args.cluster_fraction,
        size_fraction=args.size_fraction,
        embedding_mode=args.embedding_mode,
        cluster_backend=args.cluster_backend,
        keyword_selector=args.keyword_selector,
        histogram_mechanism=args.histogram_mechanism,
    )


if __name__ == "__main__":
    main()
