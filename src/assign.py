"""Compute bounded semantic assignments and aggregate semantic sketches.

The public contribution map is fixed before the protected target split is
processed.  Every protected record is mapped to a nonnegative vector with at
most ``K`` nonzero coordinates and total ``l1`` mass at most one.

The assignment layer supports explicit abstention. Records whose strongest public-atom
similarity is below a public threshold need not be forced into the dictionary.
They may contribute the zero vector or, when the dictionary contains a public
``other/uncovered`` atom, contribute to that atom.  The threshold and any
``other`` atom must be fixed using public or disjoint auxiliary information.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .embed import SUPPORTED_EMBEDDING_FAMILIES, atom_text, fit_public_embedder
from .utils import ensure_dir, read_json, read_jsonl, text_field, write_json

AbstentionMode = Literal["zero", "other_atom", "disabled"]
MassMode = Literal["unit", "confidence"]


@dataclass(frozen=True)
class AssignmentDiagnostics:
    """Aggregate diagnostics for one assignment pass.

    These diagnostics are evaluation-only artifacts.  They must not be supplied
    to the decoder unless they are separately privatized.
    """

    n_records: int
    n_atoms: int
    K: int
    gamma: float
    similarity_threshold: float | None
    abstention_mode: str
    other_atom_index: int | None
    mass_mode: str
    n_assigned: int
    n_abstained: int
    coverage: float
    total_assigned_mass: float
    mean_row_mass: float
    min_row_mass: float
    max_row_mass: float
    row_mass_quantiles: dict[str, float]
    max_similarity_quantiles: dict[str, float]
    embedding: dict[str, Any]


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"q00": 0.0, "q05": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q95": 0.0, "q100": 0.0}
    qs = np.quantile(values.astype(float), [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "q00": float(qs[0]),
        "q05": float(qs[1]),
        "q25": float(qs[2]),
        "q50": float(qs[3]),
        "q75": float(qs[4]),
        "q95": float(qs[5]),
        "q100": float(qs[6]),
    }


def _resolve_other_atom_index(
    atoms: list[dict[str, Any]],
    other_atom_id: int | None,
    other_atom_label: str | None,
) -> int | None:
    if other_atom_id is not None:
        for index, atom in enumerate(atoms):
            if int(atom.get("atom_id", atom.get("index", index))) == int(other_atom_id):
                return index
        raise ValueError(f"other_atom_id={other_atom_id} is not present in the public atom dictionary")
    if other_atom_label:
        target = str(other_atom_label).strip().casefold()
        for index, atom in enumerate(atoms):
            if str(atom.get("label", "")).strip().casefold() == target:
                return index
        raise ValueError(f"other_atom_label={other_atom_label!r} is not present in the public atom dictionary")
    return None


def _row_total_mass(max_similarity: float, threshold: float | None, mass_mode: MassMode) -> float:
    if mass_mode == "unit":
        return 1.0
    if mass_mode != "confidence":
        raise ValueError("mass_mode must be 'unit' or 'confidence'")
    # A public deterministic confidence scaling.  It preserves the l1 bound and
    # gives weakly matched records less influence even after they clear the
    # public abstention threshold.
    lower = 0.0 if threshold is None else float(threshold)
    if lower >= 1.0:
        return 1.0 if max_similarity >= lower else 0.0
    return float(np.clip((max_similarity - lower) / max(1e-12, 1.0 - lower), 0.0, 1.0))


def topK_sparse_assignments(
    records: list[dict[str, Any]],
    atoms: list[dict[str, Any]],
    K: int = 3,
    gamma: float = 10.0,
    batch_size: int = 512,
    public_fit_texts: Sequence[str] | None = None,
    allow_private_fit: bool = False,
    *,
    similarity_threshold: float | None = None,
    abstention_mode: AbstentionMode = "zero",
    other_atom_id: int | None = None,
    other_atom_label: str | None = None,
    mass_mode: MassMode = "unit",
    require_positive_similarity: bool = True,
    return_diagnostics: bool = False,
    return_record_diagnostics: bool = False,
    embedding_family: str = "tfidf_word",
    embedding_min_df: int = 1,
    embedding_max_features: int | None = 50000,
    embedding_word_ngram_range: tuple[int, int] = (1, 2),
    embedding_char_ngram_range: tuple[int, int] = (3, 5),
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_model_revision: str | None = None,
    embedding_model_batch_size: int = 64,
    embedding_model_device: str | None = None,
    embedding_model_local_files_only: bool = False,
    embedding_model_trust_remote_code: bool = False,
    show_progress: bool = True,
):
    """Compute bounded assignments and the non-private semantic sketch.

    Parameters
    ----------
    similarity_threshold:
        Public threshold applied to the maximum cosine similarity.  A record
        abstains when its maximum similarity is below the threshold.  ``None``
        disables the explicit threshold, although ``require_positive_similarity``
        can still prevent all-zero rows from being forced into the dictionary.
    abstention_mode:
        ``"zero"`` returns the zero vector for uncovered records;
        ``"other_atom"`` assigns unit mass to a fixed public other atom;
        ``"disabled"`` reproduces the historical forced-assignment behavior.
    mass_mode:
        ``"unit"`` normalizes covered rows to unit mass. ``"confidence"``
        scales total mass to a public deterministic function of the maximum
        similarity while retaining the same sensitivity bound.

    Returns
    -------
    ``(C, V)`` by default.  With ``return_diagnostics=True``, returns
    ``(C, V, diagnostics)``.
    """
    if K < 1:
        raise ValueError("K must be at least 1")
    if not atoms:
        raise ValueError("atoms must be nonempty")
    if K > len(atoms):
        raise ValueError("K cannot exceed number of atoms")
    if gamma <= 0 or not np.isfinite(gamma):
        raise ValueError("gamma must be positive and finite")
    if similarity_threshold is not None and not np.isfinite(similarity_threshold):
        raise ValueError("similarity_threshold must be finite or None")
    if abstention_mode not in {"zero", "other_atom", "disabled"}:
        raise ValueError("abstention_mode must be 'zero', 'other_atom', or 'disabled'")
    if mass_mode not in {"unit", "confidence"}:
        raise ValueError("mass_mode must be 'unit' or 'confidence'")
    if embedding_family not in SUPPORTED_EMBEDDING_FAMILIES:
        raise ValueError(
            f"embedding_family must be one of {SUPPORTED_EMBEDDING_FAMILIES}; "
            f"got {embedding_family!r}"
        )

    other_index = _resolve_other_atom_index(atoms, other_atom_id, other_atom_label)
    if abstention_mode == "other_atom" and other_index is None:
        raise ValueError("abstention_mode='other_atom' requires other_atom_id or other_atom_label")

    record_texts = [text_field(record) for record in records]
    atom_texts = [atom_text(atom) for atom in atoms]

    if allow_private_fit:
        fit_texts = itertools.chain(record_texts, atom_texts)
        fitted_on = "protected_target_demo_only_plus_public_atoms"
    elif public_fit_texts is not None:
        fit_texts = itertools.chain(public_fit_texts, atom_texts)
        fitted_on = "public_or_disjoint_auxiliary_text_plus_public_atoms"
    else:
        fit_texts = iter(atom_texts)
        fitted_on = "public_atom_text_only"

    embedder = fit_public_embedder(
        fit_texts,
        family=embedding_family,
        min_df=embedding_min_df,
        max_features=embedding_max_features,
        word_ngram_range=embedding_word_ngram_range,
        char_ngram_range=embedding_char_ngram_range,
        fitted_on=fitted_on,
        sentence_model_name=embedding_model_name,
        sentence_model_revision=embedding_model_revision,
        sentence_batch_size=embedding_model_batch_size,
        sentence_device=embedding_model_device,
        sentence_local_files_only=embedding_model_local_files_only,
        sentence_trust_remote_code=embedding_model_trust_remote_code,
    )
    atom_matrix = embedder.transform(atom_texts)
    record_matrix = embedder.transform(record_texts)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    sketch = np.zeros(len(atoms), dtype=np.float64)
    row_masses = np.zeros(len(records), dtype=np.float64)
    max_similarities = np.zeros(len(records), dtype=np.float64)
    best_atom_indices = np.full(len(records), -1, dtype=np.int64)
    n_abstained = 0

    for start in tqdm(
        range(0, record_matrix.shape[0], batch_size),
        desc="assign",
        disable=not show_progress,
    ):
        end = min(start + batch_size, record_matrix.shape[0])
        similarities = cosine_similarity(record_matrix[start:end], atom_matrix)
        for batch_index in range(similarities.shape[0]):
            row_index = start + batch_index
            scores = np.asarray(similarities[batch_index], dtype=float)
            max_similarity = float(np.max(scores)) if scores.size else 0.0
            max_similarities[row_index] = max_similarity
            best_atom_indices[row_index] = int(np.argmax(scores)) if scores.size else -1

            should_abstain = False
            if abstention_mode != "disabled":
                if require_positive_similarity and max_similarity <= 0.0:
                    should_abstain = True
                if similarity_threshold is not None and max_similarity < float(similarity_threshold):
                    should_abstain = True

            if should_abstain:
                n_abstained += 1
                if abstention_mode == "other_atom":
                    assert other_index is not None
                    rows.append(row_index)
                    cols.append(int(other_index))
                    vals.append(1.0)
                    sketch[int(other_index)] += 1.0
                    row_masses[row_index] = 1.0
                continue

            top_indices = np.argpartition(-scores, K - 1)[:K]
            top_indices = top_indices[np.lexsort((top_indices, -scores[top_indices]))]

            # With an explicit threshold, retain only coordinates that clear it.
            if similarity_threshold is not None and abstention_mode != "disabled":
                top_indices = top_indices[scores[top_indices] >= float(similarity_threshold)]
            if require_positive_similarity and abstention_mode != "disabled":
                top_indices = top_indices[scores[top_indices] > 0.0]

            if top_indices.size == 0:
                n_abstained += 1
                if abstention_mode == "other_atom":
                    assert other_index is not None
                    rows.append(row_index)
                    cols.append(int(other_index))
                    vals.append(1.0)
                    sketch[int(other_index)] += 1.0
                    row_masses[row_index] = 1.0
                continue

            logits = gamma * scores[top_indices]
            logits = logits - np.max(logits)
            weights = np.exp(logits)
            if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0.0:
                weights = np.ones_like(weights, dtype=np.float64)
            weights = weights / float(weights.sum())
            total_mass = _row_total_mass(max_similarity, similarity_threshold, mass_mode)
            weights = weights * total_mass

            for atom_index, value in zip(top_indices, weights):
                value_float = float(value)
                if value_float <= 0.0:
                    continue
                rows.append(row_index)
                cols.append(int(atom_index))
                vals.append(value_float)
                sketch[int(atom_index)] += value_float
                row_masses[row_index] += value_float

    assignments = sparse.csr_matrix((vals, (rows, cols)), shape=(len(records), len(atoms)))
    if assignments.nnz:
        max_nnz = int(np.max(np.diff(assignments.indptr)))
        if max_nnz > K:
            raise AssertionError(f"assignment construction violated sparsity: {max_nnz} > K={K}")
    if np.any(row_masses < -1e-12) or np.any(row_masses > 1.0 + 1e-10):
        raise AssertionError("assignment construction violated the per-record l1 bound")

    diagnostics = AssignmentDiagnostics(
        n_records=len(records),
        n_atoms=len(atoms),
        K=int(K),
        gamma=float(gamma),
        similarity_threshold=None if similarity_threshold is None else float(similarity_threshold),
        abstention_mode=str(abstention_mode),
        other_atom_index=None if other_index is None else int(other_index),
        mass_mode=str(mass_mode),
        n_assigned=int(np.count_nonzero(row_masses > 0.0)),
        n_abstained=int(n_abstained),
        coverage=float(np.mean(row_masses > 0.0)) if row_masses.size else 0.0,
        total_assigned_mass=float(row_masses.sum()),
        mean_row_mass=float(row_masses.mean()) if row_masses.size else 0.0,
        min_row_mass=float(row_masses.min()) if row_masses.size else 0.0,
        max_row_mass=float(row_masses.max()) if row_masses.size else 0.0,
        row_mass_quantiles=_quantiles(row_masses),
        max_similarity_quantiles=_quantiles(max_similarities),
        embedding=embedder.metadata(),
    )
    record_diagnostics = [
        {
            "record_index": int(i),
            "max_similarity": float(max_similarities[i]),
            "best_atom_index": int(best_atom_indices[i]),
            "best_atom_id": (
                None
                if best_atom_indices[i] < 0
                else int(atoms[int(best_atom_indices[i])].get("atom_id", int(best_atom_indices[i])))
            ),
            "row_mass": float(row_masses[i]),
            "abstained": bool(row_masses[i] <= 0.0),
        }
        for i in range(len(records))
    ]
    if return_diagnostics and return_record_diagnostics:
        return sketch, assignments, asdict(diagnostics), record_diagnostics
    if return_diagnostics:
        return sketch, assignments, asdict(diagnostics)
    if return_record_diagnostics:
        return sketch, assignments, record_diagnostics
    return sketch, assignments


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute sparse semantic assignments and a non-private sketch")
    parser.add_argument("--records", required=True)
    parser.add_argument("--atoms", required=True)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--public-fit-records", default=None)
    parser.add_argument("--allow-private-fit", action="store_true")
    parser.add_argument("--similarity-threshold", type=float, default=None)
    parser.add_argument("--abstention-mode", choices=["zero", "other_atom", "disabled"], default="zero")
    parser.add_argument("--other-atom-id", type=int, default=None)
    parser.add_argument("--other-atom-label", default=None)
    parser.add_argument("--mass-mode", choices=["unit", "confidence"], default="unit")
    parser.add_argument("--allow-zero-similarity", action="store_true")
    parser.add_argument("--embedding-family", choices=list(SUPPORTED_EMBEDDING_FAMILIES), default="tfidf_word")
    parser.add_argument("--embedding-min-df", type=int, default=1)
    parser.add_argument("--embedding-max-features", type=int, default=50000)
    parser.add_argument("--embedding-word-ngram-range", nargs=2, type=int, default=(1, 2))
    parser.add_argument("--embedding-char-ngram-range", nargs=2, type=int, default=(3, 5))
    parser.add_argument("--embedding-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-model-revision", default=None)
    parser.add_argument("--embedding-model-batch-size", type=int, default=64)
    parser.add_argument("--embedding-model-device", default=None)
    parser.add_argument("--embedding-model-local-files-only", action="store_true")
    parser.add_argument("--embedding-model-trust-remote-code", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output", required=True, help="Output .npz file with C and metadata")
    args = parser.parse_args()

    records = read_jsonl(args.records)
    atoms = read_json(args.atoms)
    public_fit_texts = None
    if args.public_fit_records:
        public_fit_texts = [text_field(record) for record in read_jsonl(args.public_fit_records)]
    sketch, assignments, diagnostics = topK_sparse_assignments(
        records,
        atoms,
        K=args.K,
        gamma=args.gamma,
        public_fit_texts=public_fit_texts,
        allow_private_fit=args.allow_private_fit,
        similarity_threshold=args.similarity_threshold,
        abstention_mode=args.abstention_mode,
        other_atom_id=args.other_atom_id,
        other_atom_label=args.other_atom_label,
        mass_mode=args.mass_mode,
        require_positive_similarity=not args.allow_zero_similarity,
        return_diagnostics=True,
        embedding_family=args.embedding_family,
        embedding_min_df=args.embedding_min_df,
        embedding_max_features=args.embedding_max_features,
        embedding_word_ngram_range=tuple(args.embedding_word_ngram_range),
        embedding_char_ngram_range=tuple(args.embedding_char_ngram_range),
        embedding_model_name=args.embedding_model_name,
        embedding_model_revision=args.embedding_model_revision,
        embedding_model_batch_size=args.embedding_model_batch_size,
        embedding_model_device=args.embedding_model_device,
        embedding_model_local_files_only=args.embedding_model_local_files_only,
        embedding_model_trust_remote_code=args.embedding_model_trust_remote_code,
        show_progress=not args.no_progress,
    )
    output = Path(args.output)
    ensure_dir(output.parent)
    sparse.save_npz(str(output.with_suffix(".V.npz")), assignments)
    np.savez(
        output,
        C=sketch,
        K=args.K,
        gamma=args.gamma,
        n=len(records),
        m=len(atoms),
        similarity_threshold=args.similarity_threshold,
        abstention_mode=args.abstention_mode,
        mass_mode=args.mass_mode,
        public_fit_records=args.public_fit_records,
        allow_private_fit=bool(args.allow_private_fit),
        embedding_family=args.embedding_family,
        embedding_model_name=args.embedding_model_name,
        embedding_model_revision=args.embedding_model_revision,
    )
    write_json(diagnostics, output.with_suffix(".assignment_diagnostics.json"))
    print(f"Wrote sketch to {output}")
    print(f"Wrote sparse assignments to {output.with_suffix('.V.npz')}")
    print(f"Wrote diagnostics to {output.with_suffix('.assignment_diagnostics.json')}")


if __name__ == "__main__":
    main()
