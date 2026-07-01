"""Compute bounded semantic assignments and the aggregate semantic sketch.

For each protected record ``x_i``, :func:`topK_sparse_assignments` constructs a nonnegative assignment vector ``v_i`` over a fixed public semantic atom universe.
The implementation keeps at most ``K`` nonzero coordinates and normalizes each nonempty row to unit ``l1`` mass. The aggregate sketch is ``C = sum_i v_i``.

The atom dictionary and TF-IDF representation must be fixed independently of the protected target split. 
By default, the vectorizer is fitted on atom text and optional public or auxiliary text. 
``allow_private_fit=True`` also fits on the protected target records and is therefore restricted to non-private diagnostic runs.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .embed import TfidfEmbedder, atom_text
from .utils import read_json, read_jsonl, ensure_dir, text_field


def topK_sparse_assignments(
    records: list[dict[str, Any]],
    atoms: list[dict[str, Any]],
    K: int = 3,
    gamma: float = 10.0,
    batch_size: int = 512,
    public_fit_texts: Sequence[str] | None = None,
    allow_private_fit: bool = False,
) -> tuple[np.ndarray, sparse.csr_matrix]:
    """Compute bounded assignments and their aggregate semantic sketch.
    
    Each nonempty row ``v_i`` is nonnegative, has at most ``K`` nonzero entries,
    and is normalized to unit ``l1`` mass. This implementation therefore satisfies
    the theoretical bound ``||v_i||_1 <= 1``.
    
    The fitted vectorizer must be fixed independently of the protected target split.
    By default, fitting uses atom text and optional public or auxiliary text.
    ``allow_private_fit=True`` includes target text and is valid only for explicitly
    non-private diagnostic runs.
    """
    if K < 1:
        raise ValueError("K must be at least 1.")
    if K > len(atoms):
        raise ValueError("K cannot exceed number of atoms.")

    record_texts = [text_field(r) for r in records]
    atom_texts = [atom_text(a) for a in atoms]

    # The fitted vectorizer is part of the mechanism design, not a private
    # output.  Therefore its vocabulary and IDF weights must be determined from
    # public/auxiliary text unless this is an explicitly non-private demo run.
    if allow_private_fit:
        fit_texts = itertools.chain(record_texts, atom_texts)
    elif public_fit_texts is not None:
        fit_texts = itertools.chain(public_fit_texts, atom_texts)
    else:
        fit_texts = iter(atom_texts)

    embedder = TfidfEmbedder.fit(fit_texts)
    A = embedder.transform(atom_texts)
    X = embedder.transform(record_texts)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    C = np.zeros(len(atoms), dtype=np.float64)

    for start in tqdm(range(0, X.shape[0], batch_size), desc="assign"):
        end = min(start + batch_size, X.shape[0])
        sims = cosine_similarity(X[start:end], A)
        for bi in range(sims.shape[0]):
            row_idx = start + bi
            s = sims[bi]
            idx = np.argpartition(-s, K - 1)[:K]
            idx = idx[np.argsort(-s[idx])]
            # Convert the K local similarities into a probability vector.  This
            # ensures each record contributes total mass one, which is the
            # sensitivity condition used by dp_mechanisms.sensitivity().
            z = gamma * s[idx]
            z = z - np.max(z)
            w = np.exp(z)
            if not np.all(np.isfinite(w)) or w.sum() <= 0:
                w = np.ones_like(w, dtype=np.float64)
            w = w / w.sum()
            for j, val in zip(idx, w):
                rows.append(row_idx)
                cols.append(int(j))
                vals.append(float(val))
                C[int(j)] += float(val)

    V = sparse.csr_matrix((vals, (rows, cols)), shape=(len(records), len(atoms)))
    return C, V


def main() -> None:
    """Write an aggregate semantic sketch and sparse assignment matrix from the command line."""
    parser = argparse.ArgumentParser(description="Compute sparse semantic assignments and non-private sketch.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--atoms", required=True)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--public-fit-records", default=None, help="Optional public/auxiliary JSONL corpus used only to fit the TF-IDF vectorizer.")
    parser.add_argument("--allow-private-fit", action="store_true", help="Non-private demo only: fit TF-IDF on protected records too.")
    parser.add_argument("--output", required=True, help="Output .npz file with C and V.")
    args = parser.parse_args()

    records = read_jsonl(args.records)
    atoms = read_json(args.atoms)
    public_fit_texts = None
    if args.public_fit_records:
        public_fit_texts = [text_field(r) for r in read_jsonl(args.public_fit_records)]
    C, V = topK_sparse_assignments(records, atoms, args.K, args.gamma, public_fit_texts=public_fit_texts, allow_private_fit=args.allow_private_fit)
    ensure_dir(Path(args.output).parent)
    sparse.save_npz(str(Path(args.output).with_suffix(".V.npz")), V)
    np.savez(
        args.output,
        C=C,
        K=args.K,
        gamma=args.gamma,
        n=len(records),
        m=len(atoms),
        public_fit_records=args.public_fit_records,
        allow_private_fit=bool(args.allow_private_fit),
    )
    print(f"Wrote sketch to {args.output}")
    print(f"Wrote sparse assignments to {Path(args.output).with_suffix('.V.npz')}")


if __name__ == "__main__":
    main()
