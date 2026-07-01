"""Construct the decoder-visible semantic plan from a privatized sketch.

The released plan contains admitted public atoms, released noisy masses, and prevalence bins derived using a public denominator. 
For a DP-SPIN private run, this plan is the only protected-data-dependent object supplied to the decoder. 
Raw records, per-record contributions, non-private representatives, and non-admitted atom labels are excluded.
"""
from __future__ import annotations

from typing import Any
import numpy as np

from .dp_mechanisms import topL


def prevalence_bin(p: float | None, bins: dict[str, float] | None = None) -> str | None:
    """Map a nonnegative noisy mass to a coarse prevalence bin using a public denominator.
    """
    if p is None:
        return None
    if bins is None:
        bins = {"very common": 0.20, "common": 0.10, "moderate": 0.05, "low": 0.0}
    # Sort from largest threshold to smallest.
    for name, threshold in sorted(bins.items(), key=lambda kv: -kv[1]):
        if p >= float(threshold):
            return str(name)
    return "low"


def make_private_plan(
    C_tilde: np.ndarray,
    atoms: list[dict[str, Any]],
    L: int,
    public_denominator: int | None = None,
    bins: dict[str, float] | None = None,
    clip_nonnegative: bool = True,
) -> dict[str, Any]:
    """Select top atoms from the privatized sketch and build the decoder-visible plan.
    """
    # Select using the already privatized sketch.  This selection is
    # post-processing for dense mechanisms and part of the sparse release format
    # for private_topL, where unselected entries are -inf.
    idx = topL(C_tilde, L)
    selected = []
    for rank, j in enumerate(idx, start=1):
        noisy_mass = float(C_tilde[j])
        display_mass = max(0.0, noisy_mass) if clip_nonnegative else noisy_mass
        p = display_mass / public_denominator if public_denominator else None
        atom = atoms[int(j)]
        selected.append({
            "rank": rank,
            "atom_id": int(atom.get("atom_id", j)),
            "index": int(j),
            "label": str(atom.get("label", f"atom_{j}")),
            "description": str(atom.get("description", "")),
            "noisy_mass": display_mass,
            "raw_noisy_mass": noisy_mass,
            "prevalence": p,
            "prevalence_bin": prevalence_bin(p, bins),
        })
    return {
        "selected_atoms": selected,
        "L": int(L),
        "public_denominator": public_denominator,
        "clip_nonnegative": clip_nonnegative,
    }
