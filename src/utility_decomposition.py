"""Utility decompositions separating semantic, clipping, and DP release error."""
from __future__ import annotations

import numpy as np


def _vec(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def norm_error(a, b, ord: int | float = 1) -> float:
    a = _vec(a, "a")
    b = _vec(b, "b")
    if a.shape != b.shape:
        raise ValueError("vectors must have the same shape")
    return float(np.linalg.norm(a - b, ord=ord))


def record_level_decomposition(oracle, assigned, released, ord: int | float = 1) -> dict:
    """Report oracle-to-release error and its triangle-inequality components."""
    semantic = norm_error(assigned, oracle, ord)
    dp = norm_error(released, assigned, ord)
    total = norm_error(released, oracle, ord)
    return {
        "norm_order": ord,
        "total_oracle_to_release_error": total,
        "assignment_dictionary_error": semantic,
        "dp_release_error": dp,
        "triangle_upper_bound": semantic + dp,
        "triangle_inequality_verified": total <= semantic + dp + 1e-10,
    }


def user_level_decomposition(oracle, unclipped, clipped, released, ord: int | float = 1) -> dict:
    """Report semantic, clipping, and DP components for user-level release."""
    semantic = norm_error(unclipped, oracle, ord)
    clipping = norm_error(clipped, unclipped, ord)
    dp = norm_error(released, clipped, ord)
    total = norm_error(released, oracle, ord)
    return {
        "norm_order": ord,
        "total_oracle_to_release_error": total,
        "assignment_dictionary_error": semantic,
        "clipping_error": clipping,
        "dp_release_error": dp,
        "triangle_upper_bound": semantic + clipping + dp,
        "triangle_inequality_verified": total <= semantic + clipping + dp + 1e-10,
    }


def radial_l1_clipping_distortion(user_vectors, bound: float) -> dict:
    """Characterize exact l1 distortion from radial clipping of nonnegative users."""
    U = np.asarray(user_vectors, dtype=float)
    if U.ndim != 2 or np.any(U < -1e-12):
        raise ValueError("user_vectors must be a nonnegative 2-D array")
    if bound <= 0:
        raise ValueError("bound must be positive")
    masses = np.maximum(U, 0.0).sum(axis=1)
    removed = np.maximum(masses - float(bound), 0.0)
    scales = np.ones_like(masses)
    positive = masses > 0
    scales[positive] = np.minimum(1.0, float(bound) / masses[positive])
    clipped = U * scales[:, None]
    exact = float(np.linalg.norm(U.sum(axis=0) - clipped.sum(axis=0), ord=1))
    formula = float(removed.sum())
    return {
        "n_users": int(U.shape[0]),
        "bound": float(bound),
        "n_users_clipped": int(np.sum(masses > bound)),
        "fraction_users_clipped": float(np.mean(masses > bound)) if U.shape[0] else 0.0,
        "total_mass_removed": formula,
        "aggregate_l1_distortion": exact,
        "formula_verified": bool(abs(exact - formula) <= 1e-8),
        "scales": scales.tolist(),
    }
