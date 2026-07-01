"""Differentially private release mechanisms for aggregate semantic sketches.

The input ``C`` is the internal non-private semantic sketch over a fixed public atom universe. 
``contribution_bound`` is the maximum ``l1`` mass contributed by one protected unit after aggregation and clipping: one for record-level runs and
``B`` for user-level runs.

The implemented mechanisms are:

* the Gaussian sketch mechanism, which perturbs the full sketch and admits the top-``L`` coordinates by post-processing;
* the Laplace sketch mechanism, which applies the analogous full-sketch release;
* the sparse private atom-admission mechanism, which spends part of the privacy budget on sequential exponential-mechanism admission and the remainder on noisy release of the admitted masses.

Each release returns mechanism metadata recording the configured and effective privacy parameters, adjacency relation, sensitivities, calibration method, and budget split.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

Adjacency = Literal["adddrop", "replacement"]
GaussianCalibration = Literal["analytic", "classical"]

MECHANISM_LABELS = {
    "gaussian": r"Gaussian sketch + Top-$L$ atoms",
    "laplace": r"Laplace sketch + Top-$L$ atoms",
    "private_topL": r"Exponential-mechanism Top-$L$ atoms + Gaussian values",
}

MECHANISM_DISPLAY_NAMES = {
    "gaussian": "Gaussian sketch + Top-L atoms",
    "laplace": "Laplace sketch + Top-L atoms",
    "private_topL": "Exponential-mechanism Top-L atoms + Gaussian values",
}


def normalize_mechanism_name(mechanism: str) -> str:
    """Normalize mechanism aliases used in configurations and saved reports.
    
    ``private_topL`` is the current key for sparse private atom admission. The aliases
    are accepted for compatibility with earlier outputs. ``K`` remains assignment
    sparsity, and ``L`` remains released plan size.
    """
    key = str(mechanism).strip()
    aliases = {
        "private_topk": "private_topL",
        "private_topK": "private_topL",
        "private_topl": "private_topL",
        "private_top_l": "private_topL",
        "private_top-l": "private_topL",
        "sparse_private_topL": "private_topL",
    }
    return aliases.get(key, key)

_SQRT2 = math.sqrt(2.0)


def _validate_privacy(epsilon: float, delta: float | None = None) -> None:
    """Validate epsilon and optional delta parameters used by DP mechanisms.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    if delta is not None and not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")


def _as_1d_float_array(values: np.ndarray, name: str = "values") -> np.ndarray:
    """Return a finite one-dimensional float array view/copy for mechanisms."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _phi(x: float) -> float:
    """Standard normal CDF using only the Python standard library."""
    return 0.5 * math.erfc(-x / _SQRT2)


def _gaussian_delta(epsilon: float, mu: float) -> float:
    """Exact hockey-stick divergence for Gaussian shift parameter mu=Delta/sigma.

    For the Gaussian mechanism with global l2 sensitivity Delta and noise
    standard deviation sigma, mu = Delta / sigma.  The mechanism is
    (epsilon, delta)-DP when this quantity is at most delta.
    """
    if mu <= 0:
        return 0.0
    # delta = Phi(mu/2 - eps/mu) - exp(eps) Phi(-mu/2 - eps/mu).
    a = mu / 2.0 - epsilon / mu
    b = -mu / 2.0 - epsilon / mu
    # For the epsilon values used in this project exp(epsilon) is safe.  This
    # guard avoids overflow if a user passes an extremely large epsilon.
    try:
        exp_eps = math.exp(epsilon)
    except OverflowError:
        exp_eps = float("inf")
    val = _phi(a) - exp_eps * _phi(b)
    # Numerical roundoff can produce tiny negative values near zero.
    return float(max(0.0, min(1.0, val)))


def analytic_gaussian_sigma(epsilon: float, delta: float, l2_sensitivity: float = 1.0) -> float:
    """Calibrate Gaussian noise by inverting the analytic Gaussian condition.

    This implements a monotone binary search for the largest shift
    mu = Delta/sigma whose exact Gaussian privacy loss is at most `delta`,
    then returns sigma = Delta/mu.  It is suitable for all epsilon > 0 and
    avoids the classical bound's epsilon < 1 restriction.
    """
    _validate_privacy(epsilon, delta)
    if l2_sensitivity < 0 or not np.isfinite(l2_sensitivity):
        raise ValueError("l2_sensitivity must be nonnegative and finite")
    if l2_sensitivity == 0:
        return 0.0

    lo = 0.0  # always feasible
    hi = 1.0
    # Increase hi until it is infeasible, i.e., delta(eps, hi) > target delta.
    while _gaussian_delta(epsilon, hi) <= delta:
        lo = hi
        hi *= 2.0
        if hi > 1e6:
            break

    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _gaussian_delta(epsilon, mid) <= delta:
            lo = mid
        else:
            hi = mid

    if lo <= 0:
        # Extremely conservative fallback; this should not occur for valid
        # epsilon/delta pairs in normal project settings.
        return classical_gaussian_sigma(min(epsilon, 0.999999), delta, l2_sensitivity)
    return float(l2_sensitivity / lo)


def classical_gaussian_sigma(epsilon: float, delta: float, l2_sensitivity: float = 1.0) -> float:
    """Textbook Gaussian calibration.

    This implements the common sufficient bound
    sigma = Delta_2 * sqrt(2 log(1.25 / delta)) / epsilon.  The textbook theorem
    for this bound is usually stated for 0 < epsilon < 1.  The analytic
    calibration is the default for general epsilon > 0.  This option is retained
    only for controlled comparisons.
    """
    _validate_privacy(epsilon, delta)
    if not (0 < epsilon < 1):
        raise ValueError(
            "textbook Gaussian calibration is enabled only for 0 < epsilon < 1; "
            "use calibration='analytic' for general epsilon > 0"
        )
    if l2_sensitivity < 0 or not np.isfinite(l2_sensitivity):
        raise ValueError("l2_sensitivity must be nonnegative and finite")
    return float(l2_sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon)


def gaussian_sigma(
    epsilon: float,
    delta: float,
    l2_sensitivity: float = 1.0,
    calibration: GaussianCalibration = "analytic",
) -> float:
    """Dispatch to analytic or classical Gaussian noise calibration.
    """
    if calibration == "analytic":
        return analytic_gaussian_sigma(epsilon, delta, l2_sensitivity)
    if calibration == "classical":
        return classical_gaussian_sigma(epsilon, delta, l2_sensitivity)
    raise ValueError("calibration must be 'analytic' or 'classical'")


def gaussian_release(
    C: np.ndarray,
    epsilon: float,
    delta: float,
    l2_sensitivity: float = 1.0,
    rng=None,
    calibration: GaussianCalibration = "analytic",
) -> tuple[np.ndarray, dict]:
    """Add Gaussian noise calibrated to l2 sensitivity and return mechanism metadata.
    """
    rng = np.random.default_rng() if rng is None else rng
    C = _as_1d_float_array(C, "C")
    sigma = gaussian_sigma(epsilon, delta, l2_sensitivity, calibration=calibration)
    noise = rng.normal(0.0, sigma, size=C.shape)
    return C + noise, {
        "mechanism": "gaussian",
        "mechanism_label": MECHANISM_LABELS["gaussian"],
        "mechanism_display_name": MECHANISM_DISPLAY_NAMES["gaussian"],
        "release_mode": "full_sketch_noise_then_top_L_atoms",
        "plan_atom_admission": "Top-L atoms selected by post-processing of the noisy full sketch",
        "epsilon": float(epsilon),
        "delta": float(delta),
        "l2_sensitivity": float(l2_sensitivity),
        "sigma": float(sigma),
        "gaussian_calibration": calibration,
    }


def laplace_release(C: np.ndarray, epsilon: float, l1_sensitivity: float = 1.0, rng=None) -> tuple[np.ndarray, dict]:
    """Add Laplace noise calibrated to l1 sensitivity and return mechanism metadata.
    """
    _validate_privacy(epsilon)
    if l1_sensitivity < 0 or not np.isfinite(l1_sensitivity):
        raise ValueError("l1_sensitivity must be nonnegative and finite")
    rng = np.random.default_rng() if rng is None else rng
    C = _as_1d_float_array(C, "C")
    b = l1_sensitivity / epsilon
    noise = rng.laplace(0.0, b, size=C.shape)
    return C + noise, {
        "mechanism": "laplace",
        "mechanism_label": MECHANISM_LABELS["laplace"],
        "mechanism_display_name": MECHANISM_DISPLAY_NAMES["laplace"],
        "release_mode": "full_sketch_noise_then_top_L_atoms",
        "plan_atom_admission": "Top-L atoms selected by post-processing of the noisy full sketch",
        "epsilon": float(epsilon),
        "delta": 0.0,
        "l1_sensitivity": float(l1_sensitivity),
        "scale": float(b),
    }


def sensitivity(adjacency: Adjacency, contribution_bound: float = 1.0) -> dict[str, float]:
    """Sensitivities for a sketch with bounded protected-unit contribution.

    ``contribution_bound`` is the maximum l1 mass contributed by one protected
    unit after any aggregation and clipping.  For record-level DP this is 1
    because every record assignment has l1 mass at most one.  For user-level DP
    this is the user clipping bound B after all records from a user are summed
    and clipped.

    Add/drop adjacency changes the sketch by one nonnegative vector with l1
    norm at most B.  Replacement adjacency changes it by the difference of two
    such vectors.  Therefore l1 sensitivity is B for add/drop and 2B for
    replacement; l2 sensitivity is B for add/drop and sqrt(2)B for replacement.
    Coordinate sensitivity is B under both adjacency relations.
    """
    if contribution_bound <= 0 or not math.isfinite(float(contribution_bound)):
        raise ValueError("contribution_bound must be positive and finite")
    B = float(contribution_bound)
    if adjacency == "adddrop":
        return {"l1": B, "l2": B, "linf": B, "contribution_bound": B}
    if adjacency == "replacement":
        return {"l1": 2.0 * B, "l2": math.sqrt(2.0) * B, "linf": B, "contribution_bound": B}
    raise ValueError("adjacency must be 'adddrop' or 'replacement'")


def topL(values: np.ndarray, L: int) -> np.ndarray:
    """Return indices of the largest finite entries with deterministic tie-breaking."""
    if L <= 0:
        return np.array([], dtype=int)
    values = np.asarray(values, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        return np.array([], dtype=int)
    L_eff = min(int(L), finite_idx.size)
    # Sort by decreasing value and then increasing index.  The index tie-breaker
    # makes top-L post-processing deterministic across platforms.
    order = np.lexsort((finite_idx, -values[finite_idx]))
    return finite_idx[order[:L_eff]].astype(int)


def exponential_mechanism_topL_select(
    C: np.ndarray,
    L: int,
    epsilon: float,
    utility_sensitivity: float = 1.0,
    rng=None,
) -> tuple[np.ndarray, dict]:
    """Admit ``L`` distinct atoms by sequential exponential-mechanism sampling.
    
    At round ``t``, the utility of an unselected coordinate ``j`` is ``C_j(D)``.
    With protected-unit contribution bound ``B``, coordinate sensitivity is at most
    ``B`` under both add/drop and replacement adjacency. The selection budget is
    split uniformly across ``min(L, m)`` rounds, and sequential composition gives an
    ``(epsilon, 0)`` guarantee for the admitted index set.
    
    This is the repository's transparent sequential admission procedure, not an
    optimized heavy-hitter algorithm.
    """
    _validate_privacy(epsilon)
    C = _as_1d_float_array(C, "C")
    if utility_sensitivity <= 0 or not np.isfinite(utility_sensitivity):
        raise ValueError("utility_sensitivity must be positive and finite")
    if L < 0:
        raise ValueError("L must be nonnegative")

    rng = np.random.default_rng() if rng is None else rng
    m = int(C.shape[0])
    L_eff = min(int(L), m)
    if L_eff == 0:
        return np.array([], dtype=int), {
            "selection": "sequential_exponential_mechanism",
            "selection_mechanism": "sequential_exponential_mechanism_without_replacement",
            "atom_admission": "sequential_exponential_mechanism_without_replacement",
            "admitted_atom_count": 0,
            "epsilon_selection": float(epsilon),
            "epsilon_per_round": 0.0,
            "rounds_requested": int(L),
            "rounds_effective": 0,
            "utility": "coordinate_mass",
            "utility_sensitivity": float(utility_sensitivity),
            "selection_composition": "sequential_composition",
        }

    remaining = list(range(m))
    selected: list[int] = []
    eps_round = float(epsilon) / float(L_eff)

    for _ in range(L_eff):
        scores = C[np.asarray(remaining, dtype=int)]
        logits = eps_round * scores / (2.0 * float(utility_sensitivity))
        # Stable softmax.  The inputs have already been checked finite.
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        total = float(probs.sum())
        if not np.isfinite(total) or total <= 0:
            # This should not occur for finite inputs, but a uniform fallback is
            # safer than returning NaNs if numerical underflow ever occurs.
            probs = np.full(len(remaining), 1.0 / len(remaining), dtype=float)
        else:
            probs = probs / total
        k = int(rng.choice(len(remaining), p=probs))
        selected.append(remaining.pop(k))

    return np.asarray(selected, dtype=int), {
        "selection": "sequential_exponential_mechanism",
        "selection_mechanism": "sequential_exponential_mechanism_without_replacement",
        "atom_admission": "sequential_exponential_mechanism_without_replacement",
        "admitted_atom_count": int(L_eff),
        "epsilon_selection": float(epsilon),
        "epsilon_per_round": float(eps_round),
        "rounds_requested": int(L),
        "rounds_effective": int(L_eff),
        "utility": "coordinate_mass",
        "utility_sensitivity": float(utility_sensitivity),
        "selection_composition": "sequential_composition",
    }

def sparse_private_topL_release(
    C: np.ndarray,
    L: int,
    epsilon: float,
    delta: float = 1e-6,
    value_mechanism: str = "gaussian",
    value_epsilon_fraction: float = 0.5,
    adjacency: Adjacency = "adddrop",
    rng=None,
    gaussian_calibration: GaussianCalibration = "analytic",
    contribution_bound: float = 1.0,
    support_mechanism: str = "sequential",
) -> tuple[np.ndarray, dict]:
    """Release a sparse semantic plan by private atom admission and noisy value release.
    
    The mechanism first spends ``epsilon_selection`` to admit distinct atom indices
    with sequential exponential-mechanism sampling. It then spends
    ``epsilon_value`` to release the selected mass vector with the configured value
    mechanism. The two adaptive stages compose under the reported privacy budget.
    
    The returned array has length ``m``. Non-admitted coordinates are ``-np.inf`` so
    downstream code does not treat them as released numeric values.
    """
    _validate_privacy(epsilon)
    C = _as_1d_float_array(C, "C")
    rng = np.random.default_rng() if rng is None else rng
    if L < 0:
        raise ValueError("L must be nonnegative")
    if not (0.0 < value_epsilon_fraction < 1.0):
        raise ValueError("value_epsilon_fraction must be in (0, 1)")
    if support_mechanism not in {"sequential", "sequential_em", "sequential_exponential_mechanism"}:
        raise ValueError(
            "support_mechanism must be 'sequential'; this implementation uses "
            "sequential exponential-mechanism atom admission"
        )

    value_mechanism = value_mechanism.lower()
    if value_mechanism not in {"gaussian", "laplace"}:
        raise ValueError("value_mechanism must be 'gaussian' or 'laplace'")
    if value_mechanism == "gaussian":
        _validate_privacy(epsilon, delta)
        delta_value = float(delta)
        delta_effective = float(delta)
    else:
        # Laplace value release is pure DP.  Allow delta=0 for explicit pure-DP
        # sparse runs, but keep the configured value in metadata if nonzero.
        if not (0 <= float(delta) < 1):
            raise ValueError("delta must be in [0, 1) when value_mechanism='laplace'")
        delta_value = 0.0
        delta_effective = 0.0

    eps_val = float(epsilon) * float(value_epsilon_fraction)
    eps_sel = float(epsilon) - eps_val

    sens = sensitivity(adjacency, contribution_bound=contribution_bound)
    selected, sel_info = exponential_mechanism_topL_select(
        C,
        L,
        eps_sel,
        utility_sensitivity=sens["linf"],
        rng=rng,
    )

    C_sparse = np.full(C.shape, -np.inf, dtype=float)
    vals = C[selected].copy()
    if value_mechanism == "gaussian":
        noisy_vals, val_info = gaussian_release(
            vals,
            eps_val,
            delta_value,
            l2_sensitivity=sens["l2"],
            rng=rng,
            calibration=gaussian_calibration,
        )
    else:
        noisy_vals, val_info = laplace_release(
            vals,
            eps_val,
            l1_sensitivity=sens["l1"],
            rng=rng,
        )

    C_sparse[selected] = noisy_vals
    info = {
        "mechanism": "private_topL",
        "mechanism_label": MECHANISM_LABELS["private_topL"],
        "mechanism_display_name": MECHANISM_DISPLAY_NAMES["private_topL"],
        "release_mode": "em_top_L_atoms_then_noisy_selected_values",
        "plan_atom_admission": "Top-L atoms admitted by sequential exponential mechanism",
        "sparse_release": True,
        "sparse_release_type": "sequential_em_atom_admission_then_noisy_values",
        "epsilon": float(epsilon),
        "delta": float(delta_effective),
        "configured_delta": float(delta),
        "adjacency": adjacency,
        "contribution_bound": float(contribution_bound),
        "l1_sensitivity": float(sens["l1"]),
        "l2_sensitivity": float(sens["l2"]),
        "coordinate_sensitivity": float(sens["linf"]),
        "L_requested": int(L),
        "L_effective": int(selected.shape[0]),
        "admitted_atom_count": int(selected.shape[0]),
        "selected": selected.astype(int).tolist(),
        "admitted_atom_indices": selected.astype(int).tolist(),
        "value_epsilon_fraction": float(value_epsilon_fraction),
        "epsilon_selection": float(eps_sel),
        "epsilon_value": float(eps_val),
        "delta_value": float(delta_value),
        "privacy_accounting": "sequential_composition: EM atom admission plus selected-value release",
        "atom_admission_release": sel_info,
        "selection_release": sel_info,
        "value_mechanism": value_mechanism,
        "value_release": val_info,
        # Backward-compatible top-level selection fields used by older summaries.
        **sel_info,
    }
    return C_sparse, info

def release(
    C: np.ndarray,
    mechanism: str,
    epsilon: float,
    delta: float = 1e-6,
    adjacency: Adjacency = "adddrop",
    rng=None,
    L: int | None = None,
    gaussian_calibration: GaussianCalibration = "analytic",
    contribution_bound: float = 1.0,
    **kwargs,
):
    """Dispatch to the configured sketch-release mechanism.
    """
    mechanism = normalize_mechanism_name(mechanism)
    if adjacency not in {"adddrop", "replacement"}:
        raise ValueError("adjacency must be 'adddrop' or 'replacement'")
    # All mechanisms below consume the same non-private sketch C.  The only difference
    # is how the DP release is formed: Gaussian/Laplace first perturb the full
    # sketch and later admit Top-L atoms by post-processing, while private_topL
    # first privatizes Top-L atom admission and then releases noisy selected
    # values.
    sens = sensitivity(adjacency, contribution_bound=contribution_bound)
    if mechanism == "gaussian":
        return gaussian_release(
            C,
            epsilon,
            delta,
            l2_sensitivity=sens["l2"],
            rng=rng,
            calibration=gaussian_calibration,
        )
    if mechanism == "laplace":
        return laplace_release(C, epsilon, l1_sensitivity=sens["l1"], rng=rng)
    if mechanism == "private_topL":
        if L is None:
            raise ValueError("L is required for private_topL")
        return sparse_private_topL_release(
            C,
            L=L,
            epsilon=epsilon,
            delta=delta,
            adjacency=adjacency,
            rng=rng,
            gaussian_calibration=gaussian_calibration,
            contribution_bound=contribution_bound,
            **kwargs,
        )
    raise ValueError("mechanism must be gaussian, laplace, or private_topL")
