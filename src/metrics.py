"""Compute semantic-plan, verifier, and controlled-probe metrics.

Dense full-sketch releases and sparse private atom-admission releases expose different numeric objects. 
Full-vector error is therefore reported only for dense releases; sparse releases are evaluated on admitted coordinates and support recovery. 
Summary metrics assess consistency with the released plan, not truth relative to raw protected records.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .dp_mechanisms import topL
from .utils import read_json, write_json


def sketch_metrics(C: np.ndarray, C_tilde: np.ndarray, L: int) -> dict[str, float | bool | int | str]:
    """Compute plan-level utility diagnostics for dense or sparse releases.

    Dense releases contain finite raw noisy values for all coordinates. Sparse
    releases such as private_topL use ``-inf`` for unselected coordinates. For
    sparse releases, full-vector error is not reported because
    unselected coordinates were not released as numeric values.

    Value-error metrics are computed from the raw noisy values saved in
    ``noisy_sketch.npy`` before optional nonnegative display clipping in the
    released plan. This convention matches the perturbation analysis; displayed
    masses and support bins may be computed from clipped nonnegative values.
    """
    non_private_top = topL(C, L)
    released_support = topL(C_tilde, L)
    overlap = len(set(non_private_top.tolist()) & set(released_support.tolist())) / max(1, L)
    mass_non_private_top = float(C[non_private_top].sum()) if len(non_private_top) else 0.0
    mass_released_support = float(C[released_support].sum()) if len(released_support) else 0.0
    mass_retained = mass_released_support / mass_non_private_top if mass_non_private_top > 0 else 0.0

    finite = np.isfinite(C_tilde)
    is_sparse = bool(np.any(~finite))
    out: dict[str, float | bool | int | str] = {
        "overlap_at_L": float(overlap),
        "mass_retained_fraction": float(mass_retained),
        "support_mass_loss": float(mass_non_private_top - mass_released_support),
        "num_released_coordinates": int(np.sum(finite)),
        "num_total_coordinates": int(C.shape[0]),
        "is_sparse_release": is_sparse,
        "value_error_convention": "raw_noisy_values_before_optional_nonnegative_display_clipping",
    }

    if np.any(finite):
        err = C_tilde[finite] - C[finite]
        prefix = "value_error_on_released_coordinates" if is_sparse else "full_vector_error"
        out[f"{prefix}_linf"] = float(np.max(np.abs(err)))
        out[f"{prefix}_l1"] = float(np.sum(np.abs(err)))
        out[f"{prefix}_l2"] = float(np.linalg.norm(err))
    else:
        out["value_error_on_released_coordinates_linf"] = float("nan")
        out["value_error_on_released_coordinates_l1"] = float("nan")
        out["value_error_on_released_coordinates_l2"] = float("nan")
    return out


def probe_summary_mention(summary: str, probe_strings: list[str]) -> dict[str, Any]:
    """Measure whether decoded summary text contains public probe strings."""
    low = summary.lower()
    mentioned: list[str] = []
    for c in probe_strings:
        c_low = c.lower()
        tokens = [t for t in c_low.replace('.', ' ').split() if len(t) >= 6]
        hit = c_low in low or any(t in low for t in tokens if any(ch.isdigit() for ch in t) or '-' in t)
        if hit:
            mentioned.append(c)
    return {
        "num_probes": len(probe_strings),
        "num_mentioned": len(mentioned),
        "mention_rate": len(mentioned) / max(1, len(probe_strings)),
        "mentioned": mentioned,
    }


def probe_plan_admission(
    plan: dict,
    probe_strings: list[str] | None = None,
    probe_atom_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Measure whether public probe atoms are admitted into the released plan.

    When probe atom identifiers are supplied, admission is computed by set
    membership in the admitted atom IDs. When identifiers are unavailable, the
    function falls back to deterministic public string matching over admitted
    atom labels and descriptions.
    """
    selected = plan.get("selected_atoms", []) or []
    admitted_ids = {int(a.get("atom_id", a.get("index", -1))) for a in selected}
    admitted_indices = {int(a.get("index", a.get("atom_id", -1))) for a in selected}
    admitted: list[Any] = []

    ids = [int(x) for x in (probe_atom_ids or [])]
    if ids:
        for cid in ids:
            if cid in admitted_ids or cid in admitted_indices:
                admitted.append(cid)
        denom = len(ids)
    else:
        probe_strings = probe_strings or []
        joined = " ".join(
            str(a.get("label", "")) + " " + str(a.get("description", ""))
            for a in selected
        ).lower()
        for c in probe_strings:
            c_low = c.lower()
            tokens = [
                t.strip(".,:;()[]{}")
                for t in c_low.replace("-", " ").split()
                if len(t.strip(".,:;()[]{}")) >= 5
            ]
            rare_tokens = [
                t for t in tokens
                if any(ch.isdigit() for ch in t) or t in {"blue", "lantern", "hyperion", "zorvian"}
            ]
            hit = c_low in joined or any(t in joined for t in rare_tokens)
            if hit:
                admitted.append(c)
        denom = len(probe_strings)

    rate = len(admitted) / max(1, denom)
    return {
        "num_probes": int(denom),
        "num_plan_admitted": len(admitted),
        "plan_admission_rate": rate,
        "plan_admitted": admitted,
        "admission_basis": "atom_id" if ids else "public_string_match",
    }


def summary_plan_metrics(summary: str, plan: dict, atoms: list[dict]) -> dict:
    """Compute public consistency statistics between a summary and a released plan."""
    from .verify import verify_summary
    return verify_summary(summary, plan, atoms)


def compute_run_metrics(run_dir: str) -> dict:
    """Load a run directory and compute available sketch, verifier, and probe metrics."""
    rd = Path(run_dir)
    C = np.load(rd / 'clean_sketch.npy')
    Ct = np.load(rd / 'noisy_sketch.npy')
    plan = read_json(rd / 'private_plan.json')
    atoms = read_json(rd / 'atoms_used.json')
    summary = (rd / 'summary.txt').read_text(encoding='utf-8')
    cfg = read_json(rd / 'config.json') if (rd / 'config.json').exists() else None
    L = int(plan.get('L', len(plan.get('selected_atoms', []))))
    out = {"sketch": sketch_metrics(C, Ct, L), "verifier": summary_plan_metrics(summary, plan, atoms)}
    probes: list[str] = []
    probe_atom_ids: list[int] = []
    if (rd / 'probes.json').exists():
        probes = read_json(rd / 'probes.json')
    elif cfg and cfg.get('probe_strings'):
        probes = cfg['probe_strings']
    if cfg and cfg.get('probe_atom_ids'):
        probe_atom_ids = [int(x) for x in cfg.get('probe_atom_ids')]
    elif cfg and cfg.get('probe_atom_id') is not None:
        probe_atom_ids = [int(cfg.get('probe_atom_id'))]
    if probes or probe_atom_ids:
        if probes:
            out['probe_summary'] = probe_summary_mention(summary, probes)
            out['probe'] = out['probe_summary']  # backward-compatible alias
        out['probe_plan'] = probe_plan_admission(plan, probe_strings=probes, probe_atom_ids=probe_atom_ids)
    return out


def main() -> None:
    """Command-line entry point for computing metrics for one run directory."""
    p = argparse.ArgumentParser(description="Compute metrics for one run directory.")
    p.add_argument('--run', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = compute_run_metrics(args.run)
    write_json(out, args.output)
    print(f"Wrote metrics to {args.output}")

if __name__ == '__main__':
    main()
