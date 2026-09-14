"""Execute one DP-SPIN experiment.

The runner loads a protected target split and a fixed public or auxiliary atom dictionary, computes bounded semantic contributions, aggregates the internal
semantic sketch, releases a differentially private semantic plan, verbalizes the plan, applies the public verifier, and writes run artifacts.

The configured public denominator must be independent of the protected data under add/drop adjacency. Representation fitting uses public or auxiliary text
by default, and decoder access is limited to the released plan and public instructions.
"""
from __future__ import annotations

import argparse
import heapq
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from .assign import topK_sparse_assignments
from .embed import SUPPORTED_EMBEDDING_FAMILIES
from .claims import decode_structured_claims, render_verified_claims, verify_structured_claims, claims_from_plan_template
from .decode import decode_plan, template_decode
from .dp_mechanisms import release, normalize_mechanism_name
from .metrics import sketch_metrics, probe_summary_mention, probe_plan_admission
from .model_roles import validate_distinct_decoder_judge
from .plan import make_private_plan
from .summary_eval import build_clean_reference_plan, compute_summary_text_evaluation
from .utils import read_json, read_jsonl, iter_jsonl, write_json, write_jsonl, save_yaml, ensure_dir, load_yaml, run_name, set_seed, text_field
from .verify import verify_summary


def _resolve_public_denominator(cfg: dict, n_records_effective: int) -> int | None:
    """Return the denominator used for support bins while respecting the adjacency model.
    """
    denom = cfg.get("public_denominator")
    if denom is not None:
        denom = int(denom)
        if denom <= 0:
            raise ValueError("public_denominator must be positive")
        return denom

    adjacency = str(cfg.get("adjacency", "adddrop"))
    if adjacency == "adddrop" and not cfg.get("allow_private_denominator_for_demo", False):
        raise ValueError(
            "public_denominator is required under add/drop adjacency. "
            "Using len(records) silently can make the corpus size a released statistic. "
            "Set public_denominator explicitly in the config, or set "
            "allow_private_denominator_for_demo: true only for non-private demos."
        )

    # Replacement adjacency normally has a fixed public n; this fallback is kept
    # for controlled experiments and demos.
    return int(n_records_effective)


def _atom_source_summary(atoms: list[dict]) -> dict:
    """Summarize atom provenance metadata stored in the atom dictionary.
    """
    roles = sorted({str(a.get("atom_source_role", "unspecified")) for a in atoms})
    cols = sorted({c for a in atoms for c in a.get("atom_source_columns", [])})
    return {"atom_source_roles": roles, "atom_source_columns": cols}



def _record_key(record: dict[str, Any], index: int) -> str:
    """Stable key used for deterministic record sampling."""
    rid = record.get("id") or record.get("record_id") or record.get("complaint_id")
    if rid is None:
        rid = f"idx:{index}:" + str(record.get("text", ""))[:200]
    return str(rid)


def _assignment_prediction_rows(
    records: list[dict[str, Any]],
    assignments: sparse.csr_matrix,
    atoms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create evaluation-only ordered assignment predictions.

    The returned rows contain no raw text and are written only to
    ``offline_evaluation``.  Annotators should use a separately created blinded
    annotation batch when raw record text is needed.
    """
    if assignments.shape != (len(records), len(atoms)):
        raise ValueError("assignment matrix shape does not match records and atoms")
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row = assignments.getrow(index)
        order = np.argsort(-row.data, kind="stable") if row.nnz else np.asarray([], dtype=int)
        atom_indices = row.indices[order].astype(int).tolist() if row.nnz else []
        weights = row.data[order].astype(float).tolist() if row.nnz else []
        rows.append({
            "record_id": _record_key(record, index),
            "predicted_atom_ids": [int(atoms[j].get("atom_id", j)) for j in atom_indices],
            "predicted_weights": weights,
            "abstained": not atom_indices,
            "row_mass": float(sum(weights)),
        })
    return rows


def _select_records(records: list[dict[str, Any]], cfg: dict, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a reproducible subset without defaulting to first-n truncation.

    Modes:
      - hash (default): order by SHA256(seed, stable record key), then take n.
      - random: sample without replacement using NumPy RNG and seed.
      - first: deterministic first-n truncation, retained only for legacy runs.
    """
    n_requested = cfg.get("n_records")
    info: dict[str, Any] = {
        "selection_mode": "all",
        "n_available": len(records),
        "n_requested": None,
        "seed": seed,
    }
    if not n_requested:
        info["n_selected"] = len(records)
        return records, info

    n_requested_int = int(n_requested)
    if n_requested_int > len(records) and cfg.get("strict_sample_size", True):
        raise ValueError(
            f"Requested n_records={n_requested_int}, but only {len(records)} records are available. "
            "Lower n_records, provide more data, or set strict_sample_size: false for demos."
        )
    n = min(n_requested_int, len(records))
    mode = str(cfg.get("record_selection", "hash"))
    info.update({"selection_mode": mode, "n_requested": int(n_requested), "n_selected": n})
    if mode == "first":
        return records[:n], info
    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(records), size=n, replace=False))
        info["selected_indices"] = idx.astype(int).tolist()[:100]
        return [records[int(i)] for i in idx], info
    if mode == "hash":
        keyed = []
        for i, rec in enumerate(records):
            key = f"{seed}:{_record_key(rec, i)}".encode("utf-8", errors="ignore")
            digest = hashlib.sha256(key).hexdigest()
            keyed.append((digest, i, rec))
        keyed.sort(key=lambda x: (x[0], x[1]))
        selected = keyed[:n]
        selected.sort(key=lambda x: x[1])
        info["selected_indices"] = [int(i) for _, i, _ in selected[:100]]
        return [rec for _, _, rec in selected], info
    raise ValueError("record_selection must be one of: hash, random, first")



def _select_records_from_jsonl(path: str | Path, cfg: dict, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Streaming version of :func:`_select_records` for large JSONL files.

    For ``record_selection: hash`` this returns exactly the same selected records
    as loading the full list and sorting by ``(SHA256(seed, record_key), index)``.
    It keeps only ``n_records`` records in memory.
    """
    n_requested = cfg.get("n_records")
    if not n_requested:
        records = read_jsonl(path)
        return _select_records(records, cfg, seed)

    n_requested_int = int(n_requested)
    mode = str(cfg.get("record_selection", "hash"))
    if mode not in {"first", "random", "hash"}:
        raise ValueError("record_selection must be one of: hash, random, first")

    if mode == "random":
        # Exact compatibility with the old implementation: choose indices using
        # np.random.default_rng(seed).choice(n_available, ...), then load only
        # those indices in a second pass.
        n_available = sum(1 for _ in iter_jsonl(path))
        if n_requested_int > n_available and cfg.get("strict_sample_size", True):
            raise ValueError(
                f"Requested n_records={n_requested_int}, but only {n_available} records are available. "
                "Lower n_records, provide more data, or set strict_sample_size: false for demos."
            )
        n = min(n_requested_int, n_available)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_available, size=n, replace=False))
        wanted = set(int(i) for i in idx)
        selected = [rec for i, rec in enumerate(iter_jsonl(path)) if i in wanted]
        info = {
            "selection_mode": mode,
            "n_available": n_available,
            "n_requested": int(n_requested),
            "n_selected": n,
            "seed": seed,
            "selected_indices": idx.astype(int).tolist()[:100],
            "streaming_selection": True,
        }
        return selected, info

    n_available = 0
    if mode == "first":
        selected_pairs: list[tuple[int, dict[str, Any]]] = []
        for i, rec in enumerate(iter_jsonl(path)):
            n_available += 1
            if len(selected_pairs) < n_requested_int:
                selected_pairs.append((i, rec))
        n = min(n_requested_int, n_available)
    else:
        # Keep the n smallest hash keys using a max heap.  This is equivalent to
        # sorting every record by (digest, index) and taking the first n, but it
        # does not materialize the full corpus.
        heap: list[tuple[int, int, int, dict[str, Any]]] = []
        for i, rec in enumerate(iter_jsonl(path)):
            n_available += 1
            digest = hashlib.sha256(
                f"{seed}:{_record_key(rec, i)}".encode("utf-8", errors="ignore")
            ).hexdigest()
            digest_int = int(digest, 16)
            item = (-digest_int, -i, i, rec)
            if len(heap) < n_requested_int:
                heapq.heappush(heap, item)
            else:
                current_key = (digest_int, i)
                largest_key = (-heap[0][0], -heap[0][1])
                if current_key < largest_key:
                    heapq.heapreplace(heap, item)
        n = min(n_requested_int, n_available)
        selected_pairs = sorted(((i, rec) for _, _, i, rec in heap), key=lambda x: x[0])

    if n_requested_int > n_available and cfg.get("strict_sample_size", True):
        raise ValueError(
            f"Requested n_records={n_requested_int}, but only {n_available} records are available. "
            "Lower n_records, provide more data, or set strict_sample_size: false for demos."
        )
    selected_pairs = selected_pairs[:n]
    info = {
        "selection_mode": mode,
        "n_available": n_available,
        "n_requested": int(n_requested),
        "n_selected": n,
        "seed": seed,
        "selected_indices": [int(i) for i, _ in selected_pairs[:100]],
        "streaming_selection": True,
    }
    return [rec for _, rec in selected_pairs], info




def _protected_unit(cfg: dict) -> str:
    """Normalize the protected-unit configuration."""
    unit = str(cfg.get("protected_unit", "record")).lower().replace("-", "_")
    aliases = {
        "record": "record",
        "text_record": "record",
        "conversation": "record",
        "user": "user",
        "user_level": "user",
    }
    if unit not in aliases:
        raise ValueError("protected_unit must be one of: record, text_record, conversation, user")
    return aliases[unit]



def _lookup_record_field(record: dict[str, Any], field: str) -> Any:
    """Lookup a possibly nested field for user grouping.

    Supported forms:
      - ``user_id`` for a top-level key;
      - ``metadata.Customer ID`` for a nested metadata key;
      - any field present directly in ``record['metadata']``.
    """
    if field in record:
        return record[field]
    if "." in field:
        cur: Any = record
        for part in field.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                cur = None
                break
        if cur not in {None, ""}:
            return cur
    meta = record.get("metadata")
    if isinstance(meta, dict) and field in meta:
        return meta[field]
    return None

def _user_key(record: dict[str, Any], cfg: dict, index: int) -> str:
    """Return the configured user identifier for a record."""
    field = str(cfg.get("user_id_field", "user_id"))
    value = _lookup_record_field(record, field)
    if value not in {None, ""}:
        return str(value)
    fallback_fields = cfg.get("user_id_fallback_fields", []) or []
    for f in fallback_fields:
        value = _lookup_record_field(record, str(f))
        if value not in {None, ""}:
            return str(value)
    if cfg.get("allow_record_id_as_user_id_for_demo", False):
        return "record_as_user:" + _record_key(record, index)
    raise KeyError(
        f"User-level DP requires a user id field. Missing configured user_id_field='{field}'. "
        "Set user_id_field, provide user_id_fallback_fields, or use "
        "allow_record_id_as_user_id_for_demo only for non-user-level demos."
    )


def _group_records_by_user(records: list[dict[str, Any]], cfg: dict) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    """Group records by user id while retaining original record indices."""
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for i, rec in enumerate(records):
        uid = _user_key(rec, cfg, i)
        groups.setdefault(uid, []).append((i, rec))
    return groups


def _select_users(
    groups: dict[str, list[tuple[int, dict[str, Any]]]],
    cfg: dict,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a reproducible subset of users and include all records from them.

    This is used only for user-level experiments.  The number of selected users,
    not the number of selected records, is the public sample-size control.
    """
    users = sorted(groups.keys())
    n_requested = cfg.get("n_users")
    info: dict[str, Any] = {
        "selection_mode": "all_users",
        "n_users_available": len(users),
        "n_users_requested": None,
        "seed": seed,
    }
    if cfg.get("n_records") is not None:
        raise ValueError(
            "For protected_unit='user', remove n_records and use n_users. "
            "Selecting individual records can split a user's protected contribution."
        )
    if not n_requested:
        selected_users = users
    else:
        n_requested_int = int(n_requested)
        if n_requested_int > len(users) and cfg.get("strict_sample_size", True):
            raise ValueError(
                f"Requested n_users={n_requested_int}, but only {len(users)} users are available. "
                "Lower n_users, provide more data, or set strict_sample_size: false for demos."
            )
        n = min(n_requested_int, len(users))
        mode = str(cfg.get("user_selection", cfg.get("record_selection", "hash")))
        info.update({"selection_mode": mode, "n_users_requested": int(n_requested), "n_users_selected": n})
        if mode == "first":
            selected_users = users[:n]
        elif mode == "random":
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(users), size=n, replace=False))
            selected_users = [users[int(i)] for i in idx]
            info["selected_user_positions"] = idx.astype(int).tolist()[:100]
        elif mode == "hash":
            keyed = []
            for i, uid in enumerate(users):
                digest = hashlib.sha256(f"{seed}:{uid}".encode("utf-8", errors="ignore")).hexdigest()
                keyed.append((digest, i, uid))
            keyed.sort(key=lambda x: (x[0], x[1]))
            chosen = keyed[:n]
            chosen.sort(key=lambda x: x[1])
            selected_users = [uid for _, _, uid in chosen]
            info["selected_user_positions"] = [int(i) for _, i, _ in chosen[:100]]
        else:
            raise ValueError("user_selection must be one of: hash, random, first")
    selected_set = set(selected_users)
    selected_records_with_idx = [pair for uid in users if uid in selected_set for pair in groups[uid]]
    selected_records_with_idx.sort(key=lambda pair: pair[0])
    records = [rec for _, rec in selected_records_with_idx]
    info.setdefault("n_users_selected", len(selected_users))
    info["n_records_selected"] = len(records)
    info["selected_user_ids_preview"] = selected_users[:20]
    return records, info



def _select_users_from_jsonl(path: str | Path, cfg: dict, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Streaming user-level selector for large JSONL files.

    This matches the old selection rule without keeping every record in memory:
    first collect the unique user ids, then select users, then load records only
    for the selected users.
    """
    if cfg.get("n_records") is not None:
        raise ValueError(
            "For protected_unit='user', remove n_records and use n_users. "
            "Selecting individual records can split a user's protected contribution."
        )
    n_requested = cfg.get("n_users")
    if not n_requested:
        records = read_jsonl(path)
        groups = _group_records_by_user(records, cfg)
        return _select_users(groups, cfg, seed)

    seen_users: set[str] = set()
    n_records_available = 0
    for i, rec in enumerate(iter_jsonl(path)):
        n_records_available += 1
        seen_users.add(_user_key(rec, cfg, i))
    users = sorted(seen_users)
    n_requested_int = int(n_requested)
    if n_requested_int > len(users) and cfg.get("strict_sample_size", True):
        raise ValueError(
            f"Requested n_users={n_requested_int}, but only {len(users)} users are available. "
            "Lower n_users, provide more data, or set strict_sample_size: false for demos."
        )
    n = min(n_requested_int, len(users))
    mode = str(cfg.get("user_selection", cfg.get("record_selection", "hash")))
    if mode == "first":
        selected_users = users[:n]
        selected_positions = list(range(min(n, 100)))
    elif mode == "random":
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(users), size=n, replace=False))
        selected_users = [users[int(i)] for i in idx]
        selected_positions = idx.astype(int).tolist()[:100]
    elif mode == "hash":
        keyed = []
        for i, uid in enumerate(users):
            digest = hashlib.sha256(f"{seed}:{uid}".encode("utf-8", errors="ignore")).hexdigest()
            keyed.append((digest, i, uid))
        keyed.sort(key=lambda x: (x[0], x[1]))
        chosen = keyed[:n]
        chosen.sort(key=lambda x: x[1])
        selected_users = [uid for _, _, uid in chosen]
        selected_positions = [int(i) for _, i, _ in chosen[:100]]
    else:
        raise ValueError("user_selection must be one of: hash, random, first")

    selected_set = set(selected_users)
    selected_records_with_idx: list[tuple[int, dict[str, Any]]] = []
    for i, rec in enumerate(iter_jsonl(path)):
        if _user_key(rec, cfg, i) in selected_set:
            selected_records_with_idx.append((i, rec))
    selected_records_with_idx.sort(key=lambda pair: pair[0])
    records = [rec for _, rec in selected_records_with_idx]
    info = {
        "selection_mode": mode,
        "n_users_available": len(users),
        "n_users_requested": int(n_requested),
        "n_users_selected": len(selected_users),
        "n_records_available": n_records_available,
        "n_records_selected": len(records),
        "seed": seed,
        "selected_user_positions": selected_positions,
        "selected_user_ids_preview": selected_users[:20],
        "streaming_selection": True,
    }
    return records, info


def _embedding_fit_texts_from_jsonl(cfg: dict, seed: int):
    """Return public/auxiliary fit texts, optionally using a bounded public subset."""
    path = cfg.get("embedding_fit_records")
    if not path:
        return None, {"embedding_fit_records": None}
    max_records = cfg.get("embedding_fit_max_records")
    if max_records is None:
        return (text_field(r) for r in iter_jsonl(path)), {
            "embedding_fit_records": str(path),
            "embedding_fit_selection": "all_streamed",
            "embedding_fit_max_records": None,
        }
    fit_seed = int(cfg.get("embedding_fit_seed", 1729))
    fit_cfg = {
        "n_records": int(max_records),
        "record_selection": str(cfg.get("embedding_fit_selection", "hash")),
        "strict_sample_size": bool(cfg.get("strict_sample_size", True)),
    }
    fit_records, fit_info = _select_records_from_jsonl(path, fit_cfg, fit_seed)
    return (text_field(r) for r in fit_records), {
        "embedding_fit_records": str(path),
        "embedding_fit_selection": fit_info,
        "embedding_fit_max_records": int(max_records),
        "embedding_fit_seed": fit_seed,
    }




def _clip_user_assignments(
    V: sparse.csr_matrix,
    records: list[dict[str, Any]],
    cfg: dict,
) -> tuple[np.ndarray, sparse.csr_matrix, dict[str, Any], np.ndarray]:
    """Aggregate record assignments by user and apply radial l1 clipping.

    Returns ``(clipped_sketch, clipped_user_matrix, diagnostics,
    unclipped_sketch)``.  Diagnostics expose clipping bias rather than hiding it
    inside the non-private reference.
    """
    B = float(cfg.get("user_contribution_bound", cfg.get("user_clip_B", 1.0)))
    if B <= 0 or not np.isfinite(B):
        raise ValueError("user_contribution_bound must be positive and finite")
    groups = _group_records_by_user(records, cfg)
    user_ids = sorted(groups.keys())
    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    unclipped_masses: list[float] = []
    clipped_masses: list[float] = []
    scales: list[float] = []
    unclipped_sketch = np.zeros(V.shape[1], dtype=float)
    n_clipped = 0
    for user_row, uid in enumerate(user_ids):
        rec_idx = [i for i, _ in groups[uid]]
        u_raw = np.asarray(V[rec_idx, :].sum(axis=0)).ravel()
        u_raw = np.maximum(u_raw, 0.0)
        unclipped_sketch += u_raw
        mass = float(u_raw.sum())
        unclipped_masses.append(mass)
        scale = min(1.0, B / mass) if mass > 0.0 else 1.0
        u = u_raw * scale
        if scale < 1.0:
            n_clipped += 1
        mass_after = float(u.sum())
        clipped_masses.append(mass_after)
        scales.append(float(scale))
        nz = np.flatnonzero(u)
        for j in nz:
            val = float(u[j])
            if val != 0.0:
                row_indices.append(user_row)
                col_indices.append(int(j))
                values.append(val)
    U = sparse.csr_matrix((values, (row_indices, col_indices)), shape=(len(user_ids), V.shape[1]))
    C_user = np.asarray(U.sum(axis=0)).ravel()
    removed = np.maximum(np.asarray(unclipped_masses) - B, 0.0)
    scale_arr = np.asarray(scales, dtype=float) if scales else np.asarray([1.0])
    mass_arr = np.asarray(unclipped_masses, dtype=float) if unclipped_masses else np.asarray([0.0])
    info = {
        "protected_unit": "user",
        "user_id_field": str(cfg.get("user_id_field", "user_id")),
        "user_contribution_bound": B,
        "n_users": len(user_ids),
        "n_records": len(records),
        "n_users_clipped": int(n_clipped),
        "fraction_users_clipped": float(n_clipped / max(1, len(user_ids))),
        "total_unclipped_mass": float(np.sum(unclipped_masses)),
        "total_clipped_mass": float(np.sum(clipped_masses)),
        "total_mass_removed": float(removed.sum()),
        "clipping_l1_distortion": float(np.linalg.norm(unclipped_sketch - C_user, ord=1)),
        "mass_removed_fraction": float(removed.sum() / max(np.sum(unclipped_masses), np.finfo(float).tiny)),
        "max_unclipped_user_mass": float(mass_arr.max()),
        "max_clipped_user_mass": float(max(clipped_masses) if clipped_masses else 0.0),
        "unclipped_user_mass_quantiles": {
            str(q): float(np.quantile(mass_arr, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        },
        "clipping_scale_quantiles": {
            str(q): float(np.quantile(scale_arr, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        },
        "mean_records_per_user": float(len(records) / max(1, len(user_ids))),
        "users_with_multiple_records": int(sum(1 for uid in user_ids if len(groups[uid]) >= 2)),
        "user_level_dp_meaningful_for_dataset": bool(any(len(groups[uid]) >= 2 for uid in user_ids)),
        "user_ids_preview": user_ids[:20],
    }
    return C_user, U, info, unclipped_sketch

def _verifier_feedback(verifier: dict[str, Any]) -> str:
    """Convert verifier output into public feedback for plan-only regeneration."""
    parts = []
    labels = verifier.get("unsupported_atom_labels") or []
    if labels:
        parts.append("Unsupported atom labels detected: " + ", ".join(map(str, labels)) + ".")
    bad_nums = verifier.get("unsupported_numeric_claims") or []
    if bad_nums:
        parts.append("Unsupported numeric claims detected: " + ", ".join(map(str, bad_nums)) + ".")
    raw = verifier.get("raw_detail_violations") or []
    if raw:
        parts.append("Raw-record or structured identifier patterns detected; remove examples, quotations, identifiers, and names.")
    bad_comps = verifier.get("unsupported_comparisons") or []
    if bad_comps:
        parts.append("Unsupported comparisons detected; use pairwise semantic-support comparisons only when supported by released noisy masses, and use superlative or rank language only for rank-one released atoms.")
    bad_importance = verifier.get("unsupported_importance_claims") or []
    if bad_importance:
        parts.append("Unsupported importance-ordering language detected; remove importance-ordering claims because the released plan contains no separate importance score.")
    if verifier.get("strict_markers") and verifier.get("violations"):
        parts.append("Every substantive theme mention must include its canonical [atom:<id>] marker.")
    if not parts:
        parts.append("The previous draft did not satisfy the public verifier. Use only admitted atoms, canonical markers, and supported semantic-support language.")
    return " ".join(parts)


def _decode_verify_with_regeneration(plan: dict, atoms: list[dict[str, Any]], cfg: dict) -> tuple[str, str, dict[str, Any], list[dict[str, Any]]]:
    """Decode, verify, and optionally fall back using only plan-public feedback."""
    backend = str(cfg.get("decoder_backend", "template"))
    max_attempts = int(cfg.get("verifier_max_attempts", 3 if backend == "openai" else 1))
    max_attempts = max(1, max_attempts)
    feedback = None
    attempts = []
    final_summary = ""
    final_prompt = ""
    final_verifier: dict[str, Any] = {}
    accepted_before_fallback = False

    verifier_kwargs = {
        "strict_markers": bool(cfg.get("verifier_strict_markers", True)),
        "comparison_margin": float(cfg.get("verifier_comparison_margin", 0.0)),
        "alpha_atom": float(cfg.get("verifier_alpha_atom", cfg.get("alpha_atom", 0.0))),
        "alpha_comp": float(cfg.get("verifier_alpha_comp", cfg.get("alpha_comp", 0.0))),
        "reject_raw_details": bool(cfg.get("verifier_reject_raw_details", True)),
        "reject_unsupported_numbers": bool(cfg.get("verifier_reject_unsupported_numbers", True)),
        "reject_unsupported_importance": bool(cfg.get("verifier_reject_unsupported_importance", True)),
    }

    def _attempt_record(kind: str, attempt: int, summary: str, prompt: str, verifier: dict[str, Any]) -> dict[str, Any]:
        return {
            "kind": kind,
            "attempt": attempt,
            "accept": bool(verifier.get("accept", False)),
            "UAR": float(verifier.get("UAR", 0.0)),
            "UCR": float(verifier.get("UCR", 0.0)),
            "alpha_atom": float(verifier.get("alpha_atom", 0.0)),
            "alpha_comp": float(verifier.get("alpha_comp", 0.0)),
            "atom_rate_ok": bool(verifier.get("atom_rate_ok", True)),
            "comparison_rate_ok": bool(verifier.get("comparison_rate_ok", True)),
            "unsupported_atom_ids": verifier.get("unsupported_atom_ids", []),
            "unsupported_atom_labels": verifier.get("unsupported_atom_labels", []),
            "comparisons_detected": verifier.get("comparisons_detected", []),
            "prompt": prompt,
            "summary": summary,
        }

    for attempt in range(1, max_attempts + 1):
        summary, prompt = decode_plan(
            plan,
            backend=backend,
            model=str(cfg.get("decoder_model", "gpt-4o-mini")),
            temperature=float(cfg.get("decoder_temperature", 0.0)),
            max_tokens=int(cfg.get("decoder_max_tokens", 700)),
            verifier_feedback=feedback,
            attempt=attempt,
        )
        verifier = verify_summary(summary, plan, atoms, **verifier_kwargs)
        attempts.append(_attempt_record("candidate", attempt, summary, prompt, verifier))
        final_summary, final_prompt, final_verifier = summary, prompt, verifier

        if verifier.get("accept", False):
            accepted_before_fallback = True
            break

        feedback = _verifier_feedback(verifier)
        if backend == "template":
            # The template decoder is deterministic; retrying cannot help.
            break

    fallback_used = False
    if not accepted_before_fallback:
        # Do not release the final rejected candidate. Instead, release a
        # deterministic plan-only fallback. This is still post-processing of the
        # same released plan and public atom metadata.
        fallback_summary = template_decode(plan, mode="audit")
        fallback_prompt = "deterministic template fallback computed from the released plan"
        fallback_verifier = verify_summary(fallback_summary, plan, atoms, **verifier_kwargs)
        attempts.append(
            _attempt_record(
                "fallback",
                len(attempts) + 1,
                fallback_summary,
                fallback_prompt,
                fallback_verifier,
            )
        )
        final_summary, final_prompt, final_verifier = fallback_summary, fallback_prompt, fallback_verifier
        fallback_used = True

    final_verifier["regeneration"] = {
        "enabled": (max_attempts > 1 and backend != "template"),
        "backend": backend,
        "max_attempts": max_attempts,
        "attempts_used": len([a for a in attempts if a.get("kind") == "candidate"]),
        "accepted_before_fallback": accepted_before_fallback,
        "fallback_used": fallback_used,
        "fallback_type": "deterministic_template" if fallback_used else None,
        "final_output_accepted": bool(final_verifier.get("accept", False)),
        "post_processing_only": True,
        "note": "regeneration and deterministic fallback use only the released plan and public verifier feedback",
    }
    return final_summary, final_prompt, final_verifier, attempts

def _seed_bundle(cfg: dict[str, Any]) -> dict[str, int]:
    """Resolve independent random seeds for data, DP, decoder, and judge stages."""
    base = int(cfg.get("seed", 0))
    bundle = {
        "base_seed": base,
        "data_seed": int(cfg.get("data_seed", base)),
        "mechanism_seed": int(cfg.get("mechanism_seed", base + 100003)),
        "decoder_seed": int(cfg.get("decoder_seed", base + 200003)),
        "judge_seed": int(cfg.get("judge_seed", base + 300007)),
    }
    cfg.update(bundle)
    cfg["seed_bundle"] = dict(bundle)
    return bundle


def _decode_verify_structured(plan: dict[str, Any], cfg: dict[str, Any]):
    """Decode typed claims, verify them, and render the exact released text."""
    backend = str(cfg.get("decoder_backend", "template"))
    max_attempts = max(1, int(cfg.get("verifier_max_attempts", 3 if backend == "openai" else 1)))
    margin = float(cfg.get("verifier_comparison_margin", 0.0))
    maximum_claims = int(cfg.get("structured_maximum_claims", 12))
    feedback = None
    attempts: list[dict[str, Any]] = []
    accepted_claims: list[dict[str, Any]] | None = None
    final_prompt = ""
    raw_output = ""
    final_verifier: dict[str, Any] = {}
    for attempt in range(1, max_attempts + 1):
        try:
            claims, prompt, raw = decode_structured_claims(
                plan,
                backend=backend,
                model=str(cfg.get("decoder_model", "gpt-4o-mini")),
                temperature=float(cfg.get("decoder_temperature", 0.0)),
                max_tokens=int(cfg.get("decoder_max_tokens", 700)),
                verifier_feedback=feedback,
                comparison_margin=margin,
            )
            verifier = verify_structured_claims(
                claims, plan, comparison_margin=margin, maximum_claims=maximum_claims
            )
        except (TypeError, ValueError) as exc:
            claims, prompt, raw = [], "structured claim decoder", str(exc)
            verifier = {
                "accept": False, "contract": "structured_claim_language_v1",
                "claims": [], "violations": [{"code": "decoder_parse_error", "message": str(exc)}],
            }
        attempts.append({
            "kind": "candidate", "attempt": attempt, "accept": bool(verifier.get("accept")),
            "claims": claims, "verifier": verifier, "prompt": prompt, "raw_output": raw,
        })
        final_prompt, raw_output, final_verifier = prompt, raw, verifier
        if verifier.get("accept"):
            accepted_claims = list(claims)
            break
        feedback = str(verifier.get("violations", []))
        if backend == "template":
            break
    fallback_used = False
    if accepted_claims is None:
        fallback_used = True
        accepted_claims = claims_from_plan_template(plan, comparison_margin=margin)
        final_verifier = verify_structured_claims(
            accepted_claims, plan, comparison_margin=margin, maximum_claims=maximum_claims
        )
        final_prompt = "deterministic structured-claim fallback computed from the released plan"
        raw_output = str(accepted_claims)
        attempts.append({
            "kind": "fallback", "attempt": len(attempts) + 1,
            "accept": bool(final_verifier.get("accept")), "claims": accepted_claims,
            "verifier": final_verifier, "prompt": final_prompt, "raw_output": raw_output,
        })
    summary = render_verified_claims(accepted_claims, plan)
    final_verifier["regeneration"] = {
        "enabled": max_attempts > 1 and backend != "template",
        "max_attempts": max_attempts,
        "attempts_used": len([a for a in attempts if a["kind"] == "candidate"]),
        "fallback_used": fallback_used,
        "post_processing_only": True,
        "exact_released_text_rendered_from_verified_claims": True,
    }
    return summary, final_prompt, final_verifier, attempts, accepted_claims, raw_output


def run(cfg: dict) -> Path:
    """Execute one end-to-end DP-SPIN run and write all artifacts.
    """
    seeds = _seed_bundle(cfg)
    seed = seeds["base_seed"]
    cfg["seed"] = seed
    set_seed(seed)
    data_seed = seeds["data_seed"]
    rng = np.random.default_rng(seeds["mechanism_seed"])
    protected_unit = _protected_unit(cfg)
    cfg["protected_unit"] = protected_unit

    # These are the protected target records for this run. They may influence
    # only the bounded assignments and the DP sketch, not atom labels or the
    # vectorizer fit unless an explicit demo-only flag is set.
    if cfg.get("stream_jsonl_selection", True):
        if protected_unit == "user":
            records, record_selection_info = _select_users_from_jsonl(cfg["records"], cfg, data_seed)
        else:
            records, record_selection_info = _select_records_from_jsonl(cfg["records"], cfg, data_seed)
    else:
        all_records = read_jsonl(cfg["records"])
        if protected_unit == "user":
            all_user_groups = _group_records_by_user(all_records, cfg)
            records, record_selection_info = _select_users(all_user_groups, cfg, data_seed)
        else:
            records, record_selection_info = _select_records(all_records, cfg, data_seed)
    # Atoms are treated as public/auxiliary inputs.  Their labels/descriptions
    # may be shown to the decoder and to readers of the final summary.  For
    # dictionary-size sweeps, run_grid may set cfg["atoms"] from an
    # --atoms-template such as data/public_atoms/cfpb_atoms_m{m}.json.  In that
    # protocol each m uses its separately built atom file rather than a prefix of
    # a larger atom file.
    atom_path = str(cfg["atoms"])
    atoms_raw = read_json(atom_path)
    requested_m = cfg.get("m")
    atoms = atoms_raw[: int(requested_m)] if requested_m else atoms_raw
    atom_file_info = {
        "atom_file": atom_path,
        "atom_file_size": len(atoms_raw),
        "requested_m": int(requested_m) if requested_m else None,
        "atom_count_used": len(atoms),
        "truncated_atom_file_to_requested_m": bool(requested_m and int(requested_m) < len(atoms_raw)),
    }
    for i, a in enumerate(atoms):
        a["atom_id"] = int(a.get("atom_id", i))

    cfg["m"] = len(atoms)
    cfg["atom_file_used"] = atom_path
    cfg["atom_file_size"] = len(atoms_raw)
    cfg["n_records_effective"] = len(records)
    if protected_unit == "user":
        cfg["n_users_effective"] = int(record_selection_info.get("n_users_selected", 0))
        cfg["n_protected_units_effective"] = cfg["n_users_effective"]
    else:
        cfg["n_protected_units_effective"] = len(records)
    cfg.setdefault("gaussian_calibration", "analytic")
    # The paper experiments use the marker-preserving audit-text verifier.
    # Structured typed-claim decoding remains available as an optional mode.
    cfg.setdefault("decoder_contract_mode", "legacy_text")
    cfg.setdefault("selection_budget_allocation", "uniform")
    cfg.setdefault("selection_bound_beta", 0.05)
    # Store explicit defaults for sparse private-admission value release so the
    # budget split is reproducible in saved configs and grouped reports.
    cfg["mechanism"] = normalize_mechanism_name(str(cfg.get("mechanism", "gaussian")))
    cfg["value_epsilon_fraction"] = float(cfg.get("value_epsilon_fraction", 0.5))
    cfg["value_mechanism"] = str(cfg.get("value_mechanism", "gaussian"))
    # Store both configured and mechanism-effective privacy parameters before
    # constructing the run directory. This makes run names mechanism-aware; in
    # particular, Laplace runs are named with delta_eff=0 even when the shared
    # config contains a Gaussian delta for approximate-DP mechanisms.
    cfg["epsilon_configured"] = float(cfg.get("epsilon", 1.0))
    cfg["delta_configured"] = float(cfg.get("delta", 0.0))
    cfg["epsilon_effective"] = float(cfg["epsilon_configured"])
    cfg["delta_effective"] = 0.0 if cfg["mechanism"] == "laplace" else float(cfg["delta_configured"])

    denominator_unit_count = int(record_selection_info.get("n_users_selected", len(records))) if protected_unit == "user" else len(records)
    public_denominator = _resolve_public_denominator(cfg, denominator_unit_count)
    cfg["public_denominator"] = public_denominator

    outroot = ensure_dir(cfg.get("output_dir", "outputs/runs"))
    outdir = ensure_dir(outroot / run_name(cfg))

    public_fit_texts = None
    embedding_fit_info: dict[str, Any] = {"embedding_fit_records": None}
    if cfg.get("embedding_fit_records"):
        public_fit_texts, embedding_fit_info = _embedding_fit_texts_from_jsonl(cfg, data_seed)
    elif cfg.get("allow_private_embedding_fit_for_demo", False):
        public_fit_texts = None
    # By default, assignment fitting uses atom texts only. This keeps the
    # assignment map fixed relative to the protected target corpus.
    C, V, assignment_diag = topK_sparse_assignments(
        records,
        atoms,
        K=int(cfg.get("K", 3)),
        gamma=float(cfg.get("gamma", 10.0)),
        public_fit_texts=public_fit_texts,
        allow_private_fit=bool(cfg.get("allow_private_embedding_fit_for_demo", False)),
        similarity_threshold=(
            None if cfg.get("assignment_similarity_threshold") is None
            else float(cfg.get("assignment_similarity_threshold"))
        ),
        abstention_mode=str(cfg.get("assignment_abstention_mode", "zero")),
        other_atom_id=cfg.get("assignment_other_atom_id"),
        other_atom_label=cfg.get("assignment_other_atom_label"),
        mass_mode=str(cfg.get("assignment_mass_mode", "unit")),
        require_positive_similarity=bool(cfg.get("assignment_require_positive_similarity", True)),
        return_diagnostics=True,
        embedding_family=str(cfg.get("assignment_embedding_family", "tfidf_word")),
        embedding_min_df=int(cfg.get("assignment_embedding_min_df", 1)),
        embedding_max_features=(
            None if cfg.get("assignment_embedding_max_features") is None
            else int(cfg.get("assignment_embedding_max_features", 50000))
        ),
        embedding_word_ngram_range=tuple(cfg.get("assignment_embedding_word_ngram_range", [1, 2])),
        embedding_char_ngram_range=tuple(cfg.get("assignment_embedding_char_ngram_range", [3, 5])),
        embedding_model_name=str(cfg.get(
            "assignment_embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )),
        embedding_model_revision=cfg.get("assignment_embedding_model_revision"),
        embedding_model_batch_size=int(cfg.get("assignment_embedding_model_batch_size", 64)),
        embedding_model_device=cfg.get("assignment_embedding_model_device"),
        embedding_model_local_files_only=bool(
            cfg.get("assignment_embedding_model_local_files_only", False)
        ),
        embedding_model_trust_remote_code=bool(
            cfg.get("assignment_embedding_model_trust_remote_code", False)
        ),
        show_progress=bool(cfg.get("assignment_show_progress", True)),
    )
    assignment_info: dict[str, Any] = {
        "protected_unit": "record",
        "n_records": len(records),
        "record_contribution_bound": 1.0,
        "assignment_diagnostics": assignment_diag,
    }
    contribution_bound = 1.0
    C_unclipped_user = None
    if protected_unit == "user":
        C, U, clip_info, C_unclipped_user = _clip_user_assignments(V, records, cfg)
        clip_info["assignment_diagnostics"] = assignment_diag
        assignment_info = clip_info
        contribution_bound = float(assignment_info["user_contribution_bound"])

    L = int(cfg.get("L", 10))
    C_tilde, mech_info = release(
        C,
        mechanism=str(cfg.get("mechanism", "gaussian")),
        epsilon=float(cfg.get("epsilon", 1.0)),
        delta=float(cfg.get("delta", 1e-6)),
        adjacency=str(cfg.get("adjacency", "adddrop")),
        rng=rng,
        L=L,
        value_epsilon_fraction=float(cfg["value_epsilon_fraction"]),
        value_mechanism=str(cfg["value_mechanism"]),
        gaussian_calibration=str(cfg.get("gaussian_calibration", "analytic")),
        contribution_bound=contribution_bound,
        selection_budget_allocation=str(cfg.get("selection_budget_allocation", "uniform")),
        selection_bound_beta=float(cfg.get("selection_bound_beta", 0.05)),
    )
    # Confirm the effective privacy parameters returned by the mechanism.
    # These values should agree with the pre-release values used in the run
    # directory name.
    cfg["epsilon_effective"] = float(mech_info.get("epsilon", cfg["epsilon_effective"]))
    cfg["delta_effective"] = float(mech_info.get("delta", cfg["delta_effective"]))

    bins = cfg.get("support_bins")
    # The plan is the only private-data-dependent object that the decoder sees.
    # It combines DP noisy masses with public atom labels and coarse bins.
    plan = make_private_plan(C_tilde, atoms, L=L, public_denominator=public_denominator, bins=bins)
    plan["mechanism_info"] = mech_info
    plan["privacy"] = {
        "epsilon": cfg.get("epsilon_effective"),
        "delta": cfg.get("delta_effective"),
        "epsilon_configured": cfg.get("epsilon_configured"),
        "delta_configured": cfg.get("delta_configured"),
        "adjacency": cfg.get("adjacency", "adddrop"),
        "protected_unit": protected_unit,
        "contribution_bound": contribution_bound,
    }
    plan["atom_dictionary"] = _atom_source_summary(atoms)
    plan["privacy_boundary"] = {
        "decoder_receives": "private_plan plus public atom labels/descriptions",
        "decoder_must_not_receive": ["raw records", "raw excerpts", "private target-corpus atom labels"],
        "public_denominator": public_denominator,
        "assignment_vectorizer_fit": (
            "public_or_auxiliary_records" if cfg.get("embedding_fit_records") else
            "private_records_demo_only" if cfg.get("allow_private_embedding_fit_for_demo", False) else
            "atom_texts_only"
        ),
        "assignment_vectorizer_fit_info": embedding_fit_info,
        "protected_unit_selection": record_selection_info,
        "atom_file_info": atom_file_info,
        "assignment_and_clipping": assignment_info,
        "experimental_protected_unit": protected_unit,
        "user_level_dp_status": "implemented_with_per_user_aggregation_and_l1_clipping" if protected_unit == "user" else "not_used_in_this_run",
        "verifier": (
            "structured_claim_language_v2_support_exact_rendering"
            if str(cfg.get("decoder_contract_mode", "legacy_text")) == "structured"
            else "legacy_text_verifier_restricted_scope"
        ),
    }

    contract_mode = str(cfg.get("decoder_contract_mode", "legacy_text"))
    structured_claims = None
    structured_raw_output = None
    if contract_mode == "structured":
        summary, prompt, verifier, verifier_attempts, structured_claims, structured_raw_output = (
            _decode_verify_structured(plan, cfg)
        )
    elif contract_mode == "legacy_text":
        summary, prompt, verifier, verifier_attempts = _decode_verify_with_regeneration(plan, atoms, cfg)
    else:
        raise ValueError("decoder_contract_mode must be 'structured' or 'legacy_text'")
    # Offline utility reference: non-private top-L aggregate plan and
    # non-private-reference summary. These artifacts are for evaluation only.
    # They are not DP releases and are never used by the decoder that produces
    # the released summary.
    clean_reference_plan = build_clean_reference_plan(
        C,
        atoms,
        L=L,
        public_denominator=public_denominator,
        bins=bins,
    )
    clean_reference_summary, clean_reference_prompt = decode_plan(
        clean_reference_plan,
        backend=str(cfg.get("clean_reference_decoder_backend", "template")),
        model=str(cfg.get("clean_reference_decoder_model", cfg.get("decoder_model", "gpt-4o-mini"))),
        temperature=float(cfg.get("clean_reference_decoder_temperature", 0.0)),
        max_tokens=int(cfg.get("clean_reference_decoder_max_tokens", cfg.get("decoder_max_tokens", 700))),
    )
    model_role_validation = validate_distinct_decoder_judge(
        decoder_backend=str(cfg.get("decoder_backend", "template")),
        decoder_model=str(cfg.get("decoder_model", "")),
        judge_backend=str(cfg.get("summary_comparative_eval_backend", "none")),
        judge_model=str(cfg.get("summary_comparative_eval_model", "")),
        require_distinct=bool(cfg.get("require_distinct_decoder_judge", True)),
    )
    text_eval = compute_summary_text_evaluation(
        summary,
        clean_reference_summary,
        clean_reference_plan,
        embedding_backend=str(cfg.get("summary_embedding_backend", "tfidf")),
        embedding_model=str(cfg.get("summary_embedding_model", "sentence-transformers/all-mpnet-base-v2")),
        comparative_backend=str(cfg.get("summary_comparative_eval_backend", "none")),
        comparative_model=str(cfg.get("summary_comparative_eval_model", "gpt-4o-mini")),
    )
    text_eval["model_role_validation"] = model_role_validation

    metrics = {
        "sketch": sketch_metrics(C, C_tilde, L),
        "verifier": verifier,
        "text_eval": text_eval,
        "protected_unit_selection": record_selection_info,
        "atom_file_info": atom_file_info,
        "assignment_and_clipping": assignment_info,
        "assignment_vectorizer_fit_info": embedding_fit_info,
    }
    probe_strings = list(cfg.get("probe_strings") or [])
    probe_atom_ids = list(cfg.get("probe_atom_ids") or ([] if cfg.get("probe_atom_id") is None else [cfg.get("probe_atom_id")]))
    if probe_strings or probe_atom_ids:
        if probe_strings:
            metrics["probe_summary"] = probe_summary_mention(summary, probe_strings)
            metrics["probe"] = metrics["probe_summary"]  # backward-compatible alias
        metrics["probe_plan"] = probe_plan_admission(
            plan,
            probe_strings=probe_strings,
            probe_atom_ids=[int(x) for x in probe_atom_ids],
        )

    release_dir = ensure_dir(outdir / "release_bundle")
    offline_dir = ensure_dir(outdir / "offline_evaluation")

    # The release bundle contains only post-processed DP artifacts and public
    # configuration.  Non-private references and per-unit assignments remain in
    # the explicitly separated offline-evaluation directory.
    write_json(plan, release_dir / "private_plan.json")
    (release_dir / "summary.txt").write_text(summary, encoding="utf-8")
    write_json(verifier, release_dir / "verifier_output.json")
    if structured_claims is not None:
        write_json(structured_claims, release_dir / "structured_claims.json")
    public_release_config = {
        key: cfg.get(key) for key in [
            "epsilon_effective", "delta_effective", "adjacency", "protected_unit",
            "L", "public_denominator", "decoder_contract_mode", "verifier_comparison_margin",
        ]
    }
    write_json(public_release_config, release_dir / "public_release_config.json")
    manifest_name = "privacy_boundary_manifest.json"
    allowed_release_files = sorted(
        {p.name for p in release_dir.iterdir()} | {manifest_name}
    )
    write_json({
        "schema_version": 1,
        "contains_private_raw_records": False,
        "contains_nonprivate_sketch": False,
        "contains_per_unit_assignments": False,
        "allowed_files": allowed_release_files,
        "statement": "Every data-dependent file in this directory is DP or post-processing of the released DP plan.",
    }, release_dir / manifest_name)

    np.save(offline_dir / "clean_assignment_sketch.npy", C)
    np.save(offline_dir / "noisy_released_sketch.npy", C_tilde)
    sparse.save_npz(offline_dir / "record_assignments.npz", V)
    write_jsonl(_assignment_prediction_rows(records, V, atoms), offline_dir / "assignment_predictions.jsonl")
    if protected_unit == "user":
        sparse.save_npz(offline_dir / "user_clipped_assignments.npz", U)
        if C_unclipped_user is not None:
            np.save(offline_dir / "clean_unclipped_user_sketch.npy", C_unclipped_user)
    write_json(atoms, offline_dir / "atoms_used.json")
    write_json(clean_reference_plan, offline_dir / "clean_reference_plan.json")
    (offline_dir / "clean_reference_summary.txt").write_text(clean_reference_summary, encoding="utf-8")
    (offline_dir / "clean_reference_decoder_prompt.txt").write_text(clean_reference_prompt, encoding="utf-8")
    write_json(text_eval, offline_dir / "summary_text_eval.json")
    write_json(verifier_attempts, offline_dir / "verifier_attempts.json")
    if structured_raw_output is not None:
        (offline_dir / "structured_decoder_raw.txt").write_text(str(structured_raw_output), encoding="utf-8")
    write_json(metrics, offline_dir / "metrics.json")
    write_json(record_selection_info, offline_dir / "protected_unit_selection_manifest.json")
    write_json(assignment_info, offline_dir / "assignment_and_clipping_diagnostics.json")

    # Top-level compatibility files preserve the historical run layout used by
    # reporting and reconstruction scripts. Deployment packaging must use only
    # ``release_bundle/``; the run root contains non-private evaluation files.
    np.save(outdir / "clean_sketch.npy", C)
    np.save(outdir / "noisy_sketch.npy", C_tilde)
    if protected_unit == "user":
        sparse.save_npz(outdir / "user_clipped_assignments.npz", U)
    write_json(atoms, outdir / "atoms_used.json")
    write_json(plan, outdir / "private_plan.json")
    write_json(clean_reference_plan, outdir / "clean_reference_plan.json")
    (outdir / "clean_reference_summary.txt").write_text(clean_reference_summary, encoding="utf-8")
    (outdir / "clean_reference_decoder_prompt.txt").write_text(clean_reference_prompt, encoding="utf-8")
    write_json(text_eval, outdir / "summary_text_eval.json")
    write_json(verifier, outdir / "verifier_output.json")
    write_json(verifier_attempts, outdir / "verifier_attempts.json")
    write_json(metrics, outdir / "metrics.json")
    write_json(record_selection_info, outdir / "protected_unit_selection_manifest.json")
    write_json(assignment_info, outdir / "assignment_and_clipping_diagnostics.json")
    write_json(cfg, outdir / "config.json")
    save_yaml(cfg, outdir / "config.yaml")
    if cfg.get("probe_strings"):
        write_json(list(cfg["probe_strings"]), outdir / "probes.json")
    (outdir / "summary.txt").write_text(summary, encoding="utf-8")
    (outdir / "decoder_prompt.txt").write_text(prompt, encoding="utf-8")
    print(f"Run complete: {outdir}")
    print(metrics["sketch"])
    return outdir


def main() -> None:
    """Command-line entry point for a single DP-SPIN run.
    """
    parser = argparse.ArgumentParser(description="Run one DP-SPIN experiment end to end.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--mechanism", choices=["gaussian", "laplace", "private_topL"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-records", type=int, default=None)
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--L", type=int, default=None)
    parser.add_argument("--decoder-backend", choices=["template", "openai", "hf_local"], default=None)
    parser.add_argument("--gaussian-calibration", choices=["analytic", "classical"], default=None)
    parser.add_argument("--public-denominator", type=int, default=None)
    parser.add_argument("--record-selection", choices=["hash", "random", "first"], default=None)
    parser.add_argument("--protected-unit", choices=["record", "user"], default=None)
    parser.add_argument("--user-id-field", default=None)
    parser.add_argument("--n-users", type=int, default=None)
    parser.add_argument("--user-selection", choices=["hash", "random", "first"], default=None)
    parser.add_argument("--user-contribution-bound", type=float, default=None)
    parser.add_argument("--verifier-max-attempts", type=int, default=None)
    parser.add_argument("--verifier-alpha-atom", type=float, default=None)
    parser.add_argument("--verifier-alpha-comp", type=float, default=None)
    parser.add_argument("--summary-embedding-backend", choices=["none", "tfidf", "auto", "sentence_transformer"], default=None)
    parser.add_argument("--summary-comparative-eval-backend", choices=["none", "heuristic", "openai", "hf_local"], default=None)
    parser.add_argument("--embedding-fit-max-records", type=int, default=None)
    parser.add_argument("--embedding-fit-selection", choices=["hash", "random", "first"], default=None)
    parser.add_argument("--embedding-fit-seed", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--mechanism-seed", type=int, default=None)
    parser.add_argument("--decoder-seed", type=int, default=None)
    parser.add_argument("--judge-seed", type=int, default=None)
    parser.add_argument("--assignment-similarity-threshold", type=float, default=None)
    parser.add_argument("--assignment-abstention-mode", choices=["zero", "other_atom", "disabled"], default=None)
    parser.add_argument("--assignment-mass-mode", choices=["unit", "confidence"], default=None)
    parser.add_argument("--assignment-embedding-family", choices=list(SUPPORTED_EMBEDDING_FAMILIES), default=None)
    parser.add_argument("--assignment-embedding-min-df", type=int, default=None)
    parser.add_argument("--assignment-embedding-max-features", type=int, default=None)
    parser.add_argument("--assignment-embedding-model-name", default=None)
    parser.add_argument("--assignment-embedding-model-revision", default=None)
    parser.add_argument("--assignment-embedding-model-batch-size", type=int, default=None)
    parser.add_argument("--assignment-embedding-model-device", default=None)
    parser.add_argument("--assignment-embedding-model-local-files-only", action="store_true", default=None)
    parser.add_argument("--assignment-embedding-model-trust-remote-code", action="store_true", default=None)
    parser.add_argument("--decoder-contract-mode", choices=["structured", "legacy_text"], default=None)
    parser.add_argument("--verifier-comparison-margin", type=float, default=None)
    parser.add_argument("--selection-budget-allocation", choices=["uniform", "front_loaded", "theory_optimal"], default=None)
    parser.add_argument("--selection-bound-beta", type=float, default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    for k in ["epsilon", "delta", "mechanism", "seed", "m", "K", "L", "decoder_backend", "gaussian_calibration"]:
        val = getattr(args, k, None)
        if val is not None:
            cfg[k] = val
    if "mechanism" in cfg:
        cfg["mechanism"] = normalize_mechanism_name(str(cfg["mechanism"]))
    if args.n_records is not None:
        old_n = cfg.get("n_records")
        old_den = cfg.get("public_denominator")
        cfg["n_records"] = args.n_records
        if old_den is None or old_den == old_n:
            cfg["public_denominator"] = args.n_records
    if args.public_denominator is not None:
        cfg["public_denominator"] = args.public_denominator
    if args.record_selection is not None:
        cfg["record_selection"] = args.record_selection
    if args.protected_unit is not None:
        cfg["protected_unit"] = args.protected_unit
        if args.protected_unit == "user" and args.n_users is not None:
            cfg.pop("n_records", None)
    if args.user_id_field is not None:
        cfg["user_id_field"] = args.user_id_field
    if args.n_users is not None:
        cfg["n_users"] = args.n_users
        if cfg.get("protected_unit") == "user":
            cfg.pop("n_records", None)
            if cfg.get("auto_public_denominator_from_n_users", True):
                cfg["public_denominator"] = args.n_users
    if args.user_selection is not None:
        cfg["user_selection"] = args.user_selection
    if args.user_contribution_bound is not None:
        cfg["user_contribution_bound"] = args.user_contribution_bound
    if args.verifier_max_attempts is not None:
        cfg["verifier_max_attempts"] = args.verifier_max_attempts
    if args.verifier_alpha_atom is not None:
        cfg["verifier_alpha_atom"] = args.verifier_alpha_atom
    if args.verifier_alpha_comp is not None:
        cfg["verifier_alpha_comp"] = args.verifier_alpha_comp
    if args.summary_embedding_backend is not None:
        cfg["summary_embedding_backend"] = args.summary_embedding_backend
    if args.summary_comparative_eval_backend is not None:
        cfg["summary_comparative_eval_backend"] = args.summary_comparative_eval_backend
    if args.embedding_fit_max_records is not None:
        cfg["embedding_fit_max_records"] = args.embedding_fit_max_records
    if args.embedding_fit_selection is not None:
        cfg["embedding_fit_selection"] = args.embedding_fit_selection
    if args.embedding_fit_seed is not None:
        cfg["embedding_fit_seed"] = args.embedding_fit_seed
    for field in [
        "data_seed", "mechanism_seed", "decoder_seed", "judge_seed",
        "assignment_similarity_threshold", "assignment_abstention_mode",
        "assignment_mass_mode", "assignment_embedding_family", "assignment_embedding_min_df",
        "assignment_embedding_max_features", "assignment_embedding_model_name",
        "assignment_embedding_model_revision", "assignment_embedding_model_batch_size",
        "assignment_embedding_model_device", "assignment_embedding_model_local_files_only",
        "assignment_embedding_model_trust_remote_code", "decoder_contract_mode",
        "verifier_comparison_margin", "selection_budget_allocation",
        "selection_bound_beta",
    ]:
        value = getattr(args, field, None)
        if value is not None:
            cfg[field] = value
    run(cfg)

if __name__ == "__main__":
    main()
