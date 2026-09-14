"""Append controlled probe-atom runs to Amazon or Yelp experiments.

The runner creates probe-augmented target files and configurations without repeating the ordinary non-probe grid. 
It is used by the review experiment scripts to measure admission of a known public coordinate and exact probe-text mention in the generated summary.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from .dp_mechanisms import normalize_mechanism_name
from .run_experiment import run
from .run_grid import _expected_run_dir, _is_completed_run_dir


def _split_words(values: list[str] | None, default: list[str]) -> list[str]:
    if values is None:
        return default
    return values


def _slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "all"


def _read_jsonl_first(path: Path, n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    if len(rows) < n:
        raise RuntimeError(f"{path} has only {len(rows)} records, requested {n}.")
    return rows


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _get_user_id(rec: dict[str, Any], user_field: str = "user_id") -> str:
    if user_field in rec and rec[user_field] not in (None, ""):
        return str(rec[user_field])
    meta = rec.get("metadata")
    if isinstance(meta, dict) and meta.get(user_field) not in (None, ""):
        return str(meta[user_field])
    raise KeyError(f"Missing user id field {user_field!r} in record")


def _read_first_users(path: Path, n_users: int, user_field: str = "user_id") -> tuple[list[dict[str, Any]], int]:
    chosen: list[str] = []
    seen: set[str] = set()
    for rec in _iter_jsonl(path):
        uid = _get_user_id(rec, user_field)
        if uid not in seen:
            seen.add(uid)
            chosen.append(uid)
            if len(chosen) >= n_users:
                break
    if len(chosen) < n_users:
        raise RuntimeError(f"{path} has only {len(chosen)} users, requested {n_users}.")

    chosen_set = set(chosen)
    rows = [rec for rec in _iter_jsonl(path) if _get_user_id(rec, user_field) in chosen_set]
    return rows, len(chosen_set)


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _make_probe_atoms(base_atoms_path: Path, out_atoms_path: Path, probe_text: str) -> int:
    atoms = json.load(base_atoms_path.open("r", encoding="utf-8"))
    probe_upper = probe_text.upper()
    atoms = [
        a for a in atoms
        if "BLUE-LANTERN-791" not in json.dumps(a, ensure_ascii=False).upper()
        and "BLUE LANTERN" not in json.dumps(a, ensure_ascii=False).upper()
        and probe_upper not in json.dumps(a, ensure_ascii=False).upper()
    ]
    old_ids: list[int] = []
    for i, atom in enumerate(atoms):
        try:
            old_ids.append(int(atom.get("atom_id", i)))
        except Exception:
            old_ids.append(i)
    probe_atom_id = max(old_ids, default=-1) + 1
    atoms.append({
        "atom_id": probe_atom_id,
        "label": "blue lantern synthetic probe code",
        "description": "Reviews mentioning the synthetic probe phrase BLUE-LANTERN-791.",
        "domain": "controlled_probe_atom",
    })
    out_atoms_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(atoms, out_atoms_path.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return probe_atom_id


def _add_probe_records(
    base_rows: list[dict[str, Any]],
    *,
    support: int,
    seed: int,
    probe_text: str,
    dataset_tag: str,
    user_mode: bool,
) -> list[dict[str, Any]]:
    rows = list(base_rows)
    for i in range(support):
        rec: dict[str, Any] = {
            "id": f"{dataset_tag}_probe_{seed}_{i:04d}",
            "text": probe_text,
            "metadata": {
                "is_probe": True,
                "probe_support": int(support),
                "probe_text": probe_text,
            },
        }
        if user_mode:
            uid = f"{dataset_tag}_probe_user_{seed}_{i:04d}"
            rec["user_id"] = uid
            rec["metadata"]["user_id"] = uid
        rows.append(rec)
    random.Random(seed).shuffle(rows)
    return rows


def _base_cfg(
    *,
    experiment_name: str,
    unit: str,
    records: Path,
    atoms: Path,
    aux_records: Path,
    output_dir: Path,
    m: int,
    K: int,
    L: int,
    epsilon: float,
    delta: float,
    mechanism: str,
    seed: int,
    public_denominator: int,
    support: int,
    probe_base_records: int,
    probe_generated_records: int,
    probe_atom_id: int,
    probe_text: str,
    user_id_field: str = "user_id",
    n_records: Optional[int] = None,
    n_users: Optional[int] = None,
    user_B: Optional[float] = None,
    embedding_fit_max_records: Optional[int] = None,
    embedding_fit_selection: str = "hash",
    embedding_fit_seed: int = 1729,
    decoder_backend: str = "template",
    clean_reference_decoder_backend: str = "template",
    summary_embedding_backend: str = "tfidf",
    summary_comparative_eval_backend: str = "none",
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "experiment_name": experiment_name,
        "records": str(records),
        "atoms": str(atoms),
        "embedding_fit_records": str(aux_records),
        "output_dir": str(output_dir),
        "protected_unit": unit,
        "adjacency": "adddrop",
        "m": int(m),
        "K": int(K),
        "gamma": 10.0,
        "L": int(L),
        "epsilon": float(epsilon),
        "delta": float(delta),
        "mechanism": normalize_mechanism_name(mechanism),
        "gaussian_calibration": "analytic",
        "seed": int(seed),
        "decoder_backend": decoder_backend,
        "clean_reference_decoder_backend": clean_reference_decoder_backend,
        "summary_embedding_backend": summary_embedding_backend,
        "summary_comparative_eval_backend": summary_comparative_eval_backend,
        "strict_sample_size": True,
        "public_denominator": int(public_denominator),
        "probe_strings": [probe_text],
        "probe_atom_ids": [int(probe_atom_id)],
        "probe_support": int(support),
        "probe_base_records": int(probe_base_records),
        "probe_generated_records": int(probe_generated_records),
        "support_bins": {
            "very high": 0.20,
            "high": 0.10,
            "moderate": 0.05,
            "low": 0.0,
        },
    }
    if unit == "record":
        if n_records is None:
            raise ValueError("n_records is required for record-level probe runs.")
        cfg["n_records"] = int(n_records)
        cfg["record_selection"] = "hash"
        cfg["auto_public_denominator_from_n_records"] = False
    else:
        if n_users is None or user_B is None:
            raise ValueError("n_users and user_B are required for user-level probe runs.")
        cfg["n_users"] = int(n_users)
        cfg["user_id_field"] = user_id_field
        cfg["user_selection"] = "hash"
        cfg["user_contribution_bound"] = float(user_B)
        cfg["auto_public_denominator_from_n_users"] = False
    if embedding_fit_max_records is not None:
        cfg["embedding_fit_max_records"] = int(embedding_fit_max_records)
        cfg["embedding_fit_selection"] = embedding_fit_selection
        cfg["embedding_fit_seed"] = int(embedding_fit_seed)
    return cfg


def _write_cfg(cfg: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, path.open("w", encoding="utf-8"), sort_keys=False)


def _run_or_skip(cfg: dict[str, Any], *, skip_existing: bool) -> bool:
    if skip_existing:
        expected = _expected_run_dir(cfg)
        if _is_completed_run_dir(expected):
            print(f"Skipping completed probe run: {expected}")
            return False
    run(cfg)
    return True


def main() -> None:
    """Run the configured review controlled probe-atom experiments."""
    p = argparse.ArgumentParser(description="Run controlled probe-atom review experiments.")
    p.add_argument("--dataset-tag", required=True, help="Filesystem-safe dataset tag, e.g. amazon_All_Beauty or yelp_restaurants.")
    p.add_argument("--out", required=True)
    p.add_argument("--target-records", required=True)
    p.add_argument("--aux-records", required=True)
    p.add_argument("--base-atoms", required=True)
    p.add_argument("--probe-atoms", required=True)
    p.add_argument("--m", required=True, type=int, help="Base dictionary size before adding the probe atom.")
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--L", type=int, default=10)
    p.add_argument("--supports", nargs="+", type=int, default=[1, 2, 5, 10, 25, 50, 100, 200])
    p.add_argument("--epsilons", nargs="+", type=float, default=[0.5, 1, 2, 4])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--mechanisms", nargs="+", default=["gaussian", "laplace", "private_topL"])
    p.add_argument("--delta", type=float, default=1.0e-6)
    p.add_argument("--probe-text", default="The review contains the synthetic probe phrase BLUE-LANTERN-791.")
    p.add_argument("--run-record", action="store_true")
    p.add_argument("--run-user", action="store_true")
    p.add_argument("--record-base-records", type=int, default=2000)
    p.add_argument("--user-base-users", type=int, default=1000)
    p.add_argument("--user-B", type=float, default=2.0)
    p.add_argument("--user-id-field", default="user_id")
    p.add_argument("--embedding-fit-max-records", type=int, default=None)
    p.add_argument("--embedding-fit-selection", default="hash", choices=["hash", "random", "first"])
    p.add_argument("--embedding-fit-seed", type=int, default=1729)
    p.add_argument("--decoder-backend", default="template")
    p.add_argument("--clean-reference-decoder-backend", default="template")
    p.add_argument("--summary-embedding-backend", default="tfidf")
    p.add_argument("--summary-comparative-eval-backend", default="none")
    p.add_argument("--config-root", default=None)
    p.add_argument("--run-list", default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    if not args.run_record and not args.run_user:
        args.run_record = True
        args.run_user = True

    out = Path(args.out)
    target = Path(args.target_records)
    aux = Path(args.aux_records)
    base_atoms = Path(args.base_atoms)
    probe_atoms = Path(args.probe_atoms)
    for required in [target, aux, base_atoms]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")

    probe_atom_id = _make_probe_atoms(base_atoms, probe_atoms, args.probe_text)
    work = out / "probe_data"
    cfg_root = Path(args.config_root) if args.config_root else out / "probe_configs"
    work.mkdir(parents=True, exist_ok=True)
    cfg_root.mkdir(parents=True, exist_ok=True)
    config_paths: list[Path] = []
    completed = 0
    skipped = 0

    if args.run_record:
        base_rows = _read_jsonl_first(target, args.record_base_records)
        _write_jsonl(base_rows, work / f"{args.dataset_tag}_record_probe_base_n{args.record_base_records}.jsonl")
        for support in args.supports:
            for seed in args.seeds:
                rows = _add_probe_records(
                    base_rows,
                    support=support,
                    seed=seed,
                    probe_text=args.probe_text,
                    dataset_tag=args.dataset_tag,
                    user_mode=False,
                )
                rec_path = work / f"{args.dataset_tag}_record_probe_s{support}_seed{seed}.jsonl"
                _write_jsonl(rows, rec_path)
                n_total = len(rows)
                for mechanism in args.mechanisms:
                    for eps in args.epsilons:
                        cfg = _base_cfg(
                            experiment_name=f"{args.dataset_tag}_record_probe_support{support}",
                            unit="record",
                            records=rec_path,
                            atoms=probe_atoms,
                            aux_records=aux,
                            output_dir=out / "record" / "runs",
                            m=args.m + 1,
                            K=args.K,
                            L=args.L,
                            epsilon=eps,
                            delta=args.delta,
                            mechanism=mechanism,
                            seed=seed,
                            public_denominator=n_total,
                            support=support,
                            probe_base_records=len(base_rows),
                            probe_generated_records=n_total,
                            probe_atom_id=probe_atom_id,
                            probe_text=args.probe_text,
                            n_records=n_total,
                            embedding_fit_max_records=args.embedding_fit_max_records,
                            embedding_fit_selection=args.embedding_fit_selection,
                            embedding_fit_seed=args.embedding_fit_seed,
                            decoder_backend=args.decoder_backend,
                            clean_reference_decoder_backend=args.clean_reference_decoder_backend,
                            summary_embedding_backend=args.summary_embedding_backend,
                            summary_comparative_eval_backend=args.summary_comparative_eval_backend,
                        )
                        eps_tag = str(eps).replace(".", "_")
                        cfg_path = cfg_root / "record" / f"probe_s{support}_{normalize_mechanism_name(mechanism)}_eps{eps_tag}_seed{seed}.yaml"
                        _write_cfg(cfg, cfg_path)
                        config_paths.append(cfg_path)
                        if _run_or_skip(cfg, skip_existing=args.skip_existing):
                            completed += 1
                        else:
                            skipped += 1

    if args.run_user:
        base_rows, base_user_count = _read_first_users(target, args.user_base_users, args.user_id_field)
        _write_jsonl(base_rows, work / f"{args.dataset_tag}_user_probe_base_u{base_user_count}.jsonl")
        for support in args.supports:
            for seed in args.seeds:
                rows = _add_probe_records(
                    base_rows,
                    support=support,
                    seed=seed,
                    probe_text=args.probe_text,
                    dataset_tag=args.dataset_tag,
                    user_mode=True,
                )
                rec_path = work / f"{args.dataset_tag}_user_probe_s{support}_seed{seed}.jsonl"
                _write_jsonl(rows, rec_path)
                u_total = base_user_count + support
                for mechanism in args.mechanisms:
                    for eps in args.epsilons:
                        cfg = _base_cfg(
                            experiment_name=f"{args.dataset_tag}_user_probe_support{support}",
                            unit="user",
                            records=rec_path,
                            atoms=probe_atoms,
                            aux_records=aux,
                            output_dir=out / "user" / "runs",
                            m=args.m + 1,
                            K=args.K,
                            L=args.L,
                            epsilon=eps,
                            delta=args.delta,
                            mechanism=mechanism,
                            seed=seed,
                            public_denominator=u_total,
                            support=support,
                            probe_base_records=len(base_rows),
                            probe_generated_records=len(rows),
                            probe_atom_id=probe_atom_id,
                            probe_text=args.probe_text,
                            user_id_field=args.user_id_field,
                            n_users=u_total,
                            user_B=args.user_B,
                            embedding_fit_max_records=args.embedding_fit_max_records,
                            embedding_fit_selection=args.embedding_fit_selection,
                            embedding_fit_seed=args.embedding_fit_seed,
                            decoder_backend=args.decoder_backend,
                            clean_reference_decoder_backend=args.clean_reference_decoder_backend,
                            summary_embedding_backend=args.summary_embedding_backend,
                            summary_comparative_eval_backend=args.summary_comparative_eval_backend,
                        )
                        cfg["probe_base_users"] = int(base_user_count)
                        cfg["probe_generated_users"] = int(u_total)
                        eps_tag = str(eps).replace(".", "_")
                        cfg_path = cfg_root / "user" / f"probe_s{support}_{normalize_mechanism_name(mechanism)}_eps{eps_tag}_seed{seed}.yaml"
                        _write_cfg(cfg, cfg_path)
                        config_paths.append(cfg_path)
                        if _run_or_skip(cfg, skip_existing=args.skip_existing):
                            completed += 1
                        else:
                            skipped += 1

    if args.run_list:
        run_list = Path(args.run_list)
        run_list.parent.mkdir(parents=True, exist_ok=True)
        run_list.write_text("\n".join(str(p) for p in config_paths) + "\n", encoding="utf-8")
        print(f"Wrote {len(config_paths)} controlled probe configs to {run_list}")
    print(f"Controlled probe runs complete: completed {completed}; skipped {skipped}.")


if __name__ == "__main__":
    main()
