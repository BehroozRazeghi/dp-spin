"""Re-run the public text verifier on saved summaries without rerunning DP-SPIN.

This utility is intentionally post-hoc: it reads only saved semantic plans,
public atom metadata, saved summaries, and saved public verifier settings.  It
never reads protected records, recomputes assignments, resamples DP noise, or
changes the released semantic plan.

For ordinary DP-SPIN run directories it writes ``verifier_reverified.json``.
For OpenAI representative directories containing ``summary_openai_audit.txt``
it writes ``verifier_openai_reverified.json``.  Existing verifier files are
left unchanged for provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .utils import read_json, write_json
from .verify import verify_summary

VERIFIER_REVISION = "public_contract_verifier_v6_marker_local"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_config(run_dir: Path) -> dict[str, Any]:
    json_path = run_dir / "config.json"
    if json_path.exists():
        try:
            return read_json(json_path)
        except Exception:
            pass
    yaml_path = run_dir / "config.yaml"
    if yaml_path.exists():
        try:
            import yaml
            return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
    return {}


def _find_plan(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "private_plan.json",
        run_dir / "release_bundle" / "private_plan.json",
    ]
    return next((p for p in candidates if p.exists()), None)


def _find_atoms(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "atoms_used.json",
        run_dir / "offline_evaluation" / "atoms_used.json",
    ]
    return next((p for p in candidates if p.exists()), None)


def _verifier_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "strict_markers": bool(cfg.get("verifier_strict_markers", True)),
        "comparison_margin": float(cfg.get("verifier_comparison_margin", 0.0)),
        "alpha_atom": float(cfg.get("verifier_alpha_atom", cfg.get("alpha_atom", 0.0))),
        "alpha_comp": float(cfg.get("verifier_alpha_comp", cfg.get("alpha_comp", 0.0))),
        "reject_raw_details": bool(cfg.get("verifier_reject_raw_details", True)),
        "reject_unsupported_numbers": bool(cfg.get("verifier_reject_unsupported_numbers", True)),
        "reject_unsupported_importance": bool(cfg.get("verifier_reject_unsupported_importance", True)),
    }


def _reverify_one(run_dir: Path, summary_path: Path, output_path: Path, kind: str) -> dict[str, Any]:
    plan_path = _find_plan(run_dir)
    atoms_path = _find_atoms(run_dir)
    if plan_path is None or atoms_path is None:
        return {"status": "skipped", "reason": "missing private_plan.json or atoms_used.json", "run_dir": str(run_dir), "kind": kind}

    cfg = _load_config(run_dir)
    plan = read_json(plan_path)
    atoms = read_json(atoms_path)
    summary = summary_path.read_text(encoding="utf-8")
    kwargs = _verifier_kwargs(cfg)
    verifier = verify_summary(summary, plan, atoms, **kwargs)
    verifier["reverification"] = {
        "verifier_revision": VERIFIER_REVISION,
        "reverified_from_saved_artifacts": True,
        "summary_kind": kind,
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "private_plan_path": str(plan_path),
        "private_plan_sha256": _sha256(plan_path),
        "atoms_path": str(atoms_path),
        "atoms_sha256": _sha256(atoms_path),
        "decoder_contract_mode_recorded": cfg.get("decoder_contract_mode"),
        "strict_markers": kwargs["strict_markers"],
        "comparison_margin": kwargs["comparison_margin"],
        "alpha_atom": kwargs["alpha_atom"],
        "alpha_comp": kwargs["alpha_comp"],
    }
    write_json(verifier, output_path)
    return {
        "status": "written",
        "run_dir": str(run_dir),
        "kind": kind,
        "output": str(output_path),
        "accept": verifier.get("accept"),
        "UAR": verifier.get("UAR"),
        "UCR": verifier.get("UCR"),
        "comparison_count": verifier.get("comparison_count"),
    }


def reverify_tree(root: Path, include_standard: bool = True, include_openai: bool = True) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    if include_standard:
        for summary_path in sorted(root.rglob("summary.txt")):
            run_dir = summary_path.parent
            # Skip release-bundle copies; verify the canonical run root only.
            if run_dir.name == "release_bundle":
                continue
            key = (str(run_dir.resolve()), "standard")
            if key in seen:
                continue
            seen.add(key)
            results.append(_reverify_one(run_dir, summary_path, run_dir / "verifier_reverified.json", "standard_audit"))

    if include_openai:
        for summary_path in sorted(root.rglob("summary_openai_audit.txt")):
            run_dir = summary_path.parent
            key = (str(run_dir.resolve()), "openai")
            if key in seen:
                continue
            seen.add(key)
            results.append(_reverify_one(run_dir, summary_path, run_dir / "verifier_openai_reverified.json", "openai_audit"))

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Re-run the corrected public verifier on saved summaries only.")
    p.add_argument("--root", action="append", required=True, help="Output tree to scan recursively. Repeat for multiple roots.")
    p.add_argument("--mode", choices=["both", "standard", "openai"], default="both")
    p.add_argument("--report", default=None, help="Optional JSON report path.")
    args = p.parse_args()

    results: list[dict[str, Any]] = []
    for root_s in args.root:
        root = Path(root_s)
        if not root.exists():
            raise FileNotFoundError(root)
        results.extend(reverify_tree(root, include_standard=args.mode in {"both", "standard"}, include_openai=args.mode in {"both", "openai"}))

    summary = {
        "verifier_revision": VERIFIER_REVISION,
        "num_candidates": len(results),
        "num_written": sum(r.get("status") == "written" for r in results),
        "num_skipped": sum(r.get("status") == "skipped" for r in results),
        "results": results,
    }
    if args.report:
        write_json(summary, Path(args.report))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
