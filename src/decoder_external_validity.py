"""Evaluate multiple decoder models with an automated judge in a distinct role.

The input semantic plans are already differentially private.  This module is
therefore post-processing and may be rerun without additional privacy cost.
Automated judging is a secondary diagnostic; matched common-ground human
annotations remain the primary language-usefulness evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .claims import decode_structured_claims, render_verified_claims, verify_structured_claims
from .model_roles import validate_decoder_panel
from .summary_eval import llm_comparative_eval
from .utils import load_yaml, read_json, read_jsonl, write_json, write_jsonl


def run_decoder_panel(config: dict[str, Any]) -> dict[str, Any]:
    manifest = read_jsonl(config["manifest"])
    decoders = list(config.get("decoders", []))
    judge = dict(config.get("judge", {}))
    role_validation = validate_decoder_panel(
        decoders,
        judge,
        require_at_least_two_decoders=bool(config.get("require_at_least_two_decoders", True)),
        require_nonjudge_decoder=bool(config.get("require_nonjudge_decoder", True)),
    )
    comparison_margin = float(config.get("comparison_margin", 0.0))
    rows: list[dict[str, Any]] = []
    for item in manifest:
        plan = read_json(item["plan_path"])
        reference_plan = read_json(item["reference_plan_path"])
        reference_summary = Path(item["reference_summary_path"]).read_text(encoding="utf-8")
        for decoder in decoders:
            claims, prompt, raw = decode_structured_claims(
                plan,
                backend=str(decoder.get("backend", "template")),
                model=str(decoder.get("model", "")),
                temperature=float(decoder.get("temperature", 0.0)),
                max_tokens=int(decoder.get("max_tokens", 700)),
                model_revision=decoder.get("revision"),
                device_map=str(decoder.get("device_map", "auto")),
                local_files_only=bool(decoder.get("local_files_only", False)),
                trust_remote_code=bool(decoder.get("trust_remote_code", False)),
                comparison_margin=comparison_margin,
            )
            verifier = verify_structured_claims(claims, plan, comparison_margin=comparison_margin)
            summary = render_verified_claims(claims, plan) if verifier["accept"] else ""
            judge_result = llm_comparative_eval(
                reference_plan,
                summary,
                reference_summary,
                backend=str(judge.get("backend", "none")),
                model=str(judge.get("model", "")),
                temperature=float(judge.get("temperature", 0.0)),
                max_tokens=int(judge.get("max_tokens", 500)),
            )
            rows.append(
                {
                    "dataset": item.get("dataset"),
                    "target_sample_id": item.get("target_sample_id"),
                    "plan_path": item["plan_path"],
                    "decoder_name": str(decoder.get("name", decoder.get("model", decoder.get("backend", "decoder")))),
                    "decoder_backend": str(decoder.get("backend", "template")),
                    "decoder_model": str(decoder.get("model", "")),
                    "decoder_revision": decoder.get("revision"),
                    "decoder_device_map": decoder.get("device_map"),
                    "decoder_local_files_only": bool(decoder.get("local_files_only", False)),
                    "decoder_trust_remote_code": bool(decoder.get("trust_remote_code", False)),
                    "summary": summary,
                    "structured_claims": claims,
                    "verifier_accept": bool(verifier["accept"]),
                    "verifier_violations": verifier["violations"],
                    "judge": judge_result,
                    "prompt": prompt if bool(config.get("retain_prompts", False)) else None,
                    "raw_decoder_output": raw if bool(config.get("retain_raw_outputs", False)) else None,
                }
            )
    return {
        "experiment": "decoder_model_external_validity",
        "manifest": config["manifest"],
        "role_validation": role_validation,
        "judge": judge,
        "decoders": decoders,
        "rows": rows,
        "scope_note": (
            "Verifier acceptance establishes released-plan support for the finite claim language. "
            "Automated judge scores do not establish corpus truth or general natural-language correctness."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_decoder_panel(load_yaml(args.config))
    write_json(result, args.output)
    print(json.dumps({"output": args.output, "n_rows": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
