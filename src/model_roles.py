"""Validation of generator and automated-judge model roles.

Automated evaluation is less informative when the exact same model family is
used both to generate and to judge a summary.  This module records role IDs and
can enforce that every learned decoder differs from the automated judge.
"""
from __future__ import annotations

from typing import Any, Iterable

NON_MODEL_BACKENDS = {"none", "template", "heuristic", "local"}


def canonical_model_role(backend: str | None, model: str | None) -> str:
    backend_norm = str(backend or "none").strip().lower()
    model_norm = str(model or "").strip().lower()
    return f"{backend_norm}:{model_norm}" if backend_norm not in NON_MODEL_BACKENDS else backend_norm


def validate_distinct_decoder_judge(
    *,
    decoder_backend: str,
    decoder_model: str | None,
    judge_backend: str,
    judge_model: str | None,
    require_distinct: bool = True,
) -> dict[str, Any]:
    decoder_role = canonical_model_role(decoder_backend, decoder_model)
    judge_role = canonical_model_role(judge_backend, judge_model)
    decoder_is_model = str(decoder_backend).lower() not in NON_MODEL_BACKENDS
    judge_is_model = str(judge_backend).lower() not in NON_MODEL_BACKENDS
    same_model = decoder_is_model and judge_is_model and decoder_role == judge_role
    if require_distinct and same_model:
        raise ValueError(
            "decoder and automated judge use the same backend/model; choose a "
            "different judge or decoder model, or explicitly disable the distinct-role check"
        )
    return {
        "decoder_role": decoder_role,
        "judge_role": judge_role,
        "decoder_is_learned_model": decoder_is_model,
        "judge_is_learned_model": judge_is_model,
        "same_model": same_model,
        "require_distinct": bool(require_distinct),
        "valid": not same_model or not require_distinct,
    }


def validate_decoder_panel(
    decoders: Iterable[dict[str, Any]],
    judge: dict[str, Any],
    *,
    require_at_least_two_decoders: bool = True,
    require_nonjudge_decoder: bool = True,
) -> dict[str, Any]:
    decoders_list = list(decoders)
    if require_at_least_two_decoders and len(decoders_list) < 2:
        raise ValueError("decoder panel must contain at least two decoder conditions")
    judge_role = canonical_model_role(judge.get("backend"), judge.get("model"))
    roles = []
    for decoder in decoders_list:
        role = canonical_model_role(decoder.get("backend"), decoder.get("model"))
        roles.append(role)
    learned_nonjudge = [
        role
        for role, decoder in zip(roles, decoders_list)
        if str(decoder.get("backend", "template")).lower() not in NON_MODEL_BACKENDS and role != judge_role
    ]
    if require_nonjudge_decoder and not learned_nonjudge:
        raise ValueError("at least one learned decoder must differ from the automated judge")
    return {
        "decoder_roles": roles,
        "judge_role": judge_role,
        "n_decoders": len(decoders_list),
        "learned_nonjudge_decoder_roles": learned_nonjudge,
        "valid": True,
    }
