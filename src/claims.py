"""Structured claim language for plan-grounded aggregate summaries.

The structured contract removes the audit/display mismatch of the legacy text
verifier.  A decoder proposes only typed claims over released atom identifiers.
A deterministic verifier checks those claims against the released DP plan, and a
deterministic renderer produces the exact text that is released.

No protected records, non-private sketches, or non-admitted atom labels are used
by this module.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from .llm_client import maybe_llm

ClaimType = Literal["theme", "support", "comparison", "rank"]
ComparisonRelation = Literal["higher_support_than", "lower_support_than"]


@dataclass(frozen=True)
class ClaimViolation:
    index: int
    code: str
    message: str
    claim: dict[str, Any]


def _atom_id(atom: dict[str, Any]) -> int:
    return int(atom.get("atom_id", atom.get("index", -1)))


def _plan_atom_map(plan: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {_atom_id(atom): atom for atom in plan.get("selected_atoms", []) or []}


def _mass(atom: dict[str, Any]) -> float:
    return float(atom.get("noisy_mass", 0.0))


def _normalize_claim(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("each claim must be a JSON object")
    claim = {str(key): value for key, value in raw.items()}
    if "claim_type" not in claim and "type" in claim:
        claim["claim_type"] = claim.pop("type")
    claim["claim_type"] = str(claim.get("claim_type", "")).strip().lower()
    return claim


def verify_structured_claims(
    claims: Iterable[dict[str, Any]],
    plan: dict[str, Any],
    *,
    comparison_margin: float = 0.0,
    maximum_claims: int = 12,
) -> dict[str, Any]:
    """Verify typed claims against a released semantic plan.

    Soundness is relative to this finite claim language.  The verifier does not
    establish that the public atom dictionary is semantically complete or that
    the contribution map is correct relative to raw records.
    """
    if comparison_margin < 0:
        raise ValueError("comparison_margin must be nonnegative")
    atoms = _plan_atom_map(plan)
    normalized: list[dict[str, Any]] = []
    violations: list[ClaimViolation] = []

    for index, raw in enumerate(list(claims)):
        try:
            claim = _normalize_claim(raw)
        except (TypeError, ValueError) as exc:
            violations.append(ClaimViolation(index, "invalid_schema", str(exc), {"raw": raw}))
            continue
        normalized.append(claim)
        kind = claim.get("claim_type")
        if kind not in {"theme", "support", "comparison", "rank"}:
            violations.append(ClaimViolation(index, "unsupported_claim_type", f"unsupported claim_type={kind!r}", claim))
            continue

        if kind in {"theme", "support", "rank"}:
            try:
                atom_id = int(claim["atom_id"])
            except (KeyError, TypeError, ValueError):
                violations.append(ClaimViolation(index, "missing_atom_id", "claim requires an integer atom_id", claim))
                continue
            if atom_id not in atoms:
                violations.append(ClaimViolation(index, "unadmitted_atom", f"atom_id={atom_id} is not admitted", claim))
                continue
            atom = atoms[atom_id]

            if kind == "support":
                requested = str(claim.get("bin", claim.get("support_bin", ""))).strip().casefold()
                released = str(atom.get("support_bin", "")).strip().casefold()
                if not requested:
                    violations.append(ClaimViolation(index, "missing_support_bin", "support claim requires bin", claim))
                elif requested != released:
                    violations.append(
                        ClaimViolation(
                            index,
                            "unsupported_support_bin",
                            f"requested bin={requested!r}, released bin={released!r}",
                            claim,
                        )
                    )
            elif kind == "rank":
                try:
                    requested_rank = int(claim.get("rank"))
                except (TypeError, ValueError):
                    violations.append(ClaimViolation(index, "missing_rank", "rank claim requires an integer rank", claim))
                    continue
                released_rank = int(atom.get("rank", -1))
                if requested_rank != released_rank:
                    violations.append(
                        ClaimViolation(index, "unsupported_rank", f"requested rank={requested_rank}, released rank={released_rank}", claim)
                    )

        elif kind == "comparison":
            try:
                left_id = int(claim["left_atom_id"])
                right_id = int(claim["right_atom_id"])
            except (KeyError, TypeError, ValueError):
                violations.append(
                    ClaimViolation(index, "missing_comparison_atoms", "comparison requires integer left_atom_id and right_atom_id", claim)
                )
                continue
            relation = str(claim.get("relation", "")).strip().lower()
            if relation not in {"higher_support_than", "lower_support_than"}:
                violations.append(ClaimViolation(index, "unsupported_relation", f"unsupported relation={relation!r}", claim))
                continue
            if left_id not in atoms or right_id not in atoms:
                violations.append(
                    ClaimViolation(index, "unadmitted_comparison_atom", "both comparison atoms must be admitted", claim)
                )
                continue
            if left_id == right_id:
                violations.append(ClaimViolation(index, "self_comparison", "an atom cannot be compared with itself", claim))
                continue
            left_mass = _mass(atoms[left_id])
            right_mass = _mass(atoms[right_id])
            supported = (
                left_mass >= right_mass + comparison_margin
                if relation == "higher_support_than"
                else left_mass + comparison_margin <= right_mass
            )
            if not supported:
                violations.append(
                    ClaimViolation(
                        index,
                        "unsupported_comparison",
                        f"released masses do not support {relation} with margin={comparison_margin}",
                        claim,
                    )
                )

    if len(normalized) > maximum_claims:
        violations.append(
            ClaimViolation(
                maximum_claims,
                "too_many_claims",
                f"received {len(normalized)} claims; maximum is {maximum_claims}",
                {"claim_count": len(normalized)},
            )
        )

    violation_dicts = [
        {"index": item.index, "code": item.code, "message": item.message, "claim": item.claim}
        for item in violations
    ]
    return {
        "accept": not violation_dicts,
        "contract": "structured_claim_language_v2_support",
        "claims": normalized,
        "claim_count": len(normalized),
        "violations": violation_dicts,
        "false_accept_scope_note": (
            "Acceptance establishes support relative to the released plan for the finite structured claim language; "
            "it does not establish semantic correctness or completeness relative to protected records."
        ),
    }


def claims_from_plan_template(
    plan: dict[str, Any],
    *,
    maximum_themes: int = 5,
    include_support: bool = True,
    include_top_rank: bool = True,
    include_comparison: bool = True,
    comparison_margin: float = 0.0,
) -> list[dict[str, Any]]:
    """Construct a deterministic valid claim set from a released plan."""
    atoms = list(plan.get("selected_atoms", []) or [])[: max(0, int(maximum_themes))]
    claims: list[dict[str, Any]] = []
    for atom in atoms:
        atom_id = _atom_id(atom)
        claims.append({"claim_type": "theme", "atom_id": atom_id})
        if include_support and atom.get("support_bin") is not None:
            claims.append({"claim_type": "support", "atom_id": atom_id, "bin": str(atom["support_bin"])})
    if include_top_rank and atoms:
        top = min(atoms, key=lambda item: int(item.get("rank", 10**9)))
        claims.append({"claim_type": "rank", "atom_id": _atom_id(top), "rank": int(top.get("rank", 1))})
    if include_comparison and len(atoms) >= 2:
        ordered = sorted(atoms, key=_mass, reverse=True)
        if _mass(ordered[0]) >= _mass(ordered[1]) + comparison_margin:
            claims.append(
                {
                    "claim_type": "comparison",
                    "left_atom_id": _atom_id(ordered[0]),
                    "right_atom_id": _atom_id(ordered[1]),
                    "relation": "higher_support_than",
                }
            )
    return claims


def structured_claim_prompt(plan: dict[str, Any], verifier_feedback: str | None = None) -> str:
    """Build an LLM prompt that exposes only the released plan and public rules."""
    entries = []
    for atom in plan.get("selected_atoms", []) or []:
        entries.append(
            {
                "atom_id": _atom_id(atom),
                "label": str(atom.get("label", "")),
                "description": str(atom.get("description", "")),
                "rank": int(atom.get("rank", -1)),
                "support_bin": atom.get("support_bin"),
                "noisy_mass": float(atom.get("noisy_mass", 0.0)),
            }
        )
    rules = {
        "allowed_claim_types": {
            "theme": {"claim_type": "theme", "atom_id": "integer"},
            "support": {"claim_type": "support", "atom_id": "integer", "bin": "exact released bin"},
            "comparison": {
                "claim_type": "comparison",
                "left_atom_id": "integer",
                "right_atom_id": "integer",
                "relation": "higher_support_than or lower_support_than",
            },
            "rank": {"claim_type": "rank", "atom_id": "integer", "rank": "exact released rank"},
        },
        "prohibited": [
            "free-form factual claims",
            "examples, causes, entities, subgroups, dates, quotations, or exact record counts",
            "atoms not in the released plan",
        ],
        "output": "JSON array only; no markdown and no prose",
    }
    payload = {"released_plan": entries, "rules": rules}
    prompt = (
        "Propose a concise set of structured aggregate-summary claims. Use only the released plan and the finite schema. "
        "Return a JSON array only.\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    if verifier_feedback:
        prompt += "\n\nPrevious verifier feedback:\n" + verifier_feedback
    return prompt


def parse_claims_json(text: str) -> list[dict[str, Any]]:
    """Parse an LLM response containing a JSON claim array."""
    candidate = str(text).strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start < 0 or end <= start:
            raise ValueError("decoder output did not contain a JSON array")
        data = json.loads(candidate[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("decoder output must be a JSON array")
    return [_normalize_claim(item) for item in data]


def decode_structured_claims(
    plan: dict[str, Any],
    *,
    backend: str = "template",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 700,
    model_revision: str | None = None,
    device_map: str = "auto",
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    verifier_feedback: str | None = None,
    comparison_margin: float = 0.0,
) -> tuple[list[dict[str, Any]], str, str]:
    """Decode structured claims and return ``(claims, prompt, raw_output)``."""
    prompt = structured_claim_prompt(plan, verifier_feedback=verifier_feedback)
    if backend in {"template", "none", "local"}:
        claims = claims_from_plan_template(plan, comparison_margin=comparison_margin)
        return claims, prompt, json.dumps(claims, ensure_ascii=False)
    raw = maybe_llm(
        prompt,
        backend=backend,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        model_revision=model_revision,
        device_map=device_map,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    if raw is None:
        claims = claims_from_plan_template(plan, comparison_margin=comparison_margin)
        return claims, prompt, json.dumps(claims, ensure_ascii=False)
    return parse_claims_json(raw), prompt, raw


def render_verified_claims(claims: Iterable[dict[str, Any]], plan: dict[str, Any]) -> str:
    """Render the exact final summary from already verified structured claims."""
    atoms = _plan_atom_map(plan)
    normalized = [_normalize_claim(claim) for claim in claims]
    theme_ids: list[int] = []
    support_bins_by_atom: dict[int, str] = {}
    comparisons: list[tuple[int, int, str]] = []
    ranks: dict[int, int] = {}
    for claim in normalized:
        kind = claim["claim_type"]
        if kind == "theme":
            atom_id = int(claim["atom_id"])
            if atom_id not in theme_ids:
                theme_ids.append(atom_id)
        elif kind == "support":
            support_bins_by_atom[int(claim["atom_id"])] = str(claim.get("bin", claim.get("support_bin", "")))
        elif kind == "comparison":
            comparisons.append((int(claim["left_atom_id"]), int(claim["right_atom_id"]), str(claim["relation"])))
        elif kind == "rank":
            ranks[int(claim["atom_id"])] = int(claim["rank"])

    if not theme_ids:
        return "The released semantic plan does not support any aggregate theme statement."

    phrases = []
    for atom_id in theme_ids:
        atom = atoms[atom_id]
        label = str(atom.get("label", f"atom {atom_id}"))
        if atom_id in support_bins_by_atom and support_bins_by_atom[atom_id]:
            phrases.append(f"{label} ({support_bins_by_atom[atom_id]})")
        else:
            phrases.append(label)
    paragraph = "The released semantic plan identifies " + ", ".join(phrases) + " as aggregate themes."

    extra: list[str] = []
    for atom_id, rank in sorted(ranks.items(), key=lambda item: item[1]):
        if rank == 1:
            extra.append(f"{atoms[atom_id].get('label')} has the highest released rank.")
        else:
            extra.append(f"{atoms[atom_id].get('label')} has released rank {rank}.")
    for left_id, right_id, relation in comparisons:
        left = str(atoms[left_id].get("label"))
        right = str(atoms[right_id].get("label"))
        if relation == "higher_support_than":
            extra.append(f"The released values indicate that {left} has higher semantic support than {right}.")
        else:
            extra.append(f"The released values indicate that {left} has lower semantic support than {right}.")
    if extra:
        paragraph += " " + " ".join(extra)
    return paragraph
