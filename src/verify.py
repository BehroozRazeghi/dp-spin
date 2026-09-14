"""Check generated text against the public decoder contract.

The verifier receives the released semantic plan, public atom metadata, public support-bin definitions, and the generated summary. It does not read protected
records or the non-private semantic sketch. The checks establish syntactic and numeric consistency with the released plan; they do not establish truth or completeness relative to the protected corpus.

Canonical markers such as ``[atom:17]`` provide the strongest atom-reference check. Literal public-label matching is used when a summary omits markers.
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .utils import read_json, write_json

_ATOM_MARKER_RE = re.compile(r"\[(?:atom\s*:\s*|a\s*)(-?\d+)\]", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}(?!\d)")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_ACCOUNT_RE = re.compile(r"\b(?:account|acct|card|loan|case|complaint|claim|ticket)\s*(?:#|number|no\.?|id)\s*[:#-]?\s*[A-Za-z0-9-]*\d[A-Za-z0-9-]{3,}\b", re.IGNORECASE)
_QUOTE_RE = re.compile(r"[\"“”‘’][^\"“”‘’]{12,}[\"“”‘’]")
_PROPER_NAME_RE = re.compile(r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+\b")

_COMPARISON_PATTERNS = [
    ("higher_support_than", re.compile(r"(.{0,90}?)\b(?:higher(?: semantic)? support)\s+than\b(.{0,90})", re.IGNORECASE)),
    ("lower_support_than", re.compile(r"(.{0,90}?)\b(?:lower(?: semantic)? support)\s+than\b(.{0,90})", re.IGNORECASE)),
]
_SUPERLATIVE_WORDS = {
    "highest support", "highest semantic support", "highest released rank", "top-ranked theme"
}

# Explicit importance-ordering language is rejected unless a future release
# mechanism adds a separate differentially private importance score to the plan.
# In the current DP-SPIN plan, the supported quantitative notions are released
# noisy mass, support bin, and released rank.
_UNSUPPORTED_IMPORTANCE_RE = re.compile(
    r"\b(?:more|less)\s+important\s+than\b|"
    r"\b(?:higher|lower)\s+importance\s+than\b|"
    r"\b(?:most|least)\s+important\b",
    re.IGNORECASE,
)

_ALLOWED_GENERIC_NUMBERS = {0.0, 1.0, 3.0, 5.0, 10.0}


def normalize(s: str) -> str:
    """Lowercase, normalize punctuation spacing, and collapse whitespace."""
    s = str(s).lower()
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _contains_label(text_norm: str, label: str) -> bool:
    """Return whether a normalized label occurs as a phrase in normalized text."""
    label_norm = normalize(label)
    if not label_norm or len(label_norm) < 3:
        return False
    pattern = r"(?<!\w)" + re.escape(label_norm) + r"(?!\w)"
    return re.search(pattern, text_norm) is not None


def _atom_id(atom: dict[str, Any]) -> int:
    return int(atom.get("atom_id", atom.get("index", -1)))


def _atom_index(atom: dict[str, Any]) -> int:
    return int(atom.get("index", atom.get("atom_id", -1)))


def _label(atom: dict[str, Any]) -> str:
    return str(atom.get("label", f"atom_{_atom_id(atom)}"))


def _desc(atom: dict[str, Any]) -> str:
    return str(atom.get("description", ""))


def _plan_maps(plan: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], set[int]]:
    selected = plan.get("selected_atoms", []) or []
    by_id = {_atom_id(a): a for a in selected}
    by_index = {_atom_index(a): a for a in selected}
    admitted = set(by_id)
    return by_id, by_index, admitted


def _all_atom_label_map(all_atoms: list[dict[str, Any]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for a in all_atoms:
        aid = _atom_id(a)
        out[aid] = _label(a)
    return out


def extract_atom_markers(summary: str) -> set[int]:
    """Extract canonical atom references such as [atom:17] or [A17]."""
    return {int(m.group(1)) for m in _ATOM_MARKER_RE.finditer(summary)}


def extract_atoms_by_label(summary: str, all_atoms: list[dict[str, Any]]) -> set[int]:
    """Find atom IDs whose public labels appear literally in the summary."""
    text = normalize(summary)
    found: set[int] = set()
    # Longer labels first reduces ambiguity in diagnostics; result is a set.
    for a in sorted(all_atoms, key=lambda x: len(_label(x)), reverse=True):
        if _contains_label(text, _label(a)):
            found.add(_atom_id(a))
    return found


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _atoms_in_span(span: str, all_atoms: list[dict[str, Any]]) -> set[int]:
    ids = extract_atom_markers(span)
    ids |= extract_atoms_by_label(span, all_atoms)
    return ids


def _detect_raw_detail_violations(summary: str) -> list[dict[str, str]]:
    """Detect raw-record patterns and structured identifiers barred by the contract."""
    checks = [
        ("email", _EMAIL_RE),
        ("phone_number", _PHONE_RE),
        ("ssn", _SSN_RE),
        ("account_or_case_identifier", _ACCOUNT_RE),
        ("long_quotation", _QUOTE_RE),
        ("titled_person_name", _PROPER_NAME_RE),
    ]
    out: list[dict[str, str]] = []
    for kind, pat in checks:
        for m in pat.finditer(summary):
            out.append({"type": kind, "match": m.group(0)[:120]})
    return out


def _detect_unsupported_importance_claims(summary: str) -> list[dict[str, str]]:
    """Detect explicit importance-ordering claims unsupported by the current plan.

    The current DP-SPIN plan releases noisy masses, support bins, and ranks,
    but it does not release a separate importance score.  Therefore claims such
    as ``more important than`` or ``most important`` are rejected as unsupported
    unless a future mechanism explicitly releases differentially private
    importance scores.
    """
    out: list[dict[str, str]] = []
    for m in _UNSUPPORTED_IMPORTANCE_RE.finditer(summary):
        out.append({
            "type": "unsupported_importance_claim",
            "match": m.group(0),
            "reason": (
                "the released plan contains noisy masses, support bins, and ranks, "
                "but no separate released importance score"
            ),
        })
    return out


def _allowed_numeric_values(plan: dict[str, Any]) -> list[float]:
    vals: list[float] = []
    for a in plan.get("selected_atoms", []) or []:
        for key in ("noisy_mass", "raw_noisy_mass"):
            try:
                vals.append(float(a.get(key)))
            except Exception:
                pass
        try:
            p = a.get("semantic_support")
            if p is not None:
                vals.append(float(p))
                vals.append(100.0 * float(p))
        except Exception:
            pass
    vals.extend(_ALLOWED_GENERIC_NUMBERS)
    return [v for v in vals if math.isfinite(v)]


def _detect_unsupported_numbers(summary: str, plan: dict[str, Any], tolerance_abs: float = 0.75, tolerance_rel: float = 0.02) -> list[str]:
    """Flag numeric quantities not supported by the released plan or output format."""
    allowed = _allowed_numeric_values(plan)
    atom_marker_spans = [m.span() for m in _ATOM_MARKER_RE.finditer(summary)]
    unsupported: list[str] = []
    for m in _NUMBER_RE.finditer(summary):
        token = m.group(0)
        # Ignore numbers inside atom markers, e.g. [atom:17].
        if any(m.start() >= a and m.end() <= b for a, b in atom_marker_spans):
            continue
        is_percent = token.endswith("%")
        try:
            val = float(token.rstrip("%"))
        except ValueError:
            continue
        ok = False
        for a in allowed:
            # Percent tokens may match either percentage values or exact displayed percent.
            tol = max(tolerance_abs, tolerance_rel * max(1.0, abs(a)))
            if abs(val - a) <= tol:
                ok = True
                break
        if not ok:
            unsupported.append(token)
    return unsupported



def _plan_bin_labels(plan: dict[str, Any]) -> list[str]:
    """Return support-bin labels appearing in the released plan.

    The verifier treats these labels as public vocabulary.  Longer labels are
    returned first so that ``very high`` is matched before ``high``.
    """
    labels: set[str] = set()
    for atom in plan.get("selected_atoms", []) or []:
        b = atom.get("support_bin")
        if b is not None:
            labels.add(str(b).strip().lower())
    labels.update(str(k).strip().lower() for k in (plan.get("support_bins") or {}).keys())
    return sorted((b for b in labels if b), key=len, reverse=True)


def _bin_label_in_text(text_norm: str, bin_label: str) -> bool:
    """Return whether a support-bin label occurs as a standalone phrase."""
    if not bin_label:
        return False
    pattern = r"(?<!\w)" + re.escape(bin_label) + r"(?!\w)"
    return re.search(pattern, text_norm) is not None


def _bin_labels_in_text_nonoverlap(text_norm: str, bin_labels: list[str]) -> list[str]:
    """Return support-bin labels with longest non-overlapping matching.

    Bin labels can be nested, for example ``high`` occurs inside
    ``very high``. For verifier correctness, a sentence containing
    ``very high`` should not also be interpreted as assigning the bin
    ``high`` to the same atom.  Labels must therefore be supplied in
    descending length order, as returned by :func:`_plan_bin_labels`.
    """
    matches: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for b in bin_labels:
        if not b:
            continue
        pattern = r"(?<!\w)" + re.escape(b) + r"(?!\w)"
        for m in re.finditer(pattern, text_norm):
            span = m.span()
            if any(not (span[1] <= a or span[0] >= z) for a, z in occupied):
                continue
            matches.append((span[0], span[1], b))
            occupied.append(span)
    matches.sort(key=lambda x: x[0])
    return [b for _, _, b in matches]


def _allowed_bins_for_atom(atom: dict[str, Any], plan: dict[str, Any]) -> set[str]:
    """Return support-bin labels allowed for an admitted atom.

    By default, only the released bin for that atom is allowed.  A deployment may
    optionally include a public mapping ``allowed_support_bin_coarsenings`` in
    the plan to authorize coarser labels.  This mapping is part of the released
    plan or public decoding policy and does not access protected data.
    """
    released = str(atom.get("support_bin", "")).strip().lower()
    allowed = {released} if released else set()
    mapping = plan.get("allowed_support_bin_coarsenings") or {}
    for b in mapping.get(released, []) if isinstance(mapping, dict) else []:
        if str(b).strip():
            allowed.add(str(b).strip().lower())
    return allowed


def _detect_support_bin_claims(
    summary: str,
    plan: dict[str, Any],
    all_atoms: list[dict[str, Any]],
    *,
    strict_markers: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Detect support-bin assignments and validate them against the plan.

    The check is syntactic. It considers a sentence to assign a
    support bin when the sentence mentions an admitted atom and contains a
    released bin label. Pairwise semantic-support comparisons are handled by the
    comparison checker.
    """
    by_id, _, admitted = _plan_maps(plan)
    bin_labels = _plan_bin_labels(plan)
    if not bin_labels:
        return [], []

    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    comparison_re = re.compile(r"\b(?:higher|lower)(?: semantic)? support\s+than\b", re.IGNORECASE)
    superlative_re = re.compile(r"\bhighest(?: semantic)? support\b", re.IGNORECASE)

    for sent in _sentence_split(summary):
        if comparison_re.search(sent) or superlative_re.search(sent):
            continue
        marker_matches = list(_ATOM_MARKER_RE.finditer(sent))
        if strict_markers and marker_matches:
            # In strict audit mode, canonical markers are authoritative.  A bin
            # is associated only with the marker-local span that follows its
            # marker, up to the next canonical marker (or the end of the
            # sentence).  This prevents a sentence containing several
            # ``[atom:j] label (bin)`` entries from forming the Cartesian
            # product of every bin with every atom in that sentence.
            for idx, marker in enumerate(marker_matches):
                aid = int(marker.group(1))
                if aid not in admitted:
                    continue
                span_end = marker_matches[idx + 1].start() if idx + 1 < len(marker_matches) else len(sent)
                local_text = sent[marker.end():span_end]
                present_bins = _bin_labels_in_text_nonoverlap(normalize(local_text), bin_labels)
                if not present_bins:
                    continue
                allowed = _allowed_bins_for_atom(by_id[aid], plan)
                for b in present_bins:
                    record = {
                        "type": "support_bin",
                        "text": local_text.strip() or sent,
                        "atom_id": int(aid),
                        "claimed_bin": b,
                        "released_bin": by_id[aid].get("support_bin"),
                    }
                    if b in allowed:
                        supported.append(record | {"supported_by_released_bin": True})
                    else:
                        unsupported.append(record | {"supported_by_released_bin": False})
            continue

        # Non-strict fallback for summaries without canonical markers.  This
        # retains the previous literal-label behavior and should be interpreted
        # as a heuristic diagnostic rather than the strict audit contract.
        sent_norm = normalize(sent)
        present_bins = _bin_labels_in_text_nonoverlap(sent_norm, bin_labels)
        if not present_bins:
            continue
        ids = _atoms_in_span(sent, all_atoms) & admitted
        if not ids:
            continue
        # Accept only unambiguous fallback assignments.  When a sentence
        # contains several atoms or several bins, the text does not determine a
        # unique atom--bin association without canonical markers.
        if len(ids) != 1 or len(present_bins) != 1:
            continue
        aid = next(iter(ids))
        b = present_bins[0]
        allowed = _allowed_bins_for_atom(by_id[aid], plan)
        record = {
            "type": "support_bin",
            "text": sent,
            "atom_id": int(aid),
            "claimed_bin": b,
            "released_bin": by_id[aid].get("support_bin"),
        }
        if b in allowed:
            supported.append(record | {"supported_by_released_bin": True})
        else:
            unsupported.append(record | {"supported_by_released_bin": False})
    return supported, unsupported


def _ordered_pair_supported(kind: str, left_atom: dict[str, Any], right_atom: dict[str, Any], margin: float) -> bool:
    left = float(left_atom.get("noisy_mass", 0.0))
    right = float(right_atom.get("noisy_mass", 0.0))
    if kind == "higher_support_than":
        return left >= right + margin
    if kind == "lower_support_than":
        return left + margin <= right
    return False


def _detect_comparisons(summary: str, plan: dict[str, Any], all_atoms: list[dict[str, Any]], margin: float = 0.0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Detect pairwise semantic-support comparisons and rank superlatives and validate their support."""
    by_id, _, admitted = _plan_maps(plan)
    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for kind, pat in _COMPARISON_PATTERNS:
        for match in pat.finditer(summary):
            left_span, right_span = match.group(1), match.group(2)
            left_ids = _atoms_in_span(left_span, all_atoms) & admitted
            right_ids = _atoms_in_span(right_span, all_atoms) & admitted
            record: dict[str, Any] = {
                "type": kind,
                "text": match.group(0).strip(),
                "left_atom_ids": sorted(left_ids),
                "right_atom_ids": sorted(right_ids),
            }
            if len(left_ids) == 1 and len(right_ids) == 1:
                left_id, right_id = next(iter(left_ids)), next(iter(right_ids))
                ok = _ordered_pair_supported(kind, by_id[left_id], by_id[right_id], margin)
                record["supported_by_noisy_masses"] = bool(ok)
                (supported if ok else unsupported).append(record)
            else:
                record["supported_by_noisy_masses"] = False
                record["reason"] = "comparison does not identify exactly one admitted atom on each side"
                unsupported.append(record)
    text_norm = normalize(summary)
    if any(w in text_norm for w in _SUPERLATIVE_WORDS):
        top_ids = {int(a.get("atom_id")) for a in (plan.get("selected_atoms", []) or []) if int(a.get("rank", 999999)) == 1}
        for sent in _sentence_split(summary):
            sent_norm = normalize(sent)
            if any(w in sent_norm for w in _SUPERLATIVE_WORDS):
                ids = _atoms_in_span(sent, all_atoms) & admitted
                if not ids:
                    unsupported.append({"type": "superlative", "text": sent, "reason": "no admitted atom identified"})
                elif not ids <= top_ids:
                    unsupported.append({"type": "superlative", "text": sent, "atom_ids": sorted(ids), "reason": "superlative used for non-rank-1 atom"})
                else:
                    supported.append({"type": "superlative", "text": sent, "atom_ids": sorted(ids), "supported_by_rank": True})
    return supported, unsupported


def _description_terms(description: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9_-]{3,}", normalize(description))
    stop = {"this", "that", "with", "from", "about", "theme", "themes", "record", "records", "aggregate", "summary", "private", "public", "high", "moderate", "very", "low"}
    return {w for w in words if w not in stop}


def _detect_possible_unattributed_descriptive_terms(summary: str, plan: dict[str, Any]) -> list[str]:
    """Report terms from non-admitted atom descriptions that appear in the summary.
    
    Because ordinary words can occur in several public descriptions, this diagnostic
    is excluded from the default hard acceptance criterion.
    """
    # Placeholder for future semantic checks. Kept empty to avoid false rejects.
    return []


def verify_summary(
    summary: str,
    plan: dict[str, Any],
    all_atoms: list[dict[str, Any]] | None = None,
    *,
    strict_markers: bool = False,
    comparison_margin: float = 0.0,
    alpha_atom: float = 0.0,
    alpha_comp: float = 0.0,
    reject_raw_details: bool = True,
    reject_unsupported_numbers: bool = True,
    reject_unsupported_importance: bool = True,
) -> dict[str, Any]:
    """Check whether a summary satisfies the public decoder contract.
    
    Canonical atom markers permit direct validation of atom references. Literal
    public-label matching is used when markers are absent. The result certifies only
    consistency with the released plan under the implemented checks, not truth or
    completeness relative to protected records.
    """
    selected = plan.get("selected_atoms", []) or []
    if all_atoms is None:
        all_atoms = selected
    label_map = _all_atom_label_map(all_atoms)
    by_id, _, admitted_ids = _plan_maps(plan)

    alpha_atom = float(alpha_atom)
    alpha_comp = float(alpha_comp)
    if not (0.0 <= alpha_atom <= 1.0):
        raise ValueError("alpha_atom must be in [0, 1]")
    if not (0.0 <= alpha_comp <= 1.0):
        raise ValueError("alpha_comp must be in [0, 1]")

    marker_ids = extract_atom_markers(summary)
    label_ids = extract_atoms_by_label(summary, all_atoms)
    # Canonical markers are authoritative in strict audit mode. Literal-label
    # matching is retained separately as a diagnostic because descriptions or
    # ordinary prose can contain another public atom label without making an
    # atom-level claim about that label.
    mentioned = marker_ids if strict_markers else (marker_ids | label_ids)
    unsupported = mentioned - admitted_ids
    uar = len(unsupported) / max(1, len(mentioned))

    missing_markers = False
    if strict_markers and label_ids:
        # In strict mode, each label mention should appear in a sentence with an
        # admitted canonical marker. This makes verification less ambiguous.
        for sent in _sentence_split(summary):
            if extract_atoms_by_label(sent, all_atoms) and not extract_atom_markers(sent):
                missing_markers = True
                break

    supported_comps, unsupported_comps = _detect_comparisons(summary, plan, all_atoms, margin=comparison_margin)
    ucr = len(unsupported_comps) / max(1, len(supported_comps) + len(unsupported_comps))
    supported_bins, unsupported_bins = _detect_support_bin_claims(
        summary,
        plan,
        all_atoms,
        strict_markers=bool(strict_markers),
    )

    raw_detail_violations = _detect_raw_detail_violations(summary) if reject_raw_details else []
    unsupported_numbers = _detect_unsupported_numbers(summary, plan) if reject_unsupported_numbers else []
    unsupported_importance_claims = (
        _detect_unsupported_importance_claims(summary)
        if reject_unsupported_importance
        else []
    )
    possible_unattributed = _detect_possible_unattributed_descriptive_terms(summary, plan)

    atom_rate_ok = uar <= alpha_atom
    comparison_rate_ok = ucr <= alpha_comp

    unsupported_atom_records = [
        {
            "type": "unsupported_atom",
            "atom_id": int(aid),
            "label": label_map.get(int(aid), f"atom_{aid}"),
        }
        for aid in sorted(unsupported)
    ]
    unsupported_comparison_records = [
        {**comp, "violation_type": "unsupported_comparison"}
        for comp in unsupported_comps
    ]

    violations: list[dict[str, Any]] = []
    if not atom_rate_ok:
        violations.append({
            "type": "unsupported_atom_rate_exceeded",
            "UAR": float(uar),
            "alpha_atom": float(alpha_atom),
            "unsupported_atom_count": int(len(unsupported)),
            "mentioned_atom_count": int(len(mentioned)),
            "unsupported_atoms": unsupported_atom_records,
        })
    if missing_markers:
        violations.append({"type": "missing_canonical_marker", "reason": "strict marker mode requires atom label mentions to be accompanied by [atom:<id>] markers"})
    if not comparison_rate_ok:
        violations.append({
            "type": "unsupported_comparison_rate_exceeded",
            "UCR": float(ucr),
            "alpha_comp": float(alpha_comp),
            "unsupported_comparison_count": int(len(unsupported_comps)),
            "comparison_count": int(len(supported_comps) + len(unsupported_comps)),
            "unsupported_comparisons": unsupported_comparison_records,
        })
    for bin_claim in unsupported_bins:
        violations.append({**bin_claim, "type": "unsupported_support_bin"})
    for raw in raw_detail_violations:
        violations.append({"type": "raw_record_or_structured_identifier_pattern", **raw})
    for num in unsupported_numbers:
        violations.append({"type": "unsupported_numeric_claim", "match": num})
    for claim in unsupported_importance_claims:
        violations.append(claim)

    accept = not violations
    return {
        "verifier_type": "public_contract_verifier_v6_marker_local",
        "verifier_scope": (
            "checks plan consistency using canonical atom markers, public atom labels, "
            "support-bin assignments, semantic-support comparisons, rank superlatives, "
            "numeric claims, explicit unsupported importance-ordering language, and "
            "structured identifier or raw-record patterns; accepts atom/comparison diagnostics "
            "only when UAR <= alpha_atom and UCR <= alpha_comp; does not certify semantic truth, "
            "completeness, or factuality relative to raw records"
        ),
        "strict_markers": bool(strict_markers),
        "alpha_atom": float(alpha_atom),
        "alpha_comp": float(alpha_comp),
        "atom_rate_ok": bool(atom_rate_ok),
        "comparison_rate_ok": bool(comparison_rate_ok),
        "mentioned_atom_ids": sorted(int(x) for x in mentioned),
        "mentioned_atom_labels": [label_map.get(int(x), f"atom_{int(x)}") for x in sorted(mentioned)],
        "mentioned_marker_atom_ids": sorted(int(x) for x in marker_ids),
        "mentioned_literal_label_atom_ids": sorted(int(x) for x in label_ids),
        "admitted_atom_ids": sorted(int(x) for x in admitted_ids),
        "unsupported_atom_ids": sorted(int(x) for x in unsupported),
        "unsupported_atom_labels": [label_map.get(int(x), f"atom_{int(x)}") for x in sorted(unsupported)],
        "unsupported_atom_records": unsupported_atom_records,
        "UAR": float(uar),
        "supported_comparisons": supported_comps,
        "unsupported_comparisons": unsupported_comps,
        "unsupported_comparison_records": unsupported_comparison_records,
        "comparisons_detected": [c.get("text", c.get("type")) for c in supported_comps + unsupported_comps],
        "comparison_count": int(len(supported_comps) + len(unsupported_comps)),
        "supported_comparison_count": int(len(supported_comps)),
        "unsupported_comparison_count": int(len(unsupported_comps)),
        "UCR": float(ucr),
        "supported_support_bins": supported_bins,
        "unsupported_support_bins": unsupported_bins,
        "raw_detail_violations": raw_detail_violations,
        "unsupported_numeric_claims": unsupported_numbers,
        "unsupported_importance_claims": unsupported_importance_claims,
        "possible_unattributed_descriptive_terms": possible_unattributed,
        "violations": violations,
        "accept": bool(accept),
    }


def main() -> None:
    """Command-line entry point for verifying a saved summary against a plan."""
    parser = argparse.ArgumentParser(description="Verify that a summary is consistent with a private plan.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--atoms", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict-markers", action="store_true")
    parser.add_argument("--comparison-margin", type=float, default=0.0)
    parser.add_argument("--alpha-atom", type=float, default=0.0)
    parser.add_argument("--alpha-comp", type=float, default=0.0)
    parser.add_argument("--allow-raw-detail-patterns", action="store_true")
    parser.add_argument("--allow-unsupported-numbers", action="store_true")
    parser.add_argument("--allow-unsupported-importance", action="store_true")
    args = parser.parse_args()
    plan = read_json(args.plan)
    with open(args.summary, "r", encoding="utf-8") as f:
        summary = f.read()
    atoms = read_json(args.atoms) if args.atoms else None
    out = verify_summary(
        summary,
        plan,
        atoms,
        strict_markers=bool(args.strict_markers),
        comparison_margin=float(args.comparison_margin),
        alpha_atom=float(args.alpha_atom),
        alpha_comp=float(args.alpha_comp),
        reject_raw_details=not bool(args.allow_raw_detail_patterns),
        reject_unsupported_numbers=not bool(args.allow_unsupported_numbers),
        reject_unsupported_importance=not bool(args.allow_unsupported_importance),
    )
    write_json(out, args.output)
    print(f"Wrote verifier output to {args.output}")


if __name__ == "__main__":
    main()
