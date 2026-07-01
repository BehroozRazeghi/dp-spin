"""Verbalize a released semantic plan.

The decoder is post-processing of the released plan. For a DP-SPIN release, its input is limited to admitted public atoms, released noisy masses or prevalence
bins, and public decoding instructions. It must not receive raw records, raw excerpts, per-record contribution vectors, non-private representatives, or
non-admitted atom labels.

The ``audit`` mode requests canonical atom markers for verifier checks. 
Them ``display`` mode omits markers and exact noisy masses while preserving the same plan-only input boundary.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from .utils import read_json, ensure_dir
from .llm_client import maybe_llm


DECODER_MODES = ("audit", "display")


def _atom_marker(atom: dict) -> str:
    return f"[atom:{int(atom.get('atom_id', atom.get('index', -1)))}]"


def _format_plan_entries(plan: dict) -> list[str]:
    lines: list[str] = []
    for a in plan.get("selected_atoms", []):
        noisy_mass = float(a.get("noisy_mass", 0.0))
        marker = _atom_marker(a)
        lines.append(
            f"- {marker} label={a.get('label')} | description={a.get('description')} | "
            f"bin={a.get('prevalence_bin')} | noisy_mass={noisy_mass:.3f}"
        )
    return lines


def _is_non_private_reference_plan(plan: dict) -> bool:
    """Identify an evaluation-only non-private reference plan."""
    return bool(plan.get("evaluation_only")) or str(plan.get("reference_type", "")).startswith("non_private")


def render_prompt(
    plan: dict,
    verifier_feedback: str | None = None,
    attempt: int = 1,
    mode: str = "audit",
) -> str:
    """Build the plan-only prompt for the selected decoder mode.
    
    ``audit`` requires canonical atom markers and permits released noisy masses so
    the public verifier can check atom and numeric claims. ``display`` omits atom
    identifiers and exact noisy masses and permits only prevalence language supported
    by the released bins.
    """
    if mode not in DECODER_MODES:
        raise ValueError(f"Unknown decoder mode {mode!r}; expected one of {DECODER_MODES}")

    is_reference = _is_non_private_reference_plan(plan)
    plan_kind = "non-private reference" if is_reference else "differentially private"

    if mode == "display":
        lines = [
            f"You are given a {plan_kind} aggregate summary plan.",
            "Write a concise aggregate summary using only the listed plan entries.",
            "Rules:",
            "1. Do not introduce any theme not listed in the plan.",
            "2. Do not invent examples, names, organizations, dates, quotes, events, locations, or subgroups.",
            "3. Do not mention individual records or claim to have seen raw records.",
            "4. Do not infer causes, demographics, product identities, businesses, user groups, or subgroups beyond the atom labels and descriptions.",
            "5. Use prevalence language only from the released prevalence bins. Do not report exact noisy masses.",
            "6. Do not include atom identifiers such as [atom:<id>] in the output.",
            "7. If the plan is low-support, noisy, or ambiguous, state uncertainty rather than adding unsupported content.",
            "8. Write one concise paragraph followed by at most three short bullets only if bullets improve clarity.",
        ]
    else:
        lines = [
            f"You are given a {plan_kind} aggregate summary plan.",
            "Write a concise aggregate summary using only the listed plan entries.",
            "Every substantive theme mention must include its canonical marker [atom:<id>] exactly as shown.",
            "Rules:",
            "1. Do not introduce any theme not listed in the plan.",
            "2. Do not invent examples, names, organizations, dates, quotes, or events.",
            "3. Do not mention individual records or claim to have seen raw records.",
            "4. Do not infer causes, demographics, locations, product identities, businesses, user groups, or subgroups beyond the atom labels and descriptions.",
            "5. Use prevalence language only according to the released prevalence bins and released noisy masses.",
            "6. Comparisons such as 'more common than' are allowed only when supported by the released noisy masses.",
            "7. If the plan is low-support, noisy, or ambiguous, state uncertainty rather than adding unsupported content.",
        ]
    if verifier_feedback:
        lines.extend([
            "",
            f"Regeneration attempt {attempt}.",
            "The previous draft was rejected by a public plan-consistency verifier.",
            "Verifier feedback based only on public plan information:",
            verifier_feedback,
            "Revise by removing unsupported labels, unsupported numbers, raw details, or unsupported comparisons.",
        ])
    plan_heading = "Non-private reference summary plan:" if is_reference else "Released differentially private summary plan:"
    lines.extend(["", plan_heading])
    lines.extend(_format_plan_entries(plan))
    lines.append("")
    if mode == "display":
        lines.append("Output format: one concise paragraph, optionally followed by at most three short bullets. Do not include atom IDs or exact noisy masses.")
    else:
        lines.append("Output format: one short paragraph followed by up to five bullets. Use canonical markers in both the paragraph and the bullets.")
    return "\n".join(lines)


def template_decode(plan: dict, mode: str = "audit") -> str:
    """Render the released plan deterministically, with audit markers when requested."""
    atoms = plan.get("selected_atoms", [])
    is_reference = _is_non_private_reference_plan(plan)
    plan_name = "non-private reference summary plan" if is_reference else "released differentially private summary plan"
    if not atoms:
        return f"The {plan_name} did not admit any aggregate themes."

    def phrase(a: dict) -> str:
        b = a.get("prevalence_bin")
        label = a.get("label")
        if mode == "display":
            return f"{label}" + (f" ({b})" if b else "")
        marker = _atom_marker(a)
        return f"{marker} {label}" + (f" ({b})" if b else "")

    top = atoms[: min(3, len(atoms))]
    if mode == "display":
        paragraph = (
            f"The {plan_name} identifies the following aggregate themes: "
            + ", ".join(phrase(a) for a in top)
            + ". The summary is restricted to admitted plan entries and prevalence bins shown in the plan."
        )
        bullets = "\n".join(f"- {phrase(a)}: {a.get('description','')}" for a in atoms[:3])
    else:
        paragraph = (
            f"The {plan_name} identifies the following admitted aggregate themes: "
            + ", ".join(phrase(a) for a in top)
            + ". This summary is restricted to admitted atoms and aggregate values shown in the plan."
        )
        bullets = "\n".join(f"- {phrase(a)}: {a.get('description','')}" for a in atoms[:5])
    return paragraph + "\n" + bullets


def decode_plan(
    plan: dict,
    backend: str = "template",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 700,
    verifier_feedback: str | None = None,
    attempt: int = 1,
    mode: str = "audit",
) -> tuple[str, str]:
    """Verbalize a released plan with the template or OpenAI backend."""
    prompt = render_prompt(plan, verifier_feedback=verifier_feedback, attempt=attempt, mode=mode)
    llm_text = maybe_llm(prompt, backend=backend, model=model, temperature=temperature, max_tokens=max_tokens)
    return (llm_text if llm_text is not None else template_decode(plan, mode=mode), prompt)


def main() -> None:
    """Decode a saved semantic plan from the command line."""
    p = argparse.ArgumentParser(description="Decode a private plan into a plan-only summary.")
    p.add_argument("--plan", required=True)
    p.add_argument("--summary-output", "--output", dest="summary_output", required=True)
    p.add_argument("--prompt-output", default=None)
    p.add_argument("--backend", choices=["template", "openai"], default="template")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--mode", choices=list(DECODER_MODES), default="audit")
    args = p.parse_args()
    plan = read_json(args.plan)
    summary, prompt = decode_plan(
        plan,
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        mode=args.mode,
    )
    ensure_dir(Path(args.summary_output).parent)
    open(args.summary_output, "w", encoding="utf-8").write(summary)
    if args.prompt_output:
        ensure_dir(Path(args.prompt_output).parent)
        open(args.prompt_output, "w", encoding="utf-8").write(prompt)
    print(f"Wrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()
