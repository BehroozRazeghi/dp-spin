"""Evaluate generated summaries against their released plans.

The heuristic and optional LLM judges assess summary quality and consistency with the decoder-visible release object. 
In a private DP-SPIN evaluation, the judge input is restricted to the released semantic plan and generated summary.
Non-private baselines use a separate, explicitly labeled evaluation boundary.
"""
from __future__ import annotations

import argparse
import json
import re
from .utils import read_json, write_json
from .llm_client import openai_chat

RUBRIC = """
Score the summary from 1 to 5 on each axis:
coverage: covers the admitted aggregate themes;
faithfulness: does not add unsupported details;
usefulness: useful to a decision maker;
clarity: readable and concise.
Return only valid JSON with keys coverage, faithfulness, usefulness, clarity, rationale.
"""


def heuristic_judge(summary: str, plan: dict) -> dict:
    """Score a summary using deterministic lexical checks against admitted atoms.
    """
    selected = [a.get("label", "") for a in plan.get("selected_atoms", [])]
    mentioned = sum(1 for l in selected if l.lower() in summary.lower())
    coverage = 1 + round(4 * mentioned / max(1, len(selected)))
    bad = any(w in summary.lower() for w in ["for example", "customer named", "on january", "quote"])
    return {"coverage": int(min(5, coverage)), "faithfulness": 2 if bad else 5, "usefulness": int(min(5, max(2, coverage))), "clarity": 4, "rationale": "Heuristic judge based on atom-label coverage and unsupported-detail keywords."}


def llm_judge(summary: str, plan: dict, model: str = "gpt-4o-mini") -> dict:
    """Ask an LLM judge to evaluate summary utility and plan consistency.
    """
    prompt = RUBRIC + "\nPLAN:\n" + json.dumps(plan, ensure_ascii=False, indent=2) + "\nSUMMARY:\n" + summary
    text = openai_chat(prompt, model=model, temperature=0.0, max_tokens=400)
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def main() -> None:
    """Command-line entry point for heuristic or LLM judging.
    """
    p = argparse.ArgumentParser(description="Judge summary utility with heuristic or OpenAI LLM judge.")
    p.add_argument("--plan", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--backend", choices=["heuristic", "openai"], default="heuristic")
    p.add_argument("--model", default="gpt-4o-mini")
    args = p.parse_args()
    plan = read_json(args.plan)
    summary = open(args.summary, encoding="utf-8").read()
    out = llm_judge(summary, plan, args.model) if args.backend == "openai" else heuristic_judge(summary, plan)
    write_json(out, args.output)
    print(f"Wrote judge output to {args.output}")

if __name__ == "__main__":
    main()
