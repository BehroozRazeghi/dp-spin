"""Call an OpenAI-compatible chat model for decoding and evaluation.

Privacy depends on the prompt contents, not on this wrapper. 
Prompts used for a private DP-SPIN release must contain only public information, the released semantic plan, generated text, and other post-processing artifacts permitted by the documented evaluation protocol.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional


def openai_chat(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 700) -> str:
    """Call an OpenAI-compatible chat model.
    
    The function requires ``OPENAI_API_KEY``. Template decoding and non-LLM report
    workflows remain available when no key is configured.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Use --decoder template or set the key.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def maybe_llm(prompt: str, backend: str = "template", model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 700) -> Optional[str]:
    """Return ``None`` when no API key is configured; otherwise call the chat backend."""
    if backend in {"none", "template", "local"}:
        return None
    if backend == "openai":
        return openai_chat(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    raise ValueError(f"Unknown LLM backend: {backend}")


def main() -> None:
    """Run a command-line connectivity check for the configured chat backend."""
    p = argparse.ArgumentParser(description="Test an OpenAI-compatible LLM call.")
    p.add_argument("--prompt", required=True)
    p.add_argument("--model", default="gpt-4o-mini")
    args = p.parse_args()
    print(openai_chat(args.prompt, model=args.model))


if __name__ == "__main__":
    main()
