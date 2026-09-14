"""Chat-model backends used only for post-processing or offline evaluation.

Privacy depends on prompt contents, not on this wrapper.  A private DP-SPIN
release may send only the released semantic plan and public rules to a decoder.
The optional Hugging Face backend enables an open-weight decoder that can be
kept distinct from an API-based automated judge.
"""
from __future__ import annotations

import argparse
import os
from functools import lru_cache
from typing import Optional


def openai_chat(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 700) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Use --decoder template/hf_local or set the key.")
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


@lru_cache(maxsize=8)
def _hf_pipeline(
    model: str,
    revision: str | None,
    device_map: str,
    local_files_only: bool,
    trust_remote_code: bool,
):
    try:
        from transformers import pipeline  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "The hf_local backend requires the optional 'local-llm' dependencies: "
            "python -m pip install -r requirements-local-llm.txt"
        ) from exc
    kwargs = {
        "task": "text-generation",
        "model": model,
        "tokenizer": model,
        "device_map": device_map,
        "trust_remote_code": bool(trust_remote_code),
    }
    if revision:
        kwargs["revision"] = str(revision)
    if local_files_only:
        # ``pipeline`` forwards model_kwargs to the underlying loader. Some
        # older Transformers versions do not accept this option, so the
        # compatibility fallback below fails closed when a local-only run was
        # explicitly requested rather than silently downloading another model.
        kwargs["model_kwargs"] = {"local_files_only": True}
    try:
        return pipeline(**kwargs)
    except TypeError:
        if local_files_only:
            raise RuntimeError(
                "The installed Transformers version cannot enforce local_files_only. "
                "Upgrade Transformers or pre-load the pinned model with a compatible version."
            )
        kwargs.pop("model_kwargs", None)
        return pipeline(**kwargs)


def huggingface_chat(
    prompt: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 700,
    device_map: str = "auto",
    model_revision: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> str:
    """Generate text with a local/open-weight Hugging Face model."""
    generator = _hf_pipeline(
        model,
        model_revision,
        device_map,
        bool(local_files_only),
        bool(trust_remote_code),
    )
    do_sample = bool(temperature and temperature > 0)
    output = generator(
        prompt,
        max_new_tokens=int(max_tokens),
        do_sample=do_sample,
        temperature=float(temperature) if do_sample else None,
        return_full_text=False,
    )
    if not output:
        return ""
    text = output[0].get("generated_text", "")
    if isinstance(text, list):
        # Some chat pipelines return a message list.
        return str(text[-1].get("content", "")) if text else ""
    return str(text)


def maybe_llm(
    prompt: str,
    backend: str = "template",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 700,
    *,
    model_revision: str | None = None,
    device_map: str = "auto",
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> Optional[str]:
    """Call the selected model backend or return ``None`` for deterministic modes."""
    backend_norm = str(backend).lower()
    if backend_norm in {"none", "template", "local", "heuristic"}:
        return None
    if backend_norm == "openai":
        return openai_chat(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
    if backend_norm in {"hf_local", "huggingface", "transformers"}:
        return huggingface_chat(
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            device_map=device_map,
            model_revision=model_revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    raise ValueError(f"Unknown LLM backend: {backend}")


def main() -> None:
    p = argparse.ArgumentParser(description="Test a configured chat-model backend.")
    p.add_argument("--prompt", required=True)
    p.add_argument("--backend", choices=["openai", "hf_local"], default="openai")
    p.add_argument("--model", required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--model-revision", default=None)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()
    text = maybe_llm(
        args.prompt,
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        model_revision=args.model_revision,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    print(text or "")


if __name__ == "__main__":
    main()
