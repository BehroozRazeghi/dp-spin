"""Insert controlled probe records for empirical diagnostics.

The inserted text corresponds to a known public probe atom. 
Subsequent experiments measure whether that coordinate is admitted to the released plan and whether its text appears in the generated summary. 
These measurements assess empirical behavior, not the formal privacy guarantee.
"""
from __future__ import annotations

import argparse
import random
from .utils import read_jsonl, write_jsonl, set_seed

DEFAULT_PROBES = [
    "The complaint contains the secret phrase BLUE-LANTERN-791.",
    "The customer reports a rare event called Zorvian escrow inversion.",
    "The support ticket mentions the unique code HYPERION-X17.",
]


def insert_probes(input_path: str, output_path: str, support: int, seed: int, probe_text: str | None = None) -> None:
    """Append a controlled probe string to a controlled number of randomly chosen records.
    """
    set_seed(seed)
    records = read_jsonl(input_path)
    probe_text = probe_text or DEFAULT_PROBES[0]
    for i in range(support):
        records.append({
            "id": f"probe_{seed}_{i:04d}",
            "text": probe_text,
            "metadata": {"is_probe": True, "probe_support": support, "probe_text": probe_text},
        })
    random.shuffle(records)
    write_jsonl(records, output_path)
    print(f"Wrote {len(records)} records with probe support {support} to {output_path}")


def main() -> None:
    """Command-line entry point for probe insertion.
    """
    parser = argparse.ArgumentParser(description="Insert synthetic probe records into a JSONL corpus.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--support", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text", default=None)
    args = parser.parse_args()
    insert_probes(args.input, args.output, args.support, args.seed, args.text)


if __name__ == "__main__":
    main()
