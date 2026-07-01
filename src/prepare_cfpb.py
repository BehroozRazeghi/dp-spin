"""Convert CFPB complaint data to the DP-SPIN JSONL schema.

Each output row contains ``id``, ``text``, and ``metadata``. 
The public CFPB export does not provide a stable consumer identifier suitable for grouping all contributions from one person, so this preparation path supports record-level experiments only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .load_data import csv_to_jsonl
from .utils import write_json


def prepare_cfpb(
    input_path: str,
    output_path: str,
    text_col: str | None = None,
    id_col: str | None = None,
    max_rows: int | None = None,
) -> None:
    """Prepare CFPB complaint data for record-level DP-SPIN experiments.

    The public CFPB export does not provide a stable consumer identifier suitable
    for user-level DP. The conversion therefore omits a top-level ``user_id``
    field. User-level experiments require a dataset with a stable contributor
    identifier, such as the supported Yelp or Amazon review inputs.
    """
    csv_to_jsonl(
        input_path=input_path,
        output_path=output_path,
        text_col=text_col,
        id_col=id_col,
        max_rows=max_rows,
        user_id_col=None,
    )
    manifest = {
        "dataset": "cfpb",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "protected_unit_supported": "record",
        "user_level_dp_note": "The public CFPB export does not expose a stable consumer identifier. Treat each complaint narrative as one protected record.",
    }
    write_json(manifest, str(output_path) + ".manifest.json")


def main() -> None:
    """Convert CFPB data to the repository JSONL schema from the command line."""
    p = argparse.ArgumentParser(description="Prepare CFPB complaint CSV data for DP-SPIN.")
    p.add_argument("--input", required=True, help="Path to CFPB CSV export.")
    p.add_argument("--output", required=True, help="Output JSONL path.")
    p.add_argument("--text-col", default=None)
    p.add_argument("--id-col", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    prepare_cfpb(args.input, args.output, args.text_col, args.id_col, args.max_rows)


if __name__ == "__main__":
    main()
