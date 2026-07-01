"""Convert CSV data to the JSONL record schema used by DP-SPIN.

Each output record contains a ``text`` field and a ``metadata`` dictionary. 
An optional stable user identifier is copied to a top-level field for user-level contribution aggregation. If the output is a protected target split, it must not subsequently be used to name atoms or fit the public assignment representation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from .utils import write_jsonl

COMMON_TEXT_COLS = [
    "Consumer complaint narrative",
    "complaint_what_happened",
    "narrative",
    "text",
    "body",
    "description",
]
COMMON_ID_COLS = ["Complaint ID", "complaint_id", "id", "ticket_id"]


def infer_col(cols: list[str], candidates: list[str]) -> str | None:
    """Pick the first available candidate column from a CSV header.
    """
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def csv_to_jsonl(
    input_path: str,
    output_path: str,
    text_col: str | None = None,
    id_col: str | None = None,
    max_rows: int | None = None,
    user_id_col: str | None = None,
    user_id_output_field: str = "user_id",
) -> None:
    """Convert a CSV file into JSONL records with text and metadata fields.

    If ``user_id_col`` is provided, the value is copied to a top-level
    ``user_id_output_field``.  This is required for user-level DP runs, where
    all records from the same user must be grouped before contribution clipping.
    The original column is still retained in ``metadata``.
    """
    df = pd.read_csv(input_path, nrows=max_rows, low_memory=False)
    text_col = text_col or infer_col(list(df.columns), COMMON_TEXT_COLS)
    if not text_col:
        raise ValueError(f"Could not infer text column. Available columns: {list(df.columns)}")
    id_col = id_col or infer_col(list(df.columns), COMMON_ID_COLS)
    rows = []
    for i, row in df.iterrows():
        text = row.get(text_col)
        if pd.isna(text) or not str(text).strip():
            continue
        rid = str(row.get(id_col)) if id_col and not pd.isna(row.get(id_col)) else f"record_{i:08d}"
        meta = {str(c): (None if pd.isna(row[c]) else row[c].item() if hasattr(row[c], "item") else row[c]) for c in df.columns if c != text_col}
        out = {"id": rid, "text": str(text), "metadata": meta}
        if user_id_col:
            if user_id_col not in df.columns:
                raise ValueError(f"user_id_col={user_id_col!r} is not in the CSV columns")
            uid = row.get(user_id_col)
            if not pd.isna(uid) and str(uid).strip():
                out[user_id_output_field] = str(uid)
        rows.append(out)
    write_jsonl(rows, output_path)
    print(f"Wrote {len(rows)} records to {output_path}")
    print(f"Text column: {text_col}")
    if id_col:
        print(f"ID column: {id_col}")
    if user_id_col:
        print(f"User ID column: {user_id_col} -> top-level field: {user_id_output_field}")


def main() -> None:
    """Command-line entry point for CSV-to-JSONL preprocessing.
    """
    p = argparse.ArgumentParser(description="Convert a CSV dataset to the project JSONL record format.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--text-col", default=None)
    p.add_argument("--id-col", default=None)
    p.add_argument("--user-id-col", default=None, help="Optional CSV column copied to a top-level user_id field for user-level DP runs.")
    p.add_argument("--user-id-output-field", default="user_id", help="Top-level JSONL field name for --user-id-col.")
    p.add_argument("--max-rows", type=int, default=None)
    args = p.parse_args()
    csv_to_jsonl(args.input, args.output, args.text_col, args.id_col, args.max_rows, args.user_id_col, args.user_id_output_field)


if __name__ == "__main__":
    main()
