"""Convert review datasets to the DP-SPIN JSONL schema.

The output preserves a stable contributor identifier when available so that all records from one contributor can be aggregated and clipped in user-level runs.
Metadata needed for atom construction are retained for use on a disjoint auxiliary split.

Supported inputs include Yelp review JSONL with optional business metadata,
Amazon review JSON, JSONL, or gzip-compressed files, and configurable generic review files.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

from .utils import ensure_dir, write_json, write_jsonl


YELP_REVIEW_TEXT_FIELDS = ["text"]
YELP_USER_FIELDS = ["user_id"]
YELP_ID_FIELDS = ["review_id"]

AMAZON_TEXT_FIELDS = ["text", "reviewText", "review_text", "body", "content"]
AMAZON_TITLE_FIELDS = ["title", "summary", "reviewTitle"]
AMAZON_USER_FIELDS = ["user_id", "reviewerID", "reviewer_id", "profile_id"]
AMAZON_ID_FIELDS = ["review_id", "reviewerID_asin_unixReviewTime", "id"]
AMAZON_ITEM_FIELDS = ["asin", "parent_asin", "item_id", "product_id"]
AMAZON_RATING_FIELDS = ["rating", "overall", "stars", "score"]
AMAZON_CATEGORY_FIELDS = ["category", "categories", "main_category", "product_category"]


def _open_text(path: str | Path):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, "r", encoding="utf-8", errors="replace")


def _iter_json_records(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield records from JSONL or JSON-array files, optionally gzip-compressed."""
    p = Path(path)
    with _open_text(p) as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"JSON file {path} must contain a list or JSONL records")
            for row in data:
                if isinstance(row, dict):
                    yield row
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def _iter_csv_records(path: str | Path) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def _iter_records(path: str | Path) -> Iterable[dict[str, Any]]:
    suffixes = "".join(Path(path).suffixes).lower()
    if suffixes.endswith(".csv"):
        yield from _iter_csv_records(path)
    else:
        yield from _iter_json_records(path)


def _first_present(row: dict[str, Any], fields: list[str], default: Any = None) -> Any:
    for f in fields:
        if f in row and row[f] not in {None, ""}:
            return row[f]
    return default


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_as_text(v) for v in value)
    return str(value)


def _rating_bin(value: Any) -> str | None:
    try:
        r = float(value)
    except Exception:
        return None
    if r <= 2:
        return "low rating"
    if r <= 3:
        return "medium rating"
    return "high rating"


def _normalize_category(value: Any) -> tuple[str | None, str | None]:
    """Return primary category and a pipe-joined category string."""
    if value is None or value == "":
        return None, None
    if isinstance(value, str):
        if "," in value:
            cats = [c.strip() for c in value.split(",") if c.strip()]
        elif "|" in value:
            cats = [c.strip() for c in value.split("|") if c.strip()]
        else:
            cats = [value.strip()] if value.strip() else []
    elif isinstance(value, list):
        # Amazon variants sometimes use nested category paths.
        cats = []
        for item in value:
            if isinstance(item, list):
                cats.extend(str(x).strip() for x in item if str(x).strip())
            else:
                s = str(item).strip()
                if s:
                    cats.append(s)
    else:
        cats = [str(value).strip()]
    # Deduplicate while preserving order.
    seen = set()
    clean = []
    for c in cats:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            clean.append(c)
    if not clean:
        return None, None
    return clean[0], " | ".join(clean[:10])


def _load_yelp_business(business_path: str | None) -> dict[str, dict[str, Any]]:
    if not business_path:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in _iter_json_records(business_path):
        bid = row.get("business_id")
        if not bid:
            continue
        primary, joined = _normalize_category(row.get("categories"))
        out[str(bid)] = {
            "business_name": row.get("name"),
            "business_city": row.get("city"),
            "business_state": row.get("state"),
            "business_primary_category": primary,
            "business_categories": joined,
            # Common aliases used by downstream diagnostics and optional filters.
            "primary_category": primary,
            "categories": joined,
        }
    return out


def _parse_category_filter(value: str | None) -> list[str]:
    if value is None:
        return []
    value = value.strip()
    if not value:
        return []
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def _category_matches(joined_categories: str | None, required: list[str]) -> bool:
    if not required:
        return True
    if not joined_categories:
        return False
    cats = [c.strip().lower() for c in str(joined_categories).replace("|", ",").split(",") if c.strip()]
    cat_text = " || ".join(cats)
    return any(req == c or req in cat_text for req in required for c in cats) or any(req in cat_text for req in required)


def prepare_yelp(
    review_path: str,
    output_path: str,
    business_path: str | None = None,
    max_rows: int | None = None,
    min_text_chars: int = 20,
    business_category_filter: str | None = None,
) -> None:
    """Convert Yelp review JSONL into DP-SPIN JSONL with user_id.

    If ``business_category_filter`` is provided, reviews are kept only when their
    joined business categories match at least one requested category string.  This
    requires ``business_path`` so that review ``business_id`` values can be joined
    to Yelp business metadata.  For example, ``Restaurants`` creates a coherent
    restaurant-review subset before auxiliary/target splitting.
    """
    businesses = _load_yelp_business(business_path)
    required_categories = _parse_category_filter(business_category_filter)
    if required_categories and not businesses:
        raise ValueError(
            "Yelp business-category filtering requires --businesses with a valid "
            "yelp_academic_dataset_business.json file."
        )
    rows = []
    n_seen = 0
    n_category_filtered = 0
    for row in _iter_json_records(review_path):
        n_seen += 1
        if max_rows is not None and n_seen > max_rows:
            break
        text = _as_text(_first_present(row, YELP_REVIEW_TEXT_FIELDS)).strip()
        uid = _first_present(row, YELP_USER_FIELDS)
        rid = _first_present(row, YELP_ID_FIELDS, f"yelp_review_{n_seen:09d}")
        if not text or len(text) < min_text_chars or not uid:
            continue
        bid = row.get("business_id")
        business_meta = businesses.get(str(bid)) if bid is not None else None
        if required_categories:
            joined_categories = business_meta.get("categories") if business_meta else None
            if not _category_matches(joined_categories, required_categories):
                n_category_filtered += 1
                continue
        meta = {
            "dataset": "yelp",
            "review_id": rid,
            "user_id": uid,
            "business_id": bid,
            "stars": row.get("stars"),
            "rating_bin": _rating_bin(row.get("stars")),
            "date": row.get("date"),
            "useful": row.get("useful"),
            "funny": row.get("funny"),
            "cool": row.get("cool"),
        }
        if business_meta:
            meta.update({k: v for k, v in business_meta.items() if v not in {None, ""}})
        rows.append({"id": str(rid), "user_id": str(uid), "text": text, "metadata": meta})
    write_jsonl(rows, output_path)
    manifest = {
        "dataset": "yelp",
        "review_path": str(review_path),
        "business_path": str(business_path) if business_path else None,
        "output_path": str(output_path),
        "records_written": len(rows),
        "records_seen": n_seen,
        "business_category_filter": business_category_filter,
        "records_filtered_by_business_category": n_category_filtered,
        "business_metadata_joined": bool(businesses),
        "user_id_field": "user_id",
        "text_field": "text",
        "privacy_note": "Use record-level DP for one review as the protected unit, or user-level DP by grouping all records with the same user_id.",
    }
    write_json(manifest, str(output_path) + ".manifest.json")
    print(f"Wrote {len(rows)} Yelp records to {output_path}")


def prepare_amazon(
    input_path: str,
    output_path: str,
    max_rows: int | None = None,
    min_text_chars: int = 20,
    user_field: str | None = None,
    text_field: str | None = None,
    category_name: str | None = None,
) -> None:
    """Convert Amazon review JSON/JSONL/JSON.gz into DP-SPIN JSONL with user_id.

    ``category_name`` is useful for Amazon Reviews 2023 category files, where the
    category is encoded in the downloaded file name rather than in each review
    row.  It is stored as ``metadata.primary_category`` and can be used to build
    public atom dictionaries.
    """
    rows = []
    n_seen = 0
    for row in _iter_records(input_path):
        n_seen += 1
        if max_rows is not None and n_seen > max_rows:
            break
        uid = row.get(user_field) if user_field else _first_present(row, AMAZON_USER_FIELDS)
        body = row.get(text_field) if text_field else _first_present(row, AMAZON_TEXT_FIELDS)
        title = _first_present(row, AMAZON_TITLE_FIELDS)
        text = _as_text(body).strip()
        title_text = _as_text(title).strip()
        if title_text and title_text not in text:
            text = f"{title_text}. {text}".strip()
        if not text or len(text) < min_text_chars or not uid:
            continue
        item = _first_present(row, AMAZON_ITEM_FIELDS)
        rating = _first_present(row, AMAZON_RATING_FIELDS)
        cat_raw = _first_present(row, AMAZON_CATEGORY_FIELDS)
        primary, joined = _normalize_category(cat_raw)
        rid = _first_present(row, AMAZON_ID_FIELDS)
        if not rid:
            rid = f"amazon_{uid}_{item}_{row.get('unixReviewTime', n_seen)}"
        meta = {
            "dataset": "amazon",
            "review_id": rid,
            "user_id": uid,
            "item_id": item,
            "rating": rating,
            "rating_bin": _rating_bin(rating),
            "verified": row.get("verified") if row.get("verified") is not None else row.get("verified_purchase"),
            "helpful_vote": row.get("helpful_vote") if row.get("helpful_vote") is not None else row.get("helpful_votes"),
            "timestamp": row.get("timestamp") or row.get("unixReviewTime") or row.get("reviewTime") or row.get("sort_timestamp"),
        }
        if category_name and not primary:
            primary = category_name.replace("_", " ")
            joined = primary
        if primary:
            meta["primary_category"] = primary
        if joined:
            meta["categories"] = joined
        rows.append({"id": str(rid), "user_id": str(uid), "text": text, "metadata": {k: v for k, v in meta.items() if v not in {None, ""}}})
    write_jsonl(rows, output_path)
    manifest = {
        "dataset": "amazon",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "records_written": len(rows),
        "records_seen": n_seen,
        "user_id_field": user_field or "inferred",
        "text_field": text_field or "inferred",
        "privacy_note": "Use record-level DP for one review as the protected unit, or user-level DP by grouping all records with the same user_id.",
    }
    write_json(manifest, str(output_path) + ".manifest.json")
    print(f"Wrote {len(rows)} Amazon records to {output_path}")


def main() -> None:
    """Convert Yelp, Amazon, or generic review data from the command line."""
    p = argparse.ArgumentParser(description="Prepare Yelp/Amazon/generic review data for DP-SPIN.")
    sub = p.add_subparsers(dest="cmd", required=True)

    y = sub.add_parser("yelp", help="Prepare Yelp Open Dataset review JSONL.")
    y.add_argument("--reviews", required=True, help="Path to yelp_academic_dataset_review.json")
    y.add_argument("--businesses", default=None, help="Optional path to yelp_academic_dataset_business.json")
    y.add_argument("--output", required=True)
    y.add_argument("--max-rows", type=int, default=None)
    y.add_argument("--min-text-chars", type=int, default=20)
    y.add_argument(
        "--business-category-filter",
        default=None,
        help="Optional comma-separated Yelp business category filter, e.g. Restaurants. Requires --businesses.",
    )

    a = sub.add_parser("amazon", help="Prepare Amazon review JSON/JSONL/JSON.gz.")
    a.add_argument("--input", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--max-rows", type=int, default=None)
    a.add_argument("--min-text-chars", type=int, default=20)
    a.add_argument("--user-field", default=None)
    a.add_argument("--text-field", default=None)
    a.add_argument("--category-name", default=None, help="Optional category label to store as metadata.primary_category, e.g. All_Beauty.")

    args = p.parse_args()
    ensure_dir(Path(args.output).parent)
    if args.cmd == "yelp":
        prepare_yelp(
            args.reviews,
            args.output,
            args.businesses,
            args.max_rows,
            args.min_text_chars,
            args.business_category_filter,
        )
    elif args.cmd == "amazon":
        prepare_amazon(args.input, args.output, args.max_rows, args.min_text_chars, args.user_field, args.text_field, args.category_name)


if __name__ == "__main__":
    main()
