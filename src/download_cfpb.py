"""Download CFPB complaint records from the public CFPB API.

The downloaded data are public benchmark inputs. 
Experiments that emulate a private deployment should first create disjoint auxiliary and protected target splits, use the auxiliary split for atom construction and representation fitting, and treat only the target split as protected.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlencode
import requests

DEFAULT_ENDPOINT = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"


def download_cfpb(output: str, size: int = 5000, has_narrative: bool = True, product: str | None = None, date_min: str | None = None, date_max: str | None = None, fmt: str = "csv") -> None:
    """Download a CFPB complaint subset through the public CFPB search API.

    The CFPB API may evolve. If this endpoint changes, use the CFPB export UI
    to download a CSV and then run src.load_data on that local file.
    """
    params: dict[str, str | int] = {
        "format": fmt,
        "no_aggs": "true",
        "size": int(size),
        "frm": 0,
        "sort": "created_date_desc",
    }
    if has_narrative:
        params["has_narrative"] = "true"
    if product:
        params["product"] = product
    if date_min:
        params["date_received_min"] = date_min
    if date_max:
        params["date_received_max"] = date_max
    url = DEFAULT_ENDPOINT + "?" + urlencode(params)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    print(f"Wrote CFPB data to {out}")
    print(f"URL: {url}")


def main() -> None:
    """Command-line entry point for CFPB public benchmark download.
    """
    p = argparse.ArgumentParser(description="Download a CFPB consumer-complaint subset.")
    p.add_argument("--output", default="data/raw/cfpb_complaints.csv")
    p.add_argument("--size", type=int, default=5000)
    p.add_argument("--no-narrative-filter", action="store_true")
    p.add_argument("--product", default=None)
    p.add_argument("--date-min", default=None, help="YYYY-MM-DD")
    p.add_argument("--date-max", default=None, help="YYYY-MM-DD")
    p.add_argument("--format", choices=["csv", "json"], default="csv")
    args = p.parse_args()
    download_cfpb(args.output, args.size, not args.no_narrative_filter, args.product, args.date_min, args.date_max, args.format)


if __name__ == "__main__":
    main()
