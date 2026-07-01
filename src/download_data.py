"""Acquire the public datasets used by the experiment scripts.

CFPB data are retrieved from the public CFPB API. 
Amazon Reviews 2023 category files can be downloaded from the McAuley Lab source. 
Yelp Open Dataset files are validated locally or downloaded through the Kaggle CLI after the user accepts the applicable dataset terms.

Dataset conversion to the repository JSONL schema is handled by :mod:`src.prepare_cfpb` and :mod:`src.prepare_reviews`.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from .download_cfpb import download_cfpb

AMAZON_2023_REVIEW_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "review_categories/{category}.jsonl.gz"
)
AMAZON_2023_META_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/"
    "meta_categories/meta_{category}.jsonl.gz"
)


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _check_existing(path: str | Path, name: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"{name} raw file not found: {p}\n"
            "Either download it manually and place it at this path, or rerun "
            "with the appropriate downloader option."
        )
    print(f"Found {name} raw file: {p}")
    return p


def _download_url(url: str, output: str | Path, force: bool = False) -> Path:
    out = Path(output)
    if out.exists() and not force:
        print(f"Reusing existing file: {out}")
        return out
    _ensure_parent(out)
    tmp = out.with_suffix(out.suffix + ".part")
    print(f"Downloading:\n  {url}\n  -> {out}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f, length=1024 * 1024)
    tmp.replace(out)
    print(f"Wrote {out}")
    return out


def _find_file(root: Path, names: list[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    for p in root.rglob("*"):
        if p.name in names:
            return p
    return None


def _download_yelp_with_kaggle(raw_dir: str | Path, force: bool = False) -> tuple[Path, Path | None]:
    """Download Yelp Open Dataset through the Kaggle CLI.

    The command uses the official Kaggle dataset entry. It requires the
    ``kaggle`` package, a valid API token, and prior acceptance of the dataset
    terms.
    """
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    review = _find_file(raw, ["yelp_academic_dataset_review.json"])
    business = _find_file(raw, ["yelp_academic_dataset_business.json"])
    if review and not force:
        print(f"Reusing existing Yelp files under {raw}")
        return review, business

    if shutil.which("kaggle") is None:
        raise SystemExit(
            "Kaggle CLI is not installed. Install and configure it first:\n"
            "  pip install kaggle\n"
            "  mkdir -p ~/.kaggle && place kaggle.json there\n"
            "  chmod 600 ~/.kaggle/kaggle.json\n"
            "Also accept the Yelp dataset terms on Kaggle before downloading."
        )

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "yelp-dataset/yelp-dataset",
        "-p",
        str(raw),
        "--unzip",
    ]
    if force:
        cmd.append("--force")
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    review = _find_file(raw, ["yelp_academic_dataset_review.json"])
    business = _find_file(raw, ["yelp_academic_dataset_business.json"])
    if review is None:
        raise SystemExit(f"Kaggle download completed but review file was not found under {raw}")
    print(f"Yelp review file: {review}")
    if business:
        print(f"Yelp business file: {business}")
    else:
        print("Yelp business file not found; review-only preparation is still possible.")
    return review, business


def main() -> None:
    """Run the selected public-dataset acquisition command."""
    parser = argparse.ArgumentParser(description="Acquire or validate raw datasets for DP-SPIN.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cfpb", help="Download CFPB complaint data from the public CFPB API.")
    c.add_argument("--output", default="data/raw/cfpb_complaints.csv")
    c.add_argument("--size", type=int, default=20000)
    c.add_argument("--no-narrative-filter", action="store_true")
    c.add_argument("--product", default=None)
    c.add_argument("--date-min", default=None)
    c.add_argument("--date-max", default=None)
    c.add_argument("--format", choices=["csv", "json"], default="csv")

    y = sub.add_parser("yelp", help="Download or validate Yelp Open Dataset files.")
    y.add_argument("--source", choices=["manual", "kaggle"], default="manual")
    y.add_argument("--raw-dir", default="data/raw/yelp")
    y.add_argument("--reviews", default="data/raw/yelp/yelp_academic_dataset_review.json")
    y.add_argument("--businesses", default="data/raw/yelp/yelp_academic_dataset_business.json")
    y.add_argument("--force", action="store_true")

    a = sub.add_parser("amazon", help="Download or validate Amazon Reviews 2023 category files.")
    a.add_argument("--source", choices=["official", "manual"], default="official")
    a.add_argument("--category", default="All_Beauty", help="Amazon Reviews 2023 category name, e.g. All_Beauty or Digital_Music.")
    a.add_argument("--reviews", default="data/raw/amazon_reviews.jsonl.gz")
    a.add_argument("--metadata", default=None, help="Optional output path for item metadata JSONL.GZ.")
    a.add_argument("--with-metadata", action="store_true", help="Also download item metadata for the chosen category.")
    a.add_argument("--url", default=None, help="Override review URL. Useful for mirrors or cached institutional storage.")
    a.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.cmd == "cfpb":
        download_cfpb(
            output=args.output,
            size=args.size,
            has_narrative=not args.no_narrative_filter,
            product=args.product,
            date_min=args.date_min,
            date_max=args.date_max,
            fmt=args.format,
        )
    elif args.cmd == "yelp":
        if args.source == "kaggle":
            review, business = _download_yelp_with_kaggle(args.raw_dir, force=args.force)
            print("Next step:")
            cmd = [
                "python", "-m", "src.prepare_reviews", "yelp",
                "--reviews", str(review),
                "--output", "data/processed/yelp_reviews_all.jsonl",
            ]
            if business:
                cmd.extend(["--businesses", str(business)])
            print("  " + " ".join(cmd))
        else:
            _check_existing(args.reviews, "Yelp review")
            if args.businesses:
                bpath = Path(args.businesses)
                if bpath.exists():
                    _check_existing(bpath, "Yelp business")
                else:
                    print(f"Optional Yelp business file not found: {bpath}. Continuing with review-only data.")
            print("Next step: python -m src.prepare_reviews yelp --reviews ... --output ...")
    elif args.cmd == "amazon":
        if args.source == "manual":
            _check_existing(args.reviews, "Amazon review")
            if args.metadata:
                _check_existing(args.metadata, "Amazon metadata")
            print("Next step: python -m src.prepare_reviews amazon --input ... --output ...")
        else:
            url = args.url or AMAZON_2023_REVIEW_URL.format(category=args.category)
            _download_url(url, args.reviews, force=args.force)
            if args.with_metadata:
                meta_out = args.metadata or str(Path(args.reviews).with_name(f"meta_{args.category}.jsonl.gz"))
                meta_url = AMAZON_2023_META_URL.format(category=args.category)
                _download_url(meta_url, meta_out, force=args.force)
            print("Next step: python -m src.prepare_reviews amazon --input ... --output ...")


if __name__ == "__main__":
    main()
