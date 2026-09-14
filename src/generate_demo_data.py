"""Generate synthetic support-ticket records for local pipeline checks.

The generated corpus supports installation tests and examples without external data. 
It is not an empirical privacy benchmark unless a separate experiment explicitly defines disjoint auxiliary and protected target splits.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from .utils import write_json, write_jsonl, ensure_dir, set_seed

TOPICS = [
    ("unexpected fees", "Complaints about fees perceived as unexpected, unexplained, or excessive."),
    ("billing dispute", "Disputes about bills, charges, or payment amounts."),
    ("account access problem", "Problems logging in, accessing accounts, or using online portals."),
    ("refund not received", "Refunds requested but delayed, missing, or denied."),
    ("poor customer communication", "Customers report unclear, delayed, or unhelpful communication."),
    ("identity theft or fraud", "Reports involving suspected fraud, identity theft, or unauthorized activity."),
    ("loan servicing delay", "Delays in loan servicing, processing, or account updates."),
    ("credit reporting dispute", "Disputes about incorrect or outdated credit report information."),
    ("debt collection contact", "Complaints about repeated, confusing, or aggressive collection contacts."),
    ("technical error", "Technical failures, app crashes, broken forms, or website errors."),
    ("documentation request", "Customers request documents, statements, letters, or confirmations."),
    ("payment processing delay", "Payments not posted, delayed, duplicated, or applied incorrectly."),
    ("service cancellation", "Problems cancelling, closing, or terminating an account or service."),
    ("disputed interest", "Issues concerning interest rates, interest charges, or accrued interest."),
    ("address or contact update", "Problems updating address, phone, email, or contact information."),
]

TEMPLATES = [
    "I contacted support because of {topic}. The issue has not been resolved and the response was slow.",
    "My complaint concerns {topic}. I asked for clarification but the company did not provide a clear explanation.",
    "The main problem is {topic}. I need the company to review the case and correct the record.",
    "I experienced {topic} and the support process was confusing. The response did not address my concern.",
    "This ticket is about {topic}. I have followed up several times and still need a resolution.",
]


def generate(out_records: str, out_atoms: str, n: int, seed: int, n_users: int = 0) -> None:
    """Create synthetic records with topic-specific boilerplate and optional metadata.
    """
    set_seed(seed)
    atoms = [
        {"atom_id": i, "label": label, "description": desc, "domain": "demo_support"}
        for i, (label, desc) in enumerate(TOPICS)
    ]
    weights = [0.18, 0.14, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.04, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01]
    rows = []
    for i in range(n):
        idx = random.choices(range(len(TOPICS)), weights=weights, k=1)[0]
        label, _ = TOPICS[idx]
        text = random.choice(TEMPLATES).format(topic=label)
        user_id = f"user_{(i % n_users):05d}" if n_users and n_users > 0 else f"user_{i:05d}"
        rows.append({
            "id": f"demo_{i:06d}",
            "user_id": user_id,
            "text": text,
            "metadata": {"true_topic": label, "true_atom_id": idx},
        })
    write_json(atoms, out_atoms)
    write_jsonl(rows, out_records)
    print(f"Wrote atoms: {out_atoms}")
    print(f"Wrote records: {out_records}")


def main() -> None:
    """Command-line entry point for synthetic data generation.
    """
    parser = argparse.ArgumentParser(description="Generate a synthetic demo dataset and public atom universe.")
    parser.add_argument("--records", default="data/processed/demo_records.jsonl")
    parser.add_argument("--atoms", default="data/public_atoms/demo_atoms.json")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-users", type=int, default=0, help="Number of synthetic users; records are assigned round-robin. Default 0 makes one user per record.")
    args = parser.parse_args()
    generate(args.records, args.atoms, args.n, args.seed, args.n_users)


if __name__ == "__main__":
    main()
