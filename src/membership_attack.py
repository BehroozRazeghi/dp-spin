"""Compare paired releases after deleting one protected record.

The diagnostic computes a set-distance score between outputs from ``D`` and ``D`` with one record removed. 
It measures empirical output stability under a specific paired perturbation. 
It is neither a complete membership-inference attack nor evidence replacing the formal differential-privacy guarantee.
"""
from __future__ import annotations

import argparse
import random
import numpy as np
from pathlib import Path
from .utils import read_jsonl, write_jsonl, load_yaml, write_json, ensure_dir
from .run_experiment import run


def remove_one(records, idx):
    """Return a copy of the record list with one index removed.
    """
    return [r for i,r in enumerate(records) if i != idx]


def attack_score(plan_a: dict, plan_b: dict, target_atom_labels: list[str]) -> float:
    """Compute Jaccard distance between two admitted-atom index sets."""
    labels_a = {a['label'] for a in plan_a.get('selected_atoms', [])}
    labels_b = {a['label'] for a in plan_b.get('selected_atoms', [])}
    return float(len(labels_a & set(target_atom_labels)) - len(labels_b & set(target_atom_labels)))


def main() -> None:
    """Command-line entry point for the paired-run stability probe.
    """
    p = argparse.ArgumentParser(description="Membership-style sensitivity probe by paired runs D and D\\{x*}.")
    p.add_argument('--config', required=True)
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--output', default='outputs/membership_attack.json')
    args = p.parse_args()
    base = load_yaml(args.config)
    records = read_jsonl(base['records'])
    outroot = ensure_dir(Path(args.output).parent / 'membership_tmp')
    results = []
    for t in range(args.trials):
        idx = random.Random(t).randrange(len(records))
        keep_path = outroot / f'D_trial{t}.jsonl'
        drop_path = outroot / f'Dminus_trial{t}.jsonl'
        write_jsonl(records, keep_path)
        write_jsonl(remove_one(records, idx), drop_path)
        cfg1 = dict(base); cfg1.update({'records': str(keep_path), 'seed': t, 'experiment_name': base.get('experiment_name','exp') + f'_mem_keep{t}'})
        cfg0 = dict(base); cfg0.update({'records': str(drop_path), 'seed': t, 'experiment_name': base.get('experiment_name','exp') + f'_mem_drop{t}'})
        r1 = run(cfg1); r0 = run(cfg0)
        import json
        p1 = json.load(open(r1/'private_plan.json'))
        p0 = json.load(open(r0/'private_plan.json'))
        # Diagnostic rule: infer membership when the admitted sets differ.
        pred_member = p1.get('selected_atoms') != p0.get('selected_atoms')
        results.append({'trial': t, 'target_index': idx, 'plans_differ': bool(pred_member)})
    write_json({'trials': results, 'plan_difference_rate': sum(r['plans_differ'] for r in results)/max(1,len(results))}, args.output)
    print(f"Wrote {args.output}")

if __name__ == '__main__':
    main()
