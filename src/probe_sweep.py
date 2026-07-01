"""Run controlled probe-record experiments over support and privacy settings.

For each condition, the runner appends a known probe record pattern to the protected target corpus, executes DP-SPIN, and records probe-atom admission and
summary-mention outcomes. These are empirical diagnostics and do not replace the formal differential-privacy guarantee.

Each generated corpus contains ``base_size + support`` records. Unless a fixed base denominator is requested, ``n_records`` and ``public_denominator`` are set
to that realized corpus size.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .probes import insert_probes, DEFAULT_PROBES
from .utils import load_yaml, save_yaml, ensure_dir
from .run_experiment import run
from .dp_mechanisms import normalize_mechanism_name


def _count_jsonl(path: str | Path) -> int:
    """Count non-empty JSONL records without loading the file into memory."""
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _resolve_probe_denominator(base: dict[str, Any], actual_n: int, base_n: int, support: int) -> int:
    """Resolve the public denominator used by a generated probe run.

    The default policy is ``actual_generated``: prevalence bins are defined
    relative to the generated corpus size ``base_n + support``. This is the
    least ambiguous policy for probe diagnostics. A fixed base denominator is
    still available for backward-compatible diagnostic runs by setting
    ``probe_public_denominator_mode: fixed_base`` in the base config.
    """
    mode = str(base.get("probe_public_denominator_mode", "actual_generated"))
    if mode in {"actual_generated", "actual", "generated"}:
        return int(actual_n)
    if mode in {"fixed_base", "base"}:
        return int(base.get("probe_base_size", base_n))
    if mode in {"base_plus_support", "formula"}:
        return int(base_n + support)
    raise ValueError(
        "probe_public_denominator_mode must be one of: "
        "actual_generated, fixed_base, base_plus_support"
    )


def main() -> None:
    """Run a probe admission sweep over supports, epsilons, and seeds."""
    p = argparse.ArgumentParser(description="Run probe insertion sweeps.")
    p.add_argument('--config', required=True)
    p.add_argument('--supports', nargs='+', type=int, default=[1, 2, 5, 10, 25, 50])
    p.add_argument('--epsilons', nargs='+', type=float, default=None)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--mechanisms', nargs='+', default=None, help='Release mechanisms to sweep. Defaults to the mechanism in the base config.')
    p.add_argument('--probe-text', default=DEFAULT_PROBES[0])
    p.add_argument('--work-dir', default='outputs/probe_data')
    args = p.parse_args()

    base = load_yaml(args.config)
    work = ensure_dir(args.work_dir)
    epsilons = args.epsilons or [float(base.get('epsilon', 1.0))]
    mechanisms = [normalize_mechanism_name(m) for m in (args.mechanisms or [base.get('mechanism', 'gaussian')])]
    base_n = _count_jsonl(base['records'])

    for support in args.supports:
        if support < 0:
            raise ValueError("Probe support $s$ must be nonnegative")
        for seed in args.seeds:
            probe_records = work / f"records_probe_s{support}_seed{seed}.jsonl"
            insert_probes(
                base['records'],
                str(probe_records),
                support=support,
                seed=seed,
                probe_text=args.probe_text,
            )
            actual_n = _count_jsonl(probe_records)
            expected_n = base_n + support
            if actual_n != expected_n:
                raise RuntimeError(
                    f"Generated probe corpus has {actual_n} records, but expected "
                    f"base_n + support = {base_n} + {support} = {expected_n}."
                )

            public_denominator = _resolve_probe_denominator(base, actual_n, base_n, support)

            for mechanism in mechanisms:
                for eps in epsilons:
                    cfg = dict(base)
                    cfg['records'] = str(probe_records)
                    cfg['mechanism'] = normalize_mechanism_name(mechanism)
                    cfg['epsilon'] = float(eps)
                    cfg['seed'] = int(seed)
                    cfg['probe_strings'] = [args.probe_text]
                    cfg['probe_support'] = int(support)
                    cfg['probe_base_records'] = int(base_n)
                    cfg['probe_generated_records'] = int(actual_n)
                    cfg['n_records'] = int(actual_n)
                    cfg['public_denominator'] = int(public_denominator)
                    cfg['strict_sample_size'] = True
                    cfg['record_selection'] = str(cfg.get('record_selection', 'hash'))
                    cfg['experiment_name'] = f"{base.get('experiment_name', 'exp')}_probe_support{support}_{cfg['mechanism']}"

                    # Save the exact per-run config next to the generated data for
                    # auditability. The DP run itself will also write config.yaml in
                    # its output directory.
                    eps_tag = str(eps).replace('.', '_')
                    cfg_path = work / f"config_probe_s{support}_{cfg['mechanism']}_seed{seed}_eps{eps_tag}.yaml"
                    save_yaml(cfg, cfg_path)
                    run(cfg)

    print("Controlled probe sweep complete.")


if __name__ == '__main__':
    main()
