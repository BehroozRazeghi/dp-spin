"""Execute a Cartesian grid of DP-SPIN configurations.

The driver varies privacy budgets, seeds, mechanisms, atom counts, assignment sparsity ``K``, released plan size ``L``, protected-unit counts, and user
contribution bound ``B``. It can skip completed output directories and updates the public denominator from the configured public record or user count when the corresponding automatic option is enabled.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from .utils import load_yaml, run_name
from .run_experiment import run
from .dp_mechanisms import normalize_mechanism_name


def parse_list(vals, cast):
    """Parse command-line grid values with optional type conversion.
    """
    if vals is None:
        return None
    return [cast(v) for v in vals]




def _resolve_realized_m_for_resume(cfg: dict) -> int | None:
    """Determine the realized atom count used in an existing run name.
    
    If the atom file contains fewer than the requested ``m`` entries, the experiment
    uses the file length. The resume check mirrors that behavior. On a file-reading
    failure, it falls back to the requested value.
    """
    requested_m = cfg.get("m")
    if requested_m is None:
        return None
    try:
        atom_path = cfg.get("atoms")
        if atom_path and Path(atom_path).exists():
            with open(atom_path, "r", encoding="utf-8") as f:
                atoms = json.load(f)
            return min(int(requested_m), len(atoms))
    except Exception:
        pass
    return int(requested_m)


def _expected_run_dir(cfg: dict) -> Path:
    """Construct the output directory expected for a pending grid condition.
    
    The function mirrors the configuration normalization performed by
    :func:`src.run_experiment.run`. A condition is skipped only when the expected
    directory contains the required completion artifacts.
    """
    probe = dict(cfg)
    mech = normalize_mechanism_name(str(probe.get("mechanism", "gaussian")))
    probe["mechanism"] = mech
    probe["epsilon_configured"] = float(probe.get("epsilon", 1.0))
    probe["delta_configured"] = float(probe.get("delta", 0.0))
    probe["epsilon_effective"] = float(probe["epsilon_configured"])
    probe["delta_effective"] = 0.0 if mech == "laplace" else float(probe["delta_configured"])
    m_eff = _resolve_realized_m_for_resume(probe)
    if m_eff is not None:
        probe["m"] = int(m_eff)
    protected_unit = str(probe.get("protected_unit", "record")).lower().replace("-", "_")
    probe["protected_unit"] = protected_unit
    if protected_unit == "user" and probe.get("n_users") is not None:
        probe["n_users_effective"] = int(probe["n_users"])
        probe["n_protected_units_effective"] = int(probe["n_users"])
    elif protected_unit != "user" and probe.get("n_records") is not None:
        probe["n_records_effective"] = int(probe["n_records"])
        probe["n_protected_units_effective"] = int(probe["n_records"])
    return Path(probe.get("output_dir", "outputs/runs")) / run_name(probe)


def _is_completed_run_dir(outdir: Path) -> bool:
    """Return True only when the key artifacts for a completed run exist."""
    required = ["metrics.json", "private_plan.json", "summary.txt", "config.yaml"]
    return outdir.is_dir() and all((outdir / name).exists() for name in required)


def main() -> None:
    """Run the Cartesian product of configured grid values.
    """
    p = argparse.ArgumentParser(description="Run multi-epsilon/multi-seed ablation grid.")
    p.add_argument('--config', required=True)
    p.add_argument('--epsilons', nargs='+', type=float, default=[0.5,1,2,4,8])
    p.add_argument('--seeds', nargs='+', type=int, default=[0,1,2,3,4])
    p.add_argument('--mechanisms', nargs='+', default=['gaussian','laplace'])
    p.add_argument('--m-values', nargs='+', type=int, default=None)
    p.add_argument('--K-values', nargs='+', type=int, default=None)
    p.add_argument('--L-values', nargs='+', type=int, default=None)
    p.add_argument('--n-records-values', nargs='+', type=int, default=None)
    p.add_argument('--n-users-values', nargs='+', type=int, default=None)
    p.add_argument('--user-contribution-bounds', nargs='+', type=float, default=None, help='Grid values for user_contribution_bound B in user-level runs.')
    p.add_argument('--records', default=None, help='Override records path from the config.')
    p.add_argument('--atoms', default=None, help='Override atoms path from the config.')
    p.add_argument('--atoms-template', default=None, help='Override atoms path using a format string with {m}, e.g. data/public_atoms/atoms_m{m}.json.')
    p.add_argument('--embedding-fit-records', default=None, help='Override embedding_fit_records path from the config.')
    p.add_argument('--embedding-fit-max-records', type=int, default=None, help='Use at most this many public/auxiliary records to fit the assignment vectorizer.')
    p.add_argument('--embedding-fit-selection', choices=['hash','random','first'], default=None, help='Selection rule for bounded public/auxiliary vectorizer fitting.')
    p.add_argument('--embedding-fit-seed', type=int, default=None, help='Seed for bounded public/auxiliary vectorizer fitting.')
    p.add_argument('--output-dir', default=None, help='Override output_dir from the config.')
    p.add_argument('--summary-embedding-backend', default=None, choices=['tfidf','auto','sentence_transformer'], help='Text-level evaluation embedding backend.')
    p.add_argument('--summary-comparative-eval-backend', default=None, choices=['none','heuristic','openai'], help='Optional DP-summary vs non-private-reference-summary evaluator.')
    p.add_argument('--decoder-backend', default=None, choices=['template','openai'], help='Override decoder backend for generated DP summaries.')
    p.add_argument('--decoder-model', default=None, help='Override decoder model for LLM backends.')
    p.add_argument('--decoder-temperature', default=None, type=float, help='Override decoder temperature.')
    p.add_argument('--clean-reference-decoder-backend', default=None, choices=['template','openai'], help='Override decoder backend for non-private-reference summaries used in offline evaluation.')
    p.add_argument('--clean-reference-decoder-model', default=None, help='Override non-private-reference decoder model.')
    p.add_argument('--skip-existing', action='store_true', help='Resume a grid by skipping runs whose key output artifacts already exist.')
    args = p.parse_args()
    base = load_yaml(args.config)
    m_values = args.m_values or [base.get('m')]
    K_values = args.K_values or [base.get('K',3)]
    L_values = args.L_values or [base.get('L',10)]
    n_values = args.n_records_values or [base.get('n_records')]
    u_values = args.n_users_values or [base.get('n_users')]
    b_values = args.user_contribution_bounds or [base.get('user_contribution_bound')]
    total = 0
    skipped = 0
    for eps, seed, mech, m, K, L, n, u, B in itertools.product(args.epsilons, args.seeds, args.mechanisms, m_values, K_values, L_values, n_values, u_values, b_values):
        cfg = dict(base)
        cfg.update({'epsilon': eps, 'seed': seed, 'mechanism': normalize_mechanism_name(mech), 'K': K, 'L': L})
        if args.records is not None:
            cfg['records'] = args.records
        if args.atoms_template is not None:
            try:
                cfg['atoms'] = args.atoms_template.format(m=m)
            except Exception as exc:
                raise ValueError(
                    "--atoms-template must be a Python format string containing {m}, "
                    "for example data/public_atoms/cfpb_atoms_m{m}.json"
                ) from exc
            if not Path(cfg['atoms']).exists():
                raise FileNotFoundError(
                    f"Atom file from --atoms-template does not exist for m={m}: {cfg['atoms']}"
                )
        elif args.atoms is not None:
            cfg['atoms'] = args.atoms
        if args.embedding_fit_records is not None:
            cfg['embedding_fit_records'] = args.embedding_fit_records
        if args.embedding_fit_max_records is not None:
            cfg['embedding_fit_max_records'] = int(args.embedding_fit_max_records)
        if args.embedding_fit_selection is not None:
            cfg['embedding_fit_selection'] = args.embedding_fit_selection
        if args.embedding_fit_seed is not None:
            cfg['embedding_fit_seed'] = int(args.embedding_fit_seed)
        if args.output_dir is not None:
            cfg['output_dir'] = args.output_dir
        if args.summary_embedding_backend is not None:
            cfg['summary_embedding_backend'] = args.summary_embedding_backend
        if args.summary_comparative_eval_backend is not None:
            cfg['summary_comparative_eval_backend'] = args.summary_comparative_eval_backend
        if args.decoder_backend is not None:
            cfg['decoder_backend'] = args.decoder_backend
        if args.decoder_model is not None:
            cfg['decoder_model'] = args.decoder_model
        if args.decoder_temperature is not None:
            cfg['decoder_temperature'] = float(args.decoder_temperature)
        if args.clean_reference_decoder_backend is not None:
            cfg['clean_reference_decoder_backend'] = args.clean_reference_decoder_backend
        if args.clean_reference_decoder_model is not None:
            cfg['clean_reference_decoder_model'] = args.clean_reference_decoder_model
        if m is not None:
            cfg['m'] = m
        if cfg.get('protected_unit') == 'user':
            if B is not None:
                cfg['user_contribution_bound'] = float(B)
            # User-level DP selects whole users and includes all records from each
            # selected user.  Keeping n_records here would allow accidental
            # record-level truncation and could split a protected contribution.
            cfg.pop('n_records', None)
            if u is not None:
                cfg['n_users'] = u
                if cfg.get('auto_public_denominator_from_n_users', True):
                    cfg['public_denominator'] = int(u)
        else:
            if n is not None:
                cfg['n_records'] = n
                # In these experiments the number of target records used is a public
                # protocol parameter. Keep prevalence denominators synchronized with
                # the n_records grid unless the config explicitly disables it.
                if cfg.get('auto_public_denominator_from_n_records', True):
                    cfg['public_denominator'] = int(n)
        if args.skip_existing:
            expected_outdir = _expected_run_dir(cfg)
            if _is_completed_run_dir(expected_outdir):
                print(f"Skipping completed run: {expected_outdir}")
                skipped += 1
                continue
        run(cfg)
        total += 1
    print(f"Completed {total} grid runs; skipped {skipped} existing runs.")

if __name__ == '__main__':
    main()
