# Source package

The `src` package implements DP-SPIN—**differentially private semantic plans for insight narration**—and the experiment, baseline, evaluation, and reporting code used in the accompanying paper.

The primary implementation follows a trusted-curator model. The curator maps each protected contribution to a bounded sparse vector over a fixed public semantic atom universe, aggregates those vectors into an internal semantic sketch, and releases a differentially private semantic plan. The decoder receives only the released plan and public instructions. A public verifier checks whether the generated text satisfies the decoder contract.

For installation, dataset preparation, and complete experiment commands, see the repository-level [`README.md`](../README.md) and [`scripts/README.md`](../scripts/README.md).

## Privacy boundary

A private DP-SPIN run uses the following data flow:

```text
public or auxiliary data
    -> public semantic atom dictionary
    -> fixed assignment representation

protected target split
    -> bounded semantic contributions
    -> internal aggregate semantic sketch
    -> differentially private semantic-plan release
    -> contract-constrained verbalization
    -> public plan-consistency verification
```

The decoder-visible release may contain:

- admitted public atom labels and descriptions;
- released noisy masses;
- prevalence bins derived from a public denominator;
- public decoding instructions and verifier rules.

The decoder must not receive:

- raw protected records or excerpts;
- per-record semantic contribution vectors;
- the non-private aggregate semantic sketch;
- nearest neighbors or target-derived representatives;
- labels for non-admitted atoms;
- non-private reference plans or summaries.

Non-private references and target-dependent evaluation artifacts may be generated inside the trusted evaluation workflow. They are not part of the DP-SPIN release and must remain separate from decoder inputs.

## Core pipeline

| Module | Responsibility |
|---|---|
| `assign.py` | Computes bounded record-to-atom semantic contributions and the internal aggregate semantic sketch. |
| `dp_mechanisms.py` | Implements the Gaussian sketch, Laplace sketch, and sparse private atom-admission mechanisms. |
| `plan.py` | Constructs the decoder-visible semantic plan from a privatized sketch or sparse selected-value release. |
| `decode.py` | Verbalizes a released semantic plan in audit or display mode. |
| `verify.py` | Checks generated text against the public decoder contract. |
| `run_experiment.py` | Executes one complete DP-SPIN experiment and writes its artifacts. |

The main parameters are:

- `m`: number of atoms in the public semantic atom dictionary;
- `K`: maximum assignment support per protected record;
- `L`: number of atoms admitted to the released semantic plan;
- `B`: user-level contribution bound after aggregating and clipping one user's records;
- `epsilon`, `delta`: privacy parameters;
- `protected_unit`: `record` or `user`;
- `adjacency`: add/drop or replacement, as configured by the experiment.

## Data preparation and atom construction

| Module | Responsibility |
|---|---|
| `download_cfpb.py` | Downloads public CFPB complaint data. |
| `download_data.py` | Acquires or validates CFPB, Amazon Reviews, and Yelp source files. |
| `load_data.py` | Converts CSV data to the repository JSONL schema. |
| `prepare_cfpb.py` | Converts CFPB records for record-level experiments. |
| `prepare_reviews.py` | Converts Yelp, Amazon, or generic review data and preserves stable contributor identifiers when available. |
| `split_records.py` | Creates disjoint auxiliary and protected target splits, including a streaming path for large JSONL files. |
| `build_atoms.py` | Builds decoder-visible atom dictionaries from public data or the auxiliary split. |
| `embed.py` | Defines the fixed TF-IDF representation used by the semantic contribution map. |
| `dataset_stats.py` | Computes non-private dataset diagnostics and protocol statistics. |
| `generate_demo_data.py` | Generates synthetic records for installation and pipeline checks. |

Atom naming and atom admission are separate operations. `build_atoms.py` fixes decoder-visible labels independently of the protected target split. The protected target data affect the released atom set only through a differentially private release mechanism.

## Experiment drivers and diagnostics

| Module | Responsibility |
|---|---|
| `run_grid.py` | Executes the current Cartesian-grid runner and can skip completed conditions. |
| `probe_sweep.py` | Runs controlled probe-record experiments over support, privacy budget, and seed. |
| `probes.py` | Inserts controlled probe records using the current `probe` terminology. |
| `review_probe.py` | Adds controlled probe-atom runs to Amazon or Yelp experiment families. |
| `membership_attack.py` | Compares paired releases after one-record deletion as an empirical stability diagnostic. |

Controlled probe-atom and one-record-deletion experiments are empirical diagnostics. They do not replace the formal differential-privacy guarantee.

## Metrics, text evaluation, and reporting

| Module | Responsibility |
|---|---|
| `metrics.py` | Computes semantic-plan utility, verifier, and controlled-probe metrics. |
| `summary_eval.py` | Compares generated text with an offline non-private reference plan and summary. |
| `judge_summary.py` | Applies heuristic or optional LLM evaluation relative to the decoder-visible release object. |
| `human_eval_interface.py` | Renders a static manual-evaluation page. |
| `summarize_runs.py` | Aggregates repeated run directories into grouped CSV and JSON summaries. |
| `plot_results.py` | Plots plan-utility and controlled-probe results from saved metrics. |
| `report.py` | Generates dataset reports, LaTeX tables, and figures. |
| `cfpb_report_figures.py` | Generates CFPB-specific figures from saved artifacts. |
| `main_paper_figures.py` | Generates selected cross-dataset figures for the manuscript body. |
| `paper_assets.py` | Collects selected tables, figures, diagnostics, and text into a stable asset directory. |

Dense full-sketch mechanisms and sparse private atom admission expose different numeric objects. `metrics.py` therefore reports full-vector error only for dense releases. Sparse releases are evaluated on admitted coordinates, recovered support, retained non-private mass, and support-mass loss.

## Baselines

| Module | Responsibility |
|---|---|
| `baselines.py` | Generates non-private utility references and fixed-universe DP keyword and category histogram baselines. |
| `baseline_report.py` | Summarizes available baseline artifacts and their privacy boundaries. |
| `urania_baseline.py` | Implements the repository's URANIA-style clustered public-keyword comparison. |
| `urania_report.py` | Compares saved DP-SPIN and URANIA-style release and summary artifacts. |

Raw-record summarization, heuristic redaction, and target-derived clustering are non-private references. The keyword and category histogram baselines are differentially private only when the bin universe is fixed independently of the protected target split and the documented contribution bound is enforced.

The URANIA-style implementation is a controlled repository-local comparison, not an exact reproduction of the original external implementation. Saved metadata record the mechanism and implementation differences.

## OpenAI verbalization and evaluation

| Module | Responsibility |
|---|---|
| `llm_client.py` | Provides the optional OpenAI-compatible chat call used by decoders and judges. |
| `openai_decode_representatives.py` | Decodes selected completed DP-SPIN releases in audit and display modes. |
| `openai_paper_pipeline.py` | Runs cross-dataset verbalization, baseline generation, and release-conditioned evaluation from saved releases. |
| `openai_report.py` | Aggregates saved OpenAI decoding and judging results. |
| `openai_main_tables.py` | Builds selected cross-dataset CSV and LaTeX tables without calling the API. |

For a DP-SPIN release, the OpenAI decoder receives only the released semantic plan and public instructions. Non-private baseline prompts may contain target records and are therefore labeled and stored separately. The OpenAI experiment commands and environment variables are documented in [`scripts/README_openai.md`](../scripts/README_openai.md).


## Running modules

Run package modules from the repository root so that relative imports resolve correctly:

```bash
python -m src.run_experiment --config configs/demo.yaml
python -m src.run_grid --config configs/demo.yaml --epsilons 0.5 1.0 2.0
python -m src.metrics --run-dir outputs/demo/<run-directory>
```

Use the shell workflows in `scripts/` for the complete dataset-specific experiment grids and report-generation sequences.

## Documentation conventions

Docstrings distinguish four kinds of artifacts:

1. **internal non-private state**, such as the aggregate semantic sketch and per-record contribution vectors;
2. **released differentially private objects**, primarily the semantic plan;
3. **public post-processing artifacts**, including generated summaries and verifier outputs derived only from the released plan;
4. **non-private evaluation artifacts**, including raw-record baselines and offline reference plans.

A function that reads or produces a non-private evaluation artifact is not part of the decoder-visible DP release unless its documentation states otherwise.
