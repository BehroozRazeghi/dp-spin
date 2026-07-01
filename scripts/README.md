# Scripts

Shell entry points for dataset preparation, public-atom construction, record-level and user-level DP-SPIN experiments, controlled-probe runs, OpenAI verbalization, and report generation are kept here.

Run all commands from the repository root:

```bash
bash scripts/<script-name>.sh
```

List-valued environment variables are space-delimited unless noted otherwise. Each script uses `set -euo pipefail`, so unset variables and failed commands terminate the run.

OpenAI inputs, controls, outputs, and report-only execution are documented in [`README_openai.md`](README_openai.md).

## Terminology

The names below match the manuscript and the command-line interfaces.

- The **protected unit** is the unit changed by the neighboring-dataset relation: one record for record-level privacy or one user for user-level privacy. User-level runs clip each user's aggregate semantic contribution before release.
- The **auxiliary split** is disjoint from the protected target split. It is used to construct the atom dictionary and fit the text representation used by the semantic contribution map.
- The **protected target split** is the corpus to which the differentially private plan-release mechanism is applied.
- `m` is the size of the public semantic atom universe used by a run.
- `K` is the maximum number of nonzero atom weights in each record-level semantic assignment.
- `L` is the number of atoms admitted to the released semantic plan.
- `B` is the deterministic user-contribution clipping bound used before user-level release.
- `epsilon` and `delta` are the differential-privacy parameters of the configured release.

Mechanism identifiers:

| Identifier | Implemented plan-release mechanism |
|---|---|
| `gaussian` | Gaussian perturbation of the full semantic sketch, followed by top-`L` selection as post-processing |
| `laplace` | Laplace perturbation of the full semantic sketch, followed by top-`L` selection as post-processing |
| `private_topL` | Sequential exponential-mechanism admission of `L` atoms, followed by differentially private perturbation of their selected masses |

`private_topL` is the implementation name for **sparse private atom admission**. `K` denotes assignment sparsity; `L` denotes the released plan size. New commands should use this spelling.

## Script index

| Script | Role | Runs a DP mechanism | May call OpenAI |
|---|---|---:|---:|
| `run_cfpb.sh` | CFPB record-level dataset runner | Yes | Only with `USE_OPENAI=1` |
| `run_amazon.sh` | Amazon record-level and user-level dataset runner | Yes | Only when an OpenAI decoder or atom-refinement backend is selected |
| `run_yelp.sh` | Yelp record-level and user-level dataset runner | Yes | Only when an OpenAI decoder or atom-refinement backend is selected |
| `run_demo.sh` | Synthetic demonstration | Yes | No |
| `run_amazon_missing_KL_only.sh` | Append missing Amazon `K`-by-`L` operating points | Yes | No |
| `run_review_probe_only.sh` | Append Amazon and/or Yelp controlled-probe runs | Yes | No |
| `run_openai_paper.sh` | OpenAI verbalization, baselines, release-conditioned evaluation, and manuscript tables | No | Yes, unless `REPORT_ONLY=1` |
| `run_openai_dpspin_only.sh` | OpenAI verbalization of selected saved DP-SPIN plans | No | Yes |
| `_run_openai_paper_driver.sh` | Internal argument-mapping driver | No | Indirectly |
| `build_openai_main_tables.sh` | Rebuild cross-dataset OpenAI tables from saved reports | No | No |
| `build_main_paper_figures.sh` | Build compact cross-dataset figures from saved reports | No | No |
| `refresh_cfpb_report_assets.sh` | Rebuild CFPB reports, organized figures, and manuscript assets | No | No |
| `refresh_review_report_assets.sh` | Rebuild Amazon and Yelp grouped summaries, tables, and figures | No | No |
| `refresh_paper_assets.sh` | Recollect CFPB report tables, URANIA-style assets, and the paper-assets manifest | No | No |

## Environment

Use Python 3.10 or later in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Workflow-specific requirements:

- Yelp download through Kaggle requires the Kaggle CLI, configured credentials, and prior acceptance of the applicable dataset terms.
- OpenAI verbalization, OpenAI judging, or OpenAI-assisted auxiliary atom refinement requires `OPENAI_API_KEY`.
- `summary_embedding_backend=sentence_transformer` or `auto` may require `sentence-transformers`; the default `tfidf` backend does not.

## Dataset runners

### CFPB consumer complaint narratives

```bash
bash scripts/run_cfpb.sh
```

`run_cfpb.sh`:

1. validates the Python environment and required imports;
2. downloads a 20,000-row CFPB CSV subset when `data/raw/cfpb_complaints.csv` is absent;
3. prepares complaint records and creates a deterministic 50/50 split into auxiliary and protected target corpora;
4. constructs separate metadata-based atom dictionaries of sizes `m in {50, 100, 200}` from the auxiliary split using the common schema `Product / Issue / Sub-issue`;
5. runs the main, lower-signal, and ablation grids on the protected target split;
6. runs controlled-probe sweeps with a known public probe atom;
7. runs DP histogram baselines and a URANIA-style public-keyword baseline;
8. runs the paired membership-style stability diagnostic;
9. optionally runs an integrated OpenAI decoding and judging subset; and
10. creates organized figures, tables, diagnostics, and manuscript assets.

CFPB runs are record-level only because the public export does not provide a stable identifier
for grouping complaints by consumer.

#### Default CFPB grids (`FULL=0`)

| Block | Protected target size | `m` | `K` | `L` | `epsilon` | Seeds |
|---|---:|---|---|---|---|---|
| Main | 2,000 records | `50 100 200` | `3` | `10` | `0.5 1 2 4 8` | `0 1 2` |
| Lower-signal setting | 1,000 records | `200` | `3` | `5 10` | `0.25 0.5 1 2` | `0 1 2` |
| Ablation | 1,000 records | `50 100 200` | `1 3 5` | `5 10` | `0.5 1 2` | `0 1 2` |
| Controlled probe | 2,000 base records | base `m = 50, 200` plus one public probe atom | `3` | `10` | `0.5 1 2 4` | `0 1 2` |

The default run includes all three mechanisms. Controlled-probe supports are `1 2 5 10 25 50 100 200 500`.

`FULL=1` uses seeds `0 1 2 3 4`, expands the grids, adds `L=20` to the ablation, extends the support set, and includes `epsilon=8` in the probe sweep.

#### CFPB controls

| Variable | Default | Meaning |
|---|---|---|
| `FULL` | `0` | Select the moderate (`0`) or larger (`1`) grid |
| `REUSE_PREP` | `1` | Reuse prepared records, splits, and atom dictionaries when valid |
| `FORCE_PREP` | `0` | Force regeneration of prepared records, splits, and atom dictionaries |
| `USE_OPENAI` | `0` | Run the integrated CFPB OpenAI subset after the local experiments |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model used only when `USE_OPENAI=1` |
| `URANIA_EPS_TOTAL` | `1.0` | Total privacy budget of the URANIA-style baseline |
| `URANIA_DELTA` | `1e-6` | Delta of the URANIA-style baseline |
| `URANIA_CLUSTERS` | `10` | Requested number of private clusters |
| `URANIA_KEYWORDS` | `10` | Number of released public keywords per summary |
| `URANIA_MAX_KEYWORDS` | `200` | Maximum public keyword-vocabulary size |
| `URANIA_KEYWORDS_PER_RECORD` | `5` | Per-record keyword contribution bound |
| `URANIA_SVD_DIM` | `64` | SVD dimension used by the baseline |
| `URANIA_DP_ITERATIONS` | `3` | Number of DP clustering iterations |
| `URANIA_RUN_PROBE` | `1` | Run the URANIA-style controlled-probe diagnostic |

Example:

```bash
FULL=1 bash scripts/run_cfpb.sh
```

Outputs:

```text
outputs/cfpb/
```

`outputs/cfpb/README_RESULTS.txt` indexes the generated artifacts. Manuscript-ready files are collected under:

```text
outputs/cfpb/paper_assets/figures/
outputs/cfpb/paper_assets/tables/
outputs/cfpb/paper_assets/raw_tables/
outputs/cfpb/paper_assets/MANIFEST.md
```

`USE_OPENAI=1` is retained for compatibility. Use `run_openai_paper.sh` for the maintained cross-dataset OpenAI workflow.

### Amazon Reviews 2023

```bash
bash scripts/run_amazon.sh
```

The default category is `All_Beauty`. With `DOWNLOAD_AMAZON=auto`, a missing raw file is downloaded from the configured McAuley Lab source. The runner prepares records, creates a user-disjoint auxiliary and protected target splits, constructs review-theme atoms from the auxiliary split, executes record-level and user-level grids, runs the controlled-probe conditions, and rebuilds the grouped reports.

Under the default All Beauty filters and `AMAZON_MIN_DF=25`, 46 atoms remain. Requested-cap files for 50, 100, and 200 atoms may therefore contain the same 46 atoms. The grid uses `m=46` unless `AMAZON_M_VALUES` is set explicitly.

#### Default Amazon grids (`FULL=0`)

| Setting | Values |
|---|---|
| Seeds | `0 1 2` |
| Privacy budgets | `0.5 1 2 4` |
| Record counts | `2000 5000` |
| User counts | `1000 2000` |
| User clipping bounds | `1 2` |
| Atom-universe size | `46` |
| Assignment sparsity | `1 3` |
| Released plan size | `5 10` |
| Mechanisms | `gaussian laplace private_topL` |

`FULL=1` uses five seeds, adds `epsilon=8`, record count 10,000, user count 5,000, user clipping bound 5, `K=5`, and `L=20`.

#### Amazon controls

Execution and data preparation:

| Variable | Default | Meaning |
|---|---|---|
| `FULL` | `0` | Select the moderate or larger grid |
| `REUSE_PREP` | `1` | Reuse prepared records, split files, and atom files |
| `FORCE_PREP` | `0` | Rebuild preparation artifacts |
| `RUN_RECORD` | `1` | Run record-level experiments |
| `RUN_USER` | `1` | Run user-level experiments |
| `CLEAN_RUNS` | `1` | Delete the selected main-grid run directories before execution |
| `SKIP_EXISTING` | `0` | Pass `--skip-existing` to grid and probe runners |
| `RUN_PROBE` | `1` | Run the controlled-probe experiment |
| `AMAZON_CATEGORY` | `All_Beauty` | Amazon Reviews 2023 category |
| `AMAZON_REVIEWS` | `data/raw/amazon_<category>.jsonl.gz` | Raw review file |
| `DOWNLOAD_AMAZON` | `auto` | `auto`, `1`, or `0` |
| `AMAZON_DOWNLOAD_SOURCE` | `official` | Download source passed to `src.download_data` |
| `AMAZON_USER_FIELD` | empty | Optional raw user-field override |
| `AMAZON_TEXT_FIELD` | empty | Optional raw text-field override |
| `RAW_MAX_ROWS` | empty | Optional preparation limit |

Atom construction:

| Variable | Default | Meaning |
|---|---|---|
| `AMAZON_ATOM_CAPS` | `46 50 100 200` | Requested maximum atom counts to build |
| `AMAZON_NGRAM_MIN` | `2` | Minimum phrase length |
| `AMAZON_NGRAM_MAX` | `2` | Maximum phrase length |
| `AMAZON_MIN_DF` | `25` | Minimum auxiliary document support |
| `AMAZON_MAX_DF` | `1.0` | Maximum auxiliary document-frequency fraction |
| `AMAZON_ATOM_REFINEMENT` | `deduplicate` | `none`, `deduplicate`, or `llm` |
| `AMAZON_ATOM_REFINE_BACKEND` | `template` | `template` or `openai` |
| `AMAZON_ATOM_REFINE_MODEL` | `gpt-4o-mini` | Model used only for OpenAI-assisted auxiliary atom refinement |
| `AMAZON_ATOM_REFINE_TOP_CANDIDATES` | `1000` | Maximum auxiliary candidates passed to refinement |

Grid overrides:

| Variable | Default source | Meaning |
|---|---|---|
| `AMAZON_SEEDS` | selected by `FULL` | Seed list |
| `AMAZON_EPS` | selected by `FULL` | Privacy-budget list |
| `AMAZON_MECHS` | `gaussian laplace private_topL` | Mechanism list |
| `AMAZON_N_RECORDS` | selected by `FULL` | Record-count list |
| `AMAZON_N_USERS` | selected by `FULL` | User-count list |
| `AMAZON_USER_BOUNDS` | selected by `FULL` | User clipping-bound list |
| `AMAZON_M_VALUES` | `46` | Atom-universe sizes used by the grid |
| `AMAZON_K_VALUES` | selected by `FULL` | Assignment-sparsity list |
| `AMAZON_L_VALUES` | selected by `FULL` | Released-plan-size list |

Decoder and secondary text evaluation:

| Variable | Default | Meaning |
|---|---|---|
| `DECODER_BACKEND` | `template` | Decoder used for main-grid runs |
| `DECODER_MODEL` | `gpt-4o-mini` | Model used when the decoder backend is OpenAI |
| `DECODER_TEMPERATURE` | `0.0` | Decoder temperature |
| `CLEAN_REFERENCE_DECODER_BACKEND` | `template` | Decoder for the non-private plan reference |
| `CLEAN_REFERENCE_DECODER_MODEL` | `DECODER_MODEL` | Model for the non-private plan reference |
| `SUMMARY_COMPARATIVE_EVAL_BACKEND` | `none` | `none`, `heuristic`, or `openai` |
| `SUMMARY_EMBEDDING_BACKEND` | `tfidf` | `tfidf`, `auto`, or `sentence_transformer` |
| `AMAZON_EMBEDDING_FIT_MAX_RECORDS` | empty | Optional auxiliary fit-set limit |

Controlled-probe settings:

| Variable | Default |
|---|---|
| `AMAZON_PROBE_M` | `46` |
| `AMAZON_PROBE_SUPPORTS` | `1 2 5 10 25 50 100 200` |
| `AMAZON_PROBE_EPS` | `AMAZON_EPS` or `0.5 1 2 4` |
| `AMAZON_PROBE_SEEDS` | `AMAZON_SEEDS` or `0 1 2` |
| `AMAZON_PROBE_MECHS` | `AMAZON_MECHS` or all three mechanisms |
| `AMAZON_PROBE_K` | `3` |
| `AMAZON_PROBE_L` | `10` |
| `AMAZON_PROBE_BASE_RECORDS` | `2000` |
| `AMAZON_PROBE_BASE_USERS` | `1000` |
| `AMAZON_PROBE_USER_B` | `2` |
| `AMAZON_PROBE_TEXT` | `The review contains the synthetic probe phrase BLUE-LANTERN-791.` |

Examples:

```bash
AMAZON_CATEGORY=Digital_Music bash scripts/run_amazon.sh

RUN_USER=0 AMAZON_K_VALUES="3" AMAZON_L_VALUES="10" \
  bash scripts/run_amazon.sh

CLEAN_RUNS=0 SKIP_EXISTING=1 RUN_PROBE=0 \
AMAZON_M_VALUES="46" AMAZON_EPS="1" \
  bash scripts/run_amazon.sh
```

Outputs are written under:

```text
outputs/amazon_<category>/
```

### Yelp Open Dataset

```bash
bash scripts/run_yelp.sh
```

By default, the script expects:

```text
data/raw/yelp/yelp_academic_dataset_review.json
data/raw/yelp/yelp_academic_dataset_business.json
```

The default `YELP_CATEGORY_FILTER=Restaurants` joins reviews to the business file and retains businesses whose categories include `Restaurants`. An empty filter uses all business categories. When the filter is nonempty, the business file is required.

The runner prepares the selected review corpus, creates a deterministic user-disjoint auxiliary/target split, constructs review-theme atoms from the auxiliary split, executes record-level and user-level grids, runs the controlled-probe conditions, and rebuilds the grouped reports.

#### Default Yelp grids (`FULL=0`)

| Setting | Values |
|---|---|
| Seeds | `0 1 2` |
| Privacy budgets | `0.5 1 2 4` |
| Record counts | `2000` |
| User counts | `1000 2000` |
| User clipping bounds | `1 2` |
| Atom-universe sizes | `50 100` |
| Assignment sparsity | `1 3` |
| Released plan size | `5 10` |
| Mechanisms | `gaussian laplace private_topL` |

`FULL=1` uses five seeds, adds `epsilon=8`, record counts 5,000 and 10,000, user counts 2,000 and 5,000, user clipping bound 5, `K=5`, and `L=20`.

#### Yelp controls

Execution and data preparation:

| Variable | Default | Meaning |
|---|---|---|
| `FULL` | `0` | Select the moderate or larger grid |
| `REUSE_PREP` | `1` | Reuse prepared records, split files, and atom files |
| `FORCE_PREP` | `0` | Rebuild preparation artifacts |
| `RUN_RECORD` | `1` | Run record-level experiments |
| `RUN_USER` | `1` | Run user-level experiments |
| `CLEAN_RUNS` | `1` | Delete the selected main-grid run directories before execution |
| `SKIP_EXISTING` | `0` | Pass `--skip-existing` to grid and probe runners |
| `RUN_PROBE` | `1` | Run the controlled-probe experiment |
| `SKIP_FULL_DATASET_STATS` | `1` | Skip statistics for the full processed corpus to reduce memory use |
| `YELP_ONE_RUN_PER_PROCESS` | `1` | Execute each grid point in a fresh Python process |
| `YELP_STREAM_SPLIT` | `1` | Stream the user-disjoint auxiliary and protected target split |
| `DOWNLOAD_YELP` | `0` | Enable download through `src.download_data` |
| `YELP_SOURCE` | `manual` | `manual` or `kaggle` |
| `YELP_RAW_DIR` | `data/raw/yelp` | Raw-data directory |
| `YELP_REVIEWS` | `<raw-dir>/yelp_academic_dataset_review.json` | Review JSON path |
| `YELP_BUSINESSES` | `<raw-dir>/yelp_academic_dataset_business.json` | Business JSON path |
| `YELP_CATEGORY_FILTER` | `Restaurants` | Business-category filter; empty means all categories |
| `RAW_MAX_ROWS` | empty | Optional preparation limit |

Atom construction:

| Variable | Default | Meaning |
|---|---|---|
| `YELP_NGRAM_MIN` | `2` | Minimum phrase length |
| `YELP_NGRAM_MAX` | `2` | Maximum phrase length |
| `YELP_MIN_DF` | `2` | Minimum auxiliary document support |
| `YELP_MAX_DF` | `1.0` | Maximum auxiliary document-frequency fraction |
| `YELP_ATOM_REFINEMENT` | `deduplicate` | `none`, `deduplicate`, or `llm` |
| `YELP_ATOM_REFINE_BACKEND` | `template` | `template` or `openai` |
| `YELP_ATOM_REFINE_MODEL` | `gpt-4o-mini` | Model used only for OpenAI-assisted auxiliary atom refinement |
| `YELP_ATOM_REFINE_TOP_CANDIDATES` | `1000` | Maximum auxiliary candidates passed to refinement |

Grid overrides:

| Variable | Default source | Meaning |
|---|---|---|
| `YELP_SEEDS` | selected by `FULL` | Seed list |
| `YELP_EPS` | selected by `FULL` | Privacy-budget list |
| `YELP_MECHS` | `gaussian laplace private_topL` | Mechanism list |
| `YELP_N_RECORDS` | selected by `FULL` | Record-count list |
| `YELP_N_USERS` | selected by `FULL` | User-count list |
| `YELP_USER_BOUNDS` | selected by `FULL` | User clipping-bound list |
| `YELP_M_VALUES` | `50 100` | Atom-universe-size list |
| `YELP_K_VALUES` | selected by `FULL` | Assignment-sparsity list |
| `YELP_L_VALUES` | selected by `FULL` | Released-plan-size list |

Decoder, text evaluation, and auxiliary representation fitting:

| Variable | Default | Meaning |
|---|---|---|
| `DECODER_BACKEND` | `template` | Decoder used for main-grid runs |
| `DECODER_MODEL` | `gpt-4o-mini` | Model used when the decoder backend is OpenAI |
| `DECODER_TEMPERATURE` | `0.0` | Decoder temperature |
| `CLEAN_REFERENCE_DECODER_BACKEND` | `template` | Decoder for the non-private plan reference |
| `CLEAN_REFERENCE_DECODER_MODEL` | `DECODER_MODEL` | Model for the non-private plan reference |
| `SUMMARY_COMPARATIVE_EVAL_BACKEND` | `none` | `none`, `heuristic`, or `openai` |
| `SUMMARY_EMBEDDING_BACKEND` | `tfidf` | `tfidf`, `auto`, or `sentence_transformer` |
| `YELP_EMBEDDING_FIT_MAX_RECORDS` | `50000` | Maximum auxiliary records used to fit the text representation; empty means all |
| `YELP_EMBEDDING_FIT_SELECTION` | `hash` | Auxiliary fit-subset selection rule |
| `YELP_EMBEDDING_FIT_SEED` | `1729` | Auxiliary fit-subset seed |

Controlled-probe settings:

| Variable | Default |
|---|---|
| `YELP_PROBE_M` | `100` |
| `YELP_PROBE_SUPPORTS` | `1 2 5 10 25 50 100 200` |
| `YELP_PROBE_EPS` | `YELP_EPS` or `0.5 1 2 4` |
| `YELP_PROBE_SEEDS` | `YELP_SEEDS` or `0 1 2` |
| `YELP_PROBE_MECHS` | `YELP_MECHS` or all three mechanisms |
| `YELP_PROBE_K` | `3` |
| `YELP_PROBE_L` | `10` |
| `YELP_PROBE_BASE_RECORDS` | `2000` |
| `YELP_PROBE_BASE_USERS` | `1000` |
| `YELP_PROBE_USER_B` | `2` |
| `YELP_PROBE_TEXT` | `The review contains the synthetic probe phrase BLUE-LANTERN-791.` |

Examples:

```bash
DOWNLOAD_YELP=1 YELP_SOURCE=kaggle bash scripts/run_yelp.sh

YELP_CATEGORY_FILTER="" RUN_USER=0 \
  bash scripts/run_yelp.sh

CLEAN_RUNS=0 SKIP_EXISTING=1 RUN_PROBE=0 \
YELP_M_VALUES="100" YELP_K_VALUES="3" YELP_L_VALUES="10" \
  bash scripts/run_yelp.sh
```

For the default category filter, outputs are written under:

```text
outputs/yelp_restaurants/
```

An empty category filter uses `outputs/yelp_all/`.

## Targeted and continuation workflows

### Append missing Amazon `K`-by-`L` points

```bash
bash scripts/run_amazon_missing_KL_only.sh
```

The wrapper calls `run_amazon.sh` with:

```text
CLEAN_RUNS=0
SKIP_EXISTING=1
RUN_PROBE=0
```

Its defaults append record-level and user-level runs for `m=46`, `K in {1,3}`, `L in {5,10}`,
`epsilon=1`, seeds `0 1 2`, 10,000 records, 5,000 users, and `B=1`. Completed operating points
are preserved and skipped.

Any wrapper default can be overridden at invocation time:

```bash
RUN_USER=0 AMAZON_EPS="0.5 1" AMAZON_N_RECORDS="5000" \
  bash scripts/run_amazon_missing_KL_only.sh
```

### Append controlled-probe runs

```bash
bash scripts/run_review_probe_only.sh
```

The wrapper appends Amazon and/or Yelp controlled-probe runs to existing prepared data and atom dictionaries. It does not prepare raw data or construct the base dictionaries. Each selected dataset must already provide:

```text
data/protected_target/<dataset>_target_records.jsonl
data/auxiliary/<dataset>_atom_auxiliary_records.jsonl
data/public_atoms/<dataset>_atoms_m<m>.json
```

Principal controls:

| Variable | Default |
|---|---|
| `RUN_AMAZON_PROBE` | `1` |
| `RUN_YELP_PROBE` | `1` |
| `RUN_RECORD` | `1` |
| `RUN_USER` | `1` |
| `SKIP_EXISTING` | `1` |
| `PROBE_SUPPORTS` | `1 2 5 10 25 50 100 200` |
| `PROBE_EPS` | `0.5 1 2 4` |
| `PROBE_SEEDS` | `0 1 2` |
| `PROBE_MECHS` | `gaussian laplace private_topL` |
| `PROBE_K` | `3` |
| `PROBE_L` | `10` |
| `PROBE_TEXT` | `The review contains the synthetic probe phrase BLUE-LANTERN-791.` |
| `AMAZON_CATEGORY` | `All_Beauty` |
| `AMAZON_PROBE_M` | `46` |
| `AMAZON_PROBE_BASE_RECORDS` | `2000` |
| `AMAZON_PROBE_BASE_USERS` | `1000` |
| `AMAZON_PROBE_USER_B` | `2` |
| `YELP_CATEGORY_FILTER` | `Restaurants` |
| `YELP_PROBE_M` | `100` |
| `YELP_PROBE_BASE_RECORDS` | `2000` |
| `YELP_PROBE_BASE_USERS` | `1000` |
| `YELP_PROBE_USER_B` | `2` |
| `YELP_EMBEDDING_FIT_MAX_RECORDS` | `50000` |

The primary Amazon and Yelp runners already execute these conditions when `RUN_PROBE=1`. Use this wrapper only to append or repair probe runs without rerunning the main grids.

## Synthetic demonstration

`run_demo.sh` generates synthetic records and atoms, runs record-level and user-level grids, executes a small controlled-probe sweep, computes local baselines, and builds plots and reports:

```bash
bash scripts/run_demo.sh
```

All generated artifacts are written under `outputs/demo/` or `outputs/baselines/demo/`. The script does not require an external dataset or API key.

## OpenAI workflows

The OpenAI scripts read completed DP-SPIN run directories and do not rerun a plan-release mechanism:

```bash
bash scripts/run_openai_paper.sh
bash scripts/run_openai_dpspin_only.sh
bash scripts/build_openai_main_tables.sh
```

See [`README_openai.md`](README_openai.md) for the decoder boundary, API-visible inputs, defaults, controls, output layout, and interpretation of release-conditioned scores.

`_run_openai_paper_driver.sh` only maps environment variables to CLI arguments. Invoke
`run_openai_paper.sh` instead.

## Report and manuscript-asset workflows

### Rebuild CFPB reports and organized assets

```bash
bash scripts/refresh_cfpb_report_assets.sh
```

The maintained CFPB refresh command reads saved runs, rebuilds report tables and `metrics_summary.csv`, regenerates the organized figures, removes legacy figure names, and recollects manuscript assets. It does not rerun mechanisms, rebuild dictionaries, decode summaries, call OpenAI, or regenerate baselines.

Controls:

| Variable | Default |
|---|---|
| `OUT` | `outputs/cfpb` |
| `CFPB_FIG_EPSILON` | `0.5` |
| `CFPB_FIG_K` | `3` |
| `CFPB_FIG_L` | `10` |
| `CFPB_FIG_M` | `200` |

### Recollect CFPB paper assets

```bash
bash scripts/refresh_paper_assets.sh
```

The narrower compatibility command rebuilds report tables from the saved `main`, `hard`, `ablation`, and `probe` runs, regenerates URANIA-style assets when the required inputs exist, and reruns `src.paper_assets`. It does not rebuild `metrics_summary.csv` with `src.plot_results` or regenerate the organized CFPB figures with `src.cfpb_report_figures`.

Controls:

| Variable | Default |
|---|---|
| `OUT` | `outputs/cfpb` |
| `PROBE_TEXT` | `BLUE-LANTERN-791` |

### Rebuild Amazon and Yelp reports

```bash
bash scripts/refresh_review_report_assets.sh
```

`refresh_review_report_assets.sh` reads saved Amazon and Yelp runs, rebuilds the record-level and user-level grouped summaries, and reruns `src.report` for both datasets.

Controls:

| Variable | Default |
|---|---|
| `AMAZON_CATEGORY` | `All_Beauty` |
| `AMAZON_OUT` | `outputs/amazon_<category>` |
| `YELP_CATEGORY_FILTER` | `Restaurants` |
| `YELP_OUT` | `outputs/yelp_<filter-slug>` or `outputs/yelp_all` |

### Build compact cross-dataset manuscript figures

```bash
bash scripts/build_main_paper_figures.sh
```

`build_main_paper_figures.sh` reads saved DP-SPIN reports and writes the compact manuscript figures. It does not rerun mechanisms, decoders, or baselines.

| Variable | Default |
|---|---|
| `REPORTS_ROOT` | `outputs` |
| `OUTPUT_DIR` | `outputs/main_paper_figures` |
| `MAIN_FIG_K` | `3` |
| `MAIN_FIG_L` | `10` |
| `MAIN_FIG_USER_B` | `1` |

### Build cross-dataset OpenAI tables

```bash
bash scripts/build_openai_main_tables.sh
```

`build_openai_main_tables.sh` reads dataset-scoped reports and writes compact cross-dataset tables without calling OpenAI.

| Variable | Default |
|---|---|
| `REPORTS_ROOT` | `outputs/openai_paper` |
| `OUTPUT_DIR` | `<reports-root>/main_tables` |
| `UNIT` | `record` |
| `QUALITATIVE_MAX_CHARS` | `650` |
| `DATASETS` | empty; use all available dataset reports |

## Privacy boundary of saved artifacts

Run directories contain both releasable outputs and non-private evaluation artifacts. The releasable interface is the differentially private semantic plan together with outputs computed only from that plan and public objects. The non-private semantic sketch, reference plan, reference summary, clipped user assignments, protected target records, and target-derived baselines are evaluation artifacts and must not be exposed as deployment outputs.

Controlled-probe runs add records containing a fixed public pattern and add its atom to the public dictionary before release. They measure admission of that known coordinate and its mention after verbalization. They do not test whether a new private atom can be discovered or named from protected text.
