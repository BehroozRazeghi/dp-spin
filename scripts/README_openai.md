# OpenAI verbalization and release-conditioned evaluation

The OpenAI scripts verbalize saved DP-SPIN semantic plans, generate selected baseline summaries, run release-conditioned evaluation, and build manuscript tables. They read completed local runs and do not rerun a differentially private plan-release mechanism.

See [`README.md`](README.md) for the full script index.

## Privacy boundary

For DP-SPIN verbalization, the OpenAI decoder receives only:

- the released differentially private semantic plan;
- public labels and descriptions for admitted atoms;
- released noisy masses, released ranks, or prevalence bins contained in or computed publicly from the released plan; and
- public decoder instructions.

It does not receive:

- raw protected target records or excerpts;
- nearest neighbors, exemplars, or non-private representatives from the protected target split;
- raw per-record or per-user semantic contribution vectors;
- the non-private semantic sketch;
- the non-private reference plan or reference summary as decoder context; or
- labels or descriptions for atoms outside the released plan.

Each selected plan produces two summaries:

- the **audit summary**, which preserves required atom markers used by the public plan-consistency verifier; and
- the **display summary**, which verbalizes the same released-plan content in presentation-oriented text without requiring marker preservation.

Both summaries use the same released differentially private semantic plan. The verifier receives that plan and the audit summary, not the protected corpus.

## Release-conditioned evaluation

The generic judge receives a generated summary and the method-specific object supplied to that method's decoder. It does not receive the protected target corpus. Scores therefore measure verbalization relative to the released object.

These scores:

- are secondary verbalization-quality measures;
- are not differential-privacy guarantees;
- do not measure recovery of the non-private semantic sketch;
- do not imply that different methods released equivalent semantic information; and
- do not replace the plan-level utility measures.

Primary mechanism evaluation remains plan-level: coordinate error, top-`L` overlap, retained non-private mass fraction, support-mass loss, noisy-value error, and controlled-probe admission. Verifier metrics test consistency with the released plan under fixed rules; they do not measure completeness relative to the protected corpus.

## Baselines and API-visible data

The default manuscript workflow includes:

- DP-SPIN released semantic plans;
- differentially private fixed-vocabulary keyword histograms;
- differentially private fixed-vocabulary category histograms; and
- a URANIA-style released public-keyword object.

The optional `raw`, `redacted`, and `non_private_clustering` methods are non-private utility references. They can expose raw or target-derived content to the selected model, fall outside the DP-SPIN decoder boundary, and are disabled by default.

Enable a non-private reference only after checking its prompt, dataset terms, institutional policy, and intended privacy protocol.

## Prerequisites

Install the repository requirements and set an API key:

```bash
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your_key"
```

The default model is `gpt-4o-mini`. `OPENAI_MODEL` may be set to another identifier supported by the installed client and account.

Completed dataset runs must already exist. Selected runs are expected to contain at least:

```text
private_plan.json
config.yaml
```

The pipeline selects matching operating points from saved run artifacts; it cannot create a missing DP release.

Default source-run roots are:

```text
outputs/cfpb/main/runs/
outputs/amazon_All_Beauty/record/runs/
outputs/amazon_All_Beauty/user/runs/
outputs/yelp_restaurants/record/runs/
outputs/yelp_restaurants/user/runs/
```

Run these scripts from the repository root.

## Script roles

### `run_openai_paper.sh`

The maintained manuscript workflow:

1. selects completed DP-SPIN runs;
2. generates audit and display summaries from their released semantic plans;
3. optionally computes legacy plan-conditioned DP-SPIN judgments;
4. generates the configured baseline summaries;
5. generates the URANIA-style public-keyword baseline by default;
6. runs the generic release-conditioned judge by default;
7. writes dataset-scoped CSV, JSON, Markdown, and LaTeX reports; and
8. rebuilds the compact cross-dataset manuscript tables by default.

It does not rerun `src.run_experiment`, `src.run_grid`, or any DP-SPIN plan-release mechanism.

### `run_openai_dpspin_only.sh`

Verbalizes selected saved DP-SPIN plans only. It does not run histogram baselines, the
URANIA-style baseline, cross-method comparison, or the compact table builder.

### `_run_openai_paper_driver.sh`

Maps environment variables to `src.openai_paper_pipeline` arguments. Invoke
`run_openai_paper.sh` rather than this internal driver.

### `build_openai_main_tables.sh`

Builds compact tables from saved dataset-scoped reports. It makes no OpenAI call.

## Manuscript workflow

Run the representative profile for all configured datasets:

```bash
bash scripts/run_openai_paper.sh
```

Run one dataset:

```bash
DATASET=cfpb bash scripts/run_openai_paper.sh
DATASET=amazon bash scripts/run_openai_paper.sh
DATASET=yelp bash scripts/run_openai_paper.sh
```

Run the larger OpenAI selection profile:

```bash
OPENAI_PROFILE=grid bash scripts/run_openai_paper.sh
```

Supported profile names are `representative` and `grid`.

### Default representative profile

Unless overridden, `run_openai_paper.sh` uses:

```text
DP-SPIN mechanisms             gaussian private_topL
privacy budgets                1 4
seeds                          0 1 2
assignment sparsity K          3
released plan size L           10
DP histogram baselines         dp_keyword dp_category
URANIA-style baseline          enabled
legacy DP-SPIN-only judge      none
generic release-conditioned judge  openai
comparative summary evaluator  none
summary embedding              tfidf
qualitative truncation         disabled
main-table construction        enabled
```

The `grid` profile changes the privacy-budget list to `0.5 1 2 4`; other defaults remain unchanged unless overridden.

The cross-method comparison uses one matched DP-SPIN condition:

```text
COMPARISON_DPSPIN_MECHANISM=gaussian
COMPARISON_DPSPIN_EPSILON=1.0
```

Together, these variables fix the DP-SPIN release condition used in the cross-method comparison.

## Common examples

Use one seed for both DP-SPIN and baseline selection:

```bash
SEEDS="0" BASELINE_SEEDS="0" DATASET=cfpb \
  bash scripts/run_openai_paper.sh
```

Select one mechanism and one privacy budget:

```bash
MECHS="gaussian" EPS="1" DATASET=amazon \
  bash scripts/run_openai_paper.sh
```

Restrict the protected target size used to select DP-SPIN runs:

```bash
N_RECORDS_VALUES="2000" DATASET=cfpb \
  bash scripts/run_openai_paper.sh
```

Select record-level and user-level runs and explicit clipping bounds:

```bash
UNITS="record user" USER_BOUNDS="1 2" DATASET=yelp \
  bash scripts/run_openai_paper.sh
```

Use sparse private atom admission as the matched DP-SPIN comparison condition:

```bash
COMPARISON_DPSPIN_MECHANISM=private_topL \
COMPARISON_DPSPIN_EPSILON=4 \
DATASET=cfpb \
  bash scripts/run_openai_paper.sh
```

Method names contain spaces, so method lists use `|` rather than spaces as the method separator:

```bash
COMPARISON_METHODS="DP-SPIN private plan|DP keyword histogram|DP category histogram" \
  bash scripts/run_openai_paper.sh
```

`DP-SPIN private plan` is a reporting label for verbalization from the released differentially private semantic plan, not a separate mechanism.

## `run_openai_paper.sh` controls

### Dataset and run selection

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_PROFILE` | `representative` | `representative` or `grid` |
| `DATASET` | `all` | `all`, `cfpb`, `amazon`, or `yelp` |
| `OUTPUT_DIR` | `outputs/openai_paper` | Root directory for OpenAI reports |
| `MECHS` | `gaussian private_topL` | Selected DP-SPIN mechanisms |
| `EPS` | `1 4` for representative; `0.5 1 2 4` for grid | Selected privacy budgets |
| `SEEDS` | `0 1 2` | Selected DP-SPIN seeds |
| `UNITS` | empty | Optional `record`, `user`, or both; empty uses dataset defaults |
| `M_VALUES` | empty | Optional atom-universe-size filter |
| `USER_BOUNDS` | empty | Optional user clipping-bound filter |
| `N_RECORDS_VALUES` | empty | Optional record-count filter |
| `N_USERS_VALUES` | empty | Optional user-count filter |
| `K_VALUE` | `3` | Assignment-sparsity filter |
| `L_VALUE` | `10` | Released-plan-size filter |
| `MAX_DPSPIN_RUNS_PER_DATASET` | empty | Optional cap after run selection |

Dataset defaults used when filters are empty are defined in `src.openai_paper_pipeline`:

| Dataset | Units | Default `m` | Default baseline subset | Default user `B` |
|---|---|---:|---:|---:|
| CFPB | record | 200 | 2,000 | not applicable |
| Amazon All Beauty | record, user | 46 | 2,000 | 1 |
| Yelp Restaurants | record, user | 100 | 2,000 | 1 |

### Model and decoding

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` | Model used for OpenAI verbalization and judging |
| `OPENAI_TEMPERATURE` | `0.0` | Sampling temperature |
| `OPENAI_MAX_TOKENS` | `700` | Maximum generated tokens |
| `FORCE` | `0` | Rebuild existing OpenAI outputs when set to `1` |
| `OPENAI_REQUIRED` | `1` | Require an API key outside report-only mode |

Model outputs may differ across calls even at temperature zero. Preserve the raw responses and
metadata used for reported results.

### Baselines

| Variable | Default | Meaning |
|---|---|---|
| `RUN_BASELINES` | `1` | Generate selected baseline outputs |
| `BASELINE_SUBSET_SIZE` | empty | Use dataset-specific default when empty |
| `BASELINE_EPSILON` | `1.0` | Privacy budget for DP histogram baselines |
| `BASELINE_SEED` | `0` | Backward-compatible single baseline seed |
| `BASELINE_SEEDS` | `SEEDS` | Baseline seed list |
| `BASELINE_METHODS` | `dp_keyword dp_category` | Methods passed to `src.baselines` |
| `RUN_URANIA` | `1` | Generate the URANIA-style public-keyword baseline |
| `URANIA_EPS_TOTAL` | `1.0` | Total URANIA-style baseline budget |
| `URANIA_DELTA` | `1e-6` | URANIA-style baseline delta |
| `URANIA_CLUSTERS` | `10` | Requested number of private clusters |
| `URANIA_KEYWORDS` | `10` | Released public keywords per summary |

Supported `BASELINE_METHODS` values are:

```text
raw
redacted
non_private_clustering
dp_keyword
dp_category
```

The first three are non-private references. The default two are differentially private fixed-vocabulary histogram baselines.

### Judging and secondary text evaluation

| Variable | Default | Meaning |
|---|---|---|
| `RUN_JUDGE` | `1` | Run the configured generic judge |
| `DPSPIN_JUDGE_BACKEND` | `none` | Legacy DP-SPIN-only plan-conditioned judge: `none`, `heuristic`, or `openai` |
| `GENERIC_JUDGE_BACKEND` | `openai` | Release-conditioned cross-method judge: `none`, `heuristic`, or `openai` |
| `COMPARATIVE_BACKEND` | `none` | Optional DP-summary/non-private-reference comparison: `none`, `heuristic`, or `openai` |
| `SUMMARY_EMBEDDING_BACKEND` | `tfidf` | `tfidf`, `auto`, or `sentence_transformer` |
| `JUDGE_METHODS` | empty | Empty uses the union of comparison and qualitative methods |
| `COMPARISON_METHODS` | four default methods | Pipe-delimited methods included in the comparison table |
| `COMPARISON_DPSPIN_MECHANISM` | `gaussian` | DP-SPIN mechanism used in the cross-method comparison |
| `COMPARISON_DPSPIN_EPSILON` | `1.0` | DP-SPIN privacy budget used in the cross-method comparison |
| `QUALITATIVE_METHODS` | `COMPARISON_METHODS` | Pipe-delimited methods included in qualitative tables |
| `QUALITATIVE_MAX_CHARS` | `0` | Character limit for dataset-scoped qualitative output; `0` disables truncation |

The generic judge's `privacy_safety` field is a text-level assessment, not a formal privacy guarantee.

### Execution and main-table construction

| Variable | Default | Meaning |
|---|---|---|
| `RUN_DPSPIN` | `1` | Generate DP-SPIN OpenAI verbalizations |
| `REPORT_ONLY` | `0` | Rebuild reports from saved outputs without API calls |
| `BUILD_MAIN_TABLES` | `1` | Rebuild compact cross-dataset tables after the dataset workflow |
| `MAIN_TABLE_UNIT` | `record` | Unit selected for compact main tables |
| `MAIN_TABLE_MAX_CHARS` | `650` | Qualitative excerpt limit in compact tables |
| `MAIN_TABLE_OUTPUT_DIR` | `<OUTPUT_DIR>/main_tables` | Compact table directory |

The default cross-method table uses record-level rows because the DP histogram and URANIA-style objects are record-level releases. Use `MAIN_TABLE_UNIT=user` only when every compared method and the stated interpretation support user-level analysis.

## Report-only mode

Rebuild reports and tables from existing outputs without making API calls:

```bash
REPORT_ONLY=1 bash scripts/run_openai_paper.sh
```

Report-only mode does not require an API key. It rebuilds files from saved structured outputs but cannot recover missing summaries, raw responses, or judgments.

Examples:

```bash
REPORT_ONLY=1 DATASET=cfpb \
  bash scripts/run_openai_paper.sh

REPORT_ONLY=1 BUILD_MAIN_TABLES=1 MAIN_TABLE_MAX_CHARS=500 \
  bash scripts/run_openai_paper.sh
```

## DP-SPIN-only verbalization

Verbalize selected saved DP-SPIN plans without generating baselines or cross-method manuscript tables:

```bash
DATASET=amazon bash scripts/run_openai_dpspin_only.sh
```

Example with explicit units and release settings:

```bash
DATASET=amazon \
UNITS="record user" \
MECHS="gaussian private_topL" \
EPS="1 4" \
SEEDS="0" \
  bash scripts/run_openai_dpspin_only.sh
```

Controls:

| Variable | Default | Meaning |
|---|---|---|
| `DATASET` | `all` | `all`, `cfpb`, `amazon`, or `yelp` |
| `OPENAI_MODEL` | `gpt-4o-mini` | Decoder model |
| `OPENAI_TEMPERATURE` | `0.0` | Decoder temperature |
| `OPENAI_MAX_TOKENS` | `700` | Maximum generated tokens |
| `MECHS` | `gaussian private_topL` | Mechanism filter |
| `EPS` | `1 4` | Privacy-budget filter |
| `SEEDS` | `0` | Seed filter |
| `UNITS` | empty | Optional protected-unit filter |
| `M_VALUES` | empty | Optional atom-universe-size filter |
| `USER_BOUNDS` | empty | Optional user clipping-bound filter |
| `K_VALUE` | `3` | Assignment-sparsity filter |
| `L_VALUE` | `10` | Released-plan-size filter |
| `JUDGE_BACKEND` | `none` | Optional legacy plan-conditioned judge |
| `COMPARATIVE_BACKEND` | `none` | Optional summary-reference comparison backend |
| `SUMMARY_EMBEDDING_BACKEND` | `tfidf` | Text-similarity backend |
| `FORCE_REDECODE` | `0` | Regenerate existing decoded outputs |
| `MAX_RUNS_PER_DATASET` | empty | Optional selected-run cap |
| `OUTPUT_DIR` | empty | Empty uses dataset-specific defaults |

Dataset-specific default output roots are:

```text
outputs/cfpb/openai_representative/
outputs/amazon_All_Beauty/openai_representative/
outputs/yelp_restaurants/openai_representative/
```

Representative run folders contain the selected released plan and configuration, audit and display summaries, prompts, verifier results, optional judgments, and secondary text-evaluation files.

## Output structure

The default output root is:

```text
outputs/openai_paper/
```

Dataset-scoped report directories are:

```text
outputs/openai_paper/cfpb/
outputs/openai_paper/amazon_All_Beauty/
outputs/openai_paper/yelp_restaurants/
outputs/openai_paper/all_datasets/
```

The selected `DATASET` determines which report directories are written. A dataset-scoped report includes files such as:

```text
tables/dpspin_openai_runs.csv
tables/baseline_runs.csv
judgments/judgments.csv
latex/tab_openai_dpspin.tex
latex/tab_openai_baseline_comparison.tex
latex/tab_openai_qualitative_examples.tex
latex/qualitative_examples.md
README_OPENAI_PAPER_REPORT.md
openai_paper_report_index.json
```

The main workflow also rebuilds compact cross-dataset tables:

```text
outputs/openai_paper/main_tables/tab_openai_main_comparison.tex
outputs/openai_paper/main_tables/tab_openai_main_dpspin.tex
outputs/openai_paper/main_tables/tab_openai_main_qualitative_examples.tex
```

Build those tables directly from existing reports:

```bash
bash scripts/build_openai_main_tables.sh
```

Equivalent module call:

```bash
python -m src.openai_main_tables \
  --reports-root outputs/openai_paper \
  --output-dir outputs/openai_paper/main_tables \
  --unit record \
  --qualitative-max-chars 650
```

Optional table-builder controls:

```bash
REPORTS_ROOT=outputs/openai_paper \
OUTPUT_DIR=outputs/openai_paper/main_tables \
UNIT=record \
QUALITATIVE_MAX_CHARS=650 \
DATASETS="cfpb amazon_All_Beauty yelp_restaurants" \
  bash scripts/build_openai_main_tables.sh
```

## API cost and reproducibility

OpenAI verbalization and judging may incur API charges. Call count depends on the selected runs, baselines, judges, regeneration settings, and datasets.

For reported results:

- record the exact model identifier and all environment overrides;
- retain the generated prompts, raw model responses, parsed outputs, judgments, and report metadata;
- preserve the source run directories and released plans selected for verbalization;
- distinguish audit summaries from display summaries;
- do not combine outputs from materially different model versions without reporting the distinction;
- preserve the baseline method list and the method-specific released object used by the judge; and
- use `REPORT_ONLY=1` when regenerating tables from a frozen output set.

A fixed temperature request does not guarantee byte-identical OpenAI outputs.
