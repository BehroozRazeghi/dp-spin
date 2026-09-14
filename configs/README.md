# Configuration files

The YAML files in this directory define single DP-SPIN runs or provide base
configurations for dataset-specific grids. The runners resolve each base file
into a per-run configuration before constructing bounded sparse semantic
contributions, the aggregate semantic sketch, the differentially private
semantic plan, and the final summary.

The configuration file is not, by itself, a privacy certificate. The file paths
and protocol choices must satisfy the privacy boundary described below.

## Files

| File | Role |
|---|---|
| `demo_record.yaml` | Synthetic record-level example used by `scripts/run_demo.sh`. |
| `demo_user.yaml` | Synthetic user-level example used by `scripts/run_demo.sh`. |
| `cfpb_template.yaml` | General manual CFPB record-level configuration. |
| `cfpb_m50.yaml` | Manual CFPB run with the separately constructed 50-atom dictionary. |
| `cfpb_m100.yaml` | Manual CFPB run with the separately constructed 100-atom dictionary. |
| `cfpb_m200.yaml` | Manual CFPB run with the separately constructed 200-atom dictionary. |
| `cfpb_probe.yaml` | Manual CFPB controlled probe-atom configuration. |
| `amazon_record_template.yaml` | Base configuration for Amazon record-level grids. |
| `amazon_user_template.yaml` | Base configuration for Amazon user-level grids. |
| `yelp_record_template.yaml` | Base configuration for Yelp record-level grids. |
| `yelp_user_template.yaml` | Base configuration for Yelp user-level grids. |
| `validation_demo.yaml` | Synthetic end-to-end validation configuration. |
| `synthetic_benchmark.yaml` | Controlled ground-truth benchmark grid. |
| `external_validity_template.yaml` | Assignment-family, transfer, and dictionary-stress protocol. |
| `external_validity_demo.yaml` | Fully local external-validity software example. |
| `decoder_external_validity_template.yaml` | Multi-decoder and distinct-judge protocol. |
| `scaling_benchmark.yaml` | Release-time runtime, memory, and payload benchmark. |
| `auto/*.yaml` | Generated CFPB base configurations; `run_cfpb.sh` may replace them. |

The main CFPB workflow uses generated files under `configs/auto/`; it does not
read `cfpb_template.yaml` or the three manual `cfpb_m*.yaml` files.

## Configuration precedence

For a single run, `src.run_experiment` loads the YAML file and then applies any
supported command-line overrides. For a grid, `src.run_grid` copies the base
configuration for each operating point and replaces the grid fields supplied on
the command line.

The `config.yaml` and `config.json` written inside each run directory are the
resolved configurations. They record the realized atom count, selected protected
unit count, effective privacy parameters, and other run-specific values.

## Data and output fields

```yaml
experiment_name: run-name prefix
records: protected target JSONL file
atoms: public semantic atom dictionary
embedding_fit_records: public or disjoint auxiliary JSONL used to fit the TF-IDF map
output_dir: parent directory for run folders
```

`records` is the protected target split. The public atom dictionary and the
representation used by the semantic contribution map must be fixed independently
of this split. In the reported user-level experiments, the auxiliary and
protected target splits are user-disjoint.

If `embedding_fit_records` is omitted, the implementation fits the selected fixed representation on the
public atom text only. The demo-only option that fits on protected target text is
outside the privacy guarantee used for the paper experiments.

## Protected unit and adjacency

```yaml
protected_unit: record or user
adjacency: adddrop or replacement
n_records: number of protected records selected for a record-level run
n_users: number of protected users selected for a user-level run
record_selection: hash, random, or first
user_selection: hash, random, or first
user_id_field: field used to group records by user
user_contribution_bound: l1 clipping bound B for one user's aggregate contribution
public_denominator: public denominator used to compute released normalized semantic-support values
```

For `protected_unit: record`, one record is one protected contribution. For
`protected_unit: user`, the runner selects complete users, includes all records
for each selected user, sums their record-level semantic vectors, and clips the
result to l1 norm at most `user_contribution_bound` before applying the release
mechanism.

A user-level configuration must not also impose `n_records`. Independent record
truncation can split a user's protected contribution.

The reported experiments use add/drop adjacency. Replacement adjacency is
implemented for controlled comparisons.

`public_denominator` is used only to convert nonnegative released noisy masses
into normalized semantic-support values and bins. Under add/drop adjacency, do not replace it
with a realized private corpus size unless that size is public or is released
through a separately accounted differentially private mechanism.

The grid flags
`auto_public_denominator_from_n_records` and
`auto_public_denominator_from_n_users` keep the denominator synchronized with a
publicly specified grid size.

## Semantic contribution map

```yaml
m: requested number of public atom coordinates
K: assignment sparsity parameter
L: number of atoms admitted to the released semantic plan
gamma: softmax concentration parameter for retained atom-similarity scores
```

The model permits at most `K` nonzero coordinates and total l1 mass at most one.
The assignment layer supports `assignment_embedding_family: tfidf_word | tfidf_char | sentence_transformer`,
`assignment_embedding_min_df`, `assignment_embedding_max_features`,
`assignment_embedding_word_ngram_range`, `assignment_embedding_char_ngram_range`,
and the following fixed-pretrained-encoder controls:
`assignment_embedding_model_name`, `assignment_embedding_model_revision`,
`assignment_embedding_model_batch_size`, `assignment_embedding_model_device`,
`assignment_embedding_model_local_files_only`, and
`assignment_embedding_model_trust_remote_code`. Reproducible runs should pin a
specific model revision or commit and archive the resolved metadata. The
sentence encoder performs no fitting on project records; TF-IDF families may be
fitted only on public atom text and optional public or disjoint auxiliary text.

The assignment layer also supports `assignment_similarity_threshold`,
`assignment_abstention_mode: zero | other_atom | disabled`, and
`assignment_mass_mode: unit | confidence`. The threshold and any public
`other/uncovered` atom must be fixed from public or disjoint auxiliary data.
Covered records retain at most `K` atoms; uncovered records may contribute zero
mass. Larger `gamma` values concentrate mass on the retained atoms.

The runner loads the atom file before assigning protected target records. If
`m` is smaller than the file length, it uses the first `m` entries. If the file
contains fewer than `m` atoms, it uses the full file and records the realized
count. For the CFPB grid, each value of `m` is paired with a separately
constructed dictionary through `--atoms-template`; smaller CFPB dictionaries
are not obtained by truncating the 200-atom file.

## Differentially private semantic-plan release

```yaml
epsilon: configured privacy budget
delta: configured approximate-DP parameter
mechanism: gaussian, laplace, or private_topL
gaussian_calibration: analytic or classical
value_epsilon_fraction: fraction of epsilon used for selected-value release
value_mechanism: gaussian or laplace
seed: legacy base seed
data_seed: protected-unit selection seed
mechanism_seed: DP mechanism seed
decoder_seed: decoder seed (recorded; backend support may vary)
judge_seed: automated-judge seed (recorded; backend support may vary)
```

The mechanisms are:

- `gaussian`: perturb the full semantic sketch with Gaussian noise, then admit
  the top `L` atoms by post-processing;
- `laplace`: perturb the full semantic sketch with Laplace noise, then admit the
  top `L` atoms by post-processing;
- `private_topL`: admit `L` atoms through sequential exponential-mechanism
  rounds, then release noisy masses only for the admitted atoms.

For `private_topL`, the default value-release fraction is `0.5`. If
`value_epsilon_fraction = rho`, then `rho * epsilon` is used for selected-value
release. `selection_budget_allocation` selects `uniform`, `front_loaded`, or
`theory_optimal` allocation of the remaining budget across sequential rounds;
the last option minimizes the manuscript's cumulative regret surrogate. Each
run records the epsilon schedule, per-round alpha bounds, cumulative certificate,
and sufficient exact-recovery gap. The default selected-value mechanism is Gaussian.

A Laplace full-sketch release has effective `delta = 0`. A `private_topL` run
also has effective `delta = 0` when its selected values are released with
Laplace noise, even if the shared base configuration contains a nonzero
configured `delta`.

Use classical Gaussian calibration only for `0 < epsilon < 1`. Analytic
calibration is the default for general positive `epsilon`.

## Decoder, verifier, and offline evaluation

```yaml
decoder_backend: template, openai, or hf_local
decoder_contract_mode: legacy_text (paper protocol) or structured
verifier_max_attempts: maximum number of decoder attempts before fallback
verifier_comparison_margin: public minimum released-mass comparison margin
summary_embedding_backend: none, tfidf, auto, or sentence_transformer
summary_comparative_eval_backend: none, heuristic, openai, or hf_local
clean_reference_decoder_backend: template, openai, or hf_local
```

The DP-SPIN decoder receives only the released semantic plan and public
instructions. The reported paper experiments use `legacy_text`, which verifies marker-preserving audit text against the released plan. `structured` remains an optional typed-claim mode in which the decoder proposes claims over released atom identifiers and the accepted claim list is rendered deterministically.
It does not receive raw protected records, raw excerpts, per-record semantic
vectors, the non-private sketch, non-private representatives, or labels outside
the admitted atom set.

The clean reference plan, clean reference summary, and text-level comparison
metrics are non-private evaluation artifacts. They are not part of the released
DP-SPIN interface.

`atom_source_protocol`, `atom_dictionary_protocol`, and `config_role` are
descriptive metadata saved with a run. They document the intended protocol but
do not enforce it.

## Support bins

```yaml
support_bins:
  very high: 0.20
  high: 0.10
  moderate: 0.05
  low: 0.0
```

The thresholds are fractions of `public_denominator`. The plan builder clips a
released noisy mass below zero to zero for display, divides by the public
denominator, and assigns the first matching bin in descending threshold order.

## Dataset protocols

### CFPB consumer complaint narratives

The CFPB workflow uses record-level protection because the prepared public data
do not provide a stable identifier for grouping all records from one person.
The public atom dictionaries are constructed from the disjoint auxiliary split
using the `Product / Issue / Sub-issue` hierarchy:

```text
data/public_atoms/cfpb_atoms_m50.json
data/public_atoms/cfpb_atoms_m100.json
data/public_atoms/cfpb_atoms_m200.json
```

The controlled probe experiment adds a known public probe atom before processing
the protected target corpus. It measures differentially private admission of
that public coordinate; it is not a raw-string extraction experiment.

### Amazon All Beauty reviews

The public atom dictionary is constructed from the user-disjoint auxiliary
split. Under the reported preparation protocol, the realized dictionary has 46
atoms. The record-level and user-level templates therefore use `m: 46`.

### Yelp Restaurant reviews

The workflow constructs auxiliary-only dictionaries with requested sizes 50,
100, and 200. The main grid uses 50 and 100 atoms; the 200-atom dictionary is
used for an ablation or manual diagnostics.

## Checks before a run

Verify the following conditions before treating an output as a privacy-claimed
release:

1. `records` points to the protected target split.
2. Atom labels and descriptions are fixed independently of the protected target
   split or released by a separately accounted private mechanism.
3. `embedding_fit_records` is public or disjoint auxiliary data.
4. The protected unit, adjacency relation, sample-size field, contribution
   bound, and public denominator describe the same protocol.
5. A user-level run defines `user_id_field`, `n_users`, and a positive
   `user_contribution_bound`, and does not define `n_records`.
6. `1 <= K <= m` and `1 <= L <= m` for the realized atom count.
7. `epsilon` is positive. Gaussian releases use `0 < delta < 1`.
8. Any external service that receives protected target text is inside the stated
   trusted-curator boundary. Otherwise, that disclosure is outside the
   repository's privacy guarantee.

## Examples

Run one synthetic record-level experiment:

```bash
python -m src.generate_demo_data
python -m src.run_experiment --config configs/demo_record.yaml
```

Override selected fields for a manual CFPB run:

```bash
python -m src.run_experiment \
  --config configs/cfpb_m100.yaml \
  --epsilon 2 \
  --seed 1 \
  --mechanism gaussian
```

Run the maintained dataset workflows through their shell scripts rather than by
editing generated files under `configs/auto/`.


## External-validity configurations

- `external_validity_template.yaml` defines a paired fixed-representation experiment with word TF-IDF, character TF-IDF, and a pinned pretrained sentence encoder; domain/time transfer against a matched target dictionary on the same target records; and dictionary-misspecification/abstention risk-coverage studies.
- `external_validity_demo.yaml` is a fully local software-validation example using bundled synthetic records and annotations. It intentionally omits the optional pretrained model and therefore is not a paper result.
- `decoder_external_validity_template.yaml` defines at least two decoder conditions and a distinct automated judge. For the open-weight decoder, pin `revision`, record `device_map`, and set `local_files_only: true` after the model snapshot has been archived. The `hf_local` backend requires the optional `local-llm` dependencies.
- `scaling_benchmark.yaml` controls dictionary sizes, released-plan sizes, mechanisms, warm-up runs, repetitions, and privacy parameters for release-time runtime, peak Python-tracked memory, and numeric payload measurements.

The transfer configuration requires explicit source/target domain and period metadata. If `matched_target_atoms` is supplied, both source and matched dictionaries are evaluated on the identical target records and their own compatible annotations. The misspecification configuration distinguishes deployable random/shuffle stress tests from the evaluation-only `drop_frequent_gold` adversary. A threshold is called selected only when it is chosen on a disjoint annotated calibration set; otherwise the target threshold grid is descriptive.
