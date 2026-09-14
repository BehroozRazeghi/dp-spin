# External-validity experiments

This document describes the repository workflows that evaluate the semantic layer independently of the differentially private release mechanism.

## Evaluation questions

The external-validity suite addresses five questions:

1. Does the fixed assignment map remain accurate across representation families?
2. How does a frozen dictionary transfer across time periods or domains?
3. How does dictionary misspecification affect forced assignment and abstention?
4. Do language-quality conclusions persist across decoders that are distinct from the automated judge?
5. How do release time and memory scale with dictionary size and plan size?

These experiments do not alter the privacy analysis. Every tested assignment representation and dictionary must be fixed before processing the protected target split.

## Required data layout

Use the existing repository structure:

```text
data/annotations/       completed annotations and adjudication files
data/auxiliary/         public or disjoint auxiliary fitting/calibration records
data/frozen_atoms/      immutable dictionary snapshots and manifests
data/protected_target/  target records used only after protocols are frozen
data/public_atoms/      active public dictionaries
```

Store outputs below `outputs/external_validity/`, `outputs/decoder_external_validity/`, and `outputs/scaling/`.

## Assignment-family comparison

`src.external_validity` supports the following fixed families:

- `tfidf_word`
- `tfidf_char`
- `sentence_transformer`

The sentence-transformer condition should record a pinned model revision. TF-IDF vocabularies may be fitted only on public atom text and optional disjoint auxiliary text.

Run:

```bash
bash scripts/run_external_validity.sh \
  configs/external_validity_template.yaml \
  outputs/external_validity/results.json
```

The report includes record-paired estimates and bootstrap intervals for top-1 precision, top-`K` recall, coverage, selective risk, uncovered detection, and false assignment on uncovered records.

## Dictionary transfer

Each transfer task must identify:

- source and target domain;
- source and target time period;
- target records;
- source dictionary and its public fitting corpus;
- annotations defined against the source dictionary;
- an optional matched target dictionary and matched annotations.

Source and matched conditions must be evaluated on identical target record IDs. If dictionaries use different atom identifiers, each condition needs its own annotation mapping while retaining the paired target records.

## Dictionary misspecification and abstention

Supported stress conditions include:

- random atom removal;
- evaluation-only removal of frequently annotated atoms;
- atom-description shuffling.

Compare forced assignment against public-threshold abstention over a prespecified threshold grid. Select an operating threshold only on disjoint auxiliary annotations. If no calibration set is supplied, report the complete risk-coverage curve and do not label a target-selected threshold as recommended.

Useful quantities include:

- selective top-1 error;
- coverage;
- uncovered-record detection;
- false assignment among uncovered records;
- decision cost `error_cost * errors + abstention_cost * abstentions`.

## Decoder and judge separation

`src.decoder_external_validity` evaluates a panel of decoders against a separately specified judge. The role validator rejects a learned decoder that is identical to the automated judge when distinct roles are required.

```bash
bash scripts/run_decoder_external_validity.sh \
  configs/decoder_external_validity_template.yaml \
  outputs/decoder_external_validity/results.json
```

Automated judging is secondary. Use `src.human_eval` for blinded common-ground evaluation of factual support, important-theme coverage, unsupported additions, usefulness, and clarity.

## Scaling benchmark

```bash
bash scripts/run_scaling_benchmark.sh \
  configs/scaling_benchmark.yaml \
  outputs/scaling/scaling_results.json
```

The benchmark isolates plan release from assignment and decoding. It reports warm-run wall-clock time, Python-tracked peak memory, numeric payload size, and descriptive log-log slopes over dictionary size. `tracemalloc` does not capture every native-library or accelerator allocation, so publish the hardware/software manifest and add process RSS or device-memory measurements when available.

## Interpretation

Keep the three evidence layers separate:

- mechanism fidelity measures agreement with the non-private assignment sketch;
- semantic validity measures agreement with independent annotations;
- language usefulness measures the quality of the final summary against a common reference.

A result at one layer does not establish performance at the others.
