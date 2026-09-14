# Reproducibility and release checklist

## Environment

1. Create a fresh Python environment.
2. Install `requirements.txt` and only the optional dependency files needed by the selected experiments.
3. Record Python, NumPy, SciPy, scikit-learn, PyTorch, Transformers, sentence-transformers, and OpenAI client versions as applicable.
4. Record CPU, RAM, accelerator, operating system, and relevant thread-count environment variables.

## Frozen inputs

Archive or hash:

- source dataset versions;
- preparation and split manifests;
- public and auxiliary records used to construct dictionaries;
- frozen atom dictionaries and their manifests;
- assignment-model names and revisions;
- public denominators;
- annotation guidelines, raw annotations, and adjudicated labels;
- complete resolved configurations.

Do not tune dictionaries, thresholds, models, or prompts on the protected target results that will be reported as final evaluation.

## Randomness

Use distinct values for:

- `data_seed`;
- `mechanism_seed`;
- `decoder_seed`;
- `judge_seed`.

For paired comparisons, reuse the same target sample and data seed across methods while varying mechanism randomness independently according to the documented design.

## Privacy-boundary check

For each completed run:

1. Inspect `release_bundle/privacy_boundary_manifest.json`.
2. Confirm that `release_bundle/` contains only listed files.
3. Confirm that raw records, per-unit assignments, clean sketches, and non-private references appear only in `offline_evaluation/` or other internal analysis directories.
4. Confirm that the decoder prompt was constructed only from the DP plan and public rules.
5. Confirm that any denominator is public or privately released with accounted privacy cost.

## Matched comparisons

Cross-method comparisons should use identical:

- target record or user IDs;
- auxiliary split;
- atom universe or a clearly documented alternative representation;
- protected-unit definition;
- adjacency relation;
- contribution bound;
- privacy parameters;
- public denominator;
- requested plan size;
- evaluation reference.

Use `src.matched_protocol` to identify mismatches before aggregating results.

## Regeneration sequence

A clean end-to-end run should follow this order:

1. acquire or validate raw data;
2. prepare normalized records;
3. create disjoint auxiliary and protected target splits;
4. build and freeze public atom dictionaries;
5. calibrate abstention on auxiliary annotations only;
6. run the DP mechanisms and baselines;
7. aggregate repeated trials and uncertainty intervals;
8. prepare blinded common-ground evaluation;
9. regenerate figures and tables from saved artifacts;
10. verify the release/offline artifact boundary.

## Repository validation

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
bash scripts/run_validation_smoke.sh
```

The validation workflow uses synthetic data and does not replace real-data execution.
