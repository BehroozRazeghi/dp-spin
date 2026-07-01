# Generated CFPB configurations

The YAML files in this directory are generated base configurations for the CFPB
workflow. `scripts/run_cfpb.sh` may replace them at any time. Do not use manual
edits here as the source of truth for an experiment.

The grid runner copies a base configuration and overrides the operating-point
fields supplied on the command line, including the privacy budget, seed,
mechanism, atom count, assignment sparsity, released plan size, and protected
record count. The resolved configuration saved in each run directory is the
record of the executed run.

For dictionary-size sweeps, `scripts/run_cfpb.sh` passes an atom-path template so
that each requested value of `m` uses its separately constructed CFPB atom file.
The current controlled-probe workflow generates files named
`cfpb_probe_m<base_m>.yaml`. The retained `cfpb_probe.yaml` file is a legacy
generated example and should not be used as the current probe-workflow entry
point.
