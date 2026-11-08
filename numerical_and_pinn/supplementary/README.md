# Supplementary experiments

These archived studies provide context for the methods on the SURF poster.
Run the examples from `numerical_and_pinn/` after setting `PYTHONPATH` to
`$PWD/src`. Outputs go to `runs/` or the directory set by `SURF_OUTPUT_DIR`.

## Same-grid solver comparison

[Archived results](01_solver_benchmarks/results/) and
[original report](01_solver_benchmarks/reports/poster_evidence_report.md).
Run experiment 62 with `benchmark` to use the archived frozen penalty choice.
Use `validate` first if repeating the original validation phase, then
`benchmark`. Experiment 63 reads the archived R2 convergence evidence and
writes the reference audit. Experiment 64 produces the synthesis from the
new run.

## DIRK with Projected LU

[Archived results](02_reference_audits/dirk_projected_lu/results/) and
[original report](02_reference_audits/dirk_projected_lu/reports/dirk_projected_lu_substitution_report.md).
Experiment 66 uses the archived frozen gates by default. To repeat the
freeze from the R2 evidence, run experiment 65 first. Experiment 67 reads
the new audit outputs and writes the synthesis.

## Published DIRK-P

[Archived results](02_reference_audits/published_dirk_p/results/) and
[original report](02_reference_audits/published_dirk_p/reports/published_dirk_p_technical_report.md).
Run experiments 68, 69 and 70 in order to repeat the protocol, paper example
and candidate audit. Experiment 71 summarizes the full audit. Experiment 72
documents the historical correction to the residual gate and can re-score
an earlier audit copied into a separate run directory.
