# Same-grid solver comparison

This four-solver run gave a 30.2× ratio between the pooled median times for
CN–PSOR and Projected LU. The 26.0× result on the poster comes from the
[original R1 benchmark](../../01_cn_lcp_solvers/).

The [parameter selection](results/00_protocol/) and
[benchmark results](results/01_strict_solvers/) include the frozen Penalty/Newton
configuration, validation tables, raw timings and correctness checks. The
validation record was regenerated using the original selection rule after a
stale process overwrote its manifest. The 8,040 formal timing samples were
preserved. [Full original archive](https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces/tree/3aa2d821b9c92d095658782e7bcaf1322a90a7cb/method_extensions/results/14_poster_unified_comparison)

## Reproduce the study

Run from `numerical_and_pinn/` with `PYTHONPATH=$PWD/src`.

~~~sh
python supplementary/01_solver_benchmarks/62_strict_solver_poster_comparison.py benchmark
python supplementary/01_solver_benchmarks/63_accuracy_reference_evidence_audit.py
python supplementary/01_solver_benchmarks/64_poster_experiment_synthesis.py
~~~

The benchmark uses the archived penalty choice by default. Run experiment 62
with `validate` first to repeat the validation phase. Experiment 63 reads the
[main R2 results](../../02_dirk_reference/results/) to regenerate the Gamma
summaries below.

## Benchmark roles

- CN+PSOR: **Basic / Original Classical Benchmark**.
- CN+Policy Iteration: **Strengthened Classical Benchmark 1**.
- CN+Projected LU: **Strengthened Classical Benchmark 2**.
- DIRK+Policy+sinh: **High-Accuracy Numerical Reference**.
- Penalty/Newton: candidate comparator; correctness is judged before speed.

## Strict same-CN-LCP comparison

| Arm | Median (s) | P95 (s) | Common residual gate | Strong Policy-match certification | Max normalized LCP residual |
|---|---:|---:|---:|---:|---:|
| psor | 0.309707 | 1.43094 | 67/67 | 66/67 | 9.99992e-13 |
| policy_iteration | 0.0159997 | 0.0224767 | 67/67 | 67/67 | 6.47702e-16 |
| projected_lu_single | 0.0102586 | 0.0144603 | 67/67 | 67/67 | 6.29798e-16 |
| penalty_newton | 0.020795 | 0.0622017 | 40/67 | 40/67 | 2.17228e-11 |

Penalty/Newton decision: **FAILED_CORRECTNESS**.

Paired median Policy/PSOR ratio: 0.0557; Projected-LU/Policy ratio: 0.6776; Penalty/Policy ratio: 1.0067.
For the 31 put/dividend-call regimes, Projected-LU/Policy is 0.6582 and Penalty/Policy is 2.6403.

## Accuracy-reference evidence

Audit decision: **REUSE_EXISTING_REFERENCE_EVIDENCE**.

Temporal and spatial evidence are reported separately from online solver timing. Gamma claims remain restricted to the validated stable mask.

### Temporal summary

| Method | Stable fraction | Median Gamma max error |
|---|---:|---:|
| cn_quadratic | 1.000 | 1.65742e-06 |
| rannacher_cn_quadratic | 1.000 | 1.6565e-06 |
| dirk_lstable_quadratic | 1.000 | 6.94696e-07 |
| lobatto_iiic_penalty_quadratic | 0.917 | 3.71535e-06 |

### Spatial summary

| Grid | Stable fraction | Median Gamma max error |
|---|---:|---:|
| uniform | 1.000 | 0.000152088 |
| sinh_strike_concentrated | 0.917 | 2.59175e-05 |

Reference limitation: Fine DIRK+sinh refinement is an internal numerical reference, not an analytic American-option theorem or a speed benchmark.
