# Binomial benchmark, structure-aware PINN and high-dimensional HVI

This package collects Zihan Liang's parts of the SURF 2026 poster: the
50,000-step binomial-tree benchmark, the structure-aware ETC+FB PINN (Arm D),
the high-dimensional FC-HVI and DeepSets-HVI models, the shared-reference
comparison behind the R3 table, and the R3 figure.
[Poster PDF](../poster/2026_SURF_poster.pdf)

It reuses the DIRK reference, the Arm C export and the PINN library from
[`numerical_and_pinn`](../numerical_and_pinn/) instead of copying them.

| Poster result | Code and archived evidence |
| --- | --- |
| **R2** table, CN+PSOR vs DIRK against the 50k-step tree | [01_crr50000_benchmark](01_crr50000_benchmark/) holds the tree and its convergence check. [02_crr_cross_check](02_crr_cross_check/) scores both solvers against it: price RMSE 5.01e-5 vs 2.70e-6 (18.5×), max error 2.71e-4 vs 1.01e-5. |
| **R4** boundary RMSE, CN+PSOR vs DIRK | Same check: 9.40e-3 vs 2.39e-3 against the tree, over 163 time points. |
| **P1/R3/R4** Structure-aware PINN (Arm D) | [03_structure_aware_pinn](03_structure_aware_pinn/) contains the seed-101 entry point, its frozen configuration, held-out metrics for 67 contracts and the poster-case export. Against the stored DIRK surface its price RMSE is 1.045e-3, and it finds 46 of 163 boundary points (28.2%). |
| **P2/R3** FC-HVI and DeepSets-HVI | [04_high_dimensional_hvi](04_high_dimensional_hvi/) contains the models, the final boundary-complete experiment for d = 10, 20, 50, its tests and its archived results. |
| **R3** table and 26.7× | [05_shared_reference_comparison](05_shared_reference_comparison/) rescores every model against the same DIRK reference on S/K ∈ [0.5, 1.5], τ/T ∈ [0.01, 1]. |
| **R3** figure | [06_poster_figure](06_poster_figure/) draws the τ/T = 0.5 price slice from the stored surfaces. |

## Check the saved poster results

Run from this directory with Python 3.11 or 3.12. These checks read archived
results and need only NumPy and Matplotlib; the R2/R4 check re-solves the
coarse CN+PSOR grid, which takes about a second.

~~~sh
python -m pip install -r requirements.txt
python 02_crr_cross_check/check_against_crr50000.py
python 03_structure_aware_pinn/evaluate_arm_d.py
python 05_shared_reference_comparison/evaluate_r3_table.py
python 06_poster_figure/plot_r3_price_slice.py --no-show
~~~

Expected output:

| Quantity | Value | Poster |
| --- | --- | --- |
| CN+PSOR price RMSE / max error vs tree | 5.0081e-5 / 2.7056e-4 | 5.01e-5 / 2.71e-4 |
| DIRK price RMSE / max error vs tree | 2.7009e-6 / 1.0128e-5 | 2.70e-6 / 1.01e-5 |
| CN+PSOR / DIRK price RMSE | 18.54 | 18.5× |
| Boundary RMSE vs tree, CN+PSOR / DIRK | 9.4037e-3 / 2.3945e-3 | 9.40e-3 / 2.39e-3 |
| Arm D price RMSE, boundary points found | 1.0451e-3, 46/163 | 1.05e-3, 28.2% |
| FC-HVI / DeepSets-HVI price RMSE, d = 50 | 2.3062e-3 / 4.0005e-3 | 2.31e-3 / 4.00e-3 |
| Arm C / Arm D price RMSE | 26.678 | 26.7× |

The figure is written to `runs/r3_figure/`. The image used on the poster is kept
in [06_poster_figure/results](06_poster_figure/results/). Both show the same
curves; the poster copy was exported at a lower resolution.

## Rerun the experiments

**50,000-step tree.** The two scripts in `01_crr50000_benchmark` need only
NumPy and write to `./results/` under the current directory. The 50,000-step
export took about 40 minutes on a 10-core laptop with 4 workers; both scripts
resume from their ledgers if interrupted.

~~~sh
cd 01_crr50000_benchmark
python binomial_benchmark_convergence.py   # step-doubling check up to 50,000
python binomial_benchmark_50000.py         # price slices and free boundary
~~~

**Arm D.** Arm D uses the PINN library in `numerical_and_pinn/src`, which
already contains the ETC+FB formulation. Install PyTorch, place the frozen
dataset described in the
[numerical_and_pinn README](../numerical_and_pinn/README.md#train-score-or-export-arm-c),
and run the phases in order:

~~~sh
export PYTHONPATH="$PWD/../numerical_and_pinn/src"
python 03_structure_aware_pinn/run_arm_d.py train --device cuda --max-seconds 14400 --output-dir runs/arm_d_seed101
python 03_structure_aware_pinn/run_arm_d.py reference --output-dir runs/arm_d_seed101
python 03_structure_aware_pinn/run_arm_d.py score --device cpu --output-dir runs/arm_d_seed101
~~~

**High-dimensional HVI.** The original runs used an RTX 4070 Laptop GPU on
Windows (`environment_manifest.json`). Run from `04_high_dimensional_hvi`:

~~~sh
cd 04_high_dimensional_hvi
export PYTHONPATH="$PWD/src:$PWD/experiments"
python -m pytest tests -q
python experiments/hd3_matched_american_put_boundary_complete.py --phase status
python experiments/hd3_matched_american_put_boundary_complete.py --phase preflight
python experiments/hd3_matched_american_put_boundary_complete.py --phase canonical
python experiments/hd3_matched_american_put_boundary_complete.py --phase finalize
~~~

`preflight`, `canonical` and `finalize` write into `results/`, so copy this
folder elsewhere before rerunning to keep the archived results intact.

## Scope

- One contract: American put with K = 1, T = 1, r = 0.05, q = 0.10, σ = 0.60.
- The 50,000-step tree is a converged numerical benchmark, not an exact
  solution. Going from 32,768 to 50,000 steps changed the checked prices by an
  RMSE of 9.5e-7.
- Arm D is a single-seed pilot. Seed 101 was chosen before held-out scoring
  because it had the lowest Arm D validation median price RMSE; it is not a
  seed-robustness estimate.
- The HVI models are trained with oracle labels from an internal CN/PSOR
  solve and are then scored against the DIRK reference. Their d-dimensional
  contract is a geometric basket that reduces exactly to the one-dimensional
  put above, so the comparison is fair on that contract only. All six HVI runs
  use seed 101.
- None of the six HVI models recovers a free boundary at the 1e-6 premium
  threshold, although their prices are accurate to about 2–4e-3.
- The R3 RMSEs compare predictive accuracy on a shared reference and domain.
  They are not a comparison of training cost, because the low-dimensional and
  high-dimensional models use different training protocols.

## Provenance

- The tree scripts and outputs come from a local run. The 50,000-step script
  matches the `script_sha256` in
  [its manifest](01_crr50000_benchmark/results/crr50000/manifest.json), and every
  output matches [checksums.json](01_crr50000_benchmark/results/crr50000/checksums.json).
- The cross-check tables in `02_crr_cross_check/results` come from
  [zihan-liang/american-option-risk-surfaces@f40fd05](https://github.com/zihan-liang/american-option-risk-surfaces/commit/f40fd05c843f6a0522396d1e4349493d1f72914d).
- The Arm D code and runs were stored in the shared repository
  [Ruixixu-hub/2026-surf-american-risk-surfaces](https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces):
  the entry point is adapted from `experiments/33a_arm_d_seed101_pilot.py`, the
  held-out results come from `results/08_pinn_gap/04_heldout_pilots/arm_d_seed101`,
  and the poster-case export from `results/visualization_exports/american_put_matched/`,
  all at [63ce7cf](https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces/tree/63ce7cf6ffe04a778dc2f3b0ce691a26a1aca5c6).
  `arm_d_config.json` records the SHA-256 of the frozen configuration it was
  taken from. The adapted entry point only changes where the configuration is
  read and where outputs are written, in the same way as `run_arm_c.py`.
- The HVI code, configuration and results come from tag
  [`archive/exp/hd3-matched-american-put-final-boundary-complete`](https://github.com/zihan-liang/american-option-risk-surfaces/tree/c11c3ec3c22a5304b33f394c0ecdb1f6225cb6e1)
  (c11c3ec). The run recorded implementation commit `e4e263a`. Its recorded
  `config_sha256` is for the same configuration file saved with Windows line
  endings, so `--phase status` reports a different hash on other systems.
- The shared-reference RMSEs were first computed by
  `analysis/build_consolidated_experiment_data.py` in Zihan's repository; the
  stored tables in `05_shared_reference_comparison/results` come from
  [5bcd15e](https://github.com/zihan-liang/american-option-risk-surfaces/commit/5bcd15e16a6e8f175936bd824ad61b18b020cccb).
  `evaluate_r3_table.py` repeats that calculation directly from the surfaces.
- The DIRK reference and Arm C export used here are byte-identical to the
  `63ce7cf` files used in the original comparison.
- The project used Codex-assisted development and experiment orchestration.
