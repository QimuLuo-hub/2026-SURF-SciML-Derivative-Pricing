# Numerical methods and Vanilla PINN

This package collects Ruixi Xu's numerical experiments and Arm C (Vanilla
PINN) evidence for the SURF 2026 poster. [Poster PDF](../poster/2026_SURF_poster.pdf)

| Poster result | Code and archived evidence |
| --- | --- |
| **R1** Projected LU versus CN–PSOR | [01_cn_lcp_solvers](01_cn_lcp_solvers/) contains the benchmark entry point, the frozen solver choice, 6,030 raw measurements and the summary and correctness tables. The original pooled medians were 0.243615125 s for PSOR and 0.0093837915 s for Projected LU, a ratio of 25.961 (26.0 on the poster). |
| **R2** DIRK time and space refinement | [02_dirk_reference](02_dirk_reference/) contains both refinement studies. The poster's six rows are for `put_T025_s020_r001_q000`, using `value_rmse` and `gamma_rmse`. The [reference export](02_dirk_reference/results/reference/) is for the separate poster comparison contract, `put_T100_s060_r005_q010`. |
| **R3/R4** Vanilla PINN | [03_vanilla_pinn](03_vanilla_pinn/) contains the seed-101 training/scoring entry point, stored held-out metrics for 67 contracts, and the pointwise export for the single poster contract. Against the stored DIRK surface, its price RMSE is 0.0278823 and it finds 1 of 163 boundary points (0.6%). |

## Check the saved poster result

Run these commands from the `numerical_and_pinn` directory. Use Python
3.11 or 3.12 for a fresh installation of the pinned NumPy version.

~~~sh
python -m pip install -r requirements.txt
python 03_vanilla_pinn/evaluate_arm_c.py
~~~

For R1, use the median of `measured_total_seconds` for each arm in
[runtime_samples.csv](01_cn_lcp_solvers/results/runtime_samples.csv).
Divide the `psor` median by the `projected_lu_single` median. The correctness
table records the solver checks for the same contracts. For R2, use the
`dirk_lstable_quadratic` rows of the
[temporal table](02_dirk_reference/results/temporal_convergence.csv) at
M = 240 and N = 120, 240, 480, and the `sinh_strike_concentrated` rows of the
[spatial table](02_dirk_reference/results/spatial_convergence.csv) at
N = 960 and M = 120, 240, 480. These checks use archived results, so machine
speed and the original GPU environment do not affect them.

## Run the numerical experiments

~~~sh
export PYTHONPATH="$PWD/src"
python 01_cn_lcp_solvers/run_projected_lu_benchmark.py --output-dir runs/r1
python 02_dirk_reference/run_temporal_refinement.py --output-dir runs/refinement
python 02_dirk_reference/run_spatial_refinement.py --output-dir runs/refinement
python 02_dirk_reference/generate_dirk_reference.py --output-dir runs/dirk_poster_case
~~~

The spatial study reads the temporal decision from the same output directory.
To check the setup quickly, add `--regime-limit 1` to the three study commands
and `--warmups 0 --repeats 1` to R1. These reduced runs check execution only.
Elapsed times depend on the machine.

[inputs/v1_small_grid](inputs/v1_small_grid/) holds the original contract
and split tables. The held-out split contains the same 67 contracts as the
archived R1 measurements. [inputs/tolerance_decision.json](inputs/tolerance_decision.json)
records the frozen tolerance used by the refinement studies and audits.
The experiment scripts use the implementations in [src](src/).

## Train, score or export Arm C

Install a platform-appropriate PyTorch build for this section. The original
seed-101 training used CUDA. For a fresh run, choose a new output directory
and run these phases in order:

~~~sh
export PYTHONPATH="$PWD/src"
python 03_vanilla_pinn/run_arm_c.py train --device cuda --max-seconds 14400 --output-dir runs/arm_c_seed101
python 03_vanilla_pinn/run_arm_c.py reference --output-dir runs/arm_c_seed101
python 03_vanilla_pinn/run_arm_c.py score --device cpu --output-dir runs/arm_c_seed101
~~~

The full `score` phase reads `inputs/v1_small_grid/dataset_v1_small_grid.npz`.
Download the [frozen dataset file](https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces/blob/876ec0bc69c4d8db24c3e4710e448e0cfd161832/results/04_surrogate_dataset/v1_small_grid/dataset_v1_small_grid.npz)
and place it at that path. Its SHA-256 is
`bc467831347ff73d28b631bf0a35dd5a563f46666d5af5d3ce04f192df48dc8b`.
The physics-only training phase reads the contract table. The saved poster
metric can be checked with the smaller exports already in this package.

To regenerate the compact poster export, extract the Arm C checkpoint and its
matching status JSON for `put_T100_s060_r005_q010` from the
[seed-101 release archive](https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces/releases/tag/arm-c-seed101-followup-v1).
Run `generate_dirk_reference.py` as above to get the native reference file,
then pass all three files to the export script.

~~~sh
python 03_vanilla_pinn/export_arm_c_predictions.py \
  --arm-c-checkpoint /path/to/C_put_T100_s060_r005_q010_seed101_best.pt \
  --arm-c-status /path/to/C_put_T100_s060_r005_q010_seed101_status.json \
  --reference runs/dirk_poster_case/put_T100_s060_r005_q010.npz \
  --source-git-commit "$(git rev-parse HEAD)" \
  --output-dir runs/arm_c_poster_export
~~~

## Supplementary studies

- [01_solver_benchmarks](supplementary/01_solver_benchmarks/) holds the separate
  30.2× rerun and the Policy Iteration and Penalty/Newton comparison. The
  poster's 26.0× is from R1 above.
- [DIRK with Projected LU](supplementary/02_reference_audits/dirk_projected_lu/)
  checks replacing Policy Iteration inside DIRK. It retains the original
  DIRK–Policy–sinh reference.
- [Published DIRK-P](supplementary/02_reference_audits/published_dirk_p/)
  reproduces and audits a published alternative. Its archived evidence
  includes the corrected residual gate and the earlier scoring record.

The supplementary scripts write new outputs under `runs/` by default. Set
`SURF_OUTPUT_DIR` to use a different location. The original plots and reports
are included with their archived tables. See
[supplementary/README.md](supplementary/README.md) for run order.

## Provenance and scope

- The original R1 benchmark is experiment 60 and the refinement studies are
  experiments 23 and 26 in
  [c741e6e](https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces/tree/c741e6e40efc18c233eb4dd8504c29c7095be536).
  Arm C training and scoring are adapted from experiment 33b at the same
  commit. The separate 30.2× study entered via
  [3aa2d82](https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces/commit/3aa2d821b9c92d095658782e7bcaf1322a90a7cb),
  and the two DIRK audits via
  [e31a794](https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces/commit/e31a794e1460fc9ed0c50eec86730ede3c782e33).
- The stored DIRK and Arm C pointwise files come from
  [63ce7cf](https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces/tree/63ce7cf6ffe04a778dc2f3b0ce691a26a1aca5c6/results/visualization_exports/american_put_matched/put_T100_s060_r005_q010).
  The compact exports retain the original plot coordinates and predictions.
- The two small data tables are the shared input from the original v1 study.
  Archived CSVs and JSON records preserve their upstream bytes, including
  historical paths in their metadata. The standalone code uses the package
  paths described above.
- The main timing, refinement and Arm C runs retain the original numerical
  solver versions. The later DIRK audits use their separate variants in
  `src/american_risk_surfaces/solvers/reference_audits/`.
- The original project used Codex-assisted development and experiment
  orchestration as recorded in the
  [upstream provenance note](https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces/blob/c741e6e40efc18c233eb4dd8504c29c7095be536/method_extensions/PROVENANCE.md).
  The team's Arm D, 50,000-step tree and high-dimensional HVI contributions
  are separate from this experiment package.
