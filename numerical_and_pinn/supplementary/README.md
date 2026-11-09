# Supplementary experiments

These studies provide context for the SURF poster. Each study includes its
results, scripts and run order.

| Study | What it checks |
| --- | --- |
| [01 Solver benchmarks](01_solver_benchmarks/) | Compares four solvers on the same Crank–Nicolson system in a separate run from the poster's R1 benchmark. |
| [02 DIRK with Projected LU](02_dirk_projected_lu/) | Tests replacing Policy Iteration with Projected LU inside the DIRK reference. |
| [03 Published DIRK-P](03_published_dirk_p/) | Reproduces a published example and compares the method against the poster's reference. |

Run the commands from `numerical_and_pinn/` after installing
[the dependencies](../requirements.txt) and setting `PYTHONPATH` to `$PWD/src`.
New outputs go to `runs/` by default. Set `SURF_OUTPUT_DIR` to use a separate
run directory.
