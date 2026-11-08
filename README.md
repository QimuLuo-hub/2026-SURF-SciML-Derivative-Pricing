# Scientific Machine Learning and Numerical Methods in Derivative Pricing

Summer Undergraduate Research Fellowship (SURF-2026-0159), Department of Foundational Mathematics, School of Mathematics and Physics.

Team: Zihan Liang, Qimu Luo, Xiaoya Wu, Ruixi Xu. Supervisors: Dr. Ren Zhao, Dr. Yiru Ye.

The project studies free-boundary American-option risk surfaces under the Black–Scholes model. It compares Crank–Nicolson LCP solvers (PSOR, Brennan–Schwartz projected LU), builds a high-accuracy DIRK–Policy-Iteration–sinh reference, and tests structure-aware PINNs and high-dimensional hybrid-VI networks. The final poster is in [`poster/`](poster/2026_SURF_poster.pdf).

This repository holds the **preliminary work** that the poster builds on. It goes step by step: a European call with a closed-form benchmark, then the obstacle problem as the simplest free-boundary problem, then the American put solved as an LCP.

## Repository layout

```
.
├── poster/                     Final SURF poster
├── reports/                    Weekly write-ups (theory, derivations, result tables)
├── 01_european_call/           Black–Scholes European call: FDM schemes and a PINN
├── 02_obstacle_problem/        1D / 2D obstacle problem solved with PSOR
└── 03_american_put/            American put: CN + PSOR (LCP) and a binomial-tree benchmark
```

Each code folder has a `results/` subfolder. The scripts write their figures there.

### `reports/`

| File | Content |
|---|---|
| `week0_preliminary_foundations.pdf` | Options, Black–Scholes PDE, European vs American, free boundary |
| `week3_research_directions.pdf` | Literature review and possible research directions |
| `week5_european_call_fdm_and_pinn.pdf` | Explicit / implicit / Crank–Nicolson FDM and DeepXDE PINN for the European call (code in `01_european_call/`) |
| `week6-8_free_boundary_and_american_put.pdf` | Obstacle problem, LCP formulation, PSOR, American put and free-boundary detection (code in `02_obstacle_problem/`, `03_american_put/`) |

### `01_european_call/`

European call, K = 100, T = 1, r = 0.05, σ = 0.2, S ∈ [0, 300]. The Black–Scholes closed form is the reference.

| Script | Method | Output |
|---|---|---|
| `fdm_explicit.py` | Explicit FDM, with a stability check | `results/fdm_explicit/` |
| `fdm_implicit.py` | Fully implicit FDM (tridiagonal solve) | `results/fdm_implicit/` |
| `fdm_crank_nicolson.py` | Crank–Nicolson FDM | `results/fdm_crank_nicolson/` |
| `pinn_deepxde.py` | PINN with DeepXDE (adapted from the DeepXDE Black–Scholes example) | `results/pinn/` |
| `plot_grid_schematic.py` | Figure of the (S, t) finite-difference grid | `results/grid_schematic.png` |

Each FDM script runs four studies: refining with M = N, fixing space and refining time, fixing time and refining space, and varying σ. It saves pointwise and overall error plots. The full console output is in `results/logs/`. At (M, N) = (400, 7006), all three schemes reach RMSE ≈ 1e-4 against the closed form (explicit 9.7e-5, implicit 1.6e-4, CN 1.2e-4).

### `02_obstacle_problem/`

The obstacle problem (elastic string over an obstacle) is the simplest free-boundary problem. Its LCP is solved with projected SOR.

| Script | Content |
|---|---|
| `psor_obstacle_1d.py` | 1D two-hump obstacle, PSOR solution and convergence history |
| `psor_obstacle_2d.py` | 2D two-hump obstacle on an 81×81 grid, 5-point Laplacian, PSOR |
| `plot_obstacle_schematic.py` | Exact schematic of the 1D obstacle problem (contact points A, B) |

The solver output, including complementarity checks, is in `results/*_output.txt`.

### `03_american_put/`

American put, E = 10, r = 0.10, σ = 0.40, with maturities of 3 and 6 months. This is the numerical example in Wilmott, Dewynne & Howison (1993).

| Script | Content |
|---|---|
| `psor_american_put.py` | Transform Black–Scholes to the heat equation, apply the θ-method (CN), solve the LCP by PSOR with an ω search, and detect the free boundary |
| `binomial_american_put.py` | Binomial tree (N = 4000) used as an independent benchmark |

The two methods agree to about 1e-4:

| S | Payoff | PSOR 3M | Binomial 3M | PSOR 6M | Binomial 6M |
|---|---|---|---|---|---|
| 8  | 2.0000 | 2.0202 | 2.0202 | 2.0953 | 2.0954 |
| 10 | 0.0000 | 0.6920 | 0.6923 | 0.9217 | 0.9219 |
| 12 | 0.0000 | 0.1712 | 0.1712 | 0.3624 | 0.3625 |
| 14 | 0.0000 | 0.0332 | 0.0331 | 0.1321 | 0.1322 |

At t = 0, the detected exercise boundary is S_f ≈ 7.56 for 3 months and S_f ≈ 7.12 for 6 months (`results/psor_free_boundary.png`).

## Running

```bash
pip install numpy scipy matplotlib        # FDM, PSOR, binomial
pip install deepxde torch                 # PINN only (any DeepXDE backend works)
python 01_european_call/fdm_crank_nicolson.py
```

Scripts can be run from any directory. Outputs always go to the `results/` folder next to the script.

## References

- P. Wilmott, J. Dewynne, S. Howison. *Option Pricing: Mathematical Models and Computation*. Oxford Financial Press, 1993.
- L. Lu, X. Meng, Z. Mao, G. E. Karniadakis. DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1):208–228, 2021.
- C. M. Elliott, J. R. Ockendon. *Weak and Variational Methods for Moving Boundary Problems*. Pitman, 1982.
- M. Raissi, P. Perdikaris, G. E. Karniadakis. Physics-informed neural networks. *J. Comput. Phys.*, 378:686–707, 2019.
