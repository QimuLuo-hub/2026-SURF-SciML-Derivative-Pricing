# Preliminary Work

This is the groundwork done before the main project. It moves step by step towards the American free-boundary problem:

1. **European call**: finite-difference schemes and a PINN, checked against the closed-form Black–Scholes price.
2. **Obstacle problem**: the simplest free-boundary problem, written as a linear complementarity problem (LCP) and solved with projected SOR (PSOR).
3. **American put**: the Crank–Nicolson + PSOR LCP solver from step 2, applied to the American put. A binomial tree serves as the benchmark.

The derivations and result tables are in the weekly reports in [`../reports/`](../reports). `week5` covers part 1, and `week6-8` covers parts 2 and 3.

Each subfolder has a `results/` folder. The scripts write their output there, whatever directory they are run from.

## `01_european_call/`

European call, K = 100, T = 1, r = 0.05, σ = 0.2, S ∈ [0, 300]. The Black–Scholes closed form is the reference.

| Script | Method | Output |
|---|---|---|
| `fdm_explicit.py` | Explicit FDM, with a stability check | `results/fdm_explicit/` |
| `fdm_implicit.py` | Fully implicit FDM (tridiagonal solve) | `results/fdm_implicit/` |
| `fdm_crank_nicolson.py` | Crank–Nicolson FDM | `results/fdm_crank_nicolson/` |
| `pinn_deepxde.py` | PINN with DeepXDE (adapted from the DeepXDE Black–Scholes example) | `results/pinn/` |
| `plot_grid_schematic.py` | Figure of the (S, t) finite-difference grid | `results/grid_schematic.png` |

Each FDM script runs four studies: refining with M = N, fixing space and refining time, fixing time and refining space, and varying σ. The full console output is in `results/logs/`. At (M, N) = (400, 7006), all three schemes reach RMSE ≈ 1e-4 against the closed form (explicit 9.7e-5, implicit 1.6e-4, CN 1.2e-4).

## `02_obstacle_problem/`

| Script | Content |
|---|---|
| `psor_obstacle_1d.py` | 1D two-hump obstacle, PSOR solution and convergence history |
| `psor_obstacle_2d.py` | 2D two-hump obstacle on an 81×81 grid, 5-point Laplacian, PSOR |
| `plot_obstacle_schematic.py` | Exact schematic of the 1D obstacle problem (contact points A, B) |

The solver output, including complementarity checks, is in `results/*_output.txt`.

## `03_american_put/`

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
python preliminary_work/01_european_call/fdm_crank_nicolson.py
```

## References

- P. Wilmott, J. Dewynne, S. Howison. *Option Pricing: Mathematical Models and Computation*. Oxford Financial Press, 1993.
- L. Lu, X. Meng, Z. Mao, G. E. Karniadakis. DeepXDE: A deep learning library for solving differential equations. *SIAM Review*, 63(1):208–228, 2021.
- C. M. Elliott, J. R. Ockendon. *Weak and Variational Methods for Moving Boundary Problems*. Pitman, 1982.
