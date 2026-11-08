# Scientific Machine Learning and Numerical Methods in Derivative Pricing

This project develops and evaluates numerical methods and physics-informed neural networks (PINNs) for **free-boundary American-option risk surfaces** under the Black–Scholes framework. It also extends the approach to high-dimensional states with hybrid neural models.

Early exercise turns pricing into a variational-inequality (VI) problem. The option value, the exercise boundary and the price sensitivities must all be resolved consistently. The project:

- compares Crank–Nicolson (CN) linear complementarity problem (LCP) solvers;
- builds a high-accuracy reference that combines a diagonally implicit Runge–Kutta (DIRK) method, Policy Iteration and a sinh-mapped grid;
- studies low-dimensional obstacle PINNs and high-dimensional hybrid-VI models.

Main findings: Brennan–Schwartz projected LU is **26.0× faster** than projected SOR, and the Structure-aware PINN cuts the price RMSE **26.7×** compared with the Vanilla PINN. Structural enforcement improves neural accuracy. Recovering the free boundary reliably is still harder than approximating the price.

📄 Poster: [`poster/2026_SURF_poster.pdf`](poster/2026_SURF_poster.pdf)

## Model problem

For asset price $S$, time to maturity $\tau$, risk-free rate $r$, dividend yield $q$ and volatility $\sigma$:

$$\mathcal{L}_{BS}V = \tfrac12\sigma^2 S^2 V_{SS} + (r-q) S V_S - rV.$$

- **European options**: $V_\tau - \mathcal{L}_{BS}V = 0$, with $V(S,0) = \Phi(S)$. The European call has a closed form, which is used to validate the numerical methods.
- **American options**: early exercise adds the obstacle $V \ge \Phi$. The price then satisfies the variational inequality
  $$\min\big(V - \Phi,\; V_\tau - \mathcal{L}_{BS}V\big) = 0.$$
  The exercise region ($V = \Phi$) and the continuation region ($V_\tau - \mathcal{L}_{BS}V = 0$) are separated by the free boundary $S = S^*(\tau)$.
- **Exercise regimes**: for the American put, $(K-S)^+$, exercise happens at low $S$. For a dividend-paying call, $(S-K)^+$, it happens at high $S$. A call with no dividend ($q = 0$) is never exercised early, so $C_A = C_E$.
- **Key outputs**: $V(S,\tau)$, $S^*(\tau)$, $\Delta = V_S$ and $\Gamma = V_{SS}$.

## Methods

### Numerical PDE methods

| | Method | Idea |
|---|---|---|
| N1 | **CN–PSOR** | Classical benchmark. Crank–Nicolson turns the VI into one LCP per time step, solved by projected SOR, which applies a relaxed Gauss–Seidel update and projects it onto the payoff obstacle. |
| N1 | **CN–Brennan–Schwartz (projected LU)** | Exact LCP solver that uses an LU factorisation of the tridiagonal CN system and projected back-substitution. |
| N2 | **DIRK–Policy–sinh** | High-accuracy reference. It uses L-stable second-order DIRK time stepping ($\theta = 1 - 1/\sqrt2$), solves each stage LCP by Policy Iteration, and places grid points near the strike with a sinh map. |

### Physics-informed and hybrid neural methods

Let $G_\Theta = V_\Theta - \Phi$ and $R_\Theta = V_{\Theta,\tau} - \mathcal{L}V_\Theta$. Complementarity then reads $G_\Theta \ge 0$, $R_\Theta \ge 0$, $G_\Theta R_\Theta = 0$.

| | Model | Idea |
|---|---|---|
| P1 | **Vanilla soft-LCP PINN** | Learns $V_\Theta$ directly. The terminal, boundary, obstacle and complementarity conditions enter as soft penalties. |
| P1 | **Structure-aware ETC+FB PINN** | Uses $V_\Theta = V_{\text{lift}} + s N_\Theta$ to enforce the terminal condition exactly. Complementarity is written as $\phi_{FB}(G_\Theta, R_\Theta) = 0$ with the Fischer–Burmeister function. |
| P2 | **FC-HVI** | Fully connected network on the $d$-dimensional state $x$ and time $s$, with $d \in \{10, 20, 50\}$. Setting $V_\Theta = \Phi + s\,\mathrm{softplus}(N_\Theta)$ guarantees $V_\Theta \ge \Phi$. |
| P2 | **DeepSets-HVI** | Mean pooling $z = \frac1d\sum_i \phi_\Theta(x_i)$ makes the model invariant to asset order. |

## Key results

**R1. LCP solver efficiency.** Both solvers use the same CN discretisation and a residual tolerance of $10^{-12}$. Projected LU brings the median runtime down from $2.44\times10^{-1}$ s to $9.38\times10^{-3}$ s, which is **26.0× faster** than CN–PSOR.

**R2. High-accuracy reference.** The benchmark is a 50k-step binomial tree, for an American put with K = 1, T = 1, r = 0.05, q = 0.10, σ = 0.60.

| Method | Price RMSE | Max \|price error\| |
|---|---|---|
| CN + PSOR | 5.01e-5 | 2.71e-4 |
| **DIRK + Policy + sinh** | **2.70e-6** | **1.01e-5** |

The reference converges at observed orders of about 1.9–2.5 under both temporal and spatial refinement, which is consistent with second-order behaviour.

**R3. PINN evaluation.** Price RMSE against the high-accuracy reference, over S/K ∈ [0.5, 1.5] and τ/T ∈ [0.01, 1]:

| Model | Price RMSE |
|---|---|
| Vanilla PINN | 2.79e-2 |
| **Structure-aware PINN** | **1.05e-3** |
| FC-HVI (d = 50) | 2.31e-3 |
| DeepSets-HVI (d = 50) | 4.00e-3 |

**R4. Free-boundary recovery.** Against the binomial tree, DIRK lowers the boundary RMSE from 9.40e-3 (CN–PSOR) to 2.39e-3. The Vanilla soft-LCP PINN detects only 0.6% of boundary crossings, compared with 28.2% for ETC+FB.

## Conclusion

Method design matters for American-option risk surfaces. With a fixed CN discretisation, projected LU makes the LCP solver much faster. DIRK–Policy–sinh gives a high-accuracy reference that converges regularly under refinement. How well the neural models perform depends strongly on how they represent the free-boundary structure. ETC+FB improves greatly on the vanilla PINN, and the hybrid-VI models extend the approach to high-dimensional states. Even so, an accurate price does not guarantee an accurate free boundary. Future work will focus on recovering the boundary and the Greeks with neural models, and on matched validation for high-dimensional and dividend-paying call settings.

## Repository structure

```
.
├── poster/              Final project poster
├── reports/             Weekly reports
├── preliminary_work/    Groundwork: European call FDM/PINN, obstacle problem PSOR,
│                        American put CN+PSOR with a binomial benchmark (see its README)
└── ...                  Code and results for the methods above (to be added)
```

## References

1. Jaillet, P., Lamberton, D., and Lapeyre, B. (1990). Variational inequalities and the pricing of American options. *Acta Applicandae Mathematicae*, 21, 263–289.
2. Huang, J., and Pang, J.-S. (1998). Option pricing and linear complementarity. *Journal of Computational Finance*, 2(1), 31–60.
3. Alexander, R. (1977). Diagonally implicit Runge–Kutta methods for stiff O.D.E.'s. *SIAM Journal on Numerical Analysis*, 14(6), 1006–1021.
4. Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
5. Fischer, A. (1992). A special Newton-type optimization method. *Optimization*, 24(3–4), 269–284.
6. Zaheer, M., et al. (2017). Deep Sets. *Advances in Neural Information Processing Systems*, 30.
