"""
Replicate the book-style American put computation using:

1. The Black–Scholes-to-heat-equation transformation
2. The theta-method in time
3. Crank–Nicolson (theta = 0.5)
4. PSOR to solve the linear complementarity problem at each time step
5. alpha = dt / dx^2 = 1, exactly as stated in the numerical example

Parameter setting taken from the numerical example in
Wilmott, Dewynne & Howison, Option Pricing: Mathematical Models and Computation (1993):
    r = 0.10
    sigma = 0.40
    E = 10
    alpha = 1
    theta = 0.5  (Crank–Nicolson)

We compute the American put values at time t = 0 for maturities:
    T = 3 months = 0.25 years
    T = 6 months = 0.50 years

The code follows the pseudo-code structure from the book:
    - initialize v with the transformed payoff at tau = 0
    - march forward in tau
    - at each step, build b and g
    - solve the discrete complementarity problem using PSOR

The output is a clean table with:
    Asset Price | Payoff Value | 3 Months American | 6 Months American
"""

from __future__ import annotations

import math
from dataclasses import dataclass


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class ModelParams:
    strike: float = 10.0
    rate: float = 0.10
    sigma: float = 0.40

    @property
    def k(self) -> float:
        """Dimensionless rate parameter k = 2r / sigma^2."""
        return 2.0 * self.rate / (self.sigma * self.sigma)


@dataclass(frozen=True)
class GridParams:
    dx: float = 0.025
    alpha: float = 1.0
    x_min: float = -6.0
    x_max: float = 6.0
    theta: float = 0.5  # Crank–Nicolson

    @property
    def dt(self) -> float:
        return self.alpha * self.dx * self.dx

    @property
    def n_half(self) -> int:
        """Half the number of spatial intervals, assuming a symmetric x-grid."""
        n = int(round(self.x_max / self.dx))
        if not math.isclose(self.x_min, -self.x_max, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("This script assumes a symmetric grid: x_min = -x_max.")
        return n


def transformed_payoff_american_put(x: np.ndarray, tau: float, params: ModelParams) -> np.ndarray:
    """
    Transformed obstacle/payoff for the American put in heat-equation variables.

    Original payoff: max(E - S, 0)
    Dimensionless payoff: max(1 - exp(x), 0), where x = ln(S / E)

    After the standard transformation:
        w(x, tau) = V(S,t) / E
        w = exp(a x + b tau) u
    with
        a = -(k - 1)/2,
        b = -(k + 1)^2 / 4,
    the transformed obstacle becomes
        u_payoff(x, tau) = exp((k - 1)x/2 + (k + 1)^2 tau/4) * max(1 - exp(x), 0).
    """
    k = params.k
    intrinsic = np.maximum(1.0 - np.exp(x), 0.0)
    factor = np.exp(0.5 * (k - 1.0) * x + 0.25 * (k + 1.0) ** 2 * tau)
    return factor * intrinsic


def transformed_to_original_price(u_value: float, x: float, tau: float, params: ModelParams) -> float:
    """
    Convert transformed solution u(x, tau) back to the original option price V(S,t).

        V = E * exp(-(k - 1)x/2 - (k + 1)^2 tau/4) * u
    """
    k = params.k
    factor = math.exp(-0.5 * (k - 1.0) * x - 0.25 * (k + 1.0) ** 2 * tau)
    return params.strike * factor * u_value


# --- Free boundary and transformation helpers ---

def transformed_to_original_time(tau: float, params: ModelParams, maturity_years: float) -> float:
    """Map transformed time tau back to calendar time t."""
    return maturity_years - 2.0 * tau / (params.sigma * params.sigma)


def transformed_x_to_asset_price(x_value: float, params: ModelParams) -> float:
    """Convert transformed spatial coordinate x back to asset price S."""
    return params.strike * math.exp(x_value)


# New helper: locate the American put free boundary directly on the computational x-grid.
def find_put_free_boundary_from_transformed_grid(
    x_grid: np.ndarray,
    u_grid: np.ndarray,
    tau: float,
    params: ModelParams,
    tol: float = 1.0e-6,
) -> float | None:
    """
    Detect the American put free boundary in original price space.

    Search from the strike side back into the exercise region, not from the far
    left boundary. This avoids being trapped by the imposed left boundary value.
    """
    s_grid = params.strike * np.exp(x_grid)
    payoff = np.maximum(params.strike - s_grid, 0.0)

    option_values = np.array([
        transformed_to_original_price(u_val, x_val, tau, params)
        for x_val, u_val in zip(x_grid, u_grid)
    ])
    premium = option_values - payoff

    # Only inspect the in-the-money side.
    itm = np.where((s_grid < params.strike) & (payoff > 0.0))[0]
    if len(itm) < 2:
        return None

    # Tolerance in original price units.
    contact_tol = max(tol, 1.0e-5 * params.strike)

    # Search from right to left: find the first contact point when moving
    # from continuation region back into exercise region.
    contact_idx = None
    continuation_idx = None

    for k in range(len(itm) - 1, -1, -1):
        i = itm[k]
        if premium[i] <= contact_tol:
            contact_idx = i
            if k < len(itm) - 1:
                continuation_idx = itm[k + 1]
            break

    if contact_idx is None:
        return None

    # If every ITM point is contact, just return the rightmost ITM point.
    if continuation_idx is None:
        return float(s_grid[contact_idx])

    s0 = float(s_grid[contact_idx])
    s1 = float(s_grid[continuation_idx])
    p0 = float(premium[contact_idx])
    p1 = float(premium[continuation_idx])

    # Linear interpolation of premium = 0 crossing.
    if math.isclose(p1, p0, rel_tol=0.0, abs_tol=1.0e-14):
        return s0

    s_f = s0 - p0 * (s1 - s0) / (p1 - p0)
    return float(s_f)


def psor_solve_tridiagonal(
    v_old: np.ndarray,
    rhs: np.ndarray,
    obstacle: np.ndarray,
    alpha: float,
    theta: float,
    omega: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int]:
    """
    Solve the tridiagonal LCP

        C v >= rhs,
        v >= obstacle,
        (v - obstacle) * (C v - rhs) = 0,

    where C is the Crank–Nicolson matrix with
        diagonal     = 1 + 2 alpha theta
        off-diagonal = - alpha theta

    using PSOR.
    """
    diag = 1.0 + 2.0 * alpha * theta
    off = -alpha * theta

    v = np.maximum(v_old.copy(), obstacle)
    n = len(v)

    for it in range(1, max_iter + 1):
        v_prev = v.copy()

        # Left interior unknown
        if n > 0:
            y = (rhs[0] - off * v_prev[1]) / diag if n > 1 else rhs[0] / diag
            candidate = v_prev[0] + omega * (y - v_prev[0])
            v[0] = max(obstacle[0], candidate)

        # Middle unknowns
        for i in range(1, n - 1):
            y = (rhs[i] - off * v[i - 1] - off * v_prev[i + 1]) / diag
            candidate = v_prev[i] + omega * (y - v_prev[i])
            v[i] = max(obstacle[i], candidate)

        # Right interior unknown
        if n > 1:
            y = (rhs[-1] - off * v[-2]) / diag
            candidate = v_prev[-1] + omega * (y - v_prev[-1])
            v[-1] = max(obstacle[-1], candidate)

        if np.linalg.norm(v - v_prev, ord=2) <= tol:
            return v, it

    raise RuntimeError("PSOR did not converge within max_iter.")


def american_put_heat_psor(
    maturity_years: float,
    params: ModelParams,
    grid: GridParams,
    omega: float = 1.4,
    tol: float = 1.0e-10,
    max_iter: int = 10000,
    track_free_boundary: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, list[tuple[float, float]]]:
    """
    Solve the transformed American put problem up to the specified maturity.

    Returns
    -------
    x : spatial grid in transformed coordinates
    v : transformed solution u(x, tau_final)
    tau_final : final transformed time
    free_boundary : list of (t, S_f(t)) pairs
    """
    sigma = params.sigma
    tau_final = 0.5 * sigma * sigma * maturity_years

    n_half = grid.n_half
    x = np.linspace(-n_half * grid.dx, n_half * grid.dx, 2 * n_half + 1)

    # We choose dt so that alpha = dt / dx^2 = 1 exactly.
    dt = grid.dt
    n_steps = int(round(tau_final / dt))
    if not math.isclose(n_steps * dt, tau_final, rel_tol=0.0, abs_tol=1e-14):
        raise ValueError(
            f"tau_final = {tau_final} is not an integer multiple of dt = {dt}. "
            "Adjust dx so that alpha = 1 and tau_final / dt is an integer."
        )

    # Initial value: transformed payoff at tau = 0
    v = transformed_payoff_american_put(x, tau=0.0, params=params)

    alpha = grid.alpha
    theta = grid.theta
    free_boundary: list[tuple[float, float]] = []

    # March forward in transformed time tau.
    for step in range(1, n_steps + 1):
        tau = step * dt

        g_new = transformed_payoff_american_put(x, tau=tau, params=params)
        g_old = transformed_payoff_american_put(x, tau=tau - dt, params=params)

        # Interior unknowns correspond to indices 1, 2, ..., len(x)-2.
        v_interior_old = v[1:-1]
        g_interior_new = g_new[1:-1]

        rhs = np.empty_like(v_interior_old)

        # General pattern for the interior points.
        rhs[:] = (
            v_interior_old
            + alpha * (1.0 - theta) * (v[2:] - 2.0 * v[1:-1] + v[:-2])
        )

        # Boundary corrections, exactly as in the book's discussion.
        # Left boundary x = x_min, right boundary x = x_max.
        rhs[0] += alpha * (1.0 - theta) * g_old[0] + alpha * theta * g_new[0]
        rhs[-1] += alpha * (1.0 - theta) * g_old[-1] + alpha * theta * g_new[-1]

        # Solve the LCP at the new time level.
        v_new_interior, _ = psor_solve_tridiagonal(
            v_old=v_interior_old,
            rhs=rhs,
            obstacle=g_interior_new,
            alpha=alpha,
            theta=theta,
            omega=omega,
            tol=tol,
            max_iter=max_iter,
        )

        v[0] = g_new[0]
        v[-1] = g_new[-1]
        v[1:-1] = v_new_interior

    if track_free_boundary:
        # Build the free boundary in the original price variable at each time level.
        v_track = transformed_payoff_american_put(x, tau=0.0, params=params)
        free_boundary = []

        for step in range(1, n_steps + 1):
            tau = step * dt
            g_new = transformed_payoff_american_put(x, tau=tau, params=params)
            g_old = transformed_payoff_american_put(x, tau=tau - dt, params=params)

            v_interior_old = v_track[1:-1]
            g_interior_new = g_new[1:-1]

            rhs = np.empty_like(v_interior_old)
            rhs[:] = (
                v_interior_old
                + alpha * (1.0 - theta) * (v_track[2:] - 2.0 * v_track[1:-1] + v_track[:-2])
            )
            rhs[0] += alpha * (1.0 - theta) * g_old[0] + alpha * theta * g_new[0]
            rhs[-1] += alpha * (1.0 - theta) * g_old[-1] + alpha * theta * g_new[-1]

            v_new_interior, _ = psor_solve_tridiagonal(
                v_old=v_interior_old,
                rhs=rhs,
                obstacle=g_interior_new,
                alpha=alpha,
                theta=theta,
                omega=omega,
                tol=tol,
                max_iter=max_iter,
            )

            v_track[0] = g_new[0]
            v_track[-1] = g_new[-1]
            v_track[1:-1] = v_new_interior

            s_f = find_put_free_boundary_from_transformed_grid(
                x_grid=x,
                u_grid=v_track,
                tau=tau,
                params=params,
                tol=1.0e-6,
            )
            if s_f is not None:
                t_now = transformed_to_original_time(tau, params, maturity_years)
                free_boundary.append((t_now, s_f))

    return x, v, tau_final, free_boundary


def american_price_from_transformed_solution(
    asset_price: float,
    x_grid: np.ndarray,
    u_grid: np.ndarray,
    tau_final: float,
    params: ModelParams,
) -> float:
    """
    Interpolate the transformed solution at a given asset price S and convert back
    to the original American put value.
    """
    if asset_price <= 0.0:
        # For S = 0, the American put is worth exactly E.
        return params.strike

    x_target = math.log(asset_price / params.strike)
    if x_target < x_grid[0] or x_target > x_grid[-1]:
        raise ValueError(
            f"Asset price S = {asset_price} maps to x = {x_target:.6f}, which is outside the x-grid."
        )

    u_target = float(np.interp(x_target, x_grid, u_grid))
    return transformed_to_original_price(u_target, x_target, tau_final, params)


def find_optimal_omega(params: ModelParams, grid: GridParams):
    omega_list = np.linspace(1.0, 1.9, 10)
    print("\nSearching optimal omega...")

    # Build one representative system
    x, v, tau, _ = american_put_heat_psor(0.25, params, grid)

    # take one interior snapshot
    v_old = v[1:-1]
    g = transformed_payoff_american_put(x, tau, params)[1:-1]

    alpha = grid.alpha
    theta = grid.theta

    rhs = (
        v_old
        + alpha * (1.0 - theta) * (v[2:] - 2.0 * v[1:-1] + v[:-2])
    )

    best_omega = None
    best_iter = float("inf")

    for omega in omega_list:
        try:
            _, iters = psor_solve_tridiagonal(
                v_old=v_old,
                rhs=rhs,
                obstacle=g,
                alpha=alpha,
                theta=theta,
                omega=omega,
                tol=1e-8,
                max_iter=5000,
            )

            print(f"omega={omega:.2f}, iterations={iters}")

            if iters < best_iter:
                best_iter = iters
                best_omega = omega

        except RuntimeError:
            print(f"omega={omega:.2f} did not converge")

    print(f"Best omega ≈ {best_omega:.2f}\n")
    return best_omega


def build_result_table(
    params: ModelParams,
    grid: GridParams,
    asset_prices: np.ndarray,
    omega: float,
) -> list[tuple[float, float, float, float]]:
    """Compute the final table rows for 3-month and 6-month American put values."""
    x_3m, u_3m, tau_3m, _ = american_put_heat_psor(
        maturity_years=0.25,
        params=params,
        grid=grid,
        omega=omega
    )
    x_6m, u_6m, tau_6m, _ = american_put_heat_psor(
        maturity_years=0.50,
        params=params,
        grid=grid,
        omega=omega
    )

    rows: list[tuple[float, float, float, float]] = []
    for s in asset_prices:
        payoff = max(params.strike - s, 0.0)
        amer_3m = american_price_from_transformed_solution(s, x_3m, u_3m, tau_3m, params)
        amer_6m = american_price_from_transformed_solution(s, x_6m, u_6m, tau_6m, params)
        rows.append((float(s), float(payoff), amer_3m, amer_6m))

    return rows


# --- Free boundary plotting ---

def plot_american_put_profiles_with_free_boundary(
    params: ModelParams,
    grid: GridParams,
    omega: float,
    asset_prices: np.ndarray,
) -> None:
    """
    Plot the American put value against stock price together with the payoff,
    and mark the free boundary for the 3-month and 6-month cases.
    """
    x_3m, u_3m, tau_3m, fb_3m = american_put_heat_psor(
        maturity_years=0.25,
        params=params,
        grid=grid,
        omega=omega,
        track_free_boundary=True,
    )
    x_6m, u_6m, tau_6m, fb_6m = american_put_heat_psor(
        maturity_years=0.50,
        params=params,
        grid=grid,
        omega=omega,
        track_free_boundary=True,
    )

    # Use a plotting grid directly in the original asset-price variable.
    s_min = params.strike * math.exp(x_3m[0])
    s_plot = np.linspace(s_min, max(asset_prices) + 1.0, 2000)

    # Convert transformed solutions back to original option values on the S-grid.
    v_3m = np.array([
        american_price_from_transformed_solution(s, x_3m, u_3m, tau_3m, params)
        for s in s_plot
    ])
    v_6m = np.array([
        american_price_from_transformed_solution(s, x_6m, u_6m, tau_6m, params)
        for s in s_plot
    ])

    # For display, extend the curves to S = 0 using the exact boundary value of the put.
    s_plot_full = np.concatenate(([0.0], s_plot))
    v_3m_full = np.concatenate(([params.strike], v_3m))
    v_6m_full = np.concatenate(([params.strike], v_6m))

    # Original payoff in S-variable.
    payoff = np.maximum(params.strike - s_plot_full, 0.0)

    # Use the final detected free boundary from the numerical solve itself.
    s_f_3m = fb_3m[-1][1] if fb_3m else None
    s_f_6m = fb_6m[-1][1] if fb_6m else None

    print(f"Detected 3M boundary at t=0: {s_f_3m}")
    print(f"Detected 6M boundary at t=0: {s_f_6m}")

    for s_test in [5.5, 6.0, 6.5, 7.0, 7.5, 8.0]:
        v3 = american_price_from_transformed_solution(s_test, x_3m, u_3m, tau_3m, params)
        v6 = american_price_from_transformed_solution(s_test, x_6m, u_6m, tau_6m, params)
        payoff_test = max(params.strike - s_test, 0.0)
        print(
            f"S={s_test:4.1f} | 3M premium={v3 - payoff_test:+.6e} | "
            f"6M premium={v6 - payoff_test:+.6e}"
        )

    # Interpolate option value at the free boundary for plotting markers.
    v_f_3m = float(np.interp(s_f_3m, s_plot_full, v_3m_full)) if s_f_3m is not None else None
    v_f_6m = float(np.interp(s_f_6m, s_plot_full, v_6m_full)) if s_f_6m is not None else None

    plt.figure(figsize=(8, 5), dpi=160)
    plt.plot(s_plot_full, payoff, linestyle="--", linewidth=2, label="Payoff")
    plt.plot(s_plot_full, v_3m_full, linewidth=2, label="3 Months American")
    plt.plot(s_plot_full, v_6m_full, linewidth=2, label="6 Months American")

    if s_f_3m is not None and v_f_3m is not None:
        plt.scatter([s_f_3m], [v_f_3m], s=50, marker="o", label=fr"3M exercise boundary: $S_f\approx {s_f_3m:.4f}$")

    if s_f_6m is not None and v_f_6m is not None:
        plt.scatter([s_f_6m], [v_f_6m], s=50, marker="x", label=fr"6M exercise boundary: $S_f\approx {s_f_6m:.4f}$")

    # Also show the reference stock prices used in the final table.
    for s in asset_prices:
        plt.axvline(s, linewidth=0.6, alpha=0.15)

    plt.xlim(0.0, max(s_plot))
    plt.ylim(bottom=0.0)
    plt.xlabel("Stock Price $S$")
    plt.ylabel("Option Value")
    # plt.title("American Put Value, Payoff, and Free Boundary at t = 0")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "psor_free_boundary.png", dpi=300)
    plt.show()

    if fb_3m:
        s_vals_3m = [s for _, s in fb_3m]
        print(
            f"3M free boundary: S_f(t=0) ≈ {s_vals_3m[-1]:.6f}, "
            f"S_f(t near maturity) ≈ {s_vals_3m[0]:.6f}"
        )

    if fb_6m:
        s_vals_6m = [s for _, s in fb_6m]
        print(
            f"6M free boundary: S_f(t=0) ≈ {s_vals_6m[-1]:.6f}, "
            f"S_f(t near maturity) ≈ {s_vals_6m[0]:.6f}"
        )


def print_result_table(rows: list[tuple[float, float, float, float]]) -> None:
    """Pretty-print the result table."""
    header = (
        f"{'Asset Price':>12}"
        f"{'Payoff Value':>16}"
        f"{'3 Months Amer.':>18}"
        f"{'6 Months Amer.':>18}"
    )
    print(header)
    print("-" * len(header))
    for s, payoff, a3, a6 in rows:
        print(f"{s:12.2f}{payoff:16.4f}{a3:18.4f}{a6:18.4f}")


def main() -> None:
    params = ModelParams(strike=10.0, rate=0.10, sigma=0.40)
    # grid = GridParams(dx=0.025, alpha=1.0, x_min=-6.0, x_max=6.0, theta=0.5)
    grid = GridParams(dx=0.01, alpha=1.0, x_min=-6.0, x_max=6.0, theta=0.5)#CN
    # grid = GridParams(dx=0.01, alpha=1.0, x_min=-6.0, x_max=6.0, theta=1)#implicit
    # grid = GridParams(dx=0.01, alpha=1.0, x_min=-6.0, x_max=6.0, theta=0)#explicit - explode

    # To match the book-style table, include S = 0 as well.
    asset_prices = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])

    print("Model parameters")
    print(f"  Strike E   = {params.strike}")
    print(f"  Rate r     = {params.rate}")
    print(f"  Sigma      = {params.sigma}")
    print(f"  Theta      = {grid.theta}  (0.5-Crank–Nicolson; 0-Ex; 1-Im)")
    print(f"  Alpha      = {grid.alpha}")
    print(f"  dx         = {grid.dx}")
    print(f"  dt         = {grid.dt}")
    print(f"  x-range    = [{grid.x_min}, {grid.x_max}]")
    print()

    optimal_omega = find_optimal_omega(params, grid)

    rows = build_result_table(params=params, grid=grid, asset_prices=asset_prices, omega=optimal_omega)
    print_result_table(rows)
    plot_american_put_profiles_with_free_boundary(
        params=params,
        grid=grid,
        omega=optimal_omega,
        asset_prices=asset_prices,
    )


if __name__ == "__main__":
    main()