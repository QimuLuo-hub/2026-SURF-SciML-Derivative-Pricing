"""
2D obstacle problem solved by PSOR.

We solve the complementarity system on a rectangular grid:

    -Delta_h u >= 0,
    u >= f,
    (u - f) * (-Delta_h u) = 0,

with homogeneous Dirichlet boundary conditions:

    u = 0 on the boundary.

The discrete Laplacian uses the standard 5-point stencil:

    -Delta_h u[i,j] = (4*u[i,j] - u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i,j+1]) / h^2

PSOR update at each interior point:

    y = 0.25 * (u_left_new + u_right_old + u_down_new + u_up_old)
    u_new = max(f, u_old + omega * (y - u_old))

This is the 2D analogue of the 1D obstacle-problem PSOR iteration.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def obstacle_two_humps(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Example 2D obstacle surface with two smooth humps.
    The obstacle is clipped below at zero so that the contact set is easier to see.
    """
    hump1 = 0.45 * np.exp(-(((x + 0.45) / 0.22) ** 2 + ((y + 0.10) / 0.28) ** 2))
    hump2 = 0.85 * np.exp(-(((x - 0.20) / 0.26) ** 2 + ((y - 0.15) / 0.22) ** 2))
    f = hump1 + hump2 - 0.02
    return np.maximum(f, 0.0)


def solve_obstacle_psor_2d(
    nx: int = 81,
    ny: int = 81,
    xlim: tuple[float, float] = (-1.0, 1.0),
    ylim: tuple[float, float] = (-1.0, 1.0),
    obstacle_fn=obstacle_two_humps,
    omega: float = 1.7,
    tol: float = 1e-8,
    max_iter: int = 50000,
    initial_guess: str = "obstacle",
    verbose: bool = True,
):
    """
    Solve the 2D obstacle problem using PSOR.

    Parameters
    ----------
    nx, ny : int
        Number of grid points in x and y directions, including boundaries.
    xlim, ylim : tuple[float, float]
        Domain bounds.
    obstacle_fn : callable
        Function obstacle_fn(X, Y) returning the obstacle values on the full grid.
    omega : float
        Relaxation parameter. Must satisfy 0 < omega < 2.
    tol : float
        Convergence tolerance based on ||u^{k+1} - u^k||_2.
    max_iter : int
        Maximum PSOR iterations.
    initial_guess : str
        "obstacle" or "zero".
    verbose : bool
        Print iteration progress.

    Returns
    -------
    X, Y : np.ndarray
        Full meshgrid arrays.
    u : np.ndarray
        Numerical solution on the full grid.
    f : np.ndarray
        Obstacle on the full grid.
    info : dict
        Convergence and complementarity diagnostics.
    """
    if not (0.0 < omega < 2.0):
        raise ValueError("omega must satisfy 0 < omega < 2")

    if nx < 3 or ny < 3:
        raise ValueError("nx and ny must both be at least 3")

    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    hx = x[1] - x[0]
    hy = y[1] - y[0]

    if not np.isclose(hx, hy):
        raise ValueError("This implementation assumes a uniform square grid: hx must equal hy.")

    h = hx
    X, Y = np.meshgrid(x, y, indexing="ij")
    f = obstacle_fn(X, Y)

    # Homogeneous Dirichlet boundary conditions: u = 0 on boundary.
    u = np.zeros((nx, ny), dtype=float)

    if initial_guess == "obstacle":
        u[1:-1, 1:-1] = np.maximum(f[1:-1, 1:-1], 0.0)
    elif initial_guess == "zero":
        u[1:-1, 1:-1] = 0.0
    else:
        raise ValueError("initial_guess must be 'obstacle' or 'zero'")

    # Ensure feasibility of the initial guess.
    u[1:-1, 1:-1] = np.maximum(u[1:-1, 1:-1], f[1:-1, 1:-1])

    residual_history: list[float] = []
    converged = False

    for k in range(max_iter):
        u_old = u.copy()

        # Gauss-Seidel ordering: sweep through the interior points.
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                # Use newly updated values on the "left/down" side
                # and old values on the "right/up" side.
                y_ij = 0.25 * (
                    u[i - 1, j] +        # new
                    u_old[i + 1, j] +    # old
                    u[i, j - 1] +        # new
                    u_old[i, j + 1]      # old
                )

                candidate = u_old[i, j] + omega * (y_ij - u_old[i, j])
                u[i, j] = max(f[i, j], candidate)

        diff = np.linalg.norm(u - u_old)
        residual_history.append(diff)

        if verbose and (k < 5 or (k + 1) % 500 == 0):
            print(f"Iteration {k + 1:6d}, ||u_new - u_old||_2 = {diff:.3e}")

        if diff <= tol:
            converged = True
            break

    # Complementarity diagnostics.
    # Discrete operator: -Delta_h u
    B_u = (
        4.0 * u[1:-1, 1:-1]
        - u[:-2, 1:-1]
        - u[2:, 1:-1]
        - u[1:-1, :-2]
        - u[1:-1, 2:]
    ) / (h * h)

    gap = u[1:-1, 1:-1] - f[1:-1, 1:-1]

    info = {
        "converged": converged,
        "iterations": k + 1,
        "final_diff_norm": residual_history[-1] if residual_history else None,
        "min(u-f)": float(np.min(gap)),
        "min(-Delta_h u)": float(np.min(B_u)),
        "max(|(u-f)*(-Delta_h u)|)": float(np.max(np.abs(gap * B_u))),
        "residual_history": residual_history,
        "grid_shape": u.shape,
    }

    return X, Y, u, f, info


def plot_solution(X: np.ndarray, Y: np.ndarray, u: np.ndarray, f: np.ndarray) -> None:
    """Create clean illustration plots for the 2D obstacle problem."""
    contact_tol = 1e-5
    contact = np.abs(u - f) < contact_tol

    # 3D surface plot
    fig = plt.figure(figsize=(10, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, u, rstride=1, cstride=1, alpha=0.90, cmap=cm.jet, linewidth=0)
    ax.plot_surface(X, Y, f, rstride=2, cstride=2, alpha=0.35, cmap=cm.plasma, linewidth=0)
    # ax.set_title("2D Obstacle Problem: PSOR Solution and Obstacle")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("height")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "psor_2d_obstacle_sol.png", dpi=300)
    plt.show()

    # Top view with contact set
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    contour_u = ax.contourf(X, Y, u, levels=30)
    plt.colorbar(contour_u, ax=ax, label="solution u")
    ax.contour(X, Y, f, levels=12, colors="white", linewidths=0.8)
    ax.contour(X, Y, contact.astype(float), levels=[0.5], colors="red", linewidths=2.0)
    # ax.set_title("Top View: Solution with Obstacle Contours and Contact Boundary")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "psor_2d_obstacle_top.png", dpi=300)
    plt.show()

    # Center cross-sections
    i_mid = X.shape[0] // 2
    j_mid = X.shape[1] // 2

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
    axes[0].plot(X[:, j_mid], u[:, j_mid], linewidth=2.5, label="solution u")
    axes[0].plot(X[:, j_mid], f[:, j_mid], "--", linewidth=2.0, label="obstacle f")
    axes[0].set_title("Cross-section at middle y")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("height")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(Y[i_mid, :], u[i_mid, :], linewidth=2.5, label="solution u")
    axes[1].plot(Y[i_mid, :], f[i_mid, :], "--", linewidth=2.0, label="obstacle f")
    axes[1].set_title("Cross-section at middle x")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("height")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.show()

    # Constrained gap illustration
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    gap = u - f
    contour_gap = ax.contourf(X, Y, gap, levels=30)
    plt.colorbar(contour_gap, ax=ax, label="u - f")
    ax.contour(X, Y, contact.astype(float), levels=[0.5], colors="black", linewidths=2.0)
    ax.set_title("Gap to Obstacle: u - f")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def plot_convergence(info: dict) -> None:
    """Plot the PSOR convergence history."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.semilogy(info["residual_history"])
    # ax.set_title("PSOR Convergence History (2D Obstacle Problem)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|u^{k+1} - u^k\|_2$")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "psor_2d_obstacle_convergence.png", dpi=300)
    plt.show()


def main() -> None:
    X, Y, u, f, info = solve_obstacle_psor_2d(
        nx=81,
        ny=81,
        omega=1.7,
        tol=1e-8,
        max_iter=30000,
        obstacle_fn=obstacle_two_humps,
        initial_guess="obstacle",
        verbose=True,
    )

    print("\n--- Solver info ---")
    for key, value in info.items():
        if key != "residual_history":
            print(f"{key}: {value}")

    plot_solution(X, Y, u, f)
    plot_convergence(info)


if __name__ == "__main__":
    main()