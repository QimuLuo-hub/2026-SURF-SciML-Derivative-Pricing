import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def solve_obstacle_psor(
    x: np.ndarray,
    obstacle_fn,
    omega: float = 1.4,
    tol: float = 1e-10,
    max_iter: int = 100000,
    initial_guess: str = "obstacle",
    verbose: bool = True,
):
    """
    Solve the 1D obstacle problem using PSOR:

        -v_{i+1} + 2 v_i - v_{i-1} >= 0
        v_i >= f_i
        (v_i - f_i) * (-v_{i+1} + 2 v_i - v_{i-1}) = 0

    with boundary conditions:
        v(-1) = v(1) = 0

    Parameters
    ----------
    x : np.ndarray
        Full grid including endpoints, e.g. np.linspace(-1, 1, M).
    obstacle_fn : callable
        Function f(x) defining the obstacle.
    omega : float
        Relaxation parameter. Usually 0 < omega < 2.
        omega = 1 gives projected Gauss-Seidel.
    tol : float
        Stopping tolerance based on ||v^{k+1} - v^k||_2.
    max_iter : int
        Maximum number of PSOR iterations.
    initial_guess : str
        "obstacle" -> start from max(f, 0) in the interior
        "zero"     -> start from zero
    verbose : bool
        Whether to print iteration info.

    Returns
    -------
    x : np.ndarray
        Full grid including endpoints.
    v : np.ndarray
        Full solution including boundary values.
    f : np.ndarray
        Full obstacle values.
    info : dict
        Diagnostic information.
    """
    if not (0.0 < omega < 2.0):
        raise ValueError("omega must satisfy 0 < omega < 2")

    if x.ndim != 1 or len(x) < 3:
        raise ValueError("x must be a 1D grid with at least 3 points")

    # Full obstacle values (including endpoints)
    f_full = obstacle_fn(x)

    # Enforce the boundary values of the solution:
    # v[0] = v[-1] = 0
    # Interior unknowns are v[1:-1]
    n_interior = len(x) - 2
    f = f_full[1:-1].copy()

    # Initial guess
    if initial_guess == "obstacle":
        v_old = np.maximum(f, 0.0)
    elif initial_guess == "zero":
        v_old = np.zeros(n_interior)
    else:
        raise ValueError("initial_guess must be 'obstacle' or 'zero'")

    # To match the book's assumption, start with v^0 >= f
    v_old = np.maximum(v_old, f)

    # Boundary values
    left_bc = 0.0
    right_bc = 0.0

    converged = False
    num_iter = 0
    residual_history = []

    for k in range(max_iter):
        v_new = v_old.copy()

        # ---- Leftmost interior point ----
        # y_1 = (left_bc + v_old[1]) / 2
        if n_interior == 1:
            # Special case: only one interior node
            y = 0.5 * (left_bc + right_bc)
            candidate = v_old[0] + omega * (y - v_old[0])
            v_new[0] = max(f[0], candidate)
        else:
            y = 0.5 * (left_bc + v_old[1])
            candidate = v_old[0] + omega * (y - v_old[0])
            v_new[0] = max(f[0], candidate)

            # ---- Middle interior points ----
            for i in range(1, n_interior - 1):
                # Use updated left neighbor v_new[i-1]
                # and old right neighbor v_old[i+1]
                y = 0.5 * (v_new[i - 1] + v_old[i + 1])
                candidate = v_old[i] + omega * (y - v_old[i])
                v_new[i] = max(f[i], candidate)

            # ---- Rightmost interior point ----
            y = 0.5 * (v_new[-2] + right_bc)
            candidate = v_old[-1] + omega * (y - v_old[-1])
            v_new[-1] = max(f[-1], candidate)

        # Convergence test
        diff = np.linalg.norm(v_new - v_old)
        residual_history.append(diff)

        if verbose and (k < 5 or (k + 1) % 500 == 0):
            print(f"Iteration {k + 1:6d}, ||v_new - v_old||_2 = {diff:.3e}")

        if diff <= tol:
            converged = True
            num_iter = k + 1
            v_old = v_new
            break

        v_old = v_new

    if not converged:
        num_iter = max_iter
        if verbose:
            print("Warning: PSOR did not converge within max_iter.")

    # Reconstruct full solution including boundaries
    v_full = np.zeros_like(x)
    v_full[1:-1] = v_old

    # Simple complementarity check
    # discrete operator Bv = -v_{i+1} + 2v_i - v_{i-1}
    Bv = -v_full[2:] + 2.0 * v_full[1:-1] - v_full[:-2]

    comp_1 = np.min(v_full[1:-1] - f_full[1:-1])          # should be >= 0
    comp_2 = np.min(Bv)                                   # should be >= 0
    comp_3 = np.max(np.abs((v_full[1:-1] - f_full[1:-1]) * Bv))  # should be near 0

    info = {
        "converged": converged,
        "iterations": num_iter,
        "final_diff_norm": residual_history[-1] if residual_history else None,
        "min(v-f)": comp_1,
        "min(Bv)": comp_2,
        "max(|(v-f)*Bv|)": comp_3,
        "residual_history": residual_history,
    }

    return x, v_full, f_full, info


def two_hump_obstacle(x: np.ndarray) -> np.ndarray:
    """
    Example obstacle with two humps, zero near the endpoints.
    You can change this freely.
    """
    hump1 = 0.35 * np.exp(-((x + 0.55) / 0.18) ** 2)
    hump2 = 0.85 * np.exp(-((x - 0.20) / 0.23) ** 2)
    f = hump1 + hump2 - 0.02
    return np.maximum(f, 0.0)


def main():
    # Grid on [-1, 1]
    x = np.linspace(-1.0, 1.0, 401)

    # Solve
    x, v, f, info = solve_obstacle_psor(
        x=x,
        obstacle_fn=two_hump_obstacle,
        omega=1.6,
        tol=1e-10,
        max_iter=50000,
        initial_guess="obstacle",
        verbose=True,
    )

    print("\n--- Solver info ---")
    for key, val in info.items():
        if key != "residual_history":
            print(f"{key}: {val}")

    # Identify approximate contact set
    contact_tol = 1e-6
    contact = np.abs(v - f) < contact_tol

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.plot(x, v, linewidth=3, color="black", label="PSOR solution $v(x)$")
    ax.plot(x, f, linestyle="--", linewidth=2, color="red", label="Obstacle $f(x)$")

    # Highlight contact points (make visually distinct)
    ax.scatter(
        x[contact],
        v[contact],
        s=20,
        color="green",
        marker="x",
        label="Contact set (PSOR)"
    )

    ax.axhline(0.0, linewidth=1)
    # ax.set_title("1D Obstacle Problem Solved by PSOR")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$v(x)$")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "psor_obstacle_1d.png", dpi=300)
    plt.show()

    # Plot convergence history
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.semilogy(info["residual_history"])
    # ax.set_title("PSOR Convergence History")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|v^{k+1} - v^k\|_2$")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "psor_obstacle_1d_convergence.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()