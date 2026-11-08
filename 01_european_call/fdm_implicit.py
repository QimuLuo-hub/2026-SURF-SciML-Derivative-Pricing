import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
from pathlib import Path


#The analytical solution:Black-Scholes closed-form price of a European call.

def bs_call_price(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float) #    S can be a scalar or a numpy array.

    # Avoid log(0) warning for S=0
    price = np.zeros_like(S)

    positive = S > 0
    S_pos = S[positive]

    d1 = (np.log(S_pos / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price[positive] = S_pos * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return price


#Smax : float, Maximum asset price in truncated spatial domain
# M : int, Number of spatial steps
def european_call_implicit_fdm(K, T, r, sigma, Smax, M, N):
    """
    Fully implicit finite difference solver for the Black-Scholes PDE
    for a European call option.

    Returns
    -------
    S_grid : ndarray
        Asset price grid
    V : ndarray
        Option values at time t=0 on the grid
    """
    dS = Smax / M
    dt = T / N

    S_grid = np.linspace(0, Smax, M + 1)
    t_grid = np.linspace(0, T, N + 1)

    # Solution array
    # V[i, n] would mean asset grid index i and time index n
    # To save memory, we only keep one time slice at a time
    V = np.maximum(S_grid - K, 0.0)   # terminal condition at t = T

    # Interior nodes are i = 1, 2, ..., M-1
    i = np.arange(1, M)

    # Coefficients for the implicit scheme
    a = -0.5 * dt * (sigma**2 * i**2 - r * i)
    b = 1.0 + dt * (sigma**2 * i**2 + r)
    c = -0.5 * dt * (sigma**2 * i**2 + r * i)

    # Build banded matrix for scipy.linalg.solve_banded
    # For a tridiagonal system:
    # ab[0,1:] = upper diagonal
    # ab[1,:]  = main diagonal
    # ab[2,:-1]= lower diagonal
    ab = np.zeros((3, M - 1))
    ab[0, 1:] = c[:-1]
    ab[1, :] = b
    ab[2, :-1] = a[1:]

    # March backward in time
    for n in range(N - 1, -1, -1):
        t_n = t_grid[n]

        # Boundary conditions at time t_n
        V_0 = 0.0
        V_M = Smax - K * np.exp(-r * (T - t_n))

        # Right-hand side is V at the next time level
        rhs = V[1:M].copy()

        # Adjust for known boundary values
        rhs[0] -= a[0] * V_0
        rhs[-1] -= c[-1] * V_M

        # Solve tridiagonal system
        V_interior = solve_banded((1, 1), ab, rhs) #	1 lower diagonal, 1 upper diagonal

        # Update full solution vector at current time
        V[0] = V_0
        V[1:M] = V_interior
        V[M] = V_M

    metadata = {
        "dS": dS,
        "dt": dt,
        "M": M,
        "N": N,
    }
    return S_grid, V, metadata


if __name__ == "__main__":
    # Model parameters
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2

    # Numerical parameters
    Smax = 300.0

    # Output directory for saved figures
    output_dir = Path(__file__).resolve().parent / "results" / "fdm_implicit"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Original single-run case
    original_M = 400
    # original_N = 400
    original_N = 7006 # needs to be extremely large!!

    # Comparison case 1: vary both M and N together with N = M
    grid_settings_equal = [
        (50, 50),
        (100, 100),
        (200, 200),
        (400, 400),
        (800, 800),
    ]

    # Comparison case 2: fix space, vary time
    grid_settings_fix_space = [
        (200, 50),
        (200, 100),
        (200, 200),
        (200, 400),
        (200, 800),
    ]

    # Comparison case 3: fix time, vary space
    grid_settings_fix_time = [
        (50, 200),
        (100, 200),
        (200, 200),
        (400, 200),
        (800, 200),
    ]

    # Comparison case 4: fix grid, vary sigma
    sigma_settings = [0.10, 0.15, 0.20, 0.25, 0.30]
    sigma_compare_M = 400
    sigma_compare_N = 400

    def run_case(M, N, sigma_value=None):
        if sigma_value is None:
            sigma_value = sigma
        S_grid, V_fdm, metadata = european_call_implicit_fdm(K, T, r, sigma_value, Smax, M, N)
        V_exact = bs_call_price(S_grid, K, T, r, sigma_value)
        abs_error = np.abs(V_fdm - V_exact)
        max_error = np.max(abs_error)
        rmse = np.sqrt(np.mean(abs_error**2))
        rel_l2 = np.linalg.norm(V_fdm - V_exact, 2) / np.linalg.norm(V_exact, 2)

        S0 = 100.0
        idx = np.argmin(np.abs(S_grid - S0))

        return {
            "M": M,
            "N": N,
            "sigma": sigma_value,
            "dS": metadata["dS"],
            "dt": metadata["dt"],
            "S_grid": S_grid,
            "V_fdm": V_fdm,
            "V_exact": V_exact,
            "abs_error": abs_error,
            "max_error": max_error,
            "rmse": rmse,
            "rel_l2": rel_l2,
            "S0_index": idx,
            "S0_grid": S_grid[idx],
            "S0_point_error": abs(V_fdm[idx] - V_exact[idx]),
        }

    def print_result_summary(title, result):
        print(title)
        print(f"Grid (M, N) = ({result['M']}, {result['N']}) | sigma = {result['sigma']:.4f} | dS = {result['dS']:.6f}, dt = {result['dt']:.6f}")
        print(f"  Maximum absolute error = {result['max_error']:.6e}")
        print(f"  RMSE = {result['rmse']:.6e}")
        print(f"  Relative L2 error = {result['rel_l2']:.6e}")
        print(f"  S closest to 100.0: {result['S0_grid']:.4f}")
        print(f"  FDM price at S={result['S0_grid']:.4f}:   {result['V_fdm'][result['S0_index']]:.6f}")
        print(f"  Exact price at S={result['S0_grid']:.4f}: {result['V_exact'][result['S0_index']]:.6f}")
        print(f"  Pointwise absolute error: {result['S0_point_error']:.6e}") #\text{error at } S_i = |V_{\text{FDM}}(S_i) - V_{\text{exact}}(S_i)|
        print("-" * 70)

    def run_comparison_case(grid_settings, case_name):
        comparison_results = []
        print(f"\n=== {case_name} ===")
        for M, N in grid_settings:
            result = run_case(M, N)
            comparison_results.append(result)
            print_result_summary(f"{case_name}", result)
        return comparison_results

    def run_sigma_comparison_case(sigma_values, case_name, M, N):
        comparison_results = []
        print(f"\n=== {case_name} ===")
        for sigma_value in sigma_values:
            result = run_case(M, N, sigma_value=sigma_value)
            comparison_results.append(result)
            print_result_summary(f"{case_name}", result)
        return comparison_results

    # Original non-comparison case
    original_result = run_case(original_M, original_N)
    print("\n=== Original single-run case ===")
    print_result_summary("Original single-run case", original_result)

    # Original case plots: numerical solution, absolute error, and RMSE marker
    plt.figure(figsize=(10, 6))
    plt.plot(original_result["S_grid"], original_result["V_fdm"], label="Implicit FDM")
    plt.plot(original_result["S_grid"], original_result["V_exact"], "--", label="Black-Scholes exact")
    plt.xlabel("Asset price S")
    plt.ylabel("Option value")
    plt.title(f"Original Case Numerical Solution (M={original_M}, N={original_N})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "original_numerical_solution.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(original_result["S_grid"], original_result["abs_error"], label="Absolute error")
    plt.xlabel("Asset price S")
    plt.ylabel("Absolute error")
    plt.title(f"Original Case Absolute Error (M={original_M}, N={original_N})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "original_absolute_error.png", dpi=300, bbox_inches="tight")
    plt.close()

    # plt.figure(figsize=(8, 5))
    # plt.plot([1], [original_result["rmse"]], marker="o", linestyle="None", label=f"RMSE = {original_result['rmse']:.6e}")
    # plt.xticks([1], [f"M={original_M}, N={original_N}"])
    # plt.ylabel("RMSE")
    # plt.title("Original Case RMSE")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(output_dir / "original_rmse.png", dpi=300, bbox_inches="tight")
    # plt.close()

    # Comparison studies
    results_equal = run_comparison_case(grid_settings_equal, "Comparison 1: Vary both M and N with N = M")
    results_fix_space = run_comparison_case(grid_settings_fix_space, "Comparison 2: Fix M and vary N")
    results_fix_time = run_comparison_case(grid_settings_fix_time, "Comparison 3: Fix N and vary M")
    results_sigma = run_sigma_comparison_case(
        sigma_settings,
        "Comparison 4: Fix grid and vary sigma",
        sigma_compare_M,
        sigma_compare_N,
    )

    def plot_comparison_pointwise_errors(results, case_name, filename):
        plt.figure(figsize=(10, 6))
        for result in results:
            plt.plot(
                result["S_grid"],
                result["abs_error"],
                label=f"M={result['M']}, N={result['N']}, sigma={result['sigma']:.2f}"
            )
        plt.xlabel("Asset price S")
        plt.ylabel("Absolute error")
        plt.title(f"{case_name}: Pointwise Absolute Error")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_comparison_overall_errors(results, x_key, x_label, case_name, filename):
        x_values_raw = np.array([result[x_key] for result in results], dtype=float)
        max_errors_raw = np.array([result["max_error"] for result in results], dtype=float)
        rmse_values_raw = np.array([result["rmse"] for result in results], dtype=float)
        rel_l2_values_raw = np.array([result["rel_l2"] for result in results], dtype=float)

        # Use negative log scale so that smaller dS / dt and smaller errors are shown more clearly.
        if x_key in ["dS", "dt"]:
            x_values = -np.log10(x_values_raw)
            xlabel = f"$-\\log_{{10}}({x_label})$"
        else:
            x_values = x_values_raw
            xlabel = x_label

        max_errors = max_errors_raw
        rmse_values = rmse_values_raw
        rel_l2_values = rel_l2_values_raw

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, max_errors, marker="o", label="Max absolute error")
        plt.plot(x_values, rmse_values, marker="s", label="RMSE")
        plt.plot(x_values, rel_l2_values, marker="^", label="Relative L2 error")
        plt.yscale("log")
        plt.xlabel(xlabel)
        plt.ylabel("Error (log scale)")
        plt.title(f"{case_name}: Overall Error Summary")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    plot_comparison_pointwise_errors(
        results_equal,
        "Comparison 1 (N = M)",
        "I_comparison_equal_pointwise_error.png"
    )
    plot_comparison_overall_errors(
        results_equal,
        "dS",
        "dS",
        "Comparison 1 (N = M)",
        "I_comparison_equal_overall_error.png"
    )

    plot_comparison_pointwise_errors(
        results_fix_space,
        "Comparison 2 (Fix M, vary N)",
        "I_comparison_fix_space_pointwise_error.png"
    )
    plot_comparison_overall_errors(
        results_fix_space,
        "dt",
        "dt",
        "Comparison 2 (Fix M, vary N)",
        "I_comparison_fix_space_overall_error.png"
    )

    plot_comparison_pointwise_errors(
        results_fix_time,
        "Comparison 3 (Fix N, vary M)",
        "I_comparison_fix_time_pointwise_error.png"
    )
    plot_comparison_overall_errors(
        results_fix_time,
        "dS",
        "dS",
        "Comparison 3 (Fix N, vary M)",
        "I_comparison_fix_time_overall_error.png"
    )

    plot_comparison_pointwise_errors(
        results_sigma,
        "Comparison 4 (Fix grid, vary sigma)",
        "I_comparison_sigma_pointwise_error.png"
    )
    plot_comparison_overall_errors(
        results_sigma,
        "sigma",
        "Volatility sigma",
        "Comparison 4 (Fix grid, vary sigma)",
        "I_comparison_sigma_overall_error.png"
    )