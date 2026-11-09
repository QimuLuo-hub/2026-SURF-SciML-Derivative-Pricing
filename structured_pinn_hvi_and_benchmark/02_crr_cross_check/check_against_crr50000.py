"""Compare CN+PSOR and DIRK-Policy-sinh against the 50,000-step CRR benchmark.

Reproduces the poster's R2 table (price RMSE and max |price error|), the 18.5x
ratio, and the R4 boundary RMSEs. DIRK is read from the stored reference export
in ``numerical_and_pinn``. CN+PSOR (M = N = 120, Smax = 4, omega = 1.2,
tolerance 1e-12) is re-solved with the solver in ``numerical_and_pinn/src``;
this takes a few seconds.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
TEAM_ROOT = PACKAGE.parent
CRR = PACKAGE / "01_crr50000_benchmark" / "results" / "crr50000"
REFERENCE = TEAM_ROOT / "numerical_and_pinn" / "02_dirk_reference" / "results" / "reference"
RUIXI_SRC = TEAM_ROOT / "numerical_and_pinn" / "src"

CONTRACT = {"K": 1.0, "T": 1.0, "r": 0.05, "q": 0.10, "sigma": 0.60}
SLICES = (("1", 1.0), ("05", 0.5), ("01", 0.1))
THRESHOLD = 1e-6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crr", type=Path, default=CRR)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()

    crr_prices = [_read(args.crr / f"price_slice_tau_{tag}.csv") for tag, _ in SLICES]
    crr_boundary = _read(args.crr / "free_boundary.csv")
    boundary_tau = np.array([float(row["tau_over_T"]) for row in crr_boundary])
    crr_s_star = np.array([float(row["S_star_over_K"]) for row in crr_boundary])

    dirk_price_error = []
    for (tag, _), crr_rows in zip(SLICES, crr_prices):
        dirk_rows = _read(args.reference / f"price_slice_tau_{tag}.csv")
        _same_spots(crr_rows, dirk_rows)
        dirk_price_error.extend(
            float(d["V_over_K"]) - float(c["V_over_K"]) for c, d in zip(crr_rows, dirk_rows)
        )
    dirk_boundary = _read(args.reference / "free_boundary.csv")
    if [float(row["tau_over_T"]) for row in dirk_boundary] != boundary_tau.tolist():
        raise ValueError("DIRK and CRR boundary time grids differ")
    dirk_s_star = np.array([float(row["S_star_over_K"]) for row in dirk_boundary])

    psor_price_error, psor_s_star = _cn_psor(crr_prices, boundary_tau)

    dirk = _summary(dirk_price_error, dirk_s_star - crr_s_star)
    psor = _summary(psor_price_error, psor_s_star - crr_s_star)
    print(json.dumps({
        "CN_PSOR": psor,
        "DIRK_POLICY": dirk,
        "price_rmse_ratio_CN_PSOR_over_DIRK": psor["price_rmse"] / dirk["price_rmse"],
        "price_points": len(dirk_price_error),
        "boundary_points": int(boundary_tau.size),
    }, indent=2))


def _cn_psor(crr_prices, boundary_tau):
    sys.path.insert(0, str(RUIXI_SRC))
    from american_risk_surfaces.diagnostics.boundary import extract_boundary_at_time
    from american_risk_surfaces.solvers.cn_psor import american_crank_nicolson_psor_price

    result = american_crank_nicolson_psor_price(
        "put", CONTRACT["K"], CONTRACT["T"], CONTRACT["r"], CONTRACT["q"], CONTRACT["sigma"],
        Smax=4.0, M=120, N=120, omega=1.2, tolerance=1e-12, max_iter=100_000,
    )
    if not result.converged:
        raise RuntimeError("CN+PSOR did not converge")

    errors = []
    for (_, tau), rows in zip(SLICES, crr_prices):
        index = int(np.argmin(np.abs(result.tau_grid - tau)))
        if abs(result.tau_grid[index] - tau) > 1e-12:
            raise ValueError(f"tau/T = {tau} is not a CN time level")
        spots = np.array([float(row["S_over_K"]) for row in rows])
        values = np.interp(spots, result.spot_grid, result.value_grid[index])
        errors.extend(values - np.array([float(row["V_over_K"]) for row in rows]))

    native_tau, native_s_star = [], []
    for index in range(1, len(result.tau_grid)):
        point = extract_boundary_at_time(
            result.spot_grid,
            result.value_grid[index] - result.payoff,
            "put",
            float(result.tau_grid[index]),
            index,
            THRESHOLD,
        )
        if not point.boundary_found:
            raise RuntimeError(f"no CN+PSOR boundary at time index {index}")
        native_tau.append(float(result.tau_grid[index]))
        native_s_star.append(point.boundary_spot)
    s_star = np.interp(boundary_tau, native_tau, native_s_star)
    return errors, s_star


def _summary(price_error, boundary_error) -> dict[str, float]:
    price = np.asarray(price_error, dtype=float)
    boundary = np.asarray(boundary_error, dtype=float)
    return {
        "price_rmse": float(np.sqrt(np.mean(price**2))),
        "price_max_abs": float(np.max(np.abs(price))),
        "boundary_rmse": float(np.sqrt(np.mean(boundary**2))),
    }


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _same_spots(left, right) -> None:
    if [float(row["S_over_K"]) for row in left] != [float(row["S_over_K"]) for row in right]:
        raise ValueError("price slices use different S/K grids")


if __name__ == "__main__":
    main()
