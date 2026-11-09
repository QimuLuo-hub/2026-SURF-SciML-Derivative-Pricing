"""Recompute the poster's R3 price-RMSE table from stored surfaces.

Every model is scored against the stored DIRK-Policy-sinh reference
(M = 480, N = 960) on its 401 x 163 grid over S/K in [0.5, 1.5] and
tau/T in [0.01, 1]. Arm C and Arm D already use that grid. The HVI surfaces use
their native 401 x 161 grid over S/K in [0.2, 1.5]; they are interpolated
bilinearly onto the reference grid, which needs no extrapolation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
TEAM_ROOT = PACKAGE.parent
REFERENCE = TEAM_ROOT / "numerical_and_pinn" / "02_dirk_reference" / "results" / "reference"
ARM_C = TEAM_ROOT / "numerical_and_pinn" / "03_vanilla_pinn" / "results" / "poster_case"
ARM_D = PACKAGE / "03_structure_aware_pinn" / "results" / "poster_case"
HVI = (
    PACKAGE / "04_high_dimensional_hvi" / "results"
    / "hd3_matched_american_put_boundary_complete" / "visualization"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-dimensions", action="store_true", help="also report d = 10 and 20")
    args = parser.parse_args()

    with np.load(REFERENCE / "surface.npz") as reference:
        ref_s = reference["S_over_K"].copy()
        ref_tau = reference["tau_over_T"].copy()
        ref_v = reference["V_over_K"].copy()

    rmse = {
        "Vanilla PINN (Arm C)": _grid_rmse(ARM_C, ref_s, ref_tau, ref_v),
        "Structure-aware PINN (Arm D)": _grid_rmse(ARM_D, ref_s, ref_tau, ref_v),
    }
    for dimension in (10, 20, 50) if args.all_dimensions else (50,):
        for folder, label in (("FC_HVI", "FC-HVI"), ("DEEPSETS_HVI", "DeepSets-HVI")):
            path = HVI / f"d{dimension}" / folder
            rmse[f"{label} (d = {dimension})"] = _hvi_rmse(path, ref_s, ref_tau, ref_v)

    print(json.dumps({
        "price_rmse": rmse,
        "arm_c_over_arm_d": rmse["Vanilla PINN (Arm C)"] / rmse["Structure-aware PINN (Arm D)"],
    }, indent=2))


def _grid_rmse(folder: Path, ref_s, ref_tau, ref_v) -> float:
    with np.load(folder / "surface.npz") as surface:
        if not (
            np.array_equal(surface["S_over_K"], ref_s)
            and np.array_equal(surface["tau_over_T"], ref_tau)
        ):
            raise ValueError(f"{folder} is not on the reference grid")
        return float(np.sqrt(np.mean((surface["V_over_K"] - ref_v) ** 2)))


def _hvi_rmse(folder: Path, ref_s, ref_tau, ref_v) -> float:
    with np.load(folder / "surface.npz") as surface:
        flat_s = surface["S_over_K"]
        flat_tau = surface["tau_over_T"]
        flat_v = surface["V_over_K"]
    s = np.unique(flat_s)
    tau = np.unique(flat_tau)
    if s.size * tau.size != flat_v.size:
        raise ValueError(f"{folder} is not a rectilinear grid")
    values = flat_v[np.lexsort((flat_s, flat_tau))].reshape(tau.size, s.size)
    tol = 1e-12
    if ref_s[0] < s[0] - tol or ref_s[-1] > s[-1] + tol:
        raise ValueError(f"{folder} does not cover the reference S/K range")
    if ref_tau[0] < tau[0] - tol or ref_tau[-1] > tau[-1] + tol:
        raise ValueError(f"{folder} does not cover the reference tau/T range")
    along_s = np.vstack([np.interp(ref_s, s, row) for row in values])
    on_reference = np.column_stack(
        [np.interp(ref_tau, tau, along_s[:, j]) for j in range(ref_s.size)]
    )
    return float(np.sqrt(np.mean((on_reference - ref_v) ** 2)))


if __name__ == "__main__":
    main()
