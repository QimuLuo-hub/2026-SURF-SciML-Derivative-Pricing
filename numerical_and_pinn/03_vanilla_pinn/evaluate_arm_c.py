"""Read the stored Arm C and DIRK exports and report the poster-case metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARM_C = HERE / "results" / "poster_case"
REFERENCE = HERE.parent / "02_dirk_reference" / "results" / "reference"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-c", type=Path, default=ARM_C)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()

    with (
        np.load(args.arm_c / "surface.npz") as arm_c,
        np.load(args.reference / "surface.npz") as reference,
    ):
        for axis in ("S_over_K", "tau_over_T"):
            if not np.array_equal(arm_c[axis], reference[axis]):
                raise ValueError(f"Arm C and DIRK have different {axis} grids")
        error = arm_c["V_over_K"] - reference["V_over_K"]
        price_rmse = float(np.sqrt(np.mean(error**2)))
        times = arm_c["tau_over_T"].copy()

    with (args.arm_c / "free_boundary.csv").open(encoding="utf-8", newline="") as file:
        boundaries = list(csv.DictReader(file))
    boundary_times = [float(row["tau_over_T"]) for row in boundaries]
    if not np.array_equal(boundary_times, times):
        raise ValueError("Arm C boundary rows do not match the surface time grid")
    found = sum(row["boundary_found"] == "true" for row in boundaries)
    print(json.dumps({
        "price_rmse": price_rmse,
        "boundary_found": found,
        "boundary_time_points": len(boundaries),
        "boundary_found_percentage": 100.0 * found / len(boundaries),
    }, indent=2))


if __name__ == "__main__":
    main()
