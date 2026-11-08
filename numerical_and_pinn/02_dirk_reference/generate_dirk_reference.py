"""Generate the native DIRK–Policy–sinh reference for the poster contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from american_risk_surfaces.pinn.protocol import HELDOUT_SPLITS
from american_risk_surfaces.pinn.reference import generate_reference_cache

POSTER_REGIME = "put_T100_s060_r005_q010"
SPATIAL_STEPS = 480
TIME_STEPS = 960


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    generate_reference_cache(
        args.output_dir,
        splits=HELDOUT_SPLITS,
        spatial_steps=SPATIAL_STEPS,
        time_steps=TIME_STEPS,
        regime_ids=(POSTER_REGIME,),
    )


if __name__ == "__main__":
    main()
