"""Paths for the standalone SURF poster experiment package."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PACKAGE_ROOT / "inputs" / "v1_small_grid"
TOLERANCE_DECISION = PACKAGE_ROOT / "inputs" / "tolerance_decision.json"
R1_DIR = PACKAGE_ROOT / "01_cn_lcp_solvers"
DIRK_DIR = PACKAGE_ROOT / "02_dirk_reference"
SUPPLEMENTARY_DIR = PACKAGE_ROOT / "supplementary"
RUNS_DIR = Path(os.environ.get("SURF_OUTPUT_DIR", PACKAGE_ROOT / "runs")).resolve()


def portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    return resolved.relative_to(PACKAGE_ROOT).as_posix() if resolved.is_relative_to(PACKAGE_ROOT) else str(resolved)
