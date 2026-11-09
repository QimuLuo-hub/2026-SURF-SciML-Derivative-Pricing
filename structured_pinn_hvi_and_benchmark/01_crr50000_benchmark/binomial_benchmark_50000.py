"""50,000-step CRR benchmark export for the matched American put.

Standalone canonical benchmark script.

Requirements
------------
Python 3.9+ and NumPy only.

Reads
-----
No project files and no previous reference files are required.

Writes
------
./results/binomial_benchmark_50000/

Scientific scope
----------------
This is an independent high-resolution Cox-Ross-Rubinstein (CRR) benchmark,
not an analytic/exact ground truth. The fixed contract is

    American put
    K = 1
    T = 1
    r = 0.05
    q = 0.10
    sigma = 0.60

The price-slice grid matches the compact comparison export:
S/K in [0.5, 1.5] with 401 equally spaced points, at tau/T = 1, 0.5, 0.1.

The free-boundary tau grid matches the compact comparison export:
sorted unique union of 161 equally spaced points on [0.01, 1] and
the exact slice levels {0.1, 0.5, 1.0}, giving 163 points.

Boundary extraction uses the same premium threshold 1e-6, but uses a
root-price bisection rather than the reference solver's spatial-grid
linear threshold crossing. This difference is recorded in metadata.

Greeks are deterministic second-order finite differences on the common
401-point price grid:
    Delta = dV/dS
    Gamma = d2V/dS2
    scaled_Gamma = K * Gamma
With K=1, scaled_Gamma numerically equals Gamma.

The run is resumable. Each completed price or boundary job is fsynced
to a ledger before the next result is accepted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CONTRACT = {
    "option_type": "put",
    "K": 1.0,
    "T": 1.0,
    "r": 0.05,
    "q": 0.10,
    "sigma": 0.60,
}

TREE_STEPS = 50_000

PRICE_S_MIN = 0.5
PRICE_S_MAX = 1.5
PRICE_S_POINTS = 401
PRICE_TAUS = (1.0, 0.5, 0.1)

BOUNDARY_TAU_MIN = 0.01
BOUNDARY_TAU_MAX = 1.0
BOUNDARY_TAU_BASE_POINTS = 161
BOUNDARY_THRESHOLD = 1e-6
BOUNDARY_S_LO = 0.05
BOUNDARY_S_HI = 1.0
BOUNDARY_BISECTION_ITERATIONS = 20

DEFAULT_RESULT_DIR = Path.cwd() / "results" / "binomial_benchmark_50000"


def price_spots() -> np.ndarray:
    return np.linspace(PRICE_S_MIN, PRICE_S_MAX, PRICE_S_POINTS, dtype=np.float64)


def boundary_taus() -> np.ndarray:
    base = np.linspace(
        BOUNDARY_TAU_MIN,
        BOUNDARY_TAU_MAX,
        BOUNDARY_TAU_BASE_POINTS,
        dtype=np.float64,
    )
    return np.unique(np.concatenate([base, np.asarray(PRICE_TAUS, dtype=np.float64)]))


def crr_american_put(
    spot: float,
    tau: float,
    *,
    steps: int = TREE_STEPS,
    k: float = CONTRACT["K"],
    r: float = CONTRACT["r"],
    q: float = CONTRACT["q"],
    sigma: float = CONTRACT["sigma"],
) -> float:
    """Memory-efficient float64 CRR American-put backward induction."""
    spot = float(spot)
    tau = float(tau)
    k = float(k)
    r = float(r)
    q = float(q)
    sigma = float(sigma)
    steps = int(steps)

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if spot < 0.0 or k <= 0.0 or sigma <= 0.0:
        raise ValueError("require spot>=0, K>0 and sigma>0")
    if tau <= 0.0:
        return max(k - spot, 0.0)

    dt = tau / steps
    log_u = sigma * math.sqrt(dt)
    u = math.exp(log_u)
    d = 1.0 / u
    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise RuntimeError(f"invalid_crr_probability:{p}")
    disc = math.exp(-r * dt)

    j = np.arange(steps + 1, dtype=np.float64)
    stock = spot * np.exp((2.0 * j - steps) * log_u)
    value = np.maximum(k - stock, 0.0)

    for _ in range(steps - 1, -1, -1):
        value = disc * (p * value[1:] + (1.0 - p) * value[:-1])
        stock = stock[:-1] * u
        value = np.maximum(value, np.maximum(k - stock, 0.0))

    return float(value[0])


def crr_american_put_small_reference(
    spot: float,
    tau: float,
    *,
    steps: int,
    k: float = CONTRACT["K"],
    r: float = CONTRACT["r"],
    q: float = CONTRACT["q"],
    sigma: float = CONTRACT["sigma"],
) -> float:
    """Independent, intentionally slow small-N implementation for preflight."""
    if steps < 1:
        raise ValueError("steps must be >=1")
    if tau <= 0.0:
        return max(k - spot, 0.0)

    dt = tau / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise RuntimeError("invalid CRR probability")
    disc = math.exp(-r * dt)

    values = [
        max(k - spot * (u**j) * (d ** (steps - j)), 0.0)
        for j in range(steps + 1)
    ]
    for n in range(steps - 1, -1, -1):
        new_values = []
        for j in range(n + 1):
            continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j])
            stock = spot * (u**j) * (d ** (n - j))
            exercise = max(k - stock, 0.0)
            new_values.append(max(exercise, continuation))
        values = new_values
    return float(values[0])


def finite_difference_greeks(
    spots: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Second-order finite-difference Delta and Gamma on an equally spaced grid."""
    s = np.asarray(spots, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if s.ndim != 1 or v.ndim != 1 or len(s) != len(v) or len(s) < 4:
        raise ValueError("spots and values must be 1D arrays of equal length >=4")

    ds = np.diff(s)
    if np.any(ds <= 0.0) or not np.allclose(ds, ds[0], rtol=1e-11, atol=1e-14):
        raise ValueError("finite_difference_greeks requires an equally spaced grid")
    h = float(ds[0])

    delta = np.empty_like(v)
    gamma = np.empty_like(v)

    delta[1:-1] = (v[2:] - v[:-2]) / (2.0 * h)
    delta[0] = (-3.0 * v[0] + 4.0 * v[1] - v[2]) / (2.0 * h)
    delta[-1] = (3.0 * v[-1] - 4.0 * v[-2] + v[-3]) / (2.0 * h)

    gamma[1:-1] = (v[2:] - 2.0 * v[1:-1] + v[:-2]) / (h * h)
    gamma[0] = (2.0 * v[0] - 5.0 * v[1] + 4.0 * v[2] - v[3]) / (h * h)
    gamma[-1] = (2.0 * v[-1] - 5.0 * v[-2] + 4.0 * v[-3] - v[-4]) / (h * h)

    return delta, gamma


def price_job_plan() -> list[dict[str, Any]]:
    return [
        {
            "S_over_K": float(s),
            "tau_over_T": float(tau),
            "steps": TREE_STEPS,
        }
        for tau in PRICE_TAUS
        for s in price_spots()
    ]


def _solve_price_job(job: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    s = float(job["S_over_K"])
    tau = float(job["tau_over_T"])
    steps = int(job["steps"])
    value = crr_american_put(s, tau, steps=steps)
    payoff = max(CONTRACT["K"] - s, 0.0)
    return {
        "S_over_K": s,
        "tau_over_T": tau,
        "steps": steps,
        "V_over_K": value / CONTRACT["K"],
        "payoff_over_K": payoff / CONTRACT["K"],
        "premium_over_K": (value - payoff) / CONTRACT["K"],
        "seconds": time.perf_counter() - started,
    }


def solve_boundary_tau(
    tau: float,
    *,
    steps: int = TREE_STEPS,
    iterations: int = BOUNDARY_BISECTION_ITERATIONS,
    threshold: float = BOUNDARY_THRESHOLD,
) -> dict[str, Any]:
    """Locate the root-price premium threshold by deterministic bisection."""
    started = time.perf_counter()
    tau = float(tau)
    steps = int(steps)
    iterations = int(iterations)
    threshold = float(threshold)

    def premium(s: float) -> float:
        value = crr_american_put(s, tau, steps=steps)
        return value - max(CONTRACT["K"] - s, 0.0)

    lo = float(BOUNDARY_S_LO)
    hi = float(BOUNDARY_S_HI)
    p_lo = premium(lo)
    p_hi = premium(hi)

    # Exercise-like side: premium <= threshold.
    # Continuation-like side: premium > threshold.
    if p_lo > threshold:
        return {
            "tau_over_T": tau,
            "steps": steps,
            "found": False,
            "S_star_over_K": None,
            "reason": "lower_bracket_not_exercise_like",
            "threshold": threshold,
            "lower_premium": p_lo,
            "upper_premium": p_hi,
            "final_bracket_width": hi - lo,
            "iterations": 0,
            "seconds": time.perf_counter() - started,
        }
    if p_hi <= threshold:
        return {
            "tau_over_T": tau,
            "steps": steps,
            "found": False,
            "S_star_over_K": None,
            "reason": "upper_bracket_not_continuation_like",
            "threshold": threshold,
            "lower_premium": p_lo,
            "upper_premium": p_hi,
            "final_bracket_width": hi - lo,
            "iterations": 0,
            "seconds": time.perf_counter() - started,
        }

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        p_mid = premium(mid)
        if p_mid <= threshold:
            lo = mid
            p_lo = p_mid
        else:
            hi = mid
            p_hi = p_mid

    star = 0.5 * (lo + hi)
    return {
        "tau_over_T": tau,
        "steps": steps,
        "found": True,
        "S_star_over_K": star,
        "reason": "premium_threshold_bisection",
        "threshold": threshold,
        "lower_premium": p_lo,
        "upper_premium": p_hi,
        "final_bracket_width": hi - lo,
        "iterations": iterations,
        "seconds": time.perf_counter() - started,
    }


def _solve_boundary_job(job: dict[str, Any]) -> dict[str, Any]:
    return solve_boundary_tau(float(job["tau_over_T"]))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name) for name in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _price_key(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(row["S_over_K"]),
        float(row["tau_over_T"]),
        int(row["steps"]),
    )


def _boundary_key(row: dict[str, Any]) -> tuple[float, int]:
    return (float(row["tau_over_T"]), int(row["steps"]))


def run_prices(result_dir: Path, workers: int) -> dict[str, Any]:
    ledger = result_dir / "_resume_price_ledger.csv"
    fieldnames = [
        "S_over_K",
        "tau_over_T",
        "steps",
        "V_over_K",
        "payoff_over_K",
        "premium_over_K",
        "seconds",
    ]
    existing = _read_csv(ledger)
    done = {_price_key(row) for row in existing}
    plan = price_job_plan()
    pending = [job for job in plan if _price_key(job) not in done]

    started = time.perf_counter()
    if pending:
        if workers == 1:
            iterator = enumerate((_solve_price_job(job) for job in pending), start=1)
            for i, row in iterator:
                _append_csv_row(ledger, fieldnames, row)
                print(
                    f"[price {i}/{len(pending)}] tau={row['tau_over_T']:.3g} "
                    f"S/K={row['S_over_K']:.4f} V/K={row['V_over_K']:.12g} "
                    f"sec={row['seconds']:.3f}",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_solve_price_job, job) for job in pending]
                for i, future in enumerate(as_completed(futures), start=1):
                    row = future.result()
                    _append_csv_row(ledger, fieldnames, row)
                    print(
                        f"[price {i}/{len(pending)}] tau={row['tau_over_T']:.3g} "
                        f"S/K={row['S_over_K']:.4f} V/K={row['V_over_K']:.12g} "
                        f"sec={row['seconds']:.3f}",
                        flush=True,
                    )

    rows_raw = _read_csv(ledger)
    # Deduplicate by deterministic job key, keeping the last completed record.
    unique: dict[tuple[float, float, int], dict[str, Any]] = {}
    for r in rows_raw:
        row = {
            "S_over_K": float(r["S_over_K"]),
            "tau_over_T": float(r["tau_over_T"]),
            "steps": int(r["steps"]),
            "V_over_K": float(r["V_over_K"]),
            "payoff_over_K": float(r["payoff_over_K"]),
            "premium_over_K": float(r["premium_over_K"]),
            "seconds": float(r["seconds"]),
        }
        unique[_price_key(row)] = row

    required = {_price_key(job) for job in plan}
    complete = required.issubset(unique)
    if complete:
        curated = [unique[key] for key in sorted(required, key=lambda k: (-k[1], k[0], k[2]))]
        _write_csv(result_dir / "price_slices.csv", curated)

        all_greeks: list[dict[str, Any]] = []
        tags = {1.0: "1", 0.5: "05", 0.1: "01"}
        for tau in PRICE_TAUS:
            subset = [r for r in curated if float(r["tau_over_T"]) == float(tau)]
            subset.sort(key=lambda r: r["S_over_K"])
            s = np.asarray([r["S_over_K"] for r in subset], dtype=np.float64)
            v = np.asarray([r["V_over_K"] for r in subset], dtype=np.float64) * CONTRACT["K"]
            delta, gamma = finite_difference_greeks(s * CONTRACT["K"], v)
            greek_rows = []
            for row, dlt, gma in zip(subset, delta, gamma):
                g = {
                    **row,
                    "Delta": float(dlt),
                    "Gamma": float(gma),
                    "scaled_Gamma": float(CONTRACT["K"] * gma),
                    "greek_operator": "second_order_fd_on_401_point_common_grid",
                }
                greek_rows.append(g)
                all_greeks.append(g)
            _write_csv(result_dir / f"price_slice_tau_{tags[tau]}.csv", subset)
            _write_csv(result_dir / f"greeks_slice_tau_{tags[tau]}.csv", greek_rows)
        _write_csv(result_dir / "greeks_slices.csv", all_greeks)

    return {
        "status": "completed" if complete else "partial",
        "expected_jobs": len(required),
        "completed_jobs": len(required & set(unique)),
        "pending_at_invocation_start": len(pending),
        "elapsed_this_invocation_seconds": time.perf_counter() - started,
    }


def run_boundary(result_dir: Path, workers: int) -> dict[str, Any]:
    ledger = result_dir / "_resume_boundary_ledger.csv"
    fieldnames = [
        "tau_over_T",
        "steps",
        "found",
        "S_star_over_K",
        "reason",
        "threshold",
        "lower_premium",
        "upper_premium",
        "final_bracket_width",
        "iterations",
        "seconds",
    ]
    plan = [{"tau_over_T": float(tau), "steps": TREE_STEPS} for tau in boundary_taus()]
    existing = _read_csv(ledger)
    done = {_boundary_key(row) for row in existing}
    pending = [job for job in plan if _boundary_key(job) not in done]

    started = time.perf_counter()
    if pending:
        if workers == 1:
            for i, job in enumerate(pending, start=1):
                row = _solve_boundary_job(job)
                _append_csv_row(ledger, fieldnames, row)
                print(
                    f"[boundary {i}/{len(pending)}] tau={row['tau_over_T']:.6f} "
                    f"S*/K={row['S_star_over_K']} found={row['found']} "
                    f"sec={row['seconds']:.3f}",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_solve_boundary_job, job) for job in pending]
                for i, future in enumerate(as_completed(futures), start=1):
                    row = future.result()
                    _append_csv_row(ledger, fieldnames, row)
                    print(
                        f"[boundary {i}/{len(pending)}] tau={row['tau_over_T']:.6f} "
                        f"S*/K={row['S_star_over_K']} found={row['found']} "
                        f"sec={row['seconds']:.3f}",
                        flush=True,
                    )

    rows_raw = _read_csv(ledger)
    unique: dict[tuple[float, int], dict[str, Any]] = {}
    for r in rows_raw:
        raw_star = r.get("S_star_over_K")
        star = None if raw_star in (None, "", "None") else float(raw_star)
        row = {
            "tau_over_T": float(r["tau_over_T"]),
            "steps": int(r["steps"]),
            "found": str(r["found"]).lower() == "true",
            "S_star_over_K": star,
            "reason": str(r["reason"]),
            "threshold": float(r["threshold"]),
            "lower_premium": float(r["lower_premium"]),
            "upper_premium": float(r["upper_premium"]),
            "final_bracket_width": float(r["final_bracket_width"]),
            "iterations": int(r["iterations"]),
            "seconds": float(r["seconds"]),
        }
        unique[_boundary_key(row)] = row

    required = {_boundary_key(job) for job in plan}
    complete = required.issubset(unique)
    if complete:
        curated = [unique[key] for key in sorted(required, key=lambda k: k[0])]
        _write_csv(result_dir / "free_boundary.csv", curated)

    return {
        "status": "completed" if complete else "partial",
        "expected_jobs": len(required),
        "completed_jobs": len(required & set(unique)),
        "found_count": sum(
            1 for key in required if key in unique and bool(unique[key]["found"])
        ),
        "pending_at_invocation_start": len(pending),
        "elapsed_this_invocation_seconds": time.perf_counter() - started,
    }


def preflight() -> dict[str, Any]:
    fast = crr_american_put(1.0, 0.5, steps=16)
    slow = crr_american_put_small_reference(1.0, 0.5, steps=16)
    difference = abs(fast - slow)
    if difference > 1e-12:
        raise RuntimeError(f"optimized/reference CRR mismatch:{difference}")

    spots = price_spots()
    taus = boundary_taus()
    if len(spots) != 401 or len(taus) != 163:
        raise RuntimeError("comparison grid contract mismatch")

    # Boundary smoke uses a low step count only to validate bracketing/mechanics.
    b = solve_boundary_tau(0.5, steps=256, iterations=8)
    if not b["found"] or not (0.0 < float(b["S_star_over_K"]) < 1.0):
        raise RuntimeError(f"boundary preflight failed:{b}")

    return {
        "status": "passed",
        "optimized_vs_independent_small_tree_abs_diff": difference,
        "smoke_price": fast,
        "boundary_smoke_S_star_over_K": b["S_star_over_K"],
        "price_grid_points_per_slice": len(spots),
        "price_slice_count": len(PRICE_TAUS),
        "canonical_price_jobs": len(price_job_plan()),
        "boundary_tau_points": len(taus),
        "tree_steps": TREE_STEPS,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }


def write_manifest(result_dir: Path, workers: int) -> None:
    payload = {
        "method": "converged_high_resolution_CRR_binomial_American_put",
        "ground_truth_claim": False,
        "contract": CONTRACT,
        "tree_steps": TREE_STEPS,
        "dtype": "float64",
        "workers": workers,
        "price_grid": {
            "S_over_K_min": PRICE_S_MIN,
            "S_over_K_max": PRICE_S_MAX,
            "S_points": PRICE_S_POINTS,
            "tau_over_T": list(PRICE_TAUS),
        },
        "boundary_grid": {
            "tau_over_T_min": BOUNDARY_TAU_MIN,
            "tau_over_T_max": BOUNDARY_TAU_MAX,
            "tau_points": len(boundary_taus()),
            "tau_construction": (
                "sorted unique union of 161 equally spaced points on [0.01,1.0] "
                "and exact {0.1,0.5,1.0}"
            ),
            "S_bracket": [BOUNDARY_S_LO, BOUNDARY_S_HI],
            "premium_threshold": BOUNDARY_THRESHOLD,
            "bisection_iterations": BOUNDARY_BISECTION_ITERATIONS,
            "extraction_method": "root_price_premium_threshold_bisection",
        },
        "greeks": {
            "Delta": "dV/dS",
            "Gamma": "d2V/dS2",
            "scaled_Gamma": "K * Gamma",
            "source": "second-order finite differences on the 401-point CRR price slices",
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unavailable",
            "cpu_count": os.cpu_count(),
        },
        "script_sha256": _sha256(Path(__file__).resolve()),
    }
    _atomic_json(result_dir / "manifest.json", payload)


def write_checksums(result_dir: Path) -> None:
    names = [
        "manifest.json",
        "price_slices.csv",
        "price_slice_tau_1.csv",
        "price_slice_tau_05.csv",
        "price_slice_tau_01.csv",
        "greeks_slices.csv",
        "greeks_slice_tau_1.csv",
        "greeks_slice_tau_05.csv",
        "greeks_slice_tau_01.csv",
        "free_boundary.csv",
    ]
    payload = {
        name: _sha256(result_dir / name)
        for name in names
        if (result_dir / name).is_file()
    }
    _atomic_json(result_dir / "checksums.json", payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        choices=("preflight", "prices", "boundary", "all"),
        default="all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2) // 2)),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be >=1")

    if args.part == "preflight":
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return 0

    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(result_dir, args.workers)

    started = time.perf_counter()
    status: dict[str, Any] = {
        "requested_part": args.part,
        "tree_steps": TREE_STEPS,
        "result_dir": str(result_dir),
    }

    if args.part in ("prices", "all"):
        status["prices"] = run_prices(result_dir, args.workers)

    if args.part in ("boundary", "all"):
        status["boundary"] = run_boundary(result_dir, args.workers)

    requested_sections = [
        status[name]["status"]
        for name in ("prices", "boundary")
        if name in status
    ]
    status["status"] = (
        "completed"
        if requested_sections and all(x == "completed" for x in requested_sections)
        else "partial"
    )
    status["elapsed_this_invocation_seconds"] = time.perf_counter() - started

    write_checksums(result_dir)
    _atomic_json(result_dir / "final_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
