"""One-time label-aware scoring kept separate from held-out PINN training."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from american_risk_surfaces.diagnostics.boundary import extract_boundary_at_time
from american_risk_surfaces.diagnostics.greeks import (
    finite_difference_delta_nonuniform,
    finite_difference_gamma_nonuniform,
)
from american_risk_surfaces.pinn.evaluation import (
    discrete_lcp_audit,
    evaluate_pinn_vi,
    load_pinn_checkpoint,
    predict_pinn_surface,
)
from american_risk_surfaces.pinn.protocol import (
    DATASET_PATH,
    RESULTS_DIR,
    load_regime_records,
)
from american_risk_surfaces.pinn.reference import (
    interpolate_grid_surface,
    interpolate_reference,
)
from american_risk_surfaces.solvers.black_scholes import call_payoff, put_payoff
from american_risk_surfaces.solvers.american_lcp import AmericanLCPConfig, american_cn_lcp_price


def score_checkpoints_once(
    checkpoint_rows: Iterable[dict[str, Any]],
    *,
    output_dir: Path | str = RESULTS_DIR / "04_heldout",
    device: str = "cpu",
    reference_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Open frozen labels only after checkpoints have been declared complete."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = np.load(DATASET_PATH, allow_pickle=True)
    feature_names = tuple(str(value) for value in bundle["feature_names"])
    mask_names = tuple(str(value) for value in bundle["mask_names"])
    dataset_regime_ids = tuple(str(value) for value in bundle["regime_ids"])
    audit_names = tuple(str(value) for value in bundle["audit_numeric_names"])
    feature_index = {name: index for index, name in enumerate(feature_names)}
    mask_index = {name: index for index, name in enumerate(mask_names)}
    audit_index = {name: index for index, name in enumerate(audit_names)}
    high_reference_dir = None if reference_dir is None else Path(reference_dir)
    rows: list[dict[str, Any]] = []
    checkpoint_list = list(checkpoint_rows)
    complete_rows = [row for row in checkpoint_list if row.get("status") == "COMPLETE"]
    scoring_started = perf_counter()
    complete_index = 0
    for checkpoint_row in checkpoint_list:
        if checkpoint_row.get("status") != "COMPLETE":
            continue
        complete_index += 1
        regime_id = str(checkpoint_row["regime_id"])
        print(
            f"[score {complete_index}/{len(complete_rows)} | "
            f"{100.0 * (complete_index - 1) / max(len(complete_rows), 1):5.1f}%] "
            f"arm={checkpoint_row['arm']} regime={regime_id} seed={checkpoint_row['seed']}",
            flush=True,
        )
        regime_index = dataset_regime_ids.index(regime_id)
        selected = bundle["regime_index"] == regime_index
        coordinates = bundle["X"][selected][
            :, [feature_index["log_moneyness"], feature_index["tau_fraction"]]
        ]
        loaded = load_pinn_checkpoint(checkpoint_row["checkpoint_path"], device=device)
        diagnostics = evaluate_pinn_vi(
            loaded,
            coordinates,
            device=device,
        )
        predicted = diagnostics["value"]
        if high_reference_dir is None:
            reference = bundle["y_value"][selected]
            reference_delta = bundle["y_delta"][selected]
            reference_gamma = bundle["y_scaled_gamma"][selected]
            reference_boundary = bundle["y_boundary"][selected]
            reference_source = "frozen_M120_dataset"
        else:
            path = high_reference_dir / f"{regime_id}.npz"
            if not path.exists():
                raise FileNotFoundError(f"missing high-accuracy reference: {path}")
            interpolated = interpolate_reference(path, coordinates)
            reference = interpolated["value_over_k"]
            reference_delta = interpolated["delta"]
            reference_gamma = interpolated["scaled_gamma"]
            reference_boundary = interpolated["boundary_over_k"]
            reference_source = "DIRK_policy_sinh_M480_N960"
        error = predicted - reference
        masks = bundle["masks"][selected]
        audit = bundle["audit_numeric"][selected]
        boundary_metrics, surface_prediction = _boundary_metrics(
            loaded,
            coordinates,
            reference_boundary,
            K=float(audit[0, audit_index["K"]]),
            Smax=float(audit[0, audit_index["Smax"]]),
            M=int(audit[0, audit_index["M"]]),
            device=device,
        )
        discrete_metrics = discrete_lcp_audit(
            surface_prediction,
            AmericanLCPConfig(
                loaded.problem.option_type,
                float(audit[0, audit_index["K"]]),
                loaded.problem.T,
                loaded.problem.r,
                loaded.problem.q,
                loaded.problem.sigma,
                float(audit[0, audit_index["Smax"]]),
                int(audit[0, audit_index["M"]]),
                int(audit[0, audit_index["N"]]),
                tolerance=1e-12,
                obstacle_tolerance=1e-12,
            ),
        )
        base = {
            "arm": checkpoint_row["arm"],
            "variant": checkpoint_row.get("variant", "unknown"),
            "network_spec": checkpoint_row.get("network_spec", "unknown"),
            "split": checkpoint_row["split"],
            "regime_id": regime_id,
            "seed": checkpoint_row["seed"],
            "region": "all",
            "reference_source": reference_source,
            **_error_metrics(error, reference),
            **_vi_metrics(diagnostics),
            **boundary_metrics,
            **discrete_metrics,
        }
        delta_mask = masks[:, mask_index["delta_allowed_mask"]]
        gamma_mask = masks[:, mask_index["gamma_allowed_mask"]]
        base["delta_rmse"] = _rmse(
            diagnostics["delta"][delta_mask] - reference_delta[delta_mask]
        )
        base["scaled_gamma_rmse"] = _rmse(
            diagnostics["scaled_gamma"][gamma_mask]
            - reference_gamma[gamma_mask]
        )
        rows.append(base)
        for region in (
            "exercise_region",
            "continuation_region",
            "boundary_near",
            "payoff_kink_near",
            "maturity_row",
            "strict_interior",
        ):
            region_mask = masks[:, mask_index[region]]
            if np.any(region_mask):
                rows.append(
                    {
                        **base,
                        "region": region,
                        **_error_metrics(error[region_mask], reference[region_mask]),
                    }
                )
        elapsed = perf_counter() - scoring_started
        remaining = elapsed / complete_index * (len(complete_rows) - complete_index)
        print(
            f"[score {complete_index}/{len(complete_rows)} | "
            f"{100.0 * complete_index / max(len(complete_rows), 1):5.1f}%] "
            f"COMPLETE elapsed={_format_duration(elapsed)} eta={_format_duration(remaining)}",
            flush=True,
        )
    _write_csv(output / "pinn_metrics_by_regime_seed.csv", rows)
    return rows


def score_classical_baselines(
    *,
    splits: Iterable[str],
    reference_dir: Path | str,
    output_dir: Path | str,
    regime_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Score frozen M=N=120 CN+PSOR and CN+Policy against the same reference."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    high_reference_dir = Path(reference_dir)
    bundle = np.load(DATASET_PATH, allow_pickle=True)
    feature_names = tuple(str(value) for value in bundle["feature_names"])
    mask_names = tuple(str(value) for value in bundle["mask_names"])
    dataset_regime_ids = tuple(str(value) for value in bundle["regime_ids"])
    feature_index = {name: index for index, name in enumerate(feature_names)}
    mask_index = {name: index for index, name in enumerate(mask_names)}
    rows: list[dict[str, Any]] = []
    for record in load_regime_records(splits=splits, regime_ids=regime_ids):
        regime_index = dataset_regime_ids.index(record.regime_id)
        selected = bundle["regime_index"] == regime_index
        coordinates = bundle["X"][selected][
            :, [feature_index["log_moneyness"], feature_index["tau_fraction"]]
        ]
        masks = bundle["masks"][selected]
        reference_path = high_reference_dir / f"{record.regime_id}.npz"
        if not reference_path.exists():
            raise FileNotFoundError(f"missing high-accuracy reference: {reference_path}")
        reference = interpolate_reference(reference_path, coordinates)
        config = AmericanLCPConfig(
            record.option_type,
            record.K,
            record.T,
            record.r,
            record.q,
            record.sigma,
            record.Smax,
            record.M,
            record.N,
            tolerance=1e-12,
            obstacle_tolerance=1e-12,
        )
        for arm, solver in (("A", "psor"), ("B", "policy_iteration")):
            result = american_cn_lcp_price(config, lcp_solver=solver)
            normalized_surface = result.value_grid / record.K
            prediction = interpolate_grid_surface(
                result.spot_grid / record.K,
                result.tau_grid / record.T,
                normalized_surface,
                coordinates,
            )
            delta_surface = np.vstack(
                [
                    finite_difference_delta_nonuniform(result.spot_grid, value_row)
                    for value_row in result.value_grid
                ]
            )
            gamma_surface = record.K * np.vstack(
                [
                    finite_difference_gamma_nonuniform(result.spot_grid, value_row)
                    for value_row in result.value_grid
                ]
            )
            delta_prediction = interpolate_grid_surface(
                result.spot_grid / record.K,
                result.tau_grid / record.T,
                delta_surface,
                coordinates,
            )
            gamma_prediction = interpolate_grid_surface(
                result.spot_grid / record.K,
                result.tau_grid / record.T,
                gamma_surface,
                coordinates,
            )
            error = prediction - reference["value_over_k"]
            residual_max = max(
                step.residual.normalized_lcp_residual for step in result.lcp_results
            )
            base = {
                "arm": arm,
                "variant": "CN+PSOR" if arm == "A" else "CN+Policy Iteration",
                "network_spec": "not_applicable",
                "split": record.split,
                "regime_id": record.regime_id,
                "seed": "deterministic",
                "region": "all",
                "reference_source": "DIRK_policy_sinh_M480_N960",
                **_error_metrics(error, reference["value_over_k"]),
                **_classical_boundary_metrics(result, reference["boundary_over_k"], coordinates),
                "normalized_lcp_residual_max": residual_max,
                "solver_converged": result.converged,
                "runtime_seconds": result.total_seconds,
                "total_lcp_iterations": sum(step.iterations for step in result.lcp_results),
            }
            delta_mask = masks[:, mask_index["delta_allowed_mask"]]
            gamma_mask = masks[:, mask_index["gamma_allowed_mask"]]
            base["delta_rmse"] = _rmse(
                delta_prediction[delta_mask] - reference["delta"][delta_mask]
            )
            base["scaled_gamma_rmse"] = _rmse(
                gamma_prediction[gamma_mask] - reference["scaled_gamma"][gamma_mask]
            )
            rows.append(base)
            for region in (
                "exercise_region",
                "continuation_region",
                "boundary_near",
                "payoff_kink_near",
                "maturity_row",
                "strict_interior",
            ):
                region_mask = masks[:, mask_index[region]]
                if np.any(region_mask):
                    rows.append(
                        {
                            **base,
                            "region": region,
                            **_error_metrics(
                                error[region_mask], reference["value_over_k"][region_mask]
                            ),
                        }
                    )
    _write_csv(output / "classical_baseline_metrics.csv", rows)
    return rows


def _error_metrics(error: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    return {
        "sample_count": int(len(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": _rmse(error),
        "max_abs_error": float(np.max(np.abs(error))),
        "relative_l2": float(
            np.linalg.norm(error) / max(np.linalg.norm(reference), np.finfo(float).eps)
        ),
    }


def _vi_metrics(diagnostics: dict[str, np.ndarray]) -> dict[str, float]:
    combined = np.maximum.reduce(
        (
            diagnostics["obstacle_violation"],
            diagnostics["equation_violation"],
            np.abs(diagnostics["fb"]),
        )
    )
    return {
        "obstacle_max": float(np.max(diagnostics["obstacle_violation"])),
        "equation_max": float(np.max(diagnostics["equation_violation"])),
        "complementarity_max": float(np.max(np.abs(diagnostics["complementarity"]))),
        "fb_max": float(np.max(np.abs(diagnostics["fb"]))),
        "vi_p95": float(np.quantile(combined, 0.95)),
    }


def _boundary_metrics(
    loaded: Any,
    coordinates: np.ndarray,
    reference_boundary_by_row: np.ndarray,
    *,
    K: float,
    Smax: float,
    M: int,
    device: str,
) -> tuple[dict[str, float | int], Any]:
    unique_s = np.unique(coordinates[:, 1])
    spots = np.linspace(0.0, Smax, M + 1)
    taus = unique_s * loaded.problem.T
    prediction = predict_pinn_surface(loaded, spots, taus, K=K, device=device)
    payoff_function = call_payoff if loaded.problem.option_type == "call" else put_payoff
    payoff = np.asarray(payoff_function(spots, K), dtype=float)
    premium = prediction.value_grid - payoff[np.newaxis, :]
    predicted_boundary = np.full(len(unique_s), np.nan, dtype=float)
    reference_boundary = np.full(len(unique_s), np.nan, dtype=float)
    for index, normalized_time in enumerate(unique_s):
        point = extract_boundary_at_time(
            spots,
            premium[index],
            loaded.problem.option_type,
            float(taus[index]),
            index,
            threshold=1e-6,
        )
        if point.boundary_found:
            predicted_boundary[index] = point.boundary_spot / K
        matching = np.isclose(coordinates[:, 1], normalized_time)
        candidates = reference_boundary_by_row[matching]
        finite = candidates[np.isfinite(candidates)]
        if finite.size:
            reference_boundary[index] = float(np.median(finite))
    predicted_found = np.isfinite(predicted_boundary)
    reference_found = np.isfinite(reference_boundary)
    both = predicted_found & reference_found
    differences = predicted_boundary[both] - reference_boundary[both]
    return {
        "boundary_reference_found": int(np.count_nonzero(reference_found)),
        "boundary_predicted_found": int(np.count_nonzero(predicted_found)),
        "boundary_found_rate": float(np.mean(predicted_found == reference_found)),
        "boundary_missed_count": int(np.count_nonzero(reference_found & ~predicted_found)),
        "boundary_false_count": int(np.count_nonzero(~reference_found & predicted_found)),
        "boundary_conditional_mae": (
            float(np.mean(np.abs(differences))) if differences.size else float("nan")
        ),
        "boundary_conditional_rmse": _rmse(differences) if differences.size else float("nan"),
    }, prediction


def _classical_boundary_metrics(
    result: Any,
    reference_boundary_by_row: np.ndarray,
    coordinates: np.ndarray,
) -> dict[str, float | int]:
    predicted_boundary = np.full(len(result.tau_grid), np.nan, dtype=float)
    reference_boundary = np.full(len(result.tau_grid), np.nan, dtype=float)
    premium = result.value_grid - result.payoff[np.newaxis, :]
    normalized_time = result.tau_grid / result.config.T
    for index, tau in enumerate(result.tau_grid):
        point = extract_boundary_at_time(
            result.spot_grid,
            premium[index],
            result.config.option_type,
            float(tau),
            index,
            threshold=1e-6,
        )
        if point.boundary_found:
            predicted_boundary[index] = point.boundary_spot / result.config.K
        matching = np.isclose(coordinates[:, 1], normalized_time[index])
        candidates = reference_boundary_by_row[matching]
        finite = candidates[np.isfinite(candidates)]
        if finite.size:
            reference_boundary[index] = float(np.median(finite))
    predicted_found = np.isfinite(predicted_boundary)
    reference_found = np.isfinite(reference_boundary)
    both = predicted_found & reference_found
    differences = predicted_boundary[both] - reference_boundary[both]
    return {
        "boundary_reference_found": int(np.count_nonzero(reference_found)),
        "boundary_predicted_found": int(np.count_nonzero(predicted_found)),
        "boundary_found_rate": float(np.mean(predicted_found == reference_found)),
        "boundary_missed_count": int(np.count_nonzero(reference_found & ~predicted_found)),
        "boundary_false_count": int(np.count_nonzero(~reference_found & predicted_found)),
        "boundary_conditional_mae": (
            float(np.mean(np.abs(differences))) if differences.size else float("nan")
        ),
        "boundary_conditional_rmse": _rmse(differences) if differences.size else float("nan"),
    }


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _network_name(spec: dict[str, Any]) -> str:
    if spec.get("architecture") == "resnet":
        return f"resnet_{int(spec['blocks'])}x{int(spec['layers_per_block'])}x{int(spec['width'])}"
    if spec.get("architecture") == "mlp":
        return f"mlp_{int(spec['hidden_layers'])}x{int(spec['width'])}"
    raise ValueError("unknown network specification")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
