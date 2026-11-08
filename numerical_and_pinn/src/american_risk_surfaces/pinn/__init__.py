"""SURF physics-informed American-option solver experiments."""

from american_risk_surfaces.pinn.evaluation import (
    LoadedPINN,
    PINNSurfacePrediction,
    discrete_lcp_audit,
    evaluate_pinn_vi,
    load_pinn_checkpoint,
    predict_pinn_surface,
)
from american_risk_surfaces.pinn.formulation import PINNProblem
from american_risk_surfaces.pinn.networks import NetworkSpec
from american_risk_surfaces.pinn.training import (
    PINNRunResult,
    PINNTrainingConfig,
    train_single_regime_pinn,
)

__all__ = (
    "LoadedPINN",
    "NetworkSpec",
    "PINNProblem",
    "PINNRunResult",
    "PINNSurfacePrediction",
    "PINNTrainingConfig",
    "discrete_lcp_audit",
    "evaluate_pinn_vi",
    "load_pinn_checkpoint",
    "predict_pinn_surface",
    "train_single_regime_pinn",
)
