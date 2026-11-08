"""Deterministic global sampling for the Vanilla PINN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from american_risk_surfaces.pinn.formulation import PINNProblem


@dataclass(frozen=True)
class SamplingBatch:
    interior: Any
    terminal_x: Any
    boundary_s: Any
    component_counts: dict[str, int]


class PINNSampler:
    """A seeded sampler with the archived checkpoint state format."""

    def __init__(
        self,
        problem: PINNProblem,
        *,
        seed: int,
        pool_size: int = 65536,
        candidate_size: int = 16384,
        device: str = "cpu",
        dtype: Any = None,
    ) -> None:
        torch = _torch()
        self.problem = problem
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = torch.float64 if dtype is None else dtype
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.numpy_rng = np.random.default_rng(self.seed)
        self.global_pool = self._sobol(pool_size, self.seed)
        adaptive_size = max(1, int(round(0.20 * pool_size)))
        self.adaptive_pool = self.global_pool[:adaptive_size].clone()

    def state_dict(self) -> dict[str, Any]:
        return {
            "torch_generator_state": self.generator.get_state(),
            "numpy_generator_state": self.numpy_rng.bit_generator.state,
            "adaptive_pool": self.adaptive_pool.clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # ``torch.load(..., map_location="cuda")`` also moves this CPU RNG
        # byte state to CUDA. CPU Generators only accept a CPU ByteTensor, so
        # normalize it here before restoring interrupted Windows CUDA runs.
        generator_state = _torch().as_tensor(state["torch_generator_state"]).cpu()
        if generator_state.dtype != _torch().uint8:
            generator_state = generator_state.to(dtype=_torch().uint8)
        self.generator.set_state(generator_state.contiguous())
        self.numpy_rng.bit_generator.state = state["numpy_generator_state"]
        adaptive = _torch().as_tensor(state["adaptive_pool"], dtype=self.dtype).cpu()
        if adaptive.ndim != 2 or adaptive.shape[1] != 2:
            raise ValueError("checkpoint adaptive sampler state has the wrong shape.")
        self.adaptive_pool = adaptive.clone()

    def sample(
        self,
        batch_size: int,
        boundary_batch_size: int,
        *,
        mode: str,
        time_upper: float = 1.0,
    ) -> SamplingBatch:
        torch = _torch()
        if batch_size < 4 or boundary_batch_size < 2:
            raise ValueError("batch sizes are too small.")
        if not 0.0 < time_upper <= 1.0:
            raise ValueError("time_upper must be in (0, 1].")
        if mode != "global":
            raise ValueError("the Vanilla PINN uses global sampling")
        interior = self._draw_pool(self.global_pool, batch_size)
        counts = {"global": batch_size, "maturity": 0, "strike": 0, "adaptive": 0}
        interior = interior.clone()
        interior[:, 1] *= time_upper
        terminal_indices = torch.randint(
            len(self.global_pool), (boundary_batch_size,), generator=self.generator
        )
        terminal_x = self.global_pool[terminal_indices, 0:1]
        boundary_s = torch.rand(
            (boundary_batch_size, 1), generator=self.generator, dtype=self.dtype
        ) * time_upper
        return SamplingBatch(
            interior=interior.to(self.device),
            terminal_x=terminal_x.to(self.device),
            boundary_s=boundary_s.to(self.device),
            component_counts=counts,
        )

    def _sobol(self, size: int, seed: int) -> Any:
        torch = _torch()
        if size < 1:
            raise ValueError("pool sizes must be positive.")
        unit = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed).draw(size)
        unit = unit.to(dtype=self.dtype)
        unit[:, 0] = self.problem.x_min + unit[:, 0] * (
            self.problem.x_max - self.problem.x_min
        )
        return unit

    def _draw_pool(self, pool: Any, count: int) -> Any:
        torch = _torch()
        indices = torch.randint(len(pool), (count,), generator=self.generator)
        return pool[indices]

def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for PINN sampling.") from exc
    return torch
