from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from control_abms import BaseController


class PIDController(BaseController):
    """Discrete-time PID with backward difference on error (same scaling as PI in control_abms)."""

    def __init__(
        self,
        output_key: str,
        control_key: str,
        setpoint: float,
        Kp: float = 1.0,
        Ki: float = 0.0,
        Kd: float = 0.0,
        u_min: float = 0.0,
        u_max: float = 1.0,
    ) -> None:
        self.output_key = output_key
        self.control_key = control_key
        self.setpoint = setpoint
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.u_min = u_min
        self.u_max = u_max
        self._integral: float = 0.0
        self._prev_error: float | None = None

    def compute(self, timestep: int, outputs: dict) -> dict:  # noqa: ARG002
        error = self.setpoint - outputs.get(self.output_key, 0)
        self._integral += error
        if self._prev_error is None:
            derr = 0.0
        else:
            derr = error - self._prev_error
        self._prev_error = error
        u = self.Kp * error + self.Ki * self._integral + self.Kd * derr
        u = float(max(self.u_min, min(self.u_max, u)))
        return {self.control_key: u}

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None


class ConstantCullController(BaseController):
    """Always requests a cull (used to generate a simple controlled training trajectory)."""

    def compute(self, t, outputs):  # noqa: ARG002
        return {"cull": True}


def run_abm_replicas(
    run_abm: Callable[..., dict[str, Any]],
    controller_factory: Callable[[], Any],
    n_seeds: int,
    initial_rabbits: int,
    initial_grass_prob: float,
    steps: int,
    ic_idx: int,
    seed_base: int = 10_000,
    seed_stride: int = 1_000,
    ) -> dict[str, np.ndarray]:
    """Run ``n_seeds`` ABM rollouts and stack ``rabbits``, ``grass``, ``u`` as (n_seeds, steps+1)."""
    R_mat = np.empty((n_seeds, steps + 1))
    G_mat = np.empty((n_seeds, steps + 1))
    U_mat = np.empty((n_seeds, steps + 1))
    for seed in range(n_seeds):
        hist = run_abm(
            initial_rabbits=initial_rabbits,
            initial_grass_prob=initial_grass_prob,
            steps=steps,
            seed=seed_base + seed_stride * ic_idx + seed,
            controller=controller_factory(),
        )
        R_mat[seed] = hist["rabbits"]
        G_mat[seed] = hist["grass"]
        U_mat[seed] = hist["u"]
    return {"R": R_mat, "G": G_mat, "u": U_mat}


def compute_metrics(
    R_mat: np.ndarray,
    u_mat: np.ndarray,
    setpoint: float,
    warmup: int = 20,
) -> dict[str, float]:
    """Per-replica metrics aggregated to means (and std of tracking error across replicas)."""
    R_ss = R_mat[:, warmup:]
    u_ss = u_mat[:, warmup:]
    err = np.abs(R_ss - setpoint).mean(axis=1)
    var = R_ss.var(axis=1)
    effort = u_ss.mean(axis=1)
    return {
        "mean_tracking_error": float(err.mean()),
        "std_tracking_error": float(err.std()),
        "mean_variance": float(var.mean()),
        "mean_control_effort": float(effort.mean()),
    }
