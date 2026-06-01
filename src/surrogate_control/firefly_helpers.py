from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from control_abms import BaseController, BaseModel

from surrogate_control.firefly_controllers import beacon_on
from surrogate_control.rabbit_grass_latent import blend_latent_vector
from surrogate_control.surrogate_latent_model import (
    decode_latent_to_grids,
    decode_observables,
    dynamics_one_step_flat,
    latent_vector_dim,
)


class ConstantFlashController(BaseController):
    """Centre arena beacon on every step (open-loop training trajectory)."""

    def compute(self, timestep: int, outputs: dict) -> dict:  # noqa: ARG002
        return {"external_flash": True}


def snapshot_firefly_grids(model: BaseModel) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(resting, flashing)`` occupancy grids from a ``FireflyModel``."""
    flashing = model.clock < model.threshold
    resting = np.zeros((model.width, model.height), dtype=np.float32)
    flashing_grid = np.zeros((model.width, model.height), dtype=np.float32)
    resting[model.x[~flashing], model.y[~flashing]] = 1.0
    flashing_grid[model.x[flashing], model.y[flashing]] = 1.0
    return resting, flashing_grid


def run_abm_with_firefly_grids(
    run_simulation_fn: Callable[..., dict],
    build_model_fn: Callable[[], BaseModel],
    controller: Any,
    steps: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run a firefly rollout; record ``(T+1, W, H)`` resting / flashing grids."""
    model = build_model_fn()
    resting_frames: list[np.ndarray] = []
    flashing_frames: list[np.ndarray] = []

    r0, f0 = snapshot_firefly_grids(model)
    resting_frames.append(r0)
    flashing_frames.append(f0)

    def _callback(_t: int, mdl: BaseModel, _outputs: dict, _control: dict) -> None:
        r, f = snapshot_firefly_grids(mdl)
        resting_frames.append(r)
        flashing_frames.append(f)

    hist = run_simulation_fn(model, controller=controller, steps=steps, step_callback=_callback)
    return hist, np.stack(resting_frames, axis=0), np.stack(flashing_frames, axis=0)


def flashing_to_cnn_input(flashing_stack: np.ndarray) -> np.ndarray:
    """Flashing occupancy to ``(T, 1, H, W)`` float32 in ``[0, 1]``."""
    f = np.asarray(flashing_stack, dtype=np.float32)
    if f.ndim == 2:
        return f[np.newaxis, np.newaxis, :, :]
    if f.ndim == 3:
        return f[:, np.newaxis, :, :]
    raise ValueError(f"flashing_stack must be (H, W) or (T, H, W), got shape {f.shape}")


def save_firefly_trajectory_h5(
    path: Path | str,
    resting_stack: np.ndarray,
    flashing_stack: np.ndarray,
    control: np.ndarray | None = None,
) -> None:
    """Write firefly grids (same layout as ``FireflyDataset`` / training generator)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("resting", data=resting_stack.astype(np.float32), compression="gzip")
        h5f.create_dataset("flashing", data=flashing_stack.astype(np.float32), compression="gzip")
        if control is not None:
            h5f.create_dataset("control", data=np.asarray(control, dtype=np.float32), compression="gzip")


@torch.no_grad()
def trajectory_to_latent_flashing(
    koopman: torch.nn.Module,
    flashing_stack: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    """Flashing grids → CNN input → latent sequence ``(T, D)``."""
    from surrogate_control.latent_encoding import encode_grid_stack

    x = flashing_to_cnn_input(flashing_stack)
    return encode_grid_stack(koopman, x, device=device)


@torch.no_grad()
def decode_observables_firefly(
    koopman: Any,
    z_flat: np.ndarray,
    device: torch.device,
    thresh: float = 0.5,
) -> tuple[float, float]:
    """Decode latent to approximate flashing and resting cell counts."""
    flash_map = decode_latent_to_grids(koopman, z_flat, device)
    flashing = float((flash_map > thresh).sum())
    resting = float(flash_map.size - flashing)
    return flashing, resting


class FireflySurrogateLatentLinearModel(BaseModel):
    """Linear latent surrogate with ``external_flash`` control and flashing-count output."""

    def __init__(
        self,
        koopman: Any,
        a: np.ndarray,
        b: np.ndarray,
        bias: np.ndarray,
        z0: np.ndarray,
        device: torch.device | None = None,
    ) -> None:
        self.koopman = koopman
        self.device = device or next(koopman.parameters()).device
        self.a = np.asarray(a, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self.bias = np.asarray(bias, dtype=float)
        self.z = np.asarray(z0, dtype=float).reshape(-1)
        self.timestep = 0
        self.history: dict[str, list[float]] = {"flashing": [], "resting": [], "u": [0.0]}
        f0, r0 = decode_observables_firefly(self.koopman, self.z, self.device)
        self.history["flashing"].append(f0)
        self.history["resting"].append(r0)

    def step(self, control_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        u = 1.0 if beacon_on(control_inputs) else 0.0
        self.z = self.a @ self.z + self.b * u + self.bias
        self.timestep += 1
        f, r = decode_observables_firefly(self.koopman, self.z, self.device)
        self.history["flashing"].append(f)
        self.history["resting"].append(r)
        self.history["u"].append(u)
        return self.get_outputs()

    def get_outputs(self) -> dict[str, Any]:
        return {
            "flashing_count": self.history["flashing"][-1],
            "resting_count": self.history["resting"][-1],
            "timestep": self.timestep,
        }

    def get_history(self) -> dict[str, list[float]]:
        return self.history

    def get_state_grid(self):
        raise NotImplementedError("FireflySurrogateLatentLinearModel has no spatial grid.")

    def close_h5(self) -> None:
        pass


class FireflySurrogateLatentSplineModel(BaseModel):
    """Spline-blended latent surrogate for firefly (blend weight = ``external_flash``)."""

    def __init__(
        self,
        koopman: Any,
        splines: dict[str, Any],
        z0: np.ndarray,
        device: torch.device | None = None,
    ) -> None:
        self.koopman = koopman
        self.splines = splines
        self.device = device or next(koopman.parameters()).device
        self.z = np.asarray(z0, dtype=float).reshape(-1)
        self.timestep = 0
        self.history: dict[str, list[float]] = {"flashing": [], "resting": [], "u": [0.0]}
        f, r = decode_observables_firefly(self.koopman, self.z, self.device)
        self.history["flashing"].append(f)
        self.history["resting"].append(r)

    def step(self, control_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        u = 1.0 if beacon_on(control_inputs) else 0.0
        t_next = float(self.timestep + 1)
        self.z = blend_latent_vector(self.splines, t_next, u)
        self.timestep += 1
        f, r = decode_observables_firefly(self.koopman, self.z, self.device)
        self.history["flashing"].append(f)
        self.history["resting"].append(r)
        self.history["u"].append(u)
        return self.get_outputs()

    def get_outputs(self) -> dict[str, Any]:
        return {
            "flashing_count": self.history["flashing"][-1],
            "resting_count": self.history["resting"][-1],
            "timestep": self.timestep,
        }

    def get_history(self) -> dict[str, list[float]]:
        return self.history

    def get_state_grid(self):
        raise NotImplementedError("FireflySurrogateLatentSplineModel has no spatial grid.")

    def close_h5(self) -> None:
        pass


def run_abm_replicas_firefly(
    run_abm: Callable[..., dict[str, Any]],
    controller_factory: Callable[[], Any],
    n_seeds: int,
    num_fireflies: int,
    strategy: str,
    steps: int,
    ic_idx: int,
    seed_base: int = 10_000,
    seed_stride: int = 1_000,
) -> dict[str, np.ndarray]:
    """Stack ``flashing_count``, ``resting`` proxy, and ``external_flash`` over seeds."""
    F_mat = np.empty((n_seeds, steps + 1))
    R_mat = np.empty((n_seeds, steps + 1))
    U_mat = np.empty((n_seeds, steps + 1))
    for seed in range(n_seeds):
        hist = run_abm(
            num_fireflies=num_fireflies,
            strategy=strategy,
            steps=steps,
            seed=seed_base + seed_stride * ic_idx + seed,
            controller=controller_factory(),
        )
        F_mat[seed] = hist["flashing_count"]
        R_mat[seed] = hist.get("resting_count", np.zeros(steps + 1))
        U_mat[seed] = np.asarray(hist["external_flash"], dtype=float)
    return {"F": F_mat, "R": R_mat, "u": U_mat}


def compute_metrics_firefly(
    F_mat: np.ndarray,
    u_mat: np.ndarray,
    setpoint: float,
    warmup: int = 20,
) -> dict[str, float]:
    """Tracking / variance / effort metrics on flashing-count replicas."""
    F_ss = F_mat[:, warmup:]
    u_ss = u_mat[:, warmup:]
    err = np.abs(F_ss - setpoint).mean(axis=1)
    var = F_ss.var(axis=1)
    effort = u_ss.mean(axis=1)
    return {
        "mean_tracking_error": float(err.mean()),
        "std_tracking_error": float(err.std()),
        "mean_variance": float(var.mean()),
        "mean_control_effort": float(effort.mean()),
    }
