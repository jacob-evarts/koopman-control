from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from control_abms import BaseModel


def snapshot_grids(model: BaseModel) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(grass, rabbits)`` binary grids from a ``RabbitGrassModel``-like instance."""
    grass = np.asarray(model.grass_grid, dtype=np.float32)
    rabbits = np.zeros_like(grass)
    for r in model.rabbits:
        rabbits[int(r["x"]), int(r["y"])] = 1.0
    return grass, rabbits


def run_abm_with_grids(
    run_simulation_fn: Callable[..., dict],
    build_model_fn: Callable[[], BaseModel],
    controller: Any,
    steps: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run an ABM rollout and record ``(T+1, H, W)`` grass / rabbit grids each step."""
    model = build_model_fn()
    grass_frames: list[np.ndarray] = []
    rabbit_frames: list[np.ndarray] = []

    g0, r0 = snapshot_grids(model)
    grass_frames.append(g0)
    rabbit_frames.append(r0)

    def _callback(_t: int, mdl: BaseModel, _outputs: dict, _control: dict) -> None:
        g, r = snapshot_grids(mdl)
        grass_frames.append(g)
        rabbit_frames.append(r)

    hist = run_simulation_fn(model, controller=controller, steps=steps, step_callback=_callback)
    grass_stack = np.stack(grass_frames, axis=0)
    rabbit_stack = np.stack(rabbit_frames, axis=0)
    return hist, grass_stack, rabbit_stack


def grids_to_cnn_input(rabbit_stack: np.ndarray) -> np.ndarray:
    """Rabbit occupancy to ``(T, 1, H, W)`` float32 in ``[0, 1]``."""
    r = np.asarray(rabbit_stack, dtype=np.float32)
    if r.ndim == 2:
        return r[np.newaxis, np.newaxis, :, :]
    if r.ndim == 3:
        return r[:, np.newaxis, :, :]
    raise ValueError(f"rabbit_stack must be (H, W) or (T, H, W), got shape {r.shape}")


def save_trajectory_h5(path: Path | str, grass_stack: np.ndarray, rabbit_stack: np.ndarray) -> None:
    """Write rabbit–grass grids in the same layout as ``RabbitGrassDataset``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("grass", data=grass_stack.astype(np.float32), compression="gzip")
        h5f.create_dataset("rabbits", data=rabbit_stack.astype(np.float32), compression="gzip")


@torch.no_grad()
def encode_grid_stack(
    koopman: torch.nn.Module,
    x: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    """Encode ``(T, C, H, W)`` grids to flattened latents ``(T, D)``."""
    device = device or next(koopman.parameters()).device
    xt = torch.from_numpy(x).to(device)
    latents = []
    for t in range(xt.shape[0]):
        z = koopman.encode_flat(xt[t : t + 1])
        latents.append(z.cpu().numpy().reshape(-1))
    return np.stack(latents, axis=0)


def trajectory_to_latent(
    koopman: torch.nn.Module,
    rabbit_stack: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    """Convenience: rabbit grids → CNN input → latent sequence ``(T, D)``."""
    x = grids_to_cnn_input(rabbit_stack)
    return encode_grid_stack(koopman, x, device=device)


def load_koopman_cnn(checkpoint: Path | str, map_location: str | torch.device = "cpu"):
    """Load a trained ``KoopmanCNN`` Lightning checkpoint."""
    from koopman_control.models.koopman_cnn import KoopmanCNN

    model = KoopmanCNN.load_from_checkpoint(str(checkpoint), map_location=map_location)
    model.eval()
    return model


def load_koopman_cnn_dynamics(checkpoint: Path | str, map_location: str | torch.device = "cpu"):
    """Load a trained ``KoopmanCNNDynamics`` Lightning checkpoint."""
    from koopman_control.models.koopman_cnn_dynamics import KoopmanCNNDynamics

    model = KoopmanCNNDynamics.load_from_checkpoint(str(checkpoint), map_location=map_location)
    model.eval()
    return model