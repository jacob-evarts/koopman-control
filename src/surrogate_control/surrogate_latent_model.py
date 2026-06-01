from __future__ import annotations

from typing import Any

import numpy as np
import torch

from control_abms import BaseModel

from surrogate_control.rabbit_grass_latent import blend_latent_vector


def latent_vector_dim(koopman: Any) -> int:
    """Flat latent size from ``encode`` / ``encode_flat`` (vector or spatial-flattened)."""
    if hasattr(koopman, "decode_latent"):
        return int(koopman.latent_dim)
    return int(koopman.latent_dim_total)


def _flat_to_latent_tensor(z_flat: np.ndarray, koopman: Any) -> torch.Tensor:
    """``(1, latent_dim)`` for dynamics CNN, else ``(1, C, H, W)`` for legacy ``KoopmanCNN``."""
    z = np.asarray(z_flat, dtype=np.float32).reshape(-1)
    d_vec = int(getattr(koopman, "latent_dim", -1))
    if hasattr(koopman, "decode_latent") and z.size == d_vec:
        return torch.from_numpy(z).float().unsqueeze(0)
    c = koopman.latent_channels
    h = w = koopman.latent_spatial
    if z.size == c * h * w:
        return torch.from_numpy(z.reshape(c, h, w)).float().unsqueeze(0)
    raise ValueError(
        f"Latent length {z.size} does not match vector dim {d_vec} "
        f"or spatial flat dim {c * h * w}"
    )


@torch.no_grad()
def decode_latent_to_grids(
    koopman: Any,
    z_flat: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Decode flat latent to rabbit probability map ``(H, W)`` (or grass+rabbit for legacy 2-ch)."""
    z = _flat_to_latent_tensor(z_flat, koopman).to(device)
    if hasattr(koopman, "decode_latent"):
        x = koopman.decode_latent(z)[0].cpu().numpy()
    else:
        x = koopman.decode(z)[0].cpu().numpy()
    if x.shape[0] == 1:
        return x[0]
    return x[1]


@torch.no_grad()
def dynamics_one_step_flat(
    koopman: Any,
    z_flat: np.ndarray,
    u: float = 0.0,
    device: torch.device | None = None,
) -> np.ndarray:
    """Apply trained dynamics; return flat latent for the next step."""
    device = device or next(koopman.parameters()).device
    z = _flat_to_latent_tensor(z_flat, koopman).to(device)
    if hasattr(koopman, "step_dynamics"):
        u_t = torch.tensor([[float(u)]], device=device, dtype=z.dtype)
        return koopman.step_dynamics(z, u_t)[0].cpu().numpy()
    z_next = koopman.linear_dynamics(z)
    return z_next.flatten().cpu().numpy()


@torch.no_grad()
def decode_observables(koopman: Any, z_flat: np.ndarray, device: torch.device) -> tuple[float, float]:
    """Decode a flat latent to approximate rabbit count (grass not modeled; returns 0)."""
    rabbit_map = decode_latent_to_grids(koopman, z_flat, device)
    rabbits = float((rabbit_map > 0.5).sum())
    return rabbits, 0.0


class SurrogateLatentLinearModel(BaseModel):
    """Discrete linear latent surrogate with CNN decode for scalar outputs."""

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
        self.history: dict[str, list[float]] = {"rabbits": [], "grass": [], "u": [0.0]}
        r0, g0 = decode_observables(self.koopman, self.z, self.device)
        self.history["rabbits"].append(r0)
        self.history["grass"].append(g0)

    def step(self, control_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        u = float((control_inputs or {}).get("cull", 0.0))
        self.z = self.a @ self.z + self.b * u + self.bias
        self.timestep += 1
        r, g = decode_observables(self.koopman, self.z, self.device)
        self.history["rabbits"].append(r)
        self.history["grass"].append(g)
        self.history["u"].append(u)
        return self.get_outputs()

    def get_outputs(self) -> dict[str, Any]:
        r = self.history["rabbits"][-1] if self.history["rabbits"] else 0.0
        g = self.history["grass"][-1] if self.history["grass"] else 0.0
        return {"rabbit_count": r, "grass_count": g, "timestep": self.timestep}

    def get_history(self) -> dict[str, list[float]]:
        return self.history

    def get_state_grid(self):
        raise NotImplementedError("SurrogateLatentLinearModel has no spatial grid.")

    def close_h5(self) -> None:
        pass


class SurrogateLatentSplineModel(BaseModel):
    """Latent trajectory = blend of spline fits (uncontrolled vs full cull), CNN decode for outputs."""

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
        self.history: dict[str, list[float]] = {"rabbits": [], "grass": [], "u": [0.0]}
        r, g = decode_observables(self.koopman, self.z, self.device)
        self.history["rabbits"].append(r)
        self.history["grass"].append(g)

    def step(self, control_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        u = float((control_inputs or {}).get("cull", 0.0))
        t_next = float(self.timestep + 1)
        self.z = blend_latent_vector(self.splines, t_next, u)
        self.timestep += 1
        r, g = decode_observables(self.koopman, self.z, self.device)
        self.history["rabbits"].append(r)
        self.history["grass"].append(g)
        self.history["u"].append(u)
        return self.get_outputs()

    def get_outputs(self) -> dict[str, Any]:
        return {
            "rabbit_count": self.history["rabbits"][-1],
            "grass_count": self.history["grass"][-1],
            "timestep": self.timestep,
        }

    def get_history(self) -> dict[str, list[float]]:
        return self.history

    def get_state_grid(self):
        raise NotImplementedError("SurrogateLatentSplineModel has no spatial grid.")

    def close_h5(self) -> None:
        pass
