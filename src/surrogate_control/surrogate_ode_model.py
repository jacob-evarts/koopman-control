from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from control_abms import BaseModel

from surrogate_control.rabbit_grass_ode import ode_rhs

DEFAULT_G_MAX = 64.0 * 64.0


class SurrogateODEModel(BaseModel):
    """Fitted logistic-grass surrogate (same parameter names as the notebook)."""

    def __init__(
        self,
        params: np.ndarray | Sequence[float],
        initial_rabbits: float = 100.0,
        initial_grass: float | None = None,
        g_max: float = DEFAULT_G_MAX,
    ) -> None:
        self.params = np.asarray(params, dtype=float)
        self.g_max = float(g_max)
        self.R = float(initial_rabbits)
        self.G = float(initial_grass if initial_grass is not None else self.g_max / 2)
        self.timestep = 0
        self.history: dict[str, list[float]] = {"rabbits": [self.R], "grass": [self.G], "u": [0.0]}

    def step(self, control_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        u = float((control_inputs or {}).get("cull", 0.0))
        sol = solve_ivp(
            ode_rhs,
            (0.0, 1.0),
            [self.R, self.G],
            args=(self.params, u, self.g_max),
            method="RK45",
            rtol=1e-4,
            atol=1e-6,
        )
        self.R = max(float(sol.y[0, -1]), 0.0)
        self.G = max(float(sol.y[1, -1]), 0.0)
        self.timestep += 1
        self.history["rabbits"].append(self.R)
        self.history["grass"].append(self.G)
        self.history["u"].append(u)
        return self.get_outputs()

    def get_outputs(self) -> dict[str, Any]:
        return {
            "rabbit_count": self.R,
            "grass_count": self.G,
            "timestep": self.timestep,
        }

    def get_history(self) -> dict[str, list[float]]:
        return self.history

    def get_state_grid(self):
        raise NotImplementedError("SurrogateODEModel has no spatial grid.")

    def close_h5(self) -> None:
        pass
