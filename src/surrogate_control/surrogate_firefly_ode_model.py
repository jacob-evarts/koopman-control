from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from control_abms import BaseModel

from surrogate_control.firefly_controllers import beacon_on
from surrogate_control.firefly_ode import (
    firefly_ode_rhs,
    initial_velocity,
    resting_from_flashing,
)


class SurrogateFireflyODEModel(BaseModel):
    """Van der Pol–style cyclic surrogate with ``external_flash`` control."""

    def __init__(
        self,
        params: np.ndarray | Sequence[float],
        num_fireflies: float,
        initial_flashing: float,
        initial_resting: float | None = None,
        *,
        initial_velocity_value: float | None = None,
        flashing_history: Sequence[float] | None = None,
    ) -> None:
        self.params = np.asarray(params, dtype=float)
        self.n_total = float(num_fireflies)
        self.F = float(initial_flashing)
        if initial_velocity_value is not None:
            self.V = float(initial_velocity_value)
        elif flashing_history is not None and len(flashing_history) >= 2:
            self.V = initial_velocity(flashing_history)
        else:
            self.V = 0.0
        self.R = float(
            initial_resting
            if initial_resting is not None
            else resting_from_flashing(self.F, self.n_total)
        )
        self.timestep = 0
        self.history: dict[str, list[float]] = {
            "flashing": [self.F],
            "resting": [self.R],
            "u": [0.0],
        }

    def step(self, control_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        u = 1.0 if beacon_on(control_inputs) else 0.0
        sol = solve_ivp(
            firefly_ode_rhs,
            (0.0, 1.0),
            [self.F, self.V],
            args=(self.params, u, self.n_total),
            method="RK45",
            rtol=1e-4,
            atol=1e-6,
        )
        self.F = float(np.clip(sol.y[0, -1], 0.0, self.n_total))
        self.V = float(sol.y[1, -1])
        self.R = resting_from_flashing(self.F, self.n_total)
        self.timestep += 1
        self.history["flashing"].append(self.F)
        self.history["resting"].append(self.R)
        self.history["u"].append(u)
        return self.get_outputs()

    def get_outputs(self) -> dict[str, Any]:
        return {
            "flashing_count": self.F,
            "resting_count": self.R,
            "timestep": self.timestep,
        }

    def get_history(self) -> dict[str, list[float]]:
        return self.history

    def get_state_grid(self):
        raise NotImplementedError("SurrogateFireflyODEModel has no spatial grid.")

    def close_h5(self) -> None:
        pass
