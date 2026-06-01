from __future__ import annotations

from typing import Any

from scipy.interpolate import CubicSpline

from control_abms import BaseModel

from surrogate_control.rabbit_grass_spline import blend_spline_state

DEFAULT_G_MAX = 64.0 * 64.0


class SurrogateSplineModel(BaseModel):
    """Blend of spline fits to mean uncontrolled vs mean cull-on trajectories.

    At integer time ``t``, state is ``(1-u)*traj_uc(t) + u*traj_c(t)`` in each
    component, matching open-loop schedules when ``u`` is constant.
    """

    def __init__(
        self,
        spl_uc_R: CubicSpline,
        spl_uc_G: CubicSpline,
        spl_c_R: CubicSpline,
        spl_c_G: CubicSpline,
        initial_rabbits: float,
        initial_grass: float,
        g_max: float = DEFAULT_G_MAX,
        t_min: float = 0.0,
        t_max: float | None = None,
    ) -> None:
        self.spl_uc_R = spl_uc_R
        self.spl_uc_G = spl_uc_G
        self.spl_c_R = spl_c_R
        self.spl_c_G = spl_c_G
        self.g_max = float(g_max)
        self.t_min = float(t_min)
        if t_max is None:
            self.t_max = float(
                max(
                    spl_uc_R.x[-1],
                    spl_uc_G.x[-1],
                    spl_c_R.x[-1],
                    spl_c_G.x[-1],
                )
            )
        else:
            self.t_max = float(t_max)
        self.R = float(initial_rabbits)
        self.G = float(initial_grass)
        self.timestep = 0
        self.history: dict[str, list[float]] = {"rabbits": [self.R], "grass": [self.G], "u": [0.0]}

    def step(self, control_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        u = float((control_inputs or {}).get("cull", 0.0))
        t_next = float(self.timestep + 1)
        self.R, self.G = blend_spline_state(
            self.spl_uc_R,
            self.spl_uc_G,
            self.spl_c_R,
            self.spl_c_G,
            t_next,
            u,
            self.t_min,
            self.t_max,
        )
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
        raise NotImplementedError("SurrogateSplineModel has no spatial grid.")

    def close_h5(self) -> None:
        pass
