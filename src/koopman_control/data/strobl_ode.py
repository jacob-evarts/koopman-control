"""Fixed non-spatial ODE baseline matched to the Strobl tumour model.

The equations are those reported in Strobl et al. (2022), Eqs. (2)--(4).
Controls are held constant over each supplied interval.  Parameters are shared
simulation inputs; this module deliberately provides no fitting API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class StroblODEParameters:
    """Fixed biological and treatment parameters for every ODE rollout."""

    r_s: float = 0.027
    r_r: float = 0.027
    delta_t: float = 0.0
    d_d: float = 0.75
    d_max: float = 1.0
    carrying_capacity: float = 10_000.0

    def __post_init__(self) -> None:
        values = {
            "r_s": self.r_s,
            "r_r": self.r_r,
            "delta_t": self.delta_t,
            "d_d": self.d_d,
            "d_max": self.d_max,
            "carrying_capacity": self.carrying_capacity,
        }
        for name, value in values.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.carrying_capacity <= 0:
            raise ValueError("carrying_capacity must be positive")
        if self.d_d * self.d_max > 1.0:
            raise ValueError("the model requires d_d * d_max <= 1")


@dataclass(frozen=True)
class StroblODEResult:
    """ODE states at control boundaries and their held scalar actions."""

    times: np.ndarray
    counts: np.ndarray
    actions: np.ndarray

    @property
    def sensitive(self) -> np.ndarray:
        """Sensitive-cell trajectory at interval boundaries."""
        return self.counts[:, 0]

    @property
    def resistant(self) -> np.ndarray:
        """Resistant-cell trajectory at interval boundaries."""
        return self.counts[:, 1]

    @property
    def total(self) -> np.ndarray:
        """Total tumour trajectory at interval boundaries."""
        return self.counts[:, 2]


def strobl_rhs(
    _time: float,
    state: Sequence[float] | np.ndarray,
    dose: float,
    parameters: StroblODEParameters,
) -> np.ndarray:
    """Evaluate the exact controlled Strobl ODE right-hand side.

    .. math::

       \\dot S = r_S(1-(S+R)/K)(1-2d_D D(t)/D_{max})S-\\delta_T S

       \\dot R = r_R(1-(S+R)/K)R-\\delta_T R.
    """
    sensitive, resistant = np.asarray(state, dtype=np.float64)
    dose = float(dose)
    if not np.isfinite(dose) or not 0.0 <= dose <= parameters.d_max:
        raise ValueError(f"dose must lie in [0, {parameters.d_max:g}]")
    crowding = 1.0 - (sensitive + resistant) / parameters.carrying_capacity
    treatment = (
        1.0
        if parameters.d_max == 0.0
        else 1.0 - (2.0 * parameters.d_d / parameters.d_max) * dose
    )
    ds = parameters.r_s * crowding * treatment * sensitive
    ds -= parameters.delta_t * sensitive
    dr = parameters.r_r * crowding * resistant
    dr -= parameters.delta_t * resistant
    return np.asarray([ds, dr], dtype=np.float64)


def solve_strobl_ode(
    initial_counts: Sequence[float] | np.ndarray,
    actions: Sequence[float] | np.ndarray,
    parameters: StroblODEParameters,
    *,
    interval_duration: float | Sequence[float] | np.ndarray = 1.0,
    start_time: float = 0.0,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> StroblODEResult:
    """Integrate piecewise-constant controls, one interval at a time.

    ``initial_counts`` accepts ``(S0, R0)`` or ``(S0, R0, N0)``; when ``N0``
    is supplied it is validated rather than treated as an independent state.
    Returned counts are ordered ``[S, R, N]`` and have length ``T + 1``.

    SciPy is imported lazily so the rest of the package remains importable
    before the separately required SciPy dependency is installed.
    """
    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:
        raise ImportError(
            "solve_strobl_ode requires SciPy; install the current scipy package"
        ) from exc

    initial = np.asarray(initial_counts, dtype=np.float64)
    if initial.shape not in {(2,), (3,)}:
        raise ValueError("initial_counts must contain (S0, R0) or (S0, R0, N0)")
    if not np.all(np.isfinite(initial)) or np.any(initial < 0):
        raise ValueError("initial_counts must be finite and non-negative")
    if initial.shape == (3,) and not np.isclose(initial[2], initial[0] + initial[1]):
        raise ValueError("N0 must equal S0 + R0")
    state = initial[:2].copy()

    action_array = np.asarray(actions, dtype=np.float64)
    if action_array.ndim != 1 or not np.all(np.isfinite(action_array)):
        raise ValueError("actions must be a finite one-dimensional array")
    if np.any((action_array < 0) | (action_array > parameters.d_max)):
        raise ValueError(f"actions must lie in [0, {parameters.d_max:g}]")

    durations = np.asarray(interval_duration, dtype=np.float64)
    if durations.ndim == 0:
        durations = np.full(action_array.size, float(durations))
    if durations.shape != action_array.shape:
        raise ValueError("interval_duration must be scalar or match actions")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
        raise ValueError("all control interval durations must be finite and positive")
    if not np.isfinite(start_time):
        raise ValueError("start_time must be finite")

    states = np.empty((action_array.size + 1, 2), dtype=np.float64)
    times = np.empty(action_array.size + 1, dtype=np.float64)
    states[0] = state
    times[0] = float(start_time)
    current_time = float(start_time)

    for index, (dose, duration) in enumerate(
        zip(action_array, durations, strict=True), start=1
    ):
        end_time = current_time + float(duration)
        solution = solve_ivp(
            strobl_rhs,
            (current_time, end_time),
            state,
            args=(float(dose), parameters),
            method=method,
            t_eval=[end_time],
            rtol=rtol,
            atol=atol,
        )
        if not solution.success:
            raise RuntimeError(
                f"ODE integration failed in interval {index - 1}: {solution.message}"
            )
        state = solution.y[:, -1]
        # Numerical solvers may undershoot zero by roundoff near extinction.
        if np.any(state < -max(float(atol), 1e-12)):
            raise RuntimeError("ODE solver produced materially negative cell counts")
        state = np.maximum(state, 0.0)
        states[index] = state
        times[index] = end_time
        current_time = end_time

    counts = np.column_stack((states, states.sum(axis=1)))
    return StroblODEResult(
        times=times,
        counts=counts,
        actions=action_array.astype(np.float32),
    )


__all__ = [
    "StroblODEParameters",
    "StroblODEResult",
    "solve_strobl_ode",
    "strobl_rhs",
]
