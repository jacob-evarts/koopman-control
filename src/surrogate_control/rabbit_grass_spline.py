from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def fit_mean_trajectory_splines(
    t: np.ndarray,
    R: np.ndarray,
    G: np.ndarray,
) -> tuple[CubicSpline, CubicSpline]:
    """Smoothing splines (natural cubic) through mean rabbit / grass vs time."""
    t_arr = np.asarray(t, dtype=float)
    return (
        CubicSpline(t_arr, np.asarray(R, dtype=float), bc_type="natural"),
        CubicSpline(t_arr, np.asarray(G, dtype=float), bc_type="natural"),
    )


def blend_spline_state(
    spl_uc_R: CubicSpline,
    spl_uc_G: CubicSpline,
    spl_c_R: CubicSpline,
    spl_c_G: CubicSpline,
    t: float,
    u: float,
    t_min: float,
    t_max: float,
) -> tuple[float, float]:
    """Convex blend between uncontrolled (u=0) and controlled (u=1) spline schedules."""
    tc = float(np.clip(t, t_min, t_max))
    w = float(np.clip(u, 0.0, 1.0))
    r = (1.0 - w) * float(spl_uc_R(tc)) + w * float(spl_c_R(tc))
    g = (1.0 - w) * float(spl_uc_G(tc)) + w * float(spl_c_G(tc))
    return max(r, 0.0), max(g, 0.0)
