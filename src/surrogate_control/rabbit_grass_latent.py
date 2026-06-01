from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline


def fit_latent_linear_surrogate(
    trajs: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ``z_{t+1} = A z_t + B u_t + bias`` from trajectories with keys ``Z``, ``u``."""
    rows_z, rows_u, rows_zp = [], [], []
    for traj in trajs:
        z, u = traj["Z"], traj["u"]
        rows_z.append(z[:-1])
        rows_u.append(u[:-1, None])
        rows_zp.append(z[1:])
    z_t = np.vstack(rows_z)
    u_t = np.vstack(rows_u)
    z_tp1 = np.vstack(rows_zp)
    d = z_t.shape[1]
    aug = np.hstack([z_t, u_t, np.ones((len(z_t), 1))])
    w, *_ = np.linalg.lstsq(aug, z_tp1, rcond=None)
    a = w[:d].T
    b = w[d]
    bias = w[d + 1]
    return np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(bias, dtype=float)


def fit_latent_spline_surrogates(
    traj_uc: dict[str, Any],
    traj_c: dict[str, Any],
) -> dict[str, Any]:
    """Per-latent-dimension natural cubic splines for uncontrolled vs cull-on schedules."""
    z_uc = np.asarray(traj_uc["Z"], dtype=float)
    z_c = np.asarray(traj_c["Z"], dtype=float)
    t_uc = np.arange(len(z_uc), dtype=float)
    t_c = np.arange(len(z_c), dtype=float)
    d = z_uc.shape[1]
    spl_uc, spl_c = [], []
    for j in range(d):
        spl_uc.append(CubicSpline(t_uc, z_uc[:, j], bc_type="natural"))
        spl_c.append(CubicSpline(t_c, z_c[:, j], bc_type="natural"))
    return {
        "spl_uc": spl_uc,
        "spl_c": spl_c,
        "t_max": float(max(t_uc[-1], t_c[-1])),
        "latent_dim": d,
    }


def blend_latent_vector(
    splines: dict[str, Any],
    t: float,
    u: float,
) -> np.ndarray:
    """Convex blend of spline values at time ``t`` with weight ``u``."""
    tc = float(np.clip(t, 0.0, splines["t_max"]))
    w = float(np.clip(u, 0.0, 1.0))
    z = np.empty(splines["latent_dim"], dtype=float)
    for j, (su, sc) in enumerate(zip(splines["spl_uc"], splines["spl_c"])):
        z[j] = (1.0 - w) * float(su(tc)) + w * float(sc(tc))
    return z
