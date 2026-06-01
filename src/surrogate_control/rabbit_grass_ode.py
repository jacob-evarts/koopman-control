from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize as sp_minimize


def ode_rhs(
    t: float,
    y: Sequence[float],
    params: Sequence[float],
    u: float,
    g_max: float,
) -> list[float]:
    """Surrogate RHS; params = (alpha, beta, rho, epsilon, kappa)."""
    R, G = max(float(y[0]), 0.0), max(float(y[1]), 0.0)
    alpha, beta, rho, epsilon, kappa = params
    gm = float(g_max)
    dR = (alpha * (G / gm) - beta) * R - kappa * float(u) * R
    dG = rho * G * (1.0 - G / gm) - epsilon * R * G
    return [dR, dG]


def simulate_ode(
    params: np.ndarray | Sequence[float],
    R0: float,
    G0: float,
    u_seq: Sequence[float] | None,
    T: int,
    g_max: float,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the surrogate ODE for ``T`` unit-time steps with piecewise-constant ``u``."""
    R_traj, G_traj = np.empty(T + 1), np.empty(T + 1)
    R_traj[0], G_traj[0] = R0, G0
    y = [float(R0), float(G0)]
    p = np.asarray(params, dtype=float)
    for k in range(T):
        u_k = float(u_seq[k]) if u_seq is not None else 0.0
        sol = solve_ivp(
            ode_rhs,
            (0.0, 1.0),
            y,
            args=(p, u_k, float(g_max)),
            method="RK45",
            rtol=rtol,
            atol=atol,
        )
        y = [max(float(sol.y[0, -1]), 0.0), max(float(sol.y[1, -1]), 0.0)]
        R_traj[k + 1], G_traj[k + 1] = y
    return R_traj, G_traj


def fit_ode_slm(
    trajs: list[dict[str, Any]],
    g_max: float,
    *,
    verbose: bool = True,
    minimize_fn: Callable[..., Any] | None = None,
) -> np.ndarray:
    """Fit surrogate parameters from mean trajectories (keys ``R``, ``G``, ``u``)."""
    minimize_fn = minimize_fn or sp_minimize
    gm = float(g_max)

    feat_R, feat_G, dR_obs, dG_obs = [], [], [], []
    for traj in trajs:
        R, G, u = traj["R"], traj["G"], traj["u"]
        Rm, Gm, um = R[:-1], G[:-1], u[:-1]
        feat_R.append(
            np.column_stack(
                [
                    Rm * Gm / gm,
                    -Rm,
                    -um * Rm,
                ]
            )
        )
        feat_G.append(np.column_stack([Gm * (1.0 - Gm / gm), -Rm * Gm]))
        dR_obs.append(np.diff(R))
        dG_obs.append(np.diff(G))

    feat_R = np.vstack(feat_R)
    feat_G = np.vstack(feat_G)
    dR_obs = np.concatenate(dR_obs)
    dG_obs = np.concatenate(dG_obs)

    (alpha0, beta0, kappa0), *_ = np.linalg.lstsq(feat_R, dR_obs, rcond=None)
    (rho0, epsilon0), *_ = np.linalg.lstsq(feat_G, dG_obs, rcond=None)
    alpha0, beta0, rho0, epsilon0, kappa0 = [max(float(v), 1e-4) for v in [alpha0, beta0, rho0, epsilon0, kappa0]]
    theta0 = np.array([alpha0, beta0, rho0, epsilon0, kappa0])

    if verbose:
        print("  OLS warm-start:", dict(zip(["alpha", "beta", "rho", "epsilon", "kappa"], theta0)))

    R_scale = np.mean([t["R"].max() for t in trajs]) + 1e-8

    def total_loss(params: np.ndarray) -> float:
        if np.any(np.asarray(params) < 0):
            return 1e12
        loss = 0.0
        for traj in trajs:
            T = len(traj["R"]) - 1
            R_pred, G_pred = simulate_ode(params, traj["R"][0], traj["G"][0], traj["u"][1:], T, gm)
            loss += np.mean(((R_pred - traj["R"]) / R_scale) ** 2)
            loss += np.mean(((G_pred - traj["G"]) / gm) ** 2)
        return float(loss)

    _it = [0]

    def _cb(xk: np.ndarray) -> None:
        _it[0] += 1
        if verbose and _it[0] % 10 == 0:
            print(f"    L-BFGS iter {_it[0]:4d}  loss={total_loss(xk):.6f}")

    margin = 0.60
    bounds = [
        (max(1e-6, v * (1 - margin)), v * (1 + margin))
        for v in [alpha0, beta0, rho0, epsilon0, kappa0]
    ]

    _it[0] = 0
    result = minimize_fn(
        total_loss,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        callback=_cb,
        options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-7},
    )
    if verbose:
        print(f"  L-BFGS done: success={result.success} loss={result.fun:.6f}")
    return np.asarray(result.x, dtype=float)
