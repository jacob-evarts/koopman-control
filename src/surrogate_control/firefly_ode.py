from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize as sp_minimize


def _clip_population(F: float, n_total: float) -> float:
    return float(np.clip(F, 0.0, n_total))


def firefly_ode_rhs(
    t: float,
    y: Sequence[float],
    params: Sequence[float],
    u: float,
    n_total: float,
) -> list[float]:
    """Cyclic firefly surrogate: Van der Pol–style oscillator on flashing count.

    State ``y = [F, V]`` with ``V = dF/dt``. Resting count is ``R = n_total - F``.

    Parameters ``(omega, gamma, mu, kappa, F_eq)``:

    - ``omega``: angular frequency of the linear restoring term (rad / step)
    - ``gamma``: linear damping on ``V``
    - ``mu``: nonlinear Van der Pol damping (``mu > 0`` → self-sustained oscillations)
    - ``kappa``: impulsive beacon kick when ``u`` is on (single-step velocity impulse)
    - ``F_eq``: equilibrium flashing level around which the limit cycle is centred
    """
    F = _clip_population(float(y[0]), n_total)
    V = float(y[1])
    omega, gamma, mu, kappa, F_eq = params
    nt = float(n_total)
    amp_scale = max(0.25 * nt, 1.0)
    x = (F - F_eq) / amp_scale
    pulse = 1.0 if float(u) > 0.5 else 0.0
    dF = V
    dV = (
        -(omega**2) * (F - F_eq)
        - gamma * V
        + mu * (1.0 - x * x) * V
        + kappa * pulse
    )
    return [dF, dV]


def resting_from_flashing(F: float, n_total: float) -> float:
    return max(float(n_total) - _clip_population(F, n_total), 0.0)


def initial_velocity(F: Sequence[float]) -> float:
    """Estimate ``dF/dt`` at the start of a trajectory."""
    F = np.asarray(F, dtype=float)
    if F.size < 2:
        return 0.0
    return float(F[1] - F[0])


def estimate_omega_from_traj(F: Sequence[float], dt: float = 1.0) -> float:
    """Rough dominant angular frequency from a scalar time series (rad / step)."""
    F = np.asarray(F, dtype=float)
    if F.size < 8:
        return 2.0 * np.pi / 30.0
    Fc = F - F.mean()
    spec = np.abs(np.fft.rfft(Fc))
    freqs = np.fft.rfftfreq(Fc.size, d=dt)
    if spec.size <= 1:
        return 2.0 * np.pi / 30.0
    spec[0] = 0.0
    k = int(np.argmax(spec[1:]) + 1)
    f_hz = max(float(freqs[k]), 1.0 / max(F.size * dt, 1.0))
    return float(2.0 * np.pi * f_hz)


def simulate_firefly_ode(
    params: np.ndarray | Sequence[float],
    F0: float,
    R0: float,
    u_seq: Sequence[float] | None,
    T: int,
    n_total: float,
    *,
    V0: float | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the oscillator for ``T`` steps; ``R0`` is only used if ``V0`` is omitted.

    When ``V0`` is None, ``V0 = R0 - (n_total - F0)`` is ignored and we set ``V0 = 0``.
    Prefer passing ``V0=initial_velocity(traj['F'])`` for better phase alignment.
    """
    nt = float(n_total)
    F_traj, R_traj = np.empty(T + 1), np.empty(T + 1)
    F0c = _clip_population(float(F0), nt)
    v_init = 0.0 if V0 is None else float(V0)
    F_traj[0], R_traj[0] = F0c, resting_from_flashing(F0c, nt)
    y = [F0c, v_init]
    p = np.asarray(params, dtype=float)
    for k in range(T):
        u_k = float(u_seq[k]) if u_seq is not None else 0.0
        sol = solve_ivp(
            firefly_ode_rhs,
            (0.0, 1.0),
            y,
            args=(p, u_k, nt),
            method="RK45",
            rtol=rtol,
            atol=atol,
        )
        F_next = _clip_population(float(sol.y[0, -1]), nt)
        V_next = float(sol.y[1, -1])
        y = [F_next, V_next]
        F_traj[k + 1] = F_next
        R_traj[k + 1] = resting_from_flashing(F_next, nt)
    return F_traj, R_traj


def _build_regression_mats(
    trajs: list[dict[str, Any]],
    n_total: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack features for ``dF`` and ``dV`` from discrete trajectories."""
    nt = float(n_total)
    amp_scale = max(0.25 * nt, 1.0)
    rows_dF, rows_dV, dF_obs, dV_obs = [], [], [], []

    for traj in trajs:
        F = np.asarray(traj["F"], dtype=float)
        u = np.asarray(traj["u"], dtype=float)
        V = np.gradient(F)
        dF = np.gradient(F)
        dV = np.gradient(V)
        Fm, Vm, um = F, V, u
        Rm = nt - np.clip(Fm, 0.0, nt)
        x = (Fm - Fm.mean()) / amp_scale
        rows_dF.append(np.column_stack([np.ones(len(Fm)), Vm]))
        rows_dV.append(
            np.column_stack(
                [
                    -(Fm - Fm.mean()),
                    -Vm,
                    (1.0 - x * x) * Vm,
                    um,
                ]
            )
        )
        dF_obs.append(dF)
        dV_obs.append(dV)

    return (
        np.vstack(rows_dF),
        np.vstack(rows_dV),
        np.concatenate(dF_obs),
        np.concatenate(dV_obs),
    )


def fit_firefly_ode_slm(
    trajs: list[dict[str, Any]],
    n_total: float,
    *,
    verbose: bool = True,
    minimize_fn: Callable[..., Any] | None = None,
) -> np.ndarray:
    """Fit cyclic surrogate parameters from trajectories (keys ``F``, ``R``, ``u``)."""
    minimize_fn = minimize_fn or sp_minimize
    nt = float(n_total)

    F_eq0 = float(np.mean([np.mean(t["F"]) for t in trajs]))
    omega0 = estimate_omega_from_traj(trajs[0]["F"])
    gamma0 = 0.05
    mu0 = 0.5
    kappa0 = 0.1

    _, feat_dV, _, dV_obs = _build_regression_mats(trajs, nt)
    if feat_dV.size > 0:
        coef, *_ = np.linalg.lstsq(feat_dV, dV_obs, rcond=None)
        # coef maps to -omega^2, -gamma, mu, kappa on features
        omega_ls = float(np.sqrt(max(-coef[0], 1e-6)))
        if np.isfinite(omega_ls):
            omega0 = float(np.clip(omega_ls, 0.05, 2.0 * np.pi))
        gamma0 = max(float(-coef[1]), 1e-4)
        mu0 = float(coef[2])
        kappa0 = max(float(coef[3]), 1e-4)

    theta0 = np.array([omega0, gamma0, mu0, kappa0, F_eq0], dtype=float)

    if verbose:
        print(
            "  OLS warm-start:",
            dict(
                zip(
                    ["omega", "gamma", "mu", "kappa", "F_eq"],
                    theta0,
                )
            ),
        )

    F_scale = np.mean([t["F"].max() - t["F"].min() for t in trajs]) + 1e-8

    def total_loss(params: np.ndarray) -> float:
        p = np.asarray(params, dtype=float)
        if p[0] <= 0 or p[3] < 0:
            return 1e12
        loss = 0.0
        for traj in trajs:
            T = len(traj["F"]) - 1
            V0 = initial_velocity(traj["F"])
            F_pred, R_pred = simulate_firefly_ode(
                p,
                traj["F"][0],
                traj["R"][0],
                traj["u"][1:],
                T,
                nt,
                V0=V0,
            )
            loss += np.mean(((F_pred - traj["F"]) / F_scale) ** 2)
            loss += 0.25 * np.mean(((R_pred - traj["R"]) / nt) ** 2)
        return float(loss)

    _it = [0]

    def _cb(xk: np.ndarray) -> None:
        _it[0] += 1
        if verbose and _it[0] % 10 == 0:
            print(f"    L-BFGS iter {_it[0]:4d}  loss={total_loss(xk):.6f}")

    bounds = [
        (0.05, 2.0 * np.pi),
        (1e-5, 2.0),
        (-2.0, 2.0),
        (1e-5, 50.0),
        (0.0, nt),
    ]

    _it[0] = 0
    result = minimize_fn(
        total_loss,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        callback=_cb,
        options={"maxiter": 250, "ftol": 1e-10, "gtol": 1e-7},
    )
    if verbose:
        print(f"  L-BFGS done: success={result.success} loss={result.fun:.6f}")
        p = result.x
        period = 2.0 * np.pi / p[0] if p[0] > 0 else float("inf")
        print(f"  implied period ≈ {period:.1f} steps")
    return np.asarray(result.x, dtype=float)
