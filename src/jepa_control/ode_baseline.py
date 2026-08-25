"""Macrostate ODE surrogate baseline for rabbit/grass control.

This is the "skip the latent, just fit an ODE on the populations" baseline that
the JEPA controller has to beat. The model is a discrete-time consumer-resource
system with delayed culling, fit by linear least squares on one-step deltas:

    R' = R + R * (growth * G - death - cull_now * u - cull_lag * u_prev)
    G' = G + regrow * (1 - G) - consume * R * G

``R`` is ``rabbit_count``, ``G`` is ``grass_frac``. Control history
``[u, u_prev]`` matches the ABM's one-step actuator lag. Closed-loop MPC uses
the same CEM planner as the JEPA controller, but plans in ``(R, G)`` with
**oracle** macrostate observations from the true simulator each step -- a
stronger information regime than image → latent → readout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from jepa_control.control import MPCConfig, SIRMPCConfig, TumorMPCConfig
from koopman_control.data.agentic_sir import AgenticSIRConfig, AgenticSIRModel
from koopman_control.data.rabbit_grass import RabbitGrassConfig, RabbitGrassModel
from koopman_control.data.tumor_tissue import TumorTissueConfig, TumorTissueModel

RABBIT = "rabbit_count"
GRASS = "grass_frac"


@dataclass(frozen=True)
class ResourceODE:
    """Fitted discrete consumer-resource parameters (one ABM step = one Euler step)."""

    growth: float
    death: float
    cull_now: float
    cull_lag: float
    regrow: float
    consume: float
    rabbit_one_step_r2: float
    grass_one_step_r2: float

    def serializable(self) -> dict:
        return asdict(self)


def _obs_index(names: list[str], key: str) -> int:
    try:
        return names.index(key)
    except ValueError as exc:
        raise KeyError(f"observable {key!r} not in {names}") from exc


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def fit_resource_ode(
    trajs: list[dict],
    *,
    ridge: float = 1e-6,
    rabbit_name: str = RABBIT,
    grass_name: str = GRASS,
) -> ResourceODE:
    """Least-squares fit of the consumer-resource map on trajectory observables.

    Rabbit updates are fit as ``ΔR / R ~ growth*G - death - cull_now*u - cull_lag*u_prev``
    on frames with ``R > 0``. Grass updates are fit as
    ``ΔG ~ regrow*(1-G) - consume*R*G``. Both are linear in the parameters.
    """
    names = list(trajs[0]["obs_names"])
    i_r, i_g = _obs_index(names, rabbit_name), _obs_index(names, grass_name)

    x_r, y_r = [], []
    x_g, y_g = [], []
    for tr in trajs:
        obs = np.asarray(tr["obs"], dtype=np.float64)
        u = np.asarray(tr["controls"], dtype=np.float64)
        # Dataset stores obs and control at the same length (one entry per frame);
        # the control at index t is applied on the transition t -> t+1.
        n_trans = min(len(u), len(obs) - 1)
        for t in range(n_trans):
            r, g = obs[t, i_r], obs[t, i_g]
            r1, g1 = obs[t + 1, i_r], obs[t + 1, i_g]
            u_now = float(u[t])
            u_prev = float(u[t - 1]) if t > 0 else 0.0
            if r > 1e-6:
                x_r.append([g, -1.0, -u_now, -u_prev])
                y_r.append((r1 - r) / r)
            x_g.append([1.0 - g, -r * g])
            y_g.append(g1 - g)

    x_r_a, y_r_a = np.asarray(x_r), np.asarray(y_r)
    x_g_a, y_g_a = np.asarray(x_g), np.asarray(y_g)
    if len(x_r_a) < 4 or len(x_g_a) < 2:
        raise ValueError("not enough transitions to fit the resource ODE")

    def ridge_solve(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gram = x.T @ x + ridge * np.eye(x.shape[1])
        return np.linalg.solve(gram, x.T @ y)

    th_r = ridge_solve(x_r_a, y_r_a)
    th_g = ridge_solve(x_g_a, y_g_a)
    return ResourceODE(
        growth=float(th_r[0]),
        death=float(th_r[1]),
        cull_now=float(th_r[2]),
        cull_lag=float(th_r[3]),
        regrow=float(th_g[0]),
        consume=float(th_g[1]),
        rabbit_one_step_r2=_r2(y_r_a, x_r_a @ th_r),
        grass_one_step_r2=_r2(y_g_a, x_g_a @ th_g),
    )


def ode_step(
    r: float,
    g: float,
    u: float,
    u_prev: float,
    ode: ResourceODE,
) -> tuple[float, float]:
    """One discrete Euler step of the fitted resource ODE."""
    r_next = r + r * (ode.growth * g - ode.death - ode.cull_now * u - ode.cull_lag * u_prev)
    g_next = g + ode.regrow * (1.0 - g) - ode.consume * r * g
    return max(0.0, float(r_next)), float(np.clip(g_next, 0.0, 1.0))


def ode_rollout(
    r0: float,
    g0: float,
    controls: np.ndarray,
    ode: ResourceODE,
    *,
    u_prev0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Free-run the ODE under a control sequence; returns ``(R, G)`` of length ``len(u)+1``."""
    u = np.asarray(controls, dtype=np.float64)
    rs = np.empty(len(u) + 1, dtype=np.float64)
    gs = np.empty(len(u) + 1, dtype=np.float64)
    rs[0], gs[0] = r0, g0
    u_prev = u_prev0
    for t, u_now in enumerate(u):
        rs[t + 1], gs[t + 1] = ode_step(rs[t], gs[t], float(u_now), u_prev, ode)
        u_prev = float(u_now)
    return rs, gs


def ode_prediction_skill(
    trajs: list[dict],
    ode: ResourceODE,
    *,
    horizon: int = 16,
    rabbit_name: str = RABBIT,
    grass_name: str = GRASS,
) -> dict:
    """Held-out free-rollout skill of the ODE on rabbit count (vs persistence)."""
    names = list(trajs[0]["obs_names"])
    i_r, i_g = _obs_index(names, rabbit_name), _obs_index(names, grass_name)
    mse_ode, mse_persist, var_acc = [], [], []
    for tr in trajs:
        obs = np.asarray(tr["obs"], dtype=np.float64)
        u = np.asarray(tr["controls"], dtype=np.float64)
        n_trans = min(len(u), len(obs) - 1)
        t_max = n_trans - horizon
        if t_max <= 0:
            continue
        # A few starts per trajectory keep this cheap.
        starts = np.linspace(0, t_max, num=min(8, t_max + 1), dtype=int)
        for t0 in starts:
            true_r = obs[t0 : t0 + horizon + 1, i_r]
            rs, _ = ode_rollout(obs[t0, i_r], obs[t0, i_g], u[t0 : t0 + horizon], ode)
            mse_ode.append(float(np.mean((rs - true_r) ** 2)))
            mse_persist.append(float(np.mean((true_r[0] - true_r) ** 2)))
            var_acc.append(float(np.var(true_r)))
    mse_o = float(np.mean(mse_ode)) if mse_ode else float("nan")
    mse_p = float(np.mean(mse_persist)) if mse_persist else float("nan")
    var = float(np.mean(var_acc)) if var_acc else float("nan")
    return {
        "horizon": horizon,
        "mse": mse_o,
        "mse_persistence": mse_p,
        "skill": 1.0 - mse_o / max(var, 1e-12),
        "skill_persistence": 1.0 - mse_p / max(var, 1e-12),
    }


def cem_plan_ode(
    r0: float,
    g0: float,
    target: float,
    ode: ResourceODE,
    cfg: MPCConfig,
    *,
    u_prev: float = 0.0,
    warm_start: np.ndarray | None = None,
) -> np.ndarray:
    """CEM plan that rolls the resource ODE and scores rabbit-count tracking."""
    h, s = cfg.plan_horizon, cfg.n_samples
    n_elite = max(1, int(cfg.elite_frac * s))
    mean = np.zeros(h, dtype=np.float64) if warm_start is None else warm_start.astype(np.float64)
    std = np.full(h, cfg.init_std, dtype=np.float64)

    for _ in range(cfg.n_iters):
        u = np.clip(
            mean[None, :] + std[None, :] * np.random.randn(s, h),
            cfg.u_min,
            cfg.u_max,
        )
        cost = np.zeros(s, dtype=np.float64)
        for i in range(s):
            r, g, prev = r0, g0, u_prev
            c = 0.0
            for k in range(h):
                r, g = ode_step(r, g, float(u[i, k]), prev, ode)
                c += (r - target) ** 2 + cfg.control_cost * float(u[i, k]) ** 2
                prev = float(u[i, k])
            cost[i] = c
        elite = np.argpartition(cost, n_elite)[:n_elite]
        mean = u[elite].mean(axis=0)
        std = u[elite].std(axis=0) + 1e-3
    return mean.astype(np.float32)


def closed_loop_ode(
    ode: ResourceODE,
    target: float,
    *,
    steps: int = 100,
    mpc: MPCConfig | None = None,
    cfg: RabbitGrassConfig | None = None,
    initial_rabbits: int = 120,
    initial_grass_prob: float = 0.35,
    seed: int = 0,
) -> dict:
    """Receding-horizon MPC with the ODE planner, closed on the true ABM.

    Each step the planner is given the **true** ``(rabbit_count, grass_frac)``
    from the simulator (oracle macros), plans with the fitted ODE, and applies
    the first action to the ABM. Same return schema as
    :func:`jepa_control.control.closed_loop` for side-by-side scoring.
    """
    mpc = mpc or MPCConfig()
    cfg = cfg or RabbitGrassConfig()
    sim = RabbitGrassModel(
        cfg=cfg, initial_rabbits=initial_rabbits, initial_grass_prob=initial_grass_prob, seed=seed
    )

    true_series = [sim.observables()[RABBIT]]
    controls = [0.0]
    warm: np.ndarray | None = None
    u_prev = 0.0
    for _ in range(steps):
        obs = sim.observables()
        plan = cem_plan_ode(
            obs[RABBIT],
            obs[GRASS],
            target,
            ode,
            mpc,
            u_prev=u_prev,
            warm_start=warm,
        )
        u = float(np.clip(plan[0], mpc.u_min, mpc.u_max))
        sim.step(u)
        true_series.append(sim.observables()[RABBIT])
        controls.append(u)
        warm = np.concatenate([plan[1:], plan[-1:]])
        u_prev = u

    true_arr = np.asarray(true_series, dtype=np.float32)
    return {
        "obs_name": RABBIT,
        "target": float(target),
        "control": np.asarray(controls, dtype=np.float32),
        "true": true_arr,
        "tracking_rmse": float(np.sqrt(np.mean((true_arr[1:] - target) ** 2))),
        "final_error": float(true_arr[-1] - target),
        "planner": "resource_ode",
    }


# ----------------------------------------------------------------------
# Wolf–rabbit–grass predator–prey ODE (cull acts on wolves)
# ----------------------------------------------------------------------
WOLF = "wolf_count"


@dataclass(frozen=True)
class PredatorPreyODE:
    """Discrete trophic chain with delayed wolf culling.

    Fitted by least squares on one-step deltas:

        ΔR/R ~ growth*G - death_r - pred*W
        ΔW/W ~ conv*R - death_w - cull_now*u - cull_lag*u_prev
        ΔG   ~ regrow*(1-G) - consume*R*G
    """

    growth: float
    death_r: float
    predation: float
    conversion: float
    death_w: float
    cull_now: float
    cull_lag: float
    regrow: float
    consume: float
    rabbit_one_step_r2: float
    wolf_one_step_r2: float
    grass_one_step_r2: float

    def serializable(self) -> dict:
        return asdict(self)


def fit_predator_prey_ode(
    trajs: list[dict],
    *,
    ridge: float = 1e-6,
    rabbit_name: str = RABBIT,
    wolf_name: str = WOLF,
    grass_name: str = GRASS,
) -> PredatorPreyODE:
    """Least-squares fit of the three-species map on trajectory observables."""
    names = list(trajs[0]["obs_names"])
    i_r = _obs_index(names, rabbit_name)
    i_w = _obs_index(names, wolf_name)
    i_g = _obs_index(names, grass_name)

    x_r, y_r, x_w, y_w, x_g, y_g = [], [], [], [], [], []
    for tr in trajs:
        obs = np.asarray(tr["obs"], dtype=np.float64)
        u = np.asarray(tr["controls"], dtype=np.float64)
        n_trans = min(len(u), len(obs) - 1)
        for t in range(n_trans):
            r, w, g = obs[t, i_r], obs[t, i_w], obs[t, i_g]
            r1, w1, g1 = obs[t + 1, i_r], obs[t + 1, i_w], obs[t + 1, i_g]
            u_now = float(u[t])
            u_prev = float(u[t - 1]) if t > 0 else 0.0
            if r > 1e-6:
                x_r.append([g, -1.0, -w])
                y_r.append((r1 - r) / r)
            if w > 1e-6:
                x_w.append([r, -1.0, -u_now, -u_prev])
                y_w.append((w1 - w) / w)
            x_g.append([1.0 - g, -r * g])
            y_g.append(g1 - g)

    def ridge_solve(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gram = x.T @ x + ridge * np.eye(x.shape[1])
        return np.linalg.solve(gram, x.T @ y)

    x_r_a, y_r_a = np.asarray(x_r), np.asarray(y_r)
    x_w_a, y_w_a = np.asarray(x_w), np.asarray(y_w)
    x_g_a, y_g_a = np.asarray(x_g), np.asarray(y_g)
    if min(len(x_r_a), len(x_w_a), len(x_g_a)) < 4:
        raise ValueError("not enough transitions to fit the predator–prey ODE")

    th_r, th_w, th_g = ridge_solve(x_r_a, y_r_a), ridge_solve(x_w_a, y_w_a), ridge_solve(x_g_a, y_g_a)
    return PredatorPreyODE(
        growth=float(th_r[0]),
        death_r=float(th_r[1]),
        predation=float(th_r[2]),
        conversion=float(th_w[0]),
        death_w=float(th_w[1]),
        cull_now=float(th_w[2]),
        cull_lag=float(th_w[3]),
        regrow=float(th_g[0]),
        consume=float(th_g[1]),
        rabbit_one_step_r2=_r2(y_r_a, x_r_a @ th_r),
        wolf_one_step_r2=_r2(y_w_a, x_w_a @ th_w),
        grass_one_step_r2=_r2(y_g_a, x_g_a @ th_g),
    )


def predator_prey_step(
    r: float,
    w: float,
    g: float,
    u: float,
    u_prev: float,
    ode: PredatorPreyODE,
) -> tuple[float, float, float]:
    """One discrete Euler step of the fitted predator–prey ODE."""
    r_next = r + r * (ode.growth * g - ode.death_r - ode.predation * w)
    w_next = w + w * (
        ode.conversion * r - ode.death_w - ode.cull_now * u - ode.cull_lag * u_prev
    )
    g_next = g + ode.regrow * (1.0 - g) - ode.consume * r * g
    return max(0.0, float(r_next)), max(0.0, float(w_next)), float(np.clip(g_next, 0.0, 1.0))


def predator_prey_prediction_skill(
    trajs: list[dict],
    ode: PredatorPreyODE,
    *,
    horizon: int = 16,
    obs_name: str = WOLF,
    rabbit_name: str = RABBIT,
    wolf_name: str = WOLF,
    grass_name: str = GRASS,
) -> dict:
    """Held-out free-rollout skill of the ODE on ``obs_name`` (vs persistence)."""
    names = list(trajs[0]["obs_names"])
    i_r = _obs_index(names, rabbit_name)
    i_w = _obs_index(names, wolf_name)
    i_g = _obs_index(names, grass_name)
    i_y = _obs_index(names, obs_name)
    mse_ode, mse_persist, var_acc = [], [], []
    for tr in trajs:
        obs = np.asarray(tr["obs"], dtype=np.float64)
        u = np.asarray(tr["controls"], dtype=np.float64)
        n_trans = min(len(u), len(obs) - 1)
        t_max = n_trans - horizon
        if t_max <= 0:
            continue
        starts = np.linspace(0, t_max, num=min(8, t_max + 1), dtype=int)
        for t0 in starts:
            true_y = obs[t0 : t0 + horizon + 1, i_y]
            r, w, g = float(obs[t0, i_r]), float(obs[t0, i_w]), float(obs[t0, i_g])
            u_prev = 0.0
            pred = [true_y[0]]
            for k in range(horizon):
                r, w, g = predator_prey_step(r, w, g, float(u[t0 + k]), u_prev, ode)
                u_prev = float(u[t0 + k])
                pred.append(w if obs_name == wolf_name else r if obs_name == rabbit_name else g)
            pred_a = np.asarray(pred, dtype=np.float64)
            mse_ode.append(float(np.mean((pred_a - true_y) ** 2)))
            mse_persist.append(float(np.mean((true_y[0] - true_y) ** 2)))
            var_acc.append(float(np.var(true_y)))
    mse_o = float(np.mean(mse_ode)) if mse_ode else float("nan")
    mse_p = float(np.mean(mse_persist)) if mse_persist else float("nan")
    var = float(np.mean(var_acc)) if var_acc else float("nan")
    return {
        "horizon": horizon,
        "obs_name": obs_name,
        "mse": mse_o,
        "mse_persistence": mse_p,
        "skill": 1.0 - mse_o / max(var, 1e-12),
        "skill_persistence": 1.0 - mse_p / max(var, 1e-12),
    }


def cem_plan_predator_prey(
    r0: float,
    w0: float,
    g0: float,
    target: float,
    ode: PredatorPreyODE,
    cfg: MPCConfig,
    *,
    u_prev: float = 0.0,
    warm_start: np.ndarray | None = None,
    obs_name: str = WOLF,
) -> np.ndarray:
    """CEM plan that rolls the predator–prey ODE and scores ``obs_name`` tracking."""
    h, s = cfg.plan_horizon, cfg.n_samples
    n_elite = max(1, int(cfg.elite_frac * s))
    mean = np.zeros(h, dtype=np.float64) if warm_start is None else warm_start.astype(np.float64)
    std = np.full(h, cfg.init_std, dtype=np.float64)
    track_wolf = obs_name == WOLF

    for _ in range(cfg.n_iters):
        u = np.clip(
            mean[None, :] + std[None, :] * np.random.randn(s, h),
            cfg.u_min,
            cfg.u_max,
        )
        cost = np.zeros(s, dtype=np.float64)
        for i in range(s):
            r, w, g, prev = r0, w0, g0, u_prev
            c = 0.0
            for k in range(h):
                r, w, g = predator_prey_step(r, w, g, float(u[i, k]), prev, ode)
                y = w if track_wolf else r
                c += (y - target) ** 2 + cfg.control_cost * float(u[i, k]) ** 2
                prev = float(u[i, k])
            cost[i] = c
        elite = np.argpartition(cost, n_elite)[:n_elite]
        mean = u[elite].mean(axis=0)
        std = u[elite].std(axis=0) + 1e-3
    return mean.astype(np.float32)


def closed_loop_predator_prey_ode(
    ode: PredatorPreyODE,
    target: float,
    *,
    obs_name: str = WOLF,
    steps: int = 100,
    mpc: MPCConfig | None = None,
    cfg=None,
    initial_rabbits: int = 120,
    initial_wolves: int = 16,
    initial_grass_prob: float = 0.35,
    seed: int = 0,
) -> dict:
    """Receding-horizon MPC with the predator–prey ODE, closed on the wolf ABM."""
    from koopman_control.data.wolf_rabbit_grass import WolfRabbitGrassConfig, WolfRabbitGrassModel

    mpc = mpc or MPCConfig()
    cfg = cfg or WolfRabbitGrassConfig()
    sim = WolfRabbitGrassModel(
        cfg=cfg,
        initial_rabbits=initial_rabbits,
        initial_wolves=initial_wolves,
        initial_grass_prob=initial_grass_prob,
        seed=seed,
    )

    true_series = [sim.observables()[obs_name]]
    controls = [0.0]
    warm: np.ndarray | None = None
    u_prev = 0.0
    for _ in range(steps):
        obs = sim.observables()
        plan = cem_plan_predator_prey(
            obs[RABBIT],
            obs[WOLF],
            obs[GRASS],
            target,
            ode,
            mpc,
            u_prev=u_prev,
            warm_start=warm,
            obs_name=obs_name,
        )
        u = float(np.clip(plan[0], mpc.u_min, mpc.u_max))
        sim.step(u)
        true_series.append(sim.observables()[obs_name])
        controls.append(u)
        warm = np.concatenate([plan[1:], plan[-1:]])
        u_prev = u

    true_arr = np.asarray(true_series, dtype=np.float32)
    return {
        "obs_name": obs_name,
        "target": float(target),
        "control": np.asarray(controls, dtype=np.float32),
        "true": true_arr,
        "tracking_rmse": float(np.sqrt(np.mean((true_arr[1:] - target) ** 2))),
        "final_error": float(true_arr[-1] - target),
        "planner": "predator_prey_ode",
    }


# ----------------------------------------------------------------------
# Tumor–healthy-tissue ODE (chemotherapy acts through a delayed drug pool)
# ----------------------------------------------------------------------
TUMOR = "tumor_count"
HEALTHY = "healthy_count"
NUTRIENT = "mean_nutrient"
DRUG = "mean_drug"


@dataclass(frozen=True)
class TumorODE:
    """Discrete tumor/healthy/nutrient/drug map with delayed drug delivery.

    Fitted by least squares on one-step deltas:

        ΔT/T ~ growth_t*N - death_t - kill_t*D
        ΔH/H ~ growth_h*N - death_h - kill_h*D - invasion*T
        ΔN   ~ recover*(1-N) - consume_t*T - consume_h*H
        ΔD   ~ deliver_now*u + deliver_lag*u_prev - decay*D
    """

    growth_t: float
    death_t: float
    kill_t: float
    growth_h: float
    death_h: float
    kill_h: float
    invasion: float
    recover: float
    consume_t: float
    consume_h: float
    deliver_now: float
    deliver_lag: float
    decay: float
    tumor_one_step_r2: float
    healthy_one_step_r2: float
    nutrient_one_step_r2: float
    drug_one_step_r2: float

    def serializable(self) -> dict:
        return asdict(self)


def _ridge_solve(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    gram = x.T @ x + ridge * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


def fit_tumor_ode(
    trajs: list[dict],
    *,
    ridge: float = 1e-6,
    tumor_name: str = TUMOR,
    healthy_name: str = HEALTHY,
    nutrient_name: str = NUTRIENT,
    drug_name: str = DRUG,
) -> TumorODE:
    """Least-squares fit of the four-state tumor map on trajectory observables."""
    names = list(trajs[0]["obs_names"])
    i_t = _obs_index(names, tumor_name)
    i_h = _obs_index(names, healthy_name)
    i_n = _obs_index(names, nutrient_name)
    i_d = _obs_index(names, drug_name)

    x_t, y_t, x_h, y_h, x_n, y_n, x_d, y_d = [], [], [], [], [], [], [], []
    for tr in trajs:
        obs = np.asarray(tr["obs"], dtype=np.float64)
        u = np.asarray(tr["controls"], dtype=np.float64)
        n_trans = min(len(u), len(obs) - 1)
        for t in range(n_trans):
            tumor, healthy = obs[t, i_t], obs[t, i_h]
            nutrient, drug = obs[t, i_n], obs[t, i_d]
            tumor1, healthy1 = obs[t + 1, i_t], obs[t + 1, i_h]
            nutrient1, drug1 = obs[t + 1, i_n], obs[t + 1, i_d]
            u_now = float(u[t])
            u_prev = float(u[t - 1]) if t > 0 else 0.0
            if tumor > 1e-6:
                x_t.append([nutrient, -1.0, -drug])
                y_t.append((tumor1 - tumor) / tumor)
            if healthy > 1e-6:
                x_h.append([nutrient, -1.0, -drug, -tumor])
                y_h.append((healthy1 - healthy) / healthy)
            x_n.append([1.0 - nutrient, -tumor, -healthy])
            y_n.append(nutrient1 - nutrient)
            x_d.append([u_now, u_prev, -drug])
            y_d.append(drug1 - drug)

    x_t_a, y_t_a = np.asarray(x_t), np.asarray(y_t)
    x_h_a, y_h_a = np.asarray(x_h), np.asarray(y_h)
    x_n_a, y_n_a = np.asarray(x_n), np.asarray(y_n)
    x_d_a, y_d_a = np.asarray(x_d), np.asarray(y_d)
    if min(len(x_t_a), len(x_h_a), len(x_n_a), len(x_d_a)) < 4:
        raise ValueError("not enough transitions to fit the tumor ODE")

    th_t = _ridge_solve(x_t_a, y_t_a, ridge)
    th_h = _ridge_solve(x_h_a, y_h_a, ridge)
    th_n = _ridge_solve(x_n_a, y_n_a, ridge)
    th_d = _ridge_solve(x_d_a, y_d_a, ridge)
    return TumorODE(
        growth_t=float(th_t[0]),
        death_t=float(th_t[1]),
        kill_t=float(th_t[2]),
        growth_h=float(th_h[0]),
        death_h=float(th_h[1]),
        kill_h=float(th_h[2]),
        invasion=float(th_h[3]),
        recover=float(th_n[0]),
        consume_t=float(th_n[1]),
        consume_h=float(th_n[2]),
        deliver_now=float(th_d[0]),
        deliver_lag=float(th_d[1]),
        decay=float(th_d[2]),
        tumor_one_step_r2=_r2(y_t_a, x_t_a @ th_t),
        healthy_one_step_r2=_r2(y_h_a, x_h_a @ th_h),
        nutrient_one_step_r2=_r2(y_n_a, x_n_a @ th_n),
        drug_one_step_r2=_r2(y_d_a, x_d_a @ th_d),
    )


def tumor_ode_step(
    tumor: float,
    healthy: float,
    nutrient: float,
    drug: float,
    u: float,
    u_prev: float,
    ode: TumorODE,
) -> tuple[float, float, float, float]:
    """One discrete Euler step of the fitted tumor ODE."""
    t_next = tumor + tumor * (ode.growth_t * nutrient - ode.death_t - ode.kill_t * drug)
    h_next = healthy + healthy * (
        ode.growth_h * nutrient - ode.death_h - ode.kill_h * drug - ode.invasion * tumor
    )
    n_next = (
        nutrient
        + ode.recover * (1.0 - nutrient)
        - ode.consume_t * tumor
        - ode.consume_h * healthy
    )
    d_next = drug + ode.deliver_now * u + ode.deliver_lag * u_prev - ode.decay * drug
    return (
        max(0.0, float(t_next)),
        max(0.0, float(h_next)),
        float(np.clip(n_next, 0.0, 1.0)),
        float(np.clip(d_next, 0.0, 1.0)),
    )


def tumor_ode_prediction_skill(
    trajs: list[dict],
    ode: TumorODE,
    *,
    horizon: int = 16,
    obs_name: str = TUMOR,
    tumor_name: str = TUMOR,
    healthy_name: str = HEALTHY,
    nutrient_name: str = NUTRIENT,
    drug_name: str = DRUG,
) -> dict:
    """Held-out free-rollout skill of the ODE on ``obs_name`` (vs persistence)."""
    names = list(trajs[0]["obs_names"])
    i_t = _obs_index(names, tumor_name)
    i_h = _obs_index(names, healthy_name)
    i_n = _obs_index(names, nutrient_name)
    i_d = _obs_index(names, drug_name)
    i_y = _obs_index(names, obs_name)
    mse_ode, mse_persist, var_acc = [], [], []
    for tr in trajs:
        obs = np.asarray(tr["obs"], dtype=np.float64)
        u = np.asarray(tr["controls"], dtype=np.float64)
        n_trans = min(len(u), len(obs) - 1)
        t_max = n_trans - horizon
        if t_max <= 0:
            continue
        starts = np.linspace(0, t_max, num=min(8, t_max + 1), dtype=int)
        for t0 in starts:
            true_y = obs[t0 : t0 + horizon + 1, i_y]
            tumor = float(obs[t0, i_t])
            healthy = float(obs[t0, i_h])
            nutrient = float(obs[t0, i_n])
            drug = float(obs[t0, i_d])
            u_prev = 0.0
            pred = [true_y[0]]
            for k in range(horizon):
                tumor, healthy, nutrient, drug = tumor_ode_step(
                    tumor, healthy, nutrient, drug, float(u[t0 + k]), u_prev, ode
                )
                u_prev = float(u[t0 + k])
                if obs_name == tumor_name:
                    pred.append(tumor)
                elif obs_name == healthy_name:
                    pred.append(healthy)
                elif obs_name == nutrient_name:
                    pred.append(nutrient)
                else:
                    pred.append(drug)
            pred_a = np.asarray(pred, dtype=np.float64)
            mse_ode.append(float(np.mean((pred_a - true_y) ** 2)))
            mse_persist.append(float(np.mean((true_y[0] - true_y) ** 2)))
            var_acc.append(float(np.var(true_y)))
    mse_o = float(np.mean(mse_ode)) if mse_ode else float("nan")
    mse_p = float(np.mean(mse_persist)) if mse_persist else float("nan")
    var = float(np.mean(var_acc)) if var_acc else float("nan")
    return {
        "horizon": horizon,
        "obs_name": obs_name,
        "mse": mse_o,
        "mse_persistence": mse_p,
        "skill": 1.0 - mse_o / max(var, 1e-12),
        "skill_persistence": 1.0 - mse_p / max(var, 1e-12),
    }


def cem_plan_tumor_ode(
    tumor0: float,
    healthy0: float,
    nutrient0: float,
    drug0: float,
    ode: TumorODE,
    cfg: TumorMPCConfig,
    *,
    tumor_target: float = 0.0,
    healthy_reference: float,
    u_prev: float = 0.0,
    warm_start: np.ndarray | None = None,
) -> np.ndarray:
    """CEM plan that rolls the tumor ODE with the same multi-objective cost as JEPA."""
    horizon, n_samples = cfg.plan_horizon, cfg.n_samples
    n_elite = max(1, int(cfg.elite_frac * n_samples))
    mean = (
        np.zeros(horizon, dtype=np.float64)
        if warm_start is None
        else np.asarray(warm_start, dtype=np.float64)
    )
    std = np.full(horizon, cfg.init_std, dtype=np.float64)
    tumor_scale = max(float(cfg.tumor_scale), 1e-6)
    healthy_scale = max(float(cfg.healthy_scale), 1e-6)

    for _ in range(cfg.n_iters):
        u = np.clip(
            mean[None, :] + std[None, :] * np.random.randn(n_samples, horizon),
            cfg.u_min,
            cfg.u_max,
        )
        cost = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            tumor, healthy, nutrient, drug, prev = tumor0, healthy0, nutrient0, drug0, u_prev
            c = 0.0
            for k in range(horizon):
                u_now = float(u[i, k])
                tumor, healthy, nutrient, drug = tumor_ode_step(
                    tumor, healthy, nutrient, drug, u_now, prev, ode
                )
                tumor_error = (tumor - tumor_target) / tumor_scale
                healthy_shortfall = max(healthy_reference - healthy, 0.0) / healthy_scale
                c += (
                    cfg.tumor_weight * tumor_error**2
                    + cfg.healthy_weight * healthy_shortfall**2
                    + cfg.control_cost * u_now**2
                    + cfg.slew_cost * (u_now - prev) ** 2
                )
                prev = u_now
            cost[i] = c
        elite = np.argpartition(cost, n_elite)[:n_elite]
        mean = u[elite].mean(axis=0)
        std = u[elite].std(axis=0) + 1e-3
    return mean.astype(np.float32)


def closed_loop_tumor_ode(
    ode: TumorODE,
    *,
    tumor_target: float = 0.0,
    healthy_reference: float | None = None,
    steps: int = 160,
    mpc: TumorMPCConfig | None = None,
    cfg: TumorTissueConfig | None = None,
    initial_healthy_frac: float = 0.94,
    initial_tumor_radius: float = 6.0,
    tumor_center_x: float | None = None,
    tumor_center_y: float | None = None,
    seed: int = 0,
) -> dict:
    """Receding-horizon MPC with the tumor ODE, closed on the true tissue ABM.

    Each step the planner is given oracle ``(tumor, healthy, nutrient, drug)``
    from the simulator — a stronger information regime than image → latent →
    readout. Return schema matches :func:`jepa_control.control.closed_loop_tumor`.
    """
    mpc = mpc or TumorMPCConfig()
    sim = TumorTissueModel(
        cfg=cfg or TumorTissueConfig(),
        initial_healthy_frac=initial_healthy_frac,
        initial_tumor_radius=initial_tumor_radius,
        tumor_center_x=tumor_center_x,
        tumor_center_y=tumor_center_y,
        seed=seed,
    )
    initial = sim.observables()
    healthy_reference = (
        float(initial["healthy_count"]) if healthy_reference is None else float(healthy_reference)
    )
    tumor = [initial["tumor_count"]]
    healthy = [initial["healthy_count"]]
    controls = [0.0]
    warm: np.ndarray | None = None
    u_prev = 0.0
    for _ in range(steps):
        obs = sim.observables()
        plan = cem_plan_tumor_ode(
            obs[TUMOR],
            obs[HEALTHY],
            obs[NUTRIENT],
            obs[DRUG],
            ode,
            mpc,
            tumor_target=tumor_target,
            healthy_reference=healthy_reference,
            u_prev=u_prev,
            warm_start=warm,
        )
        u = float(np.clip(plan[0], mpc.u_min, mpc.u_max))
        sim.step(u)
        obs = sim.observables()
        tumor.append(obs[TUMOR])
        healthy.append(obs[HEALTHY])
        controls.append(u)
        warm = np.concatenate([plan[1:], plan[-1:]])
        u_prev = u

    tumor_arr = np.asarray(tumor, dtype=np.float32)
    healthy_arr = np.asarray(healthy, dtype=np.float32)
    control_arr = np.asarray(controls, dtype=np.float32)
    shortfall = np.maximum(healthy_reference - healthy_arr[1:], 0.0)
    return {
        "tumor_target": float(tumor_target),
        "healthy_reference": healthy_reference,
        "tumor": tumor_arr,
        "healthy": healthy_arr,
        "control": control_arr,
        "tumor_rmse": float(np.sqrt(np.mean((tumor_arr[1:] - tumor_target) ** 2))),
        "healthy_shortfall_rmse": float(np.sqrt(np.mean(shortfall**2))),
        "final_tumor": float(tumor_arr[-1]),
        "final_healthy": float(healthy_arr[-1]),
        "cumulative_dose": float(control_arr[1:].sum()),
        "mean_dose": float(control_arr[1:].mean()),
        "abm": "tumor_tissue",
        "planner": "tumor_ode",
    }


# ----------------------------------------------------------------------
# Agentic spatial SIR ODE (vaccination acts with one-step lag)
# ----------------------------------------------------------------------
SUSCEPTIBLE = "susceptible_count"
INFECTED = "infected_count"
RECOVERED = "recovered_count"


@dataclass(frozen=True)
class SIRODE:
    """Discrete SIR map with delayed, population-scaled vaccination.

    Fitted on absolute one-step count deltas with non-negative least squares:

        ΔI = beta * S I / N - gamma * I
        ΔS = -beta * S I / N - S * (vacc_now * u + vacc_lag * u_prev)
        ΔR = gamma * I + S * (vacc_now * u + vacc_lag * u_prev)

    ``vacc_now`` and ``vacc_lag`` are per-capita rates (matching the ABM's
    Bernoulli vaccination probability scale), not raw count removals.
    """

    beta: float
    gamma: float
    vacc_now: float
    vacc_lag: float
    infected_one_step_r2: float
    susceptible_one_step_r2: float
    recovered_one_step_r2: float

    def serializable(self) -> dict:
        return asdict(self)


def _nnls_nonneg(x: np.ndarray, y: np.ndarray, *, ridge: float = 1e-6) -> np.ndarray:
    """Projected non-negative least squares (no scipy dependency)."""
    n_features = x.shape[1]
    theta = _ridge_solve(x, y, ridge)
    for _ in range(8):
        theta = np.maximum(theta, 0.0)
        active = theta > 1e-12
        if not active.any():
            return np.zeros(n_features, dtype=np.float64)
        theta_active = _ridge_solve(x[:, active], y, ridge)
        theta = np.zeros(n_features, dtype=np.float64)
        theta[active] = np.maximum(theta_active, 0.0)
    return theta


def fit_sir_ode(
    trajs: list[dict],
    *,
    ridge: float = 1e-6,
    susceptible_name: str = SUSCEPTIBLE,
    infected_name: str = INFECTED,
    recovered_name: str = RECOVERED,
) -> SIRODE:
    """Least-squares fit of the compartment map on trajectory observables."""
    names = list(trajs[0]["obs_names"])
    i_s = _obs_index(names, susceptible_name)
    i_i = _obs_index(names, infected_name)
    i_r = _obs_index(names, recovered_name)

    x_i, y_i = [], []
    vacc_rows: list[tuple[float, float, float, float, float]] = []
    for tr in trajs:
        obs = np.asarray(tr["obs"], dtype=np.float64)
        u = np.asarray(tr["controls"], dtype=np.float64)
        n_trans = min(len(u), len(obs) - 1)
        for t in range(n_trans):
            susceptible, infected, recovered = obs[t, i_s], obs[t, i_i], obs[t, i_r]
            i1, r1 = obs[t + 1, i_i], obs[t + 1, i_r]
            population = max(float(susceptible + infected + recovered), 1.0)
            u_now = float(u[t])
            u_prev = float(u[t - 1]) if t > 0 else 0.0
            contact = susceptible * infected / population
            x_i.append([contact, -infected])
            y_i.append(i1 - infected)
            vacc_rows.append((susceptible, infected, r1 - recovered, u_now, u_prev))

    x_i_a, y_i_a = np.asarray(x_i), np.asarray(y_i)
    if len(x_i_a) < 4:
        raise ValueError("not enough transitions to fit the SIR ODE")

    beta, gamma = _nnls_nonneg(x_i_a, y_i_a, ridge=ridge)
    x_v, y_v, x_s, y_s = [], [], [], []
    for susceptible, infected, delta_r, u_now, u_prev in vacc_rows:
        x_v.append([susceptible * u_now, susceptible * u_prev])
        y_v.append(delta_r - gamma * infected)
    for tr in trajs:
        obs = np.asarray(tr["obs"], dtype=np.float64)
        u = np.asarray(tr["controls"], dtype=np.float64)
        n_trans = min(len(u), len(obs) - 1)
        for t in range(n_trans):
            susceptible, infected = obs[t, i_s], obs[t, i_i]
            population = max(float(obs[t, i_s] + obs[t, i_i] + obs[t, i_r]), 1.0)
            u_now = float(u[t])
            u_prev = float(u[t - 1]) if t > 0 else 0.0
            contact = susceptible * infected / population
            x_s.append([susceptible * u_now, susceptible * u_prev])
            y_s.append(obs[t + 1, i_s] - susceptible + beta * contact)
    x_v_a, y_v_a = np.asarray(x_v), np.asarray(y_v)
    x_s_a, y_s_a = np.asarray(x_s), np.asarray(y_s)
    vacc_now, vacc_lag = _nnls_nonneg(x_v_a, y_v_a, ridge=ridge)

    contacts = x_i_a[:, 0]
    infected_levels = -x_i_a[:, 1]
    pred_i = beta * contacts - gamma * infected_levels
    pred_s = -beta * contacts - x_s_a @ np.array([vacc_now, vacc_lag])
    pred_r = gamma * infected_levels + x_v_a @ np.array([vacc_now, vacc_lag])
    return SIRODE(
        beta=float(beta),
        gamma=float(gamma),
        vacc_now=float(vacc_now),
        vacc_lag=float(vacc_lag),
        infected_one_step_r2=_r2(y_i_a, pred_i),
        susceptible_one_step_r2=_r2(y_s_a, pred_s),
        recovered_one_step_r2=_r2(y_v_a, pred_r),
    )


def sir_ode_step(
    susceptible: float,
    infected: float,
    recovered: float,
    u: float,
    u_prev: float,
    ode: SIRODE,
) -> tuple[float, float, float]:
    """One discrete Euler step of the fitted SIR ODE."""
    population = max(susceptible + infected + recovered, 1.0)
    infection = ode.beta * susceptible * infected / population
    vaccination = susceptible * (ode.vacc_now * u + ode.vacc_lag * u_prev)
    ds = -infection - vaccination
    di = infection - ode.gamma * infected
    dr = ode.gamma * infected + vaccination
    return (
        max(0.0, float(susceptible + ds)),
        max(0.0, float(infected + di)),
        max(0.0, float(recovered + dr)),
    )


def sir_ode_prediction_skill(
    trajs: list[dict],
    ode: SIRODE,
    *,
    horizon: int = 16,
    obs_name: str = INFECTED,
    susceptible_name: str = SUSCEPTIBLE,
    infected_name: str = INFECTED,
    recovered_name: str = RECOVERED,
) -> dict:
    """Held-out free-rollout skill of the ODE on ``obs_name`` (vs persistence)."""
    names = list(trajs[0]["obs_names"])
    i_s = _obs_index(names, susceptible_name)
    i_i = _obs_index(names, infected_name)
    i_r = _obs_index(names, recovered_name)
    i_y = _obs_index(names, obs_name)
    mse_ode, mse_persist, var_acc = [], [], []
    for tr in trajs:
        obs = np.asarray(tr["obs"], dtype=np.float64)
        u = np.asarray(tr["controls"], dtype=np.float64)
        n_trans = min(len(u), len(obs) - 1)
        t_max = n_trans - horizon
        if t_max <= 0:
            continue
        starts = np.linspace(0, t_max, num=min(8, t_max + 1), dtype=int)
        for t0 in starts:
            true_y = obs[t0 : t0 + horizon + 1, i_y]
            susceptible = float(obs[t0, i_s])
            infected = float(obs[t0, i_i])
            recovered = float(obs[t0, i_r])
            u_prev = 0.0
            pred = [true_y[0]]
            for k in range(horizon):
                susceptible, infected, recovered = sir_ode_step(
                    susceptible,
                    infected,
                    recovered,
                    float(u[t0 + k]),
                    u_prev,
                    ode,
                )
                u_prev = float(u[t0 + k])
                if obs_name == susceptible_name:
                    pred.append(susceptible)
                elif obs_name == infected_name:
                    pred.append(infected)
                else:
                    pred.append(recovered)
            pred_a = np.asarray(pred, dtype=np.float64)
            mse_ode.append(float(np.mean((pred_a - true_y) ** 2)))
            mse_persist.append(float(np.mean((true_y[0] - true_y) ** 2)))
            var_acc.append(float(np.var(true_y)))
    mse_o = float(np.mean(mse_ode)) if mse_ode else float("nan")
    mse_p = float(np.mean(mse_persist)) if mse_persist else float("nan")
    var = float(np.mean(var_acc)) if var_acc else float("nan")
    return {
        "horizon": horizon,
        "obs_name": obs_name,
        "mse": mse_o,
        "mse_persistence": mse_p,
        "skill": 1.0 - mse_o / max(var, 1e-12),
        "skill_persistence": 1.0 - mse_p / max(var, 1e-12),
    }


def cem_plan_sir_ode(
    susceptible0: float,
    infected0: float,
    recovered0: float,
    ode: SIRODE,
    cfg: SIRMPCConfig,
    *,
    infected_target: float = 0.0,
    susceptible_floor: float,
    u_prev: float = 0.0,
    warm_start: np.ndarray | None = None,
) -> np.ndarray:
    """CEM plan that rolls the SIR ODE with the same multi-objective cost as JEPA."""
    horizon, n_samples = cfg.plan_horizon, cfg.n_samples
    n_elite = max(1, int(cfg.elite_frac * n_samples))
    mean = (
        np.zeros(horizon, dtype=np.float64)
        if warm_start is None
        else np.asarray(warm_start, dtype=np.float64)
    )
    std = np.full(horizon, cfg.init_std, dtype=np.float64)
    infected_scale = max(float(cfg.infected_scale), 1e-6)
    susceptible_scale = max(float(cfg.susceptible_scale), 1e-6)

    for _ in range(cfg.n_iters):
        u = np.clip(
            mean[None, :] + std[None, :] * np.random.randn(n_samples, horizon),
            cfg.u_min,
            cfg.u_max,
        )
        cost = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            susceptible, infected, recovered, prev = (
                susceptible0,
                infected0,
                recovered0,
                u_prev,
            )
            c = 0.0
            for k in range(horizon):
                u_now = float(u[i, k])
                susceptible, infected, recovered = sir_ode_step(
                    susceptible, infected, recovered, u_now, prev, ode
                )
                infected_error = (infected - infected_target) / infected_scale
                susceptible_shortfall = max(susceptible_floor - susceptible, 0.0) / susceptible_scale
                c += (
                    cfg.infected_weight * infected_error**2
                    + cfg.susceptible_weight * susceptible_shortfall**2
                    + cfg.control_cost * u_now**2
                    + cfg.slew_cost * (u_now - prev) ** 2
                )
                prev = u_now
            cost[i] = c
        elite = np.argpartition(cost, n_elite)[:n_elite]
        mean = u[elite].mean(axis=0)
        std = u[elite].std(axis=0) + 1e-3
    return mean.astype(np.float32)


def closed_loop_sir_ode(
    ode: SIRODE,
    *,
    infected_target: float = 0.0,
    susceptible_floor: float | None = None,
    steps: int = 160,
    mpc: SIRMPCConfig | None = None,
    cfg: AgenticSIRConfig | None = None,
    n_agents: int = 500,
    initial_infected: int = 16,
    seed_center_x: float | None = None,
    seed_center_y: float | None = None,
    seed_radius: float = 5.0,
    seed: int = 0,
) -> dict:
    """Receding-horizon MPC with the SIR ODE, closed on the true agentic ABM.

    Each step the planner is given oracle ``(S, I, R)`` from the simulator — a
    stronger information regime than image → latent → readout. Return schema
    matches :func:`jepa_control.control.closed_loop_sir`.
    """
    mpc = mpc or SIRMPCConfig()
    sim = AgenticSIRModel(
        cfg=cfg or AgenticSIRConfig(),
        n_agents=n_agents,
        initial_infected=initial_infected,
        seed_center_x=seed_center_x,
        seed_center_y=seed_center_y,
        seed_radius=seed_radius,
        seed=seed,
    )
    initial = sim.observables()
    susceptible_floor = (
        float(initial["susceptible_count"]) * float(mpc.susceptible_floor_frac)
        if susceptible_floor is None
        else float(susceptible_floor)
    )
    infected = [initial["infected_count"]]
    susceptible = [initial["susceptible_count"]]
    recovered = [initial["recovered_count"]]
    incidence = [initial["cumulative_incidence"]]
    controls = [0.0]
    warm: np.ndarray | None = None
    u_prev = 0.0
    for _ in range(steps):
        obs = sim.observables()
        plan = cem_plan_sir_ode(
            obs[SUSCEPTIBLE],
            obs[INFECTED],
            obs[RECOVERED],
            ode,
            mpc,
            infected_target=infected_target,
            susceptible_floor=susceptible_floor,
            u_prev=u_prev,
            warm_start=warm,
        )
        u = float(np.clip(plan[0], mpc.u_min, mpc.u_max))
        sim.step(u)
        obs = sim.observables()
        infected.append(obs[INFECTED])
        susceptible.append(obs[SUSCEPTIBLE])
        recovered.append(obs[RECOVERED])
        incidence.append(obs["cumulative_incidence"])
        controls.append(u)
        warm = np.concatenate([plan[1:], plan[-1:]])
        u_prev = u

    infected_arr = np.asarray(infected, dtype=np.float32)
    susceptible_arr = np.asarray(susceptible, dtype=np.float32)
    recovered_arr = np.asarray(recovered, dtype=np.float32)
    incidence_arr = np.asarray(incidence, dtype=np.float32)
    control_arr = np.asarray(controls, dtype=np.float32)
    shortfall = np.maximum(susceptible_floor - susceptible_arr[1:], 0.0)
    return {
        "infected_target": float(infected_target),
        "susceptible_floor": susceptible_floor,
        "infected": infected_arr,
        "susceptible": susceptible_arr,
        "recovered": recovered_arr,
        "cumulative_incidence": incidence_arr,
        "control": control_arr,
        "infected_rmse": float(np.sqrt(np.mean((infected_arr[1:] - infected_target) ** 2))),
        "susceptible_shortfall_rmse": float(np.sqrt(np.mean(shortfall**2))),
        "final_infected": float(infected_arr[-1]),
        "final_susceptible": float(susceptible_arr[-1]),
        "final_incidence": float(incidence_arr[-1]),
        "cumulative_dose": float(control_arr[1:].sum()),
        "mean_dose": float(control_arr[1:].mean()),
        "abm": "agentic_sir",
        "planner": "sir_ode",
    }
