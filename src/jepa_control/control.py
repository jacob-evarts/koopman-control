"""Sampling-based MPC over the learned JEPA predictor, closed-loop vs the ABM.

The latent is not constrained to be linear, so control is model-predictive: at
each step we sample candidate control sequences, roll them out **in latent
space** with the learned predictor, score them with a cost defined in **readout
space** (drive a macrostate such as the rabbit population to a target), and apply
the first action of the best plan (receding horizon). The plan is refined with
the cross-entropy method (CEM).

Everything is scored through the post-hoc linear readout ``y = z W + b`` fitted
in :mod:`jepa_control.evaluate`, so no decoder is needed anywhere in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from jepa_control.model import JEPAControl
from koopman_control.data.agentic_sir import AgenticSIRConfig, AgenticSIRModel
from koopman_control.data.rabbit_grass import RabbitGrassConfig, RabbitGrassModel
from koopman_control.data.tumor_tissue import TumorTissueConfig, TumorTissueModel
from koopman_control.data.wolf_rabbit_grass import WolfRabbitGrassConfig, WolfRabbitGrassModel


@dataclass
class MPCConfig:
    plan_horizon: int = 15
    n_samples: int = 256
    n_iters: int = 4
    elite_frac: float = 0.1
    init_std: float = 0.3
    control_cost: float = 1.0  # weight on sum of u^2 over the plan (effort penalty)
    u_min: float = 0.0
    u_max: float = 1.0


@dataclass
class TumorMPCConfig(MPCConfig):
    """Multi-objective weights for tumor treatment.

    Population errors are normalized before weighting so the defaults remain
    meaningful when grid size or initial condition changes.
    """

    tumor_weight: float = 1.0
    healthy_weight: float = 1.0
    slew_cost: float = 0.1
    tumor_scale: float = 100.0
    healthy_scale: float = 1000.0


@dataclass
class SIRMPCConfig(MPCConfig):
    """Multi-objective weights for epidemic vaccination control.

    The planner trades off infected burden against vaccination effort and dose
    slew. Population errors are normalized before weighting.
    """

    infected_weight: float = 1.0
    slew_cost: float = 0.1
    infected_scale: float = 100.0
    # Soft penalty for depleting the remaining susceptible pool when infection
    # is already near the target -- discourages needless mass vaccination.
    susceptible_weight: float = 0.15
    susceptible_scale: float = 500.0
    susceptible_floor_frac: float = 0.35


@torch.no_grad()
def cem_plan(
    model: JEPAControl,
    z0: np.ndarray,
    target: float,
    readout: dict,
    obs_name: str,
    cfg: MPCConfig,
    *,
    u_prev: float = 0.0,
    warm_start: np.ndarray | None = None,
) -> np.ndarray:
    """Return the optimized action sequence ``(plan_horizon,)`` for one plan.

    ``target`` is the desired value of ``obs_name`` (e.g. a rabbit count). Only
    the first entry is applied when used in receding-horizon closed loop; the
    full sequence is returned so it can warm-start the next step.
    """
    n_lags = int(model.hparams.n_control_lags)
    idx = readout["names"].index(obs_name)
    w = torch.tensor(readout["W"][:, idx], dtype=torch.float32)
    b = float(readout["b"][idx])

    h, s = cfg.plan_horizon, cfg.n_samples
    n_elite = max(1, int(cfg.elite_frac * s))
    mean = np.zeros(h, dtype=np.float32) if warm_start is None else warm_start.astype(np.float32)
    std = np.full(h, cfg.init_std, dtype=np.float32)

    z0_t = torch.tensor(np.asarray(z0, dtype=np.float32)).unsqueeze(0).repeat(s, 1)
    for _ in range(cfg.n_iters):
        u = np.clip(
            mean[None, :] + std[None, :] * np.random.randn(s, h),
            cfg.u_min,
            cfg.u_max,
        ).astype(np.float32)
        u_t = torch.from_numpy(u)
        z = z0_t.clone()
        prev = torch.full((s, max(0, n_lags - 1)), float(u_prev))
        cost = torch.zeros(s)
        for k in range(h):
            u_now = u_t[:, k : k + 1]
            hist = torch.cat([u_now, prev], dim=1)[:, :n_lags]
            z = model.step(z, hist)
            y = z @ w + b
            cost = cost + (y - target) ** 2 + cfg.control_cost * u_now[:, 0] ** 2
            if n_lags > 1:
                prev = torch.cat([u_now, prev], dim=1)[:, : n_lags - 1]

        elite = torch.topk(cost, n_elite, largest=False).indices.numpy()
        mean = u[elite].mean(axis=0)
        std = u[elite].std(axis=0) + 1e-3
    return mean


@torch.no_grad()
def cem_plan_tumor(
    model: JEPAControl,
    z0: np.ndarray,
    readout: dict,
    cfg: TumorMPCConfig,
    *,
    tumor_target: float = 0.0,
    healthy_reference: float,
    u_prev: float = 0.0,
    warm_start: np.ndarray | None = None,
) -> np.ndarray:
    """CEM plan balancing tumor burden, healthy tissue, dose, and dose slew."""
    names = list(readout["names"])
    tumor_idx = names.index("tumor_count")
    healthy_idx = names.index("healthy_count")
    w_t = torch.tensor(readout["W"][:, tumor_idx], dtype=torch.float32)
    w_h = torch.tensor(readout["W"][:, healthy_idx], dtype=torch.float32)
    b_t = float(readout["b"][tumor_idx])
    b_h = float(readout["b"][healthy_idx])

    n_lags = int(model.hparams.n_control_lags)
    horizon, n_samples = cfg.plan_horizon, cfg.n_samples
    n_elite = max(1, int(cfg.elite_frac * n_samples))
    mean = (
        np.zeros(horizon, dtype=np.float32)
        if warm_start is None
        else np.asarray(warm_start, dtype=np.float32)
    )
    std = np.full(horizon, cfg.init_std, dtype=np.float32)
    z0_t = torch.tensor(np.asarray(z0, dtype=np.float32)).unsqueeze(0).repeat(n_samples, 1)

    tumor_scale = max(float(cfg.tumor_scale), 1e-6)
    healthy_scale = max(float(cfg.healthy_scale), 1e-6)
    for _ in range(cfg.n_iters):
        u = np.clip(
            mean[None, :] + std[None, :] * np.random.randn(n_samples, horizon),
            cfg.u_min,
            cfg.u_max,
        ).astype(np.float32)
        u_t = torch.from_numpy(u)
        z = z0_t.clone()
        prev_hist = torch.full((n_samples, max(0, n_lags - 1)), float(u_prev))
        previous_u = torch.full((n_samples,), float(u_prev))
        cost = torch.zeros(n_samples)
        for k in range(horizon):
            u_now = u_t[:, k : k + 1]
            hist = torch.cat([u_now, prev_hist], dim=1)[:, :n_lags]
            z = model.step(z, hist)
            tumor = z @ w_t + b_t
            healthy = z @ w_h + b_h
            tumor_error = (tumor - tumor_target) / tumor_scale
            # Preserving more than the reference is not penalized.
            healthy_shortfall = torch.relu(healthy_reference - healthy) / healthy_scale
            effort = u_now[:, 0] ** 2
            slew = (u_now[:, 0] - previous_u) ** 2
            cost += (
                cfg.tumor_weight * tumor_error**2
                + cfg.healthy_weight * healthy_shortfall**2
                + cfg.control_cost * effort
                + cfg.slew_cost * slew
            )
            previous_u = u_now[:, 0]
            if n_lags > 1:
                prev_hist = torch.cat([u_now, prev_hist], dim=1)[:, : n_lags - 1]

        elite = torch.topk(cost, n_elite, largest=False).indices.numpy()
        mean = u[elite].mean(axis=0)
        std = u[elite].std(axis=0) + 1e-3
    return mean


def _make_sim(
    *,
    abm: str,
    cfg,
    initial_rabbits: int,
    initial_wolves: int,
    initial_grass_prob: float,
    seed: int,
):
    if abm == "rabbit_grass":
        cfg = cfg or RabbitGrassConfig()
        return RabbitGrassModel(
            cfg=cfg,
            initial_rabbits=initial_rabbits,
            initial_grass_prob=initial_grass_prob,
            seed=seed,
        )
    if abm == "wolf_rabbit_grass":
        cfg = cfg or WolfRabbitGrassConfig()
        return WolfRabbitGrassModel(
            cfg=cfg,
            initial_rabbits=initial_rabbits,
            initial_wolves=initial_wolves,
            initial_grass_prob=initial_grass_prob,
            seed=seed,
        )
    raise ValueError(f"unknown abm {abm!r}; expected 'rabbit_grass' or 'wolf_rabbit_grass'")


@torch.no_grad()
def closed_loop(
    model: JEPAControl,
    readout: dict,
    target: float,
    *,
    obs_name: str = "rabbit_count",
    steps: int = 100,
    mpc: MPCConfig | None = None,
    cfg=None,
    abm: str = "rabbit_grass",
    initial_rabbits: int = 120,
    initial_wolves: int = 16,
    initial_grass_prob: float = 0.35,
    seed: int = 0,
) -> dict:
    """Run receding-horizon MPC against the true ABM and track the macrostate.

    ``abm`` selects the simulator (``rabbit_grass`` or ``wolf_rabbit_grass``).
    On the wolf ABM the actuator culls wolves; ``obs_name`` can be
    ``wolf_count`` or ``rabbit_count``.
    """
    mpc = mpc or MPCConfig()
    sim = _make_sim(
        abm=abm,
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
        frame = sim.render()[None]  # (1, C, W, H)
        z0 = model.encode(torch.from_numpy(frame.astype(np.float32))).cpu().numpy()[0]
        plan = cem_plan(model, z0, target, readout, obs_name, mpc, u_prev=u_prev, warm_start=warm)
        u = float(np.clip(plan[0], mpc.u_min, mpc.u_max))
        sim.step(u)
        true_series.append(sim.observables()[obs_name])
        controls.append(u)
        warm = np.concatenate([plan[1:], plan[-1:]])  # shift for warm start
        u_prev = u

    true_arr = np.asarray(true_series, dtype=np.float32)
    return {
        "obs_name": obs_name,
        "target": float(target),
        "control": np.asarray(controls, dtype=np.float32),
        "true": true_arr,
        "tracking_rmse": float(np.sqrt(np.mean((true_arr[1:] - target) ** 2))),
        "final_error": float(true_arr[-1] - target),
        "abm": abm,
    }


def baseline_rollouts(
    *,
    obs_name: str = "rabbit_count",
    steps: int = 100,
    levels: tuple[float, ...] = (0.0, 0.5, 1.0),
    cfg=None,
    abm: str = "rabbit_grass",
    initial_rabbits: int = 120,
    initial_wolves: int = 16,
    initial_grass_prob: float = 0.35,
    seed: int = 0,
) -> dict:
    """Open-loop constant-cull references on the true ABM."""
    out: dict[str, np.ndarray] = {}
    for level in levels:
        sim = _make_sim(
            abm=abm,
            cfg=cfg,
            initial_rabbits=initial_rabbits,
            initial_wolves=initial_wolves,
            initial_grass_prob=initial_grass_prob,
            seed=seed,
        )
        series = [sim.observables()[obs_name]]
        for _ in range(steps):
            sim.step(level)
            series.append(sim.observables()[obs_name])
        out[f"u={level}"] = np.asarray(series, dtype=np.float32)
    return out


@torch.no_grad()
def closed_loop_tumor(
    model: JEPAControl,
    readout: dict,
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
    """Run multi-objective latent MPC against the true tumor-tissue ABM."""
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
        float(initial["healthy_count"])
        if healthy_reference is None
        else float(healthy_reference)
    )
    tumor = [initial["tumor_count"]]
    healthy = [initial["healthy_count"]]
    controls = [0.0]
    warm: np.ndarray | None = None
    u_prev = 0.0
    for _ in range(steps):
        frame = sim.render()[None]
        z0 = model.encode(torch.from_numpy(frame.astype(np.float32))).cpu().numpy()[0]
        plan = cem_plan_tumor(
            model,
            z0,
            readout,
            mpc,
            tumor_target=tumor_target,
            healthy_reference=healthy_reference,
            u_prev=u_prev,
            warm_start=warm,
        )
        u = float(np.clip(plan[0], mpc.u_min, mpc.u_max))
        sim.step(u)
        obs = sim.observables()
        tumor.append(obs["tumor_count"])
        healthy.append(obs["healthy_count"])
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
    }


def tumor_baseline_rollouts(
    *,
    steps: int = 160,
    levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    cfg: TumorTissueConfig | None = None,
    initial_healthy_frac: float = 0.94,
    initial_tumor_radius: float = 6.0,
    tumor_center_x: float | None = None,
    tumor_center_y: float | None = None,
    seed: int = 0,
) -> dict:
    """Open-loop constant-dose tumor and healthy-tissue references."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for level in levels:
        sim = TumorTissueModel(
            cfg=cfg or TumorTissueConfig(),
            initial_healthy_frac=initial_healthy_frac,
            initial_tumor_radius=initial_tumor_radius,
            tumor_center_x=tumor_center_x,
            tumor_center_y=tumor_center_y,
            seed=seed,
        )
        tumor = [sim.observables()["tumor_count"]]
        healthy = [sim.observables()["healthy_count"]]
        for _ in range(steps):
            sim.step(level)
            obs = sim.observables()
            tumor.append(obs["tumor_count"])
            healthy.append(obs["healthy_count"])
        out[f"u={level}"] = {
            "tumor": np.asarray(tumor, dtype=np.float32),
            "healthy": np.asarray(healthy, dtype=np.float32),
        }
    return out


@torch.no_grad()
def cem_plan_sir(
    model: JEPAControl,
    z0: np.ndarray,
    readout: dict,
    cfg: SIRMPCConfig,
    *,
    infected_target: float = 0.0,
    susceptible_floor: float,
    u_prev: float = 0.0,
    warm_start: np.ndarray | None = None,
) -> np.ndarray:
    """CEM plan balancing infected burden, vaccination effort, and over-vaccination."""
    names = list(readout["names"])
    infected_idx = names.index("infected_count")
    susceptible_idx = names.index("susceptible_count")
    w_i = torch.tensor(readout["W"][:, infected_idx], dtype=torch.float32)
    w_s = torch.tensor(readout["W"][:, susceptible_idx], dtype=torch.float32)
    b_i = float(readout["b"][infected_idx])
    b_s = float(readout["b"][susceptible_idx])

    n_lags = int(model.hparams.n_control_lags)
    horizon, n_samples = cfg.plan_horizon, cfg.n_samples
    n_elite = max(1, int(cfg.elite_frac * n_samples))
    mean = (
        np.zeros(horizon, dtype=np.float32)
        if warm_start is None
        else np.asarray(warm_start, dtype=np.float32)
    )
    std = np.full(horizon, cfg.init_std, dtype=np.float32)
    z0_t = torch.tensor(np.asarray(z0, dtype=np.float32)).unsqueeze(0).repeat(n_samples, 1)

    infected_scale = max(float(cfg.infected_scale), 1e-6)
    susceptible_scale = max(float(cfg.susceptible_scale), 1e-6)
    for _ in range(cfg.n_iters):
        u = np.clip(
            mean[None, :] + std[None, :] * np.random.randn(n_samples, horizon),
            cfg.u_min,
            cfg.u_max,
        ).astype(np.float32)
        u_t = torch.from_numpy(u)
        z = z0_t.clone()
        prev_hist = torch.full((n_samples, max(0, n_lags - 1)), float(u_prev))
        previous_u = torch.full((n_samples,), float(u_prev))
        cost = torch.zeros(n_samples)
        for k in range(horizon):
            u_now = u_t[:, k : k + 1]
            hist = torch.cat([u_now, prev_hist], dim=1)[:, :n_lags]
            z = model.step(z, hist)
            infected = z @ w_i + b_i
            susceptible = z @ w_s + b_s
            infected_error = (infected - infected_target) / infected_scale
            # Penalize driving susceptibles below a floor once infection is low.
            susceptible_shortfall = torch.relu(susceptible_floor - susceptible) / susceptible_scale
            effort = u_now[:, 0] ** 2
            slew = (u_now[:, 0] - previous_u) ** 2
            cost += (
                cfg.infected_weight * infected_error**2
                + cfg.susceptible_weight * susceptible_shortfall**2
                + cfg.control_cost * effort
                + cfg.slew_cost * slew
            )
            previous_u = u_now[:, 0]
            if n_lags > 1:
                prev_hist = torch.cat([u_now, prev_hist], dim=1)[:, : n_lags - 1]

        elite = torch.topk(cost, n_elite, largest=False).indices.numpy()
        mean = u[elite].mean(axis=0)
        std = u[elite].std(axis=0) + 1e-3
    return mean


@torch.no_grad()
def closed_loop_sir(
    model: JEPAControl,
    readout: dict,
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
    """Run multi-objective latent MPC against the true agentic SIR ABM."""
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
        frame = sim.render()[None]
        z0 = model.encode(torch.from_numpy(frame.astype(np.float32))).cpu().numpy()[0]
        plan = cem_plan_sir(
            model,
            z0,
            readout,
            mpc,
            infected_target=infected_target,
            susceptible_floor=susceptible_floor,
            u_prev=u_prev,
            warm_start=warm,
        )
        u = float(np.clip(plan[0], mpc.u_min, mpc.u_max))
        sim.step(u)
        obs = sim.observables()
        infected.append(obs["infected_count"])
        susceptible.append(obs["susceptible_count"])
        recovered.append(obs["recovered_count"])
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
    }


def sir_baseline_rollouts(
    *,
    steps: int = 160,
    levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    cfg: AgenticSIRConfig | None = None,
    n_agents: int = 500,
    initial_infected: int = 16,
    seed_center_x: float | None = None,
    seed_center_y: float | None = None,
    seed_radius: float = 5.0,
    seed: int = 0,
) -> dict:
    """Open-loop constant-vaccination infection / susceptibility references."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for level in levels:
        sim = AgenticSIRModel(
            cfg=cfg or AgenticSIRConfig(),
            n_agents=n_agents,
            initial_infected=initial_infected,
            seed_center_x=seed_center_x,
            seed_center_y=seed_center_y,
            seed_radius=seed_radius,
            seed=seed,
        )
        infected = [sim.observables()["infected_count"]]
        susceptible = [sim.observables()["susceptible_count"]]
        incidence = [sim.observables()["cumulative_incidence"]]
        for _ in range(steps):
            sim.step(level)
            obs = sim.observables()
            infected.append(obs["infected_count"])
            susceptible.append(obs["susceptible_count"])
            incidence.append(obs["cumulative_incidence"])
        out[f"u={level}"] = {
            "infected": np.asarray(infected, dtype=np.float32),
            "susceptible": np.asarray(susceptible, dtype=np.float32),
            "cumulative_incidence": np.asarray(incidence, dtype=np.float32),
        }
    return out
