"""Matched JEPA and ODE MPC controllers for the Strobl spatial tumour ABM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from jepa_control.model import JEPAControl
from koopman_control.data.strobl_ode import StroblODEParameters
from koopman_control.data.strobl_simulator import (
    StroblLauncherConfig,
    StroblSimulator,
)


@dataclass(frozen=True)
class StroblMPCConfig:
    """Shared finite-horizon objective and CEM search settings."""

    plan_horizon: int = 20
    n_samples: int = 256
    n_iters: int = 4
    elite_frac: float = 0.1
    init_std: float = 0.3
    tumour_weight: float = 1.0
    resistant_weight: float = 0.0
    control_cost: float = 0.05
    slew_cost: float = 0.02
    terminal_weight: float = 1.0
    u_min: float = 0.0
    u_max: float = 1.0
    ode_substeps: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.plan_horizon <= 0 or self.n_samples < 3 or self.n_iters <= 0:
            raise ValueError(
                "MPC horizon and iterations must be positive; n_samples must be >= 3"
            )
        if not 0 < self.elite_frac <= 1:
            raise ValueError("elite_frac must lie in (0, 1]")
        if self.u_min < 0 or self.u_max <= self.u_min:
            raise ValueError("MPC dose bounds are invalid")
        if self.ode_substeps <= 0:
            raise ValueError("ode_substeps must be positive")


@dataclass(frozen=True)
class StroblPlantConfig:
    """One reproducible controlled-ABM initial condition and parameter set."""

    architecture: str = "resistant_edge"
    sensitive: int = 4_900
    resistant: int = 100
    width: int = 100
    height: int = 100
    r_s: float = 0.027
    r_r: float = 0.027
    delta_t: float = 0.0
    d_d: float = 0.75
    d_max: float = 1.0
    dt: float = 1.0
    seed: int = 17
    ic_seed: int = 23

    @property
    def carrying_capacity(self) -> float:
        return float(self.width * self.height)

    @property
    def initial_counts(self) -> tuple[int, int, int]:
        return self.sensitive, self.resistant, self.sensitive + self.resistant

    def ode_parameters(self) -> StroblODEParameters:
        return StroblODEParameters(
            r_s=self.r_s,
            r_r=self.r_r,
            delta_t=self.delta_t,
            d_d=self.d_d,
            d_max=self.d_max,
            carrying_capacity=self.carrying_capacity,
        )


def _initial_distribution(
    cfg: StroblMPCConfig, warm_start: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    if warm_start is None:
        mean = np.full(cfg.plan_horizon, 0.5 * (cfg.u_min + cfg.u_max))
    else:
        mean = np.asarray(warm_start, dtype=np.float64)
        if mean.shape != (cfg.plan_horizon,):
            raise ValueError("warm_start must have shape (plan_horizon,)")
    return mean, np.full(cfg.plan_horizon, cfg.init_std, dtype=np.float64)


def _candidate_actions(
    mean: np.ndarray,
    std: np.ndarray,
    cfg: StroblMPCConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    actions = np.clip(
        mean[None] + std[None] * rng.standard_normal((cfg.n_samples, cfg.plan_horizon)),
        cfg.u_min,
        cfg.u_max,
    )
    # Always retain three deterministic anchors so CEM cannot miss obvious
    # no-treatment, full-treatment, or current-mean solutions.
    actions[0] = cfg.u_min
    actions[1] = cfg.u_max
    actions[2] = np.clip(mean, cfg.u_min, cfg.u_max)
    return actions.astype(np.float32)


def _update_distribution(
    actions: np.ndarray, costs: np.ndarray, cfg: StroblMPCConfig
) -> tuple[np.ndarray, np.ndarray]:
    n_elite = max(1, int(round(cfg.elite_frac * cfg.n_samples)))
    elite = np.argpartition(costs, n_elite - 1)[:n_elite]
    return actions[elite].mean(0), actions[elite].std(0) + 1e-3


@torch.no_grad()
def cem_plan_strobl_jepa(
    model: JEPAControl,
    z0: np.ndarray,
    readout: dict,
    cfg: StroblMPCConfig,
    *,
    carrying_capacity: float,
    u_prev: float = 0.0,
    warm_start: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Plan scalar doses through JEPA using total burden as the primary target."""
    names = list(readout["names"])
    total_idx = names.index("total_count")
    resistant_idx = (
        names.index("resistant_count") if "resistant_count" in names else None
    )
    w_total = torch.as_tensor(readout["W"][:, total_idx], dtype=torch.float32)
    b_total = float(readout["b"][total_idx])
    w_resistant = (
        torch.as_tensor(readout["W"][:, resistant_idx], dtype=torch.float32)
        if resistant_idx is not None
        else None
    )
    b_resistant = (
        float(readout["b"][resistant_idx]) if resistant_idx is not None else 0.0
    )
    n_lags = int(model.hparams.n_control_lags)
    capacity = max(float(carrying_capacity), 1.0)
    rng = rng or np.random.default_rng(cfg.seed)
    mean, std = _initial_distribution(cfg, warm_start)
    z_start = torch.as_tensor(z0, dtype=torch.float32).reshape(1, -1)
    z_start = z_start.repeat(cfg.n_samples, 1)

    for _ in range(cfg.n_iters):
        actions = _candidate_actions(mean, std, cfg, rng)
        action_tensor = torch.from_numpy(actions)
        z = z_start.clone()
        history = torch.full((cfg.n_samples, max(0, n_lags - 1)), float(u_prev))
        previous = torch.full((cfg.n_samples,), float(u_prev))
        costs = torch.zeros(cfg.n_samples)
        for step in range(cfg.plan_horizon):
            current = action_tensor[:, step : step + 1]
            features = torch.cat((current, history), dim=1)[:, :n_lags]
            z = model.step(z, features)
            total = torch.clamp(z @ w_total + b_total, 0.0, 2.0 * capacity)
            stage = cfg.tumour_weight * total / capacity
            if cfg.resistant_weight and w_resistant is not None:
                resistant = torch.clamp(
                    z @ w_resistant + b_resistant, 0.0, 2.0 * capacity
                )
                stage = stage + cfg.resistant_weight * resistant / capacity
            stage = (
                stage
                + cfg.control_cost * current[:, 0].square()
                + cfg.slew_cost * (current[:, 0] - previous).square()
            )
            costs += stage
            previous = current[:, 0]
            if n_lags > 1:
                history = torch.cat((current, history), dim=1)[:, : n_lags - 1]
        costs += (
            cfg.terminal_weight
            * torch.clamp(z @ w_total + b_total, 0.0, 2.0 * capacity)
            / capacity
        )
        mean, std = _update_distribution(actions, costs.numpy(), cfg)
    return np.clip(mean, cfg.u_min, cfg.u_max).astype(np.float32)


def _ode_rhs_batch(
    state: np.ndarray,
    dose: np.ndarray,
    parameters: StroblODEParameters,
) -> np.ndarray:
    sensitive, resistant = state[:, 0], state[:, 1]
    crowding = 1.0 - (sensitive + resistant) / parameters.carrying_capacity
    treatment = (
        np.ones_like(dose)
        if parameters.d_max == 0
        else 1.0 - 2.0 * parameters.d_d * dose / parameters.d_max
    )
    ds = parameters.r_s * crowding * treatment * sensitive
    dr = parameters.r_r * crowding * resistant
    ds -= parameters.delta_t * sensitive
    dr -= parameters.delta_t * resistant
    return np.column_stack((ds, dr))


def _ode_step_batch(
    state: np.ndarray,
    dose: np.ndarray,
    parameters: StroblODEParameters,
    *,
    dt: float,
    substeps: int,
) -> np.ndarray:
    step = float(dt) / substeps
    result = state
    for _ in range(substeps):
        k1 = _ode_rhs_batch(result, dose, parameters)
        k2 = _ode_rhs_batch(result + 0.5 * step * k1, dose, parameters)
        k3 = _ode_rhs_batch(result + 0.5 * step * k2, dose, parameters)
        k4 = _ode_rhs_batch(result + step * k3, dose, parameters)
        result = result + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        result = np.maximum(result, 0.0)
    return result


def cem_plan_strobl_ode(
    current_counts: np.ndarray,
    parameters: StroblODEParameters,
    cfg: StroblMPCConfig,
    *,
    dt: float = 1.0,
    u_prev: float = 0.0,
    warm_start: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Plan with the matched aggregate ODE, initialized from observed ABM counts."""
    counts = np.asarray(current_counts, dtype=np.float64)
    if counts.shape not in {(2,), (3,)}:
        raise ValueError("current_counts must contain S,R or S,R,N")
    rng = rng or np.random.default_rng(cfg.seed)
    mean, std = _initial_distribution(cfg, warm_start)
    capacity = parameters.carrying_capacity

    for _ in range(cfg.n_iters):
        actions = _candidate_actions(mean, std, cfg, rng)
        state = np.repeat(counts[None, :2], cfg.n_samples, axis=0)
        previous = np.full(cfg.n_samples, float(u_prev))
        costs = np.zeros(cfg.n_samples)
        for step in range(cfg.plan_horizon):
            current = actions[:, step]
            state = _ode_step_batch(
                state,
                current,
                parameters,
                dt=dt,
                substeps=cfg.ode_substeps,
            )
            total = np.clip(state.sum(1), 0.0, 2.0 * capacity)
            stage = cfg.tumour_weight * total / capacity
            if cfg.resistant_weight:
                stage += cfg.resistant_weight * state[:, 1] / capacity
            costs += (
                stage
                + cfg.control_cost * current**2
                + cfg.slew_cost * (current - previous) ** 2
            )
            previous = current
        costs += (
            cfg.terminal_weight * np.clip(state.sum(1), 0.0, 2.0 * capacity) / capacity
        )
        mean, std = _update_distribution(actions, costs, cfg)
    return np.clip(mean, cfg.u_min, cfg.u_max).astype(np.float32)


def _launcher(plant: StroblPlantConfig) -> StroblLauncherConfig:
    model_args = (
        "--width",
        str(plant.width),
        "--height",
        str(plant.height),
        "--dt",
        f"{plant.dt:.17g}",
        "--division-sensitive",
        f"{plant.r_s:.17g}",
        "--division-resistant",
        f"{plant.r_r:.17g}",
        "--death-sensitive",
        f"{plant.delta_t:.17g}",
        "--death-resistant",
        f"{plant.delta_t:.17g}",
        "--drug-kill",
        f"{plant.d_d:.17g}",
    )
    return StroblLauncherConfig(model_args=model_args)


def _encode_grid(model: JEPAControl, grid: np.ndarray) -> np.ndarray:
    frame = np.moveaxis(np.eye(3, dtype=np.float32)[grid], -1, 0)[None]
    with torch.no_grad():
        return model.encode(torch.from_numpy(frame)).cpu().numpy()[0]


def _shift_plan(plan: np.ndarray, applied: int) -> np.ndarray:
    applied = min(max(int(applied), 1), len(plan))
    return np.concatenate((plan[applied:], np.repeat(plan[-1], applied)))


def run_strobl_closed_loop(
    controller: Literal["jepa", "ode", "constant", "paper_adaptive"],
    *,
    plant: StroblPlantConfig,
    steps: int,
    mpc: StroblMPCConfig | None = None,
    model: JEPAControl | None = None,
    readout: dict | None = None,
    constant_dose: float = 0.0,
    replan_interval: int = 1,
) -> dict:
    """Apply one controller to the true Java ABM and return canonical trajectories."""
    if steps <= 0 or replan_interval <= 0:
        raise ValueError("steps and replan_interval must be positive")
    if controller == "jepa" and (model is None or readout is None):
        raise ValueError("JEPA control requires model and readout")
    mpc = mpc or StroblMPCConfig()
    if mpc.u_max > plant.d_max:
        raise ValueError("MPC upper dose bound exceeds the plant d_max")
    rng = np.random.default_rng(mpc.seed)
    warm: np.ndarray | None = None
    pending: list[float] = []
    previous = 0.0
    initial_total = plant.sensitive + plant.resistant
    adaptive_on = True

    grids: list[np.ndarray] = []
    counts: list[list[int]] = []
    actions: list[float] = []
    with StroblSimulator(_launcher(plant), d_max=plant.d_max) as simulator:
        state = simulator.reset(
            family=plant.architecture,
            sensitive=plant.sensitive,
            resistant=plant.resistant,
            simulation_seed=plant.seed,
            ic_seed=plant.ic_seed,
        )
        grids.append(state.grid.copy())
        counts.append(
            [state.sensitive, state.resistant, state.sensitive + state.resistant]
        )
        for step in range(steps):
            if controller in {"jepa", "ode"} and not pending:
                if controller == "jepa":
                    assert model is not None and readout is not None
                    plan = cem_plan_strobl_jepa(
                        model,
                        _encode_grid(model, state.grid),
                        readout,
                        mpc,
                        carrying_capacity=plant.carrying_capacity,
                        u_prev=previous,
                        warm_start=warm,
                        rng=rng,
                    )
                else:
                    plan = cem_plan_strobl_ode(
                        np.asarray(counts[-1]),
                        plant.ode_parameters(),
                        mpc,
                        dt=plant.dt,
                        u_prev=previous,
                        warm_start=warm,
                        rng=rng,
                    )
                n_apply = min(replan_interval, len(plan), steps - step)
                pending = plan[:n_apply].astype(float).tolist()
                warm = _shift_plan(plan, n_apply)
            if controller in {"jepa", "ode"}:
                dose = pending.pop(0)
            elif controller == "constant":
                dose = constant_dose
            else:
                total = counts[-1][2]
                if total > initial_total:
                    adaptive_on = True
                elif total < 0.5 * initial_total:
                    adaptive_on = False
                dose = plant.d_max if adaptive_on else 0.0
            dose = float(np.clip(dose, 0.0, plant.d_max))
            state = simulator.step(dose)
            actions.append(dose)
            grids.append(state.grid.copy())
            counts.append(
                [state.sensitive, state.resistant, state.sensitive + state.resistant]
            )
            previous = dose

    count_array = np.asarray(counts, dtype=np.int64)
    action_array = np.asarray(actions, dtype=np.float32)
    stage_cost = (
        mpc.tumour_weight * count_array[1:, 2] / plant.carrying_capacity
        + mpc.resistant_weight * count_array[1:, 1] / plant.carrying_capacity
        + mpc.control_cost * action_array**2
    )
    return {
        "controller": controller,
        "grid": np.stack(grids).astype(np.uint8),
        "counts": count_array,
        "action": action_array,
        "stage_cost": stage_cost.astype(np.float32),
        "total_cost": float(stage_cost.sum()),
        "cumulative_dose": float(action_array.sum()),
        "mean_dose": float(action_array.mean()),
        "final_counts": count_array[-1].copy(),
        "plant": plant,
    }


def compare_strobl_controllers(
    model: JEPAControl,
    readout: dict,
    *,
    plant: StroblPlantConfig | None = None,
    steps: int = 160,
    mpc: StroblMPCConfig | None = None,
    replan_interval: int = 5,
    include_baselines: bool = True,
) -> dict[str, dict]:
    """Run JEPA and ODE MPC on matched ABMs, optionally with standard baselines."""
    plant = plant or StroblPlantConfig()
    mpc = mpc or StroblMPCConfig()
    results = {
        "jepa_mpc": run_strobl_closed_loop(
            "jepa",
            plant=plant,
            steps=steps,
            mpc=mpc,
            model=model,
            readout=readout,
            replan_interval=replan_interval,
        ),
        "ode_mpc": run_strobl_closed_loop(
            "ode",
            plant=plant,
            steps=steps,
            mpc=mpc,
            replan_interval=replan_interval,
        ),
    }
    if include_baselines:
        results["no_treatment"] = run_strobl_closed_loop(
            "constant",
            plant=plant,
            steps=steps,
            mpc=mpc,
            constant_dose=0.0,
        )
        results["maximum_tolerated_dose"] = run_strobl_closed_loop(
            "constant",
            plant=plant,
            steps=steps,
            mpc=mpc,
            constant_dose=plant.d_max,
        )
        results["paper_adaptive"] = run_strobl_closed_loop(
            "paper_adaptive",
            plant=plant,
            steps=steps,
            mpc=mpc,
        )
    return results


__all__ = [
    "StroblMPCConfig",
    "StroblPlantConfig",
    "cem_plan_strobl_jepa",
    "cem_plan_strobl_ode",
    "compare_strobl_controllers",
    "run_strobl_closed_loop",
]
