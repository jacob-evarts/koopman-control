"""Checks for the spatial agentic SIR case study, generator, and MPC objective."""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import torch

from jepa_control.control import (
    SIRMPCConfig,
    cem_plan_sir,
    closed_loop_sir,
    sir_baseline_rollouts,
)
from jepa_control.data import JEPAWindows, dataset_dims
from koopman_control.data.agentic_sir import (
    INFECTED_CHANNEL,
    NUM_CHANNELS,
    RECOVERED_CHANNEL,
    SUSCEPTIBLE_CHANNEL,
    AgenticSIRConfig,
    AgenticSIRModel,
    rollout,
)
from koopman_control.data.generate_sir import FRAME_SCALE, generate
from koopman_control.paths import sir_dataset_directory, sir_dataset_path


def test_outbreak_is_localized_and_channels_are_occupancy() -> None:
    cfg = AgenticSIRConfig(width=32, height=32)
    model = AgenticSIRModel(
        cfg=cfg,
        n_agents=120,
        initial_infected=10,
        seed_center_x=8,
        seed_center_y=24,
        seed_radius=3,
        seed=1,
    )
    frame = model.render()
    assert frame.shape == (NUM_CHANNELS, 32, 32)
    assert model.num_infected == 10
    assert model.num_susceptible + model.num_infected + model.num_recovered == 120
    obs = model.observables()
    assert abs(obs["infected_centroid_x"] - 8) < 3
    assert abs(obs["infected_centroid_y"] - 24) < 3
    assert set(np.unique(frame)) <= {0.0, 1.0}
    assert frame[INFECTED_CHANNEL].sum() >= 1
    assert frame[SUSCEPTIBLE_CHANNEL].sum() >= 1
    assert frame[RECOVERED_CHANNEL].sum() == 0


def test_vaccination_reduces_incidence() -> None:
    cfg = AgenticSIRConfig(width=32, height=32)

    def run(level: float) -> tuple[float, float]:
        _, _, obs = rollout(
            cfg,
            np.full(120, level, dtype=np.float32),
            n_agents=180,
            initial_infected=12,
            seed_radius=4,
            seed=2,
        )
        return float(obs["infected_count"].max()), float(obs["cumulative_incidence"][-1])

    untreated_peak, untreated_inc = run(0.0)
    treated_peak, treated_inc = run(1.0)
    assert treated_peak < untreated_peak
    assert treated_inc < untreated_inc


def test_default_vaccination_has_graded_response() -> None:
    """The calibrated v2 plant must support a meaningful infection / effort trade-off."""
    cfg = AgenticSIRConfig(vaccine_effectiveness=0.028)

    def run(level: float) -> tuple[float, float, float]:
        _, _, obs = rollout(
            cfg,
            np.full(200, level, dtype=np.float32),
            n_agents=800,
            initial_infected=32,
            seed_radius=5,
            seed=950,
        )
        return (
            float(obs["infected_count"].max()),
            float(obs["infected_count"][-1]),
            float(obs["cumulative_incidence"][-1]),
        )

    peak0, final0, inc0 = run(0.0)
    peak_mid, final_mid, inc_mid = run(0.5)
    peak1, final1, inc1 = run(1.0)
    assert peak0 > peak_mid > peak1
    assert inc0 > inc_mid > inc1
    assert inc1 <= 0.45 * inc0
    assert final0 > final_mid or peak0 > 2 * peak_mid


def test_sir_rollout_shapes() -> None:
    frames, controls, obs = rollout(
        AgenticSIRConfig(width=16, height=16),
        np.linspace(0, 1, 12, dtype=np.float32),
        n_agents=40,
        initial_infected=4,
        seed=3,
    )
    assert frames.shape == (13, NUM_CHANNELS, 16, 16)
    assert controls.shape == (13,)
    assert obs["infected_count"].shape == (13,)
    assert {"susceptible_count", "recovered_count", "cumulative_incidence"} <= set(obs)


def test_sir_rollout_is_reproducible() -> None:
    cfg = AgenticSIRConfig(width=16, height=16)
    control = np.linspace(0, 1, 10, dtype=np.float32)
    kwargs = dict(
        n_agents=50,
        initial_infected=5,
        seed_center_x=7,
        seed_center_y=9,
        seed=8,
    )
    first = rollout(cfg, control, **kwargs)
    second = rollout(cfg, control, **kwargs)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    for key in first[2]:
        np.testing.assert_array_equal(first[2][key], second[2][key])


def test_sir_dataset_path_environment_override(monkeypatch, tmp_path) -> None:
    root = tmp_path / "data"
    exact = tmp_path / "custom" / "sir.h5"
    monkeypatch.setenv("KOOPMAN_DATA_ROOT", str(root))
    assert sir_dataset_directory() == root / "agentic_sir_images_v2"
    assert sir_dataset_path() == root / "agentic_sir_images_v2" / "dataset.h5"
    monkeypatch.setenv("KOOPMAN_SIR_DATASET", str(exact))
    assert sir_dataset_path() == exact


def test_generate_sir_writes_occupancy_h5(tmp_path) -> None:
    out = tmp_path / "sir_ds"
    manifest = generate(
        out,
        steps=8,
        seeds=3,
        n_agents=(40,),
        initial_infected=(4,),
        seed_centers=((0.5, 0.5),),
        excitations=("zero", "constant"),
        cfg=AgenticSIRConfig(width=16, height=16),
    )
    path = out / "dataset.h5"
    assert path.exists() and (out / "manifest.json").exists()
    assert manifest["control_target"] == "vaccination"
    assert manifest["channel_names"] == ["susceptible", "infected", "recovered"]
    with h5py.File(path, "r") as f:
        assert float(f.attrs["frame_scale"]) == FRAME_SCALE
        assert f.attrs["abm"] == "agentic_sir"
    assert dataset_dims(path)[:2] == (3, 16)
    ds = JEPAWindows(path, "train", horizon=4)
    frames, _, _ = ds[0]
    assert frames.dtype == torch.float32
    assert set(np.unique(frames.numpy())) <= {0.0, 1.0}


class _ToySIRModel:
    """Two-state latent model used to isolate the CEM objective."""

    hparams = SimpleNamespace(n_control_lags=2)

    def step(self, z: torch.Tensor, u_hist: torch.Tensor) -> torch.Tensor:
        u = u_hist[:, :1]
        return z + torch.cat([-20.0 * u, -8.0 * u], dim=1)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        susceptible = frames[:, 0].sum(dim=(1, 2))
        infected = frames[:, 1].sum(dim=(1, 2))
        return torch.stack([infected, susceptible], dim=1)


def _identity_readout() -> dict:
    return {
        "W": np.eye(2, dtype=np.float32),
        "b": np.zeros(2, dtype=np.float32),
        "names": ["infected_count", "susceptible_count"],
    }


def test_sir_cem_penalty_reduces_dose() -> None:
    model = _ToySIRModel()
    z0 = np.array([40.0, 400.0], dtype=np.float32)
    np.random.seed(0)
    cheap = cem_plan_sir(
        model,
        z0,
        _identity_readout(),
        SIRMPCConfig(
            plan_horizon=5,
            n_samples=512,
            n_iters=5,
            control_cost=0.0,
            infected_scale=40,
            susceptible_scale=400,
            susceptible_weight=0.0,
        ),
        susceptible_floor=100.0,
    )
    np.random.seed(0)
    expensive = cem_plan_sir(
        model,
        z0,
        _identity_readout(),
        SIRMPCConfig(
            plan_horizon=5,
            n_samples=512,
            n_iters=5,
            control_cost=5.0,
            infected_scale=40,
            susceptible_scale=400,
            susceptible_weight=0.0,
        ),
        susceptible_floor=100.0,
    )
    assert cheap.shape == expensive.shape == (5,)
    assert float(cheap.mean()) > float(expensive.mean())


def test_closed_loop_sir_and_baselines_smoke() -> None:
    cfg = AgenticSIRConfig(width=16, height=16)
    loop = closed_loop_sir(
        _ToySIRModel(),
        _identity_readout(),
        infected_target=0.0,
        steps=3,
        mpc=SIRMPCConfig(plan_horizon=3, n_samples=32, n_iters=2, control_cost=0.05),
        cfg=cfg,
        n_agents=40,
        initial_infected=4,
        seed=4,
    )
    assert loop["infected"].shape == loop["susceptible"].shape == loop["control"].shape == (4,)
    assert loop["abm"] == "agentic_sir"
    assert np.isfinite(loop["cumulative_dose"])
    base = sir_baseline_rollouts(
        steps=3,
        levels=(0.0, 1.0),
        cfg=cfg,
        n_agents=40,
        initial_infected=4,
        seed=4,
    )
    assert set(base) == {"u=0.0", "u=1.0"}
    assert base["u=0.0"]["infected"].shape == (4,)
    assert (
        base["u=1.0"]["cumulative_incidence"][-1]
        <= base["u=0.0"]["cumulative_incidence"][-1]
    )


def test_sir_ode_fit_and_closed_loop(tmp_path) -> None:
    from jepa_control.evaluate import load_split_trajectories
    from jepa_control.ode_baseline import (
        closed_loop_sir_ode,
        fit_sir_ode,
        sir_ode_prediction_skill,
        sir_ode_step,
    )

    out = tmp_path / "sir_ode_ds"
    generate(
        out,
        steps=12,
        seeds=3,
        n_agents=(40,),
        initial_infected=(4,),
        seed_centers=((0.5, 0.5),),
        excitations=("zero", "constant", "rpwc"),
        cfg=AgenticSIRConfig(width=16, height=16),
    )
    train = load_split_trajectories(out / "dataset.h5", "train")
    ode = fit_sir_ode(train)
    assert np.isfinite(ode.infected_one_step_r2)
    assert np.isfinite(ode.susceptible_one_step_r2)
    assert np.isfinite(ode.recovered_one_step_r2)
    assert ode.beta >= 0.0 and ode.gamma >= 0.0
    assert ode.vacc_now >= 0.0 and ode.vacc_lag >= 0.0

    s1, i1, r1 = sir_ode_step(180.0, 12.0, 8.0, 0.5, 0.0, ode)
    assert s1 >= 0.0 and i1 >= 0.0 and r1 >= 0.0

    skill = sir_ode_prediction_skill(train, ode, horizon=3)
    assert "skill" in skill

    cfg = AgenticSIRConfig(width=16, height=16)
    loop = closed_loop_sir_ode(
        ode,
        infected_target=2.0,
        steps=3,
        cfg=cfg,
        n_agents=40,
        initial_infected=4,
        mpc=SIRMPCConfig(plan_horizon=3, n_samples=16, n_iters=2),
        seed=4,
    )
    assert loop["infected"].shape == loop["susceptible"].shape == loop["control"].shape == (4,)
    assert loop["planner"] == "sir_ode"
    assert np.isfinite(loop["infected_rmse"])
    assert np.isfinite(loop["susceptible_shortfall_rmse"])
