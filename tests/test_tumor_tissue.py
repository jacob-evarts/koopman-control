"""Checks for the spatial tumor case study, generator, and MPC objective."""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import torch

from jepa_control.control import (
    TumorMPCConfig,
    cem_plan_tumor,
    closed_loop_tumor,
    tumor_baseline_rollouts,
)
from jepa_control.data import JEPAWindows, dataset_dims
from koopman_control.data.generate_tumor import FRAME_SCALE, generate
from koopman_control.data.tumor_tissue import (
    DRUG_CHANNEL,
    NUM_CHANNELS,
    NUTRIENT_CHANNEL,
    TumorTissueConfig,
    TumorTissueModel,
    rollout,
)
from koopman_control.paths import tumor_dataset_directory, tumor_dataset_path


def test_tumor_is_localized_and_fields_are_continuous() -> None:
    cfg = TumorTissueConfig(width=32, height=32, vessel_spacing=8)
    model = TumorTissueModel(
        cfg=cfg,
        initial_healthy_frac=0.94,
        initial_tumor_radius=3,
        tumor_center_x=10,
        tumor_center_y=20,
        seed=1,
    )
    frame = model.render()
    assert frame.shape == (NUM_CHANNELS, 32, 32)
    assert 20 < model.num_tumor < 40
    obs = model.observables()
    assert abs(obs["tumor_centroid_x"] - 10) < 1
    assert abs(obs["tumor_centroid_y"] - 20) < 1
    assert 0 < frame[NUTRIENT_CHANNEL].std()
    assert frame[DRUG_CHANNEL].max() == 0

    for _ in range(10):
        model.step(1.0)
    assert model.drug.max() > 0
    assert model.drug.std() > 0


def test_chemotherapy_reduces_tumor_and_healthy_tissue() -> None:
    cfg = TumorTissueConfig(width=32, height=32, vessel_spacing=8)

    def run(level: float) -> tuple[float, float]:
        _, _, obs = rollout(
            cfg,
            np.full(120, level, dtype=np.float32),
            initial_healthy_frac=0.94,
            initial_tumor_radius=3,
            seed=2,
        )
        return float(obs["tumor_count"][-1]), float(obs["healthy_count"][-1])

    untreated_tumor, untreated_healthy = run(0.0)
    treated_tumor, treated_healthy = run(1.0)
    assert treated_tumor < untreated_tumor
    assert treated_healthy < untreated_healthy


def test_default_treatment_has_reachable_graded_target() -> None:
    """The default plant must support a meaningful tumor/toxicity trade-off."""
    cfg = TumorTissueConfig()

    def run(level: float) -> tuple[float, float, float]:
        _, _, obs = rollout(
            cfg,
            np.full(80, level, dtype=np.float32),
            initial_healthy_frac=0.94,
            initial_tumor_radius=6,
            seed=950,
        )
        return (
            float(obs["tumor_count"][0]),
            float(obs["tumor_count"][-1]),
            float(obs["healthy_count"][-1] / obs["healthy_count"][0]),
        )

    initial, untreated, _ = run(0.0)
    _, medium, _ = run(0.5)
    _, maximum, healthy_ratio = run(1.0)
    assert untreated > initial
    assert maximum < medium < untreated
    assert maximum <= 0.5 * initial
    assert 0.65 <= healthy_ratio <= 0.9


def test_tumor_rollout_shapes() -> None:
    frames, controls, obs = rollout(
        TumorTissueConfig(width=16, height=16, vessel_spacing=4),
        np.linspace(0, 1, 12, dtype=np.float32),
        initial_healthy_frac=0.9,
        initial_tumor_radius=2,
        seed=3,
    )
    assert frames.shape == (13, NUM_CHANNELS, 16, 16)
    assert controls.shape == (13,)
    assert obs["tumor_count"].shape == (13,)
    assert {"healthy_count", "mean_nutrient", "mean_drug"} <= set(obs)


def test_tumor_rollout_is_reproducible() -> None:
    cfg = TumorTissueConfig(width=16, height=16, vessel_spacing=4)
    control = np.linspace(0, 1, 10, dtype=np.float32)
    kwargs = dict(
        initial_healthy_frac=0.9,
        initial_tumor_radius=2,
        tumor_center_x=7,
        tumor_center_y=9,
        seed=8,
    )
    first = rollout(cfg, control, **kwargs)
    second = rollout(cfg, control, **kwargs)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    for key in first[2]:
        np.testing.assert_array_equal(first[2][key], second[2][key])


def test_tumor_dataset_path_environment_override(monkeypatch, tmp_path) -> None:
    root = tmp_path / "data"
    exact = tmp_path / "custom" / "tumor.h5"
    monkeypatch.setenv("KOOPMAN_DATA_ROOT", str(root))
    assert tumor_dataset_directory() == root / "tumor_tissue_images"
    assert tumor_dataset_path() == root / "tumor_tissue_images" / "dataset.h5"
    monkeypatch.setenv("KOOPMAN_TUMOR_DATASET", str(exact))
    assert tumor_dataset_path() == exact


def test_generate_tumor_writes_scaled_h5(tmp_path) -> None:
    out = tmp_path / "tumor_ds"
    manifest = generate(
        out,
        steps=8,
        seeds=3,
        initial_healthy_frac=(0.94,),
        initial_tumor_radius=(2.0,),
        tumor_centers=((0.5, 0.5),),
        excitations=("zero", "constant"),
        cfg=TumorTissueConfig(width=16, height=16, vessel_spacing=4),
    )
    path = out / "dataset.h5"
    assert path.exists() and (out / "manifest.json").exists()
    assert manifest["control_target"] == "chemotherapy"
    assert manifest["channel_names"] == ["healthy", "tumor", "nutrient", "drug"]
    with h5py.File(path, "r") as f:
        assert float(f.attrs["frame_scale"]) == FRAME_SCALE
        assert f.attrs["abm"] == "tumor_tissue"
    assert dataset_dims(path)[:2] == (4, 16)
    ds = JEPAWindows(path, "train", horizon=4)
    frames, _, _ = ds[0]
    assert frames.dtype == torch.float32
    assert 0.0 <= float(frames.min()) <= float(frames.max()) <= 1.0
    assert float(frames[:, NUTRIENT_CHANNEL].std()) > 0


class _ToyTumorModel:
    """Two-state latent model used to isolate the CEM objective."""

    hparams = SimpleNamespace(n_control_lags=2)

    def step(self, z: torch.Tensor, u_hist: torch.Tensor) -> torch.Tensor:
        u = u_hist[:, :1]
        return z + torch.cat([-12.0 * u, -3.0 * u], dim=1)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        healthy = frames[:, 0].sum(dim=(1, 2))
        tumor = frames[:, 1].sum(dim=(1, 2))
        return torch.stack([tumor, healthy], dim=1)


def _identity_readout() -> dict:
    return {
        "W": np.eye(2, dtype=np.float32),
        "b": np.zeros(2, dtype=np.float32),
        "names": ["tumor_count", "healthy_count"],
    }


def test_tumor_cem_penalty_reduces_dose() -> None:
    model = _ToyTumorModel()
    z0 = np.array([100.0, 1000.0], dtype=np.float32)
    np.random.seed(0)
    cheap = cem_plan_tumor(
        model,
        z0,
        _identity_readout(),
        TumorMPCConfig(
            plan_horizon=5,
            n_samples=512,
            n_iters=5,
            control_cost=0.0,
            tumor_scale=100,
            healthy_scale=1000,
        ),
        healthy_reference=1000,
    )
    np.random.seed(0)
    expensive = cem_plan_tumor(
        model,
        z0,
        _identity_readout(),
        TumorMPCConfig(
            plan_horizon=5,
            n_samples=512,
            n_iters=5,
            control_cost=10.0,
            tumor_scale=100,
            healthy_scale=1000,
        ),
        healthy_reference=1000,
    )
    assert cheap.shape == expensive.shape == (5,)
    assert cheap.mean() > expensive.mean()


def test_tumor_ode_fit_and_closed_loop(tmp_path) -> None:
    from jepa_control.evaluate import load_split_trajectories
    from jepa_control.ode_baseline import (
        closed_loop_tumor_ode,
        fit_tumor_ode,
        tumor_ode_prediction_skill,
        tumor_ode_step,
    )

    out = tmp_path / "tumor_ode_ds"
    generate(
        out,
        steps=12,
        seeds=3,
        initial_healthy_frac=(0.94,),
        initial_tumor_radius=(2.0,),
        tumor_centers=((0.5, 0.5),),
        excitations=("zero", "constant", "rpwc"),
        cfg=TumorTissueConfig(width=16, height=16, vessel_spacing=4),
    )
    train = load_split_trajectories(out / "dataset.h5", "train")
    ode = fit_tumor_ode(train)
    assert np.isfinite(ode.tumor_one_step_r2)
    assert np.isfinite(ode.healthy_one_step_r2)
    assert np.isfinite(ode.drug_one_step_r2)

    t1, h1, n1, d1 = tumor_ode_step(30.0, 200.0, 0.6, 0.1, 0.5, 0.0, ode)
    assert t1 >= 0.0 and h1 >= 0.0
    assert 0.0 <= n1 <= 1.0 and 0.0 <= d1 <= 1.0

    skill = tumor_ode_prediction_skill(train, ode, horizon=3)
    assert "skill" in skill

    cfg = TumorTissueConfig(width=16, height=16, vessel_spacing=4)
    loop = closed_loop_tumor_ode(
        ode,
        tumor_target=10.0,
        steps=3,
        cfg=cfg,
        initial_healthy_frac=0.9,
        initial_tumor_radius=2,
        mpc=TumorMPCConfig(plan_horizon=3, n_samples=16, n_iters=2),
        seed=4,
    )
    assert loop["tumor"].shape == loop["healthy"].shape == loop["control"].shape == (4,)
    assert loop["planner"] == "tumor_ode"
    assert np.isfinite(loop["tumor_rmse"])
    assert np.isfinite(loop["healthy_shortfall_rmse"])


def test_tumor_closed_loop_and_baselines_smoke() -> None:
    cfg = TumorTissueConfig(width=16, height=16, vessel_spacing=4)
    loop = closed_loop_tumor(
        _ToyTumorModel(),
        _identity_readout(),
        steps=3,
        cfg=cfg,
        initial_healthy_frac=0.9,
        initial_tumor_radius=2,
        mpc=TumorMPCConfig(plan_horizon=3, n_samples=32, n_iters=2),
        seed=4,
    )
    assert loop["tumor"].shape == loop["healthy"].shape == loop["control"].shape == (4,)
    assert np.isfinite(loop["cumulative_dose"])
    base = tumor_baseline_rollouts(
        steps=3,
        levels=(0.0, 1.0),
        cfg=cfg,
        initial_healthy_frac=0.9,
        initial_tumor_radius=2,
        seed=4,
    )
    assert set(base) == {"u=0.0", "u=1.0"}
    assert base["u=0.0"]["tumor"].shape == (4,)
