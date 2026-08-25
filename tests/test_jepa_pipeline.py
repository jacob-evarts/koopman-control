"""End-to-end checks for the JEPA latent-control pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from jepa_control.control import MPCConfig, baseline_rollouts, cem_plan, closed_loop
from jepa_control.data import JEPAWindows, dataset_dims, get_dataloaders
from jepa_control.evaluate import (
    encode_all,
    fit_latent_linear,
    fit_readout,
    horizon_errors,
    latent_pca,
    readout_predict,
)
from jepa_control.model import JEPAControl
from jepa_control.train import TrainConfig, train_run
from koopman_control.data.generate import generate
from koopman_control.data.rabbit_grass import RabbitGrassConfig


def _tiny_dataset_for_plots(tmp_path):
    """Slightly longer runs so rollout/PCA figures have something to draw."""
    output = tmp_path / "plot_dataset"
    generate(
        output,
        steps=24,
        seeds=3,
        initial_rabbits=(12,),
        initial_grass_prob=(0.3,),
        excitations=("zero", "constant", "rpwc"),
        cfg=RabbitGrassConfig(width=16, height=16),
    )
    return output / "dataset.h5"


def _tiny_dataset(tmp_path):
    output = tmp_path / "dataset"
    generate(
        output,
        steps=12,
        seeds=3,
        initial_rabbits=(12,),
        initial_grass_prob=(0.3,),
        excitations=("zero", "constant"),
        cfg=RabbitGrassConfig(width=16, height=16),
    )
    return output / "dataset.h5"


def test_model_forward_and_step_shapes() -> None:
    model = JEPAControl(input_size=16, base_channels=8, proj_hidden=32, latent_dim=6, n_obs=6)
    frames = torch.rand(4, 2, 16, 16)
    z = model.encode(frames)
    z_next = model.step(z, torch.rand(4, 2))
    assert z.shape == (4, 6)
    assert z_next.shape == z.shape
    assert torch.isfinite(z_next).all()


def test_linear_predictor_is_identity_at_init_and_exposes_dynamics() -> None:
    model = JEPAControl(input_size=16, base_channels=8, proj_hidden=32, latent_dim=6, n_obs=6)
    assert model.is_linear
    z = torch.randn(4, 6)
    # A = I and B = 0 at init, so the first step is a no-op for any control.
    torch.testing.assert_close(model.step(z, torch.rand(4, 2)), z)

    a, b, c = model.dynamics_matrices()
    assert (a.shape, b.shape, c.shape) == ((6, 6), (6, 2), (6,))
    assert model.spectral_radius() == pytest.approx(1.0)


def test_residual_mlp_ablation_selected_by_name() -> None:
    model = JEPAControl(
        input_size=16,
        base_channels=8,
        proj_hidden=32,
        latent_dim=6,
        n_obs=6,
        predictor="residual_mlp",
        predictor_hidden=16,
    )
    assert not model.is_linear
    assert model.spectral_radius() is None
    with pytest.raises(TypeError):
        model.dynamics_matrices()
    assert model.step(torch.randn(4, 6), torch.rand(4, 2)).shape == (4, 6)

    with pytest.raises(ValueError):
        JEPAControl(input_size=16, latent_dim=6, predictor="quadratic")


def test_probe_subset_is_stratified_across_excitations(tmp_path) -> None:
    """Guards the bug where a truncated probe set held a single excitation."""
    from jepa_control.evaluate import load_split_trajectories

    dataset = _tiny_dataset(tmp_path)
    excitations = {t["excitation"] for t in load_split_trajectories(dataset, "train")}
    assert len(excitations) > 1

    subset = load_split_trajectories(dataset, "train", max_runs=len(excitations))
    assert {t["excitation"] for t in subset} == excitations


def test_shared_step_runs_with_and_without_readout(tmp_path) -> None:
    dataset = _tiny_dataset(tmp_path)
    _, input_size, n_obs, _ = dataset_dims(dataset)
    windows = JEPAWindows(dataset, "train", horizon=3)
    frames, controls, obs = windows[0]
    assert frames.shape == (4, 2, 16, 16)
    assert controls.shape == (4,)
    assert obs.shape[0] == 4

    batch = (frames[None], controls[None], obs[None])
    for w_readout in (0.0, 1.0):
        model = JEPAControl(
            input_size=input_size,
            base_channels=8,
            proj_hidden=32,
            latent_dim=6,
            n_obs=n_obs,
            w_readout=w_readout,
        )
        loss = model._shared_step(batch, "train")
        assert torch.isfinite(loss)


def test_vicreg_penalizes_collapse() -> None:
    collapsed = torch.zeros(64, 8)
    spread = torch.randn(64, 8)
    var_c, _ = JEPAControl._vicreg(collapsed)
    var_s, _ = JEPAControl._vicreg(spread)
    assert var_c > var_s  # a collapsed batch incurs a larger variance hinge


def test_ema_target_encoder_updates_and_is_excluded_from_grads() -> None:
    model = JEPAControl(
        input_size=16,
        base_channels=8,
        proj_hidden=32,
        latent_dim=6,
        n_obs=6,
        target="ema",
        ema_decay=0.5,
    )
    assert model.target_encoder is not None
    assert all(not p.requires_grad for p in model.target_encoder.parameters())

    before = [p.detach().clone() for p in model.target_encoder.parameters()]
    # Move the online encoder away from the teacher, then EMA-update.
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(0.1)
    model._update_ema_target()
    after = list(model.target_encoder.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))

    # Stop-grad mode has no teacher module.
    sg = JEPAControl(input_size=16, base_channels=8, proj_hidden=32, latent_dim=6, target="stopgrad")
    assert sg.target_encoder is None


def test_fit_readout_excludes_centroids_by_default(tmp_path) -> None:
    from jepa_control.evaluate import EXCLUDE_FROM_READOUT, encode_all, fit_readout, load_split_trajectories

    dataset = _tiny_dataset(tmp_path)
    trajs = load_split_trajectories(dataset, "train")
    # Synthetic encodings so we do not need a trained checkpoint.
    t = trajs[0]["frames"].shape[0]
    d = 4
    enc = {
        "z": np.random.randn(len(trajs), t, d).astype(np.float32),
        "obs": np.stack([tr["obs"][:t] for tr in trajs]),
        "obs_names": trajs[0]["obs_names"],
        "controls": np.stack([tr["controls"][:t] for tr in trajs]),
    }
    present = [n for n in EXCLUDE_FROM_READOUT if n in enc["obs_names"]]
    assert present, "tiny rabbit dataset should include rabbit centroids"

    readout = fit_readout(enc)
    assert all(n not in readout["names"] for n in present)
    assert "rabbit_count" in readout["names"]

    full = fit_readout(enc, exclude=())
    assert set(present) <= set(full["names"])


def test_two_phase_freezes_encoder_and_ls_inits_predictor(tmp_path) -> None:
    dataset = _tiny_dataset(tmp_path)
    config = TrainConfig(
        dataset=dataset,
        horizon=3,
        batch_size=8,
        latent_dim=6,
        base_channels=8,
        proj_hidden=32,
        max_epochs=2,
        accelerator="cpu",
        freeze_encoder_after_epoch=1,
        ls_init_predictor=True,
        ls_init_max_batches=2,
        early_stopping_patience=5,
    )
    result = train_run(config, tmp_path / "two_phase", evaluate_test=False, enable_progress_bar=False)
    assert result.best_checkpoint.exists()

    # last.ckpt is always after phase 2 when freeze_after_epoch < max_epochs.
    last = result.run_dir / "checkpoints" / "last.ckpt"
    model = JEPAControl.load_from_checkpoint(last, map_location="cpu")
    assert model.encoder_frozen
    assert bool(model.hparams.encoder_frozen)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert float(model.hparams.w_vic_var) == 0.0
    assert float(model.hparams.w_vic_cov) == 0.0
    a, b, _c = model.dynamics_matrices()
    # LS init / phase-2 training should move A off exact identity or grow B.
    assert np.linalg.norm(a - np.eye(a.shape[0])) > 1e-4 or np.linalg.norm(b) > 1e-4


def test_two_phase_cli_defaults() -> None:
    from jepa_control.train import config_from_args, parse_args

    args = parse_args(["--two-phase", "--phase1-epochs", "3", "--dataset", "unused.h5"])
    cfg = config_from_args(args)
    assert cfg.freeze_encoder_after_epoch == 3
    assert cfg.ls_init_predictor is True

    args = parse_args(
        [
            "--two-phase",
            "--no-ls-init-predictor",
            "--freeze-encoder-after-epoch",
            "0",
            "--dataset",
            "unused.h5",
        ]
    )
    cfg = config_from_args(args)
    assert cfg.freeze_encoder_after_epoch == 0
    assert cfg.ls_init_predictor is False


def test_train_eval_readout_and_mpc(tmp_path) -> None:
    dataset = _tiny_dataset(tmp_path)
    config = TrainConfig(
        dataset=dataset,
        horizon=3,
        batch_size=8,
        latent_dim=6,
        base_channels=8,
        proj_hidden=32,
        predictor_hidden=32,
        max_epochs=1,
        accelerator="cpu",
        fast_dev_run=True,
    )
    result = train_run(config, tmp_path / "run", evaluate_test=False, enable_progress_bar=False)
    assert result.best_checkpoint.exists()

    model = JEPAControl(input_size=16, base_channels=8, proj_hidden=32, latent_dim=6, n_obs=6)
    model.eval()

    from jepa_control.evaluate import load_split_trajectories

    trajs = load_split_trajectories(dataset, "train")
    enc = encode_all(model, trajs)
    readout = fit_readout(enc)
    preds = readout_predict(readout, enc["z"])
    assert preds.shape[:-1] == enc["obs"].shape[:-1]
    assert preds.shape[-1] == len(readout["names"])
    assert preds.shape[-1] < enc["obs"].shape[-1]  # centroids dropped by default

    fitted = fit_latent_linear(model, enc)
    herr = horizon_errors(model, enc, fitted)
    assert "skill_full" in herr and "skill_ls_linear" in herr
    assert latent_pca(enc)["participation_ratio"] > 0

    z0 = enc["z"][0, 0]
    plan = cem_plan(
        model,
        z0,
        target=5.0,
        readout=readout,
        obs_name="rabbit_count",
        cfg=MPCConfig(plan_horizon=4, n_samples=32, n_iters=2),
    )
    assert plan.shape == (4,)
    assert np.all((plan >= 0.0) & (plan <= 1.0))

    loop = closed_loop(
        model,
        readout,
        target=5.0,
        steps=3,
        mpc=MPCConfig(plan_horizon=4, n_samples=32, n_iters=2),
        cfg=RabbitGrassConfig(width=16, height=16),
        initial_rabbits=12,
    )
    assert loop["control"].shape == (4,)
    assert np.isfinite(loop["tracking_rmse"])


def test_resource_ode_fit_and_closed_loop(tmp_path) -> None:
    from jepa_control.ode_baseline import (
        closed_loop_ode,
        fit_resource_ode,
        ode_prediction_skill,
        ode_rollout,
        ode_step,
    )

    dataset = _tiny_dataset(tmp_path)
    from jepa_control.evaluate import load_split_trajectories

    train = load_split_trajectories(dataset, "train")
    ode = fit_resource_ode(train)
    assert np.isfinite(ode.rabbit_one_step_r2)
    assert np.isfinite(ode.grass_one_step_r2)

    r1, g1 = ode_step(20.0, 0.4, 0.5, 0.0, ode)
    assert r1 >= 0.0 and 0.0 <= g1 <= 1.0
    rs, gs = ode_rollout(20.0, 0.4, np.zeros(5), ode)
    assert rs.shape == gs.shape == (6,)

    skill = ode_prediction_skill(train, ode, horizon=3)
    assert "skill" in skill

    loop = closed_loop_ode(
        ode,
        target=5.0,
        steps=3,
        mpc=MPCConfig(plan_horizon=3, n_samples=16, n_iters=2),
        cfg=RabbitGrassConfig(width=16, height=16),
        initial_rabbits=12,
    )
    assert loop["control"].shape == (4,)
    assert np.isfinite(loop["tracking_rmse"])


def test_notebook_compute_path_and_every_figure(tmp_path) -> None:
    """Exercise exactly what notebooks/jepa_eval.ipynb calls, headlessly."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jepa_control import evaluate as ev
    from jepa_control import plots as jp

    dataset = _tiny_dataset_for_plots(tmp_path)
    small_cfg = RabbitGrassConfig(width=16, height=16)

    config = TrainConfig(
        dataset=dataset,
        horizon=4,
        batch_size=8,
        latent_dim=6,
        base_channels=8,
        proj_hidden=32,
        predictor_hidden=32,
        max_epochs=2,
        accelerator="cpu",
        num_workers=0,
    )
    result = train_run(config, tmp_path / "nbrun", evaluate_test=False, enable_progress_bar=False)
    metrics = result.run_dir / "logs" / "metrics.csv"
    assert metrics.exists()

    model = ev.load_model(result.best_checkpoint)
    val = ev.load_split_trajectories(dataset, "val")
    probe_train = ev.load_split_trajectories(dataset, "train", 4)

    enc = ev.encode_all(model, val)
    enc_train = ev.encode_all(model, probe_train)
    readout = ev.fit_readout(enc_train)
    r2_val = ev.readout_r2(readout, enc)
    fitted = ev.fit_latent_linear(model, enc)
    herr = ev.horizon_errors(model, enc, fitted)
    pca = ev.latent_pca(enc)
    probe = ev.linear_probe(model, probe_train, val)
    skill = ev.readout_rollout_skill(model, enc, readout)
    dose = ev.dose_response(
        model, readout, u_levels=(0.0, 0.5), seeds=(0,), steps=6, initial_rabbits=12, cfg=small_cfg
    )

    card = ev.scorecard(
        model=model,
        herr=herr,
        probe=probe,
        pca=pca,
        readout_r2_test=r2_val,
        rollout_skill=skill,
        dose=dose,
        fitted=fitted,
        horizon=4,
    )
    assert "rabbit_count readout R^2" in ev.format_scorecard(card)
    assert pca["coords"].shape == enc["z"].shape

    loop = closed_loop(
        model,
        readout,
        target=5.0,
        steps=2,
        mpc=MPCConfig(plan_horizon=3, n_samples=16, n_iters=2),
        cfg=small_cfg,
        initial_rabbits=12,
    )
    base = baseline_rollouts(steps=2, levels=(0.0, 1.0), initial_rabbits=12, cfg=small_cfg)

    from jepa_control.ode_baseline import closed_loop_ode, fit_resource_ode

    ode = fit_resource_ode(probe_train)
    ode_loop = closed_loop_ode(
        ode,
        target=5.0,
        steps=2,
        mpc=MPCConfig(plan_horizon=3, n_samples=16, n_iters=2),
        cfg=small_cfg,
        initial_rabbits=12,
    )

    figures = [
        jp.fig_control_coverage(val),
        jp.fig_learning_curves(metrics),
        jp.fig_horizon_errors(herr),
        jp.fig_latent_pca(pca, stride=2, n_traj_lines=3),
        jp.fig_latent_probe(probe),
        jp.fig_latent_traces(model, val[0], n_dims=4),
        jp.fig_readout_quality(readout, enc, r2_val),
        jp.fig_readout_rollout(model, val, readout, n=2),
        jp.fig_dose_response(dose),
        jp.fig_step_response(model, enc["z"][0, 0], readout, steps=6),
        jp.fig_closed_loop(loop, base),
        jp.fig_controller_compare({"JEPA": loop, "ODE": ode_loop}, base),
    ]
    for fig in figures:
        assert fig is not None
        plt.close(fig)
