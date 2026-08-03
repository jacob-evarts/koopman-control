"""Small end-to-end checks for the retained latent-control pipeline."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from koopman_control.analysis.dmdc import run as run_dmdc
from koopman_control.data.dataset import RabbitGrassWindows
from koopman_control.data.excitation import make_control
from koopman_control.data.generate import ALL_EXCITATIONS, generate
from koopman_control.data.rabbit_grass import RabbitGrassConfig, rollout
from koopman_control.models.world_model import LatentWorldModel
from koopman_control.paths import data_root, dataset_path, output_root, search_root
from koopman_control.search import parse_args as parse_search_args
from koopman_control.search import run_search
from koopman_control.train import TrainConfig


def test_simulator_is_reproducible_and_aligned() -> None:
    control = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    kwargs = {
        "initial_rabbits": 20,
        "initial_grass_prob": 0.3,
        "seed": 7,
    }
    first = rollout(RabbitGrassConfig(width=16, height=16), control, **kwargs)
    second = rollout(RabbitGrassConfig(width=16, height=16), control, **kwargs)

    frames, controls, observables = first
    assert frames.shape == (9, 2, 16, 16)
    assert controls.shape == (9,)
    assert controls[0] == 0.0
    assert np.array_equal(frames, second[0])
    assert np.array_equal(controls[1:], control)
    assert all(values.shape == (9,) for values in observables.values())


def test_hpc_paths_are_controlled_by_environment(monkeypatch, tmp_path) -> None:
    data = tmp_path / "shared-data"
    output = tmp_path / "scratch-results"
    exact_dataset = tmp_path / "datasets" / "custom.h5"
    monkeypatch.setenv("KOOPMAN_DATA_ROOT", str(data))
    monkeypatch.setenv("KOOPMAN_OUTPUT_ROOT", str(output))

    assert data_root() == data
    assert output_root() == output
    assert dataset_path() == data / "rabbit_grass_images" / "dataset.h5"
    assert search_root() == output / "search"
    assert TrainConfig().dataset == data / "rabbit_grass_images" / "dataset.h5"

    monkeypatch.setenv("KOOPMAN_DATASET", str(exact_dataset))
    assert dataset_path() == exact_dataset
    assert TrainConfig().dataset == exact_dataset


@pytest.mark.parametrize("name", ALL_EXCITATIONS)
def test_excitation_signals_stay_in_actuator_range(name: str) -> None:
    signal = make_control(name, 32, np.random.default_rng(4))
    assert signal.shape == (32,)
    assert np.isfinite(signal).all()
    assert (signal >= 0.0).all()
    assert (signal <= 1.0).all()


def test_generation_loading_and_dmdc(tmp_path) -> None:
    output = tmp_path / "dataset"
    generate(
        output,
        steps=6,
        seeds=3,
        initial_rabbits=(12,),
        initial_grass_prob=(0.3,),
        excitations=("zero",),
        cfg=RabbitGrassConfig(width=16, height=16),
    )
    dataset = output / "dataset.h5"

    for split in ("train", "val", "test"):
        windows = RabbitGrassWindows(dataset, split, horizon=2)
        frames, controls = windows[0]
        assert frames.shape == (3, 2, 16, 16)
        assert controls.shape == (3,)

    report = run_dmdc(dataset)
    assert set(report["one_step_r2"]) == {"train", "val", "test"}
    assert report["state_dim"] == 6
    assert report["control_dim"] == 2


def test_world_model_encode_decode_and_controlled_step() -> None:
    model = LatentWorldModel(
        input_size=16,
        hidden_size=32,
        spatial_latent_channels=8,
        latent_dim=6,
        dynamics_mode="linear",
    )
    frames = torch.rand(4, 2, 16, 16)
    controls = torch.rand(4, 2)

    latent = model.encode(frames)
    reconstruction = model.decode(latent)
    next_latent = model.step(latent, controls)

    assert latent.shape == (4, 6)
    assert reconstruction.shape == frames.shape
    assert next_latent.shape == latent.shape
    assert torch.isfinite(next_latent).all()


def test_one_trial_search_writes_best_checkpoint_and_report(tmp_path) -> None:
    output = tmp_path / "dataset"
    generate(
        output,
        steps=18,
        seeds=3,
        initial_rabbits=(12,),
        initial_grass_prob=(0.3,),
        excitations=("zero", "constant"),
        cfg=RabbitGrassConfig(width=16, height=16),
    )
    study_dir = tmp_path / "study"
    args = parse_search_args(
        [
            "--dataset",
            str(output / "dataset.h5"),
            "--study-dir",
            str(study_dir),
            "--trials",
            "1",
            "--startup-trials",
            "1",
            "--pruning-warmup-epochs",
            "0",
            "--final-seeds",
            "0",
            "--max-epochs",
            "1",
            "--horizon",
            "2",
            "--accelerator",
            "cpu",
            "--fast-dev-run",
        ]
    )

    study = run_search(args)

    assert study.best_trial.number == 0
    assert (study_dir / "study.db").exists()
    assert (study_dir / "trials.csv").exists()
    assert (study_dir / "trials" / "trial_0000" / "provenance.json").exists()
    assert (study_dir / "best_model.ckpt").exists()
    assert (study_dir / "best_config.json").exists()
    assert (study_dir / "best_model.json").exists()
    assert (study_dir / "report" / "study_overview.png").exists()
    assert (study_dir / "report" / "summary.md").exists()

    selection = json.loads((study_dir / "best_model.json").read_text())
    assert selection["final_runs"][0]["best_step"] >= 0
    assert selection["final_runs"][0]["stop_reason"] == "fast_dev_run"
