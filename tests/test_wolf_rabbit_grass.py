"""Smoke checks for the three-species ABM and its generator."""

from __future__ import annotations

import numpy as np

from koopman_control.data.generate_wolves import generate
from koopman_control.data.wolf_rabbit_grass import (
    NUM_CHANNELS,
    WolfRabbitGrassConfig,
    WolfRabbitGrassModel,
    rollout,
)


def test_render_has_three_channels_and_cull_reduces_wolves() -> None:
    cfg = WolfRabbitGrassConfig(width=16, height=16, culling_effectiveness=0.5)
    model = WolfRabbitGrassModel(
        cfg=cfg, initial_rabbits=30, initial_wolves=20, initial_grass_prob=0.4, seed=0
    )
    img = model.render()
    assert img.shape == (NUM_CHANNELS, 16, 16)
    assert model.num_wolves == 20

    for _ in range(5):
        model.step(1.0)
    assert model.num_wolves < 20


def test_rollout_shapes_and_obs_include_wolf_count() -> None:
    cfg = WolfRabbitGrassConfig(width=16, height=16)
    frames, controls, obs = rollout(
        cfg,
        np.linspace(0, 1, 12, dtype=np.float32),
        initial_rabbits=40,
        initial_wolves=8,
        initial_grass_prob=0.3,
        seed=1,
    )
    assert frames.shape == (13, NUM_CHANNELS, 16, 16)
    assert controls.shape == (13,)
    assert "wolf_count" in obs and "rabbit_count" in obs
    assert obs["wolf_count"].shape == (13,)


def test_generate_wolves_writes_h5(tmp_path) -> None:
    out = tmp_path / "wolf_ds"
    manifest = generate(
        out,
        steps=8,
        seeds=3,
        initial_rabbits=(20,),
        initial_wolves=(4,),
        initial_grass_prob=(0.3,),
        excitations=("zero", "constant"),
        cfg=WolfRabbitGrassConfig(width=16, height=16),
        limit=4,
    )
    assert (out / "dataset.h5").exists()
    assert (out / "manifest.json").exists()
    assert manifest["control_target"] == "wolves"
    assert "wolf_count" in manifest["obs_names"]
