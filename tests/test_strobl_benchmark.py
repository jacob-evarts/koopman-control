"""Regression, mechanism, policy, ODE, and schema tests for Strobl benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from jepa_control.data import JEPAWindows, dataset_dims
from jepa_control.model import JEPAControl
from jepa_control.strobl_control import (
    StroblMPCConfig,
    StroblPlantConfig,
    cem_plan_strobl_ode,
    compare_strobl_controllers,
)
from koopman_control.data.generate_strobl import build_episode_plan
from koopman_control.data.strobl_ode import (
    StroblODEParameters,
    solve_strobl_ode,
    strobl_rhs,
)
from koopman_control.data.strobl_policies import (
    constant,
    open_loop,
    paper_adaptive,
    paper_text_adaptive,
    pulses,
    random_piecewise_constant,
)
from koopman_control.data.strobl_simulator import (
    DEFAULT_VENDOR_ROOT,
    StroblLauncherConfig,
    StroblSimulator,
    external_terminal_reason,
    simulate_episode,
)


def test_scalar_policies_are_bounded_reproducible_and_budgeted() -> None:
    assert np.allclose(open_loop([-1, 0.4, 2], d_max=1), [0, 0.4, 1])
    assert np.all(constant(8, 0.3) == np.float32(0.3))
    left = random_piecewise_constant(40, seed=7)
    right = random_piecewise_constant(40, seed=7)
    assert left.shape == (40,)
    assert np.array_equal(left, right)
    assert 3 <= len(np.unique(left)) <= 10
    pulse = pulses(20, period=5, width=2)
    assert pulse.ndim == 1
    assert np.all((0 <= pulse) & (pulse <= 1))
    capped = constant(10, 1.0, cumulative_cap=2.5)
    assert np.isclose(capped.sum(), 2.5)

    with pytest.raises(ValueError, match="one-dimensional"):
        open_loop(np.ones((2, 2)))


def test_adaptive_boundary_discrepancy_is_explicit() -> None:
    source = paper_adaptive(100)
    text = paper_text_adaptive(100)
    source.current_dose = 0.0
    text.current_dose = 0.0
    assert source(100) == 0.0  # released Java source uses strict N > N0
    assert text(100) == 1.0  # paper prose/equation uses N >= N0
    assert source(101) == 1.0


def test_ode_uses_paper_equations_and_resistant_is_not_directly_treated() -> None:
    params = StroblODEParameters(
        r_s=0.027,
        r_r=0.0243,
        delta_t=0.0027,
        d_d=0.75,
        d_max=1.0,
        carrying_capacity=10_000,
    )
    untreated = strobl_rhs(0, [4000, 100], 0.0, params)
    treated = strobl_rhs(0, [4000, 100], 1.0, params)
    assert treated[0] < untreated[0]
    assert treated[1] == untreated[1]


def test_ode_matched_architectures_are_identical() -> None:
    params = StroblODEParameters(carrying_capacity=400)
    actions = np.tile([0.0, 0.5, 1.0, 0.25], 5)
    first = solve_strobl_ode([180, 20, 200], actions, params)
    second = solve_strobl_ode([180, 20, 200], actions, params)
    assert np.array_equal(first.actions, actions.astype(np.float32))
    assert np.array_equal(first.counts, second.counts)
    assert first.counts.shape == (21, 3)
    assert np.allclose(first.counts[:, 2], first.counts[:, :2].sum(axis=1))


def test_strobl_ode_mpc_is_bounded_reproducible_and_cost_sensitive() -> None:
    parameters = StroblODEParameters()
    counts = np.asarray([4_900.0, 100.0, 5_000.0])
    low_effort_penalty = StroblMPCConfig(
        plan_horizon=10,
        n_samples=64,
        n_iters=3,
        control_cost=0.0,
        slew_cost=0.0,
        seed=9,
    )
    high_effort_penalty = StroblMPCConfig(
        plan_horizon=10,
        n_samples=64,
        n_iters=3,
        control_cost=10.0,
        slew_cost=0.0,
        seed=9,
    )
    first = cem_plan_strobl_ode(counts, parameters, low_effort_penalty)
    second = cem_plan_strobl_ode(counts, parameters, low_effort_penalty)
    conservative = cem_plan_strobl_ode(counts, parameters, high_effort_penalty)
    assert first.shape == (10,)
    assert np.array_equal(first, second)
    assert np.all((first >= 0) & (first <= 1))
    assert first.mean() > conservative.mean()


def test_matched_jepa_and_ode_mpc_execute_against_same_java_plant() -> None:
    model = JEPAControl(
        num_channels=3,
        input_size=20,
        base_channels=4,
        proj_hidden=8,
        latent_dim=4,
        predictor="residual_mlp",
        predictor_hidden=8,
        horizon=3,
        n_control_lags=2,
        n_obs=5,
    ).eval()
    readout = {
        "names": ["total_count"],
        "W": np.zeros((4, 1), dtype=np.float32),
        "b": np.asarray([100.0], dtype=np.float32),
    }
    plant = StroblPlantConfig(
        architecture="resistant_edge",
        sensitive=95,
        resistant=5,
        width=20,
        height=20,
        seed=31,
        ic_seed=37,
    )
    mpc = StroblMPCConfig(
        plan_horizon=3,
        n_samples=8,
        n_iters=1,
        seed=41,
    )
    results = compare_strobl_controllers(
        model,
        readout,
        plant=plant,
        steps=3,
        mpc=mpc,
        replan_interval=1,
        include_baselines=False,
    )
    jepa, ode = results["jepa_mpc"], results["ode_mpc"]
    assert np.array_equal(jepa["grid"][0], ode["grid"][0])
    assert np.array_equal(jepa["counts"][0], ode["counts"][0])
    assert jepa["counts"].shape == ode["counts"].shape == (4, 3)
    assert jepa["action"].shape == ode["action"].shape == (3,)
    assert np.all((jepa["action"] >= 0) & (jepa["action"] <= 1))
    assert np.all((ode["action"] >= 0) & (ode["action"] <= 1))


def test_external_progression_and_cure_boundaries() -> None:
    assert external_terminal_reason(initial_total=100, total=0, day=1) == "cure"
    assert external_terminal_reason(initial_total=100, total=121, day=149) is None
    assert external_terminal_reason(initial_total=100, total=120, day=150) is None
    assert (
        external_terminal_reason(initial_total=100, total=121, day=150) == "progression"
    )


@pytest.mark.skipif(
    not (DEFAULT_VENDOR_ROOT / "controlled-model.jar").exists(),
    reason="controlled Java jar has not been built",
)
def test_java_step_api_exact_counts_dose_bounds_and_determinism() -> None:
    launcher = StroblLauncherConfig(
        model_args=("--width", "12", "--height", "10", "--division-sensitive", "0.2")
    )
    trajectories = []
    for _ in range(2):
        with StroblSimulator(launcher) as simulator:
            state = simulator.reset(
                family="resistant_core",
                sensitive=70,
                resistant=10,
                simulation_seed=17,
                ic_seed=19,
            )
            grids = [state.grid.copy()]
            for dose in (0.0, 0.5, 1.0):
                state = simulator.step(dose)
                grids.append(state.grid.copy())
                assert state.counts.tolist() == [
                    int(np.sum(state.grid == 1)),
                    int(np.sum(state.grid == 2)),
                    int(np.sum(state.grid == 0)),
                ]
            trajectories.append(np.stack(grids))
            with pytest.raises(ValueError, match="dose"):
                simulator.step(1.01)
    assert np.array_equal(*trajectories)


@pytest.mark.skipif(
    not (DEFAULT_VENDOR_ROOT / "controlled-model.jar").exists(),
    reason="controlled Java jar has not been built",
)
def test_high_level_rollout_has_interval_alignment_and_mechanistic_diagnostics() -> (
    None
):
    result = simulate_episode(
        architecture="resistant_dispersed",
        actions=np.linspace(0, 1, 12, dtype=np.float32),
        initial_counts=[70, 10, 80],
        parameters={
            "r_s": 0.2,
            "r_r": 0.18,
            "delta_t": 0.01,
            "d_d": 0.75,
            "d_max": 1.0,
            "dt": 1.0,
        },
        width=12,
        height=10,
        seed=31,
        ic_seed=37,
        stop_on_terminal=False,
    )
    assert result["grid"].shape == (13, 10, 12)
    assert result["action"].shape == (12,)
    assert result["counts"].shape == (13, 3)
    assert result["occupancy"].shape == (13,)
    assert result["diagnostics"]["resistant_components"].shape == (13,)
    assert result["diagnostics"]["blocked_resistant"].shape == (12,)
    assert np.all(result["grid"] <= 2)


def _write_categorical_fixture(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs.update(
            num_channels=3,
            width=4,
            height=4,
            frame_scale=1.0,
            obs_names=np.asarray(
                (
                    "sensitive_count",
                    "resistant_count",
                    "total_count",
                    "occupancy_0",
                    "cost_0",
                ),
                dtype=h5py.string_dtype(),
            ),
        )
        episodes = h5.create_group("episodes")
        group = episodes.create_group("episode")
        group.attrs["split"] = "train"
        grid = np.zeros((6, 4, 4), dtype=np.uint8)
        grid[:, :2] = 1
        grid[:, 2, :2] = 2
        group.create_dataset("grid", data=grid)
        group.create_dataset("action", data=np.linspace(0, 1, 5, dtype=np.float32))
        group.create_dataset("counts", data=np.tile(np.asarray([8, 2, 10]), (6, 1)))
        group.create_dataset("occupancy", data=np.full(6, 10 / 16))
        group.create_dataset("cost", data=np.arange(5, dtype=np.float32))


def test_jepa_loader_one_hot_and_interval_action_alignment(tmp_path: Path) -> None:
    path = tmp_path / "categorical.h5"
    _write_categorical_fixture(path)
    dataset = JEPAWindows(path, "train", horizon=3)
    frames, controls, observables = dataset[0]
    assert frames.shape == (4, 3, 4, 4)
    assert np.allclose(frames.numpy().sum(axis=1), 1)
    assert controls.tolist() == [0.0, 0.0, 0.25, 0.5]
    assert observables.shape == (4, 5)
    assert dataset_dims(path) == (
        3,
        4,
        5,
        [
            "sensitive_count",
            "resistant_count",
            "total_count",
            "occupancy_0",
            "cost_0",
        ],
    )


def test_upstream_provenance_records_absent_license_and_pinned_commit() -> None:
    text = (DEFAULT_VENDOR_ROOT / "UPSTREAM.md").read_text()
    readme = (DEFAULT_VENDOR_ROOT / "README.md").read_text()
    assert "aa3b3c2ad2e4acf9fd7cc6ac318f1bf79f9361e2" in text
    assert "No public software license" in text
    assert "42cb0b7cba654cfe2297c47d13285ffdc143a0554ed75b75ad40cc1a48ad3983" in text
    assert "-profilingMode false" in readme
    assert "Java `17.0.1`" in readme
    assert "902c85ace97418c193ffa8bb8033c24797ca3aa2133eac30329b95c347ef129a" in readme


def test_pilot_plan_has_exact_policy_mix_and_matched_groups() -> None:
    config = json.loads(
        (Path(__file__).parents[1] / "configs" / "strobl_pilot.json").read_text()
    )
    plans = build_episode_plan(config)
    standard = [plan for plan in plans if plan.subset == "standard"]
    matched = [plan for plan in plans if plan.subset == "matched_state"]
    assert len(standard) == 120
    assert len(matched) == 50
    assert {
        architecture: sum(
            plan.split == "test" and plan.architecture == architecture
            for plan in standard
        )
        for architecture in ("resistant_edge", "two_resistant_nests")
    } == {"resistant_edge": 9, "two_resistant_nests": 9}
    assert {
        name: sum(plan.policy == name for plan in standard)
        for name in {
            "open_loop",
            "constant",
            "random_piecewise_constant",
            "pulses",
            "paper_adaptive",
        }
    } == {
        "open_loop": 24,
        "constant": 24,
        "random_piecewise_constant": 36,
        "pulses": 18,
        "paper_adaptive": 18,
    }
    for group_name in {plan.matched_group for plan in matched}:
        group = [plan for plan in matched if plan.matched_group == group_name]
        assert len(group) == 10
        assert {plan.architecture for plan in group} == {
            "random_mixed",
            "resistant_core",
            "resistant_edge",
            "resistant_dispersed",
            "two_resistant_nests",
        }
        assert len({plan.stochastic_replicate for plan in group}) == 2
        assert len({plan.initial_counts for plan in group}) == 1
        assert len({plan.parameters for plan in group}) == 1
        assert len({plan.action.tobytes() for plan in group}) == 1


def test_full_plan_has_paired_fixed_dose_evaluation() -> None:
    config = json.loads(
        (Path(__file__).parents[1] / "configs" / "strobl_full.json").read_text()
    )
    plans = build_episode_plan(config)
    controlled = [plan for plan in plans if plan.subset == "controlled_evaluation"]
    assert len(controlled) == 30
    assert len(plans) == 1280
    for group_name in {plan.evaluation_group for plan in controlled}:
        group = [plan for plan in controlled if plan.evaluation_group == group_name]
        assert len(group) == 6
        assert {plan.stochastic_replicate for plan in group} == {0, 1}
        assert {plan.dose_level for plan in group} == {0.0, 0.5, 1.0}
        assert len({plan.initial_counts for plan in group}) == 1
        assert len({plan.parameters for plan in group}) == 1
        assert len({plan.ic_seed for plan in group}) == 1
        for replicate in (0, 1):
            paired = [plan for plan in group if plan.stochastic_replicate == replicate]
            assert len({plan.seed for plan in paired}) == 1


def test_strobl_evaluation_notebook_is_valid_and_targets_final_run() -> None:
    path = Path(__file__).parents[1] / "notebooks" / "jepa_eval_strobl.ipynb"
    notebook = json.loads(path.read_text())
    sources = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    for index, source in enumerate(sources):
        compile(source, f"{path.name}:cell-{index}", "exec")
    text = "\n".join(sources)
    assert "jepa-strobl-full-h20-mlp-valloss-e50" in text
    assert "controlled_evaluation" in text
    assert "horizon_errors" in text
    assert "ode_counts" in text


def test_generated_smoke_schema_indices_and_matched_identity() -> None:
    path = Path(__file__).parents[1] / "data" / "strobl" / "smoke" / "dataset.h5"
    if not path.exists():
        pytest.skip(
            "generate the smoke profile before running dataset integration test"
        )
    with h5py.File(path, "r") as h5:
        assert h5.attrs["upstream_commit"] == (
            "aa3b3c2ad2e4acf9fd7cc6ac318f1bf79f9361e2"
        )
        episodes = h5["episodes"]
        assert len(episodes) == 28
        episode_ids_by_split = {"train": set(), "val": set(), "test": set()}
        matched = []
        controlled = []
        for episode_id, group in episodes.items():
            episode_ids_by_split[group.attrs["split"]].add(episode_id)
            assert group["grid"].dtype == np.uint8
            assert group["action"].dtype == np.float32
            assert group["counts"].shape[0] == group["grid"].shape[0]
            assert group["action"].shape[0] + 1 == group["grid"].shape[0]
            assert np.all((group["action"][:] >= 0) & (group["action"][:] <= 1))
            if group.attrs["subset"] == "matched_state":
                matched.append(group)
            elif group.attrs["subset"] == "controlled_evaluation":
                controlled.append(group)
        assert not (
            episode_ids_by_split["train"] & episode_ids_by_split["val"]
            or episode_ids_by_split["train"] & episode_ids_by_split["test"]
            or episode_ids_by_split["val"] & episode_ids_by_split["test"]
        )
        occupied_masks = [group["grid"][0] > 0 for group in matched]
        assert all(np.array_equal(occupied_masks[0], mask) for mask in occupied_masks)
        assert all(
            np.array_equal(matched[0]["action"][:], group["action"][:])
            for group in matched
        )
        assert all(
            np.array_equal(matched[0]["ode_counts"][:], group["ode_counts"][:])
            for group in matched
        )
        assert len(controlled) == 3
        assert {float(group.attrs["dose_level"]) for group in controlled} == {
            0.0,
            0.5,
            1.0,
        }
        assert all(
            np.array_equal(controlled[0]["grid"][0], group["grid"][0])
            for group in controlled
        )
        assert all(
            np.array_equal(controlled[0]["parameters"][:], group["parameters"][:])
            for group in controlled
        )
        for horizon in (1, 5, 10, 25):
            for split in ("train", "val", "test"):
                index = h5[f"transition_index/H{horizon}/{split}"]
                n = len(index["t"])
                assert index["action_window"].shape == (n, horizon)
                assert index["target_observation_index"].shape == (n, horizon)
                assert (
                    set(index["episode_id"].asstr()[:]) <= episode_ids_by_split[split]
                )

    from jepa_control.evaluate import load_split_trajectories

    controlled_trajectories = load_split_trajectories(
        path,
        "test",
        where={"subset": "controlled_evaluation"},
    )
    assert len(controlled_trajectories) == 3
    assert {trajectory["dose_level"] for trajectory in controlled_trajectories} == {
        0.0,
        0.5,
        1.0,
    }
    assert controlled_trajectories[0]["frames"].shape[1:] == (3, 20, 20)
    assert controlled_trajectories[0]["controls"].shape == (31,)
    assert controlled_trajectories[0]["obs"].shape == (31, 5)
