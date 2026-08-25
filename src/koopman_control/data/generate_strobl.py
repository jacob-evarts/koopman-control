"""Generate the focused, provenance-tracked Strobl dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from koopman_control.paths import PROJECT_ROOT, strobl_dataset_directory

HORIZONS = (1, 5, 10, 25)
POLICIES = (
    "open_loop",
    "constant",
    "random_piecewise_constant",
    "pulses",
    "paper_adaptive",
)
ARCHITECTURES = (
    "random_mixed",
    "resistant_core",
    "resistant_dispersed",
    "resistant_edge",
    "two_resistant_nests",
)
PARAMETER_NAMES = (
    "r_s",
    "r_r",
    "delta_t",
    "d_d",
    "d_max",
    "dt",
    "resistance_cost",
    "turnover_fraction",
)
POLICY_MIX = {
    "open_loop": 0.20,
    "constant": 0.20,
    "random_piecewise_constant": 0.30,
    "pulses": 0.15,
    "paper_adaptive": 0.15,
}
SCHEMA_VERSION = "strobl-focused-v1"


@dataclass(frozen=True)
class EpisodePlan:
    episode_id: str
    split: str
    subset: str
    architecture: str
    policy: str
    seed: int
    initial_counts: tuple[int, int, int]
    parameters: tuple[float, ...]
    action: np.ndarray
    ic_seed: int
    matched_group: str = ""
    stochastic_replicate: int = 0
    canonical: bool = False
    evaluation_group: str = ""
    dose_level: float = float("nan")


def _seed(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little"
    ) & ((1 << 63) - 1)


def _load_config(profile: str | Path) -> tuple[Path, dict[str, Any]]:
    candidate = Path(profile)
    path = (
        candidate
        if candidate.suffix == ".json" or candidate.parent != Path(".")
        else PROJECT_ROOT / "configs" / f"strobl_{profile}.json"
    )
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = json.loads(path.read_text())
    expected = (
        candidate.stem.removeprefix("strobl_") if candidate.suffix else str(profile)
    )
    if config.get("profile") != expected:
        raise ValueError(f"{path} declares profile={config.get('profile')!r}")
    return path, config


def _quotas(total: int, mix: Mapping[str, float]) -> dict[str, int]:
    if not np.isclose(sum(mix.values()), 1.0):
        raise ValueError("policy_mix must sum to one")
    raw = {name: total * float(mix[name]) for name in POLICIES}
    out = {name: int(np.floor(raw[name])) for name in POLICIES}
    order = sorted(
        POLICIES, key=lambda name: (-(raw[name] - out[name]), POLICIES.index(name))
    )
    for name in order[: total - sum(out.values())]:
        out[name] += 1
    return out


def _labels(
    total: int, mix: Mapping[str, float], rng: np.random.Generator
) -> list[str]:
    result = [name for name, count in _quotas(total, mix).items() for _ in range(count)]
    rng.shuffle(result)
    return result


def _action(
    policy: str,
    length: int,
    rng: np.random.Generator,
    constant_rank: int = 0,
    constant_count: int = 1,
) -> np.ndarray:
    if policy == "open_loop":
        return np.zeros(length, np.float32)
    if policy == "constant":
        level = constant_rank / (constant_count - 1) if constant_count > 1 else 0.5
        return np.full(length, level, np.float32)
    if policy == "random_piecewise_constant":
        result = np.empty(length, np.float32)
        n_segments = min(10, max(1, length))
        cuts = (
            np.sort(rng.choice(np.arange(1, length), n_segments - 1, replace=False))
            if n_segments > 1
            else np.empty(0, int)
        )
        edges = np.r_[0, cuts, length]
        for start, stop, level in zip(
            edges[:-1], edges[1:], rng.uniform(0, 1, n_segments), strict=True
        ):
            result[start:stop] = level
        return result
    if policy == "pulses":
        result = np.zeros(length, np.float32)
        period, width = max(8, length // 8), max(2, length // 40)
        for start in range(int(rng.integers(0, period)), length, period):
            result[start : min(length, start + width)] = rng.uniform(0.5, 1.0)
        return result
    if policy == "paper_adaptive":
        return np.ones(length, np.float32)  # replaced by realized feedback actions
    raise ValueError(f"unknown policy {policy!r}")


def _initial(
    config: Mapping[str, Any], rng: np.random.Generator
) -> tuple[int, int, int]:
    area = int(config["height"]) * int(config["width"])
    density = rng.uniform(*map(float, config["density_range"]))
    resistant_fraction = np.exp(
        rng.uniform(*np.log(np.asarray(config["resistant_fraction_range"], float)))
    )
    total = int(round(area * density))
    resistant = max(1, int(round(total * resistant_fraction)))
    return total - resistant, resistant, total


def _params(
    config: Mapping[str, Any],
    rng: np.random.Generator,
    edge: bool,
    canonical: bool = False,
) -> tuple[float, ...]:
    if canonical:
        sampled = {"r_s": 0.027, "resistance_cost": 0.0, "turnover_fraction": 0.0}
    else:
        sampled = {}
        for index, name in enumerate(("r_s", "resistance_cost", "turnover_fraction")):
            low, high = map(float, config["parameter_ranges"][name])
            sampled[name] = (
                (low if index % 2 == 0 else high) if edge else rng.uniform(low, high)
            )
    r_s = sampled["r_s"]
    return (
        r_s,
        r_s * (1 - sampled["resistance_cost"]),
        r_s * sampled["turnover_fraction"],
        float(config["d_d"]),
        float(config["d_max"]),
        float(config["dt"]),
        sampled["resistance_cost"],
        sampled["turnover_fraction"],
    )


def build_episode_plan(config: Mapping[str, Any]) -> list[EpisodePlan]:
    """Build deterministic standard and matched-state episode plans."""
    n = int(config["standard_episodes"])
    matched_groups = int(config["matched_groups"])
    matched_replicates = int(config["matched_replicates"])
    length, base_seed = int(config["intervals"]), int(config["seed"])
    rng = np.random.default_rng(base_seed)
    policies = _labels(n, POLICY_MIX, rng)
    constants, constant_rank = policies.count("constant"), 0
    n_train, n_val = round(n * 0.70), round(n * 0.15)
    splits = ["train"] * n_train + ["val"] * n_val
    splits += ["test"] * (n - len(splits))
    rng.shuffle(splits)
    choices = {
        "train": ARCHITECTURES[:3],
        "val": ARCHITECTURES[:3],
        "test": ARCHITECTURES[3:],
    }
    architecture_index = {"train": 0, "val": 0, "test": 0}
    plans: list[EpisodePlan] = []
    for index, (policy, split) in enumerate(zip(policies, splits, strict=True)):
        seed, erng = _seed(base_seed, "standard", index), None
        erng = np.random.default_rng(seed)
        schedule = _action(policy, length, erng, constant_rank, constants)
        constant_rank += policy == "constant"
        canonical = index < int(config.get("canonical_standard_episodes", 0))
        architecture = choices[split][architecture_index[split] % len(choices[split])]
        architecture_index[split] += 1
        plans.append(
            EpisodePlan(
                f"standard_{index:05d}",
                split,
                "standard",
                architecture,
                policy,
                seed,
                _initial(config, erng),
                _params(config, erng, split == "test", canonical),
                schedule,
                _seed(seed, "initial-condition"),
                canonical=canonical,
            )
        )
    group_policies = _labels(matched_groups, POLICY_MIX, rng)
    for index, policy in enumerate(group_policies):
        group, seed = f"matched_{index:04d}", _seed(base_seed, "matched", index)
        grng = np.random.default_rng(seed)
        initial, parameters = _initial(config, grng), _params(config, grng, False)
        schedule = _action(policy, length, grng)
        ic_seed = _seed(seed, "shared-occupied-mask")
        for replicate in range(matched_replicates):
            simulation_seed = _seed(seed, "simulation", replicate)
            for architecture in ARCHITECTURES:
                plans.append(
                    EpisodePlan(
                        f"{group}_rep{replicate}_{architecture}",
                        "test",
                        "matched_state",
                        architecture,
                        policy,
                        simulation_seed,
                        initial,
                        parameters,
                        schedule.copy(),
                        ic_seed,
                        group,
                        replicate,
                    )
                )
    controlled = config.get("controlled_evaluation", {})
    controlled_groups = int(controlled.get("groups", 0))
    controlled_replicates = int(controlled.get("replicates", 0))
    controlled_doses = tuple(map(float, controlled.get("doses", (0.0, 0.5, 1.0))))
    controlled_architecture = str(controlled.get("architecture", "random_mixed"))
    if controlled_groups and (
        controlled_replicates <= 0
        or not controlled_doses
        or any(dose < 0 or dose > float(config["d_max"]) for dose in controlled_doses)
    ):
        raise ValueError("controlled_evaluation has invalid replicates or doses")
    for index in range(controlled_groups):
        group = f"controlled_{index:04d}"
        group_seed = _seed(base_seed, "controlled-evaluation", index)
        group_rng = np.random.default_rng(group_seed)
        initial = _initial(config, group_rng)
        parameters = _params(config, group_rng, False)
        ic_seed = _seed(group_seed, "shared-initial-condition")
        for replicate in range(controlled_replicates):
            simulation_seed = _seed(group_seed, "simulation", replicate)
            for dose in controlled_doses:
                dose_label = str(dose).replace(".", "p")
                plans.append(
                    EpisodePlan(
                        f"{group}_rep{replicate}_dose{dose_label}",
                        "test",
                        "controlled_evaluation",
                        controlled_architecture,
                        "constant",
                        simulation_seed,
                        initial,
                        parameters,
                        np.full(length, dose, dtype=np.float32),
                        ic_seed,
                        stochastic_replicate=replicate,
                        evaluation_group=group,
                        dose_level=dose,
                    )
                )
    return plans


def _modules(config: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    names = config.get(
        "backend_modules",
        {
            "simulator": "koopman_control.data.strobl_simulator",
            "policies": "koopman_control.data.strobl_policies",
            "ode": "koopman_control.data.strobl_ode",
        },
    )
    return tuple(
        importlib.import_module(names[k]) for k in ("simulator", "policies", "ode")
    )


def _simulate(
    plan: EpisodePlan,
    config: Mapping[str, Any],
    simulator: Any,
    policies: Any,
    ode: Any,
    action_override: np.ndarray | None,
) -> dict[str, Any]:
    parameters = dict(zip(PARAMETER_NAMES, plan.parameters, strict=True))
    action = np.asarray(
        plan.action if action_override is None else action_override, np.float32
    )
    result = simulator.simulate_episode(
        architecture=plan.architecture,
        actions=action,
        initial_counts=plan.initial_counts,
        parameters=parameters,
        width=int(config["width"]),
        height=int(config["height"]),
        seed=plan.seed,
        ic_seed=plan.ic_seed,
        policy_name=(
            "paper_adaptive"
            if plan.policy == "paper_adaptive" and action_override is None
            else None
        ),
        stop_on_terminal=(
            bool(config.get("stop_on_terminal", True))
            and not plan.matched_group
            and not plan.evaluation_group
        ),
        shared_occupied_mask=bool(plan.matched_group),
    )
    action = np.asarray(result["action"], np.float32)
    counts = np.asarray(result["counts"], np.int64)
    ode_parameters = ode.StroblODEParameters(
        r_s=parameters["r_s"],
        r_r=parameters["r_r"],
        delta_t=parameters["delta_t"],
        d_d=parameters["d_d"],
        d_max=parameters["d_max"],
        carrying_capacity=float(int(config["width"]) * int(config["height"])),
    )
    ode_counts = ode.solve_strobl_ode(
        plan.initial_counts[:2],
        action,
        ode_parameters,
        interval_duration=parameters["dt"],
    ).counts.astype(np.float32)
    area = int(config["height"]) * int(config["width"])
    return {
        "grid": np.asarray(result["grid"], np.uint8),
        "action": action,
        "counts": counts,
        "occupancy": np.asarray(result["occupancy"], np.float32),
        "cost": (
            float(config["q_tumour"]) * counts[1:, 2] / area
            + float(config["q_dose"]) * action**2
        ).astype(np.float32),
        "ode_counts": ode_counts,
        "diagnostics": result["diagnostics"],
        "terminal_reason": result["terminal_reason"],
        "terminal_time": result["terminal_time"],
    }


def _validate(
    plan: EpisodePlan, data: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    t = len(data["action"])
    h, w = int(config["height"]), int(config["width"])
    if t > int(config["intervals"]):
        raise ValueError(f"{plan.episode_id}: episode exceeds configured duration")
    if data["grid"].shape != (t + 1, h, w) or data["grid"].dtype != np.uint8:
        raise ValueError(f"{plan.episode_id}: invalid grid shape or dtype")
    if data["grid"].size and data["grid"].max() > 2:
        raise ValueError(f"{plan.episode_id}: categorical grid labels exceed 2")
    if np.any((data["action"] < 0) | (data["action"] > plan.parameters[4])):
        raise ValueError(f"{plan.episode_id}: action lies outside [0,D_max]")
    derived = np.asarray(
        [((g == 1).sum(), (g == 2).sum(), (g > 0).sum()) for g in data["grid"]]
    )
    if not np.array_equal(derived, data["counts"]):
        raise ValueError(f"{plan.episode_id}: counts disagree with grid")
    shapes = {
        "action": (t,),
        "counts": (t + 1, 3),
        "occupancy": (t + 1,),
        "cost": (t,),
        "ode_counts": (t + 1, 3),
    }
    if any(data[name].shape != shape for name, shape in shapes.items()):
        raise ValueError(f"{plan.episode_id}: episode arrays violate schema")
    if tuple(data["counts"][0]) != plan.initial_counts:
        raise ValueError(f"{plan.episode_id}: initial counts were not preserved")


def _write_episode(
    root: h5py.Group, plan: EpisodePlan, data: Mapping[str, Any]
) -> None:
    group = root.create_group(plan.episode_id)
    group.create_dataset(
        "grid", data=data["grid"], compression="gzip", compression_opts=4
    )
    for name, dtype in (
        ("action", np.float32),
        ("counts", np.int64),
        ("occupancy", np.float32),
        ("cost", np.float32),
        ("ode_counts", np.float32),
    ):
        group.create_dataset(name, data=data[name], dtype=dtype)
    vector = group.create_dataset(
        "parameter_vector", data=np.asarray(plan.parameters, np.float32)
    )
    group["parameters"] = vector
    diagnostics = group.create_group("diagnostics")
    for name, values in data["diagnostics"].items():
        diagnostics.create_dataset(name, data=values)
    group.attrs.update(
        split=plan.split,
        subset=plan.subset,
        architecture=plan.architecture,
        policy=plan.policy,
        excitation=plan.policy,
        simulation_seed=np.uint64(plan.seed),
        ic_seed=np.uint64(plan.ic_seed),
        matched_group=plan.matched_group,
        evaluation_group=plan.evaluation_group,
        dose_level=plan.dose_level,
        stochastic_replicate=plan.stochastic_replicate,
        canonical=plan.canonical,
        initial_counts=np.asarray(plan.initial_counts, np.int64),
        terminal_reason=str(data.get("terminal_reason", "max_time")),
        terminal_time=float(data.get("terminal_time", len(plan.action))),
        schedule_sha256=hashlib.sha256(data["action"].tobytes()).hexdigest(),
        parameter_names=json.dumps(PARAMETER_NAMES),
        parameters_json=json.dumps(
            dict(zip(PARAMETER_NAMES, plan.parameters, strict=True))
        ),
    )


def _write_indices(h5: h5py.File, plans: list[EpisodePlan]) -> None:
    root, strings = h5.create_group("transition_index"), h5py.string_dtype("utf-8")
    offsets: dict[str, int] = {}
    offset = 0
    for plan in plans:
        offsets[plan.episode_id] = offset
        offset += len(h5["episodes"][plan.episode_id]["grid"])
    plan_index = {plan.episode_id: index for index, plan in enumerate(plans)}
    for horizon in HORIZONS:
        hgroup = root.create_group(f"H{horizon}")
        hgroup.attrs["horizon"] = horizon
        for split in ("train", "val", "test"):
            ids: list[str] = []
            episode_indices: list[int] = []
            starts: list[int] = []
            global_starts: list[int] = []
            action_windows: list[np.ndarray] = []
            target_windows: list[np.ndarray] = []
            for plan in plans:
                if plan.split == split:
                    actions = np.asarray(
                        h5["episodes"][plan.episode_id]["action"], dtype=np.float32
                    )
                    for start in range(max(0, len(actions) - horizon + 1)):
                        ids.append(plan.episode_id)
                        episode_indices.append(plan_index[plan.episode_id])
                        starts.append(start)
                        global_starts.append(offsets[plan.episode_id] + start)
                        action_windows.append(actions[start : start + horizon])
                        target_windows.append(
                            offsets[plan.episode_id]
                            + np.arange(start + 1, start + horizon + 1, dtype=np.int64)
                        )
            group, start = hgroup.create_group(split), np.asarray(starts, np.int32)
            group.create_dataset("episode_id", data=np.asarray(ids, dtype=strings))
            group.create_dataset("episode_index", data=episode_indices, dtype=np.int32)
            group.create_dataset("t", data=start)
            group.create_dataset(
                "global_observation_index", data=global_starts, dtype=np.int64
            )
            group.create_dataset(
                "action_window",
                data=np.asarray(action_windows, np.float32).reshape(-1, horizon),
            )
            group.create_dataset(
                "target_observation_index",
                data=np.asarray(target_windows, np.int64).reshape(-1, horizon),
            )
            group.attrs["episodes"] = len(
                {p.episode_id for p in plans if p.split == split}
            )
            group.attrs["transitions"] = len(start)


def generate(profile: str | Path, output_dir: Path | None = None) -> dict[str, Any]:
    config_path, config = _load_config(profile)
    plans, modules = build_episode_plan(config), _modules(config)
    simulator, policies, ode = modules
    profile_name = str(config["profile"])
    output = Path(output_dir or strobl_dataset_directory(profile_name))
    output.mkdir(parents=True, exist_ok=True)
    h5_path = output / "dataset.h5"
    metadata_path = output / "metadata.jsonl"
    started = time.perf_counter()
    raw_grid_bytes = 0
    compressed_grid_bytes = 0
    sources = [Path(__file__), config_path] + [
        Path(m.__file__) for m in modules if getattr(m, "__file__", None)
    ]
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "source_sha256": {
            str(p.resolve()): hashlib.sha256(p.resolve().read_bytes()).hexdigest()
            for p in sources
        },
    }
    rows, matched_actions = [], {}
    with h5py.File(h5_path, "w") as h5:
        episodes = h5.create_group("episodes")
        for plan in plans:
            data = _simulate(
                plan,
                config,
                simulator,
                policies,
                ode,
                matched_actions.get(plan.matched_group),
            )
            _validate(plan, data, config)
            if plan.matched_group:
                matched_actions.setdefault(plan.matched_group, data["action"].copy())
            _write_episode(episodes, plan, data)
            raw_grid_bytes += int(data["grid"].nbytes)
            compressed_grid_bytes += int(
                episodes[plan.episode_id]["grid"].id.get_storage_size()
            )
            rows.append(
                {
                    "episode_id": plan.episode_id,
                    "split": plan.split,
                    "subset": plan.subset,
                    "architecture": plan.architecture,
                    "policy": plan.policy,
                    "simulation_seed": plan.seed,
                    "ic_seed": plan.ic_seed,
                    "matched_group": plan.matched_group or None,
                    "evaluation_group": plan.evaluation_group or None,
                    "dose_level": (
                        None if np.isnan(plan.dose_level) else plan.dose_level
                    ),
                    "stochastic_replicate": plan.stochastic_replicate,
                    "canonical": plan.canonical,
                    "initial_counts": list(plan.initial_counts),
                    "parameters": dict(
                        zip(PARAMETER_NAMES, plan.parameters, strict=True)
                    ),
                    "schedule_sha256": hashlib.sha256(
                        data["action"].tobytes()
                    ).hexdigest(),
                    "final_counts": data["counts"][-1].tolist(),
                    "total_cost": float(data["cost"].sum()),
                    "realized_action": data["action"].tolist(),
                    "width": int(config["width"]),
                    "height": int(config["height"]),
                    "dt": float(config["dt"]),
                    "control_interval": float(config["dt"]),
                    "observation_interval": float(config["dt"]),
                    "terminal_reason": data["terminal_reason"],
                    "terminal_time": float(data["terminal_time"]),
                    "upstream_commit": str(
                        getattr(simulator, "UPSTREAM_COMMIT", "unknown")
                    ),
                    "simulator_version": "strobl-controlled-v1",
                    "no_patient_data": True,
                }
            )
        h5["runs"] = episodes
        _write_indices(h5, plans)
        h5.attrs.update(
            schema_version=SCHEMA_VERSION,
            profile=profile_name,
            abm="strobl",
            categorical_grid=True,
            height=int(config["height"]),
            width=int(config["width"]),
            num_channels=3,
            frame_scale=1.0,
            intervals=int(config["intervals"]),
            upstream_commit=str(getattr(simulator, "UPSTREAM_COMMIT", "unknown")),
            protocol_version=str(getattr(simulator, "PROTOCOL_VERSION", "unknown")),
            channel_names=np.asarray(
                ("empty", "sensitive", "resistant"), dtype=h5py.string_dtype()
            ),
            count_names=np.asarray(
                ("sensitive", "resistant", "total"), dtype=h5py.string_dtype()
            ),
            parameter_names=np.asarray(PARAMETER_NAMES, dtype=h5py.string_dtype()),
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
            action_alignment="action[t] drives grid[t] -> grid[t+1]",
            occupancy_definition="(sensitive + resistant) / (height * width)",
            warning="Synthetic methodological benchmark; no clinical validity.",
            config_json=json.dumps(config, sort_keys=True),
            provenance_json=json.dumps(provenance, sort_keys=True),
        )
    metadata_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    elapsed = time.perf_counter() - started
    disk_bytes = h5_path.stat().st_size + metadata_path.stat().st_size
    summary = {
        "profile": profile_name,
        "dataset": str(h5_path),
        "metadata": str(metadata_path),
        "episodes": len(plans),
        "splits": dict(Counter(p.split for p in plans)),
        "subsets": dict(Counter(p.subset for p in plans)),
        "policies": dict(Counter(p.policy for p in plans if p.subset == "standard")),
        "terminal_reasons": dict(Counter(row["terminal_reason"] for row in rows)),
        "runtime_seconds": elapsed,
        "disk_bytes": disk_bytes,
        "raw_grid_bytes": raw_grid_bytes,
        "compressed_grid_bytes": compressed_grid_bytes,
        "grid_compression_ratio": (
            raw_grid_bytes / compressed_grid_bytes if compressed_grid_bytes else None
        ),
    }
    (output / "qc_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", required=True, help="smoke/pilot/full or profile JSON"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    generate(args.profile, args.output_dir)


if __name__ == "__main__":
    main()
