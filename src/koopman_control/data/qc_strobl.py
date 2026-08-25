"""Validate and visualize a generated Strobl controlled benchmark dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from koopman_control.paths import strobl_dataset_path


def inspect_dataset(dataset: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Run schema/mechanism invariants and write a compact QC report and figure."""
    dataset = Path(dataset)
    output = Path(output_dir or dataset.parent)
    output.mkdir(parents=True, exist_ok=True)
    policy_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    terminal_counts: Counter[str] = Counter()
    matched: dict[str, list[str]] = defaultdict(list)
    controlled: dict[str, list[str]] = defaultdict(list)
    ode_errors: list[float] = []
    adaptive_episodes = 0
    adaptive_cycling = 0
    initial_components: dict[str, list[int]] = defaultdict(list)
    examples: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    split_ids: dict[str, set[str]] = defaultdict(set)

    with h5py.File(dataset, "r") as h5:
        if int(h5.attrs["num_channels"]) != 3:
            raise AssertionError("Strobl grid must have exactly three categories")
        episodes = h5["episodes"]
        for episode_id, group in episodes.items():
            split = str(group.attrs["split"])
            subset = str(group.attrs["subset"])
            policy = str(group.attrs["policy"])
            split_ids[split].add(episode_id)
            split_counts[split] += 1
            terminal_counts[str(group.attrs["terminal_reason"])] += 1
            if subset == "standard":
                policy_counts[policy] += 1
                if policy == "paper_adaptive":
                    adaptive_episodes += 1
            elif subset == "matched_state":
                matched[str(group.attrs["matched_group"])].append(episode_id)
            elif subset == "controlled_evaluation":
                controlled[str(group.attrs["evaluation_group"])].append(episode_id)

            grid = np.asarray(group["grid"], dtype=np.uint8)
            action = np.asarray(group["action"], dtype=np.float32)
            counts = np.asarray(group["counts"], dtype=np.int64)
            ode = np.asarray(group["ode_counts"], dtype=np.float32)
            if grid.ndim != 3 or action.shape != (len(grid) - 1,):
                raise AssertionError(f"{episode_id}: state/action alignment failed")
            if counts.shape != (len(grid), 3) or ode.shape != counts.shape:
                raise AssertionError(f"{episode_id}: count trajectory shape failed")
            if grid.size and int(grid.max()) > 2:
                raise AssertionError(f"{episode_id}: invalid categorical state")
            derived = np.asarray(
                [((g == 1).sum(), (g == 2).sum(), (g > 0).sum()) for g in grid]
            )
            if not np.array_equal(derived, counts):
                raise AssertionError(
                    f"{episode_id}: aggregate counts disagree with grid"
                )
            if np.any((action < 0) | (action > 1)):
                raise AssertionError(f"{episode_id}: scalar dose is outside bounds")
            if subset == "standard" and policy == "paper_adaptive":
                adaptive_cycling += int(np.count_nonzero(np.diff(action)) > 0)
            if "resistant_components" in group["diagnostics"]:
                initial_components[str(group.attrs["architecture"])].append(
                    int(group["diagnostics"]["resistant_components"][0])
                )
            ode_errors.append(float(np.mean(np.abs(counts - ode))))
            examples.setdefault(
                str(group.attrs["architecture"]),
                (grid[0], grid[-1], counts, action),
            )

        if (
            split_ids["train"] & split_ids["val"]
            or split_ids["train"] & split_ids["test"]
            or split_ids["val"] & split_ids["test"]
        ):
            raise AssertionError("episode leakage across splits")

        matched_checks = {}
        for group_name, episode_ids in matched.items():
            groups = [episodes[episode_id] for episode_id in episode_ids]
            masks = [np.asarray(group["grid"][0]) > 0 for group in groups]
            parameters = [np.asarray(group["parameters"]) for group in groups]
            initial_counts = [np.asarray(group["counts"][0]) for group in groups]
            actions = [np.asarray(group["action"]) for group in groups]
            odes = [np.asarray(group["ode_counts"]) for group in groups]
            invariant = (
                all(np.array_equal(masks[0], value) for value in masks)
                and all(np.array_equal(parameters[0], value) for value in parameters)
                and all(
                    np.array_equal(initial_counts[0], value) for value in initial_counts
                )
                and all(np.array_equal(actions[0], value) for value in actions)
                and all(np.array_equal(odes[0], value) for value in odes)
            )
            if not invariant:
                raise AssertionError(f"{group_name}: matched-state invariant failed")
            matched_checks[group_name] = len(episode_ids)

        controlled_checks = {}
        for group_name, episode_ids in controlled.items():
            groups = [episodes[episode_id] for episode_id in episode_ids]
            by_replicate: dict[int, list[h5py.Group]] = defaultdict(list)
            for group in groups:
                by_replicate[int(group.attrs["stochastic_replicate"])].append(group)
            doses = sorted({float(group.attrs["dose_level"]) for group in groups})
            for replicate, replicate_groups in by_replicate.items():
                grids = [np.asarray(group["grid"][0]) for group in replicate_groups]
                parameters = [
                    np.asarray(group["parameters"]) for group in replicate_groups
                ]
                initial_counts = [
                    np.asarray(group["counts"][0]) for group in replicate_groups
                ]
                invariant = (
                    all(np.array_equal(grids[0], value) for value in grids)
                    and all(
                        np.array_equal(parameters[0], value) for value in parameters
                    )
                    and all(
                        np.array_equal(initial_counts[0], value)
                        for value in initial_counts
                    )
                )
                if not invariant:
                    raise AssertionError(
                        f"{group_name}/rep{replicate}: paired-dose invariant failed"
                    )
            controlled_checks[group_name] = {
                "episodes": len(episode_ids),
                "replicates": len(by_replicate),
                "doses": doses,
            }

        transition_counts = {}
        for horizon_name, horizon_group in h5["transition_index"].items():
            horizon = int(horizon_group.attrs["horizon"])
            transition_counts[horizon_name] = {}
            for split, group in horizon_group.items():
                n = len(group["t"])
                if group["action_window"].shape != (n, horizon):
                    raise AssertionError(
                        f"{horizon_name}/{split}: action windows invalid"
                    )
                if group["target_observation_index"].shape != (n, horizon):
                    raise AssertionError(f"{horizon_name}/{split}: targets invalid")
                if not set(group["episode_id"].asstr()[:]) <= split_ids[split]:
                    raise AssertionError(f"{horizon_name}/{split}: split leakage")
                transition_counts[horizon_name][split] = n

    report = {
        "dataset": str(dataset),
        "episodes": sum(split_counts.values()),
        "splits": dict(split_counts),
        "standard_policy_mix": dict(policy_counts),
        "terminal_reasons": dict(terminal_counts),
        "matched_groups": matched_checks,
        "controlled_evaluation_groups": controlled_checks,
        "adaptive_episodes": adaptive_episodes,
        "adaptive_episodes_with_switching": adaptive_cycling,
        "initial_resistant_components": {
            name: sorted(set(values)) for name, values in initial_components.items()
        },
        "transition_counts": transition_counts,
        "mean_abm_ode_absolute_count_difference": float(np.mean(ode_errors)),
        "no_patient_data": True,
        "warning": "Synthetic methodological benchmark; no clinical validity.",
    }
    (output / "qc_report.json").write_text(json.dumps(report, indent=2))
    _plot_examples(examples, output / "qc_examples.png")
    return report


def _plot_examples(
    examples: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    destination: Path,
) -> None:
    names = list(examples)
    figure, axes = plt.subplots(
        len(names), 3, figsize=(12, 3 * len(names)), squeeze=False
    )
    cmap = ListedColormap(("white", "#75c572", "#d81b60"))
    for row, name in enumerate(names):
        initial, final, counts, action = examples[name]
        axes[row, 0].imshow(initial, vmin=0, vmax=2, cmap=cmap)
        axes[row, 0].set_title(f"{name}: initial")
        axes[row, 1].imshow(final, vmin=0, vmax=2, cmap=cmap)
        axes[row, 1].set_title("final")
        axes[row, 2].plot(counts[:, 0], label="sensitive")
        axes[row, 2].plot(counts[:, 1], label="resistant")
        axes[row, 2].step(
            np.arange(len(action)) + 1, action * counts[0, 2], label="dose × N0"
        )
        axes[row, 2].set_title("counts and scaled dose")
        axes[row, 2].legend(fontsize=8)
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=strobl_dataset_path())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(inspect_dataset(args.dataset, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
