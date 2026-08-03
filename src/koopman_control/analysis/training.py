"""Static plots and a reproducible report for an Optuna training study."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.trial import FrozenTrial, TrialState

from koopman_control.paths import search_root


def load_study(study_dir: str | Path) -> tuple[optuna.Study, dict[str, Any]]:
    study_dir = Path(study_dir)
    metadata = json.loads((study_dir / "study.json").read_text())
    study = optuna.load_study(
        study_name=metadata["study_name"],
        storage=metadata["storage"],
    )
    return study, metadata


def completed_trials(study: optuna.Study) -> list[FrozenTrial]:
    return [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE and trial.value is not None
    ]


def fig_study_overview(study: optuna.Study, study_dir: str | Path):
    trials = study.trials
    complete = completed_trials(study)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    if complete:
        numbers = np.asarray([trial.number for trial in complete])
        values = np.asarray([trial.value for trial in complete], dtype=float)
        order = np.argsort(numbers)
        running_best = np.minimum.accumulate(values[order])
        ax.scatter(numbers, values, color="C0", alpha=0.65, label="completed trial")
        ax.plot(numbers[order], running_best, color="C3", lw=2, label="best so far")
        ax.scatter(
            [study.best_trial.number],
            [study.best_value],
            marker="*",
            s=180,
            color="gold",
            edgecolor="k",
            label="selected",
            zorder=4,
        )
    ax.set_xlabel("trial")
    ax.set_ylabel("best validation loss")
    ax.set_title("Optimization history")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    counts = Counter(trial.state.name for trial in trials)
    labels = ["COMPLETE", "PRUNED", "FAIL", "RUNNING"]
    colors = ["C2", "C1", "C3", "C0"]
    values = [counts.get(label, 0) for label in labels]
    ax.bar(labels, values, color=colors)
    for index, value in enumerate(values):
        ax.text(index, value + 0.05, str(value), ha="center")
    ax.set_ylabel("trials")
    ax.set_title("Trial outcomes (pruning efficiency)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    durations = []
    objectives = []
    for trial in complete:
        if trial.duration is not None:
            durations.append(trial.duration.total_seconds() / 60.0)
            objectives.append(trial.value)
    if durations:
        ax.scatter(durations, objectives, c=[trial.number for trial in complete], cmap="viridis")
    ax.set_xlabel("trial duration (minutes)")
    ax.set_ylabel("best validation loss")
    ax.set_title("Accuracy vs compute")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    summary_path = Path(study_dir) / "best_model.json"
    ax.axis("off")
    lines = [
        f"study: {study.study_name}",
        f"trials: {len(trials)} ({len(complete)} complete)",
    ]
    if complete:
        lines.extend(
            [
                f"best trial: {study.best_trial.number}",
                f"best val_loss: {study.best_value:.6g}",
                "",
                "best parameters:",
                *[f"  {key}: {value}" for key, value in sorted(study.best_trial.params.items())],
            ]
        )
    if summary_path.exists():
        final = json.loads(summary_path.read_text())
        lines.extend(
            [
                "",
                f"selected seed: {final.get('selected_seed', 'n/a')}",
                f"final runs: {len(final.get('final_runs', []))}",
            ]
        )
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        family="monospace",
        fontsize=10,
    )
    ax.set_title("Selected model")

    fig.suptitle("Hyperparameter-search overview", y=0.995)
    fig.tight_layout()
    return fig


def fig_parameter_importance(study: optuna.Study):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    try:
        importance = optuna.importance.get_param_importances(study)
    except (ValueError, RuntimeError):
        importance = {}
    if importance:
        names = list(reversed(importance))
        values = [importance[name] for name in names]
        ax.barh(names, values, color="C0", alpha=0.85)
        for index, value in enumerate(values):
            ax.text(value + 0.01, index, f"{value:.2f}", va="center")
        ax.set_xlim(0, max(1.0, max(values) * 1.15))
    else:
        ax.text(
            0.5,
            0.5,
            "Need multiple completed, non-identical trials\nfor parameter importance.",
            ha="center",
            va="center",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("importance for validation loss")
    ax.set_title("Which hyperparameters explain trial quality?")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def fig_parameter_effects(study: optuna.Study):
    complete = completed_trials(study)
    parameters = sorted({name for trial in complete for name in trial.params})
    ncols = 3
    nrows = max(1, math.ceil(len(parameters) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3.8 * nrows),
        squeeze=False,
    )
    for ax, name in zip(axes.ravel(), parameters):
        pairs = [(trial.params[name], trial.value) for trial in complete if name in trial.params]
        x_values = [pair[0] for pair in pairs]
        y_values = [pair[1] for pair in pairs]
        numeric = all(isinstance(value, (int, float)) for value in x_values)
        if numeric:
            ax.scatter(x_values, y_values, alpha=0.65, color="C0")
            if name in {"lr", "weight_decay"}:
                ax.set_xscale("log")
        else:
            categories = list(dict.fromkeys(str(value) for value in x_values))
            positions = {value: index for index, value in enumerate(categories)}
            rng = np.random.default_rng(0)
            jitter = rng.normal(0, 0.04, len(x_values))
            ax.scatter(
                [positions[str(value)] + offset for value, offset in zip(x_values, jitter)],
                y_values,
                alpha=0.65,
                color="C0",
            )
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=20)
        ax.set_xlabel(name)
        ax.set_ylabel("best val_loss")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(parameters) :]:
        ax.axis("off")
    fig.suptitle("Marginal hyperparameter effects (correlation, not causation)", y=0.995)
    fig.tight_layout()
    return fig


def _metric_history(path: Path) -> dict[int, dict[str, float]]:
    epochs: dict[int, dict[str, float]] = defaultdict(dict)
    if not path.exists():
        return epochs
    with path.open() as handle:
        for row in csv.DictReader(handle):
            raw_epoch = row.get("epoch")
            if raw_epoch in (None, ""):
                continue
            epoch = int(float(raw_epoch))
            for name, value in row.items():
                if name in {"epoch", "step"} or value in (None, ""):
                    continue
                try:
                    epochs[epoch][name] = float(value)
                except ValueError:
                    continue
    return epochs


def fig_top_learning_curves(
    study: optuna.Study,
    study_dir: str | Path,
    *,
    top_k: int = 5,
):
    complete = sorted(completed_trials(study), key=lambda trial: trial.value)[:top_k]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for rank, trial in enumerate(complete, start=1):
        path = Path(study_dir) / "trials" / f"trial_{trial.number:04d}" / "logs" / "metrics.csv"
        history = _metric_history(path)
        epochs = sorted(history)
        label = f"#{rank} trial {trial.number}"
        axes[0].plot(
            epochs,
            [history[epoch].get("train_loss", np.nan) for epoch in epochs],
            alpha=0.8,
            label=label,
        )
        axes[1].plot(
            epochs,
            [history[epoch].get("val_loss", np.nan) for epoch in epochs],
            alpha=0.8,
            label=label,
        )
        axes[2].plot(
            epochs,
            [history[epoch].get("val_latent", np.nan) for epoch in epochs],
            alpha=0.8,
            label=label,
        )
    for ax, title, ylabel in zip(
        axes,
        ["Training objective", "Validation objective", "Validation latent error"],
        ["train_loss", "val_loss", "val_latent"],
    ):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Learning curves for the validation-selected top trials", y=0.995)
    fig.tight_layout()
    return fig


def _t_critical_95(n: int) -> float:
    """Two-sided 95% Student-t critical value for small repeated-seed samples."""
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        25: 2.060,
        30: 2.042,
    }
    degrees = max(1, n - 1)
    if degrees in table:
        return table[degrees]
    larger = [key for key in table if key >= degrees]
    return table[min(larger)] if larger else 1.96


def final_seed_statistics(study_dir: str | Path) -> dict[str, Any]:
    path = Path(study_dir) / "best_model.json"
    if not path.exists():
        return {}
    summary = json.loads(path.read_text())
    runs = summary.get("final_runs", [])
    metrics = sorted(
        {
            name
            for run in runs
            for name in run.get("test_metrics", {})
            if isinstance(run.get("test_metrics", {}).get(name), (int, float))
        }
    )
    stats: dict[str, Any] = {"n": len(runs), "metrics": {}}
    for name in metrics:
        values = np.asarray(
            [run["test_metrics"][name] for run in runs if name in run["test_metrics"]],
            dtype=float,
        )
        if not len(values):
            continue
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        stats["metrics"][name] = {
            "mean": float(values.mean()),
            "std": std,
            "ci95_half_width": float(_t_critical_95(len(values)) * std / math.sqrt(len(values))),
            "values": values.tolist(),
        }
    return stats


def fig_final_seed_results(study_dir: str | Path):
    summary = json.loads((Path(study_dir) / "best_model.json").read_text())
    runs = summary.get("final_runs", [])
    seeds = [
        Path(run.get("run_dir", f"run_{index}")).name.replace("seed_", "")
        for index, run in enumerate(runs)
    ]
    val_loss = [run.get("best_val_loss", np.nan) for run in runs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(seeds, val_loss, color="C0", alpha=0.85)
    axes[0].set_xlabel("final run seed")
    axes[0].set_ylabel("best validation loss")
    axes[0].set_title("Initialization sensitivity after selection")
    axes[0].grid(alpha=0.3, axis="y")

    metric_names = [
        name
        for name in ("test_loss", "test_latent", "test_latent_linear")
        if any(name in run.get("test_metrics", {}) for run in runs)
    ]
    x = np.arange(len(metric_names))
    width = 0.8 / max(1, len(runs))
    for index, (seed, run) in enumerate(zip(seeds, runs)):
        values = [run.get("test_metrics", {}).get(name, np.nan) for name in metric_names]
        axes[1].bar(
            x + (index - (len(runs) - 1) / 2) * width,
            values,
            width=width,
            label=f"seed {seed}",
            alpha=0.8,
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([name.replace("test_", "") for name in metric_names])
    axes[1].set_ylabel("held-out test metric")
    axes[1].set_title("Final test results (never used for search)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def _summary_markdown(
    study: optuna.Study,
    metadata: dict[str, Any],
    seed_stats: dict[str, Any],
) -> str:
    counts = Counter(trial.state.name for trial in study.trials)
    lines = [
        "# Training study report",
        "",
        f"- Study: `{study.study_name}`",
        f"- Objective: {metadata['objective']}",
        f"- Completed / pruned / failed: {counts['COMPLETE']} / {counts['PRUNED']} / {counts['FAIL']}",
        f"- Best trial: {study.best_trial.number}",
        f"- Best validation loss: {study.best_value:.6g}",
        "",
        "## Best hyperparameters",
        "",
    ]
    for name, value in sorted(study.best_params.items()):
        lines.append(f"- `{name}`: {value}")
    lines.extend(
        [
            "",
            "## Final held-out test results",
            "",
            "The test split was not used for trial selection. Values below summarize only the",
            "post-search final-seed runs (mean ± standard deviation; 95% CI half-width).",
            "",
        ]
    )
    if seed_stats.get("metrics"):
        for name, values in seed_stats["metrics"].items():
            lines.append(
                f"- `{name}`: {values['mean']:.6g} ± {values['std']:.3g} "
                f"(95% CI ±{values['ci95_half_width']:.3g}, n={len(values['values'])})"
            )
    else:
        lines.append("- No final test metrics are available yet.")
    lines.extend(
        [
            "",
            "## Interpretation cautions",
            "",
            "- Hyperparameter importance is associative, not causal.",
            "- The best trial is optimistically biased by selection over many trials.",
            "- Multi-seed final runs quantify initialization variance; they do not quantify",
            "  uncertainty over new simulator parameter regimes.",
            "- Use `notebooks/worldmodel_eval.ipynb` on `best_model.ckpt` for latent/control",
            "  diagnostics that are not captured by validation loss alone.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_report(study_dir: str | Path) -> Path:
    study_dir = Path(study_dir)
    report_dir = study_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    study, metadata = load_study(study_dir)

    figures = {
        "study_overview.png": fig_study_overview(study, study_dir),
        "parameter_importance.png": fig_parameter_importance(study),
        "parameter_effects.png": fig_parameter_effects(study),
        "top_learning_curves.png": fig_top_learning_curves(study, study_dir),
    }
    if (study_dir / "best_model.json").exists():
        figures["final_seed_results.png"] = fig_final_seed_results(study_dir)
    for name, figure in figures.items():
        figure.savefig(report_dir / name, dpi=180, bbox_inches="tight")
        plt.close(figure)

    stats = final_seed_statistics(study_dir)
    (report_dir / "final_seed_statistics.json").write_text(json.dumps(stats, indent=2))
    (report_dir / "summary.md").write_text(_summary_markdown(study, metadata, stats))
    return report_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study_dir",
        type=Path,
        nargs="?",
        default=search_root() / "worldmodel",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    report_dir = generate_report(parse_args(argv).study_dir)
    print(f"Wrote training report to {report_dir}")


if __name__ == "__main__":
    main()
