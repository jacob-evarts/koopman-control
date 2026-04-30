import csv
import json
from pathlib import Path
from typing import Any

from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)


class Writer:
    """Handles run directories and writing artifacts (loss CSVs, checkpoints, study summary, plots)."""

    def __init__(self, run_dir: Path, run_id: str):
        self.run_id = run_id
        self.results_dir = run_dir / f"results_{run_id}"
        self.figures_dir = self.results_dir / "figures"
        self.losses_dir = self.results_dir / "losses"
        self.checkpoints_dir = self.results_dir / "checkpoints"

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.losses_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def loss_csv_path(self, trial_id: int) -> Path:
        return self.losses_dir / f"trial_{trial_id:03d}_loss.csv"

    def optuna_fig_path(self, name: str) -> Path:
        return self.figures_dir / f"{name}.png"

    def trial_fig_path(self, trial_id: int, name: str) -> Path:
        return self.figures_dir / f"trial_{trial_id:03d}_{name}.png"

    def save_loss(self, trial_id: int, epoch: int, train_loss: float, val_loss: float) -> None:
        path = self.loss_csv_path(trial_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists()
        with open(path, mode="a", newline="") as csvfile:
            fieldnames = ["epoch", "train_loss", "val_loss"]
            _writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                _writer.writeheader()
            _writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    def remove_bested_checkpoints(self, best_trial_id: int) -> None:
        for checkpoint_file in self.checkpoints_dir.glob("trial-*.ckpt"):
            if f"trial-{best_trial_id:03d}" not in checkpoint_file.name:
                checkpoint_file.unlink()

    def save_study_summary(self, summary: dict[str, Any]) -> None:
        """Write study summary to JSON. Caller provides a dict (no Optuna/cfg dependency here)."""
        summary_path = self.results_dir / "study_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

    def save_optuna_plots(self, study) -> None:
        """Generate and save Optuna visualization figures to figures_dir."""
        fig1 = plot_optimization_history(study)
        fig1.write_image(self.optuna_fig_path("optimization_history"))

        fig2 = plot_param_importances(study)
        fig2.write_image(self.optuna_fig_path("param_importances"))

        fig3 = plot_parallel_coordinate(study)
        fig3.write_image(self.optuna_fig_path("parallel_coordinate"))
