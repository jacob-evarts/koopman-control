import csv
import json
from pathlib import Path

from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate
        )


class Writer:
    def __init__(self, run_dir: Path, run_id: str):
        self.run_id = run_id
        self.results_dir = run_dir / f"results_{run_id}"
        self.figures_dir = self.results_dir / "figures"
        self.losses_dir = self.results_dir / "losses"
        self.checkpoints_dir = self.results_dir / "checkpoints"

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.losses_dir.mkdir(parents=True, exist_ok=True)

    def loss_csv_path(self, trial_id: int):
        return self.losses_dir / f"trial_{trial_id:03d}_loss.csv"

    def optuna_fig_path(self, name: str):
        return self.figures_dir / f"{name}.png"

    def trial_fig_path(self, trial_id: int, name: str):
        return self.figures_dir / f"trial_{trial_id:03d}_{name}.png"

    def save_loss(self, trial_id: int, epoch: int, train_loss: float, val_loss: float):
        path = self.loss_csv_path(trial_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists()
        with open(path, mode='a', newline='') as csvfile:
            fieldnames = ['epoch', 'train_loss', 'val_loss']
            _writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                _writer.writeheader()
            _writer.writerow({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})

    def remove_bested_checkpoints(self, best_trial_id: int):
        for checkpoint_file in self.checkpoints_dir.glob("trial-*.ckpt"):
            if f"trial-{best_trial_id:03d}" not in checkpoint_file.name:
                checkpoint_file.unlink()
        
    def save_study_summary(self, study):
        best_trial = study.best_trial
        summary = {
            "best_trial_number": best_trial.number,
            "best_val_loss": best_trial.value,
            "best_params": best_trial.params,
            "all_trials": []
        }
        for trial in study.trials:
            trial_info = {
                "trial_number": trial.number,
                "val_loss": trial.value,
                "params": trial.params
            }
            summary["all_trials"].append(trial_info)
        summary_path = self.results_dir / "study_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

    def save_optuna_plots(self, study):
        fig1 = plot_optimization_history(study)
        fig1.write_image(self.optuna_fig_path("optimization_history"))

        fig2 = plot_param_importances(study)
        fig2.write_image(self.optuna_fig_path("param_importances"))

        fig3 = plot_parallel_coordinate(study)
        fig3.write_image(self.optuna_fig_path("parallel_coordinate"))