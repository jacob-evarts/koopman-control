"""Resumable, pruned hyperparameter search for the latent world model.

The search objective is the minimum ``val_loss`` reached during each run. Loss
weights and rollout horizon are held fixed across trials so objective values are
comparable; the search varies architecture and optimization parameters only.

After optimization, the best configuration is optionally retrained across
multiple random seeds. The test split is never touched by search trials and is
evaluated only for these final models.

Example
-------
    poetry run python -m koopman_control.search \
        --dataset data/rabbit_grass_images/dataset.h5 \
        --study-dir outputs/search/linear-h20 \
        --trials 40 --max-epochs 50 --final-seeds 3
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import optuna
import pytorch_lightning as pl
import torch
from optuna.trial import TrialState
from pytorch_lightning.callbacks import Callback

from koopman_control.paths import dataset_path, search_root
from koopman_control.train import TrainConfig, test_checkpoint, train_run


class OptunaPruningCallback(Callback):
    """Report validation loss each epoch and stop statistically weak trials."""

    def __init__(
        self,
        trial: optuna.Trial,
        *,
        monitor: str = "val_loss",
        warmup_epochs: int = 5,
    ) -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.warmup_epochs = warmup_epochs

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        _module: pl.LightningModule,
    ) -> None:
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        value = float(metric)
        epoch = int(trainer.current_epoch)
        self.trial.report(value, step=epoch)
        if epoch >= self.warmup_epochs and self.trial.should_prune():
            self.trial.set_user_attr("pruned_epoch", epoch)
            self.trial.set_user_attr("pruned_value", value)
            raise optuna.TrialPruned(f"Pruned at epoch {epoch}: {self.monitor}={value:.6g}")


def suggest_config(
    trial: optuna.Trial,
    base: TrainConfig,
    *,
    dynamics_modes: tuple[str, ...] = ("linear", "bilinear"),
) -> TrainConfig:
    """Sample one comparable architecture/optimizer configuration."""
    return replace(
        base,
        dynamics_mode=trial.suggest_categorical("dynamics_mode", list(dynamics_modes)),
        latent_dim=trial.suggest_categorical("latent_dim", [4, 8, 16, 32]),
        hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 128]),
        spatial_latent_channels=trial.suggest_categorical(
            "spatial_latent_channels", [8, 16, 32]
        ),
        activation=trial.suggest_categorical("activation", ["relu", "silu"]),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
    )


def _write_trials(study: optuna.Study, path: Path) -> None:
    param_names = sorted({key for trial in study.trials for key in trial.params})
    attr_names = sorted({key for trial in study.trials for key in trial.user_attrs})
    fields = [
        "number",
        "state",
        "value",
        "duration_seconds",
        "datetime_start",
        "datetime_complete",
        *[f"param_{name}" for name in param_names],
        *[f"attr_{name}" for name in attr_names],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trial in study.trials:
            duration = trial.duration.total_seconds() if trial.duration is not None else None
            row: dict[str, Any] = {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "duration_seconds": duration,
                "datetime_start": trial.datetime_start,
                "datetime_complete": trial.datetime_complete,
            }
            row.update({f"param_{key}": value for key, value in trial.params.items()})
            row.update(
                {
                    f"attr_{key}": (json.dumps(value) if isinstance(value, (dict, list)) else value)
                    for key, value in trial.user_attrs.items()
                }
            )
            writer.writerow(row)


def _config_from_serialized(values: dict[str, Any]) -> TrainConfig:
    values = dict(values)
    values["dataset"] = Path(values["dataset"])
    return TrainConfig(**values)


def _copy_best(
    checkpoint: Path,
    config: TrainConfig,
    destination: Path,
    metadata: dict[str, Any],
) -> None:
    shutil.copy2(checkpoint, destination / "best_model.ckpt")
    (destination / "best_config.json").write_text(json.dumps(config.serializable(), indent=2))
    (destination / "best_model.json").write_text(json.dumps(metadata, indent=2))


def run_search(args: argparse.Namespace) -> optuna.Study:
    study_dir = args.study_dir
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(study_dir / 'study.db').resolve()}"

    base = TrainConfig(
        dataset=args.dataset,
        horizon=args.horizon,
        num_workers=args.num_workers,
        stride=args.stride,
        n_control_lags=args.control_lags,
        w_latent=args.w_latent,
        w_recon=args.w_recon,
        w_vic_var=args.w_vic_var,
        w_vic_cov=args.w_vic_cov,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        gradient_clip_val=args.gradient_clip_val,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        deterministic=True,
        seed=args.search_seed,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        fast_dev_run=args.fast_dev_run,
    )
    modes = tuple(args.dynamics_modes)

    sampler = optuna.samplers.TPESampler(
        seed=args.search_seed,
        n_startup_trials=args.startup_trials,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.startup_trials,
        n_warmup_steps=args.pruning_warmup_epochs,
        interval_steps=1,
    )
    metadata = {
        "study_name": args.study_name,
        "storage": storage,
        "objective": "minimum val_loss",
        "selection_split": "val",
        "test_policy": "not evaluated during search; final models only",
        "base_config": base.serializable(),
        "dynamics_modes": list(modes),
        "search_seed": args.search_seed,
    }
    metadata_path = study_dir / "study.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        comparable_keys = (
            "study_name",
            "storage",
            "objective",
            "base_config",
            "dynamics_modes",
            "search_seed",
        )
        changed = [key for key in comparable_keys if existing.get(key) != metadata.get(key)]
        if changed:
            raise ValueError(
                "Refusing to mix incomparable trials in an existing study. "
                f"Changed fields: {', '.join(changed)}. Use a new --study-dir."
            )
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2))

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    if not args.no_baseline and not study.trials:
        study.enqueue_trial(
            {
                "dynamics_mode": "linear" if "linear" in modes else modes[0],
                "latent_dim": 8,
                "hidden_size": 64,
                "spatial_latent_channels": 16,
                "activation": "relu",
                "batch_size": 32,
                "lr": 1e-3,
                "weight_decay": 1e-6,
            }
        )

    def objective(trial: optuna.Trial) -> float:
        config = suggest_config(trial, base, dynamics_modes=modes)
        trial.set_user_attr("config", config.serializable())
        run_dir = study_dir / "trials" / f"trial_{trial.number:04d}"
        pruning = OptunaPruningCallback(
            trial,
            warmup_epochs=args.pruning_warmup_epochs,
        )
        try:
            result = train_run(
                config,
                run_dir,
                extra_callbacks=[pruning],
                evaluate_test=False,
                enable_progress_bar=args.progress,
            )
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                trial.set_user_attr("failure", "out_of_memory")
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
                raise optuna.TrialPruned("Out of memory") from error
            raise
        finally:
            gc.collect()

        trial.set_user_attr("checkpoint", str(result.best_checkpoint.resolve()))
        trial.set_user_attr("best_epoch", result.best_epoch)
        trial.set_user_attr("best_step", result.best_step)
        trial.set_user_attr("stopped_epoch", result.stopped_epoch)
        trial.set_user_attr("stop_reason", result.stop_reason)
        trial.set_user_attr("duration_seconds", result.duration_seconds)
        return result.best_val_loss

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=args.progress,
    )
    _write_trials(study, study_dir / "trials.csv")

    completed = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("Search finished without a completed trial.")

    best_trial = study.best_trial
    best_config = _config_from_serialized(best_trial.user_attrs["config"])
    final_results: list[dict[str, Any]] = []

    if args.final_seeds > 0:
        for index in range(args.final_seeds):
            seed = args.final_seed_start + index
            config = replace(best_config, seed=seed)
            result = train_run(
                config,
                study_dir / "final" / f"seed_{seed}",
                evaluate_test=True,
                enable_progress_bar=args.progress,
            )
            final_results.append(result.serializable())
        selected = min(final_results, key=lambda row: row["best_val_loss"])
        selected_config = replace(
            best_config, seed=args.final_seed_start + final_results.index(selected)
        )
        selected_checkpoint = Path(selected["best_checkpoint"])
    else:
        selected_config = best_config
        selected_checkpoint = Path(best_trial.user_attrs["checkpoint"])
        test_metrics = test_checkpoint(best_config, selected_checkpoint)
        final_results.append(
            {
                "source": "best_search_trial",
                "trial": best_trial.number,
                "best_checkpoint": str(selected_checkpoint),
                "best_val_loss": best_trial.value,
                "best_epoch": best_trial.user_attrs["best_epoch"],
                "best_step": best_trial.user_attrs["best_step"],
                "stop_reason": best_trial.user_attrs["stop_reason"],
                "test_metrics": test_metrics,
            }
        )

    summary = {
        "best_search_trial": best_trial.number,
        "best_search_value": best_trial.value,
        "best_search_params": best_trial.params,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_seed": selected_config.seed,
        "final_runs": final_results,
    }
    _copy_best(
        selected_checkpoint,
        selected_config,
        study_dir,
        summary,
    )
    _write_trials(study, study_dir / "trials.csv")

    # Produce static analysis immediately; the notebook can add interpretation.
    from koopman_control.analysis.training import generate_report

    generate_report(study_dir)
    return study


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=dataset_path())
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=search_root() / "worldmodel",
    )
    parser.add_argument("--study-name", default="latent-worldmodel")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=None, help="Wall-clock seconds")
    parser.add_argument("--startup-trials", type=int, default=6)
    parser.add_argument("--pruning-warmup-epochs", type=int, default=5)
    parser.add_argument("--final-seeds", type=int, default=3)
    parser.add_argument("--final-seed-start", type=int, default=100)
    parser.add_argument("--search-seed", type=int, default=17)
    parser.add_argument(
        "--dynamics-modes",
        nargs="+",
        choices=["linear", "bilinear", "mlp"],
        default=["linear", "bilinear"],
    )
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--progress", action="store_true")

    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--control-lags", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="32-true")

    # Fixed across trials: varying these makes val_loss values incomparable.
    parser.add_argument("--w-latent", type=float, default=1.0)
    parser.add_argument("--w-recon", type=float, default=0.2)
    parser.add_argument("--w-vic-var", type=float, default=1.0)
    parser.add_argument("--w-vic-cov", type=float, default=0.04)

    # Intended for CI/smoke tests, not real studies.
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    study = run_search(parse_args(argv))
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best val_loss: {study.best_value:.6g}")
    print(json.dumps(study.best_params, indent=2))


if __name__ == "__main__":
    main()
