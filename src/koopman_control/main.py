import random
import uuid
import hydra
import optuna
import torch
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback

from koopman_control.writer import Writer
from koopman_control.loaders.dataloaders import get_dataloaders
from koopman_control.models.factory import build_model
from koopman_control.utils.callbacks import LossCSVCallback, LossPlotCallback

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_id = f"{uuid.uuid4().hex[:3]}"
    run_dir = Path(HydraConfig.get().run.dir)
    run_study(cfg, run_dir, run_id)


def run_study(cfg: DictConfig, run_dir: Path, run_id: str) -> None:
    """Run Optuna study: load data, create writer and objective, optimize, save results."""
    train_loader, val_loader, test_loader, dataset_props = get_dataloaders(cfg.dataset)
    writer = Writer(run_dir=run_dir, run_id=run_id)
    objective = _make_objective(
        cfg, train_loader, val_loader, writer, dataset_props, run_id
    )

    study = optuna.create_study(
        study_name=f"koopman_{run_id}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=3,
        ),
    )

    study.optimize(objective, cfg.optim.n_trials)
    best_trial = study.best_trial
    writer.remove_bested_checkpoints(best_trial.number)

    summary = {
        "run_id": run_id,
        "model_type": dataset_props.model_type,
        "dataset": cfg.dataset.data_dir,
        "batch_size": cfg.dataset.batch_size,
        "best_trial_number": best_trial.number,
        "best_val_loss": best_trial.value,
        "best_params": best_trial.params,
        "all_trials": [
            {
                "trial_number": t.number,
                "val_loss": t.value,
                "params": t.params,
            }
            for t in study.trials
        ],
    }
    writer.save_study_summary(summary)
    writer.save_optuna_plots(study)


def _make_objective(
    cfg: DictConfig,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    writer: Writer,
    dataset_props,
    run_id: str):
    def objective(trial: optuna.trial.Trial):
        trial_params = {}
        for name, space in cfg.optim.search_space.items():
            if space.type == "int":
                trial_params[name] = trial.suggest_int(name, space.low, space.high)
            elif space.type == "loguniform":
                trial_params[name] = trial.suggest_float(
                    name, space.low, space.high, log=True
                )
            elif space.type == "categorical":
                trial_params[name] = trial.suggest_categorical(name, space.choices)

        model = build_model(cfg, trial_params, dataset_props)

        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            filename=f"trial-{trial.number:03d}-{{epoch:03d}}",
            dirpath=writer.checkpoints_dir,
        )

        trainer = pl.Trainer(
            logger=WandbLogger(
                **cfg.wandb,
                name=f"{run_id}-trial-{trial.number}",
                group=f"koopman-{run_id}",
            ),
            callbacks=[
                checkpoint_cb,
                EarlyStopping(monitor="val_loss", patience=4, mode="min"),
                PyTorchLightningPruningCallback(trial, monitor="val_loss"),
                LossPlotCallback(
                    checkpoint_cb=checkpoint_cb,
                    save_dir=writer.figures_dir,
                    trial_id=trial.number,
                ),
                LossCSVCallback(writer.save_loss, trial.number),
            ],
            **cfg.trainer,
        )

        trainer.fit(model, train_loader, val_loader)
        return trainer.callback_metrics["val_loss"].item()

    return objective


if __name__ == "__main__":
    main()
