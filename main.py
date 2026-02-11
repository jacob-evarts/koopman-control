import os
import random
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
from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate
    )

from lightning_model import SimpleMLP
from loss_plot import LossPlotCallback
from data import get_dataloaders

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    save_dir = Path(cfg.save_dirs.figures)
    save_dir.mkdir(exist_ok=True)

    def objective(trial:optuna.trial):
        trial_params = {}
        for name, space in cfg.optim.search_space.items():
            if space.type == "int":
                trial_params[name] = trial.suggest_int(name, space.low, space.high)
            elif space.type == "loguniform":
                trial_params[name] = trial.suggest_float(name, space.low, space.high, log=True)

        model = SimpleMLP(
            hidden_size=trial_params.get("hidden_size", cfg.model.hidden_size),
            lr=trial_params.get("lr", cfg.model.lr),
        )
        train_loader, val_loader = get_dataloaders(cfg)

        logger = WandbLogger(**cfg.wandb)
        checkpoint_cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=False)

        trainer = pl.Trainer(
            logger=logger,
            callbacks=[
                checkpoint_cb,
                EarlyStopping(monitor="val_loss", patience=4, mode="min"),
                PyTorchLightningPruningCallback(trial, monitor="val_loss"),
                LossPlotCallback(
                    checkpoint_cb=checkpoint_cb,
                    save_dir=save_dir,
                    trial_id=trial.number
                )
            ],
            **cfg.trainer
        )

        trainer.fit(model, train_loader, val_loader)
        return trainer.callback_metrics["val_loss"].item()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1),
    )
    study.optimize(objective, cfg.optim.n_trials)

    print("Best params:", study.best_params)
    fig1 = plot_optimization_history(study)
    fig1.write_image(save_dir / "optimization_history.png")
    fig2 = plot_param_importances(study)
    fig2.write_image(save_dir / "param_importances.png")
    fig3 = plot_parallel_coordinate(study)
    fig3.write_image(save_dir / "parallel_coordinate.png")
    
if __name__ == "__main__":
    main()