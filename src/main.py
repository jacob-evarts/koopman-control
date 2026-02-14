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

from writer import Writer
from models.koopman_model import KoopmanAE
from loaders import get_dataloaders
from utils.callbacks import LossCSVCallback, LossPlotCallback

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    run_id = f"{uuid.uuid4().hex[:3]}"
    run_dir = Path(HydraConfig.get().run.dir)

    train_loader, val_loader, test_loader = get_dataloaders(cfg.dataset.data_dir, cfg.dataset.batch_size)

    writer = Writer(run_dir=run_dir, run_id=run_id)
    objective = _make_objective(cfg, train_loader, val_loader, test_loader, writer)

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
    writer.save_study_summary(study)
    writer.save_optuna_plots(study)
    
def _make_objective(cfg: DictConfig, 
                    train_loader: torch.utils.data.DataLoader, 
                    val_loader: torch.utils.data.DataLoader, 
                    test_loader: torch.utils.data.DataLoader,
                    writer: Writer,
                    run_id: str = None) -> callable:
    def objective(trial: optuna.trial.Trial):
        trial_params = {}
        for name, space in cfg.optim.search_space.items():
            if space.type == "int":
                trial_params[name] = trial.suggest_int(name, space.low, space.high)
            elif space.type == "loguniform":
                trial_params[name] = trial.suggest_float(name, space.low, space.high, log=True)
            elif space.type == "categorical":
                trial_params[name] = trial.suggest_categorical(name, space.choices)

        model = KoopmanAE(
            hidden_size=trial_params.get("hidden_size", cfg.model.hidden_size),
            lr=trial_params.get("lr", cfg.model.lr),
            latent_dim=trial_params.get("latent_dim", cfg.model.latent_dim),
            activation=trial_params.get("activation", cfg.model.activation),
        )

        checkpoint_cb = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            filename=f"trial-{trial.number:03d}-{{epoch:03d}}",
            dirpath=writer.checkpoints_dir
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
                LossCSVCallback(writer, trial.number),
            ],
            **cfg.trainer
        )

        trainer.fit(model, train_loader, val_loader)
        return trainer.callback_metrics["val_loss"].item()

    return objective

if __name__ == "__main__":
    main()