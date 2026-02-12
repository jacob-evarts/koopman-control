from pathlib import Path
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback


class LossCSVCallback(pl.Callback):
    def __init__(self, writer, trial_id: int):
        self.writer = writer
        self.trial_id = trial_id

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")

        if train_loss is None:
            return

        train_loss = float(train_loss.detach().cpu()) if train_loss is not None else None
        val_loss = float(val_loss.detach().cpu()) if val_loss is not None else None

        self.writer.save_loss(
            self.trial_id,
            epoch=trainer.current_epoch,
            train_loss=train_loss,
            val_loss=val_loss,
        )

class LossPlotCallback(Callback):
    def __init__(self, checkpoint_cb: ModelCheckpoint, save_dir: Path, trial_id: int):
        super().__init__()
        self.checkpoint_cb = checkpoint_cb
        self.save_dir = save_dir
        self.trial_id = trial_id
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.train_losses.append(train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.val_losses.append(val_loss.item())

    def on_fit_end(self, trainer, pl_module):
        if self.checkpoint_cb.best_model_path:
            plt.figure()
            plt.plot(self.train_losses, label="Train Loss")
            plt.plot(self.val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Trial {self.trial_id} Loss Curve")
            plt.legend()
            self.save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(self.save_dir / f"trial_{self.trial_id}_loss_curve.png")
            plt.close()