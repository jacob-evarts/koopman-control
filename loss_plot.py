import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pathlib import Path

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