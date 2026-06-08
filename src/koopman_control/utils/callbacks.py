from pathlib import Path
from typing import Callable
import gc
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, Callback


class LossCSVCallback(pl.Callback):
    """Expects a callable (trial_id, epoch, train_loss, val_loss) -> None (e.g. Writer.save_loss)."""

    def __init__(self, save_loss_fn: Callable[[int, int, float, float], None], trial_id: int):
        self.save_loss_fn = save_loss_fn
        self.trial_id = trial_id

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")

        if train_loss is None:
            return

        train_loss = float(train_loss.detach().cpu()) if train_loss is not None else None
        val_loss = float(val_loss.detach().cpu()) if val_loss is not None else None

        self.save_loss_fn(
            self.trial_id,
            trainer.current_epoch,
            train_loss,
            val_loss,
        )


class MpsMemoryCallback(Callback):
    """Release MPS allocator cache at epoch boundaries.

    Frequent ``empty_cache`` during an epoch tends to worsen MPS thrashing by
    forcing repeated re-allocation; intra-epoch clearing is off by default.
    """

    def __init__(self, empty_cache_every_n_train_batches: int = 0) -> None:
        super().__init__()
        self.empty_cache_every_n_train_batches = empty_cache_every_n_train_batches

    def _on_mps(self, pl_module) -> bool:
        return torch.backends.mps.is_available() and pl_module.device.type == "mps"

    def _release_mps_cache(self) -> None:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            gc.collect()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        n = self.empty_cache_every_n_train_batches
        if n > 0 and (batch_idx + 1) % n == 0 and self._on_mps(pl_module):
            self._release_mps_cache()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._on_mps(pl_module):
            self._release_mps_cache()


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
