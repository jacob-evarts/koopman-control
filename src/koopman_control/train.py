"""Train one image-based latent world model with reproducible bookkeeping.

This module is both the single-run CLI and the reusable runner used by
``koopman_control.search``. Every run writes:

``config.json``
    Complete configuration needed to reproduce the run.
``logs/metrics.csv``
    Epoch-level training and validation metrics.
``checkpoints/best-*.ckpt``
    The step with the lowest validation loss, not merely the final epoch.
``checkpoints/last.ckpt``
    Resume point if the run is interrupted.
``result.json``
    Best epoch/step, stopping reason, duration, and optional held-out test metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

from koopman_control.data.dataset import get_dataloaders
from koopman_control.models.world_model import LatentWorldModel
from koopman_control.paths import dataset_path, training_root


@dataclass(frozen=True)
class TrainConfig:
    dataset: Path = field(default_factory=dataset_path)
    horizon: int = 20
    batch_size: int = 32
    num_workers: int = 0
    stride: int = 1

    latent_dim: int = 16
    hidden_size: int = 64
    spatial_latent_channels: int = 16
    activation: str = "relu"
    dynamics_mode: str = "linear"
    n_control_lags: int = 2

    lr: float = 1e-3
    weight_decay: float = 0.0
    lr_patience: int = 4
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    w_latent: float = 1.0
    w_recon: float = 0.2
    w_vic_var: float = 1.0
    w_vic_cov: float = 0.04

    max_epochs: int = 50
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    accelerator: str = "auto"
    devices: str | int = "auto"
    precision: str = "32-true"
    deterministic: bool = True
    seed: int = 0
    log_every_n_steps: int = 20
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    fast_dev_run: bool = False

    def serializable(self) -> dict[str, Any]:
        values = asdict(self)
        values["dataset"] = str(self.dataset)
        return values


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    best_checkpoint: Path
    best_val_loss: float
    best_epoch: int
    best_step: int
    stopped_epoch: int
    stop_reason: str
    duration_seconds: float
    test_metrics: dict[str, float]

    def serializable(self) -> dict[str, Any]:
        values = asdict(self)
        values["run_dir"] = str(self.run_dir)
        values["best_checkpoint"] = str(self.best_checkpoint)
        return values


def _dataset_dims(h5_path: Path) -> tuple[int, int]:
    with h5py.File(h5_path, "r") as f:
        return int(f.attrs["num_channels"]), int(f.attrs["width"])


@lru_cache(maxsize=8)
def _dataset_sha256(path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns  # Included in the cache key so changed files are re-hashed.
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _provenance(config: TrainConfig) -> dict[str, Any]:
    dataset = config.dataset.resolve()
    stat = dataset.stat()
    package_names = ("numpy", "torch", "pytorch-lightning", "h5py", "optuna")
    versions = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": versions,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
        "dataset_path": str(dataset),
        "dataset_bytes": stat.st_size,
        "dataset_sha256": _dataset_sha256(
            str(dataset),
            stat.st_size,
            stat.st_mtime_ns,
        ),
    }


def _checkpoint_position(path: Path) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", -1)), int(checkpoint.get("global_step", -1))


def _model(config: TrainConfig) -> LatentWorldModel:
    num_channels, input_size = _dataset_dims(config.dataset)
    return LatentWorldModel(
        num_channels=num_channels,
        input_size=input_size,
        hidden_size=config.hidden_size,
        spatial_latent_channels=config.spatial_latent_channels,
        latent_dim=config.latent_dim,
        activation=config.activation,
        lr=config.lr,
        weight_decay=config.weight_decay,
        lr_patience=config.lr_patience,
        lr_factor=config.lr_factor,
        min_lr=config.min_lr,
        horizon=config.horizon,
        n_control_lags=config.n_control_lags,
        dynamics_mode=config.dynamics_mode,
        w_latent=config.w_latent,
        w_recon=config.w_recon,
        w_vic_var=config.w_vic_var,
        w_vic_cov=config.w_vic_cov,
    )


def train_run(
    config: TrainConfig,
    run_dir: str | Path,
    *,
    extra_callbacks: list[Callback] | None = None,
    evaluate_test: bool = True,
    enable_progress_bar: bool = True,
) -> RunResult:
    """Train one run and return the validation-selected checkpoint metadata."""
    run_dir = Path(run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.serializable(), indent=2))
    (run_dir / "provenance.json").write_text(json.dumps(_provenance(config), indent=2))

    pl.seed_everything(config.seed, workers=True)
    train_loader, val_loader, test_loader = get_dataloaders(
        config.dataset,
        horizon=config.horizon,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        stride=config.stride,
    )

    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:03d}-{step}-{val_loss:.6f}",
        auto_insert_metric_name=False,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        save_weights_only=False,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        check_finite=True,
        verbose=True,
    )
    callbacks: list[Callback] = [
        checkpoint,
        early_stopping,
        LearningRateMonitor(logging_interval="epoch"),
        *(extra_callbacks or []),
    ]
    logger = CSVLogger(save_dir=run_dir, name="logs", version="")

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator=config.accelerator,
        devices=config.devices,
        precision=config.precision,
        deterministic=config.deterministic,
        benchmark=False,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=run_dir,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=config.log_every_n_steps,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        fast_dev_run=config.fast_dev_run,
        enable_progress_bar=enable_progress_bar,
    )

    started = time.perf_counter()
    trainer.fit(_model(config), train_loader, val_loader)
    duration = time.perf_counter() - started

    if checkpoint.best_model_path:
        best_path = Path(checkpoint.best_model_path)
        best_score = float(checkpoint.best_model_score)
    else:
        # Lightning suppresses checkpoint callbacks during fast_dev_run.
        best_path = checkpoint_dir / "fast-dev.ckpt"
        trainer.save_checkpoint(best_path)
        metric = trainer.callback_metrics.get("val_loss", float("nan"))
        best_score = float(metric)

    best_epoch, best_step = _checkpoint_position(best_path)
    test_metrics: dict[str, float] = {}
    if evaluate_test:
        rows = trainer.test(dataloaders=test_loader, ckpt_path=str(best_path), verbose=False)
        if rows:
            test_metrics = {key: float(value) for key, value in rows[0].items()}

    if config.fast_dev_run:
        stop_reason = "fast_dev_run"
    elif early_stopping.stopped_epoch > 0:
        stop_reason = "early_stopping"
    elif trainer.current_epoch >= config.max_epochs:
        stop_reason = "max_epochs"
    else:
        stop_reason = "completed"

    result = RunResult(
        run_dir=run_dir,
        best_checkpoint=best_path,
        best_val_loss=best_score,
        best_epoch=best_epoch,
        best_step=best_step,
        stopped_epoch=int(trainer.current_epoch),
        stop_reason=stop_reason,
        duration_seconds=duration,
        test_metrics=test_metrics,
    )
    (run_dir / "result.json").write_text(json.dumps(result.serializable(), indent=2))
    return result


def test_checkpoint(
    config: TrainConfig,
    checkpoint_path: str | Path,
) -> dict[str, float]:
    """Evaluate a validation-selected checkpoint once on the held-out test split."""
    _, _, test_loader = get_dataloaders(
        config.dataset,
        horizon=config.horizon,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        stride=config.stride,
    )
    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        precision=config.precision,
        deterministic=config.deterministic,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    rows = trainer.test(
        model=LatentWorldModel.load_from_checkpoint(str(checkpoint_path), map_location="cpu"),
        dataloaders=test_loader,
        verbose=False,
    )
    return {key: float(value) for key, value in (rows[0] if rows else {}).items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=dataset_path())
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--out-dir", type=Path, default=training_root())

    parser.add_argument("--horizon", type=int, default=TrainConfig.horizon)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--stride", type=int, default=TrainConfig.stride)
    parser.add_argument("--latent-dim", type=int, default=TrainConfig.latent_dim)
    parser.add_argument("--hidden-size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument(
        "--spatial-latent-channels",
        type=int,
        default=TrainConfig.spatial_latent_channels,
    )
    parser.add_argument("--activation", choices=["relu", "silu", "tanh"], default="relu")
    parser.add_argument(
        "--dynamics-mode",
        choices=["linear", "bilinear", "mlp"],
        default=TrainConfig.dynamics_mode,
    )
    parser.add_argument("--control-lags", type=int, default=TrainConfig.n_control_lags)

    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--lr-patience", type=int, default=TrainConfig.lr_patience)
    parser.add_argument("--lr-factor", type=float, default=TrainConfig.lr_factor)
    parser.add_argument("--min-lr", type=float, default=TrainConfig.min_lr)
    parser.add_argument("--w-latent", type=float, default=TrainConfig.w_latent)
    parser.add_argument("--w-recon", type=float, default=TrainConfig.w_recon)
    parser.add_argument("--w-vic-var", type=float, default=TrainConfig.w_vic_var)
    parser.add_argument("--w-vic-cov", type=float, default=TrainConfig.w_vic_cov)

    parser.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=TrainConfig.early_stopping_patience,
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=TrainConfig.early_stopping_min_delta,
    )
    parser.add_argument("--gradient-clip-val", type=float, default=TrainConfig.gradient_clip_val)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--accelerator", default=TrainConfig.accelerator)
    parser.add_argument("--devices", default=TrainConfig.devices)
    parser.add_argument("--precision", default=TrainConfig.precision)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        dataset=args.dataset,
        horizon=args.horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stride=args.stride,
        latent_dim=args.latent_dim,
        hidden_size=args.hidden_size,
        spatial_latent_channels=args.spatial_latent_channels,
        activation=args.activation,
        dynamics_mode=args.dynamics_mode,
        n_control_lags=args.control_lags,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        min_lr=args.min_lr,
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
        seed=args.seed,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        fast_dev_run=args.fast_dev_run,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    result = train_run(
        config_from_args(args),
        args.out_dir / run_name,
        evaluate_test=not args.no_test,
        enable_progress_bar=not args.no_progress,
    )
    print(json.dumps(result.serializable(), indent=2))


if __name__ == "__main__":
    main()
