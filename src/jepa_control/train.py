"""Train one JEPA latent model with reproducible bookkeeping.

Every run writes ``config.json`` (full configuration), ``provenance.json`` (git
commit, dataset hash, package versions), ``logs/metrics.csv`` (epoch metrics),
``checkpoints/best-*.ckpt`` (lowest validation loss) + ``last.ckpt``, and
``result.json`` (best epoch/step, stop reason, duration, optional test metrics).

The data layer and filesystem defaults are reused from :mod:`koopman_control` so
this package does not duplicate the ABM or the generated dataset.
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
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from jepa_control.data import dataset_dims, get_dataloaders
from jepa_control.model import JEPAControl
from koopman_control.paths import dataset_path, output_root


def training_root() -> Path:
    return output_root() / "jepa_training"


@dataclass(frozen=True)
class TrainConfig:
    dataset: Path = field(default_factory=dataset_path)
    horizon: int = 16
    batch_size: int = 32
    num_workers: int = 0
    stride: int = 1

    latent_dim: int = 16
    base_channels: int = 32
    proj_hidden: int = 256
    predictor: str = "linear"
    predictor_hidden: int = 128
    predictor_layers: int = 2
    activation: str = "silu"
    n_control_lags: int = 2

    lr: float = 1e-3
    weight_decay: float = 0.0
    lr_patience: int = 4
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    w_pred: float = 1.0
    w_vic_var: float = 1.0
    # Slightly above the VICReg paper's 0.04: with latent_dim=16 the first run
    # only used ~7 effective dimensions, so a stronger decorrelation push helps
    # fill the narrower latent before we shrink further.
    w_vic_cov: float = 0.1
    w_readout: float = 0.0
    # V-JEPA-style EMA teacher by default on this branch; --target stopgrad
    # recovers the single shared-encoder recipe.
    target: str = "ema"
    ema_decay: float = 0.996

    max_epochs: int = 50
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    # Checkpoint / early-stop on predictive loss, not total val_loss. Total loss
    # mixes in VICReg (and optional readout) and previously selected epoch-2
    # checkpoints while the linear predictor was still far from the LS optimum.
    early_stopping_monitor: str = "val_pred"
    gradient_clip_val: float = 1.0
    accelerator: str = "auto"
    devices: str | int = "auto"
    precision: str = "32-true"
    deterministic: bool = True
    seed: int = 0
    log_every_n_steps: int = 20
    fast_dev_run: bool = False

    # Two-phase training: jointly train encoder+predictor for ``freeze_encoder_after_epoch``
    # epochs, then freeze the encoder and fit dynamics only. ``None`` disables.
    freeze_encoder_after_epoch: int | None = None
    ls_init_predictor: bool = False
    ls_init_max_batches: int = 64

    def serializable(self) -> dict[str, Any]:
        values = asdict(self)
        values["dataset"] = str(self.dataset)
        return values


@torch.no_grad()
def ls_init_linear_predictor(
    model: JEPAControl,
    loader: DataLoader,
    *,
    max_batches: int = 64,
    ridge: float = 1e-6,
) -> dict[str, float]:
    """Ridge-fit ``z' = A z + B u + c`` on current encodings and load into the predictor.

    Targets match the training loss (EMA teacher when present). Returns fit
    diagnostics (one-step R², ``||A-I||``, ``||B||``).
    """
    if not model.is_linear:
        raise TypeError(
            "ls_init_linear_predictor requires predictor='linear'; "
            f"got {type(model.predictor).__name__}"
        )
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    d = model.latent_dim
    n_lags = int(model.hparams.n_control_lags)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for batch_idx, batch in enumerate(loader):
        frames, controls = batch[0].to(device), batch[1].to(device)
        b, hp1 = frames.shape[0], frames.shape[1]
        h = hp1 - 1
        flat = frames.reshape(b * hp1, *frames.shape[2:])
        z_all = model.encode(flat).reshape(b, hp1, d)
        z_tgt = model.encode_target(flat).reshape(b, hp1, d)
        for k in range(h):
            uf = model._control_features(controls, k)
            ones = torch.ones(b, 1, device=device, dtype=z_all.dtype)
            xs.append(torch.cat([z_all[:, k], uf, ones], dim=-1).cpu().numpy())
            ys.append(z_tgt[:, k + 1].cpu().numpy())
        if batch_idx + 1 >= max_batches:
            break
    if not xs:
        raise RuntimeError("ls_init_linear_predictor: no batches available")
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    theta = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
    a = theta[:d].T
    b_mat = theta[d : d + n_lags].T
    c = theta[d + n_lags]
    pred = x @ theta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0)) ** 2).sum())
    model.load_linear_predictor(a, b_mat, c)
    if was_training:
        model.train()
    return {
        "one_step_r2": 1.0 - ss_res / max(ss_tot, 1e-12),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(a)))),
        "a_minus_i_fro": float(np.linalg.norm(a - np.eye(d))),
        "b_fro": float(np.linalg.norm(b_mat)),
        "n_transitions": int(x.shape[0]),
    }


class TwoPhaseTraining(Callback):
    """Phase 1: joint encoder+predictor. Phase 2: freeze encoder, train predictor.

    At the start of ``freeze_after_epoch``, freezes the encoder (and zeros VICReg
    weights), optionally ridge-initializes the linear predictor from current
    latents, and resets early-stopping / best-checkpoint state so phase-1's
    near-init ``val_pred`` cannot permanently win.
    """

    def __init__(
        self,
        freeze_after_epoch: int,
        *,
        ls_init: bool = True,
        ls_init_max_batches: int = 64,
    ) -> None:
        if freeze_after_epoch < 0:
            raise ValueError(f"freeze_after_epoch must be >= 0, got {freeze_after_epoch}")
        self.freeze_after_epoch = freeze_after_epoch
        self.ls_init = ls_init
        self.ls_init_max_batches = ls_init_max_batches
        self._done = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: JEPAControl) -> None:
        if self._done or trainer.current_epoch < self.freeze_after_epoch:
            return
        if pl_module.encoder_frozen:
            self._done = True
            return

        print(
            f"[two-phase] epoch {trainer.current_epoch}: freezing encoder "
            f"(phase-2 predictor-only training)"
        )
        pl_module.freeze_encoder_for_phase2(zero_vicreg=True)

        if self.ls_init:
            if not pl_module.is_linear:
                print(
                    "[two-phase] skipping LS init: requires predictor='linear' "
                    f"(got {type(pl_module.predictor).__name__})"
                )
            else:
                loader = trainer.train_dataloader
                if loader is None:
                    raise RuntimeError("two-phase LS init needs trainer.train_dataloader")
                stats = ls_init_linear_predictor(
                    pl_module, loader, max_batches=self.ls_init_max_batches
                )
                print(
                    "[two-phase] LS-initialized linear predictor: "
                    f"one_step_r2={stats['one_step_r2']:+.4f}  "
                    f"||A-I||_F={stats['a_minus_i_fro']:.4f}  "
                    f"||B||_F={stats['b_fro']:.4f}  "
                    f"rho(A)={stats['spectral_radius']:.4f}  "
                    f"n={stats['n_transitions']}"
                )

        self._reset_monitors(trainer)
        # Drop encoder params from the optimizer so Adam state does not keep
        # updating frozen weights if any stray grad appears.
        self._rebuild_optimizer(trainer, pl_module)
        self._done = True

    @staticmethod
    def _reset_monitors(trainer: pl.Trainer) -> None:
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping):
                cb.best_score = torch.tensor(float("inf"))
                cb.wait_count = 0
            if isinstance(cb, ModelCheckpoint):
                cb.best_model_score = None
                cb.best_k_models = {}

    @staticmethod
    def _rebuild_optimizer(trainer: pl.Trainer, pl_module: JEPAControl) -> None:
        conf = pl_module.configure_optimizers()
        if not isinstance(conf, dict):
            raise TypeError("expected configure_optimizers() to return a dict")
        optimizer = conf["optimizer"]
        trainer.optimizers = [optimizer]
        sch_conf = conf.get("lr_scheduler")
        if sch_conf is not None and trainer.lr_scheduler_configs:
            # Replace the scheduler attached to the first config in place.
            scheduler = sch_conf["scheduler"]
            trainer.lr_scheduler_configs[0].scheduler = scheduler
            trainer.lr_scheduler_configs[0].optimizer = optimizer


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


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(config: TrainConfig) -> dict[str, Any]:
    dataset = config.dataset.resolve()
    stat = dataset.stat()
    versions = {}
    for name in ("numpy", "torch", "pytorch-lightning", "h5py"):
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
        "dataset_sha256": _dataset_sha256(dataset),
    }


def _model(config: TrainConfig) -> JEPAControl:
    num_channels, input_size, n_obs, _ = dataset_dims(config.dataset)
    return JEPAControl(
        num_channels=num_channels,
        input_size=input_size,
        base_channels=config.base_channels,
        proj_hidden=config.proj_hidden,
        latent_dim=config.latent_dim,
        predictor=config.predictor,
        predictor_hidden=config.predictor_hidden,
        predictor_layers=config.predictor_layers,
        activation=config.activation,
        n_obs=n_obs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        lr_patience=config.lr_patience,
        lr_factor=config.lr_factor,
        min_lr=config.min_lr,
        horizon=config.horizon,
        n_control_lags=config.n_control_lags,
        w_pred=config.w_pred,
        w_vic_var=config.w_vic_var,
        w_vic_cov=config.w_vic_cov,
        w_readout=config.w_readout,
        target=config.target,
        ema_decay=config.ema_decay,
    )


def train_run(
    config: TrainConfig,
    run_dir: str | Path,
    *,
    extra_callbacks: list[Callback] | None = None,
    evaluate_test: bool = True,
    enable_progress_bar: bool = True,
) -> RunResult:
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

    monitor = config.early_stopping_monitor
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:03d}-{step}-{" + monitor + ":.6f}",
        auto_insert_metric_name=False,
        monitor=monitor,
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    early_stopping = EarlyStopping(
        monitor=monitor,
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
    ]
    if config.freeze_encoder_after_epoch is not None:
        callbacks.append(
            TwoPhaseTraining(
                freeze_after_epoch=config.freeze_encoder_after_epoch,
                ls_init=config.ls_init_predictor,
                ls_init_max_batches=config.ls_init_max_batches,
            )
        )
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
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
        log_every_n_steps=config.log_every_n_steps,
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
        best_path = checkpoint_dir / "fast-dev.ckpt"
        trainer.save_checkpoint(best_path)
        best_score = float(trainer.callback_metrics.get(monitor, float("nan")))

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    best_epoch, best_step = int(ckpt.get("epoch", -1)), int(ckpt.get("global_step", -1))

    test_metrics: dict[str, float] = {}
    if evaluate_test:
        rows = trainer.test(dataloaders=test_loader, ckpt_path=str(best_path), verbose=False)
        if rows:
            test_metrics = {k: float(v) for k, v in rows[0].items()}

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=dataset_path())
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", type=Path, default=training_root())

    p.add_argument("--horizon", type=int, default=TrainConfig.horizon)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--stride", type=int, default=TrainConfig.stride)
    p.add_argument("--latent-dim", type=int, default=TrainConfig.latent_dim)
    p.add_argument("--base-channels", type=int, default=TrainConfig.base_channels)
    p.add_argument("--proj-hidden", type=int, default=TrainConfig.proj_hidden)
    p.add_argument(
        "--predictor",
        choices=["linear", "residual_mlp"],
        default=TrainConfig.predictor,
        help="Latent dynamics: 'linear' (z' = A z + B u + c) or the nonlinear ablation.",
    )
    p.add_argument("--predictor-hidden", type=int, default=TrainConfig.predictor_hidden)
    p.add_argument("--predictor-layers", type=int, default=TrainConfig.predictor_layers)
    p.add_argument("--activation", choices=["relu", "silu", "tanh", "leakyrelu"], default="silu")
    p.add_argument("--control-lags", type=int, default=TrainConfig.n_control_lags)

    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--w-pred", type=float, default=TrainConfig.w_pred)
    p.add_argument("--w-vic-var", type=float, default=TrainConfig.w_vic_var)
    p.add_argument("--w-vic-cov", type=float, default=TrainConfig.w_vic_cov)
    p.add_argument(
        "--w-readout",
        type=float,
        default=TrainConfig.w_readout,
        help="Anchoring weight; 0 keeps the encoder purely self-supervised (pure JEPA).",
    )
    p.add_argument(
        "--target",
        choices=["ema", "stopgrad"],
        default=TrainConfig.target,
        help="Predictive targets: EMA teacher encoder (default) or stop-grad shared encoder.",
    )
    p.add_argument(
        "--ema-decay",
        type=float,
        default=TrainConfig.ema_decay,
        help="EMA decay τ for the target encoder (θ_tgt ← τ θ_tgt + (1-τ) θ). Ignored if --target stopgrad.",
    )

    p.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    p.add_argument(
        "--early-stopping-patience", type=int, default=TrainConfig.early_stopping_patience
    )
    p.add_argument(
        "--early-stopping-monitor",
        default=TrainConfig.early_stopping_monitor,
        choices=["val_pred", "val_loss"],
        help="Metric for checkpointing / early stop. Prefer val_pred so VICReg/readout "
        "do not pick an under-trained predictor.",
    )
    p.add_argument("--gradient-clip-val", type=float, default=TrainConfig.gradient_clip_val)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--accelerator", default=TrainConfig.accelerator)
    p.add_argument("--devices", default=TrainConfig.devices)
    p.add_argument("--precision", default=TrainConfig.precision)
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--no-test", action="store_true")
    p.add_argument("--no-progress", action="store_true")

    phase = p.add_argument_group(
        "two-phase training",
        "Phase 1 learns the encoder; phase 2 freezes it and fits the predictor "
        "(optionally LS-warm-started). Use --two-phase for the default recipe.",
    )
    phase.add_argument(
        "--two-phase",
        action="store_true",
        help="Enable two-phase training: freeze encoder after --phase1-epochs and "
        "LS-init the linear predictor (unless --no-ls-init-predictor).",
    )
    phase.add_argument(
        "--phase1-epochs",
        type=int,
        default=5,
        help="Epochs of joint training before freezing when --two-phase is set "
        "(default: 5). Ignored if --freeze-encoder-after-epoch is given.",
    )
    phase.add_argument(
        "--freeze-encoder-after-epoch",
        type=int,
        default=None,
        help="Freeze encoder (+ EMA) at the start of this epoch. 0 = predictor-only "
        "from the first epoch. Overrides --phase1-epochs.",
    )
    phase.add_argument(
        "--ls-init-predictor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="At freeze time, ridge-fit A/B/c into the linear predictor from current "
        "encodings. Defaults to on with --two-phase, off otherwise.",
    )
    phase.add_argument(
        "--ls-init-max-batches",
        type=int,
        default=TrainConfig.ls_init_max_batches,
        help="Train batches used for the LS warm-start fit.",
    )
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    freeze_after = args.freeze_encoder_after_epoch
    if freeze_after is None and args.two_phase:
        freeze_after = args.phase1_epochs
    ls_init = args.ls_init_predictor
    if ls_init is None:
        ls_init = bool(args.two_phase)

    return TrainConfig(
        dataset=args.dataset,
        horizon=args.horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stride=args.stride,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        proj_hidden=args.proj_hidden,
        predictor=args.predictor,
        predictor_hidden=args.predictor_hidden,
        predictor_layers=args.predictor_layers,
        activation=args.activation,
        n_control_lags=args.control_lags,
        lr=args.lr,
        weight_decay=args.weight_decay,
        w_pred=args.w_pred,
        w_vic_var=args.w_vic_var,
        w_vic_cov=args.w_vic_cov,
        w_readout=args.w_readout,
        target=args.target,
        ema_decay=args.ema_decay,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_monitor=args.early_stopping_monitor,
        gradient_clip_val=args.gradient_clip_val,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        seed=args.seed,
        fast_dev_run=args.fast_dev_run,
        freeze_encoder_after_epoch=freeze_after,
        ls_init_predictor=ls_init,
        ls_init_max_batches=args.ls_init_max_batches,
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
