"""Decoder-free JEPA latent dynamics model with VICReg regularization.

Why this model
--------------
Goal: encode an image observation of the ABM to a latent ``z``, evolve it under
a control ``u`` with an action-conditioned predictor, and plan control with MPC
over that predictor. Unlike the sibling
:class:`koopman_control.models.world_model.LatentWorldModel`, this model:

  * has **no decoder** and **no reconstruction loss** -- the latent is judged
    only by how well it *predicts its own future* (JEPA) and by post-hoc probes,
  * places **no linearity constraint on the encoder** -- the latent is whatever
    the predictive + VICReg objective discovers, and
  * relies on **VICReg alone** (variance + covariance) to prevent the collapse
    that a stop-grad predictive loss would otherwise allow.

Choice of predictor
-------------------
The default predictor is **linear** (``z' = A z + B u + c``). This is an
empirical finding rather than an a-priori constraint: on the first trained model
the discovered latent turned out to be almost exactly linear in time (post-hoc
least-squares one-step R^2 = 0.974), and a plain least-squares linear operator
fit on the frozen latent *outperformed* the trained nonlinear MLP on 16-step
rollouts (skill 0.37 vs 0.08) -- the MLP was chasing a moving stop-grad target
and never converged to the good operator that demonstrably existed. A linear
predictor matches the data, optimizes far more reliably, and makes ``A``/``B``
directly readable for spectral-radius and controllability analysis (and LQR).

``predictor="residual_mlp"`` keeps the original nonlinear variant available as
an ablation, which is what the linear default should be compared against.

Target encoder
--------------
Predictive targets come from one of two mechanisms (``target`` hyperparameter):

  * ``"ema"`` (default on this branch): a second encoder whose weights are an
    exponential moving average of the online encoder (V-JEPA-style). Futures are
    encoded with the EMA encoder under ``torch.no_grad``. This gives the
    predictor a slowly moving target instead of a stop-grad copy of the same
    weights being updated in the same step.
  * ``"stopgrad"``: single shared encoder; target latents are
    ``online_encode(x).detach()`` (the original recipe).

Losses
------
  * ``pred`` : multi-step latent-prediction MSE against target embeddings,
    rolled out over a horizon so the dynamics stay accurate under free-running
    prediction, not just one step.
  * ``vic``  : VICReg variance hinge (std -> 1) + off-diagonal covariance
    penalty, applied over **all online** encodings in the window.
  * ``readout`` (optional, ``w_readout = 0`` by default): a light auxiliary MSE
    from ``z`` to the ground-truth macrostate observables. Off by default so the
    encoder is purely self-supervised; turned on only if the unsupervised latent
    proves hard to control.

Actuator lag: the control feature for each transition is the history
``[u_now, u_prev, ...]`` (see :mod:`jepa_control.data`).
"""

from __future__ import annotations

import copy

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "leakyrelu": nn.LeakyReLU,
}


class ConvEncoder(nn.Module):
    """From-scratch conv backbone -> flattened features -> latent projection.

    Four stride-2 convolutions downsample the input by 16x (64 -> 4, 16 -> 1).
    The spatial features are flattened (not globally pooled) before the
    projection so spatial structure is preserved in the flattened vector -- this
    keeps the door open for the phase-2 spatial-token variant without changing
    the interface.
    """

    def __init__(
        self,
        num_channels: int,
        input_size: int,
        base_channels: int,
        latent_dim: int,
        proj_hidden: int,
        activation: str,
    ) -> None:
        super().__init__()
        act = ACTIVATIONS[activation]
        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 4,
        )
        self.backbone = nn.Sequential(
            nn.Conv2d(num_channels, c1, 4, stride=2, padding=1),
            act(),
            nn.Conv2d(c1, c2, 4, stride=2, padding=1),
            act(),
            nn.Conv2d(c2, c3, 4, stride=2, padding=1),
            act(),
            nn.Conv2d(c3, c4, 4, stride=2, padding=1),
            act(),
        )
        spatial = max(1, input_size // 16)
        self.flat_dim = c4 * spatial * spatial
        self.proj = nn.Sequential(
            nn.Linear(self.flat_dim, proj_hidden),
            act(),
            nn.Linear(proj_hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.proj(feats.reshape(feats.shape[0], -1))


class LinearPredictor(nn.Module):
    """Linear action-conditioned predictor ``z_{t+1} = A z_t + B u + c``.

    ``A`` starts at the identity and ``B`` at zero, so the map begins as a no-op
    and multi-step rollouts are stable from the first gradient step (the ABM's
    population dynamics are near-identity per step).

    ``A`` and ``B`` are kept as separate matrices rather than one fused ``Linear``
    so the spectral radius and the controllability of the learned dynamics can be
    read straight off the parameters. The bias on ``A`` carries the affine offset
    ``c``.
    """

    def __init__(self, latent_dim: int, n_control: int) -> None:
        super().__init__()
        self.A = nn.Linear(latent_dim, latent_dim, bias=True)
        self.B = nn.Linear(n_control, latent_dim, bias=False)
        with torch.no_grad():
            self.A.weight.copy_(torch.eye(latent_dim))
            self.A.bias.zero_()
            self.B.weight.zero_()

    def forward(self, z: torch.Tensor, u_feat: torch.Tensor) -> torch.Tensor:
        return self.A(z) + self.B(u_feat)


class ResidualPredictor(nn.Module):
    """Nonlinear action-conditioned predictor ``z_{t+1} = z_t + f([z_t, u])``.

    The residual form keeps the one-step map close to identity at init (slow,
    stable population dynamics are near-identity in latent space), which makes
    multi-step rollouts well behaved early in training.
    """

    def __init__(
        self,
        latent_dim: int,
        n_control: int,
        hidden: int,
        n_layers: int,
        activation: str,
    ) -> None:
        super().__init__()
        act = ACTIVATIONS[activation]
        dims = [latent_dim + n_control] + [hidden] * n_layers
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            layers += [nn.Linear(a, b), act()]
        layers.append(nn.Linear(dims[-1], latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, u_feat: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, u_feat], dim=-1))


class JEPAControl(pl.LightningModule):
    def __init__(
        self,
        num_channels: int = 2,
        input_size: int = 64,
        base_channels: int = 32,
        proj_hidden: int = 256,
        latent_dim: int = 16,
        predictor: str = "linear",
        predictor_hidden: int = 128,
        predictor_layers: int = 2,
        activation: str = "silu",
        n_obs: int = 0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_patience: int = 4,
        lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        horizon: int = 8,
        n_control_lags: int = 2,
        w_pred: float = 1.0,
        w_vic_var: float = 1.0,
        w_vic_cov: float = 0.1,
        w_readout: float = 0.0,
        target: str = "ema",
        ema_decay: float = 0.996,
        encoder_frozen: bool = False,
    ) -> None:
        super().__init__()
        if target not in ("ema", "stopgrad"):
            raise ValueError(f"unknown target {target!r}; expected 'ema' or 'stopgrad'")
        if not (0.0 <= ema_decay < 1.0):
            raise ValueError(f"ema_decay must be in [0, 1), got {ema_decay}")
        self.save_hyperparameters()
        self.latent_dim = latent_dim

        self.encoder = ConvEncoder(
            num_channels=num_channels,
            input_size=input_size,
            base_channels=base_channels,
            latent_dim=latent_dim,
            proj_hidden=proj_hidden,
            activation=activation,
        )
        # EMA teacher: identical architecture, no grad, updated as a slow copy of
        # the online encoder after every training batch.
        self.target_encoder: ConvEncoder | None
        if target == "ema":
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_encoder.requires_grad_(False)
            self.target_encoder.eval()
        else:
            self.target_encoder = None

        if predictor == "linear":
            self.predictor: nn.Module = LinearPredictor(
                latent_dim=latent_dim,
                n_control=n_control_lags,
            )
        elif predictor == "residual_mlp":
            self.predictor = ResidualPredictor(
                latent_dim=latent_dim,
                n_control=n_control_lags,
                hidden=predictor_hidden,
                n_layers=predictor_layers,
                activation=activation,
            )
        else:
            raise ValueError(
                f"unknown predictor {predictor!r}; expected 'linear' or 'residual_mlp'"
            )
        # Optional macrostate anchor. Created only when observables are available
        # so the model can be turned into an "anchored" variant with one flag; it
        # contributes to the loss only when ``w_readout > 0``.
        self.readout = nn.Linear(latent_dim, n_obs) if n_obs > 0 else None
        # Set by :meth:`freeze_encoder_for_phase2` (two-phase training). Also an
        # ``__init__`` hparam so checkpoints restore the freeze on load.
        self.encoder_frozen: bool = False
        if encoder_frozen:
            self.freeze_encoder_for_phase2(zero_vicreg=False)

    # ------------------------------------------------------------------
    # Encode / dynamics
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Online encoder (used at inference / MPC and for rollout context)."""
        return self.encoder(x)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        """Target-side encoding: EMA teacher if present, else stop-grad online."""
        if self.target_encoder is None:
            return self.encoder(x)
        return self.target_encoder(x)

    def freeze_encoder_for_phase2(self, *, zero_vicreg: bool = True) -> None:
        """Freeze the encoder (and EMA teacher) so only the predictor trains.

        Used by two-phase training: after a representation-learning phase, lock
        ``z`` and fit dynamics. Optionally zeros VICReg weights so ``val_loss``
        tracks predictive loss alone during phase 2.
        """
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        if self.target_encoder is not None:
            self.target_encoder.requires_grad_(False)
            self.target_encoder.eval()
        if self.readout is not None:
            self.readout.requires_grad_(False)
        if zero_vicreg:
            # AttributeDict from save_hyperparameters; mutate so logged total loss
            # is not dominated by a frozen encoder's VICReg terms.
            self.hparams.w_vic_var = 0.0
            self.hparams.w_vic_cov = 0.0
        self.encoder_frozen = True
        self.hparams.encoder_frozen = True

    @torch.no_grad()
    def load_linear_predictor(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        """Copy a ridge-fit ``(A, B, c)`` into the linear predictor weights."""
        if not self.is_linear:
            raise TypeError(
                "load_linear_predictor() requires predictor='linear'; "
                f"this model uses {type(self.predictor).__name__}"
            )
        a_t = torch.as_tensor(a, dtype=self.predictor.A.weight.dtype)
        b_t = torch.as_tensor(b, dtype=self.predictor.B.weight.dtype)
        c_t = torch.as_tensor(c, dtype=self.predictor.A.bias.dtype)
        self.predictor.A.weight.copy_(a_t.to(self.predictor.A.weight.device))
        self.predictor.B.weight.copy_(b_t.to(self.predictor.B.weight.device))
        self.predictor.A.bias.copy_(c_t.to(self.predictor.A.bias.device))

    @torch.no_grad()
    def _update_ema_target(self) -> None:
        """``θ_tgt ← τ θ_tgt + (1-τ) θ_online``."""
        if self.target_encoder is None or self.encoder_frozen:
            return
        tau = float(self.hparams.ema_decay)
        for p_online, p_tgt in zip(
            self.encoder.parameters(), self.target_encoder.parameters(), strict=True
        ):
            p_tgt.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:  # noqa: ARG002
        self._update_ema_target()

    def train(self, mode: bool = True):
        # Lightning flips the whole module to train(); keep the EMA teacher in
        # eval so dropout/BN (if added later) do not update from target passes.
        # After phase-2 freeze, keep the online encoder in eval as well.
        super().train(mode)
        if self.target_encoder is not None:
            self.target_encoder.eval()
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def step(self, z: torch.Tensor, u_feat: torch.Tensor) -> torch.Tensor:
        """One-step latent dynamics under control history ``u_feat``."""
        return self.predictor(z, u_feat)

    @property
    def is_linear(self) -> bool:
        return isinstance(self.predictor, LinearPredictor)

    @torch.no_grad()
    def dynamics_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The learned ``(A, B, c)`` of ``z' = A z + B u + c``.

        Only defined for ``predictor="linear"``. Enables spectral-radius,
        controllability and LQR analysis directly on the trained model instead of
        on a post-hoc least-squares surrogate.
        """
        if not self.is_linear:
            raise TypeError(
                "dynamics_matrices() requires predictor='linear'; "
                f"this model uses {type(self.predictor).__name__}"
            )
        return (
            self.predictor.A.weight.detach().cpu().numpy(),
            self.predictor.B.weight.detach().cpu().numpy(),
            self.predictor.A.bias.detach().cpu().numpy(),
        )

    @torch.no_grad()
    def spectral_radius(self) -> float | None:
        """Largest ``|eigenvalue|`` of ``A``; ``None`` for a nonlinear predictor.

        Values above 1 mean free-running rollouts diverge, so this is worth
        watching during training.
        """
        if not self.is_linear:
            return None
        a, _, _ = self.dynamics_matrices()
        return float(np.max(np.abs(np.linalg.eigvals(a))))

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def _control_features(self, controls: torch.Tensor, k: int) -> torch.Tensor:
        """History ``[u_{k+1}, u_k, ...]`` for the transition ``k -> k+1``.

        ``controls`` is ``(B, H+1)``; index ``k+1`` is the control applied on the
        transition and earlier indices are the lagged (delayed-actuator) inputs.
        """
        lags = int(self.hparams.n_control_lags)
        feats = []
        for j in range(lags):
            idx = k + 1 - j
            col = controls[:, idx] if idx >= 0 else torch.zeros_like(controls[:, 0])
            feats.append(col)
        return torch.stack(feats, dim=-1)

    @staticmethod
    def _vicreg(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Variance (hinge to std>=1) + off-diagonal covariance penalty.

        ``z`` is ``(N, d)`` with ``N`` pooled over batch *and* time.
        """
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1.0 - std))
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / max(1, z.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = (off_diag**2).sum() / z.shape[1]
        return var_loss, cov_loss

    @staticmethod
    def _participation_ratio(z: torch.Tensor) -> float:
        """Smooth count of how many latent dimensions are genuinely used.

        ``(sum lambda)^2 / sum(lambda^2)`` of the covariance eigenvalues. Detects
        partial collapse that VICReg's per-feature variance term can miss.
        """
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / max(1, z.shape[0] - 1)
        # For PSD cov: sum(λ)=tr(cov), sum(λ²)=||cov||_F² — avoids eigvalsh (missing on MPS).
        total = cov.diag().sum().clamp_min(0.0)
        if total <= 0:
            return 0.0
        return float(total**2 / (cov * cov).sum().clamp_min(1e-12))

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        frames, controls, obs = batch  # (B,H+1,C,W,H), (B,H+1), (B,H+1,K)
        b, hp1 = frames.shape[0], frames.shape[1]
        h = hp1 - 1

        flat = frames.reshape(b * hp1, *frames.shape[2:])
        # Online encodings: context for the rollout + VICReg. Gradients flow here.
        z_all = self.encode(flat).reshape(b, hp1, self.latent_dim)
        # Target encodings: EMA teacher (no grad) or stop-grad copy of online.
        if self.target_encoder is not None:
            with torch.no_grad():
                z_tgt = self.target_encoder(flat).reshape(b, hp1, self.latent_dim)
        else:
            z_tgt = z_all.detach()

        z = z_all[:, 0]
        pred_loss = torch.zeros((), device=frames.device)
        for k in range(h):
            u_feat = self._control_features(controls, k)
            z = self.step(z, u_feat)
            pred_loss = pred_loss + F.mse_loss(z, z_tgt[:, k + 1])
        pred_loss = pred_loss / max(1, h)

        z_pooled = z_all.reshape(b * hp1, self.latent_dim)
        var_loss, cov_loss = self._vicreg(z_pooled)

        total = (
            self.hparams.w_pred * pred_loss
            + self.hparams.w_vic_var * var_loss
            + self.hparams.w_vic_cov * cov_loss
        )

        readout_loss = torch.zeros((), device=frames.device)
        if self.readout is not None and self.hparams.w_readout > 0:
            pred_obs = self.readout(z_pooled)
            target = obs.reshape(b * hp1, -1)
            # Predict *standardized* targets. Stats are stop-grad and std is
            # floored: a previous version divided by batch std (~0 on low-
            # diversity val batches) and produced ~1e14 val_readout, which then
            # dominated checkpointing whenever w_readout > 0.
            with torch.no_grad():
                mu = target.mean(dim=0, keepdim=True)
                sd = target.std(dim=0, keepdim=True).clamp_min(1e-2)
            readout_loss = F.mse_loss(pred_obs, (target - mu) / sd)
            total = total + self.hparams.w_readout * readout_loss

        log = dict(on_step=False, on_epoch=True, batch_size=b)
        self.log(f"{stage}_loss", total, prog_bar=True, **log)
        self.log(f"{stage}_pred", pred_loss, **log)
        self.log(f"{stage}_vic_var", var_loss, **log)
        self.log(f"{stage}_vic_cov", cov_loss, **log)
        if self.readout is not None and self.hparams.w_readout > 0:
            self.log(f"{stage}_readout", readout_loss, **log)
        return total

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "val")
        if batch_idx == 0:
            frames = batch[0]
            b, hp1 = frames.shape[0], frames.shape[1]
            with torch.no_grad():
                z = self.encode(frames.reshape(b * hp1, *frames.shape[2:]))
            self.log("participation_ratio", self._participation_ratio(z), prog_bar=True)
            radius = self.spectral_radius()
            if radius is not None:
                self.log("spectral_radius", radius, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        # Exclude EMA teacher weights (they are updated by polyak averaging only).
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Post-hoc linear diagnostic (NOT used to train or control)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def linear_diagnostic(self, z: np.ndarray, u_feat: np.ndarray, z_next: np.ndarray) -> dict:
        """Least-squares linear fit ``z_{t+1} ~ A z_t + B u_t + c`` in the latent.

        This is a reference operator fit directly to the frozen encodings, never
        used to train or control. With ``predictor="residual_mlp"`` it measures how
        (non)linear the discovered latent is; with the default linear predictor it
        becomes an *optimality* check -- the trained ``A``/``B`` should land close
        to this least-squares solution, and a large gap means the predictor is
        under-trained rather than the latent being at fault.
        """
        d = z.shape[1]
        x = np.concatenate([z, u_feat, np.ones((z.shape[0], 1))], axis=1)
        gram = x.T @ x + 1e-6 * np.eye(x.shape[1])
        theta = np.linalg.solve(gram, x.T @ z_next)
        pred = x @ theta
        ss_res = float(((z_next - pred) ** 2).sum())
        ss_tot = float(((z_next - z_next.mean(0)) ** 2).sum())
        a = theta[:d].T
        return {
            "A": a,
            "B": theta[d : d + u_feat.shape[1]].T,
            "one_step_r2": 1.0 - ss_res / max(ss_tot, 1e-12),
            "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(a)))),
        }
