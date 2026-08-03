"""Hybrid locally-linear latent world model for image-based control.

Why this model
--------------
Goal: encode an image to a latent ``z``, evolve it under a control ``u``, and
control it with standard tools. Two competing pressures shape the design:

  * **Koopman / linear control** wants ``z_{t+1} = A z_t + B u_t`` so LQR/MPC and
    controllability analysis apply directly.
  * **JEPA-style world models** predict in latent space with a (possibly
    nonlinear) action-conditioned predictor and avoid pixel reconstruction to
    prevent the latent from wasting capacity / collapsing.

This module takes the hybrid path requested during planning: a **linear core**
``A z + B u`` plus an **optional nonlinear residual**. The same forward pass
evaluates the linear-only prediction and the full prediction, so the gap between
them is logged every step as a direct measure of how nonlinear the system is in
this latent -- the quantity that tells you whether linear control tools are
justified.

Losses (and why each is here)
------------------------------
  * ``latent`` : multi-step latent-prediction loss against **stop-grad** target
    embeddings (JEPA-style). Trained over a horizon so the dynamics stay
    accurate when rolled out, not just one step.
  * ``recon``  : low-weight image decode of the encoded and predicted latents.
    Grounds the latent so it stays decodable/renderable (needed to *see* a
    controlled rollout) and provides an anti-collapse signal.
  * ``vic``    : VICReg variance + covariance regularization on the batch of
    embeddings, preventing dimensional / full collapse that the stop-grad
    latent loss alone could allow.

Actuator lag: the control feature for each transition is the history
``[u_now, u_prev]`` (see :mod:`koopman_control.data.dataset`).
"""

from __future__ import annotations

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


class _SpatialPoolToLatent(nn.Module):
    """Global avg + max pool over spatial features, then linear map to ``z``.

    Max pooling retains sparse agent occupancy; mean pooling captures global
    density. Concatenating both gives the latent access to each.
    """

    def __init__(self, in_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.proj = nn.Linear(2 * in_channels, latent_dim)

    def forward(self, z_spatial: torch.Tensor) -> torch.Tensor:
        b = z_spatial.shape[0]
        z_avg = self.avg_pool(z_spatial).view(b, -1)
        z_max = self.max_pool(z_spatial).view(b, -1)
        return self.proj(torch.cat([z_avg, z_max], dim=1))


class LatentWorldModel(pl.LightningModule):
    def __init__(
        self,
        num_channels: int = 2,
        input_size: int = 64,
        hidden_size: int = 64,
        spatial_latent_channels: int = 16,
        latent_dim: int = 32,
        activation: str = "relu",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_patience: int = 4,
        lr_factor: float = 0.5,
        min_lr: float = 1e-6,
        horizon: int = 8,
        n_control_lags: int = 2,
        dynamics_mode: str = "bilinear",  # "linear" | "bilinear" | "mlp"
        w_latent: float = 1.0,
        w_recon: float = 0.2,
        w_vic_var: float = 1.0,
        w_vic_cov: float = 0.04,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if dynamics_mode not in ("linear", "bilinear", "mlp"):
            raise ValueError(f"dynamics_mode must be linear/bilinear/mlp, got {dynamics_mode!r}")

        act = ACTIVATIONS[activation]
        self.latent_spatial = input_size // 8
        self.latent_channels = spatial_latent_channels
        self.latent_dim = latent_dim

        c1, c2, c3 = max(8, hidden_size // 4), max(16, hidden_size // 2), max(32, hidden_size)

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, c1, 3, stride=2, padding=1),
            act(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            act(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            act(),
            nn.Conv2d(c3, self.latent_channels, 1),
        )
        self.to_latent = _SpatialPoolToLatent(self.latent_channels, latent_dim)

        spatial_flat = self.latent_channels * self.latent_spatial * self.latent_spatial
        self.to_spatial = nn.Sequential(nn.Linear(latent_dim, spatial_flat), act())
        self.decoder = nn.Sequential(
            nn.Conv2d(self.latent_channels, c3, 1),
            act(),
            nn.ConvTranspose2d(c3, c2, 3, stride=2, padding=1, output_padding=1),
            act(),
            nn.ConvTranspose2d(c2, c1, 3, stride=2, padding=1, output_padding=1),
            act(),
            nn.ConvTranspose2d(c1, num_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

        # Dynamics: linear core (A, B) plus an optional nonlinear residual.
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(n_control_lags, latent_dim, bias=True)
        if dynamics_mode == "bilinear":
            # Control gates a linear map of the state (state-dependent input gain).
            self.C = nn.Linear(latent_dim, latent_dim, bias=False)
        elif dynamics_mode == "mlp":
            self.res_mlp = nn.Sequential(
                nn.Linear(latent_dim + n_control_lags, hidden_size),
                act(),
                nn.Linear(hidden_size, latent_dim),
            )

    # ------------------------------------------------------------------
    # Encode / decode / dynamics
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        flat = self.to_spatial(z)
        spatial = flat.view(b, self.latent_channels, self.latent_spatial, self.latent_spatial)
        return self.decoder(spatial)

    def linear_step(self, z: torch.Tensor, u_feat: torch.Tensor) -> torch.Tensor:
        """Linear core ``A z + B u`` (u_feat holds the control history)."""
        return self.A(z) + self.B(u_feat)

    def step(self, z: torch.Tensor, u_feat: torch.Tensor) -> torch.Tensor:
        """Full one-step latent dynamics (linear core + optional residual)."""
        z_lin = self.linear_step(z, u_feat)
        mode = self.hparams.dynamics_mode
        if mode == "linear":
            return z_lin
        if mode == "bilinear":
            u_now = u_feat[:, :1]  # instantaneous control gates the state map
            return z_lin + u_now * self.C(z)
        return z_lin + self.res_mlp(torch.cat([z, u_feat], dim=-1))

    # ------------------------------------------------------------------
    # Training
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

    def _vicreg(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Variance (hinge to std>=1) + off-diagonal covariance penalty."""
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1.0 - std))
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / max(1, z.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = (off_diag**2).sum() / z.shape[1]
        return var_loss, cov_loss

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        frames, controls = batch  # (B, H+1, C, W, H), (B, H+1)
        b, hp1 = frames.shape[0], frames.shape[1]
        h = hp1 - 1

        # Encode every frame; targets for the latent loss are stop-grad.
        flat = frames.reshape(b * hp1, *frames.shape[2:])
        z_all = self.encode(flat).reshape(b, hp1, self.latent_dim)
        z_tgt = z_all.detach()

        z = z_all[:, 0]
        z_lin = z_all[:, 0]
        latent_loss = torch.zeros((), device=frames.device)
        latent_loss_lin = torch.zeros((), device=frames.device)
        pred_pix_loss = torch.zeros((), device=frames.device)

        for k in range(h):
            u_feat = self._control_features(controls, k)
            z = self.step(z, u_feat)
            z_lin = self.linear_step(z_lin.detach(), u_feat)  # linear-only reference
            latent_loss = latent_loss + F.mse_loss(z, z_tgt[:, k + 1])
            latent_loss_lin = latent_loss_lin + F.mse_loss(z_lin, z_tgt[:, k + 1])
            pred_pix_loss = pred_pix_loss + F.binary_cross_entropy(self.decode(z), frames[:, k + 1])

        latent_loss = latent_loss / h
        latent_loss_lin = latent_loss_lin / h
        pred_pix_loss = pred_pix_loss / h
        recon0 = F.binary_cross_entropy(self.decode(z_all[:, 0]), frames[:, 0])
        var_loss, cov_loss = self._vicreg(z_all[:, 0])

        total = (
            self.hparams.w_latent * latent_loss
            + self.hparams.w_recon * (pred_pix_loss + recon0)
            + self.hparams.w_vic_var * var_loss
            + self.hparams.w_vic_cov * cov_loss
        )

        log = dict(on_step=False, on_epoch=True, batch_size=b, prog_bar=False)
        self.log(
            f"{stage}_loss",
            total,
            prog_bar=True,
            **{k: v for k, v in log.items() if k != "prog_bar"},
        )
        self.log(f"{stage}_latent", latent_loss, **log)
        self.log(f"{stage}_latent_linear", latent_loss_lin, **log)
        # Positive => the nonlinear residual helps; ~0 => system is linear here.
        self.log(f"{stage}_linearity_gap", latent_loss_lin - latent_loss, **log)
        self.log(f"{stage}_pred_pix", pred_pix_loss, **log)
        self.log(f"{stage}_recon", recon0, **log)
        self.log(f"{stage}_vic_var", var_loss, **log)
        self.log(f"{stage}_vic_cov", cov_loss, **log)
        return total

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        self._shared_step(batch, "val")
        if _ == 0:
            spec, rank = self.linear_diagnostics()
            self.log("spectral_radius", spec, on_epoch=True, prog_bar=True)
            self.log("controllability_rank", float(rank), on_epoch=True)

    def test_step(self, batch, _):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
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
    # Diagnostics / control interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def linear_diagnostics(self) -> tuple[float, int]:
        """Spectral radius of A and controllability rank of ``(A, b_now)``."""
        a = self.A.weight.detach().cpu().numpy()
        b = self.B.weight.detach().cpu().numpy()[:, 0:1]  # instantaneous input column
        spec = float(np.max(np.abs(np.linalg.eigvals(a))))
        blocks = [b]
        for _ in range(1, a.shape[0]):
            blocks.append(a @ blocks[-1])
        rank = int(np.linalg.matrix_rank(np.concatenate(blocks, axis=1), tol=1e-6))
        return spec, rank

    @torch.no_grad()
    def linear_system(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(A, B)`` numpy arrays for use by linear controllers (LQR/MPC)."""
        a = self.A.weight.detach().cpu().numpy()
        b = self.B.weight.detach().cpu().numpy()
        return a, b
