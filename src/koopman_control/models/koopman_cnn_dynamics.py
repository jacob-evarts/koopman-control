from __future__ import annotations

import torch
from torch import nn
import pytorch_lightning as pl

from koopman_control.utils.component_mappings import ACTIVATIONS


class _SpatialPoolToLatent(nn.Module):
    """Global avg + max pool over spatial features, then linear map to ``z``.

    Max pooling retains sparse rabbit occupancy; mean pooling captures global density.
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


class KoopmanCNNDynamics(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int = 64,
        lr: float = 1e-3,
        latent_dim: int = 64,
        activation: str = "relu",
        num_channels: int = 1,
        input_size: int = 64,
        spatial_latent_channels: int = 16,
        beta_koop: float = 5.0,
        beta_pred: float = 5.0,
        beta_recon: float = 0.2,
        rollout_horizon: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.activation_fn = ACTIVATIONS[activation]
        self.latent_spatial = input_size // 8
        self.latent_channels = spatial_latent_channels
        self.latent_dim = latent_dim
        self.rollout_horizon = max(1, int(rollout_horizon))
        self.criterion = nn.MSELoss()

        c1 = max(8, hidden_size // 4)
        c2 = max(16, hidden_size // 2)
        c3 = max(32, hidden_size)

        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, c1, kernel_size=3, stride=2, padding=1),
            self.activation_fn(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            self.activation_fn(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            self.activation_fn(),
            nn.Conv2d(c3, self.latent_channels, kernel_size=1, stride=1, padding=0),
        )

        spatial_flat = self.latent_channels * self.latent_spatial * self.latent_spatial
        self.to_latent = _SpatialPoolToLatent(self.latent_channels, latent_dim)
        self.to_spatial = nn.Sequential(
            nn.Linear(latent_dim, spatial_flat),
            self.activation_fn(),
        )

        self.dynamics = nn.Linear(latent_dim, latent_dim, bias=False)
        self.control_in = nn.Linear(1, latent_dim, bias=True)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.latent_channels, c3, kernel_size=1, stride=1, padding=0),
            self.activation_fn(),
            nn.ConvTranspose2d(c3, c2, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.activation_fn(),
            nn.ConvTranspose2d(c2, c1, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.activation_fn(),
            nn.ConvTranspose2d(c1, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    # Encode / decode / dynamics
    # ------------------------------------------------------------------

    def encode_spatial(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def spatial_to_latent(self, z_spatial: torch.Tensor) -> torch.Tensor:
        return self.to_latent(z_spatial)

    def latent_to_spatial(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        flat = self.to_spatial(z)
        return flat.view(b, self.latent_channels, self.latent_spatial, self.latent_spatial)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Vector latent ``(B, latent_dim)``."""
        return self.spatial_to_latent(self.encode_spatial(x))

    def encode_flat(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def decode(self, z_spatial: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_spatial)

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(self.latent_to_spatial(z))

    def step_dynamics(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """``z``: (B, d); ``u``: (B, 1) or (B,) cull intensity in [0, 1]."""
        if u.dim() == 1:
            u = u.unsqueeze(-1)
        return self.dynamics(z) + self.control_in(u)

    def linear_dynamics(self, z_spatial: torch.Tensor) -> torch.Tensor:
        """Spatial-map dynamics without control (for compatibility / inspection)."""
        z = self.spatial_to_latent(z_spatial)
        z_next = self.dynamics(z)
        return self.latent_to_spatial(z_next)

    @torch.no_grad()
    def koopman_matrix(self) -> torch.Tensor:
        return self.dynamics.weight.T.clone()

    @torch.no_grad()
    def control_matrix(self) -> torch.Tensor:
        return self.control_in.weight.clone()

    def forward(self, x: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encode(x)
        u_zero = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype) if u is None else u
        z_next = self.step_dynamics(z, u_zero)
        return self.decode_latent(z_next)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _pixel_losses(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        total = self.criterion(pred, target)
        return {
            "total": total,
            "rabbit": (pred[:, 0] - target[:, 0]).pow(2).mean(),
        }

    def _parse_batch(self, batch):
        if len(batch) == 4:
            x_0, x_1, u, meta = batch
        else:
            x_0, x_1, meta = batch
            u = torch.zeros(x_0.shape[0], 1, device=x_0.device, dtype=x_0.dtype)

        if u.dim() == 1:
            u = u.unsqueeze(-1)
        return x_0, x_1, u, meta

    def _one_step_losses(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        u: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z_0 = self.encode(x_0)
        z_1 = self.encode(x_1)
        z_1_pred = self.step_dynamics(z_0, u)
        x_1_pred = self.decode_latent(z_1_pred)

        loss_koop = self.criterion(z_1_pred, z_1)
        pred_px = self._pixel_losses(x_1_pred, x_1)
        recon_px = self._pixel_losses(self.decode_latent(z_0), x_0)

        return {
            "koop": loss_koop,
            "pred": pred_px["total"],
            "recon": recon_px["total"],
            "pred_rabbit": pred_px["rabbit"],
            "recon_rabbit": recon_px["rabbit"],
        }

    def _multi_step_losses(
        self,
        x_seq: torch.Tensor,
        u_seq: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """``x_seq``: (B, H+1, C, H, W); ``u_seq``: (B, H)."""
        b, horizon_p1 = x_seq.shape[0], x_seq.shape[1]
        h = horizon_p1 - 1
        z = self.encode(x_seq[:, 0])
        loss_koop = torch.tensor(0.0, device=x_seq.device)
        loss_pred = torch.tensor(0.0, device=x_seq.device)

        for k in range(h):
            u_k = u_seq[:, k].unsqueeze(-1)
            z_next = self.step_dynamics(z, u_k)
            z_tgt = self.encode(x_seq[:, k + 1])
            loss_koop = loss_koop + self.criterion(z_next, z_tgt)
            loss_pred = loss_pred + self._pixel_losses(
                self.decode_latent(z_next), x_seq[:, k + 1]
            )["total"]
            z = z_next

        recon_px = self._pixel_losses(
            self.decode_latent(self.encode(x_seq[:, 0])), x_seq[:, 0]
        )
        return {
            "koop": loss_koop / h,
            "pred": loss_pred / h,
            "recon": recon_px["total"],
            "pred_rabbit": None,
            "recon_rabbit": recon_px["rabbit"],
        }

    def _step(self, batch, stage: str) -> torch.Tensor:
        if len(batch) == 3 and batch[0].dim() == 5:
            x_seq, u_seq, _meta = batch
            losses = self._multi_step_losses(x_seq, u_seq)
        else:
            x_0, x_1, u, _meta = self._parse_batch(batch)
            losses = self._one_step_losses(x_0, x_1, u)

        total = (
            self.hparams.beta_koop * losses["koop"]
            + self.hparams.beta_pred * losses["pred"]
            + self.hparams.beta_recon * losses["recon"]
        )

        bs = batch[0].shape[0]
        prog = stage == "val"
        self.log(f"{stage}_koop_loss", losses["koop"], prog_bar=prog, batch_size=bs)
        self.log(f"{stage}_pred_loss", losses["pred"], prog_bar=prog, batch_size=bs)
        self.log(f"{stage}_recon_loss", losses["recon"], prog_bar=prog, batch_size=bs)
        if losses.get("pred_rabbit") is not None:
            self.log(f"{stage}_pred_rabbit", losses["pred_rabbit"], prog_bar=prog, batch_size=bs)
        if losses.get("recon_rabbit") is not None:
            self.log(f"{stage}_recon_rabbit", losses["recon_rabbit"], prog_bar=False, batch_size=bs)
        self.log(f"{stage}_loss", total, prog_bar=prog, batch_size=bs)
        return total

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
