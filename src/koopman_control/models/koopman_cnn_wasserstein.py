from __future__ import annotations

import torch

from koopman_control.models.koopman_cnn_dynamics import KoopmanCNNDynamics
from koopman_control.utils.sinkhorn import SinkhornGridLoss


class KoopmanCNNWasserstein(KoopmanCNNDynamics):
    """``KoopmanCNNDynamics`` with Sinkhorn OT reconstruction / prediction loss."""

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
        ot_grid_size: int = 16,
        ot_epsilon: float = 0.05,
        ot_iters: int = 30,
        ot_mass_weight: float = 1.0,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            lr=lr,
            latent_dim=latent_dim,
            activation=activation,
            num_channels=num_channels,
            input_size=input_size,
            spatial_latent_channels=spatial_latent_channels,
            beta_koop=beta_koop,
            beta_pred=beta_pred,
            beta_recon=beta_recon,
            rollout_horizon=rollout_horizon,
        )
        # Capture the full (child) signature so checkpoints round-trip the OT args.
        self.save_hyperparameters()

        self.ot_loss = SinkhornGridLoss(
            grid_size=ot_grid_size,
            epsilon=ot_epsilon,
            n_iters=ot_iters,
            mass_weight=ot_mass_weight,
        )

    def _pixel_losses(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Override: per-channel Sinkhorn OT instead of pixel MSE.

        ``total`` averages OT over channels; ``rabbit`` reports channel 0 for
        diagnostics (same keys the base class logs).
        """
        per_channel = self.ot_loss(pred, target)  # (B, C)
        return {
            "total": per_channel.mean(),
            "rabbit": per_channel[:, 0].mean(),
        }
