import torch
from torch import nn
import pytorch_lightning as pl

from koopman_control.utils.component_mappings import ACTIVATIONS

class KoopmanCNN(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int = 64,
        lr: float = 1e-3,
        latent_dim: int = 8,
        activation: str = "relu",
        num_channels: int = 1,
        input_size: int = 64,
        beta: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.activation_fn = ACTIVATIONS[activation]

        self.latent_spatial = input_size // 8
        self.latent_channels = latent_dim
        self.latent_dim_total = self.latent_channels * self.latent_spatial * self.latent_spatial

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

        self.K = nn.Conv2d(
            self.latent_channels,
            self.latent_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

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
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        z_next = self.linear_dynamics(z)
        x_recon = self.decode(z_next)
        return x_recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image batch to the spatial latent ``(B, lat_c, h, w)``."""
        return self.encoder(x)

    def encode_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: flatten the spatial latent to ``(B, lat_c*h*w)`` for
        downstream visualization / analysis (e.g., PCA, t-SNE)."""
        z = self.encode(x)
        return z.flatten(start_dim=1)

    def linear_dynamics(self, z: torch.Tensor) -> torch.Tensor:
        return self.K(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @torch.no_grad()
    def koopman_matrix(self) -> torch.Tensor:
        D = self.latent_dim_total
        device = next(self.parameters()).device
        eye = torch.eye(D, device=device).view(D, self.latent_channels, self.latent_spatial, self.latent_spatial)
        out = self.K(eye)  # (D, latent_channels, latent_spatial, latent_spatial)
        return out.view(D, D).T  # columns = K applied to basis vectors

    def _step(self, batch, stage: str) -> torch.Tensor:
        x_0, x_1, _ = batch
        z_0 = self.encode(x_0)
        z_1 = self.encode(x_1)

        x_0_recon = self.decode(z_0)
        x_1_recon = self.decode(z_1)
        recon_loss = self.criterion(x_0, x_0_recon) + self.criterion(x_1, x_1_recon)

        z_1_pred = self.linear_dynamics(z_0)
        koopman_loss = self.criterion(z_1_pred, z_1)

        total_loss = recon_loss + self.hparams.beta * koopman_loss

        bs = x_0.shape[0]
        prog = stage == "val"
        self.log(f"{stage}_recon_loss", recon_loss, prog_bar=prog, batch_size=bs)
        self.log(f"{stage}_koopman_loss", koopman_loss, prog_bar=prog, batch_size=bs)
        self.log(f"{stage}_loss", total_loss, prog_bar=prog, batch_size=bs)
        return total_loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
