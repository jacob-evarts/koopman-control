import torch
from torch import nn
import pytorch_lightning as pl

from koopman_control.utils.component_mappings import ACTIVATIONS

class KoopmanCNN(pl.LightningModule):
    def __init__(self, hidden_size=128, lr=1e-3, latent_dim=32, activation="relu", num_channels=1):
        super().__init__()
        self.save_hyperparameters()

        self.activation_fn = ACTIVATIONS[activation]
        self.num_channels = num_channels
        
        self.encoder = nn.Sequential(
            nn.Conv2d(self.num_channels, 16, kernel_size=3, stride=2, padding=1),
            self.activation_fn(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            self.activation_fn(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            self.activation_fn(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, hidden_size),
            self.activation_fn(),
            nn.Linear(hidden_size, latent_dim)
        )

        self.K = nn.Linear(latent_dim, latent_dim, bias=False)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            self.activation_fn(),
            nn.Linear(hidden_size, 64 * 8 * 8),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.activation_fn(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.activation_fn(),
            nn.ConvTranspose2d(16, self.num_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        z = self.encode(x)
        z_next = self.linear_dynamics(z)
        x_recon = self.decode(z_next)
        return x_recon
    
    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def linear_dynamics(self, z):
        return self.K(z)
    
    def decode(self, z):
        return self.decoder(z)

    def training_step(self, batch, _):
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
        self.log("train_recon_loss", recon_loss, batch_size=bs)
        self.log("train_koopman_loss", koopman_loss, batch_size=bs)
        self.log("train_loss", total_loss, batch_size=bs)
        return total_loss

    def validation_step(self, batch, _):
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
        self.log("val_recon_loss", recon_loss, prog_bar=True, batch_size=bs)
        self.log("val_koopman_loss", koopman_loss, prog_bar=True, batch_size=bs)
        self.log("val_loss", total_loss, prog_bar=True, batch_size=bs)
        return

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)