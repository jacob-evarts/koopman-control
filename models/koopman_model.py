import torch
from torch import nn
import pytorch_lightning as pl

from utils.component_mappings import ACTIVATIONS

class KoopmanAE(pl.LightningModule):
    def __init__(self, 
                 hidden_size=128, 
                 lr=1e-3, 
                 latent_dim=32, 
                 activation="relu",
                ):
        super().__init__()
        self.save_hyperparameters()

        self.activation_fn = ACTIVATIONS[activation]
        
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
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
            nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1),
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
        x_0, x_1 = batch
        x_pred = self(x_0)
        loss = self.criterion(x_pred, x_1)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x_0, x_1 = batch
        x_pred = self(x_0)
        loss = self.criterion(x_pred, x_1)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)