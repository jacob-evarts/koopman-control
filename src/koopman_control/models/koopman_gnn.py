import torch
from torch import nn
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from koopman_control.utils.component_mappings import ACTIVATIONS


class KoopmanGNN(pl.LightningModule):
    def __init__(self, node_input_dim: int, hidden_size: int = 128, lr: float = 1e-3, latent_dim: int = 32, activation: str = "relu", num_gnn_layers: int = 1,  beta: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        self.activation_fn = ACTIVATIONS[activation]

        graph_conv_layers: list[nn.Module] = []
        in_dim = node_input_dim
        for _ in range(self.hparams.num_gnn_layers):
            graph_conv_layers.append(GCNConv(in_dim, hidden_size))
            in_dim = hidden_size
        self.graph_conv_layers = nn.ModuleList(graph_conv_layers)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.activation_fn(),
            nn.Linear(hidden_size, latent_dim),
        )
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_size),
            self.activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            self.activation_fn(),
            nn.Linear(hidden_size, node_input_dim),
        )
        self.criterion = nn.MSELoss()   

    def on_train_start(self) -> None:
        from pytorch_lightning.utilities.rank_zero import rank_zero_info

        dev = next(self.parameters()).device
        rank_zero_info(f"{self.__class__.__name__}: module parameters are on {dev}")

    def encode(self, batch: Batch) -> torch.Tensor:
        x, edge_index, b = batch.x, batch.edge_index, batch.batch
        h = x
        for conv in self.graph_conv_layers:
            h = self.activation_fn()(conv(h, edge_index))
        h = global_mean_pool(h, b)
        return self.projection_head(h)

    def linear_dynamics(self, z: torch.Tensor) -> torch.Tensor:
        return self.K(z)

    def decode(self, z: torch.Tensor, batch: Batch) -> torch.Tensor:
        """z: (B, latent); expand per node using batch.batch and concat pos."""
        z_nodes = z[batch.batch]
        h = torch.cat([z_nodes, batch.pos], dim=-1)
        return self.node_decoder(h)

    def training_step(self, batch, _):
        batch_t, batch_tp, _meta = batch
        z_0 = self.encode(batch_t)
        z_1 = self.encode(batch_tp)

        x_0_hat = self.decode(z_0, batch_t)
        x_1_hat = self.decode(z_1, batch_tp)

        recon = self.criterion(batch_t.x, x_0_hat) + self.criterion(batch_tp.x, x_1_hat)
        z_1_pred = self.linear_dynamics(z_0)
        koop = self.criterion(z_1_pred, z_1)
        total = recon + self.hparams.beta * koop

        bs = int(batch_t.num_graphs)
        self.log("train_recon_loss", recon, batch_size=bs)
        self.log("train_koopman_loss", koop, batch_size=bs)
        self.log("train_loss", total, batch_size=bs)
        return total

    def validation_step(self, batch, _):
        batch_t, batch_tp, _meta = batch
        z_0 = self.encode(batch_t)
        z_1 = self.encode(batch_tp)

        x_0_hat = self.decode(z_0, batch_t)
        x_1_hat = self.decode(z_1, batch_tp)

        recon = self.criterion(batch_t.x, x_0_hat) + self.criterion(batch_tp.x, x_1_hat)
        z_1_pred = self.linear_dynamics(z_0)
        koop = self.criterion(z_1_pred, z_1)
        total = recon + self.hparams.beta * koop

        bs = int(batch_t.num_graphs)
        self.log("val_recon_loss", recon, prog_bar=True, batch_size=bs)
        self.log("val_koopman_loss", koop, prog_bar=True, batch_size=bs)
        self.log("val_loss", total, prog_bar=True, batch_size=bs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
