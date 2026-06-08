import torch
from torch import nn
import pytorch_lightning as pl
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import scatter

from koopman_control.utils.component_mappings import ACTIVATIONS


def _masked_mean_pool(
    h: torch.Tensor,
    batch: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool node features per graph, restricted to nodes where ``mask`` is true."""
    mask_f = mask.to(h.dtype)
    summed = scatter(h * mask_f.unsqueeze(-1), batch, dim=0, reduce="sum")
    counts = scatter(mask_f, batch, dim=0, reduce="sum")
    pooled = summed / counts.clamp(min=1.0).unsqueeze(-1)
    return pooled * (counts > 0).unsqueeze(-1).to(h.dtype)


class KoopmanGNN(pl.LightningModule):
    def __init__(
        self,
        node_input_dim: int,
        hidden_size: int = 128,
        lr: float = 1e-3,
        latent_dim: int = 32,
        activation: str = "relu",
        num_gnn_layers: int = 1,
        beta: float = 1.0,
        decode_with_pos: bool = True,
        latent_mode: str = "global",
        num_populations: int = 2,
        type_feature_start: int = 7,
        latent_dim_per_type: int | None = None,
        include_control: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        act_cls = ACTIVATIONS[activation]
        self.act = act_cls()

        if latent_mode not in ("global", "per_type"):
            raise ValueError(f"latent_mode must be 'global' or 'per_type', got {latent_mode!r}")

        n_pop = int(num_populations)
        d_per = int(latent_dim_per_type or (latent_dim // n_pop))
        if latent_mode == "per_type" and d_per * n_pop != latent_dim:
            raise ValueError(
                f"latent_dim ({latent_dim}) must equal num_populations ({n_pop}) "
                f"* latent_dim_per_type ({d_per}) in per_type mode"
            )

        graph_conv_layers: list[nn.Module] = []
        in_dim = node_input_dim
        for _ in range(self.hparams.num_gnn_layers):
            graph_conv_layers.append(GCNConv(in_dim, hidden_size))
            in_dim = hidden_size
        self.graph_conv_layers = nn.ModuleList(graph_conv_layers)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_cls(),
            nn.Linear(hidden_size, d_per if latent_mode == "per_type" else latent_dim),
        )
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)
        self.control_in = nn.Linear(1, latent_dim, bias=True)
        decode_in_dim = (d_per if latent_mode == "per_type" else latent_dim) + (
            2 if decode_with_pos else 0
        )
        self.node_decoder = nn.Sequential(
            nn.Linear(decode_in_dim, hidden_size),
            act_cls(),
            nn.Linear(hidden_size, hidden_size),
            act_cls(),
            nn.Linear(hidden_size, node_input_dim),
        )
        self.criterion = nn.MSELoss()

    @property
    def _latent_dim_per_type(self) -> int:
        if self.hparams.latent_mode == "per_type":
            return int(
                self.hparams.latent_dim_per_type
                or (self.hparams.latent_dim // self.hparams.num_populations)
            )
        return int(self.hparams.latent_dim)

    def _node_type_ids(self, x: torch.Tensor) -> torch.Tensor:
        start = int(self.hparams.type_feature_start)
        end = start + int(self.hparams.num_populations)
        return x[:, start:end].argmax(dim=-1)

    def _encode_per_type(self, h: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Return ``(B, num_populations * d_per)`` with one block per population."""
        b = batch.batch
        types = self._node_type_ids(batch.x)
        parts: list[torch.Tensor] = []
        for t in range(int(self.hparams.num_populations)):
            pooled = _masked_mean_pool(h, b, types == t)
            parts.append(self.projection_head(pooled))
        return torch.cat(parts, dim=-1)

    def encode(self, batch: Batch) -> torch.Tensor:
        x, edge_index, b = batch.x, batch.edge_index, batch.batch
        h = x
        for conv in self.graph_conv_layers:
            h = self.act(conv(h, edge_index))
        if self.hparams.latent_mode == "per_type":
            return self._encode_per_type(h, batch)
        h = global_mean_pool(h, b)
        return self.projection_head(h)

    def _expand_latent_to_nodes(self, z: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Map graph latent(s) to per-node vectors for decoding."""
        if self.hparams.latent_mode == "global":
            return z[batch.batch]
        d = self._latent_dim_per_type
        n_pop = int(self.hparams.num_populations)
        z_by_type = z.view(z.shape[0], n_pop, d)
        type_ids = self._node_type_ids(batch.x)
        return z_by_type[batch.batch, type_ids]

    def linear_dynamics(self, z: torch.Tensor) -> torch.Tensor:
        return self.K(z)

    def step_dynamics(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """``z``: (B, d); ``u``: (B, 1) or (B,) cull intensity in [0, 1]."""
        if u.dim() == 1:
            u = u.unsqueeze(-1)
        if self.hparams.include_control:
            return self.K(z) + self.control_in(u)
        return self.K(z)

    def _parse_batch(self, batch):
        if len(batch) == 4:
            batch_t, batch_tp, u, meta = batch
        else:
            batch_t, batch_tp, meta = batch
            u = torch.zeros(
                batch_t.num_graphs,
                1,
                device=batch_t.x.device,
                dtype=batch_t.x.dtype,
            )
        if u.dim() == 1:
            u = u.unsqueeze(-1)
        return batch_t, batch_tp, u, meta

    def decode(self, z: torch.Tensor, batch: Batch) -> torch.Tensor:
        z_nodes = self._expand_latent_to_nodes(z, batch)
        if self.hparams.decode_with_pos:
            h = torch.cat([z_nodes, batch.pos], dim=-1)
        else:
            h = z_nodes
        return self.node_decoder(h)

    def _koopman_loss(
        self,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        z_1_pred = self.step_dynamics(z_0, u)
        loss = self.criterion(z_1_pred, z_1)
        if (
            self.hparams.latent_mode == "per_type"
            and int(self.hparams.num_populations) == 2
        ):
            d = self._latent_dim_per_type
            self.log("koop_loss_pop0", self.criterion(z_1_pred[:, :d], z_1[:, :d]))
            self.log("koop_loss_pop1", self.criterion(z_1_pred[:, d:], z_1[:, d:]))
        return loss

    @torch.no_grad()
    def control_matrix(self) -> torch.Tensor:
        return self.control_in.weight.clone()

    @torch.no_grad()
    def koopman_matrix(self) -> torch.Tensor:
        return self.K.weight.T.clone()

    @torch.no_grad()
    def koopman_blocks(self) -> dict[str, torch.Tensor]:
        """Return K sub-blocks when ``latent_mode=per_type`` (for coupling analysis)."""
        k = self.koopman_matrix()
        if self.hparams.latent_mode != "per_type":
            return {"full": k}
        d = self._latent_dim_per_type
        return {
            "pop0_pop0": k[:d, :d],
            "pop0_pop1": k[:d, d:],
            "pop1_pop0": k[d:, :d],
            "pop1_pop1": k[d:, d:],
        }

    def training_step(self, batch, _):
        batch_t, batch_tp, u, _meta = self._parse_batch(batch)
        z_0 = self.encode(batch_t)
        z_1 = self.encode(batch_tp)

        x_0_hat = self.decode(z_0, batch_t)
        x_1_hat = self.decode(z_1, batch_tp)

        recon = self.criterion(batch_t.x, x_0_hat) + self.criterion(batch_tp.x, x_1_hat)
        koop = self._koopman_loss(z_0, z_1, u)
        total = recon + self.hparams.beta * koop

        bs = int(batch_t.num_graphs)
        self.log("train_loss", total, prog_bar=True, batch_size=bs, on_step=False, on_epoch=True)
        return total

    def validation_step(self, batch, _):
        batch_t, batch_tp, u, _meta = self._parse_batch(batch)
        z_0 = self.encode(batch_t)
        z_1 = self.encode(batch_tp)

        x_0_hat = self.decode(z_0, batch_t)
        x_1_hat = self.decode(z_1, batch_tp)

        recon = self.criterion(batch_t.x, x_0_hat) + self.criterion(batch_tp.x, x_1_hat)
        koop = self._koopman_loss(z_0, z_1, u)
        total = recon + self.hparams.beta * koop

        bs = int(batch_t.num_graphs)
        self.log("val_loss", total, prog_bar=True, batch_size=bs, on_step=False, on_epoch=True)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
