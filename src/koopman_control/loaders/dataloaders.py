"""
Unified data loader API. Dispatches to H5 or CSV loaders based on config.
"""
from dataclasses import dataclass
from pathlib import Path
from omegaconf import DictConfig

from koopman_control.loaders.h5_loaders import get_dataloaders_h5
from koopman_control.loaders.csv_loader import get_dataloaders_csv
from koopman_control.loaders.graph_npz_loaders import get_dataloaders_npz, GraphNpzDataset


@dataclass
class DatasetProps:
    """Properties of the loaded dataset needed to build the model."""
    model_type: str  # "cnn" | "mlp" | "gnn"
    input_dim: int | None = None  # for MLP
    num_channels: int | None = None  # for CNN
    node_input_dim: int | None = None  # for GNN (.npz graph node features)


def _infer_format(dataset_cfg: DictConfig) -> str:
    """Infer dataset format from config. Prefer explicit format, else derive from csv_file."""
    if hasattr(dataset_cfg, "format") and dataset_cfg.format is not None:
        return dataset_cfg.format
    csv_file = getattr(dataset_cfg, "csv_file", None)
    return "csv" if csv_file else "h5"


def get_dataloaders(dataset_cfg: DictConfig):
    """
    Return (train_loader, val_loader, test_loader, dataset_props) from dataset config.

    dataset_cfg should have:
      - format: "h5" | "csv" | "npz" (optional; inferred from csv_file if missing, unless npz)
      - data_dir, batch_size, dataset_name
      - csv_file: required when format is "csv"
    """
    fmt = _infer_format(dataset_cfg)
    data_dir = dataset_cfg.data_dir
    batch_size = dataset_cfg.batch_size
    dataset_name = dataset_cfg.dataset_name

    if fmt == "npz":
        num_workers = int(getattr(dataset_cfg, "num_workers", 4) or 0)
        train_loader, val_loader, test_loader = get_dataloaders_npz(
            data_dir,
            batch_size,
            dataset=dataset_name,
            num_workers=num_workers,
        )
        first_npz = sorted(Path(data_dir).glob("*.npz"))[0]
        node_input_dim = GraphNpzDataset.read_node_input_dim(first_npz)
        dataset_props = DatasetProps(
            model_type="gnn",
            node_input_dim=node_input_dim,
        )
    elif fmt == "h5":
        train_loader, val_loader, test_loader = get_dataloaders_h5(
            data_dir, batch_size, dataset=dataset_name
        )
        num_channels = train_loader.dataset.num_channels
        dataset_props = DatasetProps(
            model_type="cnn",
            num_channels=num_channels,
        )
    else:
        csv_file = getattr(dataset_cfg, "csv_file", None)
        if not csv_file:
            raise ValueError("dataset.format is 'csv' but dataset.csv_file is not set")
        train_loader, val_loader, test_loader = get_dataloaders_csv(
            data_dir, csv_file, batch_size, dataset=dataset_name
        )
        input_dim = train_loader.dataset.input_dim
        dataset_props = DatasetProps(
            model_type="mlp",
            input_dim=input_dim,
        )

    return train_loader, val_loader, test_loader, dataset_props
