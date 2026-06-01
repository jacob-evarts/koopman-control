"""
Unified data loader API. Dispatches to H5 or CSV loaders based on config.
"""
from dataclasses import dataclass
from pathlib import Path
from omegaconf import DictConfig

from koopman_control.loaders.h5_loaders import get_dataloaders_h5
from koopman_control.loaders.csv_loader import get_dataloaders_csv
from koopman_control.loaders.graph_npz_loaders import get_dataloaders_npz, GraphNpzDataset
from koopman_control.loaders.graph_h5_loaders import get_dataloaders_gnn_h5, GraphH5Dataset


@dataclass
class DatasetProps:
    """Properties of the loaded dataset needed to build the model."""
    model_type: str  # "cnn" | "mlp" | "gnn"
    input_dim: int | None = None  # for MLP
    num_channels: int | None = None  # for CNN
    channels: list[str] | None = None  # HDF5 keys stacked as CNN channels
    node_input_dim: int | None = None  # for GNN (.npz graph node features)


def _channels_from_cfg(dataset_cfg: DictConfig) -> list[str] | None:
    raw = getattr(dataset_cfg, "channels", None)
    if raw is None:
        return None
    return [str(c) for c in list(raw)]


def _infer_format(dataset_cfg: DictConfig) -> str:
    """Infer dataset format from config. Prefer explicit format, else derive from csv_file."""
    if hasattr(dataset_cfg, "format") and dataset_cfg.format is not None:
        return dataset_cfg.format
    csv_file = getattr(dataset_cfg, "csv_file", None)
    return "csv" if csv_file else "h5"


def get_dataloaders(dataset_cfg: DictConfig, model_cfg: DictConfig | None = None):
    """
    Return (train_loader, val_loader, test_loader, dataset_props) from dataset config.

    dataset_cfg should have:
      - format: "h5" | "csv" | "npz" | "gnn_h5" (optional; inferred from csv_file if missing, unless npz)
      - data_dir, batch_size, dataset_name
      - channels: list of HDF5 keys for CNN input (optional; see conf/dataset/*.yaml)
      - csv_file: required when format is "csv"
    """
    fmt = _infer_format(dataset_cfg)
    data_dir = dataset_cfg.data_dir
    batch_size = dataset_cfg.batch_size
    dataset_name = dataset_cfg.dataset_name

    if fmt == "gnn_h5":
        num_workers = int(getattr(dataset_cfg, "num_workers", 4) or 0)
        dataset_file = str(getattr(dataset_cfg, "dataset_file", "dataset.h5"))
        manifest_file = str(getattr(dataset_cfg, "manifest_file", "manifest.json"))
        train_loader, val_loader, test_loader = get_dataloaders_gnn_h5(
            data_dir,
            batch_size,
            dataset=dataset_name,
            dataset_file=dataset_file,
            manifest_file=manifest_file,
            num_workers=num_workers,
        )
        h5_path = Path(data_dir) / dataset_file
        node_input_dim = GraphH5Dataset.read_node_input_dim(h5_path)
        dataset_props = DatasetProps(
            model_type="gnn",
            node_input_dim=node_input_dim,
        )
    elif fmt == "npz":
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
        rollout_horizon = int(getattr(dataset_cfg, "rollout_horizon", 1) or 1)
        include_control = bool(getattr(dataset_cfg, "include_control", False))
        if model_cfg is not None:
            if hasattr(model_cfg, "rollout_horizon") and model_cfg.rollout_horizon is not None:
                rollout_horizon = int(model_cfg.rollout_horizon)
            if hasattr(model_cfg, "include_control") and model_cfg.include_control is not None:
                include_control = bool(model_cfg.include_control)
        train_loader, val_loader, test_loader = get_dataloaders_h5(
            data_dir,
            batch_size,
            dataset=dataset_name,
            channels=_channels_from_cfg(dataset_cfg),
            rollout_horizon=rollout_horizon,
            include_control=include_control,
        )
        num_channels = train_loader.dataset.num_channels
        dataset_props = DatasetProps(
            model_type="cnn",
            num_channels=num_channels,
            channels=list(train_loader.dataset.channels),
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
