import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
import h5py

import numpy as np
from abc import ABC, abstractmethod

# Default CNN input channels when ``dataset.channels`` is omitted from config
DEFAULT_H5_CHANNELS: dict[str, list[str]] = {
    "rabbit": ["rabbits"],
    "firefly": ["flashing"],
    "arcade": ["1cell", "2cell", "3cell", "4cell"],
}


def resolve_h5_channels(dataset_name: str, channels: list[str] | None) -> list[str]:
    """Resolve channel list from config or dataset-specific defaults."""
    if channels is not None and len(channels) > 0:
        return [str(c) for c in channels]
    if dataset_name not in DEFAULT_H5_CHANNELS:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}; set dataset.channels in config "
            f"or add a default in DEFAULT_H5_CHANNELS."
        )
    return list(DEFAULT_H5_CHANNELS[dataset_name])


class H5Dataset(Dataset, ABC):
    def __init__(
        self,
        data_files: list[Path],
        channels: list[str],
        rollout_horizon: int = 1,
        include_control: bool = False,
    ):
        if not channels:
            raise ValueError("channels must be a non-empty list of HDF5 dataset keys")
        self.channels = list(channels)
        self.rollout_horizon = max(1, int(rollout_horizon))
        self.include_control = include_control
        self.data = []
        self.index_map = []

        for file_idx, file_path in enumerate(data_files):
            with h5py.File(file_path, "r") as h5f:
                data_dict = self.load_h5_data(h5f)
                missing = [ch for ch in self.channels if ch not in data_dict]
                if missing:
                    raise KeyError(
                        f"{file_path}: missing channel keys {missing}; "
                        f"available {sorted(data_dict)}"
                    )
                n_steps = next(iter(data_dict.values())).shape[0]
                self.data.append(data_dict)

                last_t = n_steps - self.rollout_horizon
                for t in range(max(0, last_t)):
                    self.index_map.append((file_idx, t))

    @abstractmethod
    def load_h5_data(self, h5f) -> dict:
        pass

    @property
    def num_channels(self):
        return len(self.channels)

    def __len__(self):
        return len(self.index_map)

    def _control_at(self, data_dict: dict, t: int) -> float:
        if "control" in data_dict:
            return float(data_dict["control"][t])
        return 0.0

    def __getitem__(self, idx):
        file_idx, t = self.index_map[idx]
        data_dict = self.data[file_idx]
        h = self.rollout_horizon

        meta = {
            "idx": file_idx,
            "time": t,
        }

        if h > 1:
            frames = [
                np.stack([data_dict[ch][t + k].astype(np.float32) for ch in self.channels], axis=0)
                for k in range(h + 1)
            ]
            u_seq = np.array(
                [self._control_at(data_dict, t + k) for k in range(h)],
                dtype=np.float32,
            )
            return (
                torch.from_numpy(np.stack(frames, axis=0)),
                torch.from_numpy(u_seq),
                meta,
            )

        x_t = np.stack([data_dict[ch][t].astype(np.float32) for ch in self.channels], axis=0)
        x_tp1 = np.stack([data_dict[ch][t + 1].astype(np.float32) for ch in self.channels], axis=0)

        if self.include_control:
            u_t = np.array([self._control_at(data_dict, t)], dtype=np.float32)
            return torch.from_numpy(x_t), torch.from_numpy(x_tp1), torch.from_numpy(u_t), meta

        return torch.from_numpy(x_t), torch.from_numpy(x_tp1), meta


class RabbitGrassDataset(H5Dataset):
    """Loads ``grass``, ``rabbits``, and optional ``control``; CNN channels come from config."""

    def load_h5_data(self, h5f):
        out = {
            "grass": h5f["grass"][:],
            "rabbits": h5f["rabbits"][:],
        }
        if "control" in h5f:
            out["control"] = h5f["control"][:]
        return out


class FireflyDataset(H5Dataset):
    """Loads ``flashing``, ``resting``, and optional ``control`` (centre beacon on/off)."""

    def load_h5_data(self, h5f):
        out = {
            "flashing": h5f["flashing"][:],
            "resting": h5f["resting"][:],
        }
        if "control" in h5f:
            out["control"] = h5f["control"][:]
        return out


class ArcadeDataset(H5Dataset):
    def load_h5_data(self, h5f):
        return {
            "1cell": h5f["1cell"][:],
            "2cell": h5f["2cell"][:],
            "3cell": h5f["3cell"][:],
            "4cell": h5f["4cell"][:],
        }


def get_dataloaders_h5(
    data_folder: str,
    batch_size: int,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    dataset: str = "rabbit",
    channels: list[str] | None = None,
    rollout_horizon: int = 1,
    include_control: bool = False,
):
    folder = Path(data_folder)
    all_files = sorted(folder.glob("*.h5"))

    random.seed(42)
    all_files = list(all_files)
    random.shuffle(all_files)

    n_files = len(all_files)
    n_train = int(train_frac * n_files)
    n_val = int(val_frac * n_files)

    train_files = all_files[:n_train]
    val_files = all_files[n_train : n_train + n_val]
    test_files = all_files[n_train + n_val :]

    if dataset == "rabbit":
        dataset_cls = RabbitGrassDataset
    elif dataset == "firefly":
        dataset_cls = FireflyDataset
    elif dataset == "arcade":
        dataset_cls = ArcadeDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    channel_list = resolve_h5_channels(dataset, channels)

    train_dataset = dataset_cls(
        train_files,
        channel_list,
        rollout_horizon=rollout_horizon,
        include_control=include_control,
    )
    val_dataset = dataset_cls(
        val_files,
        channel_list,
        rollout_horizon=rollout_horizon,
        include_control=include_control,
    )
    test_dataset = dataset_cls(
        test_files,
        channel_list,
        rollout_horizon=rollout_horizon,
        include_control=include_control,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
