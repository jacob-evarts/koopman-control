import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
import h5py

import numpy as np
from abc import ABC, abstractmethod

class H5Dataset(Dataset, ABC):
    def __init__(self, data_files: list[Path]):
        self.data = []
        self.index_map = []

        for file_idx, file_path in enumerate(data_files):
            with h5py.File(file_path, 'r') as h5f:
                data_dict = self.load_h5_data(h5f)
                n_steps = next(iter(data_dict.values())).shape[0]
                self.data.append(data_dict)

                for t in range(n_steps - 1):
                    self.index_map.append((file_idx, t))

    @abstractmethod
    def load_h5_data(self, h5f) -> dict:
        pass

    @abstractmethod
    def get_channels(self) -> list[str]:
        pass

    @property
    def num_channels(self):
        return len(self.get_channels())

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, t = self.index_map[idx]
        data_dict = self.data[file_idx]

        channels = self.get_channels()
        x_t = np.stack([data_dict[ch][t].astype(np.float32) for ch in channels], axis=0)
        x_tp1 = np.stack([data_dict[ch][t + 1].astype(np.float32) for ch in channels], axis=0)

        meta = {
            "idx": file_idx,
            "time": t,
        }

        return torch.from_numpy(x_t), torch.from_numpy(x_tp1), meta

class RabbitGrassDataset(H5Dataset):
    def load_h5_data(self, h5f):
        return {
            'grass': h5f['grass'][:],
            'rabbits': h5f['rabbits'][:]
        }

    def get_channels(self):
        return ['grass', 'rabbits']

class FireflyDataset(H5Dataset):
    def load_h5_data(self, h5f):
        return {
            'flashing': h5f['flashing'][:],
            'resting': h5f['resting'][:]
        }

    def get_channels(self):
        return ['flashing', 'resting']
    
class ArcadeDataset(H5Dataset):
    def load_h5_data(self, h5f):
        return {
            '1cell': h5f['1cell'][:],
            '2cell': h5f['2cell'][:],
            '3cell': h5f['3cell'][:],
            '4cell': h5f['4cell'][:],
        }

    def get_channels(self):
        return ['1cell', '2cell', "3cell", "4cell"]

def get_dataloaders_h5(data_folder: str, batch_size: int, train_frac=0.7, val_frac=0.2, dataset="rabbit"):
    folder = Path(data_folder)
    all_files = sorted(folder.glob("*.h5"))

    random.seed(42)
    all_files = list(all_files)
    random.shuffle(all_files)

    n_files = len(all_files)
    n_train = int(train_frac * n_files)
    n_val = int(val_frac * n_files)

    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]

    if dataset == "rabbit":
        dataset_cls = RabbitGrassDataset
    elif dataset == "firefly":
        dataset_cls = FireflyDataset
    elif dataset == "arcade":
        dataset_cls = ArcadeDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    train_dataset = dataset_cls(train_files)
    val_dataset = dataset_cls(val_files)
    test_dataset = dataset_cls(test_files)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


