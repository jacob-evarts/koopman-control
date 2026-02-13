import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
import h5py
import numpy as np

class RabbitGrassDataset(Dataset):
    def __init__(self, data_files: list[Path]):
        self.data = []
        self.index_map = []

        for file_idx, file_path in enumerate(data_files):
            with h5py.File(file_path, 'r') as h5f:
                grass = h5f['grass'][:]      # shape: [time, width, height]
                rabbits = h5f['rabbits'][:]
                n_steps = grass.shape[0]

                self.data.append({'grass': grass, 'rabbits': rabbits})

                # Create index map for each timestep
                for t in range(n_steps - 1):
                    self.index_map.append((file_idx, t))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, t = self.index_map[idx]
        data = self.data[file_idx]

        grass_t = data['grass'][t].astype(np.float32)
        rabbits_t = data['rabbits'][t].astype(np.float32)
        x_t = np.stack([grass_t, rabbits_t], axis=0)

        grass_tp1 = data['grass'][t + 1].astype(np.float32)
        rabbits_tp1 = data['rabbits'][t + 1].astype(np.float32)
        x_tp1 = np.stack([grass_tp1, rabbits_tp1], axis=0)

        meta = {
            "file_idx": file_idx,
            "time": t,
        }

        return torch.from_numpy(x_t), torch.from_numpy(x_tp1), meta


def get_dataloaders(data_folder: str, batch_size: int, train_frac=0.7, val_frac=0.2):
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

    train_dataset = RabbitGrassDataset(train_files)
    val_dataset = RabbitGrassDataset(val_files)
    test_dataset = RabbitGrassDataset(test_files)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader