import torch
from torch.utils.data import Dataset
from pathlib import Path
import h5py
import numpy as np

class RabbitGrassDataset(Dataset):
    def __init__(self, data_folder: str):
        self.folder = Path(data_folder)
        self.h5_files = sorted(self.folder.glob("*.h5"))

        self.index_map = []
        self._files = []
        for file_idx, file_path in enumerate(self.h5_files):
            h5f = h5py.File(file_path, 'r')
            n_steps = h5f['grass'].shape[0]
            self._files.append(h5f)
            for t in range(n_steps - 1):
                self.index_map.append((file_idx, t))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, t = self.index_map[idx]
        h5f = self._files[file_idx]

        grass_t = h5f['grass'][t].astype(np.float32)
        rabbits_t = h5f['rabbits'][t].astype(np.float32)
        x_t = np.stack([grass_t, rabbits_t], axis=0)

        grass_tp1 = h5f['grass'][t + 1].astype(np.float32)
        rabbits_tp1 = h5f['rabbits'][t + 1].astype(np.float32)
        x_tp1 = np.stack([grass_tp1, rabbits_tp1], axis=0)

        return torch.from_numpy(x_t), torch.from_numpy(x_tp1)

    def close(self):
        for h5f in self._files:
            h5f.close()

def get_dataloaders(data_folder: str, batch_size: int):
    dataset = RabbitGrassDataset(data_folder)
    
    files = dataset.h5_files
    n_files = len(files)
    n_train = int(0.7 * n_files)
    n_val = int(0.2 * n_files)

    train_dataset = torch.utils.data.Subset(dataset, range(n_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                                               
    val_dataset = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = torch.utils.data.Subset(dataset, range(n_train + n_val, n_files))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader