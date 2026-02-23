import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np

class CSVDataset(Dataset):
    def __init__(self, df, feature_cols):
        self.data = []
        self.index_map = []

        df = df.sort_values(["filename", "timepoint"]).reset_index(drop=True)

        for sim_name, sim_df in df.groupby("filename"):
            sim_df = sim_df.sort_values("timepoint").reset_index(drop=True)

            features = sim_df[feature_cols].values.astype(np.float32)
            n_steps = len(features)

            self.data.append(features)

            for t in range(n_steps - 1):
                self.index_map.append((len(self.data) - 1, t))

        self.feature_dim = self.data[0].shape[1]

    @property
    def input_dim(self):
        return self.feature_dim

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        seq_idx, t = self.index_map[idx]
        seq = self.data[seq_idx]

        x_t = seq[t]
        x_tp1 = seq[t + 1]

        meta = {"idx": seq_idx, "time": t}

        return torch.from_numpy(x_t), torch.from_numpy(x_tp1), meta

class RabbitGrassCSVDataset(CSVDataset):
    def __init__(self, df):
        feature_cols = [
            "rabbit_population",
            "grass_population",
            "rabbit_density",
            "grass_density",
            "log_rabbit_population",
            "log_grass_population",
            "rabbit_grass_ratio",
        ]
        super().__init__(df, feature_cols=feature_cols)

class FireflyCSVDataset(CSVDataset):
    def __init__(self, df, feature_cols=None):
        if feature_cols is None:
            feature_cols = ["firefly"]
        super().__init__(df, feature_cols=feature_cols)

def get_dataloaders_csv(
    data_folder: str,
    csv_file: str,
    batch_size: int,
    train_frac=0.7,
    val_frac=0.2,
    dataset="rabbit",
    feature_cols=None,
):
    folder = Path(data_folder)
    csv_path = folder / csv_file

    df = pd.read_csv(csv_path)

    all_sims = df["filename"].unique().tolist()

    random.seed(42)
    random.shuffle(all_sims)

    n_total = len(all_sims)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)

    train_sims = all_sims[:n_train]
    val_sims = all_sims[n_train:n_train + n_val]
    test_sims = all_sims[n_train + n_val:]

    train_df = df[df["filename"].isin(train_sims)]
    val_df = df[df["filename"].isin(val_sims)]
    test_df = df[df["filename"].isin(test_sims)]

    if dataset == "rabbit":
        dataset_cls = RabbitGrassCSVDataset
    elif dataset == "firefly":
        dataset_cls = FireflyCSVDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if feature_cols is not None:
        train_dataset = dataset_cls(train_df, feature_cols=feature_cols)
        val_dataset = dataset_cls(val_df, feature_cols=feature_cols)
        test_dataset = dataset_cls(test_df, feature_cols=feature_cols)
    else:
        train_dataset = dataset_cls(train_df)
        val_dataset = dataset_cls(val_df)
        test_dataset = dataset_cls(test_df)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader