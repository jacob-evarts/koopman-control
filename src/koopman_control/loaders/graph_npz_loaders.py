"""
Load precomputed graph simulations from .npz (see data/h5_to_graphs.py).

Each .npz stores a disjoint union over timesteps: x, pos, edge_index, ptr, batch, ...
We slice consecutive timesteps (t, t+1) as pairs of PyG Data objects for Koopman training.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data


def _slice_timestep(
    ptr: np.ndarray,
    edge_index: np.ndarray,
    x: np.ndarray,
    pos: np.ndarray,
    t: int,
) -> Data | None:
    """Build PyG Data for timestep t. Returns None if the graph has no nodes."""
    lo = int(ptr[t])
    hi = int(ptr[t + 1])
    if hi <= lo:
        return None
    ei = edge_index
    mask = (ei[0] >= lo) & (ei[0] < hi) & (ei[1] >= lo) & (ei[1] < hi)
    ei_local = ei[:, mask].astype(np.int64) - lo
    x_t = np.asarray(x[lo:hi], dtype=np.float32)
    pos_t = np.asarray(pos[lo:hi], dtype=np.float32)
    return Data(
        x=torch.from_numpy(x_t),
        edge_index=torch.from_numpy(ei_local),
        pos=torch.from_numpy(pos_t),
    )


class GraphNpzDataset(Dataset):
    """Pairs of consecutive graphs from each .npz trajectory."""

    @staticmethod
    def read_node_input_dim(npz_path: Path) -> int:
        with np.load(npz_path) as d:
            return int(d["x"].shape[1])

    def __init__(self, npz_paths: list[Path]):
        self.paths = [Path(p) for p in npz_paths]
        self.index_map: list[tuple[int, int]] = []
        for fi, path in enumerate(self.paths):
            with np.load(path, mmap_mode="r") as d:
                ptr = np.asarray(d["ptr"])
            n_steps = int(len(ptr) - 1)
            for t in range(n_steps - 1):
                lo0, hi0 = int(ptr[t]), int(ptr[t + 1])
                lo1, hi1 = int(ptr[t + 1]), int(ptr[t + 2])
                if hi0 > lo0 and hi1 > lo1:
                    self.index_map.append((fi, t))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> tuple[Data, Data, dict]:
        fi, t = self.index_map[idx]
        path = self.paths[fi]
        with np.load(path, mmap_mode="r") as d:
            ptr = d["ptr"]
            ei = d["edge_index"]
            x = d["x"]
            pos = d["pos"]
            d0 = _slice_timestep(ptr, ei, x, pos, t)
            d1 = _slice_timestep(ptr, ei, x, pos, t + 1)
        assert d0 is not None and d1 is not None
        meta = {"file": path.name, "t": t}
        return d0, d1, meta


def collate_graph_pairs(
    batch: list[tuple[Data, Data, dict]],
) -> tuple[Batch, Batch, list[dict]]:
    batch_t = Batch.from_data_list([b[0] for b in batch])
    batch_tp = Batch.from_data_list([b[1] for b in batch])
    meta = [b[2] for b in batch]
    return batch_t, batch_tp, meta


def get_dataloaders_npz(
    data_folder: str,
    batch_size: int,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    dataset: str | None = None,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    data_folder: directory containing one or more ``*.npz`` graph trajectories.

    ``dataset`` is accepted for API compatibility with other loaders and is ignored.
    """
    folder = Path(data_folder)
    all_files = sorted(p for p in folder.glob("*.npz") if p.is_file())
    if not all_files:
        raise ValueError(f"No .npz files found in {folder}")

    _ = dataset  # unused; kept for parity with H5/CSV loader signatures

    random.seed(42)
    all_files = list(all_files)
    random.shuffle(all_files)

    n = len(all_files)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_files = all_files[:n_train]
    val_files = all_files[n_train : n_train + n_val]
    test_files = all_files[n_train + n_val :]
    if not train_files:
        raise ValueError(
            f"No training .npz files after split ({n} files under {folder}). "
            "Add more runs or lower train_frac / val_frac."
        )

    train_ds = GraphNpzDataset(train_files)
    val_ds = GraphNpzDataset(val_files)
    test_ds = GraphNpzDataset(test_files)
    collate = collate_graph_pairs

    kw: dict = dict(
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kw["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    return train_loader, val_loader, test_loader
