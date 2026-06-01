"""
Load cluster-tracked GNN trajectories from a consolidated ``dataset.h5``.

Each run is stored under ``runs/{run_id}/`` with padded node slots per frame:

- ``x``           (T, N_max, F) node features
- ``present``     (T, N_max)    active-node mask
- ``edge_src`` / ``edge_dst`` (T, E_max) with ``num_edges`` (T,)
- ``node_type``   (N_max,) static slot types

Consecutive timesteps are exposed as PyG ``Data`` pairs for KoopmanGNN training.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from koopman_control.loaders.graph_npz_loaders import collate_graph_pairs, graph_dataloader_kwargs


def _read_lattice_shape(run_group: h5py.Group) -> tuple[int, int]:
    if "lattice_shape" in run_group.attrs:
        shape = np.asarray(run_group.attrs["lattice_shape"], dtype=np.int64).reshape(-1)
        if shape.size >= 2:
            return int(shape[0]), int(shape[1])
    return 64, 64


def _timestep_to_data(
    x_t: np.ndarray,
    present_t: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    num_edges: int,
    lattice_hw: tuple[int, int],
) -> Data | None:
    """Build one PyG graph for timestep ``t``; remap padded slots to local indices."""
    active = np.flatnonzero(np.asarray(present_t, dtype=np.uint8) > 0)
    if active.size == 0:
        return None

    n_slots = int(present_t.shape[0])
    slot_to_local = np.full(n_slots, -1, dtype=np.int64)
    slot_to_local[active] = np.arange(active.size, dtype=np.int64)

    x_active = np.asarray(x_t[active], dtype=np.float32)
    h, w = lattice_hw
    # centroid_x / centroid_y are grid coordinates (see run attrs feature_names)
    pos = np.stack(
        [
            x_active[:, 1] / max(w, 1),
            x_active[:, 2] / max(h, 1),
        ],
        axis=1,
    ).astype(np.float32)

    n_e = max(0, int(num_edges))
    if n_e > 0:
        src = np.asarray(edge_src[:n_e], dtype=np.int64)
        dst = np.asarray(edge_dst[:n_e], dtype=np.int64)
        valid = (src >= 0) & (dst >= 0) & (present_t[src] > 0) & (present_t[dst] > 0)
        src = slot_to_local[src[valid]]
        dst = slot_to_local[dst[valid]]
        edge_index = (
            np.stack([src, dst], axis=0).astype(np.int64)
            if src.size > 0
            else np.zeros((2, 0), dtype=np.int64)
        )
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return Data(
        x=torch.from_numpy(x_active),
        edge_index=torch.from_numpy(edge_index),
        pos=torch.from_numpy(pos),
    )


class GraphH5Dataset(Dataset):
    """Consecutive graph pairs from one or more runs inside ``dataset.h5``."""

    @staticmethod
    def read_node_input_dim(h5_path: Path, run_key: str | None = None) -> int:
        with h5py.File(h5_path, "r") as h5f:
            runs = h5f["runs"]
            key = run_key or next(iter(runs.keys()))
            grp = runs[key]
            if "num_node_features" in grp.attrs:
                return int(grp.attrs["num_node_features"])
            return int(grp["x"].shape[-1])

    def __init__(self, h5_path: Path, run_ids: list[str]):
        self.h5_path = Path(h5_path)
        self.run_ids = list(run_ids)
        self.index_map: list[tuple[str, int]] = []
        self._h5f: h5py.File | None = None
        self._lattice_cache: dict[str, tuple[int, int]] = {}

        with h5py.File(self.h5_path, "r") as h5f:
            runs = h5f["runs"]
            for run_id in self.run_ids:
                if run_id not in runs:
                    raise KeyError(f"Run {run_id!r} not found in {self.h5_path}")
                n_steps = int(runs[run_id]["x"].shape[0])
                for t in range(max(0, n_steps - 1)):
                    self.index_map.append((run_id, t))

    def _get_h5(self) -> h5py.File:
        if self._h5f is None:
            self._h5f = h5py.File(self.h5_path, "r")
        return self._h5f

    def _lattice_for_run(self, grp: h5py.Group, run_id: str) -> tuple[int, int]:
        if run_id not in self._lattice_cache:
            self._lattice_cache[run_id] = _read_lattice_shape(grp)
        return self._lattice_cache[run_id]

    def __del__(self) -> None:
        if self._h5f is not None:
            try:
                self._h5f.close()
            except Exception:  # noqa: BLE001
                pass
            self._h5f = None

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> tuple[Data, Data, dict]:
        run_id, t = self.index_map[idx]
        h5f = self._get_h5()
        grp = h5f["runs"][run_id]
        lattice = self._lattice_for_run(grp, run_id)
        x = grp["x"]
        present = grp["present"]
        edge_src = grp["edge_src"]
        edge_dst = grp["edge_dst"]
        num_edges = grp["num_edges"]

        d0 = _timestep_to_data(
            x[t],
            present[t],
            edge_src[t],
            edge_dst[t],
            int(num_edges[t]),
            lattice,
        )
        d1 = _timestep_to_data(
            x[t + 1],
            present[t + 1],
            edge_src[t + 1],
            edge_dst[t + 1],
            int(num_edges[t + 1]),
            lattice,
        )

        if d0 is None or d1 is None:
            raise RuntimeError(f"Empty graph at {run_id} t={t} or t+1 (idx={idx})")

        meta = {"run_id": run_id, "t": t}
        return d0, d1, meta


def list_run_ids(h5_path: Path, manifest_path: Path | None = None) -> list[str]:
    """Return run ids from ``manifest.json`` if present, else HDF5 group keys."""
    if manifest_path is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        runs = manifest.get("runs")
        if isinstance(runs, list) and runs:
            if isinstance(runs[0], dict) and "run_id" in runs[0]:
                return [str(r["run_id"]) for r in runs]
            return [str(r) for r in runs]
    with h5py.File(h5_path, "r") as h5f:
        return sorted(h5f["runs"].keys())


def get_dataloaders_gnn_h5(
    data_folder: str,
    batch_size: int,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    dataset_file: str = "dataset.h5",
    manifest_file: str = "manifest.json",
    num_workers: int = 4,
    dataset: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load graph pairs from ``{data_folder}/{dataset_file}``.

    Runs are split by trajectory (not by timestep). ``dataset`` is ignored (API parity).
    """
    folder = Path(data_folder)
    h5_path = folder / dataset_file
    if not h5_path.is_file():
        raise FileNotFoundError(f"GNN dataset not found: {h5_path}")

    manifest_path = folder / manifest_file
    all_runs = list_run_ids(h5_path, manifest_path if manifest_path.is_file() else None)
    if not all_runs:
        raise ValueError(f"No runs found in {h5_path}")

    _ = dataset

    random.seed(42)
    shuffled = list(all_runs)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_runs = shuffled[:n_train]
    val_runs = shuffled[n_train : n_train + n_val]
    test_runs = shuffled[n_train + n_val :]

    if not train_runs:
        raise ValueError(
            f"No training runs after split ({n} runs in {h5_path}). "
            "Lower train_frac / val_frac."
        )

    train_ds = GraphH5Dataset(h5_path, train_runs)
    val_ds = GraphH5Dataset(h5_path, val_runs)
    test_ds = GraphH5Dataset(h5_path, test_runs)

    kw = graph_dataloader_kwargs(num_workers, collate_graph_pairs)
    kw["batch_size"] = batch_size

    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    return train_loader, val_loader, test_loader
