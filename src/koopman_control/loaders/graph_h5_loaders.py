"""
Load cluster-tracked GNN trajectories from HDF5 graph exports.

Supports two layouts:

1. **Consolidated** ``dataset.h5`` — each run under ``runs/{run_id}/`` plus
   ``manifest.json`` listing all runs.
2. **Single-run flat** ``gnn.h5`` — arrays at the file root with ``run_id`` in
   attrs / ``metadata.json`` (one trajectory per folder).

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
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch, Data

from koopman_control.loaders.graph_npz_loaders import (
    PrebatchedGraphDataset,
    collate_graph_pairs,
    collate_prebatched,
    graph_dataloader_kwargs,
)


def _read_lattice_shape(run_group: h5py.Group) -> tuple[int, int]:
    if "lattice_shape" in run_group.attrs:
        shape = np.asarray(run_group.attrs["lattice_shape"], dtype=np.int64).reshape(-1)
        if shape.size >= 2:
            return int(shape[0]), int(shape[1])
    return 64, 64


def _active_count(present_t: np.ndarray) -> int:
    return int(np.asarray(present_t, dtype=np.uint8).sum())


def _valid_pair_indices(n_steps: int, present: np.ndarray) -> list[int]:
    """Return timestep starts ``t`` where both ``t`` and ``t+1`` have active nodes."""
    valid: list[int] = []
    for t in range(max(0, n_steps - 1)):
        if _active_count(present[t]) > 0 and _active_count(present[t + 1]) > 0:
            valid.append(t)
    return valid


def _timestep_to_data(
    x_t: np.ndarray,
    present_t: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    num_edges: int,
    lattice_hw: tuple[int, int],
    edge_weight_t: np.ndarray | None = None,
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
        edge_weight = (
            np.asarray(edge_weight_t[:n_e], dtype=np.float32)[valid]
            if edge_weight_t is not None
            else None
        )
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_weight = None

    data_kwargs: dict = {
        "x": torch.from_numpy(x_active),
        "edge_index": torch.from_numpy(edge_index),
        "pos": torch.from_numpy(pos),
    }
    if edge_weight is not None and edge_weight.size > 0:
        data_kwargs["edge_weight"] = torch.from_numpy(edge_weight)
    else:
        data_kwargs["edge_weight"] = torch.zeros(0, dtype=torch.float32)
    return Data(**data_kwargs)


def control_type_to_scalar(control_type: str) -> float:
    """Map run ``control_type`` attr to scalar cull intensity in ``[0, 1]``."""
    ct = str(control_type).lower()
    if ct in ("uncontrolled", "none", ""):
        return 0.0
    if ct == "cull_on":
        return 1.0
    if ct.startswith("cull_p") and ct[6:].isdigit():
        return float(ct[6:]) / 100.0
    if ct in ("cull_periodic", "cull_pulse"):
        return 0.5
    return 0.0


def _control_scalar_at_t(grp: h5py.Group, t: int) -> float:
    if "control" in grp:
        return float(np.asarray(grp["control"][t], dtype=np.float32))
    return control_type_to_scalar(str(grp.attrs.get("control_type", "uncontrolled")))


def _run_group(h5f: h5py.File, run_id: str) -> h5py.Group:
    """Return the HDF5 group for ``run_id`` (consolidated or flat layout)."""
    if "runs" in h5f:
        return h5f["runs"][run_id]
    flat_id = str(h5f.attrs.get("run_id", "single_run"))
    if run_id != flat_id:
        raise KeyError(f"Run {run_id!r} not found in flat {h5f.filename!r} (has {flat_id!r})")
    return h5f


def _build_pair_at_t(
    run_id: str,
    t: int,
    lattice: tuple[int, int],
    x: np.ndarray,
    present: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    num_edges: np.ndarray,
    edge_weight: np.ndarray | None = None,
    include_control: bool = False,
    control_u: float | None = None,
) -> tuple[Data, Data, dict]:
    w0 = edge_weight[t] if edge_weight is not None else None
    w1 = edge_weight[t + 1] if edge_weight is not None else None
    d0 = _timestep_to_data(
        x[t],
        present[t],
        edge_src[t],
        edge_dst[t],
        int(num_edges[t]),
        lattice,
        w0,
    )
    d1 = _timestep_to_data(
        x[t + 1],
        present[t + 1],
        edge_src[t + 1],
        edge_dst[t + 1],
        int(num_edges[t + 1]),
        lattice,
        w1,
    )
    if d0 is None or d1 is None:
        raise RuntimeError(f"Empty graph at {run_id} t={t} or t+1")
    meta: dict = {"run_id": run_id, "t": t}
    if include_control:
        meta["u"] = float(control_u if control_u is not None else 0.0)
    return d0, d1, meta


class GraphH5Dataset(Dataset):
    """Consecutive graph pairs from one or more runs inside a graph HDF5 file."""

    @staticmethod
    def read_node_input_dim(h5_path: Path, run_key: str | None = None) -> int:
        with h5py.File(h5_path, "r") as h5f:
            if "runs" in h5f:
                runs = h5f["runs"]
                key = run_key or next(iter(runs.keys()))
                grp = runs[key]
            else:
                grp = h5f
            if "num_node_features" in grp.attrs:
                return int(grp.attrs["num_node_features"])
            return int(grp["x"].shape[-1])

    def __init__(
        self,
        h5_path: Path,
        run_ids: list[str],
        preload: bool = False,
        include_control: bool = False,
    ):
        self.h5_path = Path(h5_path)
        self.run_ids = list(run_ids)
        self.preload = bool(preload)
        self.include_control = bool(include_control)
        self.index_map: list[tuple[str, int]] = []
        self._cache: list[tuple[Data, Data, dict]] | None = None
        self._h5f: h5py.File | None = None
        self._lattice_cache: dict[str, tuple[int, int]] = {}

        with h5py.File(self.h5_path, "r") as h5f:
            cache: list[tuple[Data, Data, dict]] = [] if self.preload else []
            for run_id in self.run_ids:
                grp = _run_group(h5f, run_id)
                present = np.asarray(grp["present"][:], dtype=np.uint8)
                n_steps = int(grp["x"].shape[0])
                valid_ts = _valid_pair_indices(n_steps, present)
                if self.preload:
                    lattice = _read_lattice_shape(grp)
                    x = np.asarray(grp["x"][:], dtype=np.float32)
                    edge_src = np.asarray(grp["edge_src"][:], dtype=np.int32)
                    edge_dst = np.asarray(grp["edge_dst"][:], dtype=np.int32)
                    num_edges = np.asarray(grp["num_edges"][:], dtype=np.int32)
                    edge_weight_arr = (
                        np.asarray(grp["edge_weight"][:], dtype=np.float32)
                        if "edge_weight" in grp
                        else None
                    )
                    for t in valid_ts:
                        self.index_map.append((run_id, t))
                        u_t = _control_scalar_at_t(grp, t) if self.include_control else None
                        cache.append(
                            _build_pair_at_t(
                                run_id,
                                t,
                                lattice,
                                x,
                                present,
                                edge_src,
                                edge_dst,
                                num_edges,
                                edge_weight_arr,
                                include_control=self.include_control,
                                control_u=u_t,
                            )
                        )
                else:
                    for t in valid_ts:
                        self.index_map.append((run_id, t))

            if self.preload:
                self._cache = cache

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
        if self._cache is not None:
            return self._cache[idx]

        run_id, t = self.index_map[idx]
        h5f = self._get_h5()
        grp = _run_group(h5f, run_id)
        lattice = self._lattice_for_run(grp, run_id)
        x = grp["x"]
        present = grp["present"]
        edge_src = grp["edge_src"]
        edge_dst = grp["edge_dst"]
        num_edges = grp["num_edges"]
        edge_weight = grp["edge_weight"] if "edge_weight" in grp else None
        u_t = _control_scalar_at_t(grp, t) if self.include_control else None

        return _build_pair_at_t(
            run_id,
            t,
            lattice,
            x,
            present,
            edge_src,
            edge_dst,
            num_edges,
            edge_weight,
            include_control=self.include_control,
            control_u=u_t,
        )


def list_run_ids(h5_path: Path, manifest_path: Path | None = None) -> list[str]:
    """Return run ids from manifest/metadata or HDF5 (consolidated or flat)."""
    if manifest_path is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        runs = manifest.get("runs")
        if isinstance(runs, list) and runs:
            if isinstance(runs[0], dict) and "run_id" in runs[0]:
                return [str(r["run_id"]) for r in runs]
            return [str(r) for r in runs]
        if "run_id" in manifest:
            return [str(manifest["run_id"])]
    with h5py.File(h5_path, "r") as h5f:
        if "runs" in h5f:
            return sorted(h5f["runs"].keys())
        return [str(h5f.attrs.get("run_id", "single_run"))]


def _wrap_dataloader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    precollate: bool,
    batch_device: str | None,
) -> DataLoader:
    if precollate:
        if batch_device is not None and num_workers > 0:
            num_workers = 0
        ds = PrebatchedGraphDataset(ds, batch_size, batch_device=batch_device)
        kw = graph_dataloader_kwargs(num_workers, collate_prebatched)
        kw["batch_size"] = 1
    else:
        kw = graph_dataloader_kwargs(num_workers, collate_graph_pairs)
        kw["batch_size"] = batch_size
    return DataLoader(ds, shuffle=shuffle, **kw)


def get_dataloaders_gnn_h5(
    data_folder: str,
    batch_size: int,
    train_frac: float = 0.7,
    val_frac: float = 0.2,
    dataset_file: str = "dataset.h5",
    manifest_file: str = "manifest.json",
    num_workers: int = 4,
    dataset: str | None = None,
    preload: bool = False,
    precollate: bool = False,
    batch_device: str | None = None,
    include_control: bool = False,
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
    if not manifest_path.is_file():
        metadata_path = folder / "metadata.json"
        if metadata_path.is_file():
            manifest_path = metadata_path

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

    if n == 1:
        # Single flat export: split consecutive timestep pairs, not runs.
        full_ds = GraphH5Dataset(
            h5_path, all_runs, preload=preload, include_control=include_control
        )
        n_pairs = len(full_ds)
        if n_pairs < 2:
            raise ValueError(f"Need at least 2 timestep pairs; got {n_pairs} in {h5_path}")
        indices = list(range(n_pairs))
        random.shuffle(indices)
        n_train = max(1, int(train_frac * n_pairs))
        n_val = max(1, int(val_frac * n_pairs)) if n_pairs > 2 else 0
        if n_train + n_val >= n_pairs:
            n_train = max(1, n_pairs - 1)
            n_val = 0
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :] or train_idx[-1:]
        train_ds = Subset(full_ds, train_idx)
        val_ds = Subset(full_ds, val_idx) if val_idx else Subset(full_ds, test_idx[:1])
        test_ds = Subset(full_ds, test_idx)
    else:
        if not train_runs:
            raise ValueError(
                f"No training runs after split ({n} runs in {h5_path}). "
                "Lower train_frac / val_frac."
            )
        train_ds = GraphH5Dataset(
            h5_path, train_runs, preload=preload, include_control=include_control
        )
        val_ds = GraphH5Dataset(
            h5_path, val_runs, preload=preload, include_control=include_control
        )
        test_ds = GraphH5Dataset(
            h5_path, test_runs, preload=preload, include_control=include_control
        )

    train_loader = _wrap_dataloader(
        train_ds, batch_size, shuffle=True, num_workers=num_workers,
        precollate=precollate, batch_device=batch_device,
    )
    val_loader = _wrap_dataloader(
        val_ds, batch_size, shuffle=False, num_workers=num_workers,
        precollate=precollate, batch_device=batch_device,
    )
    test_loader = _wrap_dataloader(
        test_ds, batch_size, shuffle=False, num_workers=num_workers,
        precollate=precollate, batch_device=batch_device,
    )
    return train_loader, val_loader, test_loader
