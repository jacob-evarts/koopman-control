"""Dataset + loaders for the JEPA latent model.

This mirrors :mod:`koopman_control.data.dataset` but each window also carries the
ground-truth macrostate observables ``obs``. They are unused when the encoder is
trained purely self-supervised (``w_readout = 0``) and become the target of the
optional anchoring loss when it is turned on. Observables are tiny (a handful of
scalars per frame), so always serving them keeps the anchored variant a
one-flag change with no separate loader.

Control alignment matches both supported schemas. Legacy ``control[T+1]`` stores
the action that produced each frame (with ``control[0] = 0``). Categorical-grid
datasets store interval ``action[T]``, where ``action[t]`` drives
``grid[t] -> grid[t+1]``; the loader converts this to the same state-aligned
``T+1`` representation before constructing windows.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _strings(values) -> list[str]:
    return [s.decode() if isinstance(s, bytes) else str(s) for s in values]


def _run_collection(h5: h5py.File) -> h5py.Group:
    if "runs" in h5:
        return h5["runs"]
    if "episodes" in h5:
        return h5["episodes"]
    raise KeyError("Dataset must contain a /runs or /episodes group")


def _categorical_frames(grid: np.ndarray, num_channels: int) -> np.ndarray:
    """Convert integer ``[T,H,W]`` labels to float-ready ``[T,C,H,W]`` one-hot."""
    grid = np.asarray(grid, dtype=np.uint8)
    if grid.ndim != 3:
        raise ValueError(f"categorical grid must have shape [T,H,W], got {grid.shape}")
    if grid.size and int(grid.max()) >= num_channels:
        raise ValueError(
            f"grid label {int(grid.max())} exceeds num_channels={num_channels}"
        )
    return np.moveaxis(np.eye(num_channels, dtype=np.uint8)[grid], -1, 1)


def _state_aligned_actions(action: np.ndarray, n_states: int) -> np.ndarray:
    """Represent interval actions using the legacy state-aligned convention."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if len(action) != n_states - 1:
        raise ValueError(
            f"interval action has length {len(action)}, expected {n_states - 1}"
        )
    # controls[k+1] drives state k -> k+1. controls[0] is the unavailable
    # pre-episode action and is deliberately zero rather than action[0].
    return np.concatenate((np.zeros(1, dtype=np.float32), action))


def _categorical_obs(grp: h5py.Group) -> tuple[np.ndarray, list[str]]:
    """Build frame-aligned observables from the focused dataset fields."""
    n_states = int(grp["grid"].shape[0])
    return _categorical_obs_window(grp, 0, n_states)


def _categorical_obs_window(
    grp: h5py.Group, start: int, stop: int
) -> tuple[np.ndarray, list[str]]:
    """Read one frame-aligned observable window without loading an episode."""
    parts: list[np.ndarray] = []
    names: list[str] = []
    if "counts" in grp:
        counts = np.asarray(grp["counts"][start:stop], dtype=np.float32)
        parts.append(counts)
        names.extend(
            ("sensitive_count", "resistant_count", "total_count")[: counts.shape[1]]
        )
    if "occupancy" in grp:
        occupancy = np.asarray(grp["occupancy"][start:stop], dtype=np.float32)
        if occupancy.ndim == 1:
            occupancy = occupancy[:, None]
        parts.append(occupancy)
        names.extend(f"occupancy_{i}" for i in range(occupancy.shape[1]))
    if "cost" in grp:
        cost_dataset = grp["cost"]
        n_states = int(grp["grid"].shape[0])
        if len(cost_dataset) == n_states - 1:
            if start == 0:
                cost = np.concatenate(
                    (
                        np.zeros((1,) + cost_dataset.shape[1:], dtype=np.float32),
                        np.asarray(cost_dataset[: stop - 1], dtype=np.float32),
                    )
                )
            else:
                cost = np.asarray(cost_dataset[start - 1 : stop - 1], dtype=np.float32)
        else:
            cost = np.asarray(cost_dataset[start:stop], dtype=np.float32)
        if cost.ndim == 1:
            cost = cost[:, None]
        parts.append(cost)
        names.extend(f"cost_{i}" for i in range(cost.shape[1]))
    if not parts:
        raise KeyError("categorical episode requires counts, occupancy, or cost")
    n = parts[0].shape[0]
    if any(part.shape[0] != n for part in parts):
        raise ValueError("categorical observables are not frame-aligned")
    return np.concatenate(parts, axis=1), names


class JEPAWindows(Dataset):
    """Fixed-length ``(frames, controls, obs)`` windows from one split."""

    def __init__(
        self,
        h5_path: str | Path,
        split: str,
        horizon: int = 8,
        stride: int = 1,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.split = split
        self.horizon = int(horizon)
        self.stride = int(stride)

        self._frames: list[np.ndarray | None] = []
        self._controls: list[np.ndarray | None] = []
        self._obs: list[np.ndarray | None] = []
        self._episode_ids: list[str | None] = []
        self._windows: list[tuple[int, int]] = []
        self._h5: h5py.File | None = None
        self._h5_pid: int | None = None

        with h5py.File(self.h5_path, "r") as f:
            self.num_channels = int(f.attrs["num_channels"])
            self.width = int(f.attrs["width"])
            self.height = int(f.attrs["height"])
            # Binary legacy datasets use scale=1. Continuous-channel datasets
            # (e.g. nutrient/drug) are quantized to uint8 with scale=255.
            self.frame_scale = float(f.attrs.get("frame_scale", 1.0))
            self.obs_names = _strings(f.attrs.get("obs_names", ()))
            categorical_names: list[str] | None = None
            for episode_id, grp in _run_collection(f).items():
                if grp.attrs["split"] != split:
                    continue
                if "frames" in grp:
                    frames = np.asarray(grp["frames"][:], dtype=np.uint8)
                    controls = np.asarray(grp["control"][:], dtype=np.float32)
                    obs = np.asarray(grp["obs"][:], dtype=np.float32)
                    n_frames = len(frames)
                    stored_episode_id = None
                elif "grid" in grp:
                    frames = controls = obs = None
                    n_frames = int(grp["grid"].shape[0])
                    stored_episode_id = str(episode_id)
                    _, names = _categorical_obs_window(grp, 0, min(1, n_frames))
                    if categorical_names is None:
                        categorical_names = names
                    elif categorical_names != names:
                        raise ValueError(
                            "categorical observable schema differs across episodes"
                        )
                else:
                    raise KeyError("episode requires frames/control/obs or grid/action")
                run_idx = len(self._frames)
                self._frames.append(frames)
                self._controls.append(controls)
                self._obs.append(obs)
                self._episode_ids.append(stored_episode_id)
                last_start = n_frames - (self.horizon + 1)
                for s in range(0, max(0, last_start + 1), self.stride):
                    self._windows.append((run_idx, s))

            if not self.obs_names and categorical_names is not None:
                self.obs_names = categorical_names
        if not self._windows:
            raise ValueError(
                f"No windows for split={split!r} with horizon={horizon} in {self.h5_path}"
            )

    def __len__(self) -> int:
        return len(self._windows)

    def _file(self) -> h5py.File:
        pid = os.getpid()
        if self._h5 is None or self._h5_pid != pid:
            if self._h5 is not None:
                self._h5.close()
            self._h5 = h5py.File(self.h5_path, "r")
            self._h5_pid = pid
        return self._h5

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_h5_pid"] = None
        return state

    def __del__(self) -> None:
        if self._h5 is not None:
            self._h5.close()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        run_idx, s = self._windows[idx]
        h = self.horizon
        stop = s + h + 1
        episode_id = self._episode_ids[run_idx]
        if episode_id is not None:
            group = _run_collection(self._file())[episode_id]
            frames = (
                _categorical_frames(group["grid"][s:stop], self.num_channels).astype(
                    np.float32
                )
                / self.frame_scale
            )
            if s == 0:
                controls = np.concatenate(
                    (
                        np.zeros(1, dtype=np.float32),
                        np.asarray(group["action"][: stop - 1], dtype=np.float32),
                    )
                )
            else:
                controls = np.asarray(
                    group["action"][s - 1 : stop - 1], dtype=np.float32
                )
            obs, _ = _categorical_obs_window(group, s, stop)
        else:
            stored_frames = self._frames[run_idx]
            stored_controls = self._controls[run_idx]
            stored_obs = self._obs[run_idx]
            assert (
                stored_frames is not None
                and stored_controls is not None
                and stored_obs is not None
            )
            frames = stored_frames[s:stop].astype(np.float32) / self.frame_scale
            controls = stored_controls[s:stop].astype(np.float32)
            obs = stored_obs[s:stop].astype(np.float32)
        return (
            torch.from_numpy(frames),
            torch.from_numpy(controls),
            torch.from_numpy(obs),
        )


def dataset_dims(h5_path: str | Path) -> tuple[int, int, int, list[str]]:
    """Return ``(num_channels, input_size, n_obs, obs_names)`` from the file."""
    with h5py.File(h5_path, "r") as f:
        obs_names = _strings(f.attrs.get("obs_names", ()))
        if not obs_names:
            runs = _run_collection(f)
            first = next(iter(runs.values()), None)
            if first is not None and "grid" in first:
                _, obs_names = _categorical_obs(first)
        return (
            int(f.attrs["num_channels"]),
            int(f.attrs["width"]),
            len(obs_names),
            obs_names,
        )


def get_dataloaders(
    h5_path: str | Path,
    *,
    horizon: int = 8,
    batch_size: int = 32,
    num_workers: int = 0,
    stride: int = 1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders over rollout windows."""

    def _make(split: str, shuffle: bool) -> DataLoader:
        ds = JEPAWindows(h5_path, split, horizon=horizon, stride=stride)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=shuffle,
            persistent_workers=num_workers > 0,
        )

    return _make("train", True), _make("val", False), _make("test", False)
