"""Dataset + loaders for the image-based latent world model.

Why this file exists
--------------------
The world model learns ``z_{t+1} = f(z_t, u_t)`` and is trained on *multi-step*
rollouts (predicting several steps ahead), because one-step training lets a model
look accurate while drifting badly over a horizon -- exactly the failure the
Phase-0 DMDc check quantified. So the loader serves fixed-length windows of
consecutive frames together with the controls that drive each transition.

Control alignment (and the actuator lag)
----------------------------------------
In a generated trajectory, ``control[t]`` is the input that produced
``frames[t]`` (``control[0] = 0`` for the initial frame). The transition
``frames[t] -> frames[t+1]`` is therefore driven by ``control[t+1]``. Because the
simulator's actuator has a one-step lag, each transition carries a control
*history* ``[u_now, u_prev]``:

    transition k in a window starting at s:
        u_now  = control[s + k + 1]
        u_prev = control[s + k]

The loader returns the raw window controls so the model can form these features.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class RabbitGrassWindows(Dataset):
    """Fixed-length rollout windows from one split of a generated dataset.

    Each item is ``(frames, controls)`` where:
      * ``frames``   : ``(H + 1, C, W, H)`` float32 in ``[0, 1]``
      * ``controls`` : ``(H + 1,)`` float32, the per-frame control (see module
        docstring for how transitions consume it).
    """

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

        # Preload frames/controls for this split into RAM (a full 64x64 dataset
        # is a few hundred MB as uint8, which fits comfortably and avoids h5
        # read overhead every step).
        self._frames: list[np.ndarray] = []
        self._controls: list[np.ndarray] = []
        self._windows: list[tuple[int, int]] = []  # (run_index, start_t)

        with h5py.File(self.h5_path, "r") as f:
            self.num_channels = int(f.attrs["num_channels"])
            self.width = int(f.attrs["width"])
            self.height = int(f.attrs["height"])
            for grp in f["runs"].values():
                if grp.attrs["split"] != split:
                    continue
                frames = np.asarray(grp["frames"][:], dtype=np.uint8)
                controls = np.asarray(grp["control"][:], dtype=np.float32)
                run_idx = len(self._frames)
                self._frames.append(frames)
                self._controls.append(controls)
                last_start = frames.shape[0] - (self.horizon + 1)
                for s in range(0, max(0, last_start + 1), self.stride):
                    self._windows.append((run_idx, s))

        if not self._windows:
            raise ValueError(
                f"No windows for split={split!r} with horizon={horizon} in {self.h5_path}"
            )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        run_idx, s = self._windows[idx]
        h = self.horizon
        frames = self._frames[run_idx][s : s + h + 1].astype(np.float32)
        controls = self._controls[run_idx][s : s + h + 1].astype(np.float32)
        return torch.from_numpy(frames), torch.from_numpy(controls)


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
        ds = RabbitGrassWindows(h5_path, split, horizon=horizon, stride=stride)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=shuffle,
        )

    return _make("train", True), _make("val", False), _make("test", False)
