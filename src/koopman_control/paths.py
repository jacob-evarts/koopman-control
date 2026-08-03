"""Central filesystem locations, configurable without editing source code.

Environment variables
---------------------
``KOOPMAN_DATA_ROOT``
    Parent directory for generated datasets. Defaults to ``<repo>/data``.
``KOOPMAN_OUTPUT_ROOT``
    Parent directory for training runs and studies. Defaults to ``<repo>/outputs``.
``KOOPMAN_DATASET``
    Optional exact path to ``dataset.h5``. When unset, the default is
    ``$KOOPMAN_DATA_ROOT/rabbit_grass_images/dataset.h5``.

Explicit CLI paths always take precedence over these defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _environment_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


def data_root() -> Path:
    return _environment_path("KOOPMAN_DATA_ROOT", PROJECT_ROOT / "data")


def output_root() -> Path:
    return _environment_path("KOOPMAN_OUTPUT_ROOT", PROJECT_ROOT / "outputs")


def dataset_directory() -> Path:
    return data_root() / "rabbit_grass_images"


def dataset_path() -> Path:
    return _environment_path("KOOPMAN_DATASET", dataset_directory() / "dataset.h5")


def training_root() -> Path:
    return output_root() / "training"


def search_root() -> Path:
    return output_root() / "search"
