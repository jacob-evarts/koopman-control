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
``KOOPMAN_WOLF_DATASET`` / ``KOOPMAN_TUMOR_DATASET`` / ``KOOPMAN_SIR_DATASET``
    Optional exact paths for the wolf, tumor, and SIR case-study datasets.
``KOOPMAN_STROBL_DATA_ROOT``
    Optional parent directory for focused Strobl datasets. Defaults to
    ``$KOOPMAN_DATA_ROOT/strobl``.
``KOOPMAN_STROBL_DATASET``
    Optional exact path to a focused Strobl ``dataset.h5``.

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


def wolf_dataset_directory() -> Path:
    """Default directory for the three-species (grass/rabbit/wolf) dataset.

    Named with cull strength so regenerations at different
    ``culling_effectiveness`` do not overwrite each other. Current default is
    ``0.03`` (see :mod:`koopman_control.data.generate_wolves`).
    """
    return data_root() / "wolf_rabbit_grass_images_cull03"


def wolf_dataset_path() -> Path:
    return _environment_path(
        "KOOPMAN_WOLF_DATASET", wolf_dataset_directory() / "dataset.h5"
    )


def tumor_dataset_directory() -> Path:
    """Default directory for the spatial tumor–healthy-tissue dataset."""
    return data_root() / "tumor_tissue_images"


def tumor_dataset_path() -> Path:
    return _environment_path(
        "KOOPMAN_TUMOR_DATASET", tumor_dataset_directory() / "dataset.h5"
    )


def sir_dataset_directory() -> Path:
    """Default directory for the spatial agentic SIR dataset (v2 sweep)."""
    return data_root() / "agentic_sir_images_v2"


def sir_dataset_path() -> Path:
    return _environment_path(
        "KOOPMAN_SIR_DATASET", sir_dataset_directory() / "dataset.h5"
    )


def strobl_data_root() -> Path:
    """Parent directory containing profile-specific focused Strobl datasets."""
    return _environment_path("KOOPMAN_STROBL_DATA_ROOT", data_root() / "strobl")


def strobl_dataset_directory(profile: str = "pilot") -> Path:
    """Directory for one focused Strobl profile (smoke, pilot, or full)."""
    return strobl_data_root() / profile


def strobl_dataset_path(profile: str = "pilot") -> Path:
    """Focused Strobl HDF5 path, overridable with ``KOOPMAN_STROBL_DATASET``."""
    return _environment_path(
        "KOOPMAN_STROBL_DATASET", strobl_dataset_directory(profile) / "dataset.h5"
    )


def training_root() -> Path:
    return output_root() / "training"


def search_root() -> Path:
    return output_root() / "search"
