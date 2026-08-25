"""Spatial tumor–healthy-tissue ABM with diffusing chemotherapy.

This is Case A of the spatial-control study. A localized tumor grows into a
healthy tissue sheet while nutrient and chemotherapy diffuse from a fixed
vascular pattern. The scalar control ``u in [0, 1]`` is the systemic
chemotherapy dose. Drug kills both populations, with greater toxicity to the
faster-dividing tumor cells.

The model is intentionally phenomenological, not a calibrated cancer model. Its
purpose is to create a control problem with:

* localized, non-homogeneous growth,
* delayed spatial drug transport,
* competing tumor-reduction and healthy-tissue objectives, and
* image states whose geometry matters beyond population totals.

Rendered channels are ``[healthy, tumor, nutrient, drug]`` in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HEALTHY_CHANNEL = 0
TUMOR_CHANNEL = 1
NUTRIENT_CHANNEL = 2
DRUG_CHANNEL = 3
NUM_CHANNELS = 4


def _laplacian(field: np.ndarray) -> np.ndarray:
    """Five-point Laplacian with no-flux (edge-replicating) boundaries."""
    p = np.pad(field, 1, mode="edge")
    return (
        p[2:, 1:-1]
        + p[:-2, 1:-1]
        + p[1:-1, 2:]
        + p[1:-1, :-2]
        - 4.0 * field
    )


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    """Eight-neighbor count with no wrapping across tissue boundaries."""
    p = np.pad(mask.astype(np.uint8), 1, mode="constant")
    return (
        p[:-2, :-2]
        + p[:-2, 1:-1]
        + p[:-2, 2:]
        + p[1:-1, :-2]
        + p[1:-1, 2:]
        + p[2:, :-2]
        + p[2:, 1:-1]
        + p[2:, 2:]
    )


@dataclass(frozen=True)
class TumorTissueConfig:
    """Fixed dynamics parameters for the tumor–tissue ABM."""

    width: int = 64
    height: int = 64

    # Vessels are vertical source lines, offset between trajectories. Nutrient
    # and systemic drug enter through the same source map.
    vessel_spacing: int = 16
    nutrient_source: float = 1.0
    nutrient_diffusion: float = 0.18
    nutrient_recovery: float = 0.08
    healthy_consumption: float = 0.002
    tumor_consumption: float = 0.004

    # Calibrated so u=1 can reduce a radius-6 tumor to roughly 40% of its
    # initial burden in 80 steps, while sacrificing roughly 20% of healthy
    # tissue. Lower doses retain a graded response instead of saturating early.
    drug_diffusion: float = 0.25
    drug_delivery: float = 0.20
    drug_decay: float = 0.08

    healthy_growth_rate: float = 0.06
    tumor_growth_rate: float = 0.12
    tumor_invasion_rate: float = 0.035
    healthy_growth_threshold: float = 0.25
    tumor_growth_threshold: float = 0.12
    healthy_starvation_rate: float = 0.015
    tumor_starvation_rate: float = 0.01

    # Per-step death probability multiplier at local drug concentration 1.
    healthy_drug_kill: float = 0.08
    tumor_drug_kill: float = 0.60


@dataclass
class TumorTissueModel:
    """One stochastic spatial tumor–tissue rollout."""

    cfg: TumorTissueConfig
    initial_healthy_frac: float = 0.94
    initial_tumor_radius: float = 4.0
    tumor_center_x: float | None = None
    tumor_center_y: float | None = None
    vessel_offset: int | None = None
    seed: int | None = None

    rng: np.random.Generator = field(init=False)
    healthy_grid: np.ndarray = field(init=False)
    tumor_grid: np.ndarray = field(init=False)
    nutrient: np.ndarray = field(init=False)
    drug: np.ndarray = field(init=False)
    vessel_mask: np.ndarray = field(init=False)
    initial_healthy_count: int = field(init=False)
    timestep: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        w, h = self.cfg.width, self.cfg.height

        self.healthy_grid = self.rng.random((w, h)) < float(self.initial_healthy_frac)
        self.tumor_grid = np.zeros((w, h), dtype=bool)

        cx = float(self.tumor_center_x) if self.tumor_center_x is not None else w / 2.0
        cy = float(self.tumor_center_y) if self.tumor_center_y is not None else h / 2.0
        xx, yy = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
        seed_disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= float(self.initial_tumor_radius) ** 2
        self.tumor_grid[seed_disk] = True
        self.healthy_grid[seed_disk] = False

        spacing = max(2, int(self.cfg.vessel_spacing))
        offset = (
            int(self.vessel_offset)
            if self.vessel_offset is not None
            else int(self.rng.integers(0, spacing))
        )
        self.vessel_mask = np.zeros((w, h), dtype=bool)
        self.vessel_mask[np.arange(offset, w, spacing), :] = True

        self.nutrient = np.full((w, h), 0.65, dtype=np.float32)
        self.nutrient[self.vessel_mask] = float(self.cfg.nutrient_source)
        self.drug = np.zeros((w, h), dtype=np.float32)
        self.initial_healthy_count = self.num_healthy
        self.timestep = 0

    @property
    def num_healthy(self) -> int:
        return int(self.healthy_grid.sum())

    @property
    def num_tumor(self) -> int:
        return int(self.tumor_grid.sum())

    def render(self) -> np.ndarray:
        """Return ``(4, W, H)`` float32 state in ``[0, 1]``."""
        return np.stack(
            [
                self.healthy_grid.astype(np.float32),
                self.tumor_grid.astype(np.float32),
                self.nutrient,
                self.drug,
            ],
            axis=0,
        ).astype(np.float32)

    def observables(self) -> dict[str, float]:
        w, h = self.cfg.width, self.cfg.height
        area = float(w * h)
        xy = np.argwhere(self.tumor_grid)
        if len(xy):
            centroid_x = float(xy[:, 0].mean())
            centroid_y = float(xy[:, 1].mean())
            spread = float(
                np.sqrt(
                    ((xy[:, 0] - centroid_x) ** 2 + (xy[:, 1] - centroid_y) ** 2).mean()
                )
            )
        else:
            centroid_x = centroid_y = spread = 0.0

        tumor_exposure = (
            float(self.drug[self.tumor_grid].mean()) if self.num_tumor else 0.0
        )
        healthy_exposure = (
            float(self.drug[self.healthy_grid].mean()) if self.num_healthy else 0.0
        )
        return {
            "tumor_count": float(self.num_tumor),
            "healthy_count": float(self.num_healthy),
            "tumor_frac": float(self.num_tumor) / area,
            "healthy_frac": float(self.num_healthy) / area,
            "healthy_loss": float(self.initial_healthy_count - self.num_healthy),
            "tumor_centroid_x": centroid_x,
            "tumor_centroid_y": centroid_y,
            "tumor_spread": spread,
            "mean_nutrient": float(self.nutrient.mean()),
            "mean_drug": float(self.drug.mean()),
            "tumor_drug_exposure": tumor_exposure,
            "healthy_drug_exposure": healthy_exposure,
        }

    def step(self, u: float = 0.0) -> None:
        """Advance one step under systemic chemotherapy dose ``u``."""
        u = float(np.clip(u, 0.0, 1.0))
        self.timestep += 1
        self._update_fields(u)
        self._cell_death()
        self._cell_growth()
        self._consume_nutrient()

    def _update_fields(self, u: float) -> None:
        c = self.cfg
        self.nutrient += c.nutrient_diffusion * _laplacian(self.nutrient)
        self.nutrient[self.vessel_mask] += c.nutrient_recovery * (
            c.nutrient_source - self.nutrient[self.vessel_mask]
        )
        np.clip(self.nutrient, 0.0, 1.0, out=self.nutrient)

        self.drug += c.drug_diffusion * _laplacian(self.drug)
        self.drug *= 1.0 - c.drug_decay
        self.drug[self.vessel_mask] += c.drug_delivery * u
        np.clip(self.drug, 0.0, 1.0, out=self.drug)

    def _cell_death(self) -> None:
        c = self.cfg
        healthy_starve = np.clip(
            c.healthy_growth_threshold - self.nutrient, 0.0, 1.0
        ) * c.healthy_starvation_rate
        tumor_starve = np.clip(
            c.tumor_growth_threshold - self.nutrient, 0.0, 1.0
        ) * c.tumor_starvation_rate
        healthy_p = np.clip(healthy_starve + c.healthy_drug_kill * self.drug, 0.0, 1.0)
        tumor_p = np.clip(tumor_starve + c.tumor_drug_kill * self.drug, 0.0, 1.0)
        self.healthy_grid &= self.rng.random(self.healthy_grid.shape) >= healthy_p
        self.tumor_grid &= self.rng.random(self.tumor_grid.shape) >= tumor_p

    def _cell_growth(self) -> None:
        c = self.cfg
        empty = ~(self.healthy_grid | self.tumor_grid)
        h_neighbors = _neighbor_count(self.healthy_grid)
        t_neighbors = _neighbor_count(self.tumor_grid)

        h_prob = (
            c.healthy_growth_rate
            * (h_neighbors / 8.0)
            * np.clip(
                (self.nutrient - c.healthy_growth_threshold)
                / max(1e-6, 1.0 - c.healthy_growth_threshold),
                0.0,
                1.0,
            )
        )
        t_prob = (
            c.tumor_growth_rate
            * (t_neighbors / 8.0)
            * np.clip(
                (self.nutrient - c.tumor_growth_threshold)
                / max(1e-6, 1.0 - c.tumor_growth_threshold),
                0.0,
                1.0,
            )
        )
        h_birth = empty & (h_neighbors > 0) & (self.rng.random(empty.shape) < h_prob)
        t_birth = empty & (t_neighbors > 0) & (self.rng.random(empty.shape) < t_prob)

        # If both populations claim a site, tumor wins with probability
        # proportional to its local growth pressure.
        conflict = h_birth & t_birth
        if conflict.any():
            tumor_share = t_prob / np.maximum(h_prob + t_prob, 1e-8)
            tumor_wins = self.rng.random(empty.shape) < tumor_share
            h_birth[conflict & tumor_wins] = False
            t_birth[conflict & ~tumor_wins] = False

        self.healthy_grid |= h_birth
        self.tumor_grid |= t_birth

        # Local invasion makes geometry consequential even in confluent tissue:
        # an expanding tumor can displace neighboring healthy cells rather than
        # waiting for stochastic vacancies.
        invasion_prob = (
            c.tumor_invasion_rate
            * (t_neighbors / 8.0)
            * np.clip(
                (self.nutrient - c.tumor_growth_threshold)
                / max(1e-6, 1.0 - c.tumor_growth_threshold),
                0.0,
                1.0,
            )
        )
        invaded = (
            self.healthy_grid
            & (t_neighbors > 0)
            & (self.rng.random(self.healthy_grid.shape) < invasion_prob)
        )
        self.healthy_grid[invaded] = False
        self.tumor_grid[invaded] = True

    def _consume_nutrient(self) -> None:
        self.nutrient -= (
            self.cfg.healthy_consumption * self.healthy_grid
            + self.cfg.tumor_consumption * self.tumor_grid
        )
        np.clip(self.nutrient, 0.0, 1.0, out=self.nutrient)


def rollout(
    cfg: TumorTissueConfig,
    control_seq: np.ndarray,
    *,
    initial_healthy_frac: float,
    initial_tumor_radius: float,
    tumor_center_x: float | None = None,
    tumor_center_y: float | None = None,
    vessel_offset: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Simulate one trajectory with the shared dataset return layout."""
    control_seq = np.asarray(control_seq, dtype=np.float32).reshape(-1)
    model = TumorTissueModel(
        cfg=cfg,
        initial_healthy_frac=initial_healthy_frac,
        initial_tumor_radius=initial_tumor_radius,
        tumor_center_x=tumor_center_x,
        tumor_center_y=tumor_center_y,
        vessel_offset=vessel_offset,
        seed=seed,
    )
    frames = [model.render()]
    controls = [0.0]
    first = model.observables()
    obs_series: dict[str, list[float]] = {k: [v] for k, v in first.items()}

    for u in control_seq:
        model.step(float(u))
        frames.append(model.render())
        controls.append(float(u))
        obs = model.observables()
        for key in obs_series:
            obs_series[key].append(obs[key])

    return (
        np.stack(frames).astype(np.float32),
        np.asarray(controls, dtype=np.float32),
        {k: np.asarray(v, dtype=np.float32) for k, v in obs_series.items()},
    )
