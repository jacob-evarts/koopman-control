"""Grass → rabbit → wolf ABM with continuous *wolf* culling.

Sibling of :mod:`koopman_control.data.rabbit_grass`. The control ``u in [0, 1]``
removes wolves (not rabbits), with the same one-step actuator lag. Downstream
control objectives can target either wolf or rabbit population — both are
image-visible and recorded as observables.

State / render
--------------
Three occupancy channels ``(grass, rabbit, wolf)``. Agents are stored as
position + energy arrays (same style as the two-species model).

Per step
--------
1. Cull wolves under ``u`` (lagged continuous actuator).
2. Grow grass on empty cells.
3. Rabbits move toward grass, eat, reproduce, starve.
4. Wolves move toward rabbits, eat one rabbit on their cell, reproduce, starve.

Stochastic and deliberately simple: enough trophic structure that a 2-macro ODE
on ``(R, G)`` is misspecified, while still matching the HDF5 layout the JEPA /
Koopman loaders already understand (``num_channels`` comes from file attrs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRASS_CHANNEL = 0
RABBIT_CHANNEL = 1
WOLF_CHANNEL = 2
NUM_CHANNELS = 3

_NEIGHBORHOOD = np.array(
    [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)],
    dtype=np.int32,
)


@dataclass(frozen=True)
class WolfRabbitGrassConfig:
    """Fixed dynamics parameters for the three-species ABM."""

    width: int = 64
    height: int = 64
    grass_growth_rate: float = 0.03
    grass_energy: float = 5.0
    rabbit_move_cost: float = 1.5
    rabbit_birth_threshold: float = 50.0
    wolf_move_cost: float = 2.0
    # Energy a wolf gains from eating one rabbit.
    rabbit_food_energy: float = 25.0
    wolf_birth_threshold: float = 60.0
    # Max per-step wolf removal probability at u = 1 (before lag term).
    # Tuned so constant-u dose response stays graded on [0,1] (wolves not wiped
    # by mid-range culls); see generate_wolves defaults.
    culling_effectiveness: float = 0.03
    cull_delay_fraction: float = 0.5


@dataclass
class WolfRabbitGrassModel:
    """One stochastic rollout of the grass–rabbit–wolf ABM."""

    cfg: WolfRabbitGrassConfig
    initial_rabbits: int = 120
    initial_wolves: int = 16
    initial_grass_prob: float = 0.3
    seed: int | None = None

    rng: np.random.Generator = field(init=False)
    grass_grid: np.ndarray = field(init=False)
    rabbit_xy: np.ndarray = field(init=False)
    rabbit_energy: np.ndarray = field(init=False)
    wolf_xy: np.ndarray = field(init=False)
    wolf_energy: np.ndarray = field(init=False)
    timestep: int = field(init=False, default=0)
    _u_prev: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        w, h = self.cfg.width, self.cfg.height
        self.grass_grid = (self.rng.random((w, h)) < self.initial_grass_prob).astype(np.uint8)
        self.rabbit_xy, self.rabbit_energy = self._spawn(int(self.initial_rabbits), energy_hi=11)
        self.wolf_xy, self.wolf_energy = self._spawn(int(self.initial_wolves), energy_hi=21)
        self.timestep = 0
        self._u_prev = 0.0

    def _spawn(self, n: int, *, energy_hi: int) -> tuple[np.ndarray, np.ndarray]:
        w, h = self.cfg.width, self.cfg.height
        if n <= 0:
            return (
                np.zeros((0, 2), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
            )
        xy = np.stack(
            [self.rng.integers(0, w, size=n), self.rng.integers(0, h, size=n)],
            axis=1,
        ).astype(np.int32)
        energy = self.rng.integers(0, energy_hi, size=n).astype(np.float32)
        return xy, energy

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    @property
    def num_rabbits(self) -> int:
        return int(self.rabbit_xy.shape[0])

    @property
    def num_wolves(self) -> int:
        return int(self.wolf_xy.shape[0])

    @property
    def num_grass(self) -> int:
        return int(self.grass_grid.sum())

    def render(self) -> np.ndarray:
        """``(3, W, H)`` float32 occupancy: grass / rabbit / wolf."""
        w, h = self.cfg.width, self.cfg.height
        img = np.zeros((NUM_CHANNELS, w, h), dtype=np.float32)
        img[GRASS_CHANNEL] = self.grass_grid.astype(np.float32)
        if self.num_rabbits > 0:
            xs, ys = self.rabbit_xy[:, 0], self.rabbit_xy[:, 1]
            img[RABBIT_CHANNEL, xs, ys] = 1.0
        if self.num_wolves > 0:
            xs, ys = self.wolf_xy[:, 0], self.wolf_xy[:, 1]
            img[WOLF_CHANNEL, xs, ys] = 1.0
        return img

    def observables(self) -> dict[str, float]:
        w, h = self.cfg.width, self.cfg.height

        def _spatial(xy: np.ndarray) -> tuple[float, float, float]:
            if xy.shape[0] == 0:
                return 0.0, 0.0, 0.0
            return (
                float(xy[:, 0].mean()),
                float(xy[:, 1].mean()),
                float(xy.std()),
            )

        rcx, rcy, rspread = _spatial(self.rabbit_xy)
        wcx, wcy, wspread = _spatial(self.wolf_xy)
        return {
            "rabbit_count": float(self.num_rabbits),
            "wolf_count": float(self.num_wolves),
            "grass_count": float(self.num_grass),
            "grass_frac": float(self.num_grass) / float(w * h),
            "rabbit_centroid_x": rcx,
            "rabbit_centroid_y": rcy,
            "rabbit_spread": rspread,
            "wolf_centroid_x": wcx,
            "wolf_centroid_y": wcy,
            "wolf_spread": wspread,
        }

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def step(self, u: float = 0.0) -> None:
        u = float(np.clip(u, 0.0, 1.0))
        self.timestep += 1
        self._apply_wolf_cull(u)
        self._grow_grass()
        if self.num_rabbits > 0:
            self._rabbit_move_eat_reproduce()
        if self.num_wolves > 0:
            self._wolf_hunt_reproduce()
        self._u_prev = u

    def _apply_wolf_cull(self, u: float) -> None:
        eff = self.cfg.culling_effectiveness
        total = u * eff + self._u_prev * self.cfg.cull_delay_fraction * eff
        if total <= 0.0 or self.num_wolves == 0:
            return
        keep = self.rng.random(self.num_wolves) > total
        self.wolf_xy = self.wolf_xy[keep]
        self.wolf_energy = self.wolf_energy[keep]

    def _grow_grass(self) -> None:
        empty = self.grass_grid == 0
        sprout = empty & (self.rng.random(self.grass_grid.shape) < self.cfg.grass_growth_rate)
        self.grass_grid[sprout] = 1

    def _step_toward(
        self, x: int, y: int, target_mask: np.ndarray, *, random_if_none: bool = True
    ) -> tuple[int, int]:
        """Greedy step onto an adjacent cell where ``target_mask`` is true."""
        w, h = self.cfg.width, self.cfg.height
        for dx, dy in _NEIGHBORHOOD:
            nx, ny = (x + int(dx)) % w, (y + int(dy)) % h
            if target_mask[nx, ny]:
                return nx, ny
        if random_if_none:
            x = (x + int(self.rng.integers(-1, 2))) % w
            y = (y + int(self.rng.integers(-1, 2))) % h
        return x, y

    def _rabbit_move_eat_reproduce(self) -> None:
        w, h = self.cfg.width, self.cfg.height
        new_xy: list[np.ndarray] = []
        new_energy: list[float] = []
        survive = np.zeros(self.num_rabbits, dtype=bool)
        grass = self.grass_grid.astype(bool)

        for i in range(self.num_rabbits):
            x, y = int(self.rabbit_xy[i, 0]), int(self.rabbit_xy[i, 1])
            x, y = self._step_toward(x, y, grass)
            e = float(self.rabbit_energy[i]) - self.cfg.rabbit_move_cost
            if self.grass_grid[x, y] == 1:
                self.grass_grid[x, y] = 0
                grass[x, y] = False
                e += self.cfg.grass_energy
            if e > self.cfg.rabbit_birth_threshold:
                e /= 2.0
                new_xy.append(np.array([x, y], dtype=np.int32))
                new_energy.append(e)
            self.rabbit_xy[i, 0], self.rabbit_xy[i, 1] = x, y
            self.rabbit_energy[i] = e
            if e >= 0.0:
                survive[i] = True

        xy = self.rabbit_xy[survive]
        en = self.rabbit_energy[survive]
        if new_xy:
            xy = np.concatenate([xy, np.stack(new_xy, axis=0)], axis=0)
            en = np.concatenate([en, np.asarray(new_energy, dtype=np.float32)], axis=0)
        self.rabbit_xy = xy.astype(np.int32)
        self.rabbit_energy = en.astype(np.float32)

    def _wolf_hunt_reproduce(self) -> None:
        w, h = self.cfg.width, self.cfg.height
        # Presence map for greedy moves; eating removes individual rabbits below.
        rabbit_present = np.zeros((w, h), dtype=bool)
        if self.num_rabbits > 0:
            rabbit_present[self.rabbit_xy[:, 0], self.rabbit_xy[:, 1]] = True

        # Mutable rabbit list so wolves can eat sequentially this step.
        r_xy = self.rabbit_xy
        r_en = self.rabbit_energy
        alive_r = np.ones(r_xy.shape[0], dtype=bool) if r_xy.shape[0] else np.zeros(0, dtype=bool)

        new_xy: list[np.ndarray] = []
        new_energy: list[float] = []
        survive_w = np.zeros(self.num_wolves, dtype=bool)

        for i in range(self.num_wolves):
            x, y = int(self.wolf_xy[i, 0]), int(self.wolf_xy[i, 1])
            x, y = self._step_toward(x, y, rabbit_present)
            e = float(self.wolf_energy[i]) - self.cfg.wolf_move_cost

            # Eat at most one living rabbit on this cell.
            if alive_r.any():
                on_cell = alive_r & (r_xy[:, 0] == x) & (r_xy[:, 1] == y)
                if on_cell.any():
                    victim = int(np.flatnonzero(on_cell)[0])
                    alive_r[victim] = False
                    e += self.cfg.rabbit_food_energy
                    # Refresh presence if that was the last rabbit on the cell.
                    still = alive_r & (r_xy[:, 0] == x) & (r_xy[:, 1] == y)
                    if not still.any():
                        rabbit_present[x, y] = False

            if e > self.cfg.wolf_birth_threshold:
                e /= 2.0
                new_xy.append(np.array([x, y], dtype=np.int32))
                new_energy.append(e)

            self.wolf_xy[i, 0], self.wolf_xy[i, 1] = x, y
            self.wolf_energy[i] = e
            if e >= 0.0:
                survive_w[i] = True

        self.rabbit_xy = r_xy[alive_r].astype(np.int32)
        self.rabbit_energy = r_en[alive_r].astype(np.float32)

        w_xy = self.wolf_xy[survive_w]
        w_en = self.wolf_energy[survive_w]
        if new_xy:
            w_xy = np.concatenate([w_xy, np.stack(new_xy, axis=0)], axis=0)
            w_en = np.concatenate([w_en, np.asarray(new_energy, dtype=np.float32)], axis=0)
        self.wolf_xy = w_xy.astype(np.int32)
        self.wolf_energy = w_en.astype(np.float32)


def rollout(
    cfg: WolfRabbitGrassConfig,
    control_seq: np.ndarray,
    *,
    initial_rabbits: int,
    initial_wolves: int,
    initial_grass_prob: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Simulate one trajectory; same return layout as :func:`rabbit_grass.rollout`."""
    control_seq = np.asarray(control_seq, dtype=np.float32).reshape(-1)
    model = WolfRabbitGrassModel(
        cfg=cfg,
        initial_rabbits=initial_rabbits,
        initial_wolves=initial_wolves,
        initial_grass_prob=initial_grass_prob,
        seed=seed,
    )

    frames = [model.render()]
    controls = [0.0]
    obs_keys = list(model.observables().keys())
    obs_series: dict[str, list[float]] = {k: [model.observables()[k]] for k in obs_keys}

    for u in control_seq:
        model.step(float(u))
        frames.append(model.render())
        controls.append(float(u))
        o = model.observables()
        for k in obs_keys:
            obs_series[k].append(o[k])

    frames_arr = np.stack(frames, axis=0).astype(np.float32)
    controls_arr = np.asarray(controls, dtype=np.float32)
    obs_arr = {k: np.asarray(v, dtype=np.float32) for k, v in obs_series.items()}
    return frames_arr, controls_arr, obs_arr
