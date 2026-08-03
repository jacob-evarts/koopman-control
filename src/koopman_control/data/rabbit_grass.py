"""Rabbit-grass agent-based model with a *continuous* control actuator.

Why this file exists
--------------------
The research goal is to control a biological system from image observations by
evolving a latent representation of the image forward under a control input. To
identify how the control enters the dynamics (the ``B`` matrix in
``z_{t+1} = A z_t + B u_t``), the training data must *excite* the control input
across a range of amplitudes. The original simulator only supported a binary
cull (on/off), which can only ever reveal the control effect at ``u in {0, 1}``
and cannot test whether the effect scales linearly with input amplitude.

This vendored model therefore accepts a **continuous** cull intensity
``u in [0, 1]`` and applies it as a per-rabbit removal probability. This is the
single most important change for the "training data" concern: it turns a switch
into a knob, so system-identification can probe the input-response curve.

Model dynamics (a minimal predator-resource ABM)
------------------------------------------------
State:
  * ``grass_grid`` : (W, H) uint8, 1 where grass is present.
  * ``rabbits``    : list of agents, each with integer position and an energy.

Per step:
  1. Culling: each rabbit is removed independently with probability
     ``total_cull_effect`` (see the actuator note below).
  2. Grass regrows: empty cells become grass with probability
     ``grass_growth_rate``.
  3. Each surviving rabbit moves toward adjacent grass (else randomly), pays a
     move cost in energy, eats grass on its new cell (gaining energy), and
     reproduces when energy exceeds a threshold. Rabbits with negative energy
     die.

Actuator (the control input ``u``)
----------------------------------
``total_cull_effect = u * culling_effectiveness + u_prev * delay_fraction *
culling_effectiveness``. Two deliberate properties:
  * **Continuous & monotonic** in ``u`` so the control authority can be
    identified across amplitudes.
  * **One-step lag** via the ``u_prev`` term. Real actuators have delay, and
    keeping it here forces the downstream world model to condition on the
    control *history* ``[u_t, u_{t-1}]`` rather than assuming instantaneous,
    memoryless control. The generator records the full ``u`` sequence so this
    is recoverable.

The ABM is stochastic. A single ``(x_t, u_t) -> x_{t+1}`` transition is a draw,
not a deterministic map, which is why downstream prediction targets are treated
distributionally (e.g. via aggregate observables and multi-seed averaging).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Channel indices for the rendered image observation.
GRASS_CHANNEL = 0
RABBIT_CHANNEL = 1
NUM_CHANNELS = 2


@dataclass(frozen=True)
class RabbitGrassConfig:
    """Fixed dynamics parameters for the rabbit-grass ABM.

    Initial conditions (``initial_rabbits``, ``initial_grass_prob``) are passed
    separately to :class:`RabbitGrassModel` so a single config can seed a whole
    sweep of initial conditions.
    """

    width: int = 64
    height: int = 64
    grass_growth_rate: float = 0.02
    grass_energy: int = 5
    rabbit_move_cost: float = 1.5
    rabbit_birth_threshold: int = 50
    # Maximum per-step removal probability at u = 1 (before the delay term).
    culling_effectiveness: float = 0.10
    # Fraction of the previous step's cull that carries over (actuator lag).
    cull_delay_fraction: float = 0.5


@dataclass
class RabbitGrassModel:
    """A single stochastic rollout of the rabbit-grass ABM.

    Use :meth:`step` to advance one timestep with a continuous control input,
    and :meth:`render` to obtain the 2-channel image observation.
    """

    cfg: RabbitGrassConfig
    initial_rabbits: int = 100
    initial_grass_prob: float = 0.3
    seed: int | None = None

    # Internal state (initialized in __post_init__).
    rng: np.random.Generator = field(init=False)
    grass_grid: np.ndarray = field(init=False)
    rabbit_xy: np.ndarray = field(init=False)  # (N, 2) int32 positions
    rabbit_energy: np.ndarray = field(init=False)  # (N,) float32
    timestep: int = field(init=False, default=0)
    _u_prev: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        w, h = self.cfg.width, self.cfg.height
        self.grass_grid = (self.rng.random((w, h)) < self.initial_grass_prob).astype(np.uint8)
        n = int(self.initial_rabbits)
        self.rabbit_xy = np.stack(
            [
                self.rng.integers(0, w, size=n),
                self.rng.integers(0, h, size=n),
            ],
            axis=1,
        ).astype(np.int32)
        self.rabbit_energy = self.rng.integers(0, 11, size=n).astype(np.float32)
        self.timestep = 0
        self._u_prev = 0.0

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    @property
    def num_rabbits(self) -> int:
        return int(self.rabbit_xy.shape[0])

    @property
    def num_grass(self) -> int:
        return int(self.grass_grid.sum())

    def render(self) -> np.ndarray:
        """Return the ``(2, W, H)`` float32 image: [grass occupancy, rabbit occupancy].

        Both channels are in ``[0, 1]``. Rabbit occupancy is a binary presence
        map (multiple rabbits on a cell still render as 1); aggregate counts are
        available via :meth:`observables` for control objectives.
        """
        w, h = self.cfg.width, self.cfg.height
        img = np.zeros((NUM_CHANNELS, w, h), dtype=np.float32)
        img[GRASS_CHANNEL] = self.grass_grid.astype(np.float32)
        if self.num_rabbits > 0:
            xs, ys = self.rabbit_xy[:, 0], self.rabbit_xy[:, 1]
            img[RABBIT_CHANNEL, xs, ys] = 1.0
        return img

    def observables(self) -> dict[str, float]:
        """Low-dimensional summary statistics (used for the Phase-0 DMDc check)."""
        w, h = self.cfg.width, self.cfg.height
        n = self.num_rabbits
        if n > 0:
            cx = float(self.rabbit_xy[:, 0].mean())
            cy = float(self.rabbit_xy[:, 1].mean())
            spread = float(self.rabbit_xy.std())
        else:
            cx = cy = spread = 0.0
        return {
            "rabbit_count": float(n),
            "grass_count": float(self.num_grass),
            "grass_frac": float(self.num_grass) / float(w * h),
            "rabbit_centroid_x": cx,
            "rabbit_centroid_y": cy,
            "rabbit_spread": spread,
        }

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def step(self, u: float = 0.0) -> None:
        """Advance one timestep under continuous control ``u in [0, 1]``."""
        u = float(np.clip(u, 0.0, 1.0))
        self.timestep += 1

        self._apply_cull(u)
        self._grow_grass()
        if self.num_rabbits > 0:
            self._move_eat_reproduce()

        self._u_prev = u

    def _apply_cull(self, u: float) -> None:
        eff = self.cfg.culling_effectiveness
        total = u * eff + self._u_prev * self.cfg.cull_delay_fraction * eff
        if total <= 0.0 or self.num_rabbits == 0:
            return
        keep = self.rng.random(self.num_rabbits) > total
        self.rabbit_xy = self.rabbit_xy[keep]
        self.rabbit_energy = self.rabbit_energy[keep]

    def _grow_grass(self) -> None:
        empty = self.grass_grid == 0
        sprout = empty & (self.rng.random(self.grass_grid.shape) < self.cfg.grass_growth_rate)
        self.grass_grid[sprout] = 1

    def _move_eat_reproduce(self) -> None:
        w, h = self.cfg.width, self.cfg.height
        neighborhood = np.array(
            [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)],
            dtype=np.int32,
        )
        new_xy: list[np.ndarray] = []
        new_energy: list[float] = []
        survive_mask = np.zeros(self.num_rabbits, dtype=bool)

        for i in range(self.num_rabbits):
            x, y = int(self.rabbit_xy[i, 0]), int(self.rabbit_xy[i, 1])

            # Move toward the first adjacent grass cell, else a random step.
            moved = False
            for dx, dy in neighborhood:
                nx, ny = (x + int(dx)) % w, (y + int(dy)) % h
                if self.grass_grid[nx, ny] == 1:
                    x, y = nx, ny
                    moved = True
                    break
            if not moved:
                x = (x + int(self.rng.integers(-1, 2))) % w
                y = (y + int(self.rng.integers(-1, 2))) % h

            e = float(self.rabbit_energy[i]) - self.cfg.rabbit_move_cost

            # Eat grass on the current cell.
            if self.grass_grid[x, y] == 1:
                self.grass_grid[x, y] = 0
                e += self.cfg.grass_energy

            # Reproduce when well-fed (child spawns on the same cell).
            if e > self.cfg.rabbit_birth_threshold:
                e /= 2.0
                new_xy.append(np.array([x, y], dtype=np.int32))
                new_energy.append(e)

            self.rabbit_xy[i, 0], self.rabbit_xy[i, 1] = x, y
            self.rabbit_energy[i] = e
            if e >= 0.0:
                survive_mask[i] = True

        xy = self.rabbit_xy[survive_mask]
        en = self.rabbit_energy[survive_mask]
        if new_xy:
            xy = np.concatenate([xy, np.stack(new_xy, axis=0)], axis=0)
            en = np.concatenate([en, np.asarray(new_energy, dtype=np.float32)], axis=0)
        self.rabbit_xy = xy.astype(np.int32)
        self.rabbit_energy = en.astype(np.float32)


def rollout(
    cfg: RabbitGrassConfig,
    control_seq: np.ndarray,
    *,
    initial_rabbits: int,
    initial_grass_prob: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Simulate one trajectory driven by a control sequence.

    Parameters
    ----------
    control_seq:
        Length-``T`` array of control inputs in ``[0, 1]``. One step is taken
        per entry.

    Returns
    -------
    frames:
        ``(T + 1, 2, W, H)`` float32 image observations, including the initial
        frame before any control is applied.
    controls:
        ``(T + 1,)`` float32 control aligned so ``controls[t]`` is the input that
        produced ``frames[t]`` (``controls[0] = 0`` for the initial frame). This
        alignment matches how the loaders build ``(x_t, x_{t+1}, u_t)`` pairs.
    observables:
        dict of ``(T + 1,)`` arrays of summary statistics per frame.
    """
    control_seq = np.asarray(control_seq, dtype=np.float32).reshape(-1)
    model = RabbitGrassModel(
        cfg=cfg,
        initial_rabbits=initial_rabbits,
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
