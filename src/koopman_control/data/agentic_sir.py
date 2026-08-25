"""Spatial agent-based SIR epidemic with vaccination control.

This is Case B of the spatial-control study: a lattice SIR ABM in the spirit of
classic neighborhood epidemic models (e.g. Sirakoulis et al.; urban mobility /
contact-radius ABMs). Agents move on a grid, transmit infection locally, and
recover permanently. The scalar control ``u in [0, 1]`` is a systemic
vaccination intensity that converts susceptibles to recovered.

The plant is phenomenological and intended as a control benchmark, not a
calibrated disease model. It creates:

* localized outbreak geometry that spreads as a spatial wave,
* delayed / incomplete containment under partial vaccination,
* competing infection-reduction and vaccination-effort objectives, and
* image states whose geometry matters beyond ``(S, I, R)`` totals.

Rendered channels are binary occupancy ``[susceptible, infected, recovered]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SUSCEPTIBLE_CHANNEL = 0
INFECTED_CHANNEL = 1
RECOVERED_CHANNEL = 2
NUM_CHANNELS = 3

STATE_S = 0
STATE_I = 1
STATE_R = 2

# Eight Moore neighbors used for movement and local infectious pressure.
_MOORE = np.array(
    [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class AgenticSIRConfig:
    """Fixed dynamics parameters for the spatial SIR ABM."""

    width: int = 64
    height: int = 64

    # Per-contact infection probability when a susceptible shares a cell with,
    # or is adjacent to, at least one infectious agent.
    infection_prob: float = 0.45
    # Per-step recovery probability I -> R.
    recovery_prob: float = 0.05
    # Probability an agent attempts a random Moore move each step.
    move_prob: float = 0.85
    # Max per-step S -> R vaccination probability at u = 1.
    # Tuned so constant-u dose response stays graded on [0, 1].
    vaccine_effectiveness: float = 0.045
    # One-step actuator lag on vaccination (campaign rollout delay).
    vaccine_delay_fraction: float = 0.4


@dataclass
class AgenticSIRModel:
    """One stochastic spatial SIR rollout with mobile agents."""

    cfg: AgenticSIRConfig
    n_agents: int = 500
    initial_infected: int = 12
    seed_center_x: float | None = None
    seed_center_y: float | None = None
    seed_radius: float = 4.0
    seed: int | None = None

    rng: np.random.Generator = field(init=False)
    xy: np.ndarray = field(init=False)
    state: np.ndarray = field(init=False)
    timestep: int = field(init=False, default=0)
    cumulative_incidence: float = field(init=False, default=0.0)
    _u_prev: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        w, h = self.cfg.width, self.cfg.height
        n = int(self.n_agents)
        if n <= 0:
            raise ValueError(f"n_agents must be positive, got {n}")

        self.xy = np.stack(
            [
                self.rng.integers(0, w, size=n),
                self.rng.integers(0, h, size=n),
            ],
            axis=1,
        ).astype(np.int32)
        self.state = np.full(n, STATE_S, dtype=np.int8)

        cx = (w - 1) / 2.0 if self.seed_center_x is None else float(self.seed_center_x)
        cy = (h - 1) / 2.0 if self.seed_center_y is None else float(self.seed_center_y)
        dist = np.sqrt((self.xy[:, 0] - cx) ** 2 + (self.xy[:, 1] - cy) ** 2)
        order = np.argsort(dist)
        n_seed = min(int(self.initial_infected), n)
        # Prefer agents already near the seed; if too few fall inside the
        # radius, take the nearest agents so outbreaks always start localized.
        in_disk = order[dist[order] <= float(self.seed_radius)]
        if in_disk.size >= n_seed:
            chosen = in_disk[:n_seed]
        else:
            chosen = order[:n_seed]
        self.state[chosen] = STATE_I

        self.timestep = 0
        self.cumulative_incidence = float(n_seed)
        self._u_prev = 0.0

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    @property
    def num_susceptible(self) -> int:
        return int((self.state == STATE_S).sum())

    @property
    def num_infected(self) -> int:
        return int((self.state == STATE_I).sum())

    @property
    def num_recovered(self) -> int:
        return int((self.state == STATE_R).sum())

    def render(self) -> np.ndarray:
        """``(3, W, H)`` float32 occupancy: susceptible / infected / recovered."""
        w, h = self.cfg.width, self.cfg.height
        img = np.zeros((NUM_CHANNELS, w, h), dtype=np.float32)
        for channel, label in (
            (SUSCEPTIBLE_CHANNEL, STATE_S),
            (INFECTED_CHANNEL, STATE_I),
            (RECOVERED_CHANNEL, STATE_R),
        ):
            mask = self.state == label
            if not mask.any():
                continue
            xs, ys = self.xy[mask, 0], self.xy[mask, 1]
            img[channel, xs, ys] = 1.0
        return img

    def observables(self) -> dict[str, float]:
        w, h = self.cfg.width, self.cfg.height
        n = float(self.xy.shape[0])
        infected = self.state == STATE_I
        if infected.any():
            ix = self.xy[infected, 0].astype(np.float64)
            iy = self.xy[infected, 1].astype(np.float64)
            infected_centroid_x = float(ix.mean())
            infected_centroid_y = float(iy.mean())
            infected_spread = float(np.sqrt(((ix - ix.mean()) ** 2 + (iy - iy.mean()) ** 2).mean()))
        else:
            infected_centroid_x = infected_centroid_y = infected_spread = 0.0
        return {
            "susceptible_count": float(self.num_susceptible),
            "infected_count": float(self.num_infected),
            "recovered_count": float(self.num_recovered),
            "susceptible_frac": float(self.num_susceptible) / n,
            "infected_frac": float(self.num_infected) / n,
            "recovered_frac": float(self.num_recovered) / n,
            "cumulative_incidence": float(self.cumulative_incidence),
            "infected_centroid_x": infected_centroid_x,
            "infected_centroid_y": infected_centroid_y,
            "infected_spread": infected_spread,
            "population": n,
            "grid_cells": float(w * h),
        }

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def step(self, u: float = 0.0) -> None:
        """Advance one step under vaccination intensity ``u``."""
        u = float(np.clip(u, 0.0, 1.0))
        self.timestep += 1
        self._move()
        self._transmit()
        self._recover()
        self._vaccinate(u)
        self._u_prev = u

    def _move(self) -> None:
        w, h = self.cfg.width, self.cfg.height
        moving = self.rng.random(self.xy.shape[0]) < self.cfg.move_prob
        if not moving.any():
            return
        # Uniform choice among the eight Moore neighbors (no stay).
        dirs = self.rng.integers(0, 8, size=int(moving.sum()))
        deltas = _MOORE[dirs]
        self.xy[moving, 0] = (self.xy[moving, 0] + deltas[:, 0]) % w
        self.xy[moving, 1] = (self.xy[moving, 1] + deltas[:, 1]) % h

    def _infectious_pressure(self) -> np.ndarray:
        """Boolean grid: cell is exposed if any infected agent is here or adjacent."""
        w, h = self.cfg.width, self.cfg.height
        infected = self.state == STATE_I
        pressure = np.zeros((w, h), dtype=bool)
        if not infected.any():
            return pressure
        xs, ys = self.xy[infected, 0], self.xy[infected, 1]
        # Same-cell contact.
        pressure[xs, ys] = True
        for dx, dy in _MOORE:
            pressure[(xs + int(dx)) % w, (ys + int(dy)) % h] = True
        return pressure

    def _transmit(self) -> None:
        susceptible = self.state == STATE_S
        if not susceptible.any() or self.num_infected == 0:
            return
        pressure = self._infectious_pressure()
        exposed = pressure[self.xy[:, 0], self.xy[:, 1]] & susceptible
        if not exposed.any():
            return
        newly = exposed & (self.rng.random(self.xy.shape[0]) < self.cfg.infection_prob)
        n_new = int(newly.sum())
        if n_new:
            self.state[newly] = STATE_I
            self.cumulative_incidence += float(n_new)

    def _recover(self) -> None:
        infected = self.state == STATE_I
        if not infected.any():
            return
        recover = infected & (self.rng.random(self.xy.shape[0]) < self.cfg.recovery_prob)
        self.state[recover] = STATE_R

    def _vaccinate(self, u: float) -> None:
        """Convert susceptibles to recovered under lagged vaccination intensity."""
        eff = self.cfg.vaccine_effectiveness
        rate = u * eff + self._u_prev * self.cfg.vaccine_delay_fraction * eff
        rate = float(np.clip(rate, 0.0, 1.0))
        if rate <= 0.0:
            return
        susceptible = self.state == STATE_S
        if not susceptible.any():
            return
        vaccinated = susceptible & (self.rng.random(self.xy.shape[0]) < rate)
        self.state[vaccinated] = STATE_R


def rollout(
    cfg: AgenticSIRConfig,
    control_seq: np.ndarray,
    *,
    n_agents: int = 500,
    initial_infected: int = 12,
    seed_center_x: float | None = None,
    seed_center_y: float | None = None,
    seed_radius: float = 4.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Simulate one trajectory with the shared dataset return layout."""
    control_seq = np.asarray(control_seq, dtype=np.float32).reshape(-1)
    model = AgenticSIRModel(
        cfg=cfg,
        n_agents=n_agents,
        initial_infected=initial_infected,
        seed_center_x=seed_center_x,
        seed_center_y=seed_center_y,
        seed_radius=seed_radius,
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
