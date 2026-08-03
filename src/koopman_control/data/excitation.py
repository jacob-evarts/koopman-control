"""Control-excitation signal generators for system identification.

Why this file exists
--------------------
To learn how control enters the dynamics, the training trajectories must
*excite* the input. In classical system identification the quality of an
identified model is limited by the "persistency of excitation" of the input:
the input must be rich enough (in amplitude and frequency) to reveal the
system's response. The old data used a handful of fixed policies with a binary
actuator, which is poor excitation.

These generators produce randomized, continuous control sequences in ``[0, 1]``
designed to cover the amplitude/frequency space:

  * ``random_piecewise_constant`` (RPWC): random amplitudes held for random
    dwell times. Broadband and the workhorse for identification.
  * ``prbs``: pseudo-random binary-ish switching at random amplitudes; strong,
    broadband, near-persistently-exciting.
  * ``amplitude_staircase``: holds a sequence of increasing/decreasing levels;
    directly tests whether the response scales linearly with amplitude.
  * ``ramp``: slow linear sweep of amplitude.
  * ``chirp``: sinusoid with increasing frequency, offset into ``[0, 1]``;
    probes the frequency response.
  * ``constant`` / ``zero``: baselines (uncontrolled and saturated) so the model
    also sees the un-actuated manifold and the extreme.

Each generator takes an ``rng`` so trajectories are reproducible from a seed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def zero(length: int, rng: np.random.Generator) -> np.ndarray:  # noqa: ARG001
    """Uncontrolled baseline (the natural, un-actuated dynamics)."""
    return np.zeros(length, dtype=np.float32)


def constant(
    length: int,
    rng: np.random.Generator,
    *,
    level: float | None = None,
) -> np.ndarray:
    """Hold a single (possibly random) amplitude for the whole trajectory."""
    lvl = float(rng.uniform(0.2, 1.0)) if level is None else float(level)
    return np.full(length, np.clip(lvl, 0.0, 1.0), dtype=np.float32)


def random_piecewise_constant(
    length: int,
    rng: np.random.Generator,
    *,
    min_dwell: int = 3,
    max_dwell: int = 20,
) -> np.ndarray:
    """Random amplitudes in ``[0, 1]`` held for random dwell times.

    Broadband and amplitude-rich: the primary identification signal.
    """
    out = np.empty(length, dtype=np.float32)
    i = 0
    while i < length:
        dwell = int(rng.integers(min_dwell, max_dwell + 1))
        level = float(rng.uniform(0.0, 1.0))
        out[i : i + dwell] = level
        i += dwell
    return out[:length]


def prbs(
    length: int,
    rng: np.random.Generator,
    *,
    switch_prob: float = 0.3,
) -> np.ndarray:
    """Pseudo-random switching between random hold levels.

    Like a PRBS but with random (not just binary) amplitudes, giving strong
    broadband excitation while still testing amplitude scaling.
    """
    out = np.empty(length, dtype=np.float32)
    level = float(rng.uniform(0.0, 1.0))
    for t in range(length):
        if rng.random() < switch_prob:
            level = float(rng.uniform(0.0, 1.0))
        out[t] = level
    return out


def amplitude_staircase(
    length: int,
    rng: np.random.Generator,
    *,
    n_levels: int | None = None,
) -> np.ndarray:
    """Monotonic staircase of held levels; directly probes amplitude linearity."""
    k = int(rng.integers(3, 7)) if n_levels is None else int(n_levels)
    levels = np.linspace(0.0, 1.0, k)
    if rng.random() < 0.5:
        levels = levels[::-1]
    seg = max(1, length // k)
    out = np.concatenate([np.full(seg, lv, dtype=np.float32) for lv in levels])
    if out.shape[0] < length:
        out = np.concatenate([out, np.full(length - out.shape[0], out[-1], dtype=np.float32)])
    return out[:length].astype(np.float32)


def ramp(length: int, rng: np.random.Generator) -> np.ndarray:
    """Slow linear amplitude sweep (up or down)."""
    lo, hi = float(rng.uniform(0.0, 0.3)), float(rng.uniform(0.7, 1.0))
    out = np.linspace(lo, hi, length).astype(np.float32)
    if rng.random() < 0.5:
        out = out[::-1].copy()
    return out


def chirp(length: int, rng: np.random.Generator) -> np.ndarray:
    """Frequency-swept sinusoid offset into ``[0, 1]``; probes frequency response."""
    t = np.arange(length, dtype=np.float32)
    f0 = float(rng.uniform(0.005, 0.02))
    f1 = float(rng.uniform(0.05, 0.2))
    k = (f1 - f0) / max(1, length)
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
    amp = float(rng.uniform(0.3, 0.5))
    offset = float(rng.uniform(amp, 1.0 - amp))
    return np.clip(offset + amp * np.sin(phase), 0.0, 1.0).astype(np.float32)


# Registry mapping a signal name to its generator. The names double as the
# "excitation type" recorded per trajectory so splits can hold out whole types.
SIGNALS: dict[str, Callable[..., np.ndarray]] = {
    "zero": zero,
    "constant": constant,
    "rpwc": random_piecewise_constant,
    "prbs": prbs,
    "staircase": amplitude_staircase,
    "ramp": ramp,
    "chirp": chirp,
}


def make_control(name: str, length: int, rng: np.random.Generator) -> np.ndarray:
    """Build a control sequence by registry name."""
    if name not in SIGNALS:
        raise KeyError(f"Unknown excitation signal {name!r}; choices: {sorted(SIGNALS)}")
    return SIGNALS[name](length, rng)
