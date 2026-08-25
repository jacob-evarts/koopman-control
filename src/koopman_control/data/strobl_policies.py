"""Scalar global-dose policies for the Strobl tumour model.

All schedule constructors return one-dimensional ``float32`` arrays.  Dose
bounds and an optional cumulative-dose budget are applied centrally so no
policy can accidentally produce spatial or per-cell actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _validate_horizon(horizon: int) -> int:
    horizon = int(horizon)
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    return horizon


def _bounded_schedule(
    doses: Iterable[float] | np.ndarray,
    *,
    d_max: float,
    cumulative_cap: float | None,
) -> np.ndarray:
    """Clip a scalar schedule to its pointwise and cumulative dose limits."""
    d_max = float(d_max)
    if not np.isfinite(d_max) or d_max < 0:
        raise ValueError(f"d_max must be finite and non-negative, got {d_max}")
    values = np.asarray(doses, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("a Strobl action schedule must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("dose schedule contains non-finite values")
    values = np.clip(values, 0.0, d_max)

    if cumulative_cap is not None:
        remaining = float(cumulative_cap)
        if not np.isfinite(remaining) or remaining < 0:
            raise ValueError("cumulative_cap must be finite and non-negative")
        capped = np.zeros_like(values)
        for index, dose in enumerate(values):
            administered = min(float(dose), remaining)
            capped[index] = administered
            remaining -= administered
            if remaining <= 0:
                break
        values = capped
    return values.astype(np.float32)


def open_loop(
    doses: Iterable[float] | np.ndarray,
    *,
    d_max: float = 1.0,
    cumulative_cap: float | None = None,
) -> np.ndarray:
    """Return a validated pre-specified scalar global-dose schedule."""
    return _bounded_schedule(doses, d_max=d_max, cumulative_cap=cumulative_cap)


def constant(
    horizon: int,
    dose: float,
    *,
    d_max: float = 1.0,
    cumulative_cap: float | None = None,
) -> np.ndarray:
    """Construct a constant global-dose schedule."""
    horizon = _validate_horizon(horizon)
    return _bounded_schedule(
        np.full(horizon, float(dose)),
        d_max=d_max,
        cumulative_cap=cumulative_cap,
    )


@dataclass
class AdaptivePolicy:
    """Stateful paper-style 50%-off/baseline-on scalar-dose policy."""

    initial_population: float
    d_max: float = 1.0
    restart_at_equal: bool = False
    cumulative_cap: float | None = None
    current_dose: float | None = None
    cumulative_dose: float = 0.0

    def __post_init__(self) -> None:
        self.initial_population = float(self.initial_population)
        self.d_max = float(self.d_max)
        if self.initial_population <= 0:
            raise ValueError("initial_population must be positive")
        if not np.isfinite(self.d_max) or self.d_max < 0:
            raise ValueError("d_max must be finite and non-negative")
        if self.cumulative_cap is not None and (
            not np.isfinite(self.cumulative_cap) or self.cumulative_cap < 0
        ):
            raise ValueError("cumulative_cap must be finite and non-negative")
        if self.current_dose is None:
            self.current_dose = self.d_max
        self.current_dose = float(np.clip(self.current_dose, 0.0, self.d_max))

    def reset(self) -> None:
        """Reset treatment to on and clear the cumulative administered dose."""
        self.current_dose = self.d_max
        self.cumulative_dose = 0.0

    def __call__(self, total_population: float) -> float:
        """Return the next dose after observing aggregate population ``N``."""
        population = float(total_population)
        if not np.isfinite(population) or population < 0:
            raise ValueError("total_population must be finite and non-negative")
        if population < 0.5 * self.initial_population:
            self.current_dose = 0.0
        elif population > self.initial_population or (
            self.restart_at_equal and population >= self.initial_population
        ):
            self.current_dose = self.d_max

        dose = float(self.current_dose)
        if self.cumulative_cap is not None:
            dose = min(dose, max(0.0, self.cumulative_cap - self.cumulative_dose))
        self.cumulative_dose += dose
        return dose


def paper_adaptive(
    initial_population: float,
    *,
    d_max: float = 1.0,
    cumulative_cap: float | None = None,
) -> AdaptivePolicy:
    """Create the released-source-compatible policy (restart only for ``N>N0``)."""
    return AdaptivePolicy(
        initial_population=initial_population,
        d_max=d_max,
        restart_at_equal=False,
        cumulative_cap=cumulative_cap,
    )


def paper_text_adaptive(
    initial_population: float,
    *,
    d_max: float = 1.0,
    cumulative_cap: float | None = None,
) -> AdaptivePolicy:
    """Create the paper-text variant that restarts for ``N>=N0``."""
    return AdaptivePolicy(
        initial_population=initial_population,
        d_max=d_max,
        restart_at_equal=True,
        cumulative_cap=cumulative_cap,
    )


def random_piecewise_constant(
    horizon: int,
    *,
    d_max: float = 1.0,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    n_segments: int | None = None,
    min_segments: int = 3,
    max_segments: int = 10,
    cumulative_cap: float | None = None,
) -> np.ndarray:
    """Draw a deterministic-when-seeded random 3--10 segment schedule."""
    horizon = _validate_horizon(horizon)
    if rng is not None and seed is not None:
        raise ValueError("pass either rng or seed, not both")
    generator = np.random.default_rng(seed) if rng is None else rng
    if horizon == 0:
        return np.empty(0, dtype=np.float32)

    lower = max(1, int(min_segments))
    upper = min(int(max_segments), horizon)
    if lower > upper:
        raise ValueError(
            f"segment range [{lower}, {max_segments}] is incompatible with horizon {horizon}"
        )
    if n_segments is None:
        n_segments = int(generator.integers(lower, upper + 1))
    n_segments = int(n_segments)
    if not lower <= n_segments <= upper:
        raise ValueError(f"n_segments must lie in [{lower}, {upper}]")

    cuts = (
        np.sort(generator.choice(np.arange(1, horizon), n_segments - 1, replace=False))
        if n_segments > 1
        else np.empty(0, dtype=int)
    )
    edges = np.concatenate(([0], cuts, [horizon]))
    levels = generator.uniform(0.0, float(d_max), size=n_segments)
    schedule = np.empty(horizon, dtype=np.float64)
    for level, start, stop in zip(levels, edges[:-1], edges[1:], strict=True):
        schedule[start:stop] = level
    return _bounded_schedule(schedule, d_max=d_max, cumulative_cap=cumulative_cap)


def pulses(
    horizon: int,
    *,
    d_max: float = 1.0,
    pulse_dose: float | None = None,
    width: int = 1,
    period: int | None = None,
    starts: Iterable[int] | None = None,
    cumulative_cap: float | None = None,
) -> np.ndarray:
    """Construct rectangular scalar-dose pulses from a period or start indices."""
    horizon = _validate_horizon(horizon)
    width = int(width)
    if width <= 0:
        raise ValueError("width must be positive")
    if (period is None) == (starts is None):
        raise ValueError("provide exactly one of period or starts")
    if period is not None:
        period = int(period)
        if period <= 0:
            raise ValueError("period must be positive")
        pulse_starts = range(0, horizon, period)
    else:
        pulse_starts = [int(start) for start in starts or ()]

    schedule = np.zeros(horizon, dtype=np.float64)
    level = float(d_max if pulse_dose is None else pulse_dose)
    for start in pulse_starts:
        if start < 0:
            raise ValueError("pulse starts must be non-negative")
        if start < horizon:
            schedule[start : min(horizon, start + width)] = level
    return _bounded_schedule(schedule, d_max=d_max, cumulative_cap=cumulative_cap)


__all__ = [
    "AdaptivePolicy",
    "constant",
    "open_loop",
    "paper_adaptive",
    "paper_text_adaptive",
    "pulses",
    "random_piecewise_constant",
]
