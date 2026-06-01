"""Sync-oriented beacon controllers for the firefly ABM (boolean ``external_flash``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from control_abms import BaseController


def beacon_on(control_inputs: dict[str, Any] | None) -> bool:
    """Interpret ABM / surrogate control dict as a boolean beacon command."""
    if not control_inputs:
        return False
    val = control_inputs.get("external_flash", False)
    if isinstance(val, bool):
        return val
    return float(val) > 0.5


def sliding_peak_flashing(F: np.ndarray, window: int) -> np.ndarray:
    """Causal sliding maximum of flashing counts (length ``len(F)``)."""
    F = np.asarray(F, dtype=float)
    w = max(1, int(window))
    out = np.empty_like(F)
    for i in range(len(F)):
        out[i] = float(F[max(0, i - w + 1) : i + 1].max())
    return out


def sync_target_count(n_total: float, sync_fraction: float) -> float:
    return float(sync_fraction) * float(n_total)


def sync_control_cost(
    hist: dict[str, Any],
    n_total: float,
    *,
    sync_fraction: float = 0.45,
    window: int = 20,
    warmup: int = 10,
    w_time: float = 1.5,
    w_peak: float = 2.0,
    w_effort: float = 4.0,
    w_always_on: float = 50.0,
    w_switch: float = 8.0,
) -> float:
    """Lower is better: reach high *peak* sync quickly with pulsed (not always-on) control.

    Uses sliding-window peak flashing (proxy for collective synchrony), not mean count.
    """
    F = np.asarray(hist.get("flashing", hist.get("flashing_count", [])), dtype=float)
    u = np.asarray(hist.get("u", hist.get("external_flash", [])), dtype=float)
    if F.size == 0:
        return 1e6

    target = sync_target_count(n_total, sync_fraction)
    peaks = sliding_peak_flashing(F, window)
    w = min(max(int(warmup), 0), len(F) - 1)

    time_to_sync = float(len(F) - w)
    for i in range(w, len(F)):
        if peaks[i] >= target:
            time_to_sync = float(i - w)
            break

    tail = peaks[w:]
    final_peak_deficit = max(0.0, target - float(tail[-min(15, len(tail)) :].max()))

    u_tail = u[w:]
    duty = float(u_tail.mean()) if u_tail.size else 0.0
    always_on_penalty = max(0.0, duty - 0.25) ** 2
    switch_rate = float(np.mean(np.abs(np.diff(u_tail)))) if u_tail.size > 1 else 0.0

    return (
        w_time * time_to_sync
        + w_peak * final_peak_deficit
        + w_effort * duty
        + w_always_on * always_on_penalty
        - w_switch * switch_rate
    )


class PeriodicBeaconController(BaseController):
    """Open-loop periodic centre beacon (entrainment prior)."""

    def __init__(self, period: int, duty: int, phase: int = 0) -> None:
        self.period = max(1, int(period))
        self.duty = max(1, min(int(duty), self.period))
        self.phase = int(phase) % self.period

    def compute(self, timestep: int, outputs: dict) -> dict:  # noqa: ARG002
        pos = (int(timestep) + self.phase) % self.period
        return {"external_flash": pos < self.duty}

    def reset(self) -> None:
        pass


class PeakSyncBeaconController(BaseController):
    """Pulse the beacon when recent peak sync is below target; enforced off-time between pulses."""

    def __init__(
        self,
        n_total: float,
        sync_fraction: float = 0.45,
        window: int = 25,
        flash_interval: int = 10,
    ) -> None:
        self.n_total = float(n_total)
        self.sync_fraction = float(sync_fraction)
        self.window = max(1, int(window))
        self.flash_interval = max(1, int(flash_interval))
        self._recent: list[int] = []
        self._last_flash_step = -self.flash_interval

    def compute(self, timestep: int, outputs: dict) -> dict:
        count = int(outputs.get("flashing_count", 0))
        self._recent.append(count)
        if len(self._recent) > self.window:
            self._recent.pop(0)

        peak = max(self._recent) if self._recent else count
        target = sync_target_count(self.n_total, self.sync_fraction)
        if peak >= target:
            return {"external_flash": False}
        if timestep - self._last_flash_step < self.flash_interval:
            return {"external_flash": False}

        self._last_flash_step = timestep
        return {"external_flash": True}

    def reset(self) -> None:
        self._recent.clear()
        self._last_flash_step = -self.flash_interval


def make_periodic_controller(period: float, duty: float, phase: float) -> PeriodicBeaconController:
    p = max(1, int(round(period)))
    d = max(1, min(int(round(duty)), p))
    ph = int(round(phase)) % p
    return PeriodicBeaconController(period=p, duty=d, phase=ph)


def make_peak_sync_controller(
    n_total: float,
    sync_fraction: float,
    window: float,
    flash_interval: float,
) -> PeakSyncBeaconController:
    return PeakSyncBeaconController(
        n_total=n_total,
        sync_fraction=float(sync_fraction),
        window=max(1, int(round(window))),
        flash_interval=max(1, int(round(flash_interval))),
    )
