from __future__ import annotations

import numpy as np

from surrogate_control.firefly_controllers import (
    sliding_peak_flashing,
    sync_target_count,
)


def compute_sync_metrics_firefly(
    F_mat: np.ndarray,
    u_mat: np.ndarray,
    n_total: float,
    *,
    sync_fraction: float = 0.45,
    window: int = 20,
    warmup: int = 20,
) -> dict[str, float]:
    """Replica-wise sync metrics; returns means across seeds."""
    target = sync_target_count(n_total, sync_fraction)
    times, peaks, duties, switches = [], [], [], []

    for F, u in zip(F_mat, u_mat, strict=True):
        peaks_arr = sliding_peak_flashing(F, window)
        w = min(warmup, len(F) - 1)
        t_sync = len(F) - w
        for i in range(w, len(F)):
            if peaks_arr[i] >= target:
                t_sync = i - w
                break
        times.append(float(t_sync))
        peaks.append(float(peaks_arr[w:].max()))
        u_ss = u[w:]
        duties.append(float(u_ss.mean()) if u_ss.size else 0.0)
        switches.append(float(np.mean(np.abs(np.diff(u_ss)))) if u_ss.size > 1 else 0.0)

    return {
        "mean_time_to_sync": float(np.mean(times)),
        "std_time_to_sync": float(np.std(times)),
        "mean_peak_flashing": float(np.mean(peaks)),
        "mean_control_duty": float(np.mean(duties)),
        "mean_switch_rate": float(np.mean(switches)),
        "sync_fraction_target": float(sync_fraction),
        "sync_target_count": float(target),
    }
