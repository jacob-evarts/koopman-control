"""Phase 0: DMDc identifiability check on low-dimensional observables.

Why this file exists
--------------------
Before training a deep image encoder, we want to know whether the *data* can
support a linear-with-control model at all. This is a cheap, model-free
sanity check: fit Dynamic Mode Decomposition with control (DMDc)

    z_{t+1} = A z_t + B u_t + c

directly on hand-crafted aggregate observables (rabbit/grass counts, rabbit
centroid, spread). It separates two failure modes:

  * "the data / actuator is uninformative" (fails here), vs
  * "the deep model is wrong" (would fail later).

If even this low-dimensional linear model tracks held-out trajectories, then a
learned latent has a reasonable linear target to aim for. If it does not, the
data-generation design must change before spending effort on the encoder.

The fit augments the input with control *history* ``[u_t, u_{t-1}]`` because the
simulator's actuator has a one-step lag; a purely memoryless input model would
be misspecified.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from koopman_control.paths import dataset_path


@dataclass
class DMDcModel:
    """A fitted ``z_{t+1} = A z_t + B u_t + c`` model on standardized observables."""

    A: np.ndarray  # (d, d)
    B: np.ndarray  # (d, m) where m = number of control lags
    c: np.ndarray  # (d,)
    mean: np.ndarray  # (d,) standardization mean
    std: np.ndarray  # (d,) standardization std
    obs_names: list[str]
    n_control_lags: int

    def standardize(self, z: np.ndarray) -> np.ndarray:
        return (z - self.mean) / self.std

    def unstandardize(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean

    def eig(self) -> np.ndarray:
        return np.linalg.eigvals(self.A)

    def controllability_rank(self) -> int:
        """Rank of the controllability matrix [B, AB, A^2 B, ...]."""
        d = self.A.shape[0]
        blocks = [self.B]
        for _ in range(1, d):
            blocks.append(self.A @ blocks[-1])
        ctrb = np.concatenate(blocks, axis=1)
        return int(np.linalg.matrix_rank(ctrb, tol=1e-8))


def _control_features(u: np.ndarray, t: int, n_lags: int) -> np.ndarray:
    """Return ``[u_t, u_{t-1}, ...]`` (zero-padded at the start)."""
    return np.array([u[t - k] if t - k >= 0 else 0.0 for k in range(n_lags)], dtype=np.float64)


def load_runs(h5_path: Path) -> tuple[list[dict], list[str]]:
    """Load observable trajectories grouped by split from a generated dataset."""
    runs: list[dict] = []
    with h5py.File(h5_path, "r") as f:
        obs_names = [s.decode() if isinstance(s, bytes) else str(s) for s in f.attrs["obs_names"]]
        for rid, grp in f["runs"].items():
            runs.append(
                {
                    "run_id": rid,
                    "split": grp.attrs["split"],
                    "excitation": grp.attrs["excitation"],
                    "Z": np.asarray(grp["obs"][:], dtype=np.float64),
                    "u": np.asarray(grp["control"][:], dtype=np.float64),
                }
            )
    return runs, obs_names


def fit_dmdc(
    runs: list[dict],
    obs_names: list[str],
    *,
    n_control_lags: int = 2,
) -> DMDcModel:
    """Least-squares fit of ``z_{t+1} = A z_t + B [u_t..u_{t-L+1}] + c``."""
    train = [r for r in runs if r["split"] == "train"]
    if not train:
        raise ValueError("No training runs found in dataset.")

    all_z = np.concatenate([r["Z"] for r in train], axis=0)
    mean = all_z.mean(axis=0)
    std = all_z.std(axis=0)
    std[std < 1e-6] = 1.0
    d = all_z.shape[1]

    rows_in, rows_out = [], []
    for r in train:
        z = (r["Z"] - mean) / std
        u = r["u"]
        for t in range(len(z) - 1):
            uc = _control_features(u, t, n_control_lags)
            rows_in.append(np.concatenate([z[t], uc, [1.0]]))
            rows_out.append(z[t + 1])

    phi = np.asarray(rows_in)  # (N, d + L + 1)
    y = np.asarray(rows_out)  # (N, d)
    w, *_ = np.linalg.lstsq(phi, y, rcond=None)
    w = w.T  # (d, d + L + 1)

    A = w[:, :d]
    B = w[:, d : d + n_control_lags]
    c = w[:, d + n_control_lags]
    return DMDcModel(
        A=A,
        B=B,
        c=c,
        mean=mean,
        std=std,
        obs_names=obs_names,
        n_control_lags=n_control_lags,
    )


def one_step_r2(model: DMDcModel, runs: list[dict], split: str) -> float:
    """Coefficient of determination for one-step predictions on a split."""
    sel = [r for r in runs if r["split"] == split]
    if not sel:
        return float("nan")
    preds, tgts = [], []
    for r in sel:
        z = model.standardize(r["Z"])
        u = r["u"]
        for t in range(len(z) - 1):
            uc = _control_features(u, t, model.n_control_lags)
            preds.append(model.A @ z[t] + model.B @ uc + model.c)
            tgts.append(z[t + 1])
    preds = np.asarray(preds)
    tgts = np.asarray(tgts)
    ss_res = float(((preds - tgts) ** 2).sum())
    ss_tot = float(((tgts - tgts.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def rollout_error(model: DMDcModel, runs: list[dict], split: str) -> dict[str, float]:
    """Free-running multi-step rollout error per observable (original units).

    Starting from each trajectory's first observation, apply the learned model
    with the trajectory's true control sequence and compare against the true
    (simulated) observables. This is the honest test of whether the model
    captured the controlled dynamics, not just one-step correlation.
    """
    sel = [r for r in runs if r["split"] == split]
    if not sel:
        return {}
    per_obs_sq: list[np.ndarray] = []
    for r in sel:
        z0 = model.standardize(r["Z"])[0]
        u = r["u"]
        z = z0.copy()
        pred_series = [z.copy()]
        for t in range(len(r["Z"]) - 1):
            uc = _control_features(u, t, model.n_control_lags)
            z = model.A @ z + model.B @ uc + model.c
            pred_series.append(z.copy())
        pred = model.unstandardize(np.asarray(pred_series))
        per_obs_sq.append(((pred - r["Z"]) ** 2).mean(axis=0))
    rmse = np.sqrt(np.mean(per_obs_sq, axis=0))
    return {name: float(v) for name, v in zip(model.obs_names, rmse)}


def run(h5_path: Path, *, n_control_lags: int = 2) -> dict:
    runs, obs_names = load_runs(h5_path)
    model = fit_dmdc(runs, obs_names, n_control_lags=n_control_lags)

    report = {
        "obs_names": obs_names,
        "n_control_lags": n_control_lags,
        "one_step_r2": {
            "train": one_step_r2(model, runs, "train"),
            "val": one_step_r2(model, runs, "val"),
            "test": one_step_r2(model, runs, "test"),
        },
        "rollout_rmse": {
            "train": rollout_error(model, runs, "train"),
            "val": rollout_error(model, runs, "val"),
            "test": rollout_error(model, runs, "test"),
        },
        "eigenvalues_abs": sorted(np.abs(model.eig()).tolist(), reverse=True),
        "controllability_rank": model.controllability_rank(),
        "state_dim": int(model.A.shape[0]),
        "control_dim": int(model.B.shape[1]),
    }
    return report


def _fmt(report: dict) -> str:
    lines = ["=== Phase 0 DMDc identifiability report ===", ""]
    lines.append(f"observables ({report['state_dim']}): {', '.join(report['obs_names'])}")
    lines.append(f"control lags: {report['n_control_lags']}")
    lines.append("")
    lines.append("one-step R^2 (standardized):")
    for k, v in report["one_step_r2"].items():
        lines.append(f"  {k:5s}: {v:.4f}")
    lines.append("")
    lines.append("free-running rollout RMSE (original units), test split:")
    for name, v in (report["rollout_rmse"].get("test") or {}).items():
        lines.append(f"  {name:18s}: {v:.3f}")
    lines.append("")
    lines.append(f"controllability rank: {report['controllability_rank']} / {report['state_dim']}")
    lines.append("spectral radius (max |eig|): " f"{report['eigenvalues_abs'][0]:.4f}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=dataset_path(),
        help="Path to generated dataset.h5",
    )
    p.add_argument("--control-lags", type=int, default=2)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(args.dataset, n_control_lags=args.control_lags)
    print(_fmt(report))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"\nWrote JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
