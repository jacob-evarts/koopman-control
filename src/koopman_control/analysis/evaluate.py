"""Evaluate a trained latent world model.

Why this file exists
--------------------
The system is stochastic at the pixel level (individual agents move
pseudo-randomly), so exact next-frame reconstruction is the wrong yardstick.
These analyses instead judge the model on the **macrostate** -- aggregate
occupancy mass (a proxy for population), multi-step latent-prediction accuracy,
whether the learned control has the right *sign and magnitude* of effect, and
the structure of the linear operator ``A``.

Every function is plain and importable so the companion notebook
(``worldmodel_eval.ipynb``) is a thin display layer over tested code.

Occupancy "mass"
----------------
For a decoded (or true) frame, the per-channel spatial sum of the occupancy
probability is used as the macrostate observable: ``rabbit_mass`` ~ number of
occupied rabbit cells, ``grass_mass`` ~ amount of grass. This is the coarse
quantity we care about and can predict, unlike exact pixel positions.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from koopman_control.data.rabbit_grass import (
    GRASS_CHANNEL,
    RABBIT_CHANNEL,
    RabbitGrassConfig,
    rollout,
)
from koopman_control.models.world_model import LatentWorldModel


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_model(ckpt_path: str | Path) -> LatentWorldModel:
    model = LatentWorldModel.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model.eval()
    return model


def load_split_trajectories(
    h5_path: str | Path,
    split: str,
    max_runs: int | None = None,
) -> list[dict]:
    """Return full-length trajectories (frames/controls/obs + metadata) for a split.

    ``frames`` stay ``uint8`` (the observations are binary occupancy) and are cast
    to float only where they are consumed. A split is ~80 runs x 201 frames x
    2 x 64 x 64, which is 138 MB as ``uint8`` but 550 MB as ``float32``, and the
    probe analyses need the train split resident at the same time.
    """
    trajs: list[dict] = []
    with h5py.File(h5_path, "r") as f:
        obs_names = [s.decode() if isinstance(s, bytes) else str(s) for s in f.attrs["obs_names"]]
        for rid, grp in f["runs"].items():
            if grp.attrs["split"] != split:
                continue
            trajs.append(
                {
                    "run_id": rid,
                    "excitation": grp.attrs["excitation"],
                    "initial_rabbits": int(grp.attrs["initial_rabbits"]),
                    "initial_grass_prob": float(grp.attrs["initial_grass_prob"]),
                    "seed": int(grp.attrs["seed"]),
                    "frames": np.asarray(grp["frames"][:], dtype=np.uint8),
                    "controls": np.asarray(grp["control"][:], dtype=np.float32),
                    "obs": np.asarray(grp["obs"][:], dtype=np.float32),
                }
            )
            if max_runs is not None and len(trajs) >= max_runs:
                break
    for t in trajs:
        t["obs_names"] = obs_names
    return trajs


# ----------------------------------------------------------------------
# Core latent operations
# ----------------------------------------------------------------------
@torch.no_grad()
def encode_trajectory(model: LatentWorldModel, frames: np.ndarray) -> np.ndarray:
    """Encode ``(T+1, C, W, H)`` frames to ``(T+1, latent_dim)``."""
    x = torch.from_numpy(np.asarray(frames, dtype=np.float32))
    return model.encode(x).cpu().numpy()


def _u_feat(controls: np.ndarray, k: int, n_lags: int) -> torch.Tensor:
    """Control history ``[u_{k+1}, u_k, ...]`` for the transition ``k -> k+1``."""
    feats = [controls[k + 1 - j] if (k + 1 - j) >= 0 else 0.0 for j in range(n_lags)]
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)


@torch.no_grad()
def latent_rollout(
    model: LatentWorldModel,
    z0: np.ndarray,
    controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Free-running latent rollout from ``z0`` under ``controls``.

    Returns ``(z_full, z_lin)`` each ``(T+1, d)`` where index 0 is ``z0`` and the
    rest are predicted. ``z_lin`` uses only the linear core ``A z + B u``.
    """
    n_lags = int(model.hparams.n_control_lags)
    t = len(controls) - 1
    z = torch.from_numpy(np.asarray(z0, dtype=np.float32)).unsqueeze(0)
    z_lin = z.clone()
    full = [z.squeeze(0).numpy()]
    lin = [z_lin.squeeze(0).numpy()]
    for k in range(t):
        uf = _u_feat(controls, k, n_lags)
        z = model.step(z, uf)
        z_lin = model.linear_step(z_lin, uf)
        full.append(z.squeeze(0).numpy())
        lin.append(z_lin.squeeze(0).numpy())
    return np.asarray(full), np.asarray(lin)


@torch.no_grad()
def encode_all(model: LatentWorldModel, trajs: list[dict]) -> dict:
    """Encode every trajectory once and stack the results.

    Trajectories are encoded one at a time (a whole split of raw frames does not
    fit comfortably in memory) but the latents are tiny, so everything
    downstream can work on dense arrays.

    Returns arrays aligned on ``(n_runs, T+1, ...)``: ``z``, ``controls``,
    ``obs``, plus ``obs_names`` and per-run ``meta``.
    """
    t_min = min(len(t["controls"]) for t in trajs)
    z = np.stack([encode_trajectory(model, tr["frames"])[:t_min] for tr in trajs])
    controls = np.stack([tr["controls"][:t_min] for tr in trajs])
    obs = np.stack([tr["obs"][:t_min] for tr in trajs])
    return {
        "z": z,
        "controls": controls,
        "obs": obs,
        "obs_names": trajs[0]["obs_names"],
        "meta": [
            {k: tr[k] for k in ("run_id", "excitation", "initial_rabbits", "seed")} for tr in trajs
        ],
    }


def _u_feat_batch(controls: np.ndarray, k: int, n_lags: int) -> torch.Tensor:
    """Batched control history ``(N, n_lags)`` for the transition ``k -> k+1``."""
    cols = []
    for j in range(n_lags):
        idx = k + 1 - j
        cols.append(controls[:, idx] if idx >= 0 else np.zeros(controls.shape[0]))
    return torch.tensor(np.stack(cols, axis=-1), dtype=torch.float32)


@torch.no_grad()
def latent_rollout_batch(
    model: LatentWorldModel,
    z0: np.ndarray,
    controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Free-running rollout for many trajectories at once.

    ``z0`` is ``(N, d)`` and ``controls`` is ``(N, T+1)``. Returns
    ``(z_full, z_lin)`` each ``(N, T+1, d)``. Batching matters: a split has ~80
    runs of ~200 steps, and looping one run at a time in Python dominates the
    notebook's runtime.
    """
    n_lags = int(model.hparams.n_control_lags)
    z = torch.from_numpy(np.asarray(z0, dtype=np.float32))
    z_lin = z.clone()
    full, lin = [z.numpy()], [z_lin.numpy()]
    for k in range(controls.shape[1] - 1):
        uf = _u_feat_batch(controls, k, n_lags)
        z = model.step(z, uf)
        z_lin = model.linear_step(z_lin, uf)
        full.append(z.numpy())
        lin.append(z_lin.numpy())
    return np.stack(full, axis=1), np.stack(lin, axis=1)


@torch.no_grad()
def decode_mass(model: LatentWorldModel, z: np.ndarray, *, chunk: int = 512) -> np.ndarray:
    """Decode latents ``(N, d)`` to per-channel occupancy mass ``(N, C)``.

    Decoded in chunks: the latents are tiny but the images are not, and a whole
    split at once is several hundred megabytes.
    """
    zt = torch.from_numpy(np.asarray(z, dtype=np.float32).reshape(-1, z.shape[-1]))
    out = []
    for i in range(0, len(zt), chunk):
        imgs = model.decode(zt[i : i + chunk]).cpu().numpy()
        out.append(imgs.reshape(imgs.shape[0], imgs.shape[1], -1).sum(axis=-1))
    mass = np.concatenate(out, axis=0)
    return mass.reshape(*np.asarray(z).shape[:-1], mass.shape[-1])


def true_mass(frames: np.ndarray) -> np.ndarray:
    """Per-channel occupancy mass ``(T+1, C)`` from true frames."""
    f = np.asarray(frames, dtype=np.float32)
    return f.reshape(f.shape[0], f.shape[1], -1).sum(axis=-1)


# ----------------------------------------------------------------------
# Aggregate metrics
# ----------------------------------------------------------------------
def macrostate_correlation(
    model: LatentWorldModel, trajs: list[dict], *, enc: dict | None = None
) -> dict:
    """Agreement between predicted and true occupancy mass over free rollouts."""
    enc = enc if enc is not None else encode_all(model, trajs)
    z_full, _ = latent_rollout_batch(model, enc["z"][:, 0], enc["controls"])
    t_min = enc["z"].shape[1]
    mass = decode_mass(model, z_full)
    p = mass.reshape(-1, mass.shape[-1])
    q = np.concatenate([true_mass(tr["frames"])[:t_min] for tr in trajs], axis=0)

    out: dict[str, float] = {}
    for name, ch in (("rabbit", RABBIT_CHANNEL), ("grass", GRASS_CHANNEL)):
        a, b = p[:, ch], q[:, ch]
        out[f"{name}_mass_corr"] = (
            float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        )
        out[f"{name}_mass_rmse"] = float(np.sqrt(np.mean((a - b) ** 2)))
        out[f"{name}_mass_bias"] = float(np.mean(a - b))
    return out


# ----------------------------------------------------------------------
# Prediction accuracy with baselines
# ----------------------------------------------------------------------
def horizon_errors(
    model: LatentWorldModel,
    trajs: list[dict],
    *,
    enc: dict | None = None,
    fitted: dict | None = None,
) -> dict:
    """Latent error vs rollout horizon for the model and reference baselines.

    A raw latent MSE is uninterpretable on its own -- the latent has no physical
    units and its scale is set by the VICReg variance target. Three references
    make it readable:

    ``persistence``
        Freeze the initial latent (``z_hat_k = z_0``). Any useful dynamics model
        must beat this.
    ``ls_linear``
        The *least-squares optimal* linear model in this same latent (see
        :func:`fit_latent_linear`). This is the best a linear core could possibly
        do here, so the gap to ``full`` isolates genuine nonlinearity from an
        undertrained ``A``/``B``.
    ``skill_*``
        ``1 - mse / var(z)``, i.e. variance explained relative to predicting the
        dataset-mean latent. 1.0 is perfect, 0.0 is worthless.
    """
    enc = enc if enc is not None else encode_all(model, trajs)
    z_true, controls = enc["z"], enc["controls"]
    z_full, z_lin = latent_rollout_batch(model, z_true[:, 0], controls)

    def per_step(pred: np.ndarray) -> np.ndarray:
        return ((pred - z_true) ** 2).mean(axis=(0, 2))

    var = float(z_true.reshape(-1, z_true.shape[-1]).var(axis=0).mean())
    out = {
        "steps": np.arange(z_true.shape[1]),
        "full": per_step(z_full),
        "linear": per_step(z_lin),
        "persistence": per_step(np.repeat(z_true[:, :1], z_true.shape[1], axis=1)),
        "latent_var": var,
    }
    if fitted is not None:
        out["ls_linear"] = per_step(linear_rollout_np(fitted, z_true[:, 0], controls))
    for key in ("full", "linear", "persistence", "ls_linear"):
        if key in out:
            out[f"skill_{key}"] = 1.0 - out[key] / max(var, 1e-12)
    return out


# ----------------------------------------------------------------------
# Best-possible linear model in the learned latent (DMDc)
# ----------------------------------------------------------------------
def fit_latent_linear(
    model: LatentWorldModel,
    trajs: list[dict],
    *,
    enc: dict | None = None,
    ridge: float = 1e-6,
) -> dict:
    """Least-squares fit of ``z_{t+1} = A z_t + B u_t + c`` on encoded latents.

    This is DMDc applied *inside the learned latent space*. It answers a question
    the trained model cannot: if the encoder is held fixed, how well can **any**
    linear operator predict this latent? If this fit is much better than the
    model's own linear core, the linear core is simply undertrained; if it is
    still much worse than the full nonlinear model, the latent really is
    nonlinear.
    """
    enc = enc if enc is not None else encode_all(model, trajs)
    z, controls = enc["z"], enc["controls"]
    n_lags = int(model.hparams.n_control_lags)
    d = z.shape[-1]

    xs, ys = [], []
    for k in range(z.shape[1] - 1):
        uf = _u_feat_batch(controls, k, n_lags).numpy()
        xs.append(np.concatenate([z[:, k], uf, np.ones((z.shape[0], 1))], axis=1))
        ys.append(z[:, k + 1])
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    gram = x.T @ x + ridge * np.eye(x.shape[1])
    theta = np.linalg.solve(gram, x.T @ y)  # (d + lags + 1, d)
    pred = x @ theta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0)) ** 2).sum())
    return {
        "A": theta[:d].T,
        "B": theta[d : d + n_lags].T,
        "c": theta[d + n_lags],
        "n_lags": n_lags,
        "one_step_r2": 1.0 - ss_res / max(ss_tot, 1e-12),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(theta[:d].T)))),
    }


def linear_rollout_np(fitted: dict, z0: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Free rollout of a fitted linear model. ``z0`` ``(N, d)``, out ``(N, T+1, d)``."""
    a, b, c = fitted["A"], fitted["B"], fitted["c"]
    n_lags = int(fitted["n_lags"])
    z = np.asarray(z0, dtype=np.float64)
    out = [z]
    for k in range(controls.shape[1] - 1):
        uf = _u_feat_batch(controls, k, n_lags).numpy().astype(np.float64)
        z = z @ a.T + uf @ b.T + c
        out.append(z)
    return np.stack(out, axis=1).astype(np.float32)


# ----------------------------------------------------------------------
# Latent-space geometry
# ----------------------------------------------------------------------
def latent_pca(model: LatentWorldModel, trajs: list[dict], *, enc: dict | None = None) -> dict:
    """PCA of the encoded latents, with per-sample metadata for coloring.

    Two things to read off. First, *structure*: if the latent is a useful state,
    points should organize along interpretable axes (population, grass cover)
    rather than forming an unstructured blob. Second, *effective
    dimensionality*: 32 nominal dimensions are wasted if 2 components explain
    everything, which is partial representation collapse.
    """
    enc = enc if enc is not None else encode_all(model, trajs)
    z = enc["z"]
    n, t, d = z.shape
    flat = z.reshape(-1, d)
    mean = flat.mean(axis=0)
    centered = flat - mean
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    var = sv**2
    evr = var / var.sum()
    return {
        "coords": (centered @ vt.T).reshape(n, t, d),
        "evr": evr,
        "cumulative": np.cumsum(evr),
        "components": vt,
        "mean": mean,
        # Participation ratio: a smooth "how many dimensions are really used".
        "participation_ratio": float(var.sum() ** 2 / (var**2).sum()),
        "n95": int(np.searchsorted(np.cumsum(evr), 0.95) + 1),
        "obs": enc["obs"],
        "controls": enc["controls"],
        "obs_names": enc["obs_names"],
        "meta": enc["meta"],
    }


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> dict:
    xm, xs = x.mean(axis=0), x.std(axis=0) + 1e-8
    xz = np.concatenate([(x - xm) / xs, np.ones((len(x), 1))], axis=1)
    reg = lam * np.eye(xz.shape[1])
    reg[-1, -1] = 0.0  # never penalize the intercept
    w = np.linalg.solve(xz.T @ xz + reg, xz.T @ y)
    return {"w": w, "xm": xm, "xs": xs}


def _ridge_predict(fit: dict, x: np.ndarray) -> np.ndarray:
    xz = np.concatenate([(x - fit["xm"]) / fit["xs"], np.ones((len(x), 1))], axis=1)
    return xz @ fit["w"]


def linear_probe(
    model: LatentWorldModel,
    train_trajs: list[dict],
    test_trajs: list[dict],
    *,
    ridge: float = 1.0,
) -> dict:
    """How much interpretable information is linearly readable from the latent?

    Fits ridge regressions from ``z`` to each ground-truth observable on the
    train split and reports held-out R^2. This is the sharpest test of whether
    the encoder kept what matters: a high R^2 for ``rabbit_count`` means the
    macrostate you want to control is a *linear function* of the latent, so a
    linear controller can target it directly.

    ``u_applied`` / ``u_previous`` probe whether the control that produced the
    frame is legible in the latent at all -- if it is not, the encoder is
    discarding the actuator's signature.
    """
    tr = encode_all(model, train_trajs)
    te = encode_all(model, test_trajs)
    names = list(tr["obs_names"])

    def targets(enc: dict) -> tuple[np.ndarray, list[str]]:
        obs = enc["obs"]
        ctrl = enc["controls"]
        u_now = ctrl[:, :, None]
        u_prev = np.concatenate([np.zeros_like(u_now[:, :1]), u_now[:, :-1]], axis=1)
        y = np.concatenate([obs, u_now, u_prev], axis=2)
        return y.reshape(-1, y.shape[-1]), names + ["u_applied", "u_previous"]

    x_tr, x_te = tr["z"].reshape(-1, tr["z"].shape[-1]), te["z"].reshape(-1, te["z"].shape[-1])
    y_tr, labels = targets(tr)
    y_te, _ = targets(te)

    fit = _ridge_fit(x_tr, y_tr, ridge)
    pred = _ridge_predict(fit, x_te)
    ss_res = ((y_te - pred) ** 2).sum(axis=0)
    ss_tot = ((y_te - y_te.mean(axis=0)) ** 2).sum(axis=0)
    return {
        "names": labels,
        "r2": 1.0 - ss_res / np.maximum(ss_tot, 1e-12),
        "n_train": len(x_tr),
        "n_test": len(x_te),
    }


# ----------------------------------------------------------------------
# Control-theoretic structure of the learned linear system
# ----------------------------------------------------------------------
def mode_analysis(model: LatentWorldModel) -> dict:
    """Eigen-decomposition of ``A`` annotated with how reachable each mode is.

    Each eigenvalue is a *mode* -- an independent pattern in the latent with its
    own decay rate and oscillation period. ``mode_ctrl`` is the normalized
    overlap between the control direction and each mode's left eigenvector
    (the modal / Popov-Belevitch-Hautus controllability test): near 0 means the
    actuator cannot excite that mode no matter how long you push, so a slow,
    persistent, *uncontrollable* mode is a genuine obstruction to control.
    """
    a, b_all = model.linear_system()
    b = b_all[:, 0]  # instantaneous input column
    lam, vecs = np.linalg.eig(a)
    left = np.linalg.inv(vecs)  # rows are left eigenvectors
    overlap = np.abs(left @ b)
    overlap /= np.linalg.norm(left, axis=1) * np.linalg.norm(b) + 1e-12

    mag = np.abs(lam)
    stable = mag < 1.0
    half_life = np.where(
        stable & (mag > 1e-12), np.log(0.5) / np.log(np.clip(mag, 1e-12, 1 - 1e-12)), np.inf
    )
    angle = np.abs(np.angle(lam))
    period = np.where(angle > 1e-6, 2 * np.pi / np.maximum(angle, 1e-12), np.inf)
    order = np.argsort(-mag)
    return {
        "eig": lam[order],
        "magnitude": mag[order],
        "half_life": half_life[order],
        "period": period[order],
        "mode_ctrl": overlap[order],
        "n_unstable": int((mag >= 1.0).sum()),
    }


def controllability_spectrum(model: LatentWorldModel) -> np.ndarray:
    """Normalized singular values of ``[b, Ab, A^2 b, ...]``.

    The integer rank reported during training is a thresholded version of this
    curve and is brittle -- a direction you can only reach with 1e-8 gain counts
    toward the rank but is useless in practice. The decay of these singular
    values is the honest picture of how much control authority exists in each
    direction.
    """
    a, b_all = model.linear_system()
    blocks = [b_all[:, 0:1]]
    for _ in range(1, a.shape[0]):
        blocks.append(a @ blocks[-1])
    sv = np.linalg.svd(np.concatenate(blocks, axis=1), compute_uv=False)
    return sv / max(sv[0], 1e-12)


@torch.no_grad()
def step_response(
    model: LatentWorldModel,
    z0: np.ndarray,
    *,
    u_levels: tuple[float, ...] = (0.25, 0.5, 1.0),
    steps: int = 80,
    impulse: bool = False,
) -> dict:
    """Response of decoded rabbit mass to a step (or impulse) in the control.

    The textbook system-identification experiment: hold the input at a constant
    level from t=0 and watch the output. Reported as a *deviation* from the
    ``u=0`` rollout so the plot isolates the control's contribution from the
    system's natural drift. Slope tells you gain (how much authority you have);
    time to settle tells you how far ahead a controller must plan.
    """

    def roll(u: float) -> tuple[np.ndarray, np.ndarray]:
        ctrl = np.zeros(steps + 1, dtype=np.float32)
        if impulse:
            ctrl[1] = u
        else:
            ctrl[1:] = u
        z_full, z_lin = latent_rollout(model, z0, ctrl)
        return (
            decode_mass(model, z_full)[:, RABBIT_CHANNEL],
            decode_mass(model, z_lin)[:, RABBIT_CHANNEL],
        )

    base_full, base_lin = roll(0.0)
    out = {"steps": np.arange(steps + 1), "baseline": base_full, "u_levels": u_levels}
    for u in u_levels:
        full, lin = roll(u)
        out[f"full_{u}"] = full - base_full
        out[f"linear_{u}"] = lin - base_lin
    return out


def dose_response(
    model: LatentWorldModel,
    *,
    u_levels: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 100,
    initial_rabbits: int = 120,
    initial_grass_prob: float = 0.35,
    settle_frac: float = 0.25,
    cfg: RabbitGrassConfig | None = None,
) -> dict:
    """Late-window population vs constant cull level, true simulator vs model.

    Collapses the control-response time series into the input-output curve a
    controller actually relies on. The requirement is *monotonicity*: more
    culling must mean fewer rabbits, in the model as well as in reality. A model
    with the ordering scrambled will drive the control in the wrong direction no
    matter how good its prediction error looks.
    """
    cfg = cfg or RabbitGrassConfig()
    tail = max(1, int(steps * settle_frac))
    true_m, pred_m = [], []
    for u in u_levels:
        t_vals, p_vals = [], []
        for seed in seeds:
            useq = np.full(steps, float(u), dtype=np.float32)
            frames, ctrl, _ = rollout(
                cfg,
                useq,
                initial_rabbits=initial_rabbits,
                initial_grass_prob=initial_grass_prob,
                seed=seed,
            )
            t_vals.append(true_mass(frames)[-tail:, RABBIT_CHANNEL].mean())
            z0 = encode_trajectory(model, frames[:1])[0]
            z_full, _ = latent_rollout(model, z0, ctrl)
            p_vals.append(decode_mass(model, z_full)[-tail:, RABBIT_CHANNEL].mean())
        true_m.append(t_vals)
        pred_m.append(p_vals)
    true_arr, pred_arr = np.asarray(true_m), np.asarray(pred_m)
    return {
        "u": np.asarray(u_levels, dtype=float),
        "true_mean": true_arr.mean(axis=1),
        "true_std": true_arr.std(axis=1),
        "pred_mean": pred_arr.mean(axis=1),
        "pred_std": pred_arr.std(axis=1),
        "spearman": _spearman(np.asarray(u_levels, float), pred_arr.mean(axis=1)),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, used here purely as a monotonicity score."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ----------------------------------------------------------------------
# Headline summary
# ----------------------------------------------------------------------
def _verdict(value: float, good: float, warn: float, higher_is_better: bool = True) -> str:
    if not np.isfinite(value):
        return "n/a"
    if higher_is_better:
        return "ok" if value >= good else ("warn" if value >= warn else "BAD")
    return "ok" if value <= good else ("warn" if value <= warn else "BAD")


def scorecard(
    *,
    model: LatentWorldModel,
    herr: dict,
    probe: dict,
    pca: dict,
    modes: dict,
    ctrl_sv: np.ndarray,
    dose: dict,
    macro: dict,
    fitted: dict,
    horizon: int = 20,
) -> list[dict]:
    """Condense the analyses into a pass/warn/fail table of headline numbers.

    Consumes already-computed results so nothing is recomputed. Thresholds are
    deliberately loose -- they are meant to draw the eye to the one or two things
    that need attention, not to certify anything.
    """
    h = min(horizon, len(herr["steps"]) - 1)
    names = list(probe["names"])
    r2 = np.asarray(probe["r2"])
    rabbit_r2 = float(r2[names.index("rabbit_count")])
    u_r2 = float(r2[names.index("u_applied")])
    skill_full = float(herr["skill_full"][h])
    skill_ls = float(herr["skill_ls_linear"][h]) if "skill_ls_linear" in herr else np.nan
    skill_lin = float(herr["skill_linear"][h])
    skill_persist = float(herr["skill_persistence"][h])
    used_dims = float(pca["participation_ratio"])
    authority = int((ctrl_sv > 1e-3).sum())
    slowest_uncontrollable = float(modes["magnitude"][np.argmin(modes["mode_ctrl"])])

    rows = [
        {
            "name": f"prediction skill @ h={h}",
            "value": f"{skill_full:+.3f}",
            "status": _verdict(skill_full, 0.5, 0.2),
            "note": f"vs {skill_persist:+.3f} for the persistence baseline",
        },
        {
            "name": "beats persistence?",
            "value": "yes" if skill_full > skill_persist else "no",
            "status": "ok" if skill_full > skill_persist else "BAD",
            "note": "a model that loses to a frozen latent has learned no dynamics",
        },
        {
            "name": "rabbit_count linearly decodable",
            "value": f"{rabbit_r2:+.3f}",
            "status": _verdict(rabbit_r2, 0.85, 0.6),
            "note": "held-out R^2 of a ridge readout from z",
        },
        {
            "name": "control legible in latent",
            "value": f"{u_r2:+.3f}",
            "status": _verdict(u_r2, 0.5, 0.2),
            "note": "R^2 predicting the applied u from z alone",
        },
        {
            "name": "latent dimensions in use",
            "value": f"{used_dims:.1f} / {model.latent_dim}",
            "status": _verdict(used_dims / model.latent_dim, 0.5, 0.25),
            "note": f"participation ratio; {pca['n95']} components cover 95%",
        },
        {
            "name": "model linear core skill",
            "value": f"{skill_lin:+.3f}",
            "status": _verdict(skill_lin, 0.4, 0.0),
            "note": "the A z + B u part of the trained model, on its own",
        },
        {
            "name": "best linear fit skill (DMDc)",
            "value": f"{skill_ls:+.3f}",
            "status": _verdict(skill_ls, 0.4, 0.0),
            "note": f"one-step R^2 = {fitted['one_step_r2']:.3f}; ceiling for any linear model here",
        },
        {
            "name": "is this latent linear enough?",
            "value": "yes" if skill_ls >= skill_full - 0.05 else "no",
            "status": "ok" if skill_ls >= skill_full - 0.05 else "warn",
            "note": (
                "a fitted linear operator matches the nonlinear model, so linear "
                "control tools apply"
                if skill_ls >= skill_full - 0.05
                else "the nonlinear residual is doing real work"
            ),
        },
        {
            "name": "spectral radius of A",
            "value": f"{float(modes['magnitude'][0]):.4f}",
            "status": _verdict(float(modes["magnitude"][0]), 1.0, 1.02, higher_is_better=False),
            "note": f"{modes['n_unstable']} modes with |lambda| >= 1",
        },
        {
            "name": "usable control directions",
            "value": f"{authority} / {model.latent_dim}",
            "status": _verdict(authority / model.latent_dim, 0.3, 0.1),
            "note": (
                "singular values of [b, Ab, ...] above 1e-3; the reported integer "
                f"rank is looser. Least-reachable mode has |lambda|={slowest_uncontrollable:.3f}"
            ),
        },
        {
            "name": "dose-response monotonic",
            "value": f"{dose['spearman']:+.2f}",
            "status": _verdict(-dose["spearman"], 0.99, 0.7),
            "note": "-1 means more culling reliably predicts fewer rabbits",
        },
        {
            "name": "macrostate correlation",
            "value": f"{macro['rabbit_mass_corr']:+.3f}",
            "status": _verdict(macro["rabbit_mass_corr"], 0.85, 0.6),
            "note": f"rabbit-mass bias {macro['rabbit_mass_bias']:+.1f} over free rollouts",
        },
    ]
    return rows


def format_scorecard(rows: list[dict]) -> str:
    """Render :func:`scorecard` output as a fixed-width table."""
    mark = {"ok": "[ ok ]", "warn": "[warn]", "BAD": "[FAIL]", "n/a": "[ -- ]"}
    w = max(len(r["name"]) for r in rows)
    lines = [f"{'metric':<{w}}  {'value':>14}  status  note", "-" * (w + 34)]
    for r in rows:
        lines.append(f"{r['name']:<{w}}  {r['value']:>14}  {mark[r['status']]}  {r['note']}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Image-space views
# ----------------------------------------------------------------------
@torch.no_grad()
def reconstruct(model: LatentWorldModel, frames: np.ndarray) -> np.ndarray:
    """Encode then immediately decode: pure autoencoder fidelity, no dynamics."""
    x = torch.from_numpy(np.asarray(frames, dtype=np.float32))
    return model.decode(model.encode(x)).cpu().numpy()


@torch.no_grad()
def predicted_frames(
    model: LatentWorldModel,
    traj: dict,
    *,
    start: int = 0,
    horizons: tuple[int, ...] = (0, 1, 5, 10, 25, 50),
) -> dict:
    """True vs decoded-predicted frames at several horizons from one start point.

    Expect the prediction to blur into a *density field* rather than tracking
    individual agents. That is the correct behaviour for a stochastic ABM, not a
    failure: agent positions are unpredictable in detail, so the best possible
    prediction is the probability of occupancy.
    """
    z_true = encode_trajectory(model, traj["frames"])
    ctrl = traj["controls"][start:]
    z_full, _ = latent_rollout(model, z_true[start], ctrl)
    picks = [h for h in horizons if h < len(z_full)]
    pred = model.decode(torch.from_numpy(np.asarray(z_full[picks], dtype=np.float32))).cpu().numpy()
    return {
        "horizons": picks,
        "true": np.asarray(traj["frames"], dtype=np.float32)[[start + h for h in picks]],
        "pred": pred,
        "controls": ctrl,
    }


@torch.no_grad()
def control_effect_map(
    model: LatentWorldModel,
    z: np.ndarray,
    u: float = 1.0,
) -> dict:
    """Where in the image does the control act?

    Pushes the latent along the control direction ``B[:, 0] * u`` (the column of
    ``B`` acting on the instantaneous input) and decodes before and after. The
    difference map makes the learned actuator visible in pixel space, which is
    the check that the model attributes the control to *rabbits* rather than to
    grass or to an image-wide brightness offset.
    """
    b = model.B.weight.detach().cpu().numpy()[:, 0]
    z = np.asarray(z, dtype=np.float32)
    pair = torch.from_numpy(np.stack([z, z + b * float(u)]).astype(np.float32))
    imgs = model.decode(pair).cpu().numpy()
    return {"base": imgs[0], "perturbed": imgs[1], "delta": imgs[1] - imgs[0], "u": u}
