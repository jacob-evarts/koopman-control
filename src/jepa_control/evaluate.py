"""Evaluate a trained JEPA latent model.

There is no decoder, so nothing here scores pixel reconstruction. The latent is
judged by:

  * **Prediction** -- free-running latent rollout error vs horizon, against a
    ``persistence`` floor and the least-squares-optimal linear operator in the
    same latent (``ls_linear``), so a unitless MSE becomes readable.
  * **Interpretability** -- a post-hoc **linear readout** ``z -> macrostates``
    (held-out R^2), plus control legibility (can the applied ``u`` be read from
    ``z``) and latent geometry (PCA + participation ratio for partial collapse).
  * **Linearity (diagnostic only)** -- a least-squares linear fit in the latent.
    With the default linear predictor this is an optimality reference: the trained
    ``A``/``B`` should match it, and a large gap means the predictor is
    under-trained rather than the latent being at fault. With the ``residual_mlp``
    ablation it instead measures how nonlinear the discovered latent is.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from jepa_control.data import (
    _categorical_frames,
    _categorical_obs,
    _state_aligned_actions,
)
from jepa_control.model import JEPAControl

# Spatial centroids are recorded in the ABM for diagnostics but are not useful
# control targets; exclude them from post-hoc readout / probe prediction by default.
EXCLUDE_FROM_READOUT: tuple[str, ...] = (
    "rabbit_centroid_x",
    "rabbit_centroid_y",
    "wolf_centroid_x",
    "wolf_centroid_y",
    "tumor_centroid_x",
    "tumor_centroid_y",
    "infected_centroid_x",
    "infected_centroid_y",
)


def _select_obs_names(
    all_names: list[str],
    *,
    obs_names: list[str] | None = None,
    exclude: tuple[str, ...] | None = EXCLUDE_FROM_READOUT,
) -> list[str]:
    """Pick which macrostate columns to fit / score."""
    if obs_names is not None:
        missing = [n for n in obs_names if n not in all_names]
        if missing:
            raise KeyError(
                f"obs_names not in dataset: {missing}; available={all_names}"
            )
        return list(obs_names)
    skip = set(exclude or ())
    return [n for n in all_names if n not in skip]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def _predictor_from_state_dict(state_dict: dict) -> str | None:
    """Infer ``predictor`` from weight names when a notebook kernel is stale.

    Linear checkpoints store ``predictor.A.*`` / ``predictor.B.*``; residual-MLP
    checkpoints store ``predictor.net.*``. Prefer the checkpoint's own
    hyperparameters when present; this is a fallback for kernels that still have
    an older ``JEPAControl`` default cached.
    """
    keys = state_dict.keys()
    has_linear = any(k.startswith("predictor.A.") for k in keys)
    has_mlp = any(k.startswith("predictor.net.") for k in keys)
    if has_linear and not has_mlp:
        return "linear"
    if has_mlp and not has_linear:
        return "residual_mlp"
    return None


def _target_from_state_dict(state_dict: dict) -> str:
    """Infer ``target``: EMA checkpoints store ``target_encoder.*`` weights."""
    if any(k.startswith("target_encoder.") for k in state_dict):
        return "ema"
    return "stopgrad"


def load_model(ckpt_path: str | Path) -> JEPAControl:
    """Load a trained ``JEPAControl``, forcing architecture flags to match the ckpt.

    Explicitly passes ``predictor`` and ``target`` into ``load_from_checkpoint`` so
    a stale notebook import / new default (EMA) cannot reconstruct the wrong
    modules. If the imported class is older than these hyperparameters, raises a
    clear restart-kernel message.
    """
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters") or {}
    predictor = hp.get("predictor") or _predictor_from_state_dict(ckpt["state_dict"])
    target = hp.get("target") or _target_from_state_dict(ckpt["state_dict"])
    kwargs: dict = {"map_location": "cpu", "target": target}
    if predictor is not None:
        kwargs["predictor"] = predictor
    try:
        model = JEPAControl.load_from_checkpoint(str(ckpt_path), **kwargs)
    except (TypeError, RuntimeError) as exc:
        msg = str(exc)
        stale = (
            "unexpected keyword argument 'predictor'" in msg
            or "unexpected keyword argument 'target'" in msg
            or "predictor.A." in msg
            or "predictor.net." in msg
            or "target_encoder." in msg
        )
        if stale:
            raise RuntimeError(
                f"Failed to load {ckpt_path} into the currently imported JEPAControl "
                f"(checkpoint predictor={predictor!r}, target={target!r}). Restart the "
                "notebook kernel and re-run from the top so it picks up the current class."
            ) from exc
        raise
    model.eval()
    return model


def _select_run_ids(
    run_ids: list[str], excitations: list[str], max_runs: int | None, seed: int
) -> set[str]:
    """Pick ``max_runs`` run ids stratified across excitation regimes.

    Runs are written to the HDF5 file grouped by excitation, so naively taking the
    first ``max_runs`` yields a single regime (e.g. all ``chirp``). Any readout or
    probe fit on such a subset fails to generalize across the other regimes. This
    draws a deterministic, excitation-balanced sample instead.
    """
    if max_runs is None or max_runs >= len(run_ids):
        return set(run_ids)
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[str]] = {}
    for rid, exc in zip(run_ids, excitations):
        buckets.setdefault(exc, []).append(rid)
    for ids in buckets.values():
        rng.shuffle(ids)
    order = sorted(buckets)
    selected: list[str] = []
    cursors = {k: 0 for k in order}
    while len(selected) < max_runs:
        progressed = False
        for k in order:
            if cursors[k] < len(buckets[k]):
                selected.append(buckets[k][cursors[k]])
                cursors[k] += 1
                progressed = True
                if len(selected) >= max_runs:
                    break
        if not progressed:
            break
    return set(selected)


def load_split_trajectories(
    h5_path: str | Path,
    split: str,
    max_runs: int | None = None,
    seed: int = 0,
    where: dict[str, object] | None = None,
) -> list[dict]:
    """Full-length trajectories (frames/controls/obs + metadata) for a split.

    When ``max_runs`` is smaller than the split, runs are sampled deterministically
    and stratified by excitation regime rather than taken in file order (which would
    return a single regime and bias every downstream readout/probe).
    """
    trajs: list[dict] = []
    with h5py.File(h5_path, "r") as f:
        obs_names = [
            s.decode() if isinstance(s, bytes) else str(s) for s in f.attrs["obs_names"]
        ]
        frame_scale = float(f.attrs.get("frame_scale", 1.0))
        run_ids, excitations = [], []
        for rid, grp in f["runs"].items():
            if grp.attrs["split"] != split:
                continue
            if where and any(
                str(grp.attrs.get(key, "")) != str(value)
                for key, value in where.items()
            ):
                continue
            run_ids.append(rid)
            excitations.append(str(grp.attrs["excitation"]))
        keep = _select_run_ids(run_ids, excitations, max_runs, seed)
        for rid in run_ids:
            if rid not in keep:
                continue
            grp = f["runs"][rid]
            metadata: dict = {}
            for key, value in grp.attrs.items():
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, bytes):
                    value = value.decode()
                metadata[key] = value
            if "frames" in grp:
                frames = np.asarray(grp["frames"][:], dtype=np.uint8)
                controls = np.asarray(grp["control"][:], dtype=np.float32)
                obs = np.asarray(grp["obs"][:], dtype=np.float32)
            elif "grid" in grp:
                frames = _categorical_frames(
                    grp["grid"][:], int(f.attrs["num_channels"])
                )
                controls = _state_aligned_actions(grp["action"][:], len(frames))
                obs, categorical_names = _categorical_obs(grp)
                if categorical_names != obs_names:
                    raise ValueError(
                        f"{rid}: categorical observables {categorical_names} "
                        f"do not match file schema {obs_names}"
                    )
            else:
                raise KeyError(f"{rid}: requires frames/control/obs or grid/action")
            trajs.append(
                {
                    **metadata,
                    "run_id": rid,
                    "excitation": str(grp.attrs["excitation"]),
                    "seed": int(
                        grp.attrs.get("seed", grp.attrs.get("simulation_seed", 0))
                    ),
                    "frames": frames,
                    "frame_scale": frame_scale,
                    "controls": controls,
                    "obs": obs,
                }
            )
    for t in trajs:
        t["obs_names"] = obs_names
    return trajs


# ----------------------------------------------------------------------
# Core latent operations
# ----------------------------------------------------------------------
@torch.no_grad()
def encode_trajectory(
    model: JEPAControl, frames: np.ndarray, frame_scale: float = 1.0
) -> np.ndarray:
    """Encode frames, undoing optional uint8 quantization."""
    x = torch.from_numpy(np.asarray(frames, dtype=np.float32) / float(frame_scale))
    return model.encode(x).cpu().numpy()


def _u_feat_batch(controls: np.ndarray, k: int, n_lags: int) -> torch.Tensor:
    """Batched control history ``(N, n_lags)`` for the transition ``k -> k+1``."""
    cols = []
    for j in range(n_lags):
        idx = k + 1 - j
        cols.append(controls[:, idx] if idx >= 0 else np.zeros(controls.shape[0]))
    return torch.tensor(np.stack(cols, axis=-1), dtype=torch.float32)


@torch.no_grad()
def latent_rollout_batch(
    model: JEPAControl, z0: np.ndarray, controls: np.ndarray
) -> np.ndarray:
    """Free-running latent rollout for many trajectories. Out ``(N, T+1, d)``."""
    n_lags = int(model.hparams.n_control_lags)
    z = torch.from_numpy(np.asarray(z0, dtype=np.float32))
    out = [z.numpy()]
    for k in range(controls.shape[1] - 1):
        z = model.step(z, _u_feat_batch(controls, k, n_lags))
        out.append(z.numpy())
    return np.stack(out, axis=1)


@torch.no_grad()
def encode_all(model: JEPAControl, trajs: list[dict]) -> dict:
    """Encode every trajectory once and stack ``(n_runs, T+1, ...)``."""
    t_min = min(len(t["controls"]) for t in trajs)
    z = np.stack(
        [
            encode_trajectory(model, tr["frames"], tr.get("frame_scale", 1.0))[:t_min]
            for tr in trajs
        ]
    )
    controls = np.stack([tr["controls"][:t_min] for tr in trajs])
    obs = np.stack([tr["obs"][:t_min] for tr in trajs])
    return {
        "z": z,
        "controls": controls,
        "obs": obs,
        "obs_names": trajs[0]["obs_names"],
        "meta": [
            {
                k: tr[k]
                for k in (
                    "run_id",
                    "excitation",
                    "initial_rabbits",
                    "initial_wolves",
                    "initial_healthy_frac",
                    "initial_tumor_radius",
                    "tumor_center_x",
                    "tumor_center_y",
                    "split",
                    "subset",
                    "architecture",
                    "policy",
                    "evaluation_group",
                    "dose_level",
                    "matched_group",
                    "stochastic_replicate",
                    "seed",
                )
                if k in tr
            }
            for tr in trajs
        ],
    }


# ----------------------------------------------------------------------
# Post-hoc linear readout (replaces the decoder)
# ----------------------------------------------------------------------
def fit_readout(
    enc: dict,
    ridge: float = 1.0,
    *,
    obs_names: list[str] | None = None,
    exclude: tuple[str, ...] | None = EXCLUDE_FROM_READOUT,
) -> dict:
    """Ridge fit ``obs ~ z W + b``, returned as a raw-``z`` map (composable for MPC).

    The fit is done on **centered and scaled** ``z`` and then folded back into raw
    latent coordinates. This matters here: a VICReg latent is only variance- (not
    mean-) controlled and typically uses far fewer directions than its nominal
    width, so the raw design matrix ``[z, 1]`` is badly conditioned -- fitting it
    directly produces huge, unstable weights that generalize terribly (negative
    held-out R^2) even though the information is present. Standardizing first
    regularizes the collinear directions sensibly; the returned ``W, b`` still
    satisfy ``obs ~ z W + b`` so nothing downstream needs to change.

    By default spatial centroid observables are skipped (see
    :data:`EXCLUDE_FROM_READOUT`); pass ``exclude=()`` to fit every column.
    """
    names = _select_obs_names(
        list(enc["obs_names"]), obs_names=obs_names, exclude=exclude
    )
    cols = [enc["obs_names"].index(n) for n in names]
    z = enc["z"].reshape(-1, enc["z"].shape[-1])
    y = enc["obs"][..., cols].reshape(-1, len(cols))
    mu = z.mean(axis=0)
    sd = z.std(axis=0) + 1e-8
    zs = (z - mu) / sd
    x = np.concatenate([zs, np.ones((zs.shape[0], 1))], axis=1)
    reg = ridge * np.eye(x.shape[1])
    reg[-1, -1] = 0.0
    theta = np.linalg.solve(x.T @ x + reg, x.T @ y)
    w_std, b_std = theta[:-1], theta[-1]
    # Fold standardization back so W, b act on raw z: obs = z W + b.
    w = w_std / sd[:, None]
    b = b_std - (mu / sd) @ w_std
    return {"W": w, "b": b, "names": names}


def readout_predict(readout: dict, z: np.ndarray) -> np.ndarray:
    """Map latents ``(..., d)`` to macrostates ``(..., K)``."""
    z = np.asarray(z, dtype=np.float64)
    flat = z.reshape(-1, z.shape[-1]) @ readout["W"] + readout["b"]
    return flat.reshape(*z.shape[:-1], flat.shape[-1])


def readout_r2(readout: dict, enc: dict) -> dict:
    """Held-out R^2 per observable for a fitted readout."""
    names = list(readout["names"])
    cols = [enc["obs_names"].index(n) for n in names]
    pred = readout_predict(readout, enc["z"]).reshape(-1, len(names))
    y = enc["obs"][..., cols].reshape(-1, len(cols))
    ss_res = ((y - pred) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    return dict(zip(names, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), strict=True))


# ----------------------------------------------------------------------
# Prediction accuracy with baselines
# ----------------------------------------------------------------------
def horizon_errors(model: JEPAControl, enc: dict, fitted: dict | None = None) -> dict:
    """Latent MSE vs rollout horizon for the model, persistence, and ls_linear."""
    z_true, controls = enc["z"], enc["controls"]
    z_full = latent_rollout_batch(model, z_true[:, 0], controls)

    def per_step(pred: np.ndarray) -> np.ndarray:
        return ((pred - z_true) ** 2).mean(axis=(0, 2))

    var = float(z_true.reshape(-1, z_true.shape[-1]).var(axis=0).mean())
    out = {
        "steps": np.arange(z_true.shape[1]),
        "full": per_step(z_full),
        "persistence": per_step(np.repeat(z_true[:, :1], z_true.shape[1], axis=1)),
        "latent_var": var,
    }
    if fitted is not None:
        out["ls_linear"] = per_step(linear_rollout_np(fitted, z_true[:, 0], controls))
    for key in ("full", "persistence", "ls_linear"):
        if key in out:
            out[f"skill_{key}"] = 1.0 - out[key] / max(var, 1e-12)
    return out


# ----------------------------------------------------------------------
# Linear diagnostic (DMDc in the learned latent) -- NOT used for control
# ----------------------------------------------------------------------
def fit_latent_linear(model: JEPAControl, enc: dict, ridge: float = 1e-6) -> dict:
    z, controls = enc["z"], enc["controls"]
    n_lags = int(model.hparams.n_control_lags)
    d = z.shape[-1]
    xs, ys = [], []
    for k in range(z.shape[1] - 1):
        uf = _u_feat_batch(controls, k, n_lags).numpy()
        xs.append(np.concatenate([z[:, k], uf, np.ones((z.shape[0], 1))], axis=1))
        ys.append(z[:, k + 1])
    x, y = np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
    theta = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
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
# Latent geometry + probes
# ----------------------------------------------------------------------
def latent_pca(enc: dict) -> dict:
    """PCA of the encoded latents, with metadata for coloring.

    Two things to read off: whether the cloud organizes along interpretable axes
    (population, grass cover), and the **participation ratio** -- a smooth count
    of how many dimensions are genuinely used. A ratio far below the nominal
    latent width is partial collapse, which VICReg's per-feature variance hinge
    does not catch on its own.
    """
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
    reg[-1, -1] = 0.0
    w = np.linalg.solve(xz.T @ xz + reg, xz.T @ y)
    return {"w": w, "xm": xm, "xs": xs}


def _ridge_predict(fit: dict, x: np.ndarray) -> np.ndarray:
    xz = np.concatenate([(x - fit["xm"]) / fit["xs"], np.ones((len(x), 1))], axis=1)
    return xz @ fit["w"]


def linear_probe(
    model: JEPAControl,
    train_trajs: list[dict],
    test_trajs: list[dict],
    ridge: float = 1.0,
    *,
    obs_names: list[str] | None = None,
    exclude: tuple[str, ...] | None = EXCLUDE_FROM_READOUT,
) -> dict:
    """Held-out R^2 of ridge readouts from ``z`` to macrostates and to ``u``.

    A high ``rabbit_count`` R^2 means the macrostate to be controlled is a linear
    function of the latent; ``u_applied`` R^2 checks the actuator's signature is
    not discarded (a subtle form of collapse for control). Centroid locations are
    excluded by default (see :data:`EXCLUDE_FROM_READOUT`).
    """
    tr, te = encode_all(model, train_trajs), encode_all(model, test_trajs)
    names = _select_obs_names(
        list(tr["obs_names"]), obs_names=obs_names, exclude=exclude
    )
    cols = [tr["obs_names"].index(n) for n in names]

    def targets(enc: dict) -> np.ndarray:
        u_now = enc["controls"][:, :, None]
        u_prev = np.concatenate([np.zeros_like(u_now[:, :1]), u_now[:, :-1]], axis=1)
        y = np.concatenate([enc["obs"][..., cols], u_now, u_prev], axis=2)
        return y.reshape(-1, y.shape[-1])

    x_tr = tr["z"].reshape(-1, tr["z"].shape[-1])
    x_te = te["z"].reshape(-1, te["z"].shape[-1])
    y_tr, y_te = targets(tr), targets(te)
    fit = _ridge_fit(x_tr, y_tr, ridge)
    pred = _ridge_predict(fit, x_te)
    ss_res = ((y_te - pred) ** 2).sum(axis=0)
    ss_tot = ((y_te - y_te.mean(axis=0)) ** 2).sum(axis=0)
    return {
        "names": names + ["u_applied", "u_previous"],
        "r2": 1.0 - ss_res / np.maximum(ss_tot, 1e-12),
    }


def readout_rollout_skill(
    model: JEPAControl, enc: dict, readout: dict, obs_name: str = "rabbit_count"
) -> dict:
    """Free-rollout agreement between predicted and true macrostate over horizon."""
    idx = readout["names"].index(obs_name)
    true_idx = enc["obs_names"].index(obs_name)
    z_full = latent_rollout_batch(model, enc["z"][:, 0], enc["controls"])
    pred = readout_predict(readout, z_full)[..., idx].reshape(-1)
    true = enc["obs"][..., true_idx].reshape(-1)
    corr = (
        float(np.corrcoef(pred, true)[0, 1])
        if pred.std() > 0 and true.std() > 0
        else float("nan")
    )
    return {
        "obs_name": obs_name,
        "corr": corr,
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
    }


# ----------------------------------------------------------------------
# Control response (all scored through the readout, no decoder)
# ----------------------------------------------------------------------
@torch.no_grad()
def step_response(
    model: JEPAControl,
    z0: np.ndarray,
    readout: dict,
    *,
    obs_name: str = "rabbit_count",
    u_levels: tuple[float, ...] = (0.25, 0.5, 1.0),
    steps: int = 80,
    impulse: bool = False,
) -> dict:
    """Response of the predicted macrostate to a step (or impulse) in the control.

    Reported as a *deviation* from the ``u = 0`` rollout so the curve isolates the
    control's own contribution from the system's natural drift. Read the sign
    (down is correct for culling), the settling time (how far ahead MPC must
    plan), and whether levels scale proportionally.
    """
    idx = readout["names"].index(obs_name)

    def roll(u: float) -> np.ndarray:
        ctrl = np.zeros((1, steps + 1), dtype=np.float32)
        if impulse:
            ctrl[0, 1] = u
        else:
            ctrl[0, 1:] = u
        z = latent_rollout_batch(model, np.asarray(z0, dtype=np.float32)[None], ctrl)
        return readout_predict(readout, z)[0, :, idx]

    base = roll(0.0)
    out = {"steps": np.arange(steps + 1), "baseline": base, "u_levels": u_levels}
    for u in u_levels:
        out[f"resp_{u}"] = roll(u) - base
    return out


def dose_response(
    model: JEPAControl,
    readout: dict,
    *,
    obs_name: str = "rabbit_count",
    u_levels: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 100,
    initial_rabbits: int = 120,
    initial_wolves: int = 16,
    initial_grass_prob: float = 0.35,
    settle_frac: float = 0.25,
    cfg=None,
    abm: str = "rabbit_grass",
) -> dict:
    """Late-window macrostate vs constant cull level, true simulator vs model.

    The input-output curve a controller relies on. The requirement is
    *monotonicity*: more culling must move the controlled population the right
    way in the model as well as in reality. Magnitudes can be miscalibrated and
    MPC still works, but a scrambled ordering makes it push the wrong way.

    ``abm`` selects ``rabbit_grass`` (cull rabbits) or ``wolf_rabbit_grass``
    (cull wolves).
    """
    if abm == "rabbit_grass":
        from koopman_control.data.rabbit_grass import RabbitGrassConfig, rollout

        cfg = cfg or RabbitGrassConfig()

        def _roll(useq, seed):
            return rollout(
                cfg,
                useq,
                initial_rabbits=initial_rabbits,
                initial_grass_prob=initial_grass_prob,
                seed=seed,
            )

    elif abm == "wolf_rabbit_grass":
        from koopman_control.data.wolf_rabbit_grass import (
            WolfRabbitGrassConfig,
            rollout,
        )

        cfg = cfg or WolfRabbitGrassConfig()

        def _roll(useq, seed):
            return rollout(
                cfg,
                useq,
                initial_rabbits=initial_rabbits,
                initial_wolves=initial_wolves,
                initial_grass_prob=initial_grass_prob,
                seed=seed,
            )

    else:
        raise ValueError(f"unknown abm {abm!r}")

    idx = readout["names"].index(obs_name)
    tail = max(1, int(steps * settle_frac))
    true_m, pred_m = [], []
    for u in u_levels:
        t_vals, p_vals = [], []
        for seed in seeds:
            useq = np.full(steps, float(u), dtype=np.float32)
            frames, ctrl, obs = _roll(useq, seed)
            t_vals.append(obs[obs_name][-tail:].mean())
            z0 = encode_trajectory(model, frames[:1])[0]
            z = latent_rollout_batch(model, z0[None], ctrl[None])
            p_vals.append(readout_predict(readout, z)[0, -tail:, idx].mean())
        true_m.append(t_vals)
        pred_m.append(p_vals)
    true_arr, pred_arr = np.asarray(true_m), np.asarray(pred_m)
    return {
        "u": np.asarray(u_levels, dtype=float),
        "obs_name": obs_name,
        "abm": abm,
        "true_mean": true_arr.mean(axis=1),
        "true_std": true_arr.std(axis=1),
        "pred_mean": pred_arr.mean(axis=1),
        "pred_std": pred_arr.std(axis=1),
        "spearman": _spearman(np.asarray(u_levels, float), pred_arr.mean(axis=1)),
    }


def tumor_dose_response(
    model: JEPAControl,
    readout: dict,
    *,
    u_levels: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 160,
    initial_healthy_frac: float = 0.94,
    initial_tumor_radius: float = 6.0,
    tumor_center_x: float | None = None,
    tumor_center_y: float | None = None,
    settle_frac: float = 0.25,
    cfg=None,
) -> dict:
    """True vs latent-model chemotherapy response for tumor and healthy tissue."""
    from koopman_control.data.tumor_tissue import TumorTissueConfig, rollout

    cfg = cfg or TumorTissueConfig()
    tumor_idx = readout["names"].index("tumor_count")
    healthy_idx = readout["names"].index("healthy_count")
    tail = max(1, int(steps * settle_frac))
    true_tumor, pred_tumor = [], []
    true_healthy, pred_healthy = [], []
    for u in u_levels:
        t_tumor, p_tumor, t_healthy, p_healthy = [], [], [], []
        for seed in seeds:
            controls = np.full(steps, float(u), dtype=np.float32)
            frames, control_seq, obs = rollout(
                cfg,
                controls,
                initial_healthy_frac=initial_healthy_frac,
                initial_tumor_radius=initial_tumor_radius,
                tumor_center_x=tumor_center_x,
                tumor_center_y=tumor_center_y,
                seed=seed,
            )
            z0 = encode_trajectory(model, frames[:1])[0]
            z = latent_rollout_batch(model, z0[None], control_seq[None])
            predicted = readout_predict(readout, z)[0]
            t_tumor.append(obs["tumor_count"][-tail:].mean())
            p_tumor.append(predicted[-tail:, tumor_idx].mean())
            t_healthy.append(obs["healthy_count"][-tail:].mean())
            p_healthy.append(predicted[-tail:, healthy_idx].mean())
        true_tumor.append(t_tumor)
        pred_tumor.append(p_tumor)
        true_healthy.append(t_healthy)
        pred_healthy.append(p_healthy)

    true_t = np.asarray(true_tumor)
    pred_t = np.asarray(pred_tumor)
    true_h = np.asarray(true_healthy)
    pred_h = np.asarray(pred_healthy)
    return {
        "u": np.asarray(u_levels, dtype=float),
        "obs_name": "tumor_count",
        "true_tumor_mean": true_t.mean(axis=1),
        "true_tumor_std": true_t.std(axis=1),
        "pred_tumor_mean": pred_t.mean(axis=1),
        "pred_tumor_std": pred_t.std(axis=1),
        "true_healthy_mean": true_h.mean(axis=1),
        "true_healthy_std": true_h.std(axis=1),
        "pred_healthy_mean": pred_h.mean(axis=1),
        "pred_healthy_std": pred_h.std(axis=1),
        "spearman": _spearman(np.asarray(u_levels, float), pred_t.mean(axis=1)),
    }


def sir_dose_response(
    model: JEPAControl,
    readout: dict,
    *,
    u_levels: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    seeds: tuple[int, ...] = (0, 1, 2),
    steps: int = 160,
    n_agents: int = 500,
    initial_infected: int = 16,
    seed_center_x: float | None = None,
    seed_center_y: float | None = None,
    seed_radius: float = 5.0,
    settle_frac: float = 0.25,
    cfg=None,
) -> dict:
    """True vs latent-model vaccination response for infection and incidence."""
    from koopman_control.data.agentic_sir import AgenticSIRConfig, rollout

    cfg = cfg or AgenticSIRConfig()
    infected_idx = readout["names"].index("infected_count")
    susceptible_idx = readout["names"].index("susceptible_count")
    tail = max(1, int(steps * settle_frac))
    true_infected, pred_infected = [], []
    true_susceptible, pred_susceptible = [], []
    true_incidence = []
    for u in u_levels:
        t_i, p_i, t_s, p_s, t_inc = [], [], [], [], []
        for seed in seeds:
            controls = np.full(steps, float(u), dtype=np.float32)
            frames, control_seq, obs = rollout(
                cfg,
                controls,
                n_agents=n_agents,
                initial_infected=initial_infected,
                seed_center_x=seed_center_x,
                seed_center_y=seed_center_y,
                seed_radius=seed_radius,
                seed=seed,
            )
            z0 = encode_trajectory(model, frames[:1])[0]
            z = latent_rollout_batch(model, z0[None], control_seq[None])
            predicted = readout_predict(readout, z)[0]
            t_i.append(obs["infected_count"][-tail:].mean())
            p_i.append(predicted[-tail:, infected_idx].mean())
            t_s.append(obs["susceptible_count"][-tail:].mean())
            p_s.append(predicted[-tail:, susceptible_idx].mean())
            t_inc.append(float(obs["cumulative_incidence"][-1]))
        true_infected.append(t_i)
        pred_infected.append(p_i)
        true_susceptible.append(t_s)
        pred_susceptible.append(p_s)
        true_incidence.append(t_inc)

    true_i = np.asarray(true_infected)
    pred_i = np.asarray(pred_infected)
    true_s = np.asarray(true_susceptible)
    pred_s = np.asarray(pred_susceptible)
    true_inc = np.asarray(true_incidence)
    return {
        "u": np.asarray(u_levels, dtype=float),
        "obs_name": "infected_count",
        "true_infected_mean": true_i.mean(axis=1),
        "true_infected_std": true_i.std(axis=1),
        "pred_infected_mean": pred_i.mean(axis=1),
        "pred_infected_std": pred_i.std(axis=1),
        "true_susceptible_mean": true_s.mean(axis=1),
        "true_susceptible_std": true_s.std(axis=1),
        "pred_susceptible_mean": pred_s.mean(axis=1),
        "pred_susceptible_std": pred_s.std(axis=1),
        "true_incidence_mean": true_inc.mean(axis=1),
        "true_incidence_std": true_inc.std(axis=1),
        "spearman": _spearman(np.asarray(u_levels, float), pred_i.mean(axis=1)),
    }


def sir_control_beliefs(
    model: JEPAControl,
    readout: dict,
    *,
    u_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    steps: int = 80,
    n_agents: int = 800,
    initial_infected: int = 32,
    seed_center_x: float | None = None,
    seed_center_y: float | None = None,
    seed_radius: float = 5.0,
    seed: int = 950,
    cfg=None,
    include_true: bool = True,
) -> dict:
    """Free-running predicted I/S trajectories under constant vaccination levels.

    Encodes one initial frame, then rolls the latent model under each constant
    ``u`` through the post-hoc readout. Optionally overlays matched true-ABM
    rollouts from the same seed so you can see whether the planner's internal
    world has the correct vaccination sign.
    """
    from koopman_control.data.agentic_sir import AgenticSIRConfig, rollout

    cfg = cfg or AgenticSIRConfig()
    infected_idx = readout["names"].index("infected_count")
    susceptible_idx = readout["names"].index("susceptible_count")
    # Shared initial latent: encode the first frame of the uncontrolled rollout.
    frames0, _, _ = rollout(
        cfg,
        np.zeros(1, dtype=np.float32),
        n_agents=n_agents,
        initial_infected=initial_infected,
        seed_center_x=seed_center_x,
        seed_center_y=seed_center_y,
        seed_radius=seed_radius,
        seed=seed,
    )
    z0 = encode_trajectory(model, frames0[:1])[0]

    pred: dict[float, dict[str, np.ndarray]] = {}
    true: dict[float, dict[str, np.ndarray]] = {}
    for u in u_levels:
        # Match ABM layout: length steps+1 with a leading unused control at t=0.
        controls = np.concatenate(
            [np.zeros(1, dtype=np.float32), np.full(steps, float(u), dtype=np.float32)]
        )
        z = latent_rollout_batch(model, z0[None], controls[None])[0]
        predicted = readout_predict(readout, z)
        pred[float(u)] = {
            "infected": predicted[:, infected_idx].astype(np.float32),
            "susceptible": predicted[:, susceptible_idx].astype(np.float32),
        }
        if include_true:
            _, _, obs = rollout(
                cfg,
                np.full(steps, float(u), dtype=np.float32),
                n_agents=n_agents,
                initial_infected=initial_infected,
                seed_center_x=seed_center_x,
                seed_center_y=seed_center_y,
                seed_radius=seed_radius,
                seed=seed,
            )
            true[float(u)] = {
                "infected": np.asarray(
                    obs["infected_count"][: steps + 1], dtype=np.float32
                ),
                "susceptible": np.asarray(
                    obs["susceptible_count"][: steps + 1], dtype=np.float32
                ),
            }

    out = {
        "u_levels": tuple(float(u) for u in u_levels),
        "steps": np.arange(steps + 1),
        "z0": z0,
        "pred": pred,
        "n_agents": n_agents,
        "initial_infected": initial_infected,
        "seed": seed,
    }
    if include_true:
        out["true"] = true
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, used purely as a monotonicity score."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ----------------------------------------------------------------------
# Headline summary
# ----------------------------------------------------------------------
def _verdict(
    value: float, good: float, warn: float, higher_is_better: bool = True
) -> str:
    if not np.isfinite(value):
        return "n/a"
    if higher_is_better:
        return "ok" if value >= good else ("warn" if value >= warn else "BAD")
    return "ok" if value <= good else ("warn" if value <= warn else "BAD")


def scorecard(
    *,
    model: JEPAControl,
    herr: dict,
    probe: dict,
    pca: dict,
    readout_r2_test: dict,
    rollout_skill: dict,
    dose: dict,
    fitted: dict,
    horizon: int = 16,
    primary_obs: str | None = None,
) -> list[dict]:
    """Condense the analyses into a pass/warn/fail table.

    Thresholds are deliberately loose -- they exist to draw the eye to the one or
    two things that need attention, not to certify anything.

    ``primary_obs`` is the macrostate the MPC cost is written in (defaults to the
    dose-response / rollout-skill obs, else ``rabbit_count``).
    """
    h = min(horizon, len(herr["steps"]) - 1)
    names = list(probe["names"])
    r2 = np.asarray(probe["r2"])
    skill_full = float(herr["skill_full"][h])
    skill_persist = float(herr["skill_persistence"][h])
    skill_ls = (
        float(herr["skill_ls_linear"][h]) if "skill_ls_linear" in herr else float("nan")
    )
    used = float(pca["participation_ratio"])
    primary = (
        primary_obs
        or dose.get("obs_name")
        or rollout_skill.get("obs_name")
        or "rabbit_count"
    )
    primary_r2 = float(readout_r2_test.get(primary, float("nan")))
    u_r2 = float(r2[names.index("u_applied")])

    return [
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
            "note": "a model that loses to a frozen latent learned no dynamics",
        },
        {
            "name": "no collapse (dims in use)",
            "value": f"{used:.1f} / {model.latent_dim}",
            "status": _verdict(used / model.latent_dim, 0.5, 0.25),
            "note": f"participation ratio; {pca['n95']} components cover 95%",
        },
        {
            "name": f"{primary} readout R^2",
            "value": f"{primary_r2:+.3f}",
            "status": _verdict(primary_r2, 0.85, 0.6),
            "note": "held-out; this is what the MPC cost is written in",
        },
        {
            "name": "control legible in latent",
            "value": f"{u_r2:+.3f}",
            "status": _verdict(u_r2, 0.5, 0.2),
            "note": "R^2 predicting the applied u from z alone",
        },
        {
            "name": "macrostate rollout corr",
            "value": f"{rollout_skill['corr']:+.3f}",
            "status": _verdict(rollout_skill["corr"], 0.85, 0.6),
            "note": f"free-running {primary}; RMSE {rollout_skill['rmse']:.1f}",
        },
        {
            "name": "dose-response monotonic",
            "value": f"{dose['spearman']:+.2f}",
            "status": _verdict(-dose["spearman"], 0.99, 0.7),
            "note": f"-1 means more culling reliably predicts fewer {primary}",
        },
        {
            "name": "ls_linear skill (diagnostic)",
            "value": f"{skill_ls:+.3f}",
            "status": "n/a",
            "note": (
                f"post-hoc LS operator skill; one-step R^2 = {fitted['one_step_r2']:.3f}. "
                "With predictor='linear', full should match this; a gap means under-training"
            ),
        },
    ]


def format_scorecard(rows: list[dict]) -> str:
    """Render :func:`scorecard` output as a fixed-width table."""
    mark = {"ok": "[ ok ]", "warn": "[warn]", "BAD": "[FAIL]", "n/a": "[ -- ]"}
    w = max(len(r["name"]) for r in rows)
    lines = [f"{'metric':<{w}}  {'value':>14}  status  note", "-" * (w + 34)]
    for r in rows:
        lines.append(
            f"{r['name']:<{w}}  {r['value']:>14}  {mark[r['status']]}  {r['note']}"
        )
    return "\n".join(lines)
