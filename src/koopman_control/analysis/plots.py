"""Plotting helpers for world-model evaluation.

Kept separate from :mod:`koopman_control.analysis.evaluate` (which holds the
numeric logic) so the notebook can import compute and plotting independently.
Each ``fig_*`` returns a Matplotlib Figure so it renders inline in a notebook.

The figures are grouped the way the companion notebook reads them:

1. ``fig_learning_curves``, ``fig_horizon_errors`` -- did training work, and does
   the model beat trivial baselines?
2. ``fig_reconstructions``, ``fig_prediction_frames`` -- what does the model
   actually see and imagine, in pixel space?
3. ``fig_latent_pca``, ``fig_latent_probe``, ``fig_latent_traces`` -- is the
   latent a usable state?
4. ``fig_macrostate_rollout``, ``fig_control_response``, ``fig_dose_response``,
   ``fig_step_response`` -- does the model understand the control?
5. ``fig_spectrum``, ``fig_mode_map``, ``fig_control_direction`` -- is the
   learned linear system fit for a controller?
6. ``fig_control_coverage`` -- does the *data* even identify the actuator?
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from koopman_control.data.rabbit_grass import (
    GRASS_CHANNEL,
    RABBIT_CHANNEL,
    RabbitGrassConfig,
    rollout,
)
from koopman_control.analysis import evaluate as ev
from koopman_control.models.world_model import LatentWorldModel


# ======================================================================
# 1. Training and prediction accuracy
# ======================================================================
def fig_learning_curves(metrics_csv: str | Path):
    rows = list(csv.DictReader(open(metrics_csv)))
    agg: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        e = r.get("epoch", "")
        if not e:
            continue
        for k, v in r.items():
            if v not in ("", None) and k != "epoch":
                try:
                    agg[e][k] = float(v)
                except ValueError:
                    pass
    epochs = sorted((int(e) for e in agg), key=int)

    def series(key):
        return [agg[str(e)].get(key, np.nan) for e in epochs]

    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    for key in ("train_loss", "val_loss", "val_latent"):
        axes[0].plot(epochs, series(key), marker="o", label=key)
    axes[0].set_title("Losses")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, series("val_latent"), marker="o", label="full")
    axes[1].plot(epochs, series("val_latent_linear"), marker="o", label="linear-only")
    axes[1].set_title("Latent prediction: full vs linear")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Anti-collapse terms: vic_var pinned at 0 means every latent dimension is
    # comfortably above the variance floor, which is what we want.
    for key in ("val_vic_var", "val_vic_cov", "val_recon", "val_pred_pix"):
        vals = series(key)
        if not np.all(np.isnan(vals)):
            axes[2].plot(epochs, vals, marker="o", label=key.replace("val_", ""))
    axes[2].set_title("Regularizers and reconstruction")
    axes[2].set_xlabel("epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    ax3b = axes[3].twinx()
    axes[3].plot(epochs, series("spectral_radius"), marker="o", color="C3")
    axes[3].axhline(1.0, color="k", ls="--", alpha=0.4)
    ax3b.plot(epochs, series("controllability_rank"), marker="s", color="C4")
    axes[3].set_title("Linear-system diagnostics")
    axes[3].set_xlabel("epoch")
    axes[3].set_ylabel("spectral radius", color="C3")
    ax3b.set_ylabel("controllability rank", color="C4")
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def fig_horizon_errors(res: dict, *, max_step: int | None = None):
    """Latent error vs horizon against baselines, plus variance-explained skill.

    Left panel is raw MSE on a log scale; right panel is ``1 - mse/var``, which
    is the readable version: above 0 means better than guessing the mean latent,
    and 1.0 would be perfect. ``persistence`` (freeze ``z_0``) is the bar the
    model must clear to have learned any dynamics at all.
    """
    keys = [
        ("full", "full dynamics", "C0", "-"),
        ("linear", "model linear core", "C1", "--"),
        ("ls_linear", "best linear fit (DMDc)", "C2", "-."),
        ("persistence", "persistence baseline", "0.5", ":"),
    ]
    sl = slice(0, max_step)
    steps = res["steps"][sl]

    fig, (ax_mse, ax_skill) = plt.subplots(1, 2, figsize=(14, 5))
    for key, label, color, ls in keys:
        if key not in res:
            continue
        ax_mse.plot(steps, res[key][sl], label=label, color=color, ls=ls)
        ax_skill.plot(steps, res[f"skill_{key}"][sl], label=label, color=color, ls=ls)

    ax_mse.axhline(res["latent_var"], color="k", alpha=0.4, lw=1)
    ax_mse.annotate(
        "latent variance (error of predicting the mean)",
        xy=(steps[len(steps) // 3], res["latent_var"]),
        fontsize=8,
        va="bottom",
        color="k",
        alpha=0.7,
    )
    ax_mse.set_yscale("log")
    ax_mse.set_ylabel("mean latent MSE")
    ax_mse.set_title("Free-running latent error vs horizon")

    ax_skill.axhline(0.0, color="k", alpha=0.4, lw=1)
    ax_skill.set_ylim(-0.5, 1.02)
    ax_skill.set_ylabel("variance explained  $1 - \\mathrm{mse}/\\mathrm{var}$")
    ax_skill.set_title("Prediction skill (higher is better, 0 = useless)")

    for ax in (ax_mse, ax_skill):
        ax.set_xlabel("rollout step")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ======================================================================
# 2. Pixel-space views
# ======================================================================
def _show(ax, img, *, title=None, cmap="viridis", vmin=0.0, vmax=1.0):
    ax.imshow(
        np.asarray(img, dtype=np.float32).T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def _row_vmax(data: np.ndarray) -> float:
    """Color ceiling for a row of decoded panels.

    Decoded occupancy is a probability field whose values sit far below 1, so a
    fixed 0-1 scale renders every prediction as an almost-black square and hides
    whatever spatial structure exists. Each row is scaled to its own maximum and
    that maximum is printed in the row label so the panels stay honest.
    """
    return float(max(np.max(np.asarray(data, dtype=np.float32)), 1e-3))


def _image_grid(rows, times, *, labels_prefix="t=", suptitle=""):
    """Render ``rows`` of ``(channel, stack, label)`` as a panel grid.

    Each row gets its own color scale (see :func:`_row_vmax`) because the true
    frames are binary while the decoded ones are diffuse probabilities.
    """
    fig, axes = plt.subplots(
        len(rows), len(times), figsize=(2.3 * len(times), 2.4 * len(rows)), squeeze=False
    )
    for r, (ch, data, label) in enumerate(rows):
        vmax = _row_vmax(data[:, ch])
        for c, t in enumerate(times):
            _show(
                axes[r, c],
                data[c, ch],
                vmax=vmax,
                title=f"{labels_prefix}{t}  mass={data[c, ch].sum():.0f}",
            )
        axes[r, 0].set_ylabel(f"{label}\n(scale 0–{vmax:.2f})", fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, y=0.995)
    fig.tight_layout()
    return fig


def fig_reconstructions(model: LatentWorldModel, traj: dict, times=(0, 25, 75, 150, 200)):
    """Autoencoder fidelity: encode a real frame and decode it straight back.

    No dynamics involved, so this isolates the encoder/decoder bottleneck. The
    decoded rabbit channel is a *probability* field, so it will look softer and
    more diffuse than the binary truth -- read the spatial pattern and the total
    mass (in each title), not the crispness.
    """
    times = [t for t in times if t < len(traj["frames"])]
    frames = np.asarray(traj["frames"], dtype=np.float32)[times]
    recon = ev.reconstruct(model, frames)
    return _image_grid(
        [
            (GRASS_CHANNEL, frames, "true grass"),
            (GRASS_CHANNEL, recon, "decoded grass"),
            (RABBIT_CHANNEL, frames, "true rabbit"),
            (RABBIT_CHANNEL, recon, "decoded rabbit"),
        ],
        times,
        suptitle=(f"Reconstruction (no dynamics) — run {traj['run_id']} [{traj['excitation']}]"),
    )


def fig_prediction_frames(
    model: LatentWorldModel,
    traj: dict,
    *,
    start: int = 0,
    horizons=(0, 1, 5, 10, 25, 50),
):
    """Free-running *imagination*: roll the latent forward and decode each step.

    Unlike :func:`fig_reconstructions`, only frame ``start`` is ever seen; the
    rest is predicted from the control sequence alone. Blurring with horizon is
    expected and correct -- individual agent positions are unpredictable, so the
    model should converge to a density. The failure modes to look for instead are
    the mass drifting to zero or saturating, or the spatial pattern dissolving
    into a uniform wash that carries no information.
    """
    res = ev.predicted_frames(model, traj, start=start, horizons=horizons)
    return _image_grid(
        [
            (GRASS_CHANNEL, res["true"], "true grass"),
            (GRASS_CHANNEL, res["pred"], "predicted grass"),
            (RABBIT_CHANNEL, res["true"], "true rabbit"),
            (RABBIT_CHANNEL, res["pred"], "predicted rabbit"),
        ],
        res["horizons"],
        labels_prefix="+",
        suptitle=(
            f"Free-running prediction from t={start} — run {traj['run_id']} "
            f"[{traj['excitation']}] (only the first column was observed)"
        ),
    )


# ======================================================================
# 3. Latent-space structure
# ======================================================================
def fig_latent_pca(pca: dict, *, stride: int = 3, n_traj_lines: int = 6):
    """The latent point cloud in its top two principal components.

    Each panel is the same cloud colored by a different ground-truth quantity. If
    the coloring forms a smooth gradient, that quantity is encoded in the
    latent's dominant directions -- exactly what a controller needs. Salt-and-
    pepper coloring means the quantity is either absent or buried in minor
    directions. The bottom-right panel overlays individual trajectories to show
    whether the latent traces coherent paths rather than jumping around.
    """
    coords = pca["coords"][:, ::stride]
    obs = pca["obs"][:, ::stride]
    ctrl = pca["controls"][:, ::stride]
    names = pca["obs_names"]
    x = coords[..., 0].ravel()
    y = coords[..., 1].ravel()

    time_idx = np.broadcast_to(
        np.arange(coords.shape[1])[None, :] * stride, coords.shape[:2]
    ).ravel()
    panels = [
        (obs[..., names.index("rabbit_count")].ravel(), "rabbit count", "viridis"),
        (obs[..., names.index("grass_frac")].ravel(), "grass fraction", "YlGn"),
        (ctrl.ravel(), "control $u$ applied", "magma"),
        (time_idx, "timestep", "cividis"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    axes = axes.ravel()
    for ax, (color, label, cmap) in zip(axes, panels):
        sc = ax.scatter(x, y, c=color, s=3, cmap=cmap, alpha=0.55, linewidths=0)
        fig.colorbar(sc, ax=ax, fraction=0.046).set_label(label, fontsize=9)
        ax.set_title(f"colored by {label}", fontsize=10)

    # Panel 5: trajectories as paths, to check temporal coherence. Runs are
    # picked across excitation types, otherwise the split's ordering hands back
    # several copies of the same signal.
    ax = axes[4]
    seen: set[str] = set()
    picks = []
    for i, m in enumerate(pca["meta"]):
        if m["excitation"] not in seen:
            seen.add(m["excitation"])
            picks.append(i)
        if len(picks) >= n_traj_lines:
            break
    for i in picks:
        ax.plot(
            coords[i, :, 0], coords[i, :, 1], lw=1, alpha=0.8, label=pca["meta"][i]["excitation"]
        )
        ax.scatter(coords[i, 0, 0], coords[i, 0, 1], marker="o", s=25, color="k", zorder=3)
    ax.set_title("individual trajectories (dot = start)", fontsize=10)
    ax.legend(fontsize=7)

    # Panel 6: how much of the latent is actually used.
    ax = axes[5]
    k = min(16, len(pca["evr"]))
    ax.bar(np.arange(1, k + 1), pca["evr"][:k], color="C0", alpha=0.8)
    ax.plot(np.arange(1, k + 1), pca["cumulative"][:k], color="C3", marker="o", ms=3)
    ax.axhline(0.95, color="k", ls="--", alpha=0.4)
    ax.set_title(
        f"variance per component\nparticipation ratio = {pca['participation_ratio']:.1f}"
        f" of {len(pca['evr'])} dims, {pca['n95']} comps for 95%",
        fontsize=9,
    )
    ax.set_xlabel("component")

    for ax in axes[:5]:
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.grid(alpha=0.2)
    fig.suptitle("Latent space: PCA of encoded frames across held-out runs", y=0.995)
    fig.tight_layout()
    return fig


def fig_latent_probe(probe: dict):
    """Held-out R^2 of a linear readout from the latent to known quantities.

    This is the most direct answer to "is the latent capturing anything useful".
    ``rabbit_count`` near 1 means the population is a linear function of the
    latent, so a linear controller can regulate it directly. ``u_applied`` tests
    whether the encoder even registers the control -- an image of culled rabbits
    should look different from an unculled one.
    """
    names, r2 = probe["names"], np.asarray(probe["r2"])
    order = np.argsort(r2)
    colors = ["C2" if v > 0.75 else "C1" if v > 0.4 else "C3" for v in r2[order]]

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 2))
    ax.barh(np.arange(len(names)), r2[order], color=colors, alpha=0.85)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([names[i] for i in order])
    for i, v in enumerate(r2[order]):
        ax.text(max(v, 0) + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlim(min(-0.05, r2.min() - 0.05), 1.12)
    ax.set_xlabel("held-out $R^2$ of a ridge readout from $z$")
    ax.set_title(
        f"Linear decodability of the latent\n({probe['n_train']:,} train / "
        f"{probe['n_test']:,} test frames)"
    )
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def fig_latent_traces(model: LatentWorldModel, traj: dict, *, n_dims: int = 6):
    """Per-dimension latent time series: encoded truth vs free-running prediction.

    Picks the dimensions that vary most in this run. Smooth encoded traces mean
    the encoder is extracting a slow macrostate from noisy frames; jagged traces
    mean it is tracking per-frame agent jitter, which is unpredictable by
    construction and will cap the achievable latent accuracy. Where prediction
    and truth diverge tells you *which* directions the dynamics fail on.
    """
    z_true = ev.encode_trajectory(model, traj["frames"])
    z_full, z_lin = ev.latent_rollout(model, z_true[0], traj["controls"])
    picks = np.argsort(-z_true.std(axis=0))[:n_dims]

    ncols = 3
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.1 * nrows), squeeze=False)
    for ax, d in zip(axes.ravel(), picks):
        ax.plot(z_true[:, d], color="C0", lw=1.4, label="encoded (truth)")
        ax.plot(z_full[:, d], color="C1", ls="--", lw=1.2, label="predicted (full)")
        ax.plot(z_lin[:, d], color="C2", ls=":", lw=1.2, label="predicted (linear)")
        ax.set_title(f"latent dim {d}  (std={z_true[:, d].std():.2f})", fontsize=9)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(picks) :]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Latent trajectories, highest-variance dimensions — run {traj['run_id']} "
        f"[{traj['excitation']}]",
        y=0.995,
    )
    fig.tight_layout()
    return fig


# ======================================================================
# 4. Macrostate and control response
# ======================================================================
def _diverse_by_excitation(trajs: list[dict], n: int) -> list[dict]:
    """Pick up to ``n`` trajectories spanning distinct excitation types."""
    seen: set[str] = set()
    picked: list[dict] = []
    for tr in trajs:
        if tr["excitation"] not in seen:
            seen.add(tr["excitation"])
            picked.append(tr)
        if len(picked) >= n:
            return picked
    for tr in trajs:  # backfill if fewer distinct types than n
        if tr not in picked:
            picked.append(tr)
        if len(picked) >= n:
            break
    return picked


def fig_macrostate_rollout(model: LatentWorldModel, trajs: list[dict], n: int = 4):
    n = min(n, len(trajs))
    picks = _diverse_by_excitation(trajs, n)
    fig, axes = plt.subplots(2, n, figsize=(4.4 * n, 7.5), sharex=True)
    axes = np.atleast_2d(axes).reshape(2, n)
    for col, tr in enumerate(picks):
        z_true = ev.encode_trajectory(model, tr["frames"])
        z_full, z_lin = ev.latent_rollout(model, z_true[0], tr["controls"])
        pred = ev.decode_mass(model, z_full)
        pred_lin = ev.decode_mass(model, z_lin)
        true = ev.true_mass(tr["frames"])

        for row, ch in ((0, RABBIT_CHANNEL), (1, GRASS_CHANNEL)):
            ax = axes[row, col]
            ax.plot(true[:, ch], color="C0", label="true")
            ax.plot(pred[:, ch], color="C1", ls="--", label="predicted (full)")
            ax.plot(pred_lin[:, ch], color="C2", ls=":", lw=1, label="predicted (linear)")
            axc = ax.twinx()
            axc.fill_between(np.arange(len(tr["controls"])), tr["controls"], color="C3", alpha=0.12)
            axc.set_ylim(-0.05, 1.05)
            axc.set_yticks([])
            ax.grid(alpha=0.3)
            if row == 1:
                ax.set_xlabel("step")
        axes[0, col].set_title(
            f"{tr['excitation']}  r0={tr['initial_rabbits']} seed={tr['seed']}", fontsize=10
        )
    axes[0, 0].set_ylabel("rabbit occupancy mass")
    axes[1, 0].set_ylabel("grass occupancy mass")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Macrostate rollout: predicted vs true occupancy mass (control shaded red)",
        y=0.995,
    )
    fig.tight_layout()
    return fig


def fig_control_response(
    model: LatentWorldModel,
    *,
    initial_rabbits: int = 120,
    initial_grass_prob: float = 0.35,
    seed: int = 0,
    steps: int = 80,
    controls=(0.0, 0.3, 0.6, 0.9),
    cfg: RabbitGrassConfig | None = None,
):
    """Does the model reproduce the true control authority (sign + magnitude)?

    For each constant cull level, roll the true simulator and the model (from the
    same initial frame) and overlay decoded rabbit mass. A correct model shows
    higher cull -> lower mass, tracking the simulator.
    """
    cfg = cfg or RabbitGrassConfig()
    fig, (ax_true, ax_model) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for i, u in enumerate(controls):
        useq = np.full(steps, float(u), dtype=np.float32)
        frames, ctrl, _ = rollout(
            cfg,
            useq,
            initial_rabbits=initial_rabbits,
            initial_grass_prob=initial_grass_prob,
            seed=seed,
        )
        true_r = ev.true_mass(frames)[:, RABBIT_CHANNEL]
        z0 = ev.encode_trajectory(model, frames[:1])[0]
        z_full, _ = ev.latent_rollout(model, z0, ctrl)
        pred_r = ev.decode_mass(model, z_full)[:, RABBIT_CHANNEL]
        ax_true.plot(true_r, color=f"C{i}", label=f"u={u}")
        ax_model.plot(pred_r, color=f"C{i}", ls="--", label=f"u={u}")
    ax_true.set_title("True simulator")
    ax_model.set_title("Model prediction")
    for a in (ax_true, ax_model):
        a.set_xlabel("step")
        a.grid(alpha=0.3)
        a.legend()
    ax_true.set_ylabel("rabbit occupancy mass")
    fig.suptitle("Control response: constant cull levels")
    fig.tight_layout()
    return fig


def fig_dose_response(dose: dict):
    """Input-output curve: sustained cull level vs resulting population.

    The single most control-relevant plot. A usable model does not need the
    magnitudes right, but it *must* get the ordering right -- if predicted
    population does not fall as ``u`` rises, any controller built on it will push
    the wrong way. Left panel is absolute mass on each side (the scales differ
    because decoded mass is a probability sum); right panel normalizes both to
    their ``u=0`` value so the shapes can be compared directly.
    """
    u = dose["u"]
    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(13, 5))

    ax_t = ax_abs
    ax_p = ax_abs.twinx()
    ax_t.errorbar(
        u,
        dose["true_mean"],
        yerr=dose["true_std"],
        color="C0",
        marker="o",
        capsize=3,
        label="true simulator",
    )
    ax_p.errorbar(
        u,
        dose["pred_mean"],
        yerr=dose["pred_std"],
        color="C1",
        marker="s",
        ls="--",
        capsize=3,
        label="model",
    )
    ax_t.set_ylabel("true rabbit mass", color="C0")
    ax_p.set_ylabel("predicted rabbit mass", color="C1")
    ax_abs.set_title("Absolute (independent y-axes)")
    lines = ax_t.get_lines()[:1] + ax_p.get_lines()[:1]
    ax_t.legend(lines, ["true simulator", "model"], fontsize=9)

    for vals, color, marker, label in (
        (dose["true_mean"], "C0", "o", "true simulator"),
        (dose["pred_mean"], "C1", "s", "model"),
    ):
        ref = vals[0] if abs(vals[0]) > 1e-9 else 1.0
        ax_rel.plot(u, vals / ref, color=color, marker=marker, label=label)
    ax_rel.axhline(1.0, color="k", alpha=0.3, lw=1)
    ax_rel.set_ylabel("population relative to $u=0$")
    ax_rel.set_title(f"Normalized shape (model monotonicity: Spearman = {dose['spearman']:+.2f})")
    ax_rel.legend(fontsize=9)

    for a in (ax_abs, ax_rel):
        a.set_xlabel("constant cull level $u$")
        a.grid(alpha=0.3)
    fig.suptitle("Dose-response: does more culling mean fewer rabbits?")
    fig.tight_layout()
    return fig


def fig_step_response(model: LatentWorldModel, z0: np.ndarray, *, steps: int = 80):
    """Classic step and impulse responses of the learned system.

    Both are plotted as the *deviation* from a zero-control rollout, so the curve
    is the control's own contribution. Read three things: the sign (down is
    correct for culling), the settling time (how many steps until the effect
    saturates, which sets how far a controller must look ahead), and whether the
    curves for different ``u`` scale in proportion -- proportional spacing means
    the actuator behaves linearly and one gain works everywhere.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, impulse, title in (
        (axes[0], False, "Step response (u held from t=0)"),
        (axes[1], True, "Impulse response (single-step u)"),
    ):
        res = ev.step_response(model, z0, steps=steps, impulse=impulse)
        for i, u in enumerate(res["u_levels"]):
            ax.plot(res["steps"], res[f"full_{u}"], color=f"C{i}", label=f"u={u} full")
            ax.plot(res["steps"], res[f"linear_{u}"], color=f"C{i}", ls=":", alpha=0.7)
        ax.axhline(0, color="k", lw=1, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="dotted = linear core", title_fontsize=8)
    axes[0].set_ylabel("change in decoded rabbit mass vs $u=0$")
    fig.suptitle("Learned actuator dynamics (deviation from the uncontrolled rollout)")
    fig.tight_layout()
    return fig


# ======================================================================
# 5. Structure of the learned linear system
# ======================================================================
def fig_spectrum(model: LatentWorldModel):
    a, _ = model.linear_system()
    eig = np.linalg.eigvals(a)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="k", alpha=0.4)
    ax.scatter(eig.real, eig.imag, c="C1")
    ax.axhline(0, color="k", alpha=0.2)
    ax.axvline(0, color="k", alpha=0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    spec = float(np.max(np.abs(eig)))
    ax.set_title(f"Eigenvalues of A (spectral radius = {spec:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_mode_map(modes: dict, ctrl_sv: np.ndarray):
    """Which dynamic modes exist, and which of them can the actuator reach?

    Left: the spectrum again, but each eigenvalue sized and colored by how
    strongly the control excites that mode. The dangerous pattern is a marker
    near the unit circle (slow, long-lived) that is dark (unreachable) -- that is
    a persistent behaviour you cannot steer. Middle: mode half-lives, i.e. how
    many steps each mode takes to decay by half, which sets the timescale a
    controller has to work on. Right: singular values of the controllability
    matrix, the honest continuous version of the integer rank.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="k", alpha=0.4)
    sc = ax.scatter(
        modes["eig"].real,
        modes["eig"].imag,
        c=modes["mode_ctrl"],
        s=30 + 320 * modes["mode_ctrl"],
        cmap="plasma",
        alpha=0.85,
        edgecolors="k",
        linewidths=0.4,
    )
    fig.colorbar(sc, ax=ax, fraction=0.046).set_label("mode controllability", fontsize=9)
    ax.axhline(0, color="k", alpha=0.2)
    ax.axvline(0, color="k", alpha=0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("Modes of A, sized by reachability")
    ax.grid(alpha=0.3)

    ax = axes[1]
    finite = np.isfinite(modes["half_life"])
    idx = np.arange(len(modes["half_life"]))
    ax.bar(idx[finite], modes["half_life"][finite], color="C0", alpha=0.8)
    if (~finite).any():
        ax.bar(
            idx[~finite],
            np.nanmax(modes["half_life"][finite], initial=1.0),
            color="C3",
            alpha=0.6,
            label="non-decaying ($|\\lambda|\\geq 1$)",
        )
        ax.legend(fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("mode (sorted by $|\\lambda|$)")
    ax.set_ylabel("half-life (steps)")
    ax.set_title("How long each mode persists")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ax.semilogy(
        np.arange(1, len(ctrl_sv) + 1), np.maximum(ctrl_sv, 1e-18), marker="o", ms=3, color="C4"
    )
    ax.axhline(1e-6, color="k", ls="--", alpha=0.4)
    ax.annotate("rank tolerance", xy=(1, 1.3e-6), fontsize=8, alpha=0.7)
    ax.set_xlabel("direction")
    ax.set_ylabel("normalized singular value")
    ax.set_title("Control authority per direction")
    ax.grid(alpha=0.3)

    fig.suptitle("Control-theoretic structure of the learned linear system", y=0.99)
    fig.tight_layout()
    return fig


def fig_control_direction(model: LatentWorldModel, z_ref: np.ndarray, *, u: float = 1.0):
    """What the control does, in latent coordinates and in pixel space.

    Left: the entries of ``B``, i.e. which latent dimensions the actuator moves,
    for the instantaneous input and its one-step lag. A visible lag column
    confirms the model learned the simulator's delayed actuator instead of
    assuming instantaneous control. Right: decode the latent before and after
    pushing it along the control direction. The difference map should be
    concentrated on the rabbit channel and negative (fewer rabbits) -- if the
    control mostly repaints grass or shifts the whole image, the model has
    attributed the actuator to the wrong thing.
    """
    b = model.B.weight.detach().cpu().numpy()
    eff = ev.control_effect_map(model, z_ref, u=u)

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.5, 1, 1, 1])

    ax = fig.add_subplot(gs[:, 0])
    idx = np.arange(b.shape[0])
    width = 0.4
    labels = ["$u_t$ (instant)", "$u_{t-1}$ (lag)"]
    for j in range(b.shape[1]):
        ax.barh(
            idx + (j - 0.5) * width,
            b[:, j],
            height=width,
            label=labels[j] if j < len(labels) else f"lag {j}",
            alpha=0.85,
        )
    ax.axvline(0, color="k", lw=1)
    ax.set_ylabel("latent dimension")
    ax.set_xlabel("B entry")
    ax.set_title("Control input matrix $B$", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    # Per-channel color limits: the grass response is an order of magnitude
    # larger than the rabbit response, so a shared scale would hide the rabbit
    # channel entirely -- which is the one that matters here.
    for r, ch, name in ((0, GRASS_CHANNEL, "grass"), (1, RABBIT_CHANNEL, "rabbit")):
        occ = _row_vmax(np.stack([eff["base"][ch], eff["perturbed"][ch]]))
        lim = float(np.abs(eff["delta"][ch]).max()) or 1.0
        for c, (img, title, cmap, vmin, vmax) in enumerate(
            (
                (eff["base"][ch], f"{name}: before", "viridis", 0.0, occ),
                (eff["perturbed"][ch], f"{name}: after", "viridis", 0.0, occ),
                (
                    eff["delta"][ch],
                    f"{name}: change\nsum={eff['delta'][ch].sum():+.1f}",
                    "bwr",
                    -lim,
                    lim,
                ),
            ),
            start=1,
        ):
            _show(fig.add_subplot(gs[r, c]), img, title=title, cmap=cmap, vmin=vmin, vmax=vmax)

    fig.suptitle(
        f"Learned actuator: pushing the latent along $B\\,u$ with $u={u}$ "
        "(blue = decrease, red = increase)",
        y=0.98,
    )
    fig.tight_layout()
    return fig


# ======================================================================
# 6. Dataset-side diagnostics
# ======================================================================
def fig_control_coverage(trajs: list[dict]):
    """What control signals does the held-out data actually contain?

    Identifiability is a property of the *data*, not the model: you can only
    learn how the actuator behaves at amplitudes and rates the data visits. Left
    shows one example of each excitation type; middle is the marginal
    distribution of ``u``; right is the joint distribution of consecutive values,
    where mass off the diagonal means the data contains fast changes (which is
    what pins down the lag term) and mass on the diagonal means sustained holds
    (which pins down the steady-state gain).
    """
    by_type: dict[str, dict] = {}
    for tr in trajs:
        by_type.setdefault(tr["excitation"], tr)
    u_all = np.concatenate([tr["controls"] for tr in trajs])
    u_now = np.concatenate([tr["controls"][1:] for tr in trajs])
    u_prev = np.concatenate([tr["controls"][:-1] for tr in trajs])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for i, (name, tr) in enumerate(sorted(by_type.items())):
        axes[0].plot(tr["controls"], lw=1.1, alpha=0.85, color=f"C{i}", label=name)
    axes[0].set_title("One example per excitation type")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("control $u$")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)

    axes[1].hist(u_all, bins=40, color="C0", alpha=0.85)
    axes[1].set_title(f"Amplitude coverage (mean={u_all.mean():.2f})")
    axes[1].set_xlabel("control $u$")
    axes[1].set_ylabel("frames")
    axes[1].grid(alpha=0.3)

    h = axes[2].hist2d(u_prev, u_now, bins=30, cmap="magma")
    fig.colorbar(h[3], ax=axes[2], fraction=0.046).set_label("frames", fontsize=9)
    axes[2].set_xlabel("$u_{t-1}$")
    axes[2].set_ylabel("$u_t$")
    axes[2].set_title("Rate coverage (off-diagonal = fast changes)")

    fig.suptitle("Control excitation present in the held-out data", y=0.99)
    fig.tight_layout()
    return fig
