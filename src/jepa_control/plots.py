"""Plotting helpers for JEPA latent-model evaluation.

Kept separate from :mod:`jepa_control.evaluate` (the numeric logic) so the
companion notebook can import compute and plotting independently. Each ``fig_*``
returns a Matplotlib Figure so it renders inline.

There is no decoder, so there are no pixel-space panels here. The figures are
grouped the way the notebook reads them:

1. ``fig_learning_curves``, ``fig_horizon_errors`` -- did training work, and does
   the model beat trivial baselines?
2. ``fig_latent_pca``, ``fig_latent_probe``, ``fig_latent_traces`` -- is the
   latent uncollapsed and interpretable?
3. ``fig_readout_quality``, ``fig_readout_rollout`` -- can the macrostate be read
   out of the latent, and does it survive a free rollout?
4. ``fig_dose_response``, ``fig_step_response`` -- does the model understand the
   control well enough to plan with?
5. ``fig_closed_loop``, ``fig_controller_compare``, ``fig_tumor_controller_compare``
   -- does MPC actually steer the true ABM, and does it beat a fitted ODE?

``fig_control_coverage`` (what excitation the *data* contains) is reused from the
sibling package, since it is a property of the dataset and not of the model.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jepa_control import evaluate as ev
from jepa_control.model import JEPAControl
from koopman_control.analysis.plots import fig_control_coverage  # noqa: F401  (re-export)


# ======================================================================
# 1. Training and prediction accuracy
# ======================================================================
def fig_learning_curves(metrics_csv: str | Path):
    """Losses, the VICReg anti-collapse terms, and the participation ratio.

    ``vic_var`` pinned near 0 means every latent dimension sits comfortably above
    the variance floor, which is exactly what prevents collapse. A participation
    ratio that decays over training is the warning sign to catch early: the model
    is quietly shrinking into a subspace even while the loss improves.
    """
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

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for key in ("train_loss", "val_loss", "val_pred"):
        vals = series(key)
        if not np.all(np.isnan(vals)):
            axes[0].plot(epochs, vals, marker="o", label=key)
    axes[0].set_title("Losses")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for key in ("val_vic_var", "val_vic_cov", "val_readout"):
        vals = series(key)
        if not np.all(np.isnan(vals)):
            axes[1].plot(epochs, vals, marker="o", label=key.replace("val_", ""))
    axes[1].set_title("VICReg / anti-collapse terms")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    pr = series("participation_ratio")
    axes[2].plot(epochs, pr, marker="o", color="C4")
    axes[2].set_title("Participation ratio (latent dims in use)")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("effective dimensions")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def fig_horizon_errors(res: dict, *, max_step: int | None = None):
    """Free-running latent error vs horizon, against baselines.

    Left is raw MSE on a log scale; right is ``1 - mse/var``, the readable
    version (above 0 beats guessing the mean latent, 1.0 is perfect).
    ``persistence`` is the bar the model must clear to have learned any dynamics.
    ``ls_linear`` is the best a linear operator could do *in this same latent*. For
    the default linear predictor, ``full`` should sit on top of it -- a visible gap
    means the predictor is under-trained. For the ``residual_mlp`` ablation, the gap
    instead measures what the nonlinearity is buying.
    """
    keys = [
        ("full", "JEPA predictor", "C0", "-"),
        ("ls_linear", "best linear fit (diagnostic)", "C2", "-."),
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
# 2. Latent structure and collapse
# ======================================================================
def fig_latent_pca(pca: dict, *, stride: int = 3, n_traj_lines: int = 6):
    """The latent point cloud in its top two principal components.

    Same cloud in each panel, colored by a different ground-truth quantity. A
    smooth gradient means that quantity lives in the latent's dominant
    directions -- what MPC needs. Salt-and-pepper means it is absent or buried.
    The last panel is the collapse check: with VICReg working, variance should be
    spread over many components rather than concentrated in one or two.
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
    # Prefer the controlled population and its coupled population for each case.
    if "infected_count" in names:
        primary, secondary = "infected_count", "susceptible_count"
    elif "tumor_count" in names:
        primary, secondary = "tumor_count", "healthy_count"
    elif "wolf_count" in names:
        primary, secondary = "wolf_count", "rabbit_count"
    else:
        primary, secondary = "rabbit_count", "grass_frac"
    panels = [
        (obs[..., names.index(primary)].ravel(), primary.replace("_", " "), "viridis"),
        (obs[..., names.index(secondary)].ravel(), secondary.replace("_", " "), "YlGn"),
        (ctrl.ravel(), "control $u$ applied", "magma"),
        (time_idx, "timestep", "cividis"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    axes = axes.ravel()
    for ax, (color, label, cmap) in zip(axes, panels, strict=False):
        sc = ax.scatter(x, y, c=color, s=3, cmap=cmap, alpha=0.55, linewidths=0)
        fig.colorbar(sc, ax=ax, fraction=0.046).set_label(label, fontsize=9)
        ax.set_title(f"colored by {label}", fontsize=10)

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

    The most direct answer to "did the self-supervised encoder keep anything
    useful". ``rabbit_count`` near 1 means the macrostate is a linear function of
    the latent, which is the precondition for writing the MPC cost in readout
    space. ``u_applied`` tests whether the encoder registers the control at all.
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
    ax.set_title("Linear decodability of the JEPA latent")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def fig_latent_traces(model: JEPAControl, traj: dict, *, n_dims: int = 6):
    """Per-dimension latent time series: encoded truth vs free-running prediction.

    Picks the dimensions that vary most in this run. Smooth encoded traces mean
    the encoder extracted a slow macrostate from noisy frames; jagged traces mean
    it is tracking per-frame agent jitter, which is unpredictable by construction
    and caps achievable accuracy. Divergence shows *which* directions fail.
    """
    z_true = ev.encode_trajectory(
        model, traj["frames"], traj.get("frame_scale", 1.0)
    )
    z_pred = ev.latent_rollout_batch(model, z_true[0][None], traj["controls"][None])[0]
    picks = np.argsort(-z_true.std(axis=0))[:n_dims]

    ncols = 3
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.1 * nrows), squeeze=False)
    for ax, d in zip(axes.ravel(), picks, strict=False):
        ax.plot(z_true[:, d], color="C0", lw=1.4, label="encoded (truth)")
        ax.plot(z_pred[:, d], color="C1", ls="--", lw=1.2, label="predicted")
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
# 3. The readout (the decoder's replacement)
# ======================================================================
def fig_readout_quality(readout: dict, enc: dict, r2: dict, *, max_points: int = 4000):
    """Fitted readout vs truth, per observable.

    This is the decoder-free equivalent of a reconstruction panel: instead of
    asking whether pixels come back, it asks whether the *quantities we care
    about* are recoverable from ``z`` by a linear map. Points on the diagonal mean
    the latent carries that observable in linearly accessible form.
    """
    names = readout["names"]
    pred = ev.readout_predict(readout, enc["z"]).reshape(-1, len(names))
    # Align columns with the readout: enc["obs"] may include centroids (or other
    # fields) that were excluded at fit time. Reshaping all obs into len(names)
    # columns invents extra rows and then indexes past pred (IndexError).
    cols = [enc["obs_names"].index(n) for n in names]
    true = enc["obs"][..., cols].reshape(-1, len(names))
    if true.shape[0] != pred.shape[0]:
        raise ValueError(
            f"readout pred/true length mismatch: pred={pred.shape[0]} true={true.shape[0]}"
        )
    rng = np.random.default_rng(0)
    idx = rng.choice(len(true), size=min(max_points, len(true)), replace=False)

    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows), squeeze=False)
    for ax, (j, name) in zip(axes.ravel(), enumerate(names), strict=False):
        t, p = true[idx, j], pred[idx, j]
        ax.scatter(t, p, s=3, alpha=0.3, linewidths=0, color="C0")
        lo, hi = float(min(t.min(), p.min())), float(max(t.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=1, alpha=0.6)
        ax.set_title(f"{name}   $R^2$={r2.get(name, float('nan')):+.3f}", fontsize=10)
        ax.set_xlabel("true")
        ax.set_ylabel("readout from $z$")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")
    fig.suptitle("Post-hoc linear readout: does the latent carry the macrostate?", y=0.995)
    fig.tight_layout()
    return fig


def _diverse_by_excitation(trajs: list[dict], n: int) -> list[dict]:
    seen: set[str] = set()
    picked: list[dict] = []
    for tr in trajs:
        if tr["excitation"] not in seen:
            seen.add(tr["excitation"])
            picked.append(tr)
        if len(picked) >= n:
            return picked
    for tr in trajs:
        if tr not in picked:
            picked.append(tr)
        if len(picked) >= n:
            break
    return picked


def fig_readout_rollout(
    model: JEPAControl,
    trajs: list[dict],
    readout: dict,
    *,
    n: int = 4,
    obs_name: str = "rabbit_count",
):
    """Free-running macrostate prediction vs truth (control shaded).

    Only the first frame is observed; everything after is predicted from the
    control sequence alone and mapped through the readout. This is the closest
    thing to "what MPC will believe when it plans", so systematic bias here
    translates directly into steady-state tracking error later.
    """
    n = min(n, len(trajs))
    picks = _diverse_by_excitation(trajs, n)
    idx = readout["names"].index(obs_name)

    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.0), squeeze=False)
    for col, tr in enumerate(picks):
        z_true = ev.encode_trajectory(
            model, tr["frames"], tr.get("frame_scale", 1.0)
        )
        z_pred = ev.latent_rollout_batch(model, z_true[0][None], tr["controls"][None])[0]
        pred = ev.readout_predict(readout, z_pred)[:, idx]
        true = tr["obs"][:, tr["obs_names"].index(obs_name)]

        ax = axes[0, col]
        ax.plot(true, color="C0", label="true")
        ax.plot(pred, color="C1", ls="--", label="predicted (readout)")
        axc = ax.twinx()
        axc.fill_between(np.arange(len(tr["controls"])), tr["controls"], color="C3", alpha=0.12)
        axc.set_ylim(-0.05, 1.05)
        axc.set_yticks([])
        if "initial_tumor_radius" in tr:
            ic = f"radius={tr['initial_tumor_radius']:g}"
        elif "initial_infected" in tr:
            ic = f"i0={tr['initial_infected']}"
        elif "initial_rabbits" in tr:
            ic = f"r0={tr['initial_rabbits']}"
        else:
            ic = ""
        ax.set_title(f"{tr['excitation']}  {ic} seed={tr['seed']}", fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel(obs_name)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Free-running {obs_name}: predicted through the readout vs true (control shaded)",
        y=0.99,
    )
    fig.tight_layout()
    return fig


# ======================================================================
# 4. Control understanding
# ======================================================================
def fig_dose_response(dose: dict):
    """Input-output curve: sustained cull level vs resulting population.

    The single most control-relevant plot. MPC does not need the magnitudes
    right, but it *must* get the ordering right -- if predicted population does
    not fall as ``u`` rises, the planner will push the wrong way. Left is
    absolute on independent axes; right normalizes both to their ``u=0`` value so
    the shapes are directly comparable.
    """
    u = dose["u"]
    obs_name = dose.get("obs_name", "rabbit_count")
    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(13, 5))

    ax_t = ax_abs
    ax_p = ax_abs.twinx()
    ax_t.errorbar(u, dose["true_mean"], yerr=dose["true_std"], color="C0", marker="o", capsize=3)
    ax_p.errorbar(
        u, dose["pred_mean"], yerr=dose["pred_std"], color="C1", marker="s", ls="--", capsize=3
    )
    ax_t.set_ylabel(f"true {obs_name}", color="C0")
    ax_p.set_ylabel("predicted (readout)", color="C1")
    ax_abs.set_title("Absolute (independent y-axes)")
    ax_t.legend(
        ax_t.get_lines()[:1] + ax_p.get_lines()[:1], ["true simulator", "model"], fontsize=9
    )

    for vals, color, marker, label in (
        (dose["true_mean"], "C0", "o", "true simulator"),
        (dose["pred_mean"], "C1", "s", "model"),
    ):
        ref = vals[0] if abs(vals[0]) > 1e-9 else 1.0
        ax_rel.plot(u, vals / ref, color=color, marker=marker, label=label)
    ax_rel.axhline(1.0, color="k", alpha=0.3, lw=1)
    ax_rel.set_ylabel("population relative to $u=0$")
    ax_rel.set_title(f"Normalized shape (monotonicity: Spearman = {dose['spearman']:+.2f})")
    ax_rel.legend(fontsize=9)

    for a in (ax_abs, ax_rel):
        a.set_xlabel("constant cull level $u$")
        a.grid(alpha=0.3)
    fig.suptitle(f"Dose-response: does more culling mean fewer {obs_name}?")
    fig.tight_layout()
    return fig


def fig_tumor_dose_response(dose: dict):
    """Chemotherapy dose response for tumor burden and healthy tissue."""
    u = dose["u"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    panels = (
        (
            "tumor",
            "Late-window tumor count",
            "C3",
        ),
        (
            "healthy",
            "Late-window healthy-cell count",
            "C2",
        ),
    )
    for ax, (name, title, color) in zip(axes, panels, strict=True):
        true_mean = dose[f"true_{name}_mean"]
        true_std = dose[f"true_{name}_std"]
        pred_mean = dose[f"pred_{name}_mean"]
        pred_std = dose[f"pred_{name}_std"]
        ax.errorbar(
            u, true_mean, yerr=true_std, marker="o", color="k", label="true ABM"
        )
        ax.errorbar(
            u,
            pred_mean,
            yerr=pred_std,
            marker="s",
            ls="--",
            color=color,
            label="latent rollout + readout",
        )
        ax.set_xlabel("constant chemotherapy dose $u$")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Tumor-treatment dose response (predicted tumor Spearman = {dose['spearman']:+.2f})",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def fig_sir_dose_response(dose: dict):
    """Vaccination dose response for infection and remaining susceptibles."""
    u = dose["u"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    panels = (
        ("infected", "Late-window infected count", "C3"),
        ("susceptible", "Late-window susceptible count", "C0"),
    )
    for ax, (name, title, color) in zip(axes, panels, strict=True):
        true_mean = dose[f"true_{name}_mean"]
        true_std = dose[f"true_{name}_std"]
        pred_mean = dose[f"pred_{name}_mean"]
        pred_std = dose[f"pred_{name}_std"]
        ax.errorbar(u, true_mean, yerr=true_std, marker="o", color="k", label="true ABM")
        ax.errorbar(
            u,
            pred_mean,
            yerr=pred_std,
            marker="s",
            ls="--",
            color=color,
            label="latent rollout + readout",
        )
        ax.set_xlabel("constant vaccination intensity $u$")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"SIR vaccination dose response "
        f"(predicted infected Spearman = {dose['spearman']:+.2f})",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def fig_sir_control_beliefs(
    beliefs: dict,
    *,
    title: str = "What the latent model believes under constant vaccination",
):
    """Overlay predicted (and optional true) I/S trajectories for several constant ``u``.

    ``beliefs`` is the return value of :func:`jepa_control.evaluate.sir_control_beliefs`.
    Solid lines are free-running latent rollouts through the readout — exactly the
    dynamics CEM uses when planning. Dashed lines, when present, are matched true
    ABM rollouts from the same initial condition.
    """
    u_levels = list(beliefs["u_levels"])
    steps = beliefs["steps"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for i, u in enumerate(u_levels):
        color = f"C{i}"
        pred = beliefs["pred"][u]
        axes[0].plot(steps, pred["infected"], color=color, lw=2.0, label=f"model $u={u:g}$")
        axes[1].plot(steps, pred["susceptible"], color=color, lw=2.0, label=f"model $u={u:g}$")
        if "true" in beliefs and u in beliefs["true"]:
            true = beliefs["true"][u]
            axes[0].plot(steps, true["infected"], color=color, ls="--", alpha=0.55, lw=1.4)
            axes[1].plot(steps, true["susceptible"], color=color, ls="--", alpha=0.55, lw=1.4)
    axes[0].set_ylabel("infected count")
    axes[1].set_ylabel("susceptible count")
    axes[1].set_xlabel("step")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=min(3, len(u_levels)))
    if "true" in beliefs:
        fig.suptitle(f"{title}\n(solid = latent+readout, dashed = true ABM)", y=0.995)
    else:
        fig.suptitle(title, y=0.995)
    fig.tight_layout()
    return fig


def fig_step_response(
    model: JEPAControl,
    z0: np.ndarray,
    readout: dict,
    *,
    steps: int = 80,
    obs_name: str = "rabbit_count",
):
    """Step and impulse response of the learned system, in readout units.

    Plotted as the deviation from a zero-control rollout, so the curve is the
    control's own contribution. Read the sign (down is correct for culling), the
    settling time (which sets the MPC planning horizon), and whether different
    ``u`` scale proportionally.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, impulse, title in (
        (axes[0], False, "Step response (u held from t=0)"),
        (axes[1], True, "Impulse response (single-step u)"),
    ):
        res = ev.step_response(
            model, z0, readout, steps=steps, impulse=impulse, obs_name=obs_name
        )
        for i, u in enumerate(res["u_levels"]):
            ax.plot(res["steps"], res[f"resp_{u}"], color=f"C{i}", label=f"u={u}")
        ax.axhline(0, color="k", lw=1, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"change in predicted {obs_name} vs $u=0$")
    fig.suptitle("Learned actuator dynamics (deviation from the uncontrolled rollout)")
    fig.tight_layout()
    return fig


# ======================================================================
# 5. Closed-loop MPC
# ======================================================================
def fig_closed_loop(loop: dict, baselines: dict | None = None):
    """Did MPC actually steer the true ABM to the target?

    Top: the true macrostate against the target, with open-loop constant-cull
    references for context -- the controller has to beat the best fixed dose, not
    just move in the right direction. Bottom: the control it chose. Saturation at
    0 or 1 means the target is outside the achievable range; heavy chattering
    means the planning horizon or the effort penalty needs attention.
    """
    fig, (ax_y, ax_u) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    t = np.arange(len(loop["true"]))

    if baselines:
        for i, (label, series) in enumerate(baselines.items()):
            ax_y.plot(series[: len(t)], color=f"C{i + 2}", ls=":", lw=1.2, alpha=0.8, label=label)
    ax_y.plot(t, loop["true"], color="C0", lw=2, label="MPC (closed loop)")
    ax_y.axhline(loop["target"], color="k", ls="--", alpha=0.7, label="target")
    ax_y.set_ylabel(loop["obs_name"])
    ax_y.set_title(
        f"Closed-loop MPC vs the true ABM — tracking RMSE {loop['tracking_rmse']:.1f}, "
        f"final error {loop['final_error']:+.1f}"
    )
    ax_y.legend(fontsize=9)
    ax_y.grid(alpha=0.3)

    ax_u.step(t, loop["control"], where="post", color="C3")
    ax_u.set_ylim(-0.05, 1.05)
    ax_u.set_ylabel("cull $u$")
    ax_u.set_xlabel("step")
    ax_u.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def fig_controller_compare(
    loops: dict[str, dict],
    baselines: dict | None = None,
    *,
    title: str = "Closed-loop controllers vs the true ABM",
):
    """Overlay several closed-loop planners (e.g. JEPA vs resource ODE).

    ``loops`` maps a legend label to a :func:`closed_loop`-style dict. Constant-cull
    open-loop references can still be passed via ``baselines``.
    """
    fig, (ax_y, ax_u) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    first = next(iter(loops.values()))
    t = np.arange(len(first["true"]))
    target = first["target"]
    obs_name = first["obs_name"]

    if baselines:
        for i, (label, series) in enumerate(baselines.items()):
            ax_y.plot(series[: len(t)], color=f"C{i + 3}", ls=":", lw=1.2, alpha=0.75, label=label)

    for i, (label, loop) in enumerate(loops.items()):
        color = f"C{i}"
        ax_y.plot(
            loop["true"][: len(t)],
            color=color,
            lw=2,
            label=f"{label} (RMSE {loop['tracking_rmse']:.1f})",
        )
        ax_u.step(
            np.arange(len(loop["control"])),
            loop["control"],
            where="post",
            color=color,
            label=label,
        )

    ax_y.axhline(target, color="k", ls="--", alpha=0.7, label="target")
    ax_y.set_ylabel(obs_name)
    ax_y.set_title(title)
    ax_y.legend(fontsize=9)
    ax_y.grid(alpha=0.3)

    ax_u.set_ylim(-0.05, 1.05)
    ax_u.set_ylabel("cull $u$")
    ax_u.set_xlabel("step")
    ax_u.legend(fontsize=9)
    ax_u.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def fig_tumor_controller_compare(
    loops: dict[str, dict],
    baselines: dict | None = None,
    *,
    title: str = "JEPA vs tumor-ODE controllers (closed on true ABM)",
):
    """Overlay multi-objective tumor planners (e.g. JEPA vs fitted ODE).

    ``loops`` maps a legend label to a :func:`closed_loop_tumor`-style dict.
    Constant-dose open-loop references from :func:`tumor_baseline_rollouts` can
    still be passed via ``baselines``.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    first = next(iter(loops.values()))
    t_len = len(first["tumor"])

    if baselines:
        for label, result in baselines.items():
            axes[0].plot(result["tumor"][:t_len], ls="--", alpha=0.45, label=label)
            axes[1].plot(result["healthy"][:t_len], ls="--", alpha=0.45, label=label)

    for i, (label, loop) in enumerate(loops.items()):
        color = f"C{i}"
        axes[0].plot(
            loop["tumor"][:t_len],
            color=color,
            lw=2.2,
            label=f"{label} (tumor RMSE {loop['tumor_rmse']:.1f})",
        )
        axes[1].plot(
            loop["healthy"][:t_len],
            color=color,
            lw=2.2,
            label=f"{label} (healthy shortfall {loop['healthy_shortfall_rmse']:.1f})",
        )
        axes[2].step(
            np.arange(len(loop["control"])),
            loop["control"],
            where="post",
            color=color,
            label=f"{label} (mean u {loop['mean_dose']:.2f})",
        )

    axes[0].axhline(first["tumor_target"], color="k", ls=":", label="tumor target")
    axes[1].axhline(first["healthy_reference"], color="k", ls=":", label="healthy reference")
    axes[0].set_ylabel("tumor cells")
    axes[1].set_ylabel("healthy cells")
    axes[2].set_ylabel("dose $u$")
    axes[2].set_xlabel("closed-loop step")
    axes[2].set_ylim(-0.05, 1.05)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=3)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    return fig


def fig_sir_controller_compare(
    loops: dict[str, dict],
    baselines: dict | None = None,
    *,
    title: str = "JEPA vs SIR ODE controllers (closed on true ABM)",
):
    """Overlay multi-objective SIR planners (e.g. JEPA vs fitted ODE).

    ``loops`` maps a legend label to a :func:`closed_loop_sir`-style dict.
    Constant-vaccination open-loop references from :func:`sir_baseline_rollouts`
    can still be passed via ``baselines``.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    first = next(iter(loops.values()))
    t_len = len(first["infected"])

    if baselines:
        for label, result in baselines.items():
            axes[0].plot(result["infected"][:t_len], ls="--", alpha=0.45, label=label)
            axes[1].plot(result["susceptible"][:t_len], ls="--", alpha=0.45, label=label)

    for i, (label, loop) in enumerate(loops.items()):
        color = f"C{i}"
        axes[0].plot(
            loop["infected"][:t_len],
            color=color,
            lw=2.2,
            label=f"{label} (infected RMSE {loop['infected_rmse']:.1f})",
        )
        axes[1].plot(
            loop["susceptible"][:t_len],
            color=color,
            lw=2.2,
            label=f"{label} (shortfall RMSE {loop['susceptible_shortfall_rmse']:.1f})",
        )
        axes[2].plot(
            loop["recovered"][:t_len],
            color=color,
            lw=2.0,
            label=f"{label} (final incidence {loop['final_incidence']:.0f})",
        )
        axes[3].step(
            np.arange(len(loop["control"])),
            loop["control"],
            where="post",
            color=color,
            label=f"{label} (mean u {loop['mean_dose']:.2f})",
        )

    axes[0].axhline(first["infected_target"], color="k", ls=":", label="infected target")
    axes[1].axhline(first["susceptible_floor"], color="k", ls=":", label="susceptible floor")
    axes[0].set_ylabel("infected")
    axes[1].set_ylabel("susceptible")
    axes[2].set_ylabel("recovered")
    axes[3].set_ylabel("vaccination $u$")
    axes[3].set_xlabel("closed-loop step")
    axes[3].set_ylim(-0.05, 1.05)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=3)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    return fig
