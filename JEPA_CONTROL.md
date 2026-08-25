# JEPA + VICReg latent control (decoder-free, MPC)

Branch: `feature/jepa-vicreg-control`. This is a research sibling of the
`koopman_control` pipeline (documented in [LATENT_CONTROL.md](LATENT_CONTROL.md))
that tests a different modeling philosophy. The `koopman_control` package is left
untouched for comparison; all new code lives in `src/jepa_control/`.

## Motivation

The `koopman_control` world model is JEPA-flavored but keeps a **decoder** and a
reconstruction loss. The decoder does double duty: it is an anti-collapse crutch
and it is how every macrostate is read back out of the latent. This branch asks a
sharper question:

> Can a **pure JEPA** encoder -- no decoder, no reconstruction -- learn a latent
> of the ABM that is (a) kept from collapsing by **VICReg alone**, (b)
> interpretable via a **post-hoc linear readout** to physical macrostates, and
> (c) **controllable with MPC** over a learned latent dynamics model?

Empirical finding from the first (nonlinear, `d=32`) run: the discovered latent
was almost exactly linear in time (post-hoc LS one-step R² ≈ 0.97), and a plain
least-squares linear operator *outperformed* the trained residual MLP on 16-step
rollouts. The default is therefore a **linear** predictor
`z' = A z + B u + c` in a **16-d** latent. The nonlinear residual MLP remains
available as `--predictor residual_mlp` for ablation.

**Target encoder (this branch):** predictive targets default to a second encoder
whose weights are an **EMA** of the online encoder (`--target ema`,
`--ema-decay 0.996`), V-JEPA-style. Use `--target stopgrad` for the original
single shared-encoder recipe.

## What is built

### `src/jepa_control/model.py` -- `JEPAControl`

```
frames (B, C, W, H)
        │
        ▼
   ConvEncoder          4× stride-2 conv → flatten → MLP → z ∈ R^{16}
        │
        ▼
 LinearPredictor        z_{t+1} = A z_t + B u_hist + c
                        u_hist = [u_now, u_prev]   (actuator lag)
```

Losses (pure JEPA by default):

- `pred`: multi-step latent prediction vs **stop-grad** target embeddings
  (shared-encoder JEPA), rolled out over the horizon.
- `vic`: VICReg variance hinge + off-diagonal covariance, applied over **all**
  frames in each window (not just the first). Default `w_vic_cov=0.1`.
- `readout` (optional, `w_readout = 0` by default): a light, batch-standardized
  MSE from `z` to the ground-truth observables. Off by default so the encoder is
  purely self-supervised.

`participation_ratio` and `spectral_radius(|eig(A)|)` are logged each validation
epoch. `dynamics_matrices()` exposes the trained `(A, B, c)` directly;
`linear_diagnostic` is the post-hoc LS reference operator in the same latent.

### `src/jepa_control/data.py`
`JEPAWindows` serves fixed-length `(frames, controls, obs)` windows from one split
of the shared `dataset.h5`. Observables are always included (they are tiny) so the
anchored variant is a one-flag change with no separate loader. Control alignment
matches the generator's `[u_now, u_prev]` convention.

### `src/jepa_control/train.py`
Single-run trainer + CLI with the same reproducibility guarantees as the sibling
(`config.json`, `provenance.json` with git commit + dataset hash, `logs/metrics.csv`,
best/last checkpoints, `result.json`). Reuses `koopman_control`'s `paths` and the
shared dataset; runs land under `outputs/jepa_training/`.

```bash
poetry run python -m jepa_control.train \
    --dataset data/rabbit_grass_images/dataset.h5 \
    --run-name jepa-h16-d16-linear \
    --latent-dim 16 --horizon 16 --max-epochs 50
# nonlinear ablation: add --predictor residual_mlp
# anchored ablation:  add --w-readout 1.0
```

### `src/jepa_control/evaluate.py`
Decoder-free evaluation:

- `fit_readout` / `readout_predict` / `readout_r2`: post-hoc linear map
  `z -> macrostates`, the decoder's replacement. Fits on standardized `z` and
  folds the transform back so MPC still scores `C z + b`.
- `horizon_errors`: free-running latent error vs horizon against a `persistence`
  floor and the least-squares-optimal linear operator in the same latent
  (`ls_linear`). With the default linear predictor, `full` should match
  `ls_linear`; a gap means under-training.
- `latent_pca`: geometry + participation ratio.
- `linear_probe`: held-out R^2 of `rabbit_count` (macrostate decodability) and of
  the applied `u` (control legibility -- a subtle collapse mode for control).
- `readout_rollout_skill`: predicted vs true macrostate over free rollouts.
- `fit_latent_linear`: the LS reference operator (one-step R^2, spectral radius).

Probe / readout subsets are stratified by excitation regime (file-order truncation
used to silently return a single regime and destroy held-out R²).

### `src/jepa_control/control.py`
Sampling-based MPC over the learned predictor:

- `cem_plan`: cross-entropy-method planner that rolls candidate control sequences
  through the predictor in latent space and scores them by a readout-space cost
  `sum (C z_t - y_target)^2 + lambda u^2`.
- `closed_loop`: receding-horizon control against the true ABM (encode current
  frame -> plan -> step the ABM -> repeat), reporting tracking RMSE to a target
  population.
- `baseline_rollouts`: open-loop constant-cull references for comparison.

### `src/jepa_control/plots.py` + `notebooks/jepa_eval.ipynb`
`plots.py` holds the `fig_*` helpers (numeric logic stays in `evaluate.py`) and
`jepa_eval.ipynb` is a thin display layer over them. There are no pixel panels,
since there is no decoder; `fig_control_coverage` is re-exported from the sibling
package because excitation coverage is a property of the dataset, not the model.

The notebook's flow: one expensive cell computes everything and prints a
pass/warn/fail scorecard, then each section renders instantly.

| section | question |
| --- | --- |
| 3 | did VICReg hold (variance term, participation ratio)? |
| 4 | does it beat persistence, and does the trained predictor match `ls_linear`? |
| 5 | is the latent structured, and is `u` still legible in it? |
| 6 | can the macrostate be linearly read out, and does it survive a free rollout? |
| 7 | is the dose-response monotonic? |
| 8 | does closed-loop MPC beat the constant-cull baselines on the true ABM? |
| 9 | held-out test split, touched once |

Point `RUN_DIR` at a training run and run top to bottom. The readout is fit on
train encodings and scored on val/test, so every R² quoted is held-out.

```bash
poetry run jupyter lab notebooks/jepa_eval.ipynb
```

## Probing plan (for publishable rigor)

- **No collapse:** VICReg variance/covariance terms, participation ratio, and the
  ablation `w_vic_* = 0` (which should visibly collapse).
- **Interpretable latent:** held-out `rabbit_count` readout R^2 and control
  legibility R^2.
- **Real dynamics:** `full` vs `persistence` vs `ls_linear` skill; with the
  linear predictor, `full ≈ ls_linear` is the training-quality check.
- **Control works:** closed-loop tracking vs baselines, across seeds and targets.

Planned ablations: VICReg on/off, stop-grad vs EMA target, pure vs anchored
(`w_readout`), `linear` vs `residual_mlp` predictor, latent width / horizon.

## Phase 2 (not yet built): spatial / localized control

Partition the arena into a coarse `KxK` region grid; control becomes
`u in [0,1]^{K^2}` (per-region cull). To keep `koopman_control` untouched, the
spatial actuator will be a fork inside this package (`abm_spatial.py`,
`generate_spatial.py`), with per-region readout probes, a localized step-response
matrix (region-i excitation vs region-i/j response), and a per-region MPC
objective. If regional counts are not decodable from a vector latent, the encoder
escalates to a spatial-token latent.

## Environment note

Registering the new package required `poetry install`, which synced the
(already-modified) lockfile and bumped `torch` to 2.12.1. That left the
pre-installed `torchvision` (0.24.1, built for torch 2.9) ABI-incompatible and
breaking `pytorch_lightning`'s import chain. Fixed by installing the matching
`torchvision==0.27.1`. This is **not** captured in `poetry.lock` (torchvision is
not a declared dependency), so a future `poetry install` may revert it; add an
explicit compatible `torchvision` pin if this recurs.
