# Image-based latent control (from-scratch pipeline)

This branch (`feature/image-latent-control`) rebuilds the project around a single
research goal:

> Control a biological system from **image observations** by encoding each image
> into a latent state, evolving that latent forward under a **control input**, and
> planning the control with standard tools.

It follows the approved plan `image latent control`. This document explains what
has been built so far and, importantly, *why* each piece exists.

## Motivation / diagnosis

The prior code had two issues that blocked the goal:

1. **The latent was not a controllable state.** In the GNN branch, `decode()`
   discarded the latent and re-encoded the true graph, so there was no
   `latent -> state` map to roll out or control. Mean-pooling to a small vector
   was also non-injective.
2. **The training data did not excite the control input.** The actuator was a
   binary cull driven by a few fixed policies. A binary input can only reveal the
   control effect at `u in {0, 1}` and can never test whether the response scales
   with amplitude — so the control matrix `B` was effectively unidentifiable.

Locked design decisions (from planning): **image** observations, a **hybrid
locally-linear** latent (linear `z' = Az + Bu` baseline with an optional
nonlinear/bilinear variant so linear vs nonlinear control can be compared), and
the **rabbit-grass** ABM as the first system.

## What is built so far

### Filesystem configuration

All pipeline entry points use `src/koopman_control/paths.py`. Local defaults are
the repository's `data/` and `outputs/` directories. On a cluster, point them at
scratch or project storage once:

```bash
export KOOPMAN_DATA_ROOT="/scratch/$USER/koopman-control/data"
export KOOPMAN_OUTPUT_ROOT="/scratch/$USER/koopman-control/results"
# Optional exact HDF5 override:
export KOOPMAN_DATASET="/shared/datasets/rabbit_grass_images/dataset.h5"
```

Explicit CLI path arguments take precedence, so individual jobs can still
override these defaults.

### 1. A vendored, self-contained simulator with a continuous actuator
`src/koopman_control/data/rabbit_grass.py`

The external `control_abms` package is installed compiled-only (`.pyc`, no
source) and is not importable, so it is unsuitable as a reproducible dependency.
A small, readable rabbit-grass ABM is vendored here instead. The key change is a
**continuous cull actuator** `u in [0, 1]` applied as a per-rabbit removal
probability, with a deliberate **one-step actuation lag**. Continuous control
turns the input from a switch into a knob so its effect can be identified across
amplitudes; the lag forces downstream models to condition on control *history*.

Observation is a 2-channel image `[grass, rabbit]` occupancy map. Aggregate
`observables()` (counts, centroid, spread) are also exposed for the Phase-0 check.

### 2. Control-excitation signal generators
`src/koopman_control/data/excitation.py`

Randomized, amplitude-rich control sequences designed for system identification:
`rpwc` (random piecewise-constant), `prbs`, `staircase` (tests amplitude
linearity), `ramp`, `chirp` (tests frequency response), plus `zero`/`constant`
baselines. Good excitation is what makes `A` and `B` identifiable.

### 3. Image dataset generator
`src/koopman_control/data/generate.py`

Rolls out the ABM over a sweep of initial conditions, seeds, and excitation types
and writes a self-describing `dataset.h5` (uint8 image frames + aligned
continuous control + summary observables) plus `manifest.json`. Splitting is
**seed-stratified** (~70/15/15 by trajectory) so every excitation shape and
initial condition appears in all splits -- a split is a set of independent
rollouts rather than a biased slice of the state space. Out-of-distribution
control shapes (`chirp`, `staircase`) and the extreme initial condition are
additionally **tagged** (`ood_excitation`, `ood_ic`) so generalization can be
measured as a subset without starving training.

Run it:

```bash
poetry run python -m koopman_control.data.generate \
    --output-dir data/rabbit_grass_images --steps 200 --seeds 10
```

### 4. Phase-0 DMDc identifiability check
`src/koopman_control/analysis/dmdc.py`

Before training any deep encoder, fit `z_{t+1} = A z_t + B [u_t, u_{t-1}] + c`
directly on the low-dimensional observables. This separates "is the data
informative?" from "is the deep model good?" It reports one-step R^2, honest
free-running multi-step rollout RMSE, controllability rank, and the spectral
radius of `A`.

Run it:

```bash
poetry run python -m koopman_control.analysis.dmdc \
    --dataset data/rabbit_grass_images/dataset.h5 --control-lags 2
```

### 5. Image dataset loader
`src/koopman_control/data/dataset.py`

Serves fixed-length rollout **windows** of consecutive frames plus the controls
that drive each transition. Multi-step windows (not single pairs) are the point:
one-step training lets a model look accurate while drifting over a horizon.
Each transition carries the control history `[u_now, u_prev]` to account for the
actuator's one-step lag.

### 6. Hybrid locally-linear world model
`src/koopman_control/models/world_model.py`

CNN encoder -> latent `z` -> dynamics -> CNN decoder. The dynamics are a
**linear core `A z + B u` plus an optional nonlinear residual** (`linear`,
`bilinear`, or `mlp`). The same forward pass computes both the linear-only and
the full prediction, so a `linearity_gap` metric is logged every step: a small
gap means linear control tools are justified in this latent.

Training objective:
  * multi-step latent-prediction loss vs **stop-grad** targets (JEPA-style),
  * a low-weight image decode that keeps the latent grounded/renderable, and
  * VICReg variance/covariance regularization to prevent collapse.

Diagnostics logged during training: spectral radius of `A`, controllability rank
of `(A, b)`, and the linearity gap. `linear_system()` exposes `(A, B)` for the
control phase.

Train it:

```bash
poetry run python -m koopman_control.train \
    --dataset data/rabbit_grass_images/dataset.h5 \
    --dynamics-mode linear --horizon 20 --latent-dim 8 --max-epochs 30
```

### Removed: legacy experiment pipelines
The old Hydra/Optuna entry point, alternative CNN/MLP/GNN models, graph and
legacy loaders, surrogate-control package, old configuration tree, and
exploratory notebooks were deleted. The package now contains only `data/`,
`models/`, `analysis/`, and the training entry point for this image pipeline.

## Phase-0 results (on the generated 64x64 dataset)

| metric | value | reading |
| --- | --- | --- |
| one-step R^2 (train / val / test) | 0.970 / 0.962 / 0.961 | linear-with-control fits well and generalizes to unseen control shapes and ICs |
| controllability rank | 5 / 6 | the single scalar cull input controls 5 of 6 aggregate directions |
| spectral radius of A | 0.993 | slow, near-marginally-stable population dynamics |
| free-running rollout RMSE (rabbit_count, test) | ~66 over 120 steps | one-step-linear is a good local model but accumulates error long-horizon |

Takeaway: the redesigned data **is** informative and the actuator **is**
identifiable, which was the core worry. The large multi-step rollout error is
expected on a stochastic, nonlinear system and is the quantitative motivation for
the hybrid (locally-linear / nonlinear) latent model and multi-step training
losses that come next.

## World-model results (3-epoch capped smoke run, `bilinear`)

| metric | trend | reading |
| --- | --- | --- |
| train loss | 1.03 -> 0.68 | learning |
| latent prediction loss | 0.22 -> ~0.10 | multi-step latent dynamics improving |
| linearity gap (linear - full) | 0.078 -> 0.048 | nonlinear residual helps less over time; latent is becoming approximately linear |
| VICReg variance term | 0.51 -> 0.27 | no collapse (embedding variance growing toward the target) |
| spectral radius of A | ~0.99 | slow, stable dynamics (matches Phase-0 DMDc) |
| controllability rank | ~20 / 32 | the single cull actuator controls ~20 latent directions |

This is a capped sanity run, not a tuned model, but it confirms the pipeline
trains, does not collapse, and produces the linearity/controllability signals
needed to decide between linear and nonlinear control.

### 7. Evaluation suite
`src/koopman_control/analysis/evaluate.py` (numerics),
`analysis/plots.py` (figures), and `notebooks/worldmodel_eval.ipynb`
(display layer).

Because the ABM is stochastic at the pixel level, none of these analyses score
exact next-frame reconstruction. The suite is built around four questions.

**Did it learn dynamics?** `horizon_errors` reports free-running latent error vs
horizon against three references, which is what makes an otherwise unitless MSE
readable:

| reference | purpose |
| --- | --- |
| `persistence` (freeze `z_0`) | floor; losing to it means no dynamics were learned |
| `linear` | the trained model's `A z + B u` with the residual switched off |
| `ls_linear` (`fit_latent_linear`) | least-squares-optimal linear operator in the *same* latent -- the ceiling for any linear model here |

The `full` vs `ls_linear` comparison is the one that matters, because it separates
"this latent is genuinely nonlinear" from "the trained linear core is
undertrained" -- a distinction the training-time `linearity_gap` cannot make,
since that metric compares against a core that was trained *with* the residual
absorbing part of the behaviour.

**Is the latent a usable state?** `latent_pca` gives the geometry plus a
participation ratio (a smooth count of how many dimensions are genuinely used,
which detects partial collapse that VICReg's variance term does not catch).
`linear_probe` fits ridge readouts from `z` to each ground-truth observable and
reports held-out R^2; a high `rabbit_count` R^2 means the macrostate to be
regulated is a *linear function* of the latent, so a controller can target it
without the decoder in the loop.

**Does it understand the control?** `dose_response` sweeps constant cull levels
through both the true simulator and the model and scores monotonicity by rank
correlation -- magnitudes can be miscalibrated and a controller still works, but
a scrambled ordering makes it push the wrong way. `step_response` gives the
classic step/impulse experiment as a deviation from the uncontrolled rollout.
`control_effect_map` decodes the latent before and after a push along `B u`, so
the learned actuator is visible in pixel space.

**Is `(A, B)` fit for a controller?** `mode_analysis` annotates each eigenvalue
with its half-life and its modal (PBH) controllability, which surfaces the
dangerous case of a slow, long-lived mode that the actuator cannot excite.
`controllability_spectrum` returns the singular values of `[b, Ab, ...]` --  the
continuous version of the integer rank logged during training, which counts
directions reachable only at `1e-8` gain.

`scorecard` / `format_scorecard` condense all of the above into a pass/warn/fail
table so the notebook opens with a verdict.

### 8. Rigorous training and hyperparameter search
`src/koopman_control/train.py` contains the reusable single-run trainer and
`src/koopman_control/search.py` wraps it in a persistent Optuna study.

The search varies architecture and optimization parameters (`dynamics_mode`,
latent width, CNN width, activation, batch size, learning rate, and weight
decay). Rollout horizon and loss weights are intentionally fixed within a study:
changing either would make validation losses across trials answer different
questions and therefore become invalid to rank.

Every run saves its full configuration, epoch metrics, the last checkpoint, and
the exact **best validation step**. There are two independent ways to stop
wasted computation:

1. within-run early stopping when validation loss stops improving, and
2. cross-trial median pruning when a run's validation trajectory is
   statistically uncompetitive with earlier trials.

The SQLite study is resumable. The test split is not touched during search.
After selecting the best validation configuration, it is retrained from several
random seeds and only those final models are evaluated on test.

```bash
poetry run python -m koopman_control.search \
    --dataset data/rabbit_grass_images/dataset.h5 \
    --study-dir outputs/search/linear-vs-bilinear-h20 \
    --dynamics-modes linear bilinear \
    --horizon 20 --trials 40 --max-epochs 50 --final-seeds 3
```

`src/koopman_control/analysis/training.py` automatically produces an audit
table, optimization history, trial outcomes, parameter importance, marginal
parameter effects, top-trial learning curves, and a Markdown report with
multi-seed test mean, standard deviation, and 95% confidence interval.

## Next steps (not yet built)

- Control + evaluation: LQR / linear-MPC on `(A, B)` from `linear_system()` vs
  CEM / MPPI on the full nonlinear predictor, evaluated closed-loop against the
  true ABM (e.g. drive the rabbit population to a target).
