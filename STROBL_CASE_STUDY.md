# Strobl spatial adaptive-therapy benchmark

> **Synthetic methodological benchmark only.** This simulator and its outputs
> are not clinically validated, are not treatment recommendations, and must not
> be used for patient care.

This case study wraps the spatial tumour model released with Strobl et al.
(2022), *Spatial structure impacts adaptive therapy by shaping intra-tumoral
competition*. The model is a stochastic 2-D lattice ABM containing empty sites,
drug-sensitive cells, and drug-resistant cells. The only control is one
spatially homogeneous scalar drug dose per day.

## Provenance and build

The permitted, pinned upstream snapshot and the unmodified release artifact are
under `vendor/strobl2021_space_modulates_competition_AT/`. See its
`UPSTREAM.md` for the repository URL, exact commit, citation, license status,
release-jar checksum, unmodified regression run, and local-change ledger.

Build the controlled runner:

```bash
bash vendor/strobl2021_space_modulates_competition_AT/build.sh
```

The build targets Java 8 bytecode. A Java 8-compatible runtime is required to
run the resulting jar; newer JDKs may be used to compile it.

Run one controlled episode with any named policy:

```bash
poetry run python -m koopman_control.data.strobl_simulator \
  --policy random_piecewise_constant --steps 100 --seed 7 \
  --output data/strobl/example_episode.npz
```

## Generate data

Start with the bounded smoke profile:

```bash
poetry run python -m koopman_control.data.generate_strobl \
  --profile configs/strobl_smoke.json
```

After smoke-test QC, generate the 120-standard plus 50-matched pilot:

```bash
poetry run python -m koopman_control.data.generate_strobl \
  --profile configs/strobl_pilot.json
```

After pilot QC passes, generate the full profile: 1,000 standard episodes, 250
matched-state episodes, and 30 paired fixed-dose evaluation episodes:

```bash
poetry run python -m koopman_control.data.generate_strobl \
  --profile configs/strobl_full.json
```

Use `--output-dir` to override the profile destination. The
`KOOPMAN_STROBL_DATA_ROOT` and `KOOPMAN_STROBL_DATASET` environment variables
provide persistent defaults.

## Train JEPA

The categorical loader reads HDF5 windows lazily, so training does not expand
the full dataset into memory. A practical local-MPS run that samples every
episode at several temporal offsets is:

```bash
poetry run python -m jepa_control.train \
  --dataset data/strobl/full/dataset.h5 \
  --run-name jepa-strobl-full-h20-mlp-valloss-e50 \
  --predictor residual_mlp \
  --predictor-hidden 128 --predictor-layers 2 \
  --horizon 20 --stride 100 --batch-size 32 --num-workers 4 \
  --latent-dim 16 --base-channels 16 \
  --w-pred 3.0 --w-vic-var 1.0 --w-vic-cov 0.1 --w-readout 0.1 \
  --target ema --ema-decay 0.996 \
  --lr 5e-4 --max-epochs 50 --early-stopping-monitor val_loss \
  --accelerator mps
```

The nonzero readout anchor helps preserve sensitive, resistant, total-burden,
occupancy, and cost information needed for downstream control. Reduce
`--stride` on faster hardware to use more overlapping windows. Monitoring total
validation loss avoids selecting an early low-diversity latent merely because
its near-collapsed predictive targets are easy to match.

## JEPA-versus-ODE MPC

`jepa_control.strobl_control` provides matched scalar-dose controllers:

- `cem_plan_strobl_jepa` plans through the learned latent dynamics and readout;
- `cem_plan_strobl_ode` plans through the fixed non-spatial paper equations;
- `compare_strobl_controllers` applies both to independently reset, identically
  seeded Java ABMs and also supports no-treatment, MTD, and paper-adaptive
  baselines.

Both MPCs use the same burden, dose-effort, dose-slew, terminal, horizon, and
dose-bound settings. At each replanning time, JEPA observes the exact grid while
the ODE is initialized from the exact aggregate sensitive/resistant counts.
Realized costs are always recomputed from the true ABM trajectory. The final
section of `notebooks/jepa_eval_strobl.ipynb` runs this comparison.

## Model and control contract

- The lattice has no-flux boundaries and von Neumann neighbourhoods.
- Site values are `0 = empty`, `1 = sensitive`, and `2 = resistant`.
- Cell movement is disabled.
- Cells divide only into an empty local neighbour.
- Drug is spatially homogeneous and directly kills only dividing sensitive
  cells.
- Every action is one scalar `u_t` in `[0, D_max]`; there is no coordinate,
  mask, partition, or per-cell action interface.
- One action drives `grid[t] -> grid[t+1]`. Stored actions therefore have
  length `T` while grids and counts have length `T+1`.

The policy set includes no treatment, constant dose, random piecewise-constant
dose, pulses, and adaptive therapy. The data-collection mix is 20%, 20%, 30%,
15%, and 15%, respectively. The released Java source restarts treatment only
when `N > N0`; the paper describes restarting at `N >= N0`. Both semantics are
named explicitly and never silently conflated.

## Initial spatial architectures

All constructors are deterministic for a seed and preserve requested aggregate
counts:

- `random_mixed`
- `resistant_core`
- `resistant_edge`
- `resistant_dispersed`
- `two_resistant_nests`

Matched-state groups reuse the occupied-site mask, `S(0)`, `R(0)`, parameters,
and planned dose schedule across all five architectures. Only resistant-cell
arrangement and stochastic replicate seed vary. This makes the set a direct
test of spatial information that an aggregate non-spatial model cannot see.

The full profile balances its standard held-out architectures exactly (75
`resistant_edge`, 75 `two_resistant_nests`). Its controlled-evaluation subset
uses five parameter/initial-state groups, two stochastic replicates, and
constant doses `u={0,0.5,1}`. Within each replicate, the initial grid,
parameters, and simulation seed are identical across doses.

## Parameters and termination

The canonical lattice is 100 by 100 with one-day simulation, control, and save
intervals. Defaults are centred on `r_S = 0.027/day`, `D_max = 1`, and
`d_D = 0.75`. Dataset profiles vary initial density over 0.25–0.75, resistant
fraction log-uniformly over 0.001–0.05, resistance cost over 0–0.3, turnover
relative to `r_S` over 0–0.3, and `r_S` over 0.02–0.04.

Internal upstream stopping is disabled. Dataset progression is recorded
externally when `N > 1.2 N0` after day 150. Cure and maximum time are separate
terminal reasons. The released source instead checks two consecutive
above-threshold daily observations and does not apply the day-150 exclusion;
that discrepancy is retained in provenance and regression tests.

## Dataset schema

The HDF5 file contains complete episode groups and transition indices:

- `grid[T+1,H,W]`: `uint8` categorical lattice
- `action[T]`: `float32` realized global dose
- `counts[T+1,3]`: sensitive, resistant, and total counts
- `occupancy[T+1]`
- `cost[T]`: tumour burden plus dose cost
- `ode_counts[T+1,3]`: matched non-spatial baseline
- spatial diagnostics and per-phenotype event diagnostics
- episode parameters and provenance attributes

Companion JSONL stores complete metadata. Transition-index groups for horizons
1, 5, 10, and 25 identify whole-episode windows, action windows, and target
observation indices. Splits never cross episode boundaries.

The ODE baseline uses the paper's fixed sensitive/resistant equations with the
same carrying capacity, aggregate initial counts, biological parameters, and
piecewise-constant action trajectory as the ABM. It is not fitted separately to
episodes, and therefore gives identical trajectories for matched cases with
identical counts, parameters, and actions.

## Quality control

Run focused tests:

```bash
poetry run pytest tests/test_strobl_benchmark.py -q
```

Run the reusable QC inspector:

```bash
poetry run python -m koopman_control.data.qc_strobl \
  --dataset "$KOOPMAN_STROBL_DATA_ROOT/pilot/dataset.h5"
```

The generated pilot contains 120 standard and 50 matched-state episodes. On
the development host it completed in 128.6 seconds. The HDF5 plus JSONL occupy
94.76 MB; 656.45 MB of raw grid arrays compress to 34.58 MB inside HDF5
(18.98:1 grid compression). External termination classified 110 episodes as
progression and 60 as maximum time. `qc_report.json` and `qc_examples.png`
record the schema, policy quotas, dose bounds, complete-episode split
isolation, transition horizons, matched occupied-mask/action/ODE identity,
ABM-vs-ODE aggregate differences, and representative grids/trajectories.
Open `notebooks/strobl_dataset_qc.ipynb` for the lightweight interactive view.
After training, open `notebooks/jepa_eval_strobl.ipynb` for training curves,
held-out architecture examples, latent/readout diagnostics, rollout skill,
paired-dose control beliefs, and ABM-versus-ODE comparisons.

## Citation

Strobl, M. A. R. et al. Spatial structure impacts adaptive therapy by shaping
intra-tumoral competition. *Communications Medicine* **2**, 46 (2022).
https://doi.org/10.1038/s43856-022-00110-x
