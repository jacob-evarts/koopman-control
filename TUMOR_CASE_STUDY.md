# Case A: spatial tumor–healthy-tissue control

This case study adds a deliberately simple, phenomenological tumor ABM for
testing image-based latent dynamics and multi-objective MPC. It is a control
benchmark, **not a calibrated biological model**.

## State and dynamics

Each frame has four spatial channels in `[0, 1]`:

1. `healthy`: healthy-cell occupancy
2. `tumor`: tumor-cell occupancy
3. `nutrient`: diffusing nutrient concentration
4. `drug`: diffusing chemotherapy concentration

The initial tumor is a localized disk embedded in a nearly confluent healthy
tissue sheet. Fixed vessel lines replenish nutrient and deliver systemic
chemotherapy. Tumor cells grow and invade neighboring healthy tissue; both cell
types can starve and both are killed by drug, with greater tumor sensitivity.

The scalar control `u in [0, 1]` is systemic chemotherapy dose. Drug transport
and decay make its effect delayed and spatially heterogeneous.

Main observables:

- `tumor_count`, `healthy_count`
- `tumor_frac`, `healthy_frac`, `healthy_loss`
- tumor centroid and spread
- mean nutrient/drug and population-specific drug exposure

## Generate the dataset

The default sweep contains 630 trajectories:

- 7 excitation schedules
- 3 initial tumor radii
- 2 healthy-cell densities
- 3 tumor locations
- 5 seeds

```bash
poetry run python -m koopman_control.data.generate_tumor \
  --output-dir data/tumor_tissue_images_v2 \
  --steps 200 --seeds 5
```

Frames are stored compactly as uint8. The HDF5 attribute `frame_scale=255`
instructs the JEPA loader to decode the continuous nutrient and drug channels
back to `[0, 1]`. Legacy binary datasets default to `frame_scale=1`.

The calibrated defaults make a 50%-of-initial tumor target reachable. For a
radius-6 tumor over 80 steps, averaged across five seeds, untreated burden grows
to about 126% of its initial value. Constant doses `u=(0.25, 0.50, 0.75, 1.0)`
end near `(80%, 58%, 46%, 38%)`, while the maximum dose preserves about 79% of
the initial healthy-cell count. This produces both a graded treatment signal
and a genuine tumor/toxicity trade-off. The corresponding generator options are
`--drug-diffusion`, `--drug-delivery`, `--drug-decay`,
`--healthy-drug-kill`, and `--tumor-drug-kill`.

## Train JEPA

```bash
poetry run python -m jepa_control.train \
  --dataset data/tumor_tissue_images_v2/dataset.h5 \
  --run-name jepa-tumor-v2-h20-mlp \
  --predictor residual_mlp \
  --predictor-hidden 256 --predictor-layers 2 \
  --horizon 20 --latent-dim 16 --base-channels 32 \
  --w-pred 3.0 --w-vic-var 1.0 --w-vic-cov 0.1 \
  --target ema --ema-decay 0.996 \
  --lr 5e-4 --max-epochs 50 --accelerator mps
```

The generic trainer reads channel count and observable names from the dataset,
so no tumor-specific model architecture is required.

## Multi-objective MPC

`jepa_control.control` provides:

- `TumorMPCConfig`
- `cem_plan_tumor`
- `closed_loop_tumor`
- `tumor_baseline_rollouts`

The planner minimizes, over the latent rollout,

```text
tumor_weight  * normalized_tumor_error²
+ healthy_weight * normalized_healthy_shortfall²
+ control_cost   * u²
+ slew_cost      * (u - u_previous)²
```

Only healthy-cell **shortfall** is penalized; preserving more tissue than the
reference is not a cost. As with the existing MPC, CEM optimizes a continuous
dose sequence, applies only the first dose, observes the true ABM again, and
replans.

Fit a post-hoc readout containing both `tumor_count` and `healthy_count` before
using tumor MPC. The essential go/no-go checks are held-out readout R² for both
populations, multi-step rollout skill, dose-response ordering, and the
tumor-control/healthy-loss Pareto curve.
