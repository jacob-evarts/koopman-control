# Case B: spatial agentic SIR control

This case study adds a lattice SIR agent-based model for testing image-based
latent dynamics and multi-objective vaccination MPC. It is a control benchmark
inspired by classic spatial epidemic ABMs (neighborhood / contact-radius
models), **not a calibrated public-health model**.

## State and dynamics

Each frame has three binary occupancy channels:

1. `susceptible`
2. `infected`
3. `recovered`

A fixed population of mobile agents lives on a periodic grid. Each step:

1. agents attempt a random Moore-neighborhood move,
2. susceptibles adjacent to (or co-located with) infecteds become infected with
   probability `infection_prob`,
3. infecteds recover with probability `recovery_prob`,
4. vaccination converts susceptibles to recovered with intensity `u`.

The scalar control `u in [0, 1]` is systemic vaccination intensity, with a
one-step campaign-lag term so the actuator is not instantaneous.

Initial outbreaks are localized disks of infecteds, so geometry matters: the
epidemic spreads as a spatial wave rather than a well-mixed ODE.

Main observables:

- `susceptible_count`, `infected_count`, `recovered_count`
- population fractions
- `cumulative_incidence`
- infected centroid and spread

## Generate the dataset

The default sweep contains 630 trajectories:

- 7 excitation schedules
- 2 population sizes (800 and 1200 agents on a 64×64 grid)
- 3 initial outbreak sizes (20, 32, 48 infecteds)
- 3 outbreak locations
- 5 seeds

Larger populations slow spatial mixing relative to outbreak size, which widens
the dynamic range of peak infection and produces a more graded vaccination
dose response than the original 400–550-agent sweep.

```bash
poetry run python -m koopman_control.data.generate_sir \
  --output-dir data/agentic_sir_images_v2 \
  --steps 200 --seeds 5
```

Frames are stored as uint8 occupancy with `frame_scale=1`.

With the v2 defaults (`n_agents=(800, 1200)`, `initial_infected=(20, 32, 48)`,
`vaccine_effectiveness=0.028`), untreated outbreaks reach high peak infection,
while constant vaccination `u=(0.25, 0.50, 0.75, 1.0)` reduces peak burden
and cumulative incidence in a graded way. Tune with `--infection-prob`,
`--recovery-prob`, `--vaccine-effectiveness`, and `--n-agents`.

The legacy v1 dataset remains at `data/agentic_sir_images/` (400–550 agents).

## Train JEPA

```bash
poetry run python -m jepa_control.train \
  --dataset data/agentic_sir_images_v2/dataset.h5 \
  --run-name jepa-sir-v2-h10-linear \
  --predictor linear \
  --horizon 10 \
  --latent-dim 16 \
  --base-channels 32 \
  --w-pred 3.0 \
  --w-vic-var 2.0 \
  --w-vic-cov 0.1 \
  --target ema \
  --ema-decay 0.999 \
  --lr 3e-4 \
  --max-epochs 40 \
  --accelerator mps
```

The generic trainer reads channel count and observable names from the dataset.

## Multi-objective MPC

`jepa_control.control` provides:

- `SIRMPCConfig`
- `cem_plan_sir`
- `closed_loop_sir`
- `sir_baseline_rollouts`

The planner minimizes, over the latent rollout,

```text
infected_weight   * normalized_infected_error²
+ susceptible_weight * normalized_susceptible_shortfall²
+ control_cost       * u²
+ slew_cost          * (u - u_previous)²
```

Susceptible shortfall only penalizes driving the remaining susceptible pool
below a floor (default 35% of the initial susceptible count). That creates the
trade-off between containing the outbreak and avoiding needless mass
vaccination.

Fit a post-hoc readout containing both `infected_count` and
`susceptible_count` before using SIR MPC. The essential checks are held-out
readout R² for both populations, multi-step rollout skill, vaccination
dose-response ordering, and the infection / cumulative-dose Pareto curve
against constant-`u` baselines.
