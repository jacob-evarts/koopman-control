# Koopman Control

Image-based latent dynamics and control for a stochastic rabbit-grass agent-based
model. The model encodes two-channel occupancy images, predicts the latent under a
continuous culling input, and exposes a linear `(A, B)` core for system
identification and control.

## Repository layout

```text
src/koopman_control/
├── data/
│   ├── rabbit_grass.py   # simulator and continuous actuator
│   ├── excitation.py     # identification control signals
│   ├── generate.py       # HDF5 trajectory generation
│   └── dataset.py        # multi-step training windows
├── models/
│   └── world_model.py    # encoder, latent dynamics, decoder, losses
├── analysis/
│   ├── dmdc.py           # pre-training identifiability check
│   ├── evaluate.py       # numeric model diagnostics
│   └── plots.py          # evaluation figures
└── train.py              # training entry point

notebooks/worldmodel_eval.ipynb
```

The old Hydra/Optuna experiment framework, alternative Koopman models, graph
pipeline, surrogate-control package, and exploratory notebooks have been removed.

## Install

```bash
poetry install
```

## Data and output locations

Local defaults are `<repo>/data` and `<repo>/outputs`. On an HPC, set the
locations once in your shell or scheduler script:

```bash
export KOOPMAN_DATA_ROOT="/scratch/$USER/koopman-control/data"
export KOOPMAN_OUTPUT_ROOT="/scratch/$USER/koopman-control/results"
```

An editable template is provided in `hpc.env.example`; it can be sourced from an
interactive shell or scheduler job.

The default dataset then becomes:

```text
$KOOPMAN_DATA_ROOT/rabbit_grass_images/dataset.h5
```

If the HDF5 file has a different name or layout, set its exact location:

```bash
export KOOPMAN_DATASET="/shared/project/datasets/rabbit_grass_v2.h5"
```

All generation, DMDc, training, search, and evaluation commands use these
variables automatically. Explicit `--dataset`, `--output-dir`, `--out-dir`, and
`--study-dir` arguments still take precedence.

## SLURM

Set up the environment once on a login or interactive node:

```bash
poetry config virtualenvs.in-project true
poetry install
cp hpc.env.example hpc.env
# Edit hpc.env for the cluster's shared and scratch filesystems.
```

Create a site-specific job script:

```bash
cp slurm/search.sbatch.example slurm/search.sbatch
# Edit account, partition, GPU request, memory, and wall time.
sbatch slurm/search.sbatch
```

Monitor it with:

```bash
squeue -u "$USER"
tail -f slurm-koopman-search-<job-id>.out
sacct -j <job-id> --format=JobID,State,Elapsed,MaxRSS,AllocTRES
```

The search database and every checkpoint are written under
`$KOOPMAN_OUTPUT_ROOT`. Set this to persistent project storage or persistent
scratch—not node-local `$TMPDIR`—so the study survives after the allocation
ends. Running the same script again resumes that SQLite study and adds another
`TRIALS` trials. Do not run multiple jobs concurrently against the same SQLite
file on a network filesystem; use separate study directories or a PostgreSQL
Optuna backend for parallel workers.

## Pipeline

Generate controlled image trajectories:

```bash
poetry run python -m koopman_control.data.generate \
  --steps 200 --seeds 10
```

Check whether the dataset identifies a linear system with control before training:

```bash
poetry run python -m koopman_control.analysis.dmdc
```

Train the latent world model:

```bash
poetry run python -m koopman_control.train \
  --dynamics-mode linear \
  --horizon 20 \
  --latent-dim 8 \
  --max-epochs 30
```

Run a resumable hyperparameter study:

```bash
poetry run python -m koopman_control.search \
  --study-dir "$KOOPMAN_OUTPUT_ROOT/search/linear-vs-bilinear-h20" \
  --trials 40 \
  --max-epochs 50 \
  --final-seeds 3 \
  --accelerator auto
```

Each invocation adds the requested number of trials to the existing SQLite
study. Weak trials are pruned after their validation-loss trajectory falls
behind comparable runs; each trial also has ordinary early stopping. Loss
weights and rollout horizon remain fixed so trial objectives are comparable.

The selected output contains:

```text
outputs/search/linear-vs-bilinear-h20/
├── study.db                 # resumable Optuna study
├── trials.csv               # flat audit table
├── trials/trial_*/          # configs, metrics, best + last checkpoints
├── final/seed_*/            # post-search repeated fits
├── best_model.ckpt          # validation-selected final checkpoint
├── best_config.json
├── best_model.json          # best epoch/step and test metrics
└── report/                  # optimization, importance, and learning-curve plots
```

The search never evaluates the test split while choosing hyperparameters. It is
used only after selection for the final repeated-seed fits.

Open `notebooks/worldmodel_eval.ipynb` and run it top to bottom to inspect
reconstruction, latent geometry, linear probes, multi-step prediction, actuator
response, and controllability.

See `LATENT_CONTROL.md` for the design rationale and interpretation of the
diagnostics.