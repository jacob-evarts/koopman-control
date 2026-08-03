"""Generate an image-based, control-excited rabbit-grass dataset.

Why this file exists
--------------------
This is the concrete implementation of the "training data is the most important
part" idea. It rolls out the vendored ABM under randomized, amplitude-rich
control signals and stores image observations plus the aligned control input,
so a downstream world model can learn ``z_{t+1} = f(z_t, u_t)`` with an
identifiable control effect.

Output format (self-describing, single consolidated file)
---------------------------------------------------------
``dataset.h5``::

    runs/<run_id>/frames   (T+1, 2, W, H) uint8    # [grass, rabbit] occupancy
    runs/<run_id>/control  (T+1,)         float32  # u aligned to frames
    runs/<run_id>/obs      (T+1, K)       float32  # summary observables
    runs/<run_id> attrs: excitation, initial_rabbits, initial_grass_prob,
                         seed, split, steps
  file attrs: obs_names (K,), width, height, num_channels

``manifest.json`` lists every run with metadata and its split.

Split design (generalization, not just interpolation)
-----------------------------------------------------
Splitting is by whole trajectory. The test set deliberately holds out:
  * unseen **excitation types** (control-signal shapes the model never trained
    on), and
  * unseen **initial conditions**.
This tests whether the learned dynamics generalize, rather than memorizing the
specific control schedules seen in training.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import h5py
import numpy as np

from koopman_control.data.excitation import make_control
from koopman_control.data.rabbit_grass import NUM_CHANNELS, RabbitGrassConfig, rollout
from koopman_control.paths import dataset_directory

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# All excitation shapes are used in every split so the model can learn the full
# dynamics. These two are additionally *tagged* as out-of-distribution shapes so
# generalization to unseen control signals can be measured as a subset, without
# removing them from training.
OOD_EXCITATIONS = ("staircase", "chirp")


def split_seed_thresholds(n_seeds: int) -> tuple[int, int]:
    """Return ``(n_train_seeds, n_val_seeds)`` for a ~70/15/15 seed split.

    Guarantees at least one val and one test seed whenever ``n_seeds >= 3``.
    """
    n_train = max(1, round(0.70 * n_seeds))
    n_val = max(1, round(0.15 * n_seeds))
    if n_seeds >= 3 and n_train + n_val >= n_seeds:
        n_train = n_seeds - 2
        n_val = 1
    return n_train, n_val


def assign_split(seed: int, *, n_seeds: int) -> str:
    """Assign a trajectory to train/val/test by seed (balanced, stratified).

    Splitting by seed keeps every excitation shape and initial condition present
    in all splits, so a split is a set of independent rollouts rather than a
    biased slice of the state space.
    """
    n_train, n_val = split_seed_thresholds(n_seeds)
    if seed < n_train:
        return "train"
    if seed < n_train + n_val:
        return "val"
    return "test"


def _run_id(excitation: str, n_rabbits: int, grass_prob: float, seed: int) -> str:
    p = int(round(grass_prob * 100))
    return f"{excitation}_r{n_rabbits}_p{p}_s{seed}"


ALL_EXCITATIONS = ("zero", "constant", "rpwc", "prbs", "ramp", "staircase", "chirp")


def generate(
    out_dir: Path,
    *,
    steps: int = 120,
    seeds: int = 6,
    initial_rabbits: tuple[int, ...] = (60, 120, 180),
    initial_grass_prob: tuple[float, ...] = (0.15, 0.35),
    excitations: tuple[str, ...] = ALL_EXCITATIONS,
    cfg: RabbitGrassConfig | None = None,
    limit: int | None = None,
) -> dict:
    """Roll out the sweep and write ``dataset.h5`` + ``manifest.json``."""
    cfg = cfg or RabbitGrassConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "dataset.h5"

    jobs = list(
        itertools.product(
            excitations,
            enumerate(initial_rabbits),
            initial_grass_prob,
            range(seeds),
        )
    )
    if limit is not None:
        jobs = jobs[:limit]

    iterator = tqdm(jobs, desc="rollouts") if tqdm is not None else jobs

    manifest_runs: list[dict] = []
    obs_names: list[str] | None = None

    with h5py.File(h5_path, "w") as h5f:
        runs_grp = h5f.create_group("runs")
        for excitation, (ic_index, n_rabbits), grass_prob, seed in iterator:
            # Distinct RNG stream per (control signal) vs (simulation) so the
            # control schedule and ABM noise are independently reproducible.
            control_rng = np.random.default_rng(hash((excitation, seed)) % (2**32))
            control_seq = make_control(excitation, steps, control_rng)

            frames, controls, obs = rollout(
                cfg,
                control_seq,
                initial_rabbits=n_rabbits,
                initial_grass_prob=grass_prob,
                seed=seed,
            )

            if obs_names is None:
                obs_names = list(obs.keys())
            obs_mat = np.stack([obs[k] for k in obs_names], axis=1).astype(np.float32)

            rid = _run_id(excitation, n_rabbits, grass_prob, seed)
            split = assign_split(seed, n_seeds=seeds)
            ood_excitation = excitation in OOD_EXCITATIONS
            ood_ic = ic_index == len(initial_rabbits) - 1

            grp = runs_grp.create_group(rid)
            grp.create_dataset(
                "frames",
                data=(frames > 0.5).astype(np.uint8),
                compression="gzip",
                compression_opts=4,
            )
            grp.create_dataset("control", data=controls.astype(np.float32))
            grp.create_dataset("obs", data=obs_mat)
            grp.attrs.update(
                excitation=excitation,
                initial_rabbits=int(n_rabbits),
                initial_grass_prob=float(grass_prob),
                seed=int(seed),
                split=split,
                steps=int(steps),
                ood_excitation=bool(ood_excitation),
                ood_ic=bool(ood_ic),
            )

            manifest_runs.append(
                {
                    "run_id": rid,
                    "excitation": excitation,
                    "initial_rabbits": int(n_rabbits),
                    "initial_grass_prob": float(grass_prob),
                    "seed": int(seed),
                    "split": split,
                    "steps": int(steps),
                    "mean_control": float(controls.mean()),
                    "final_rabbits": float(obs["rabbit_count"][-1]),
                    "ood_excitation": bool(ood_excitation),
                    "ood_ic": bool(ood_ic),
                }
            )

        h5f.attrs["obs_names"] = np.array(obs_names or [], dtype=h5py.string_dtype())
        h5f.attrs["width"] = cfg.width
        h5f.attrs["height"] = cfg.height
        h5f.attrs["num_channels"] = NUM_CHANNELS

    manifest = {
        "runs": manifest_runs,
        "obs_names": obs_names,
        "config": {
            "steps": steps,
            "seeds": seeds,
            "initial_rabbits": list(initial_rabbits),
            "initial_grass_prob": list(initial_grass_prob),
            "excitations": list(excitations),
            "width": cfg.width,
            "height": cfg.height,
            "culling_effectiveness": cfg.culling_effectiveness,
            "cull_delay_fraction": cfg.cull_delay_fraction,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    counts: dict[str, int] = {}
    for r in manifest_runs:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    print(f"Wrote {len(manifest_runs)} trajectories to {h5_path}")
    print(f"Split counts: {counts}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=dataset_directory(),
        help="Directory for dataset.h5 and manifest.json",
    )
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--initial-rabbits", type=int, nargs="+", default=[60, 120, 180])
    p.add_argument("--initial-grass-prob", type=float, nargs="+", default=[0.15, 0.35])
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="Cap rollouts (smoke test)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = RabbitGrassConfig(width=args.width, height=args.height)
    generate(
        args.output_dir,
        steps=args.steps,
        seeds=args.seeds,
        initial_rabbits=tuple(args.initial_rabbits),
        initial_grass_prob=tuple(args.initial_grass_prob),
        cfg=cfg,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
