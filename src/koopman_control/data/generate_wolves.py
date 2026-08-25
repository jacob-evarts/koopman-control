"""Generate a control-excited grass–rabbit–wolf image dataset.

Control ``u`` culls **wolves** (continuous, lagged). Frames are 3-channel
occupancy ``[grass, rabbit, wolf]``. Observables include both ``rabbit_count``
and ``wolf_count`` so either can be a downstream MPC target.

Output layout matches :mod:`koopman_control.data.generate` (``dataset.h5`` +
``manifest.json``) so existing JEPA / Koopman loaders work unchanged — they read
``num_channels`` and ``obs_names`` from file attrs.

Default output directory: ``data/wolf_rabbit_grass_images_cull03/``
(culling effectiveness ``0.03``; see :func:`koopman_control.paths.wolf_dataset_directory`).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import h5py
import numpy as np

from koopman_control.data.excitation import make_control
from koopman_control.data.generate import ALL_EXCITATIONS, OOD_EXCITATIONS, assign_split
from koopman_control.data.wolf_rabbit_grass import NUM_CHANNELS, WolfRabbitGrassConfig, rollout
from koopman_control.paths import wolf_dataset_directory

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _run_id(
    excitation: str, n_rabbits: int, n_wolves: int, grass_prob: float, seed: int
) -> str:
    p = int(round(grass_prob * 100))
    return f"{excitation}_r{n_rabbits}_w{n_wolves}_p{p}_s{seed}"


def generate(
    out_dir: Path,
    *,
    steps: int = 200,
    seeds: int = 8,
    seed_offset: int = 0,
    initial_rabbits: tuple[int, ...] = (80, 140),
    initial_wolves: tuple[int, ...] = (10, 20),
    initial_grass_prob: tuple[float, ...] = (0.20, 0.35),
    excitations: tuple[str, ...] = ALL_EXCITATIONS,
    cfg: WolfRabbitGrassConfig | None = None,
    limit: int | None = None,
) -> dict:
    """Roll out the sweep and write ``dataset.h5`` + ``manifest.json``.

    ``seed_offset`` shifts the ABM / control RNG seeds to ``offset + i`` while
    train/val/test assignment still uses the index ``i`` in ``0 .. seeds-1``,
    so regenerating with a new offset keeps the same split structure.
    """
    cfg = cfg or WolfRabbitGrassConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "dataset.h5"

    jobs = list(
        itertools.product(
            excitations,
            enumerate(initial_rabbits),
            initial_wolves,
            initial_grass_prob,
            range(seeds),
        )
    )
    if limit is not None:
        jobs = jobs[:limit]

    iterator = tqdm(jobs, desc="wolf rollouts") if tqdm is not None else jobs

    manifest_runs: list[dict] = []
    obs_names: list[str] | None = None

    with h5py.File(h5_path, "w") as h5f:
        runs_grp = h5f.create_group("runs")
        for excitation, (ic_index, n_rabbits), n_wolves, grass_prob, seed_index in iterator:
            seed = int(seed_offset + seed_index)
            control_rng = np.random.default_rng(
                hash((excitation, "wolf", seed, n_wolves)) % (2**32)
            )
            control_seq = make_control(excitation, steps, control_rng)

            frames, controls, obs = rollout(
                cfg,
                control_seq,
                initial_rabbits=n_rabbits,
                initial_wolves=n_wolves,
                initial_grass_prob=grass_prob,
                seed=seed,
            )

            if obs_names is None:
                obs_names = list(obs.keys())
            obs_mat = np.stack([obs[k] for k in obs_names], axis=1).astype(np.float32)

            rid = _run_id(excitation, n_rabbits, n_wolves, grass_prob, seed)
            # Split by index so a nonzero seed_offset does not dump everything into test.
            split = assign_split(seed_index, n_seeds=seeds)
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
                initial_wolves=int(n_wolves),
                initial_grass_prob=float(grass_prob),
                seed=int(seed),
                split=split,
                steps=int(steps),
                ood_excitation=bool(ood_excitation),
                ood_ic=bool(ood_ic),
                control_target="wolves",
            )

            manifest_runs.append(
                {
                    "run_id": rid,
                    "excitation": excitation,
                    "initial_rabbits": int(n_rabbits),
                    "initial_wolves": int(n_wolves),
                    "initial_grass_prob": float(grass_prob),
                    "seed": int(seed),
                    "split": split,
                    "steps": int(steps),
                    "mean_control": float(controls.mean()),
                    "final_rabbits": float(obs["rabbit_count"][-1]),
                    "final_wolves": float(obs["wolf_count"][-1]),
                    "ood_excitation": bool(ood_excitation),
                    "ood_ic": bool(ood_ic),
                }
            )

        h5f.attrs["obs_names"] = np.array(obs_names or [], dtype=h5py.string_dtype())
        h5f.attrs["width"] = cfg.width
        h5f.attrs["height"] = cfg.height
        h5f.attrs["num_channels"] = NUM_CHANNELS
        h5f.attrs["control_target"] = "wolves"
        h5f.attrs["abm"] = "wolf_rabbit_grass"

    manifest = {
        "runs": manifest_runs,
        "obs_names": obs_names,
        "abm": "wolf_rabbit_grass",
        "control_target": "wolves",
        "config": {
            "steps": steps,
            "seeds": seeds,
            "seed_offset": seed_offset,
            "initial_rabbits": list(initial_rabbits),
            "initial_wolves": list(initial_wolves),
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
    print(f"Channels: {NUM_CHANNELS} (grass, rabbit, wolf); control culls wolves")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=wolf_dataset_directory(),
        help="Directory for dataset.h5 and manifest.json",
    )
    p.add_argument("--steps", type=int, default=200)
    p.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Seeds per (excitation, IC) cell. Default 5 → 630 trajectories "
        "with the default IC grid (7 excitations × 3×3×2 ICs).",
    )
    p.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Shift ABM/control seeds to offset..offset+seeds-1 (default 0). "
        "Use a new offset to regenerate a statistically independent dataset "
        "without changing the train/val/test split structure.",
    )
    p.add_argument("--initial-rabbits", type=int, nargs="+", default=[80, 110, 140])
    p.add_argument("--initial-wolves", type=int, nargs="+", default=[10, 16, 22])
    p.add_argument("--initial-grass-prob", type=float, nargs="+", default=[0.20, 0.35])
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument(
        "--culling-effectiveness",
        type=float,
        default=0.03,
        help="Max per-step wolf removal probability at u=1 (before lag).",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap rollouts (smoke test)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = WolfRabbitGrassConfig(
        width=args.width,
        height=args.height,
        culling_effectiveness=args.culling_effectiveness,
    )
    generate(
        args.output_dir,
        steps=args.steps,
        seeds=args.seeds,
        seed_offset=args.seed_offset,
        initial_rabbits=tuple(args.initial_rabbits),
        initial_wolves=tuple(args.initial_wolves),
        initial_grass_prob=tuple(args.initial_grass_prob),
        cfg=cfg,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
