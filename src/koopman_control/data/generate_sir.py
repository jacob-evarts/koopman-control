"""Generate the Case-B spatial agentic SIR dataset.

The scalar control is systemic vaccination intensity. Frames contain three
binary occupancy channels: susceptible, infected, and recovered. Storage uses
uint8 with ``frame_scale=1`` (occupancy already in ``{0, 1}``).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import h5py
import numpy as np

from koopman_control.data.agentic_sir import NUM_CHANNELS, AgenticSIRConfig, rollout
from koopman_control.data.excitation import make_control
from koopman_control.data.generate import ALL_EXCITATIONS, OOD_EXCITATIONS, assign_split
from koopman_control.paths import sir_dataset_directory

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

CHANNEL_NAMES = ("susceptible", "infected", "recovered")
FRAME_SCALE = 1.0


def _stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _run_id(
    excitation: str,
    n_agents: int,
    initial_infected: int,
    center: tuple[float, float],
    seed: int,
) -> str:
    cx, cy = (int(round(100 * v)) for v in center)
    return f"{excitation}_n{n_agents}_i{initial_infected}_x{cx}_y{cy}_s{seed}"


def generate(
    out_dir: Path,
    *,
    steps: int = 200,
    seeds: int = 5,
    seed_offset: int = 0,
    n_agents: tuple[int, ...] = (800, 1200),
    initial_infected: tuple[int, ...] = (20, 32, 48),
    seed_centers: tuple[tuple[float, float], ...] = (
        (0.30, 0.30),
        (0.50, 0.50),
        (0.70, 0.70),
    ),
    seed_radius: float = 5.0,
    excitations: tuple[str, ...] = ALL_EXCITATIONS,
    cfg: AgenticSIRConfig | None = None,
    limit: int | None = None,
) -> dict:
    """Roll out the sweep and write ``dataset.h5`` plus ``manifest.json``."""
    cfg = cfg or AgenticSIRConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "dataset.h5"

    jobs = list(
        itertools.product(
            excitations,
            enumerate(initial_infected),
            n_agents,
            seed_centers,
            range(seeds),
        )
    )
    if limit is not None:
        jobs = jobs[:limit]
    iterator = tqdm(jobs, desc="sir rollouts") if tqdm is not None else jobs

    manifest_runs: list[dict] = []
    obs_names: list[str] | None = None
    with h5py.File(h5_path, "w") as h5f:
        runs_grp = h5f.create_group("runs")
        for (
            excitation,
            (infected_index, n_infected),
            agents,
            center,
            seed_index,
        ) in iterator:
            seed = int(seed_offset + seed_index)
            control_rng = np.random.default_rng(
                _stable_seed(excitation, agents, n_infected, center, seed, "control")
            )
            control_seq = make_control(excitation, steps, control_rng)
            cx = center[0] * (cfg.width - 1)
            cy = center[1] * (cfg.height - 1)
            frames, controls, obs = rollout(
                cfg,
                control_seq,
                n_agents=int(agents),
                initial_infected=int(n_infected),
                seed_center_x=cx,
                seed_center_y=cy,
                seed_radius=float(seed_radius),
                seed=seed,
            )
            if obs_names is None:
                obs_names = list(obs)
            obs_mat = np.stack([obs[k] for k in obs_names], axis=1).astype(np.float32)

            rid = _run_id(excitation, int(agents), int(n_infected), center, seed)
            split = assign_split(seed_index, n_seeds=seeds)
            ood_excitation = excitation in OOD_EXCITATIONS
            ood_ic = infected_index == len(initial_infected) - 1
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
                n_agents=int(agents),
                initial_infected=int(n_infected),
                seed_center_x=float(cx),
                seed_center_y=float(cy),
                seed_radius=float(seed_radius),
                seed=seed,
                split=split,
                steps=int(steps),
                ood_excitation=bool(ood_excitation),
                ood_ic=bool(ood_ic),
                control_target="vaccination",
            )
            manifest_runs.append(
                {
                    "run_id": rid,
                    "excitation": excitation,
                    "n_agents": int(agents),
                    "initial_infected": int(n_infected),
                    "seed_center_x": float(cx),
                    "seed_center_y": float(cy),
                    "seed_radius": float(seed_radius),
                    "seed": seed,
                    "split": split,
                    "steps": int(steps),
                    "mean_control": float(controls.mean()),
                    "peak_infected": float(obs["infected_count"].max()),
                    "final_infected": float(obs["infected_count"][-1]),
                    "final_incidence": float(obs["cumulative_incidence"][-1]),
                    "ood_excitation": bool(ood_excitation),
                    "ood_ic": bool(ood_ic),
                }
            )

        h5f.attrs["obs_names"] = np.array(obs_names or [], dtype=h5py.string_dtype())
        h5f.attrs["channel_names"] = np.array(CHANNEL_NAMES, dtype=h5py.string_dtype())
        h5f.attrs["width"] = cfg.width
        h5f.attrs["height"] = cfg.height
        h5f.attrs["num_channels"] = NUM_CHANNELS
        h5f.attrs["frame_scale"] = FRAME_SCALE
        h5f.attrs["control_target"] = "vaccination"
        h5f.attrs["abm"] = "agentic_sir"

    manifest = {
        "runs": manifest_runs,
        "obs_names": obs_names,
        "channel_names": list(CHANNEL_NAMES),
        "abm": "agentic_sir",
        "control_target": "vaccination",
        "frame_scale": FRAME_SCALE,
        "config": {
            "steps": steps,
            "seeds": seeds,
            "seed_offset": seed_offset,
            "n_agents": list(n_agents),
            "initial_infected": list(initial_infected),
            "seed_centers": [list(c) for c in seed_centers],
            "seed_radius": float(seed_radius),
            "excitations": list(excitations),
            **cfg.__dict__,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    counts: dict[str, int] = {}
    for row in manifest_runs:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    print(f"Wrote {len(manifest_runs)} trajectories to {h5_path}")
    print(f"Split counts: {counts}")
    print(f"Channels: {', '.join(CHANNEL_NAMES)}; control administers vaccination")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=sir_dataset_directory())
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--n-agents", type=int, nargs="+", default=[800, 1200])
    p.add_argument("--initial-infected", type=int, nargs="+", default=[20, 32, 48])
    p.add_argument(
        "--seed-centers",
        type=float,
        nargs="+",
        default=[0.30, 0.30, 0.50, 0.50, 0.70, 0.70],
        metavar=("X", "Y"),
        help="Flat list of normalized x y pairs for the outbreak seed.",
    )
    p.add_argument("--seed-radius", type=float, default=5.0)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument(
        "--infection-prob",
        type=float,
        default=AgenticSIRConfig.infection_prob,
    )
    p.add_argument(
        "--recovery-prob",
        type=float,
        default=AgenticSIRConfig.recovery_prob,
    )
    p.add_argument(
        "--vaccine-effectiveness",
        type=float,
        default=0.028,
        help="Per-step S→R probability at u=1 (lower yields more graded dose response).",
    )
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if len(args.seed_centers) % 2:
        raise ValueError("--seed-centers requires x y pairs")
    centers = tuple(
        (float(args.seed_centers[i]), float(args.seed_centers[i + 1]))
        for i in range(0, len(args.seed_centers), 2)
    )
    generate(
        args.output_dir,
        steps=args.steps,
        seeds=args.seeds,
        seed_offset=args.seed_offset,
        n_agents=tuple(args.n_agents),
        initial_infected=tuple(args.initial_infected),
        seed_centers=centers,
        seed_radius=args.seed_radius,
        cfg=AgenticSIRConfig(
            width=args.width,
            height=args.height,
            infection_prob=args.infection_prob,
            recovery_prob=args.recovery_prob,
            vaccine_effectiveness=args.vaccine_effectiveness,
        ),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
