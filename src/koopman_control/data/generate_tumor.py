"""Generate the Case-A spatial tumor–healthy-tissue dataset.

The scalar control is systemic chemotherapy dose. Frames contain four channels:
healthy occupancy, tumor occupancy, nutrient concentration, and drug
concentration. Continuous channels are quantized to uint8 with
``frame_scale=255`` in the HDF5 attributes; loaders divide by that scale.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import h5py
import numpy as np

from koopman_control.data.excitation import make_control
from koopman_control.data.generate import ALL_EXCITATIONS, OOD_EXCITATIONS, assign_split
from koopman_control.data.tumor_tissue import (
    NUM_CHANNELS,
    TumorTissueConfig,
    rollout,
)
from koopman_control.paths import tumor_dataset_directory

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

CHANNEL_NAMES = ("healthy", "tumor", "nutrient", "drug")
FRAME_SCALE = 255.0


def _stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _run_id(
    excitation: str,
    healthy_frac: float,
    tumor_radius: float,
    center: tuple[float, float],
    seed: int,
) -> str:
    hf = int(round(healthy_frac * 100))
    r = int(round(tumor_radius))
    cx, cy = (int(round(100 * v)) for v in center)
    return f"{excitation}_h{hf}_r{r}_x{cx}_y{cy}_s{seed}"


def generate(
    out_dir: Path,
    *,
    steps: int = 200,
    seeds: int = 5,
    seed_offset: int = 0,
    initial_healthy_frac: tuple[float, ...] = (0.90, 0.94),
    initial_tumor_radius: tuple[float, ...] = (4.0, 6.0, 8.0),
    tumor_centers: tuple[tuple[float, float], ...] = (
        (0.30, 0.30),
        (0.50, 0.50),
        (0.70, 0.70),
    ),
    excitations: tuple[str, ...] = ALL_EXCITATIONS,
    cfg: TumorTissueConfig | None = None,
    limit: int | None = None,
) -> dict:
    """Roll out the sweep and write ``dataset.h5`` plus ``manifest.json``."""
    cfg = cfg or TumorTissueConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "dataset.h5"

    jobs = list(
        itertools.product(
            excitations,
            enumerate(initial_tumor_radius),
            initial_healthy_frac,
            tumor_centers,
            range(seeds),
        )
    )
    if limit is not None:
        jobs = jobs[:limit]
    iterator = tqdm(jobs, desc="tumor rollouts") if tqdm is not None else jobs

    manifest_runs: list[dict] = []
    obs_names: list[str] | None = None
    with h5py.File(h5_path, "w") as h5f:
        runs_grp = h5f.create_group("runs")
        for excitation, (radius_index, radius), healthy_frac, center, seed_index in iterator:
            seed = int(seed_offset + seed_index)
            control_rng = np.random.default_rng(
                _stable_seed(excitation, healthy_frac, radius, center, seed, "control")
            )
            control_seq = make_control(excitation, steps, control_rng)
            cx, cy = center[0] * (cfg.width - 1), center[1] * (cfg.height - 1)
            frames, controls, obs = rollout(
                cfg,
                control_seq,
                initial_healthy_frac=healthy_frac,
                initial_tumor_radius=radius,
                tumor_center_x=cx,
                tumor_center_y=cy,
                vessel_offset=seed_index % max(2, cfg.vessel_spacing),
                seed=seed,
            )
            if obs_names is None:
                obs_names = list(obs)
            obs_mat = np.stack([obs[k] for k in obs_names], axis=1).astype(np.float32)

            rid = _run_id(excitation, healthy_frac, radius, center, seed)
            split = assign_split(seed_index, n_seeds=seeds)
            ood_excitation = excitation in OOD_EXCITATIONS
            ood_ic = radius_index == len(initial_tumor_radius) - 1
            grp = runs_grp.create_group(rid)
            grp.create_dataset(
                "frames",
                data=np.rint(np.clip(frames, 0.0, 1.0) * FRAME_SCALE).astype(np.uint8),
                compression="gzip",
                compression_opts=4,
            )
            grp.create_dataset("control", data=controls.astype(np.float32))
            grp.create_dataset("obs", data=obs_mat)
            grp.attrs.update(
                excitation=excitation,
                initial_healthy_frac=float(healthy_frac),
                initial_tumor_radius=float(radius),
                tumor_center_x=float(cx),
                tumor_center_y=float(cy),
                initial_healthy_count=float(obs["healthy_count"][0]),
                initial_tumor_count=float(obs["tumor_count"][0]),
                seed=seed,
                split=split,
                steps=int(steps),
                ood_excitation=bool(ood_excitation),
                ood_ic=bool(ood_ic),
                control_target="chemotherapy",
            )
            manifest_runs.append(
                {
                    "run_id": rid,
                    "excitation": excitation,
                    "initial_healthy_frac": float(healthy_frac),
                    "initial_tumor_radius": float(radius),
                    "tumor_center_x": float(cx),
                    "tumor_center_y": float(cy),
                    "seed": seed,
                    "split": split,
                    "steps": int(steps),
                    "mean_control": float(controls.mean()),
                    "final_tumor": float(obs["tumor_count"][-1]),
                    "final_healthy": float(obs["healthy_count"][-1]),
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
        h5f.attrs["control_target"] = "chemotherapy"
        h5f.attrs["abm"] = "tumor_tissue"

    manifest = {
        "runs": manifest_runs,
        "obs_names": obs_names,
        "channel_names": list(CHANNEL_NAMES),
        "abm": "tumor_tissue",
        "control_target": "chemotherapy",
        "frame_scale": FRAME_SCALE,
        "config": {
            "steps": steps,
            "seeds": seeds,
            "seed_offset": seed_offset,
            "initial_healthy_frac": list(initial_healthy_frac),
            "initial_tumor_radius": list(initial_tumor_radius),
            "tumor_centers": [list(x) for x in tumor_centers],
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
    print(f"Channels: {', '.join(CHANNEL_NAMES)}; control administers chemotherapy")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=tumor_dataset_directory())
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--initial-healthy-frac", type=float, nargs="+", default=[0.90, 0.94])
    p.add_argument("--initial-tumor-radius", type=float, nargs="+", default=[4.0, 6.0, 8.0])
    p.add_argument(
        "--tumor-centers",
        type=float,
        nargs="+",
        default=[0.30, 0.30, 0.50, 0.50, 0.70, 0.70],
        metavar=("X", "Y"),
        help="Flat list of normalized x y pairs.",
    )
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument(
        "--drug-diffusion",
        type=float,
        default=TumorTissueConfig.drug_diffusion,
    )
    p.add_argument(
        "--drug-delivery",
        type=float,
        default=TumorTissueConfig.drug_delivery,
    )
    p.add_argument("--drug-decay", type=float, default=TumorTissueConfig.drug_decay)
    p.add_argument(
        "--healthy-drug-kill",
        type=float,
        default=TumorTissueConfig.healthy_drug_kill,
    )
    p.add_argument(
        "--tumor-drug-kill",
        type=float,
        default=TumorTissueConfig.tumor_drug_kill,
    )
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if len(args.tumor_centers) % 2:
        raise ValueError("--tumor-centers requires x y pairs")
    centers = tuple(
        (float(args.tumor_centers[i]), float(args.tumor_centers[i + 1]))
        for i in range(0, len(args.tumor_centers), 2)
    )
    generate(
        args.output_dir,
        steps=args.steps,
        seeds=args.seeds,
        seed_offset=args.seed_offset,
        initial_healthy_frac=tuple(args.initial_healthy_frac),
        initial_tumor_radius=tuple(args.initial_tumor_radius),
        tumor_centers=centers,
        cfg=TumorTissueConfig(
            width=args.width,
            height=args.height,
            drug_diffusion=args.drug_diffusion,
            drug_delivery=args.drug_delivery,
            drug_decay=args.drug_decay,
            healthy_drug_kill=args.healthy_drug_kill,
            tumor_drug_kill=args.tumor_drug_kill,
        ),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
