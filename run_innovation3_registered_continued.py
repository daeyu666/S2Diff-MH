"""Run same-backbone registered-continuation controls for simulated datasets.

Purpose:
  Original registered GIGI checkpoint (normally 60 ep)
      -> additional 100 ep on REGISTERED samples only
using the SAME frozen final CDRDI-stage2 diffusion backbone, refiner
initialization, optimizer hyperparameters and loss weights as the mixed
GIGI-CDRDI experiment.  The principal controlled difference versus mixed
training is the extra-training geometry distribution.

Default main simulated datasets:
  PaviaU,Houston13,Chikusei,CAVE

Expected checkpoint naming follows the existing PaviaU convention:
  geometry: ./checkpoints/cdrdi_stage1/{D}_recursive_k6_finalonly_300ep_lr1e4.pth
  diffusion: ./checkpoints/cdrdi_stage2/{D}_estimated_deform_diffusion_k9_stage2d_A.pth
  init GIGI: ./checkpoints/innovation3_gigi/{D}_innovation3_gigi_full.pth

Use --dry_run first. Missing prerequisites are reported together before launch.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys


MAIN_DATASETS = ("PaviaU", "Houston13", "Chikusei", "CAVE")
ALL_DATASETS = MAIN_DATASETS + ("Botswana", "Augsburg")


def paths(dataset, args):
    return {
        "geometry": os.path.join(
            args.geometry_root,
            f"{dataset}_recursive_k6_finalonly_300ep_lr1e4.pth",
        ),
        "diffusion": os.path.join(
            args.diffusion_root,
            f"{dataset}_estimated_deform_diffusion_k9_stage2d_A.pth",
        ),
        "init": os.path.join(
            args.registered_gigi_root,
            f"{dataset}_innovation3_gigi_full.pth",
        ),
    }


def command(dataset, args, *, stage):
    p = paths(dataset, args)
    save_name = f"{dataset}_innovation3_gigi_registered_continued_full_k9"
    cmd = [
        sys.executable, "train_innovation3_gigi_cdrdi.py",
        "--stage", stage,
        "--dataset", dataset,
        "--variant", "full",
        "--train_geometry_mode", "registered",
        "--geometry_checkpoint", p["geometry"],
        "--diffusion_checkpoint", p["diffusion"],
        "--init_refiner_checkpoint", p["init"],
        "--geometry_steps", "9",
        "--epochs", str(args.epochs),
        "--lr", "2e-4",
        "--batch_size", str(args.batch_size),
        "--lambda_l1", "1.0",
        "--lambda_ang", "0.1",
        "--lambda_phy", "0.1",
        "--lambda_msi", "0.1",
        "--monitor", "registered_sam_high",
        "--eval_interval", "5",
        "--save_interval", "10",
        "--seed", str(args.seed),
        "--device", args.device,
        "--checkpoint_root", args.checkpoint_root,
        "--log_root", args.log_root,
        "--save_name", save_name,
    ]
    if stage == "test":
        cmd += [
            "--refiner_checkpoint",
            os.path.join(args.checkpoint_root, save_name + ".pth"),
        ]
    return cmd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        default=",".join(MAIN_DATASETS),
        help="Comma-separated datasets. Add Botswana/Augsburg only if used in the final simulated table.",
    )
    p.add_argument("--stage", choices=["train", "test", "train_test"], default="train")
    p.add_argument("--epochs", type=int, default=100,
                   help="ADDITIONAL epochs after loading the original registered GIGI weights")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--geometry_root", default="./checkpoints/cdrdi_stage1")
    p.add_argument("--diffusion_root", default="./checkpoints/cdrdi_stage2")
    p.add_argument("--registered_gigi_root", default="./checkpoints/innovation3_gigi")
    p.add_argument(
        "--checkpoint_root",
        default="./checkpoints/innovation3_gigi_registered_continued",
    )
    p.add_argument("--log_root", default="./logs")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    unknown = [x for x in datasets if x not in ALL_DATASETS]
    if not datasets or unknown:
        raise ValueError(f"Invalid datasets={unknown or datasets}; allowed={ALL_DATASETS}")
    if args.epochs != 100:
        print(
            f"WARNING continuation epochs={args.epochs}; mixed-training default is 100. "
            "Use 100 for the controlled comparison.",
            flush=True,
        )

    missing = []
    for dataset in datasets:
        for role, filename in paths(dataset, args).items():
            if not os.path.isfile(filename):
                missing.append((dataset, role, filename))
    if missing and not args.dry_run:
        details = "\n".join(
            f"  {dataset} {role}: {filename}"
            for dataset, role, filename in missing
        )
        raise FileNotFoundError(
            "Registered-continuation prerequisites missing:\n" + details
            + "\nDo not silently substitute another backbone/checkpoint; "
              "pass the actual checkpoint roots or keep dataset out of --datasets."
        )

    stages = ("train", "test") if args.stage == "train_test" else (args.stage,)
    for dataset in datasets:
        for stage in stages:
            cmd = command(dataset, args, stage=stage)
            print(
                f"REGISTERED_CONTINUED dataset={dataset} stage={stage} "
                f"additional_epochs={args.epochs}",
                flush=True,
            )
            print(shlex.join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
