"""Launch strict Augsburg-2 Wald-D2 A/B/C geometry comparison.

Only changes across branches:
 A: identity (original unregistered MSI30)
 B: fixed (dx=-0.5,dy=0) in MSI30 grid
 C: frozen Wald-CDRDI with rigid-only residual + physical-closure backtracking

All branches: observed HSI30 GT, Wald HSI90, real S2 MSI30, same seed,
train/eval splits, official SRF, Gaussian sigma1.2, calibration, D2 architecture
and schedule.  Never use old EnMAP10-supervised pretraining or labels.
"""
import argparse
import os
import shlex
import subprocess
import sys


BRANCH_MODES = {
    "A": "identity",
    "B": "wald_fixed",
    "C": "wald_cdrdi",
}
CHECKPOINT_C_DEFAULT = "./checkpoints/augsburg_real/Augsburg2_Wald_C_seeded_minus05.pth"


def make_command(args):
    branch = str(args.branch).upper()
    if branch not in BRANCH_MODES:
        raise ValueError(f"Unknown Wald branch {branch}")
    mode = BRANCH_MODES[branch]
    if args.stage not in ("train", "test"):
        raise ValueError("stage must be train or test")
    common = [
        sys.executable, "train_augsburg_real_diffusion.py",
        "--stage", args.stage,
        "--cache_root", args.wald_root,
        "--psf_json", args.psf_json,
        "--radiometry_json", args.radiometry_json,
        "--geometry_mode", mode,
        "--fixed_dx_px", "-0.5",
        "--fixed_dy_px", "0",
        "--train_patch_size", "72",
        "--train_stride", "6",
        "--eval_patch_size", "48",
        "--batch_size", str(args.batch_size),
        "--num_workers", "0",
        "--diffusion_steps", "12",
        "--base_channels", "64",
        "--time_dim", "256",
        "--spectral_hidden", "8",
        "--epochs", str(args.epochs),
        "--lr", "1e-4",
        "--weight_decay", "0",
        "--eval_interval", "5",
        "--monitor", "ref_psnr",
        "--seed", str(args.seed),
        "--device", args.device,
        "--save_name", f"Augsburg2_Wald_D2_{branch}",
        "--checkpoint_root", args.checkpoint_root,
        "--log_root", args.log_root,
    ]
    if branch == "C":
        common += [
            "--geometry_checkpoint", args.geometry_checkpoint,
            "--geometry_steps", "9",
            "--guard_window", "5",
            "--guard_min_jac", "0.5",
            "--guard_relative_gain", "0.0001",
        ]
    if args.stage == "train":
        common.append("--from_scratch")
    else:
        common += [
            "--diffusion_checkpoint", os.path.join(
                args.checkpoint_root, f"Augsburg2_Wald_D2_{branch}.pth"
            ),
            "--metrics_json", os.path.join(
                args.log_root, f"Augsburg2_Wald_D2_{branch}_test.json"
            ),
        ]
    return common


def parse_args():
    p = argparse.ArgumentParser(description="Strict Augsburg-2 Wald A/B/C D2 launch")
    p.add_argument("--branch", required=True, choices=("A", "B", "C"))
    p.add_argument("--stage", choices=("train", "test"), default="train")
    p.add_argument("--wald_root", default="./data/augsburg2_wald")
    p.add_argument("--psf_json", default="./data/augsburg2_wald/wald_psf.json")
    p.add_argument("--radiometry_json",
                   default="./data/calibration/Augsburg2_Wald_radiometry.json")
    p.add_argument("--geometry_checkpoint", default=CHECKPOINT_C_DEFAULT)
    p.add_argument("--checkpoint_root", default="./checkpoints/augsburg_real")
    p.add_argument("--log_root", default="./logs/augsburg_real")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Use 1 on 12GB GPU; raise to 2 only after memory verification")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cmd = make_command(args)
    if not args.dry_run:
        for path in (args.psf_json, args.radiometry_json):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
        if args.branch == "C" and not os.path.isfile(args.geometry_checkpoint):
            raise FileNotFoundError(
                "Frozen strictly Wald-trained CDRDI checkpoint is required: "
                + args.geometry_checkpoint
            )
        if args.stage == "test":
            check_path = os.path.join(
                args.checkpoint_root, f"Augsburg2_Wald_D2_{args.branch}.pth"
            )
            if not os.path.isfile(check_path):
                raise FileNotFoundError(check_path)
    print(f"WALD_ABC branch={args.branch} stage={args.stage} model_init="
          f"{'independent_scratch' if args.stage == 'train' else 'heldout_checkpoint'}",
          flush=True)
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
