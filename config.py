"""Clean experiment configuration for S2Diff-MH.

Only Innovation 1 and the registered Raw-MSI Direct baseline live here.
Non-registration methods will be added later in separate files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    stage: str = "train"
    dataset: str = "PaviaU"
    data_root: str = "./data/raw"
    checkpoint_root: str = "./checkpoints"
    log_root: str = "./logs"
    output_root: str = "./outputs"

    patch_size: int = 64
    stride: int = 32
    test_size: int = 128
    scale_ratio: int = 4
    srf_interp: str = "pchip"

    degradation_mode: str = "physical"
    diffusion_steps: int = 12
    lift_mode: str = "auto"
    mtf_nyquist: float = 0.2
    psf_truncate: float = 3.0
    gaussian_sigma: float = 2.0
    gaussian_kernel_size: int = 5
    boundary_probability: float = 0.2
    boundary_radius: int = 1

    predictor: str = "raw_direct"
    base_channels: int = 64
    time_dim: int = 256
    dropout: float = 0.0
    spectral_hidden: int = 8

    epochs: int = 200
    batch_size: int = 4
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lambda_l1: float = 1.0
    lambda_sam: float = 0.1
    lambda_deg: float = 0.0
    seed: int = 10
    device: str = "cuda"
    eval_interval: int = 5
    save_interval: int = 20
    resume: str = ""
    init_checkpoint: str = ""
    legacy_raw_direct_checkpoint: str = ""
    save_name: str = ""


def parse_args(argv=None) -> TrainConfig:
    p = argparse.ArgumentParser(description="S2Diff-MH clean Innovation1 baseline")
    p.add_argument("--stage", choices=["train", "test", "diagnose"], default="train")
    p.add_argument("--dataset", choices=["PaviaU", "Houston13", "Chikusei"], default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--checkpoint_root", default="./checkpoints")
    p.add_argument("--log_root", default="./logs")
    p.add_argument("--output_root", default="./outputs")
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--test_size", type=int, default=128)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--srf_interp", choices=["pchip", "linear"], default="pchip")

    p.add_argument("--degradation_mode", choices=["physical", "gaussian_bicubic", "bicubic"], default="physical")
    p.add_argument("--diffusion_steps", type=int, default=12)
    p.add_argument("--lift_mode", choices=["auto", "bilinear", "nearest", "adjoint", "normalized_adjoint"], default="auto")
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)
    p.add_argument("--gaussian_sigma", type=float, default=2.0)
    p.add_argument("--gaussian_kernel_size", type=int, default=5)
    p.add_argument("--boundary_probability", type=float, default=0.2)
    p.add_argument("--boundary_radius", type=int, default=1)

    p.add_argument("--predictor", choices=["v1", "v2", "raw_direct"], default="raw_direct")
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--spectral_hidden", type=int, default=8)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_sam", type=float, default=0.1)
    p.add_argument("--lambda_deg", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=20)
    p.add_argument("--resume", default="")
    p.add_argument("--init_checkpoint", default="")
    p.add_argument("--legacy_raw_direct_checkpoint", default="")
    p.add_argument("--save_name", default="")
    return TrainConfig(**vars(p.parse_args(argv)))


def print_config(cfg: TrainConfig):
    print("=" * 80)
    for key, value in vars(cfg).items():
        print(f"{key}: {value}")
    print("=" * 80)
