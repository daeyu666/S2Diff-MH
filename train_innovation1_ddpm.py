"""Standard Gaussian DDPM ablation for S2Diff-MH Innovation 1.

The clean HR-HSI is diffused with a conventional 1000-step Gaussian process.
The registered LR-HSI observation and HR-MSI are supplied as fixed spatial
conditions. Training predicts Gaussian noise with the standard epsilon-MSE
objective; inference uses 12-step deterministic DDIM by default.

This entry point is intentionally separate from main.py and does not modify
the physical Innovation-1 implementation or its checkpoints.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from config import TrainConfig
from data_loader import build_loaders, build_train_val_test_loaders
from diffusion import GaussianDDPMSchedule, ddpm_sample
from innovation1 import build_progressive_process
from metrics import MetricAverager, calc_metrics
from models import RawMSIDirectPredictor
from utils import (
    AverageMeter,
    CSVLogger,
    count_parameters,
    ensure_dir,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standard Gaussian DDPM ablation for S2Diff-MH Innovation 1"
    )
    parser.add_argument("--stage", choices=["train", "test"], default="train")
    parser.add_argument(
        "--dataset",
        choices=["PaviaU", "Houston13", "Chikusei", "CAVE", "Botswana", "Augsburg"],
        default="PaviaU",
    )
    parser.add_argument("--data_root", default="./data/raw")
    parser.add_argument("--checkpoint_root", default="./checkpoints")
    parser.add_argument("--log_root", default="./logs")

    # Keep the registered benchmark/data protocol identical to Innovation 1.
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--test_size", type=int, default=128)
    parser.add_argument("--scale_ratio", type=int, default=4)
    parser.add_argument("--srf_interp", choices=["pchip", "linear"], default="pchip")
    parser.add_argument("--mtf_nyquist", type=float, default=0.2)
    parser.add_argument("--psf_truncate", type=float, default=3.0)

    # Standard Gaussian diffusion settings.
    parser.add_argument("--total_diffusion_steps", type=int, default=12)
    parser.add_argument(
        "--sampling_steps", type=int, default=12,
        help="Must equal total_diffusion_steps; standard ancestral DDPM uses every step.",
    )
    parser.add_argument("--cosine_offset", type=float, default=0.008)

    # Match the physical Innovation-1 Raw-MSI Direct backbone.
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=256)
    parser.add_argument("--spectral_hidden", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Match the original training budget and optimizer.
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=20)

    parser.add_argument("--resume", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--save_name", default="")
    parser.add_argument(
        "--overwrite_run",
        action="store_true",
        help="Remove an existing DDPM run with the same names before fresh training.",
    )
    return parser.parse_args()


def build_data_config(args) -> TrainConfig:
    """Create the existing data-loader config without changing its protocol."""
    return TrainConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        patch_size=args.patch_size,
        stride=args.stride,
        test_size=args.test_size,
        scale_ratio=args.scale_ratio,
        srf_interp=args.srf_interp,
        degradation_mode="physical",
        # This 12-step process is used only to generate the matched LR-HSI
        # observation and its normalized-adjoint spatial condition.
        diffusion_steps=12,
        lift_mode="auto",
        mtf_nyquist=args.mtf_nyquist,
        psf_truncate=args.psf_truncate,
        predictor="raw_direct",
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        dropout=args.dropout,
        spectral_hidden=args.spectral_hidden,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def build_model(args, info: Dict, device: torch.device) -> RawMSIDirectPredictor:
    n_hsi = int(info["n_bands"])
    n_msi = int(info["n_msi_bands"])
    # The condition tensor concatenates normalized-adjoint LR-HSI and HR-MSI.
    condition_channels = n_hsi + n_msi
    model = RawMSIDirectPredictor(
        n_bands=n_hsi,
        n_msi_bands=condition_channels,
        total_steps=args.total_diffusion_steps,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        dropout=args.dropout,
        residual_prediction=False,  # predict epsilon, not x_0
        spectral_hidden=args.spectral_hidden,
    ).to(device)
    print(
        f"predictor=RawMSIDirectPredictor(epsilon), "
        f"condition_channels={condition_channels}, "
        f"params={count_parameters(model):.3f}M"
    )
    return model


def build_condition(
    gt: torch.Tensor,
    hr_msi: torch.Tensor,
    physical_process,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build matched LR-HSI/HR-MSI conditions; never used as the DDPM state."""
    with torch.no_grad():
        lr_hsi = physical_process.terminal_observation(gt)
        lr_hsi_up = physical_process.terminal_state(
            lr_hsi,
            target_size=tuple(gt.shape[-2:]),
        )
        condition = torch.cat([lr_hsi_up, hr_msi], dim=1)
        condition = condition.mul(2.0).sub(1.0)
    if condition.shape[-2:] != gt.shape[-2:]:
        raise RuntimeError("Condition and HR-HSI spatial grids do not match")
    return condition, lr_hsi_up


def checkpoint_paths(args) -> Tuple[str, str]:
    root = os.path.join(args.checkpoint_root, "innovation1_ddpm")
    ensure_dir(root)
    name = args.save_name or f"{args.dataset}_innovation1_standard_ddpm.pth"
    if not name.endswith(".pth"):
        name += ".pth"
    stem, ext = os.path.splitext(name)
    return os.path.join(root, name), os.path.join(root, stem + "_last" + ext)


def log_path(args) -> str:
    ensure_dir(args.log_root)
    name = args.save_name or f"{args.dataset}_innovation1_standard_ddpm.pth"
    stem = os.path.splitext(os.path.basename(name))[0]
    return os.path.join(args.log_root, stem + ".csv")


def run_metadata(args, info: Dict) -> Dict:
    metadata = vars(args).copy()
    metadata.update(
        {
            "diffusion_type": "standard_12_step_cosine_gaussian_ddpm",
            "prediction_type": "epsilon",
            "training_objective": "mean_squared_error",
            "state_normalization": "reflectance_[0,1]_mapped_to_[-1,1]",
            "beta_schedule": "cosine",
            "reverse_process": "12_step_ancestral_ddpm",
            "condition": "normalized_adjoint_LR_HSI_plus_registered_HR_MSI_mapped_to_[-1,1]",
            "n_hsi_bands": int(info["n_bands"]),
            "n_msi_bands": int(info["n_msi_bands"]),
            "condition_channels": int(info["n_bands"]) + int(info["n_msi_bands"]),
            "physical_condition_scale_ratio": int(args.scale_ratio),
        }
    )
    return metadata


def train_one_epoch(
    model,
    loader,
    optimizer,
    physical_process,
    schedule: GaussianDDPMSchedule,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        condition, _ = build_condition(gt, hr_msi, physical_process)

        batch_size = gt.shape[0]
        timesteps = torch.randint(
            low=1,
            high=schedule.total_steps + 1,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )
        # Standard DDPM is trained in a zero-centered data domain.
        x0 = gt.mul(2.0).sub(1.0)
        noise = torch.randn_like(x0)
        noisy_hsi = schedule.q_sample(x0, timesteps, noise)
        predicted_noise = model(noisy_hsi, condition, timesteps)

        if predicted_noise.shape != noise.shape:
            raise RuntimeError(
                f"Predicted noise shape {tuple(predicted_noise.shape)} "
                f"does not match target noise {tuple(noise.shape)}"
            )
        loss = F.mse_loss(predicted_noise, noise)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite Gaussian DDPM epsilon-MSE loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(grad_clip),
                error_if_nonfinite=True,
            )
        optimizer.step()
        loss_meter.update(loss.item(), batch_size)

    return loss_meter.avg


@torch.no_grad()
def evaluate(
    model,
    loader,
    physical_process,
    schedule: GaussianDDPMSchedule,
    device: torch.device,
    *,
    scale_ratio: int,
    sampling_steps: int,
    seed: int,
) -> Dict[str, float]:
    model.eval()
    final_meter = MetricAverager()
    initial_meter = MetricAverager()

    # A dedicated generator makes validation/test sampling reproducible and
    # independent of the training RNG state.
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        condition, lr_hsi_up = build_condition(gt, hr_msi, physical_process)

        prediction_normalized = ddpm_sample(
            model,
            condition,
            sample_shape=gt.shape,
            schedule=schedule,
            generator=generator,
            clip_x0=True,
        )
        prediction = ((prediction_normalized + 1.0) * 0.5).clamp(0.0, 1.0)
        final_meter.update(calc_metrics(prediction, gt, scale_ratio))
        initial_meter.update(calc_metrics(lr_hsi_up, gt, scale_ratio))

    metrics = final_meter.average()
    initial = initial_meter.average()
    metrics["INIT_PSNR"] = initial.get("PSNR", float("nan"))
    metrics["INIT_SAM"] = initial.get("SAM", float("nan"))
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    keys = ["PSNR", "SAM", "RMSE", "ERGAS", "SSIM", "CC", "INIT_PSNR", "INIT_SAM"]
    return " ".join(
        f"{key}={metrics[key]:.6f}" for key in keys if key in metrics
    )


def run_train(args, train_loader, val_loader, info, device):
    data_cfg = build_data_config(args)
    physical_process = build_progressive_process(data_cfg)
    schedule = GaussianDDPMSchedule(
        total_steps=args.total_diffusion_steps,
        cosine_offset=args.cosine_offset,
        device=device,
    )
    model = build_model(args, info, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_path, last_path = checkpoint_paths(args)
    csv_path = log_path(args)
    start_epoch = 1
    best_psnr = float("-inf")

    if args.resume:
        loaded_epoch, best_psnr = load_checkpoint(
            model,
            args.resume,
            optimizer=optimizer,
            map_location=str(device),
        )
        start_epoch = loaded_epoch + 1
    else:
        existing = [path for path in (best_path, last_path, csv_path) if os.path.exists(path)]
        if existing and not args.overwrite_run:
            raise FileExistsError(
                "A DDPM run already exists at:\n  "
                + "\n  ".join(existing)
                + "\nUse --resume <*_last.pth>, choose another --save_name, "
                "or pass --overwrite_run to discard this run."
            )
        if existing:
            for path in existing:
                os.remove(path)
                print(f"OVERWRITE_RUN removed={path}")

    print(
        f"DDPM schedule: T={schedule.total_steps}, "
        f"schedule=cosine, terminal_alpha_bar={schedule.alpha_bars[-1].item():.6e}, "
        f"ancestral reverse steps={args.total_diffusion_steps}"
    )
    print(
        f"Physical LR-HSI condition: scale={args.scale_ratio}, "
        f"MTF={args.mtf_nyquist}, PSF truncate={args.psf_truncate}"
    )

    logger = CSVLogger(
        csv_path,
        ["epoch", "epsilon_mse", "PSNR", "SAM", "best_PSNR"],
    )
    metadata = run_metadata(args, info)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            physical_process,
            schedule,
            device,
            grad_clip=args.grad_clip,
        )
        row = {
            "epoch": epoch,
            "epsilon_mse": train_loss,
            "PSNR": "",
            "SAM": "",
            "best_PSNR": best_psnr,
        }
        print(f"epoch={epoch:03d} epsilon_mse={train_loss:.7f}")

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            metrics = evaluate(
                model,
                val_loader,
                physical_process,
                schedule,
                device,
                scale_ratio=args.scale_ratio,
                sampling_steps=args.sampling_steps,
                seed=args.seed + 100000,
            )
            print("val", format_metrics(metrics))
            row["PSNR"] = metrics["PSNR"]
            row["SAM"] = metrics["SAM"]

            if metrics["PSNR"] > best_psnr:
                best_psnr = metrics["PSNR"]
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_psnr,
                    best_path,
                    extra=metadata,
                )
                print("saved best:", best_path)
            row["best_PSNR"] = best_psnr

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_psnr,
                last_path,
                extra=metadata,
            )
            print("saved last:", last_path)

        logger.write(row)


def run_test(args, test_loader, info, device):
    data_cfg = build_data_config(args)
    physical_process = build_progressive_process(data_cfg)
    schedule = GaussianDDPMSchedule(
        total_steps=args.total_diffusion_steps,
        cosine_offset=args.cosine_offset,
        device=device,
    )
    model = build_model(args, info, device)
    best_path, _ = checkpoint_paths(args)
    checkpoint = args.checkpoint or args.resume or best_path
    load_checkpoint(model, checkpoint, map_location=str(device))

    metrics = evaluate(
        model,
        test_loader,
        physical_process,
        schedule,
        device,
        scale_ratio=args.scale_ratio,
        sampling_steps=args.sampling_steps,
        seed=args.seed + 200000,
    )
    print("test", format_metrics(metrics))


def main():
    args = parse_args()
    if args.sampling_steps != args.total_diffusion_steps:
        raise ValueError(
            "Standard ancestral DDPM must use all steps: set sampling_steps "
            "equal to total_diffusion_steps."
        )
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    set_seed(args.seed)
    device = resolve_device(args.device)
    data_cfg = build_data_config(args)

    if args.stage == "train":
        train_loader, val_loader, _, info = build_train_val_test_loaders(data_cfg)
        print("dataset info:", info)
        run_train(args, train_loader, val_loader, info, device)
    else:
        _, test_loader, info = build_loaders(data_cfg)
        print("dataset info:", info)
        run_test(args, test_loader, info, device)


if __name__ == "__main__":
    main()
