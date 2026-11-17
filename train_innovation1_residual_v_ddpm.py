"""Residual-conditioned standard Gaussian DDPM ablation using v-prediction.

The forward process remains the conventional linear-beta Gaussian Markov chain.
Only the prediction parameterization changes from epsilon to velocity v.
Residual statistics are fitted on the training split only and stored in the
checkpoint. Reconstruction adds the denoised residual back to the LR-HSI
baseline.

This entry point is separate from the earlier epsilon-prediction residual
baseline, the full-HSI Gaussian DDPM baseline, and physical S2Diff-MH.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion import GaussianDDPMSchedule
from diffusion.gaussian_v_ddpm import ddpm_v_sample
from innovation1 import build_progressive_process
from metrics import MetricAverager, calc_metrics
from train_innovation1_ddpm import (
    build_condition,
    build_data_config,
    resolve_device,
)
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
        description="Residual-conditioned standard Gaussian DDPM with v-prediction"
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

    # Conventional Gaussian DDPM.
    parser.add_argument("--total_diffusion_steps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)

    # Residual normalization: per-band scale = factor * train-split std.
    parser.add_argument("--residual_scale_factor", type=float, default=4.0)
    parser.add_argument("--min_residual_scale", type=float, default=1e-4)

    # Keep the Raw-MSI Direct backbone matched to the previous experiment.
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
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=20)

    parser.add_argument("--resume", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--save_name",
        default="PaviaU_innovation1_residual_v_gaussian_ddpm.pth",
    )
    parser.add_argument(
        "--overwrite_run",
        action="store_true",
        help="Remove an existing residual-DDPM run with the same names before fresh training.",
    )
    return parser.parse_args()




def build_v_model(args, info: Dict, device: torch.device):
    """Instantiate the same Raw-MSI Direct backbone with a v-prediction target."""
    n_hsi = int(info["n_bands"])
    n_msi = int(info["n_msi_bands"])
    condition_channels = n_hsi + n_msi
    model = RawMSIDirectPredictor(
        n_bands=n_hsi,
        n_msi_bands=condition_channels,
        total_steps=args.total_diffusion_steps,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        dropout=args.dropout,
        residual_prediction=False,
        spectral_hidden=args.spectral_hidden,
    ).to(device)
    print(
        "predictor=RawMSIDirectPredictor(v-prediction), "
        f"condition_channels={condition_channels}, "
        f"params={count_parameters(model):.3f}M"
    )
    return model


def validate_v_checkpoint(checkpoint_path: str) -> Dict:
    """Reject epsilon-prediction checkpoints to prevent incompatible resumes."""
    try:
        state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    extra = state.get("extra", {}) if isinstance(state, dict) else {}
    prediction_type = extra.get("prediction_type")
    if prediction_type != "v":
        raise RuntimeError(
            f"Checkpoint prediction_type={prediction_type!r}; expected 'v'. "
            "Do not resume or evaluate an epsilon-prediction checkpoint with "
            "the v-prediction script. Start a fresh v-prediction run."
        )
    return extra


def normalize_residual(
    residual: torch.Tensor,
    residual_mean: torch.Tensor,
    residual_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize residuals and return the fraction clipped to [-1,1]."""
    normalized_raw = (residual - residual_mean) / residual_scale
    clip_fraction = (
        (normalized_raw < -1.0) | (normalized_raw > 1.0)
    ).float().mean()
    return normalized_raw.clamp(-1.0, 1.0), clip_fraction


def denormalize_residual(
    normalized_residual: torch.Tensor,
    residual_mean: torch.Tensor,
    residual_scale: torch.Tensor,
) -> torch.Tensor:
    return normalized_residual * residual_scale + residual_mean


def fit_residual_normalizer(
    train_loader,
    physical_process,
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    scale_factor: float,
    min_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Fit per-band residual mean/std using training samples only.

    Dataset augmentation is temporarily disabled so the statistics are
    deterministic. The original augmentation flag is restored afterwards.
    """
    if scale_factor <= 0:
        raise ValueError("residual_scale_factor must be > 0")
    if min_scale <= 0:
        raise ValueError("min_residual_scale must be > 0")

    dataset = train_loader.dataset
    has_augment = hasattr(dataset, "augment")
    old_augment = getattr(dataset, "augment", None)
    if has_augment:
        dataset.augment = False

    stats_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    sum_residual = None
    sum_square = None
    pixel_count = 0
    clipped_count = 0
    total_count = 0

    try:
        with torch.no_grad():
            for batch in stats_loader:
                gt = batch["gt"].to(device, non_blocking=True)
                hr_msi = batch["hr_msi"].to(device, non_blocking=True)
                _, baseline = build_condition(gt, hr_msi, physical_process)
                residual = (gt - baseline).double()
                batch_sum = residual.sum(dim=(0, 2, 3), keepdim=True)
                batch_square = residual.square().sum(
                    dim=(0, 2, 3), keepdim=True
                )
                if sum_residual is None:
                    sum_residual = batch_sum
                    sum_square = batch_square
                else:
                    sum_residual += batch_sum
                    sum_square += batch_square
                pixel_count += int(gt.shape[0] * gt.shape[2] * gt.shape[3])
    finally:
        if has_augment:
            dataset.augment = old_augment

    if sum_residual is None or pixel_count == 0:
        raise RuntimeError("Cannot fit residual normalization: empty training split")

    mean = sum_residual / float(pixel_count)
    variance = (sum_square / float(pixel_count) - mean.square()).clamp_min(0.0)
    std = variance.sqrt()
    scale = (std * float(scale_factor)).clamp_min(float(min_scale))

    # Measure clipping on the same training split after fitting its statistics.
    if has_augment:
        dataset.augment = False
    clip_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    try:
        with torch.no_grad():
            mean_device = mean.to(device=device, dtype=torch.float32)
            scale_device = scale.to(device=device, dtype=torch.float32)
            for batch in clip_loader:
                gt = batch["gt"].to(device, non_blocking=True)
                hr_msi = batch["hr_msi"].to(device, non_blocking=True)
                _, baseline = build_condition(gt, hr_msi, physical_process)
                raw = (gt - baseline - mean_device) / scale_device
                clipped_count += int(
                    ((raw < -1.0) | (raw > 1.0)).sum().item()
                )
                total_count += int(raw.numel())
    finally:
        if has_augment:
            dataset.augment = old_augment

    clip_fraction = clipped_count / max(total_count, 1)
    mean = mean.float().cpu()
    scale = scale.float().cpu()
    print(
        "Residual normalization fitted on TRAIN split only: "
        f"mean_abs_avg={mean.abs().mean().item():.6e}, "
        f"std_avg={(scale / float(scale_factor)).mean().item():.6e}, "
        f"scale_avg={scale.mean().item():.6e}, "
        f"train_clip_fraction={clip_fraction:.6%}"
    )
    return mean, scale, clip_fraction


def move_normalizer(
    residual_mean: torch.Tensor,
    residual_scale: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        residual_mean.to(device=device, dtype=torch.float32),
        residual_scale.to(device=device, dtype=torch.float32),
    )


def checkpoint_paths(args) -> Tuple[str, str]:
    root = os.path.join(args.checkpoint_root, "innovation1_residual_v_ddpm")
    ensure_dir(root)
    name = args.save_name or f"{args.dataset}_innovation1_residual_gaussian_ddpm.pth"
    if not name.endswith(".pth"):
        name += ".pth"
    stem, ext = os.path.splitext(name)
    return os.path.join(root, name), os.path.join(root, stem + "_last" + ext)


def log_path(args) -> str:
    ensure_dir(args.log_root)
    stem = os.path.splitext(os.path.basename(args.save_name))[0]
    return os.path.join(args.log_root, stem + ".csv")


def run_metadata(
    args,
    info: Dict,
    residual_mean: torch.Tensor,
    residual_scale: torch.Tensor,
) -> Dict:
    metadata = vars(args).copy()
    metadata.update(
        {
            "diffusion_type": "standard_1000_step_linear_gaussian_residual_v_ddpm",
            "diffusion_state": "HR_HSI_minus_upsampled_LR_HSI",
            "prediction_type": "v",
            "training_objective": "velocity_mean_squared_error",
            "residual_normalization": (
                f"per_band_train_split_mean_and_{args.residual_scale_factor}std_"
                "clipped_to_minus1_plus1"
            ),
            "residual_mean": residual_mean.detach().cpu(),
            "residual_scale": residual_scale.detach().cpu(),
            "beta_schedule": "linear",
            "reverse_process": "1000_step_ancestral_ddpm_v_parameterization",
            "baseline": "physical_terminal_LR_HSI_normalized_adjoint_upsampling",
            "condition": "upsampled_LR_HSI_plus_registered_HR_MSI_mapped_to_minus1_plus1",
            "n_hsi_bands": int(info["n_bands"]),
            "n_msi_bands": int(info["n_msi_bands"]),
            "condition_channels": int(info["n_bands"]) + int(info["n_msi_bands"]),
            "physical_condition_scale_ratio": int(args.scale_ratio),
        }
    )
    return metadata


def load_saved_normalizer(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    extra = state.get("extra", {}) if isinstance(state, dict) else {}
    if "residual_mean" not in extra or "residual_scale" not in extra:
        raise RuntimeError(
            "Checkpoint does not contain residual_mean/residual_scale. "
            "Use a checkpoint produced by train_innovation1_residual_ddpm.py."
        )
    mean = torch.as_tensor(extra["residual_mean"], dtype=torch.float32)
    scale = torch.as_tensor(extra["residual_scale"], dtype=torch.float32)
    if mean.ndim != 4 or scale.shape != mean.shape:
        raise RuntimeError(
            f"Invalid residual normalizer shapes: mean={tuple(mean.shape)}, "
            f"scale={tuple(scale.shape)}"
        )
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise RuntimeError("Residual normalization checkpoint contains non-finite values")
    if torch.any(scale <= 0):
        raise RuntimeError("Residual normalization scales must be positive")
    return move_normalizer(mean, scale, device)


def train_one_epoch(
    model,
    loader,
    optimizer,
    physical_process,
    schedule: GaussianDDPMSchedule,
    residual_mean: torch.Tensor,
    residual_scale: torch.Tensor,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Tuple[float, float]:
    model.train()
    loss_meter = AverageMeter()
    clip_meter = AverageMeter()

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        condition, baseline = build_condition(gt, hr_msi, physical_process)

        residual = gt - baseline
        residual_state, clip_fraction = normalize_residual(
            residual, residual_mean, residual_scale
        )

        batch_size = gt.shape[0]
        timesteps = torch.randint(
            low=1,
            high=schedule.total_steps + 1,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn_like(residual_state)
        noisy_residual = schedule.q_sample(residual_state, timesteps, noise)
        alpha_bar = schedule.alpha_bars.gather(
            0, timesteps - 1
        ).view(batch_size, 1, 1, 1).to(
            device=residual_state.device, dtype=residual_state.dtype
        )
        target_v = (
            alpha_bar.sqrt() * noise
            - (1.0 - alpha_bar).clamp_min(0.0).sqrt() * residual_state
        )
        predicted_v = model(noisy_residual, condition, timesteps)

        if predicted_v.shape != target_v.shape:
            raise RuntimeError(
                f"Predicted velocity shape {tuple(predicted_v.shape)} "
                f"does not match target velocity {tuple(target_v.shape)}"
            )
        loss = F.mse_loss(predicted_v, target_v)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite residual-DDPM velocity-MSE loss")

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
        clip_meter.update(clip_fraction.item(), batch_size)

    return loss_meter.avg, clip_meter.avg


@torch.no_grad()
def evaluate(
    model,
    loader,
    physical_process,
    schedule: GaussianDDPMSchedule,
    residual_mean: torch.Tensor,
    residual_scale: torch.Tensor,
    device: torch.device,
    *,
    scale_ratio: int,
    seed: int,
) -> Dict[str, float]:
    model.eval()
    final_meter = MetricAverager()
    initial_meter = MetricAverager()
    clip_meter = AverageMeter()

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        condition, baseline = build_condition(gt, hr_msi, physical_process)

        target_residual = gt - baseline
        _, clip_fraction = normalize_residual(
            target_residual, residual_mean, residual_scale
        )
        clip_meter.update(clip_fraction.item(), gt.shape[0])

        normalized_residual = ddpm_v_sample(
            model,
            condition,
            sample_shape=gt.shape,
            schedule=schedule,
            generator=generator,
            clip_x0=True,
        )
        predicted_residual = denormalize_residual(
            normalized_residual, residual_mean, residual_scale
        )
        prediction = (baseline + predicted_residual).clamp(0.0, 1.0)

        final_meter.update(calc_metrics(prediction, gt, scale_ratio))
        initial_meter.update(calc_metrics(baseline, gt, scale_ratio))

    metrics = final_meter.average()
    initial = initial_meter.average()
    metrics["INIT_PSNR"] = initial.get("PSNR", float("nan"))
    metrics["INIT_SAM"] = initial.get("SAM", float("nan"))
    metrics["RES_CLIP_FRAC"] = clip_meter.avg
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    keys = [
        "PSNR", "SAM", "RMSE", "ERGAS", "SSIM", "CC",
        "INIT_PSNR", "INIT_SAM", "RES_CLIP_FRAC",
    ]
    return " ".join(
        f"{key}={metrics[key]:.6f}" for key in keys if key in metrics
    )


def run_train(args, train_loader, val_loader, info, device):
    data_cfg = build_data_config(args)
    physical_process = build_progressive_process(data_cfg)
    schedule = GaussianDDPMSchedule(
        total_steps=args.total_diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    model = build_v_model(args, info, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_path, last_path = checkpoint_paths(args)
    csv_path = log_path(args)
    start_epoch = 1
    best_psnr = float("-inf")

    existing = [] if args.resume else [
        path for path in (best_path, last_path, csv_path) if os.path.exists(path)
    ]
    if existing and not args.overwrite_run:
        raise FileExistsError(
            "A residual-DDPM run already exists at:\n  "
            + "\n  ".join(existing)
            + "\nUse --resume <*_last.pth>, choose another --save_name, "
            "or pass --overwrite_run to discard this run."
        )
    if existing and args.overwrite_run:
        for path in existing:
            os.remove(path)
            print(f"OVERWRITE_RUN removed={path}")

    if args.resume:
        validate_v_checkpoint(args.resume)
        loaded_epoch, best_psnr = load_checkpoint(
            model,
            args.resume,
            optimizer=optimizer,
            map_location=str(device),
        )
        residual_mean, residual_scale = load_saved_normalizer(args.resume, device)
        start_epoch = loaded_epoch + 1
        if residual_mean.shape[1] != int(info["n_bands"]):
            raise RuntimeError("Resume checkpoint residual statistics do not match HSI bands")
        print(
            f"resumed epoch={loaded_epoch}, best_PSNR={best_psnr:.6f}; "
            "loaded residual statistics from checkpoint"
        )
    else:
        residual_mean_cpu, residual_scale_cpu, train_clip_fraction = (
            fit_residual_normalizer(
                train_loader,
                physical_process,
                device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                scale_factor=args.residual_scale_factor,
                min_scale=args.min_residual_scale,
            )
        )
        residual_mean, residual_scale = move_normalizer(
            residual_mean_cpu, residual_scale_cpu, device
        )
        print(f"fitted train residual clipping fraction={train_clip_fraction:.6%}")

    print(
        f"Gaussian residual v-DDPM: T={schedule.total_steps}, "
        f"linear beta=[{args.beta_start}, {args.beta_end}], "
        f"terminal_alpha_bar={schedule.alpha_bars[-1].item():.6e}, "
        f"ancestral reverse steps={args.total_diffusion_steps}"
    )
    print(
        f"residual normalizer: bands={residual_mean.shape[1]}, "
        f"mean_abs_avg={residual_mean.abs().mean().item():.6e}, "
        f"scale_avg={residual_scale.mean().item():.6e}"
    )
    print(
        f"baseline: physical terminal LR-HSI + normalized-adjoint lift; "
        f"condition channels={int(info['n_bands']) + int(info['n_msi_bands'])}; "
        f"parameters={count_parameters(model):.3f}M"
    )

    logger = CSVLogger(
        csv_path,
        [
            "epoch", "velocity_mse", "train_residual_clip_fraction",
            "PSNR", "SAM", "val_residual_clip_fraction", "best_PSNR",
        ],
    )
    metadata = run_metadata(args, info, residual_mean, residual_scale)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_clip_fraction = train_one_epoch(
            model,
            train_loader,
            optimizer,
            physical_process,
            schedule,
            residual_mean,
            residual_scale,
            device,
            grad_clip=args.grad_clip,
        )
        row = {
            "epoch": epoch,
            "velocity_mse": train_loss,
            "train_residual_clip_fraction": train_clip_fraction,
            "PSNR": "",
            "SAM": "",
            "val_residual_clip_fraction": "",
            "best_PSNR": best_psnr,
        }
        print(
            f"epoch={epoch:03d} velocity_mse={train_loss:.7f} "
            f"train_residual_clip_fraction={train_clip_fraction:.6%}"
        )

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            metrics = evaluate(
                model,
                val_loader,
                physical_process,
                schedule,
                residual_mean,
                residual_scale,
                device,
                scale_ratio=args.scale_ratio,
                seed=args.seed + 100000,
            )
            print("val", format_metrics(metrics))
            row["PSNR"] = metrics["PSNR"]
            row["SAM"] = metrics["SAM"]
            row["val_residual_clip_fraction"] = metrics["RES_CLIP_FRAC"]

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
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    model = build_v_model(args, info, device)
    best_path, _ = checkpoint_paths(args)
    checkpoint = args.checkpoint or args.resume or best_path
    validate_v_checkpoint(checkpoint)
    load_checkpoint(model, checkpoint, map_location=str(device))
    residual_mean, residual_scale = load_saved_normalizer(checkpoint, device)
    if residual_mean.shape[1] != int(info["n_bands"]):
        raise RuntimeError(
            f"Checkpoint has {residual_mean.shape[1]} residual bands, "
            f"but dataset provides {int(info['n_bands'])}"
        )

    metrics = evaluate(
        model,
        test_loader,
        physical_process,
        schedule,
        residual_mean,
        residual_scale,
        device,
        scale_ratio=args.scale_ratio,
        seed=args.seed + 200000,
    )
    print("test", format_metrics(metrics))


def main():
    args = parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    set_seed(args.seed)
    device = resolve_device(args.device)
    data_cfg = build_data_config(args)

    if args.stage == "train":
        from data_loader import build_train_val_test_loaders
        train_loader, val_loader, _, info = build_train_val_test_loaders(data_cfg)
        print("dataset info:", info)
        run_train(args, train_loader, val_loader, info, device)
    else:
        from data_loader import build_loaders
        _, test_loader, info = build_loaders(data_cfg)
        print("dataset info:", info)
        run_test(args, test_loader, info, device)


if __name__ == "__main__":
    main()
