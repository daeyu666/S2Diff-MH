"""Stage-2B oracle deformation-aware fine-tuning for Innovation-1 Raw-Direct.

Stage-2A showed that directly replacing D~_t with A~_(t,phi) while freezing the
registered Raw-Direct predictor causes a large state-distribution shift.  This
trainer isolates that issue by giving the diffusion process the *ground-truth*
synthetic acquisition geometry during training and evaluation.

The geometry is never estimated here.  For each batch we sample a fixed phi per
sample, build

    A_(t,phi) = D_t o W_phi,
    x_t       = A~_(t,phi)(X),

and fine-tune the existing Raw-Direct predictor to recover X from those states.
The HR-MSI remains in the reliable reference coordinate system.  LR-HSI is
never inverse-warped.

Only after this oracle path recovers strongly should the learned CDRDI estimate
replace GT phi in the end-to-end Stage-2 experiment.
"""

from __future__ import annotations

import argparse
import os
from statistics import mean
from typing import List

import torch
import torch.nn.functional as F

from cdrdi_geometry import SyntheticGeometry, sample_synthetic_geometry
from config import TrainConfig
from data_loader import build_loaders
from degradations.deformation_aware import DeformationAwareProgressiveDegradation
from innovation1 import build_progressive_process, model_predict, reconstruct_from_terminal_lr
from losses import SAMLoss
from main import build_model
from metrics import MetricAverager, calc_metrics
from models import load_legacy_raw_direct_checkpoint
from utils import CSVLogger, AverageMeter, ensure_dir, get_device, load_checkpoint, save_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Stage-2B oracle deformation-aware diffusion fine-tuning")
    p.add_argument("--dataset", choices=["PaviaU", "Houston13", "Chikusei"], default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--checkpoint_root", default="./checkpoints/cdrdi_stage2")
    p.add_argument("--log_root", default="./logs/cdrdi_stage2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)

    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--test_size", type=int, default=128)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--srf_interp", choices=["pchip", "linear"], default="pchip")
    p.add_argument("--diffusion_steps", type=int, default=12)
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)

    p.add_argument("--max_translation", type=float, default=4.0)
    p.add_argument("--max_rotation_deg", type=float, default=2.0)
    p.add_argument("--max_local_px", type=float, default=4.0)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--min_jacobian", type=float, default=0.5)
    p.add_argument("--identity_probability", type=float, default=0.10)
    p.add_argument("--min_strength", type=float, default=0.20)

    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_sam", type=float, default=0.1)
    p.add_argument("--boundary_probability", type=float, default=0.2)
    p.add_argument("--boundary_radius", type=int, default=1)
    p.add_argument("--eval_cases", type=int, default=3)
    p.add_argument("--eval_interval", type=int, default=5)

    p.add_argument("--init_checkpoint", default="")
    p.add_argument(
        "--legacy_raw_direct_checkpoint",
        default="./checkpoints/legacy/PaviaU_innovation1_physical_v3_raw_direct.pth",
    )
    p.add_argument("--resume", default="")
    p.add_argument("--save_name", default="")
    return p.parse_args()


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def _identity_geometry(batch: int, h: int, w: int, *, device, dtype, control_grid: int) -> SyntheticGeometry:
    return SyntheticGeometry(
        dx=torch.zeros(batch, device=device, dtype=dtype),
        dy=torch.zeros(batch, device=device, dtype=dtype),
        theta_deg=torch.zeros(batch, device=device, dtype=dtype),
        control=torch.zeros(batch, 2, control_grid, control_grid, device=device, dtype=dtype),
        local_field=torch.zeros(batch, 2, h, w, device=device, dtype=dtype),
    )


def _cat_geometry(items: List[SyntheticGeometry]) -> SyntheticGeometry:
    return SyntheticGeometry(
        dx=torch.cat([g.dx for g in items], dim=0),
        dy=torch.cat([g.dy for g in items], dim=0),
        theta_deg=torch.cat([g.theta_deg for g in items], dim=0),
        control=torch.cat([g.control for g in items], dim=0),
        local_field=torch.cat([g.local_field for g in items], dim=0),
    )


def sample_training_geometry(
    batch: int,
    h: int,
    w: int,
    *,
    device,
    dtype,
    generator,
    args,
) -> SyntheticGeometry:
    items: List[SyntheticGeometry] = []
    for _ in range(int(batch)):
        identity_draw = float(torch.rand((), device=device, generator=generator).item())
        if identity_draw < float(args.identity_probability):
            items.append(
                _identity_geometry(1, h, w, device=device, dtype=dtype, control_grid=args.control_grid)
            )
            continue

        phi = sample_synthetic_geometry(
            h,
            w,
            device=device,
            dtype=dtype,
            generator=generator,
            max_translation=args.max_translation,
            max_rotation_deg=args.max_rotation_deg,
            max_local_px=args.max_local_px,
            control_grid=args.control_grid,
            min_jacobian=args.min_jacobian,
        )
        alpha = float(args.min_strength) + (1.0 - float(args.min_strength)) * float(
            torch.rand((), device=device, generator=generator).item()
        )
        # Scale an already non-folding field toward identity.  This preserves
        # smoothness and cannot introduce a new fold when alpha lies in [0,1].
        phi = SyntheticGeometry(
            dx=phi.dx * alpha,
            dy=phi.dy * alpha,
            theta_deg=phi.theta_deg * alpha,
            control=phi.control * alpha,
            local_field=phi.local_field * alpha,
        )
        items.append(phi)
    return _cat_geometry(items)


def geometry_process(base_process, geometry: SyntheticGeometry) -> DeformationAwareProgressiveDegradation:
    rigid = torch.stack([geometry.dx, geometry.dy, geometry.theta_deg], dim=1)
    return DeformationAwareProgressiveDegradation(
        base_process,
        rigid=rigid,
        local_field=geometry.local_field,
    )


def batch_state_at(process, gt: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """Evaluate per-sample timesteps while keeping batch geometry paired to samples."""
    out = torch.empty_like(gt)
    for t_value in torch.unique(timesteps, sorted=True):
        # The process stores a full B-sample geometry state, so evaluate the
        # whole batch and select the samples assigned to this timestep.
        full_state = process.state_at(gt, int(t_value.item()))
        mask = timesteps == t_value
        out[mask] = full_state[mask]
    return out


def train_one_epoch(model, loader, optimizer, *, base_process, generator, args, device):
    model.train()
    sam_fn = SAMLoss()
    loss_meter = AverageMeter()
    l1_meter = AverageMeter()
    sam_meter = AverageMeter()

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        b, _, h, w = gt.shape
        geometry = sample_training_geometry(
            b,
            h,
            w,
            device=device,
            dtype=gt.dtype,
            generator=generator,
            args=args,
        )
        process = geometry_process(base_process, geometry)
        timesteps = base_process.sample_timesteps(
            b,
            boundary_probability=args.boundary_probability,
            boundary_radius=args.boundary_radius,
            device=device,
        )
        with torch.no_grad():
            x_t = batch_state_at(process, gt, timesteps)

        pred_x0 = model_predict(model, x_t, timesteps, hr_msi)
        l1 = F.l1_loss(pred_x0, gt)
        sam = sam_fn(pred_x0, gt)
        loss = float(args.lambda_l1) * l1 + float(args.lambda_sam) * sam
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Stage-2 oracle diffusion loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip), error_if_nonfinite=True)
        optimizer.step()

        loss_meter.update(float(loss.detach().item()), b)
        l1_meter.update(float(l1.detach().item()), b)
        sam_meter.update(float(sam.detach().item()), b)

    return loss_meter.avg, l1_meter.avg, sam_meter.avg


@torch.no_grad()
def evaluate(model, test_loader, *, base_process, args, device):
    model.eval()
    batch = next(iter(test_loader))
    gt = batch["gt"].to(device)
    hr_msi = batch["hr_msi"].to(device)
    if gt.shape[0] != 1:
        raise ValueError("Stage-2 oracle evaluator expects the standard single test patch")
    h, w = gt.shape[-2:]

    registered_meter = MetricAverager()
    naive_meter = MetricAverager()
    oracle_meter = MetricAverager()

    registered_lr = base_process.terminal_observation(gt)
    registered_pred = reconstruct_from_terminal_lr(
        model,
        base_process,
        registered_lr,
        target_size=(h, w),
        hr_msi=hr_msi,
    )
    registered_metrics = calc_metrics(registered_pred, gt, args.scale_ratio)

    generator = _make_generator(device, args.seed + 92000)
    for _ in range(int(args.eval_cases)):
        phi = sample_synthetic_geometry(
            h,
            w,
            device=device,
            dtype=gt.dtype,
            generator=generator,
            max_translation=args.max_translation,
            max_rotation_deg=args.max_rotation_deg,
            max_local_px=args.max_local_px,
            control_grid=args.control_grid,
            min_jacobian=args.min_jacobian,
        )
        process = geometry_process(base_process, phi)
        y_h = process.terminal_observation(gt)
        oracle_pred = reconstruct_from_terminal_lr(
            model,
            process,
            y_h,
            target_size=(h, w),
            hr_msi=hr_msi,
        )
        naive_pred = reconstruct_from_terminal_lr(
            model,
            base_process,
            y_h,
            target_size=(h, w),
            hr_msi=hr_msi,
        )
        registered_meter.update(registered_metrics)
        naive_meter.update(calc_metrics(naive_pred, gt, args.scale_ratio))
        oracle_meter.update(calc_metrics(oracle_pred, gt, args.scale_ratio))

    return {
        "registered": registered_meter.average(),
        "naive": naive_meter.average(),
        "oracle": oracle_meter.average(),
    }


def _short(metrics):
    return (
        f"PSNR={metrics['PSNR']:.4f} SAM={metrics['SAM']:.4f} "
        f"RMSE={metrics['RMSE']:.6f} SSIM={metrics['SSIM']:.6f}"
    )


def print_eval(metrics, *, prefix="EVAL"):
    reg = metrics["registered"]
    naive = metrics["naive"]
    oracle = metrics["oracle"]
    print(f"{prefix}_REGISTERED {_short(reg)}")
    print(f"{prefix}_NAIVE      {_short(naive)}")
    print(f"{prefix}_ORACLE     {_short(oracle)}")
    print(
        f"{prefix}_GAPS NAIVE_DROP={reg['PSNR']-naive['PSNR']:+.4f}dB "
        f"ORACLE_RECOVERY={oracle['PSNR']-naive['PSNR']:+.4f}dB "
        f"ORACLE_TO_REGISTERED={oracle['PSNR']-reg['PSNR']:+.4f}dB"
    )


def main():
    args = parse_args()
    if not 0.0 <= args.identity_probability <= 1.0:
        raise ValueError("--identity_probability must lie in [0,1]")
    if not 0.0 <= args.min_strength <= 1.0:
        raise ValueError("--min_strength must lie in [0,1]")
    if args.eval_cases < 1:
        raise ValueError("--eval_cases must be >=1")

    set_seed(args.seed)
    device = get_device(args.device)
    cfg = TrainConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        patch_size=args.patch_size,
        stride=args.stride,
        test_size=args.test_size,
        scale_ratio=args.scale_ratio,
        srf_interp=args.srf_interp,
        degradation_mode="physical",
        diffusion_steps=args.diffusion_steps,
        lift_mode="normalized_adjoint",
        mtf_nyquist=args.mtf_nyquist,
        psf_truncate=args.psf_truncate,
        predictor="raw_direct",
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        dropout=args.dropout,
        spectral_hidden=args.spectral_hidden,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader, test_loader, info = build_loaders(cfg)
    base_process = build_progressive_process(cfg)
    model = build_model(cfg, info, device)

    if args.resume:
        # Resume is handled after optimizer creation below.
        pass
    elif args.init_checkpoint:
        load_checkpoint(model, args.init_checkpoint, map_location=str(device), load_optimizer=False)
        print("Loaded Stage-2 init checkpoint:", args.init_checkpoint)
    else:
        report = load_legacy_raw_direct_checkpoint(
            model,
            args.legacy_raw_direct_checkpoint,
            map_location=str(device),
        )
        print("Loaded legacy Raw-Direct checkpoint:", report)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_name = args.save_name or f"{args.dataset}_oracle_deform_diffusion_local{args.max_local_px:g}_seed{args.seed}"
    ensure_dir(args.checkpoint_root)
    ensure_dir(args.log_root)
    best_path = os.path.join(args.checkpoint_root, run_name + ".pth")
    last_path = os.path.join(args.checkpoint_root, run_name + "_last.pth")
    log_path = os.path.join(args.log_root, run_name + ".csv")

    start_epoch = 1
    best_oracle_psnr = float("-inf")
    if args.resume:
        loaded_epoch, loaded_best = load_checkpoint(
            model,
            args.resume,
            optimizer=optimizer,
            map_location=str(device),
        )
        start_epoch = loaded_epoch + 1
        best_oracle_psnr = loaded_best
        print("Resumed Stage-2 oracle diffusion:", args.resume)

    logger = CSVLogger(
        log_path,
        [
            "epoch", "train_loss", "train_l1", "train_sam",
            "registered_psnr", "naive_psnr", "oracle_psnr",
            "oracle_recovery", "oracle_to_registered", "best_oracle_psnr",
        ],
    )
    train_generator = _make_generator(device, args.seed + 91000)

    initial = evaluate(model, test_loader, base_process=base_process, args=args, device=device)
    print("=" * 104)
    print("STAGE2B_INITIAL")
    print_eval(initial, prefix="INITIAL")

    for epoch in range(start_epoch, int(args.epochs) + 1):
        loss, l1, sam = train_one_epoch(
            model,
            train_loader,
            optimizer,
            base_process=base_process,
            generator=train_generator,
            args=args,
            device=device,
        )
        print(f"epoch={epoch:03d} loss={loss:.6f} l1={l1:.6f} sam={sam:.6f}")
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "train_l1": l1,
            "train_sam": sam,
            "registered_psnr": "",
            "naive_psnr": "",
            "oracle_psnr": "",
            "oracle_recovery": "",
            "oracle_to_registered": "",
            "best_oracle_psnr": best_oracle_psnr,
        }

        if epoch % int(args.eval_interval) == 0 or epoch == int(args.epochs):
            metrics = evaluate(model, test_loader, base_process=base_process, args=args, device=device)
            print_eval(metrics)
            reg_psnr = metrics["registered"]["PSNR"]
            naive_psnr = metrics["naive"]["PSNR"]
            oracle_psnr = metrics["oracle"]["PSNR"]
            row.update(
                registered_psnr=reg_psnr,
                naive_psnr=naive_psnr,
                oracle_psnr=oracle_psnr,
                oracle_recovery=oracle_psnr - naive_psnr,
                oracle_to_registered=oracle_psnr - reg_psnr,
            )
            if oracle_psnr > best_oracle_psnr:
                best_oracle_psnr = oracle_psnr
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_oracle_psnr,
                    best_path,
                    extra={"stage": "cdrdi_stage2_oracle", **vars(args)},
                )
                print(f"BEST_ORACLE epoch={epoch} PSNR={oracle_psnr:.4f} saved={best_path}")
            row["best_oracle_psnr"] = best_oracle_psnr

        save_checkpoint(
            model,
            optimizer,
            epoch,
            best_oracle_psnr,
            last_path,
            extra={"stage": "cdrdi_stage2_oracle", **vars(args)},
        )
        logger.write(row)

    print(
        "Stage-2B interpretation: a strong ORACLE recovery after fine-tuning confirms that the previous "
        "27-28 dB result was predictor state-distribution shift.  Only then should the learned CDRDI phi "
        "replace GT geometry for the estimated end-to-end path."
    )


if __name__ == "__main__":
    main()
