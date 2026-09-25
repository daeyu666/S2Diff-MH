"""Stage-2D estimated-geometry-aware diffusion fine-tuning.

Stage-2B adapted Raw-Direct to deformation-aware states using GT acquisition
geometry.  Stage-2C then showed that a frozen CDRDI estimate recovers most of
the non-registration loss, but an estimated-vs-oracle gap remains.

This trainer closes that train/test operator gap without flow supervision:

  1. synthesize the real HSI observation with hidden synthetic GT geometry;
  2. estimate phi_hat from fixed P0/R0 physical closure using frozen CDRDI;
  3. build the diffusion operator with phi_hat, never inverse-warping LR-HSI;
  4. fine-tune Raw-Direct on observation-anchored phi_hat trajectories.

GT geometry is used only to create the synthetic sensor observation and to
report geometry EPE.  It is never passed to CDRDI or to the diffusion model.
"""

from __future__ import annotations

import argparse
import os
from statistics import mean
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from cdrdi_geometry import (
    SyntheticGeometry,
    sample_synthetic_geometry,
    sampling_coordinates,
    spectral_project,
)
from config import TrainConfig
from data_loader import build_loaders, build_train_val_test_loaders
from degradations.deformation_aware import DeformationAwareProgressiveDegradation
from innovation1 import build_progressive_process, model_predict, reconstruct_from_terminal_lr
from losses import SAMLoss
from main import build_model
from metrics import MetricAverager, calc_metrics
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver
from train_cdrdi_diffusion_oracle import sample_training_geometry
from utils import CSVLogger, AverageMeter, ensure_dir, get_device, load_checkpoint, save_checkpoint, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Stage-2D estimated-phi-aware diffusion fine-tuning")
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

    p.add_argument("--geometry_steps", type=int, default=4)
    p.add_argument("--geometry_base_channels", type=int, default=32)
    p.add_argument(
        "--geometry_checkpoint",
        default="./checkpoints/cdrdi_stage1/PaviaU_recursive_k4_local4_seed10.pth",
    )

    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_sam", type=float, default=0.1)
    p.add_argument("--boundary_probability", type=float, default=0.2)
    p.add_argument("--boundary_radius", type=int, default=1)
    p.add_argument("--eval_cases", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=5)

    p.add_argument(
        "--init_checkpoint",
        default="./checkpoints/cdrdi_stage2/PaviaU_oracle_deform_diffusion_local4_seed10.pth",
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


def _process_from_geometry(base_process, geometry: SyntheticGeometry):
    rigid = torch.stack([geometry.dx, geometry.dy, geometry.theta_deg], dim=1)
    return DeformationAwareProgressiveDegradation(
        base_process,
        rigid=rigid,
        local_field=geometry.local_field,
    )


def _process_from_estimate(base_process, rigid: torch.Tensor, local: torch.Tensor):
    return DeformationAwareProgressiveDegradation(
        base_process,
        rigid=rigid.detach(),
        local_field=local.detach(),
    )


def _estimate_geometry(
    geometry_model,
    *,
    y_h: torch.Tensor,
    hr_msi: torch.Tensor,
    p0,
    srf: torch.Tensor,
    steps: int,
):
    z_h = spectral_project(y_h, srf)
    out = geometry_model(z_h, hr_msi, p0, steps=steps)
    return out["final_rigid"].detach(), out["final_local_field"].detach()


def observation_anchored_batch_state(
    process: DeformationAwareProgressiveDegradation,
    clean_x: torch.Tensor,
    observed_terminal: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Teacher-forced state on the *observed* estimated-phi reverse trajectory.

    With an imperfect phi_hat, the actual terminal lift is

        x_T^obs = A_hat_T^dagger Y_H,

    whereas the model-consistent terminal state is A~_hat_T(X).  Their residual
    must not be silently dropped during training.  Under exact clean-X reverse
    increments, that terminal residual is carried through every t:

        x_t^obs = x_T^obs + A~_hat_t(X) - A~_hat_T(X).

    This matches the inference initialization while keeping the training target
    X in the reliable MSI coordinate system.
    """
    if timesteps.ndim != 1 or timesteps.shape[0] != clean_x.shape[0]:
        raise ValueError("timesteps must have shape [B]")
    target_size = tuple(clean_x.shape[-2:])
    with torch.no_grad():
        x_terminal_obs = process.terminal_state(observed_terminal, target_size=target_size)
        x_terminal_model = process.state_at(clean_x, process.total_steps)
        terminal_residual = x_terminal_obs - x_terminal_model
        out = torch.empty_like(clean_x)
        for t_value in torch.unique(timesteps, sorted=True):
            mask = timesteps == t_value
            model_state = process.state_at(clean_x, int(t_value.item()))
            out[mask] = (model_state + terminal_residual)[mask]
    return out


def _geometry_epe(
    h: int,
    w: int,
    gt_phi: SyntheticGeometry,
    pred_rigid: torch.Tensor,
    pred_local: torch.Tensor,
) -> float:
    gt_x, gt_y = sampling_coordinates(
        h, w, gt_phi.dx, gt_phi.dy, gt_phi.theta_deg, gt_phi.local_field
    )
    pred_x, pred_y = sampling_coordinates(
        h,
        w,
        pred_rigid[:, 0],
        pred_rigid[:, 1],
        pred_rigid[:, 2],
        pred_local,
    )
    epe = torch.sqrt((pred_x - gt_x).pow(2) + (pred_y - gt_y).pow(2))
    return float(epe.mean().item())


def train_one_epoch(
    model,
    geometry_model,
    loader,
    optimizer,
    *,
    base_process,
    p0,
    srf,
    generator,
    args,
    device,
):
    model.train()
    geometry_model.eval()
    sam_fn = SAMLoss()
    loss_meter, l1_meter, sam_meter = (AverageMeter() for _ in range(3))

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        b, _, h, w = gt.shape

        with torch.no_grad():
            gt_phi = sample_training_geometry(
                b,
                h,
                w,
                device=device,
                dtype=gt.dtype,
                generator=generator,
                args=args,
            )
            true_process = _process_from_geometry(base_process, gt_phi)
            y_h = true_process.terminal_observation(gt)
            pred_rigid, pred_local = _estimate_geometry(
                geometry_model,
                y_h=y_h,
                hr_msi=hr_msi,
                p0=p0,
                srf=srf,
                steps=args.geometry_steps,
            )
            estimated_process = _process_from_estimate(base_process, pred_rigid, pred_local)
            timesteps = base_process.sample_timesteps(
                b,
                boundary_probability=args.boundary_probability,
                boundary_radius=args.boundary_radius,
                device=device,
            )
            x_t = observation_anchored_batch_state(
                estimated_process, gt, y_h, timesteps
            )

        pred_x0 = model_predict(model, x_t, timesteps, hr_msi)
        l1 = F.l1_loss(pred_x0, gt)
        sam = sam_fn(pred_x0, gt)
        loss = float(args.lambda_l1) * l1 + float(args.lambda_sam) * sam
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Stage-2D estimated-phi diffusion loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(args.grad_clip), error_if_nonfinite=True
            )
        optimizer.step()

        loss_meter.update(float(loss.detach().item()), b)
        l1_meter.update(float(l1.detach().item()), b)
        sam_meter.update(float(sam.detach().item()), b)

    return loss_meter.avg, l1_meter.avg, sam_meter.avg


@torch.no_grad()
def evaluate(model, geometry_model, test_loader, *, base_process, p0, srf, args, device):
    model.eval()
    geometry_model.eval()
    batch = next(iter(test_loader))
    gt = batch["gt"].to(device)
    hr_msi = batch["hr_msi"].to(device)
    if gt.shape[0] != 1:
        raise ValueError("Stage-2D evaluator expects the standard single test patch")
    h, w = gt.shape[-2:]

    meters = {name: MetricAverager() for name in ("registered", "naive", "oracle", "estimated")}
    epes: List[float] = []
    generator = _make_generator(device, args.seed + 94000)

    registered_lr = base_process.terminal_observation(gt)
    registered_pred = reconstruct_from_terminal_lr(
        model, base_process, registered_lr, target_size=(h, w), hr_msi=hr_msi
    )
    registered_metrics = calc_metrics(registered_pred, gt, args.scale_ratio)

    for _ in range(int(args.eval_cases)):
        gt_phi = sample_synthetic_geometry(
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
        true_process = _process_from_geometry(base_process, gt_phi)
        y_h = true_process.terminal_observation(gt)
        pred_rigid, pred_local = _estimate_geometry(
            geometry_model,
            y_h=y_h,
            hr_msi=hr_msi,
            p0=p0,
            srf=srf,
            steps=args.geometry_steps,
        )
        estimated_process = _process_from_estimate(base_process, pred_rigid, pred_local)
        epes.append(_geometry_epe(h, w, gt_phi, pred_rigid, pred_local))

        naive_pred = reconstruct_from_terminal_lr(
            model, base_process, y_h, target_size=(h, w), hr_msi=hr_msi
        )
        oracle_pred = reconstruct_from_terminal_lr(
            model, true_process, y_h, target_size=(h, w), hr_msi=hr_msi
        )
        estimated_pred = reconstruct_from_terminal_lr(
            model, estimated_process, y_h, target_size=(h, w), hr_msi=hr_msi
        )
        meters["registered"].update(registered_metrics)
        meters["naive"].update(calc_metrics(naive_pred, gt, args.scale_ratio))
        meters["oracle"].update(calc_metrics(oracle_pred, gt, args.scale_ratio))
        meters["estimated"].update(calc_metrics(estimated_pred, gt, args.scale_ratio))

    return {name: meter.average() for name, meter in meters.items()}, mean(epes)


def _short(m: Dict[str, float]) -> str:
    return (
        f"PSNR={m['PSNR']:.4f} SAM={m['SAM']:.4f} "
        f"RMSE={m['RMSE']:.6f} SSIM={m['SSIM']:.6f}"
    )


def print_eval(metrics, geom_epe: float, *, prefix: str):
    for name in ("registered", "naive", "oracle", "estimated"):
        print(f"{prefix}_{name.upper():>10s} {_short(metrics[name])}")
    reg = metrics["registered"]["PSNR"]
    naive = metrics["naive"]["PSNR"]
    oracle = metrics["oracle"]["PSNR"]
    est = metrics["estimated"]["PSNR"]
    print(
        f"{prefix}_GEOMETRY AVG_EPE_HR={geom_epe:.6f}px"
    )
    print(
        f"{prefix}_GAPS NAIVE_DROP={reg-naive:+.4f}dB "
        f"ORACLE_RECOVERY={oracle-naive:+.4f}dB "
        f"EST_RECOVERY={est-naive:+.4f}dB "
        f"EST_TO_ORACLE={est-oracle:+.4f}dB "
        f"EST_TO_REGISTERED={est-reg:+.4f}dB"
    )


def main():
    args = parse_args()
    if args.geometry_steps < 1:
        raise ValueError("--geometry_steps must be >=1")
    if args.eval_cases < 1:
        raise ValueError("--eval_cases must be >=1")
    if not 0.0 <= args.identity_probability <= 1.0:
        raise ValueError("--identity_probability must lie in [0,1]")
    if not 0.0 <= args.min_strength <= 1.0:
        raise ValueError("--min_strength must lie in [0,1]")

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
    train_loader, val_loader, val_loader, info = build_train_val_val_loaders(cfg)
    base_process = build_progressive_process(cfg)
    p0 = base_process.operator
    srf = torch.as_tensor(info["srf_weights"], device=device, dtype=torch.float32)

    geometry_model = LearnedPhysicalResidualSolver(
        info["n_msi_bands"],
        base_channels=args.geometry_base_channels,
        control_grid=args.control_grid,
        max_translation=args.max_translation,
        max_rotation_deg=args.max_rotation_deg,
        max_local_px=args.max_local_px,
    ).to(device)
    load_checkpoint(
        geometry_model,
        args.geometry_checkpoint,
        map_location=str(device),
        load_optimizer=False,
    )
    geometry_model.eval()
    for parameter in geometry_model.parameters():
        parameter.requires_grad_(False)

    model = build_model(cfg, info, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name = args.save_name or f"{args.dataset}_estimated_deform_diffusion_k{args.geometry_steps}_local{args.max_local_px:g}_seed{args.seed}"
    ensure_dir(args.checkpoint_root)
    ensure_dir(args.log_root)
    best_path = os.path.join(args.checkpoint_root, run_name + ".pth")
    last_path = os.path.join(args.checkpoint_root, run_name + "_last.pth")
    log_path = os.path.join(args.log_root, run_name + ".csv")

    start_epoch = 1
    best_est_psnr = float("-inf")
    if args.resume:
        epoch, best_est_psnr = load_checkpoint(
            model, args.resume, optimizer=optimizer, map_location=str(device)
        )
        start_epoch = epoch + 1
        print("Resumed diffusion checkpoint:", args.resume)
    else:
        load_checkpoint(
            model,
            args.init_checkpoint,
            map_location=str(device),
            load_optimizer=False,
        )
        print("Loaded Stage-2B oracle-adapted diffusion checkpoint:", args.init_checkpoint)

    logger = CSVLogger(
        log_path,
        ["epoch", "loss", "l1", "sam", "registered_psnr", "naive_psnr", "oracle_psnr", "estimated_psnr", "geom_epe", "best_estimated_psnr"],
    )

    initial_metrics, initial_epe = evaluate(
        model,
        geometry_model,
        val_loader,
        base_process=base_process,
        p0=p0,
        srf=srf,
        args=args,
        device=device,
    )
    print_eval(initial_metrics, initial_epe, prefix="INITIAL")

    generator = _make_generator(device, args.seed + 93000)
    for epoch in range(start_epoch, args.epochs + 1):
        loss, l1, sam = train_one_epoch(
            model,
            geometry_model,
            train_loader,
            optimizer,
            base_process=base_process,
            p0=p0,
            srf=srf,
            generator=generator,
            args=args,
            device=device,
        )
        print(f"epoch={epoch:03d} loss={loss:.6f} l1={l1:.6f} sam={sam:.6f}")
        row = {
            "epoch": epoch,
            "loss": loss,
            "l1": l1,
            "sam": sam,
            "registered_psnr": "",
            "naive_psnr": "",
            "oracle_psnr": "",
            "estimated_psnr": "",
            "geom_epe": "",
            "best_estimated_psnr": best_est_psnr,
        }

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            metrics, geom_epe = evaluate(
                model,
                geometry_model,
                val_loader,
                base_process=base_process,
                p0=p0,
                srf=srf,
                args=args,
                device=device,
            )
            print_eval(metrics, geom_epe, prefix="EVAL")
            row.update(
                registered_psnr=metrics["registered"]["PSNR"],
                naive_psnr=metrics["naive"]["PSNR"],
                oracle_psnr=metrics["oracle"]["PSNR"],
                estimated_psnr=metrics["estimated"]["PSNR"],
                geom_epe=geom_epe,
            )
            if metrics["estimated"]["PSNR"] > best_est_psnr:
                best_est_psnr = metrics["estimated"]["PSNR"]
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    best_est_psnr,
                    best_path,
                    extra=vars(args),
                )
                print(f"BEST_ESTIMATED epoch={epoch} PSNR={best_est_psnr:.4f} saved={best_path}")
            row["best_estimated_psnr"] = best_est_psnr

        save_checkpoint(
            model,
            optimizer,
            epoch,
            best_est_psnr,
            last_path,
            extra=vars(args),
        )
        logger.write(row)

    print("=" * 104)
    print(f"Stage-2D complete. best estimated-phi PSNR={best_est_psnr:.4f} checkpoint={best_path}")


if __name__ == "__main__":
    main()
