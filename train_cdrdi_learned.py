"""Stage-1 learned CDRDI experiment.

Compares a one-shot deformation predictor with a shared-weight recursive
physical-residual solver.  Both variants use the same network parameters and
the same fixed P0/R0 closure.  No deformation/flow GT enters the training loss
or checkpoint selection; synthetic GT geometry is retained only for evaluation
EPE after inference.
"""

from __future__ import annotations

import argparse
import os
from statistics import mean
from typing import Dict, List

import torch

from cdrdi_geometry import (
    SyntheticGeometry,
    charbonnier_mean,
    deformation_regularizer,
    forward_warp,
    jacobian_determinant,
    sample_synthetic_geometry,
    sampling_coordinates,
    spectral_project,
)
from config import TrainConfig
from data_loader import build_loaders
from degradations.physical import PhysicalDegradation
from models.cdrdi_residual_solver import LearnedPhysicalResidualSolver
from utils import (
    AverageMeter,
    CSVLogger,
    count_parameters,
    ensure_dir,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


VARIANTS = ("one_shot", "recursive")
LOSS_MODES = ("all_steps", "final_only")


def parse_args():
    p = argparse.ArgumentParser(description="Stage-1 learned physical-residual deformation solver")
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--recursive_steps", type=int, default=3)
    p.add_argument(
        "--loss_mode",
        choices=LOSS_MODES,
        default="all_steps",
        help="all_steps averages closure over every update; final_only supervises only the final unrolled state",
    )
    p.add_argument("--dataset", choices=["PaviaU", "Houston13", "Chikusei"], default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--checkpoint_root", default="./checkpoints/cdrdi_stage1")
    p.add_argument("--log_root", default="./logs/cdrdi_stage1")
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--test_size", type=int, default=128)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--srf_interp", choices=["pchip", "linear"], default="pchip")
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)

    p.add_argument("--max_translation", type=float, default=4.0)
    p.add_argument("--max_rotation_deg", type=float, default=2.0)
    p.add_argument("--max_local_px", type=float, default=4.0)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--min_jacobian", type=float, default=0.5)

    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--lambda_def", type=float, default=1e-4)
    p.add_argument("--lambda_jac", type=float, default=1e-3)
    p.add_argument("--jac_margin", type=float, default=0.1)
    p.add_argument("--eval_cases", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--device", default="cuda")
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


def _cat_geometry(items: List[SyntheticGeometry]) -> SyntheticGeometry:
    return SyntheticGeometry(
        dx=torch.cat([g.dx for g in items], dim=0),
        dy=torch.cat([g.dy for g in items], dim=0),
        theta_deg=torch.cat([g.theta_deg for g in items], dim=0),
        control=torch.cat([g.control for g in items], dim=0),
        local_field=torch.cat([g.local_field for g in items], dim=0),
    )


def sample_geometry_batch(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    max_translation: float,
    max_rotation_deg: float,
    max_local_px: float,
    control_grid: int,
    min_jacobian: float,
    randomize_local_strength: bool,
) -> SyntheticGeometry:
    items = []
    for _ in range(int(batch)):
        local_cap = float(max_local_px)
        if randomize_local_strength and max_local_px > 0.0:
            # Train across the whole deformation range rather than only near the
            # maximum amplitude used by the Stage-0 stress test.
            if float(torch.rand((), generator=generator, device=device).item()) < 0.10:
                local_cap = 0.0
            else:
                local_cap = float(
                    torch.rand((), generator=generator, device=device).item()
                ) * float(max_local_px)
        items.append(
            sample_synthetic_geometry(
                height,
                width,
                device=device,
                dtype=dtype,
                generator=generator,
                max_translation=max_translation,
                max_rotation_deg=max_rotation_deg,
                max_local_px=local_cap,
                control_grid=control_grid,
                min_jacobian=min_jacobian,
            )
        )
    return _cat_geometry(items)


def synthesize_target(
    gt_hr_hsi: torch.Tensor,
    geometry: SyntheticGeometry,
    *,
    p0,
    srf: torch.Tensor,
) -> torch.Tensor:
    """Generate Z_H=R0(P0(W_phi(X))) for the self-supervised solver input."""
    y_h = p0.degrade(
        forward_warp(
            gt_hr_hsi,
            geometry.dx,
            geometry.dy,
            geometry.theta_deg,
            geometry.local_field,
        )
    )
    return spectral_project(y_h, srf)


def _solver_steps(args) -> int:
    return 1 if args.variant == "one_shot" else int(args.recursive_steps)


def unsupervised_geometry_loss(outputs: Dict[str, object], target: torch.Tensor, args):
    predictions = outputs["predictions"]
    closures = torch.stack([charbonnier_mean(target - pred) for pred in predictions])
    if args.loss_mode == "final_only":
        closure_loss = closures[-1]
    else:
        closure_loss = closures.mean()
    final_local = outputs["final_local_field"]
    regularizer = deformation_regularizer(final_local)
    jac = jacobian_determinant(final_local)
    jac_penalty = torch.relu(float(args.jac_margin) - jac).mean()
    total = (
        closure_loss
        + float(args.lambda_def) * regularizer
        + float(args.lambda_jac) * jac_penalty
    )
    return total, closure_loss, regularizer, jac_penalty


def train_one_epoch(model, loader, optimizer, *, p0, srf, generator, args, device):
    model.train()
    loss_meter = AverageMeter()
    closure_meter = AverageMeter()
    reg_meter = AverageMeter()
    steps = _solver_steps(args)

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        with torch.no_grad():
            geometry = sample_geometry_batch(
                gt.shape[0],
                gt.shape[-2],
                gt.shape[-1],
                device=device,
                dtype=gt.dtype,
                generator=generator,
                max_translation=args.max_translation,
                max_rotation_deg=args.max_rotation_deg,
                max_local_px=args.max_local_px,
                control_grid=args.control_grid,
                min_jacobian=args.min_jacobian,
                randomize_local_strength=True,
            )
            # Geometry GT is used only to synthesize the observable target and is
            # not passed to the learned solver or any training loss.
            target = synthesize_target(gt, geometry, p0=p0, srf=srf)

        outputs = model(target, hr_msi, p0, steps=steps)
        loss, closure, reg, _ = unsupervised_geometry_loss(outputs, target, args)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Stage-1 geometry loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()

        n = gt.shape[0]
        loss_meter.update(float(loss.detach().item()), n)
        closure_meter.update(float(closure.detach().item()), n)
        reg_meter.update(float(reg.detach().item()), n)

    return {
        "loss": loss_meter.avg,
        "closure": closure_meter.avg,
        "regularizer": reg_meter.avg,
    }


def _epe_from_sampling(
    pred_x: torch.Tensor,
    pred_y: torch.Tensor,
    gt_x: torch.Tensor,
    gt_y: torch.Tensor,
):
    return torch.sqrt((pred_x - gt_x).pow(2) + (pred_y - gt_y).pow(2))


@torch.no_grad()
def evaluate(model, test_loader, *, p0, srf, args, device):
    model.eval()
    batch = next(iter(test_loader))
    gt = batch["gt"].to(device)
    hr_msi = batch["hr_msi"].to(device)
    if gt.shape[0] != 1:
        raise ValueError("Stage-1 evaluator expects the standard single test patch")

    steps = _solver_steps(args)
    generator = _make_generator(device, args.seed + 50000)
    closure_by_step = [[] for _ in range(steps + 1)]
    epe_by_step = [[] for _ in range(steps + 1)]
    p95_final = []
    reductions = []
    min_jacs = []

    h, w = gt.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=gt.dtype),
        torch.arange(w, device=device, dtype=gt.dtype),
        indexing="ij",
    )
    identity_x = xx.unsqueeze(0)
    identity_y = yy.unsqueeze(0)

    for _ in range(int(args.eval_cases)):
        geometry = sample_geometry_batch(
            1,
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
            randomize_local_strength=False,
        )
        target = synthesize_target(gt, geometry, p0=p0, srf=srf)
        outputs = model(target, hr_msi, p0, steps=steps)
        gt_x, gt_y = sampling_coordinates(
            h,
            w,
            geometry.dx,
            geometry.dy,
            geometry.theta_deg,
            geometry.local_field,
        )

        initial_closure = float(
            charbonnier_mean(target - outputs["initial_prediction"]).item()
        )
        initial_epe = _epe_from_sampling(identity_x, identity_y, gt_x, gt_y)
        closure_by_step[0].append(initial_closure)
        epe_by_step[0].append(float(initial_epe.mean().item()))

        final_epe_tensor = None
        for step_idx in range(steps):
            pred = outputs["predictions"][step_idx]
            pred_x = outputs["sampling_x"][step_idx]
            pred_y = outputs["sampling_y"][step_idx]
            closure_by_step[step_idx + 1].append(
                float(charbonnier_mean(target - pred).item())
            )
            epe_tensor = _epe_from_sampling(pred_x, pred_y, gt_x, gt_y)
            epe_by_step[step_idx + 1].append(float(epe_tensor.mean().item()))
            final_epe_tensor = epe_tensor

        final_closure = closure_by_step[-1][-1]
        reductions.append(1.0 - final_closure / max(initial_closure, 1e-12))
        p95_final.append(float(torch.quantile(final_epe_tensor.flatten(), 0.95).item()))
        min_jacs.append(float(jacobian_determinant(outputs["final_local_field"]).amin().item()))

    return {
        "step_closure": [mean(values) for values in closure_by_step],
        "step_epe": [mean(values) for values in epe_by_step],
        "final_closure": mean(closure_by_step[-1]),
        "final_epe": mean(epe_by_step[-1]),
        "final_p95": mean(p95_final),
        "closure_reduction": mean(reductions),
        "min_pred_jacobian": min(min_jacs),
    }


def print_eval(metrics, *, variant: str):
    print("-" * 96)
    for idx, (closure, epe) in enumerate(
        zip(metrics["step_closure"], metrics["step_epe"])
    ):
        label = "initial" if idx == 0 else f"update_{idx}"
        print(
            f"EVAL_STEP variant={variant} {label} "
            f"closure={closure:.8f} EPE_HR={epe:.6f}px"
        )
    print(
        f"EVAL_SUMMARY variant={variant} FINAL_EPE_HR={metrics['final_epe']:.6f}px "
        f"FINAL_P95_HR={metrics['final_p95']:.6f}px "
        f"FINAL_CLOSURE={metrics['final_closure']:.8f} "
        f"CLOSURE_REDUCTION={100.0*metrics['closure_reduction']:.3f}% "
        f"MIN_PRED_JAC={metrics['min_pred_jacobian']:.6f}"
    )


def main():
    args = parse_args()
    if args.variant == "recursive" and args.recursive_steps < 2:
        raise ValueError("recursive variant requires --recursive_steps >=2")
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader, test_loader, info = build_loaders(cfg)
    srf = torch.as_tensor(info["srf_weights"], device=device, dtype=torch.float32)
    p0 = PhysicalDegradation(
        scale_ratio=args.scale_ratio,
        mtf_nyquist=args.mtf_nyquist,
        truncate=args.psf_truncate,
    ).to(device)
    model = LearnedPhysicalResidualSolver(
        info["n_msi_bands"],
        base_channels=args.base_channels,
        control_grid=args.control_grid,
        max_translation=args.max_translation,
        max_rotation_deg=args.max_rotation_deg,
        max_local_px=args.max_local_px,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    steps = _solver_steps(args)
    run_name = args.save_name or (
        f"{args.dataset}_{args.variant}_k{steps}_{args.loss_mode}_local{args.max_local_px:g}_seed{args.seed}"
    )
    ensure_dir(args.checkpoint_root)
    ensure_dir(args.log_root)
    checkpoint_path = os.path.join(args.checkpoint_root, run_name + ".pth")
    log_path = os.path.join(args.log_root, run_name + ".csv")
    logger = CSVLogger(
        log_path,
        [
            "epoch",
            "train_loss",
            "train_closure",
            "train_regularizer",
            "val_closure",
            "val_epe_hr",
            "val_p95_hr",
            "val_closure_reduction",
            "val_min_jac",
        ],
    )

    start_epoch = 0
    best_closure = float("inf")
    if args.resume:
        start_epoch, stored_best = load_checkpoint(
            model,
            args.resume,
            optimizer=optimizer,
            strict=True,
            map_location=str(device),
        )
        if stored_best > 0.0:
            best_closure = stored_best

    print("=" * 96)
    print(
        f"CDRDI_STAGE1 variant={args.variant} steps={steps} loss_mode={args.loss_mode} "
        f"dataset={args.dataset} params={count_parameters(model):.4f}M"
    )
    print(
        "Training supervision: fixed P0/R0 physical closure only; "
        "flow GT is not used in loss or model selection."
    )
    print(f"checkpoint={checkpoint_path}")
    print("=" * 96)

    train_generator = _make_generator(device, args.seed + 10000)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            p0=p0,
            srf=srf,
            generator=train_generator,
            args=args,
            device=device,
        )
        print(
            f"epoch={epoch:03d}/{args.epochs} "
            f"loss={train_metrics['loss']:.8f} "
            f"closure={train_metrics['closure']:.8f} "
            f"reg={train_metrics['regularizer']:.8f}"
        )

        if epoch % args.eval_interval != 0 and epoch != args.epochs:
            continue
        metrics = evaluate(model, test_loader, p0=p0, srf=srf, args=args, device=device)
        print_eval(metrics, variant=args.variant)
        logger.write(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_closure": train_metrics["closure"],
                "train_regularizer": train_metrics["regularizer"],
                "val_closure": metrics["final_closure"],
                "val_epe_hr": metrics["final_epe"],
                "val_p95_hr": metrics["final_p95"],
                "val_closure_reduction": metrics["closure_reduction"],
                "val_min_jac": metrics["min_pred_jacobian"],
            }
        )

        # Keep the experiment genuinely flow-GT-free: checkpoint selection uses
        # only observable physical closure, never EPE.
        if metrics["final_closure"] < best_closure:
            best_closure = metrics["final_closure"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_closure,
                checkpoint_path,
                extra={
                    "variant": args.variant,
                    "steps": steps,
                    "loss_mode": args.loss_mode,
                    "dataset": args.dataset,
                    "max_translation": args.max_translation,
                    "max_rotation_deg": args.max_rotation_deg,
                    "max_local_px": args.max_local_px,
                    "control_grid": args.control_grid,
                },
            )
            print(
                f"BEST_BY_CLOSURE epoch={epoch} closure={best_closure:.8f} "
                f"saved={checkpoint_path}"
            )

    print("=" * 96)
    print(f"STAGE1_DONE variant={args.variant} best_observable_closure={best_closure:.8f}")


if __name__ == "__main__":
    main()
