"""Train/test terminal GIGI spectral refinement for Innovation 3.

The diffusion backbone is frozen.  Each training sample is reconstructed by the
complete reverse process first.  The new refiner then performs exactly one
full-resolution correction using:
    - reconstructed HR-HSI,
    - HR-MSI,
    - observable terminal physical residual,
    - MSI local heterogeneity.

This keeps the Innovation-3 intervention terminal-only, matching the earlier
physical-residual ablation where repeated in-loop injection was inferior.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from config import TrainConfig
from data_loader import build_loaders, build_train_val_test_loaders
from innovation1 import build_progressive_process, reconstruct_from_terminal_lr
from losses import SAMLoss
from metrics import MetricAverager, calc_metrics
from models import (
    HeterogeneityGuidedSpectralRefiner,
    RawMSIDirectPredictor,
    TerminalGIGISpectralRefiner,
    load_legacy_raw_direct_checkpoint,
    ranked_msi_heterogeneity,
)
from utils import CSVLogger, ensure_dir, get_device, load_checkpoint, set_seed


VARIANTS = ("conv", "gigi", "gigi_hetero", "gigi_phy", "full")


def parse_args():
    p = argparse.ArgumentParser(
        description="Innovation-3 terminal GIGI spectral-fidelity refinement"
    )
    p.add_argument("--stage", choices=["train", "test"], default="train")
    p.add_argument("--variant", choices=VARIANTS, default="full")
    p.add_argument(
        "--dataset",
        choices=["PaviaU", "Houston13", "Chikusei"],
        default="PaviaU",
    )
    p.add_argument("--data_root", default="./data/raw")
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

    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--refine_hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument(
        "--disable_tangent",
        action="store_true",
        help="Do not project the learned terminal correction to the tangent of x_hat.",
    )

    p.add_argument(
        "--baseline_checkpoint",
        default="./checkpoints/legacy/PaviaU_innovation1_physical_v3_raw_direct.pth",
    )
    p.add_argument(
        "--baseline_checkpoint_type",
        choices=["legacy", "standard"],
        default="legacy",
    )
    p.add_argument("--refiner_checkpoint", default="")
    p.add_argument("--resume", default="")

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--lambda_l1", type=float, default=1.0)
    p.add_argument("--lambda_ang", type=float, default=0.1)
    p.add_argument("--lambda_phy", type=float, default=0.1)
    p.add_argument("--lambda_msi", type=float, default=0.1)

    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--region_fraction", type=float, default=0.25)
    p.add_argument("--checkpoint_root", default="./checkpoints/innovation3_gigi")
    p.add_argument("--log_root", default="./logs")
    p.add_argument("--save_name", default="")
    return p.parse_args()


def _config(args) -> TrainConfig:
    return TrainConfig(
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
        spectral_hidden=args.spectral_hidden,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )


def _build_baseline(args, info, device):
    model = RawMSIDirectPredictor(
        n_bands=info["n_bands"],
        n_msi_bands=info["n_msi_bands"],
        total_steps=args.diffusion_steps,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        dropout=args.dropout,
        residual_prediction=True,
        spectral_hidden=args.spectral_hidden,
    ).to(device)
    if args.baseline_checkpoint_type == "legacy":
        report = load_legacy_raw_direct_checkpoint(
            model,
            args.baseline_checkpoint,
            map_location=str(device),
        )
        print("BASELINE_LOAD legacy", report)
    else:
        epoch, best = load_checkpoint(
            model,
            args.baseline_checkpoint,
            map_location=str(device),
            load_optimizer=False,
        )
        print(f"BASELINE_LOAD standard epoch={epoch} best={best}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _build_projector(args, info, device):
    """Reuse the already-validated C4:L + tangent projection for R_phy."""
    projector = HeterogeneityGuidedSpectralRefiner(
        n_bands=info["n_bands"],
        total_steps=args.diffusion_steps,
        hidden_channels=1,
        variant="full",
    ).to(device)
    projector.eval()
    for parameter in projector.parameters():
        parameter.requires_grad_(False)
    return projector


def _build_refiner(args, info, device):
    model = TerminalGIGISpectralRefiner(
        n_bands=info["n_bands"],
        n_msi_bands=info["n_msi_bands"],
        hidden_channels=args.refine_hidden,
        heads=args.heads,
        variant=args.variant,
        tangent_output=not args.disable_tangent,
    ).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"REFINER variant={args.variant} hidden={args.refine_hidden} heads={args.heads} "
        f"tangent={not args.disable_tangent} trainable={trainable/1e6:.4f}M"
    )
    return model


def _checkpoint_path(args) -> str:
    ensure_dir(args.checkpoint_root)
    name = args.save_name or f"{args.dataset}_innovation3_gigi_{args.variant}.pth"
    if not name.endswith(".pth"):
        name += ".pth"
    return os.path.join(args.checkpoint_root, name)


def _save_checkpoint(model, optimizer, epoch, best_high_sam, path, args):
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "epoch": int(epoch),
            "best_high_sam": float(best_high_sam),
            "variant": model.variant,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "extra": vars(args),
        },
        path,
    )


def _load_refiner_checkpoint(model, path, device, optimizer=None):
    if not path:
        raise ValueError("refiner checkpoint path is empty")
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    if state.get("variant", model.variant) != model.variant:
        raise ValueError(
            f"checkpoint variant={state.get('variant')} but model variant={model.variant}"
        )
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    return int(state.get("epoch", 0)), float(
        state.get("best_high_sam", float("inf"))
    )


def _hsi_to_msi(x: torch.Tensor, srf: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or srf.ndim != 2 or x.shape[1] != srf.shape[1]:
        raise ValueError("incompatible HSI/SRF shapes")
    return torch.einsum("mc,bchw->bmhw", srf, x)


@torch.no_grad()
def _physical_residual(
    process,
    projector,
    terminal_lr: torch.Tensor,
    base_pred: torch.Tensor,
) -> torch.Tensor:
    pred_terminal = process.terminal_observation(base_pred)
    native = terminal_lr - pred_terminal
    lifted = process.terminal_state(
        native,
        target_size=tuple(base_pred.shape[-2:]),
    )
    broad = projector.project_broadshape(lifted)
    return projector.project_tangent(
        broad,
        base_pred,
        preserve_broadshape=True,
    )


def _pixel_sam_deg(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    p = pred.float()
    t = target.float()
    dot = (p * t).sum(dim=1)
    pn = torch.linalg.vector_norm(p, dim=1)
    tn = torch.linalg.vector_norm(t, dim=1)
    cosine = dot / (pn * tn).clamp_min(eps)
    return torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)) * (
        180.0 / math.pi
    )


def _region_sam(pred, target, heterogeneity, fraction: float) -> Tuple[float, float]:
    sam = _pixel_sam_deg(pred, target)[:, 1:-1, 1:-1].reshape(-1)
    h = heterogeneity[:, 1:-1, 1:-1].reshape(-1)
    valid = torch.isfinite(sam) & torch.isfinite(h)
    sam = sam[valid]
    h = h[valid]
    lo = torch.quantile(h, fraction)
    hi = torch.quantile(h, 1.0 - fraction)
    return (
        float(sam[h >= hi].mean().item()),
        float(sam[h <= lo].mean().item()),
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def _mechanism_stats(details, fraction: float) -> Dict[str, float]:
    update = details["update"]
    heterogeneity = details["heterogeneity"]
    r_phy = details["physical_input"]

    energy = torch.linalg.vector_norm(update.float(), dim=1)
    e = energy[:, 1:-1, 1:-1].reshape(-1)
    h = heterogeneity[:, 1:-1, 1:-1].reshape(-1)
    lo = torch.quantile(h, fraction)
    hi = torch.quantile(h, 1.0 - fraction)
    high = float(e[h >= hi].mean().item())
    low = float(e[h <= lo].mean().item())

    uf = update.float().permute(0, 2, 3, 1).reshape(-1, update.shape[1])
    rf = r_phy.float().permute(0, 2, 3, 1).reshape(-1, r_phy.shape[1])
    denom = (
        torch.linalg.vector_norm(uf, dim=1)
        * torch.linalg.vector_norm(rf, dim=1)
    )
    valid = denom > 1e-10
    if valid.any():
        cos = (uf[valid] * rf[valid]).sum(dim=1) / denom[valid]
        update_phy_cos = float(cos.mean().item())
    else:
        update_phy_cos = 0.0

    stats = {
        "UPDATE_HIGH": high,
        "UPDATE_LOW": low,
        "UPDATE_HIGH_LOW_RATIO": high / max(low, 1e-12),
        "UPDATE_PHY_COS": update_phy_cos,
        "UPDATE_ABS_MEAN": float(update.abs().mean().item()),
    }
    attention = details.get("attention")
    if attention is not None and attention.shape[-1] > 1:
        p = attention.float().clamp_min(1e-12)
        entropy = -(p * p.log()).sum(dim=-1)
        entropy = entropy / math.log(attention.shape[-1])
        stats["ATTN_ENTROPY"] = float(entropy.mean().item())
    return stats


@torch.no_grad()
def _reconstruct_base(baseline, process, gt, hr_msi):
    terminal_lr = process.terminal_observation(gt)
    base = reconstruct_from_terminal_lr(
        baseline,
        process,
        terminal_lr,
        target_size=tuple(gt.shape[-2:]),
        hr_msi=hr_msi,
    )
    return terminal_lr, base


@torch.no_grad()
def evaluate(
    refiner,
    baseline,
    projector,
    loader,
    process,
    device,
    scale_ratio: int,
    region_fraction: float,
):
    refiner.eval()
    base_meter = MetricAverager()
    refined_meter = MetricAverager()
    base_high: List[float] = []
    base_low: List[float] = []
    refined_high: List[float] = []
    refined_low: List[float] = []
    mechanism_rows: List[Dict[str, float]] = []

    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        terminal_lr, base = _reconstruct_base(
            baseline,
            process,
            gt,
            hr_msi,
        )
        r_phy = _physical_residual(process, projector, terminal_lr, base)
        heterogeneity = ranked_msi_heterogeneity(hr_msi)

        refined, details = refiner(
            base,
            hr_msi,
            r_phy,
            heterogeneity,
            return_details=True,
        )

        base_meter.update(calc_metrics(base, gt, scale_ratio))
        refined_meter.update(calc_metrics(refined, gt, scale_ratio))
        bh, bl = _region_sam(base, gt, heterogeneity, region_fraction)
        rh, rl = _region_sam(refined, gt, heterogeneity, region_fraction)
        base_high.append(bh)
        base_low.append(bl)
        refined_high.append(rh)
        refined_low.append(rl)
        mechanism_rows.append(_mechanism_stats(details, region_fraction))

    base_metrics = base_meter.average()
    refined_metrics = refined_meter.average()
    base_metrics["SAM_HIGH"] = _mean(base_high)
    base_metrics["SAM_LOW"] = _mean(base_low)
    refined_metrics["SAM_HIGH"] = _mean(refined_high)
    refined_metrics["SAM_LOW"] = _mean(refined_low)

    mechanism: Dict[str, float] = {}
    if mechanism_rows:
        for key in mechanism_rows[0]:
            mechanism[key] = _mean(row[key] for row in mechanism_rows if key in row)
    return base_metrics, refined_metrics, mechanism


def _print_eval(base, refined, mechanism):
    print(
        "A0_BASE "
        f"PSNR={base['PSNR']:.6f} SAM={base['SAM']:.6f} "
        f"SAM_HIGH={base['SAM_HIGH']:.6f} SAM_LOW={base['SAM_LOW']:.6f}"
    )
    print(
        "REFINED "
        f"PSNR={refined['PSNR']:.6f} SAM={refined['SAM']:.6f} "
        f"SAM_HIGH={refined['SAM_HIGH']:.6f} SAM_LOW={refined['SAM_LOW']:.6f}"
    )
    print(
        "DELTA "
        f"dPSNR={refined['PSNR']-base['PSNR']:+.6f} "
        f"dSAM={refined['SAM']-base['SAM']:+.6f} "
        f"dSAM_HIGH={refined['SAM_HIGH']-base['SAM_HIGH']:+.6f} "
        f"dSAM_LOW={refined['SAM_LOW']-base['SAM_LOW']:+.6f}"
    )
    if mechanism:
        print(
            "MECHANISM "
            + " ".join(f"{key}={value:.6f}" for key, value in mechanism.items())
        )


def train(args):
    if not 0.0 < args.region_fraction < 0.5:
        raise ValueError("--region_fraction must lie in (0,0.5)")
    set_seed(args.seed)
    device = get_device(args.device)
    cfg = _config(args)
    train_loader, val_loader, val_loader, info = build_train_val_val_loaders(cfg)
    process = build_progressive_process(cfg)
    baseline = _build_baseline(args, info, device)
    projector = _build_projector(args, info, device)
    refiner = _build_refiner(args, info, device)
    srf = torch.as_tensor(info["srf_weights"], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(
        refiner.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    start_epoch = 1
    best_high_sam = float("inf")
    if args.resume:
        loaded_epoch, best_high_sam = _load_refiner_checkpoint(
            refiner,
            args.resume,
            device,
            optimizer=optimizer,
        )
        start_epoch = loaded_epoch + 1
        print(
            f"RESUME epoch={loaded_epoch} best_high_sam={best_high_sam:.6f}"
        )

    sam_loss_fn = SAMLoss()
    best_path = _checkpoint_path(args)
    stem, ext = os.path.splitext(best_path)
    last_path = stem + "_last" + ext
    logger = CSVLogger(
        os.path.join(
            args.log_root,
            f"{args.dataset}_innovation3_gigi_{args.variant}.csv",
        ),
        [
            "epoch",
            "loss",
            "l1",
            "ang",
            "phy",
            "msi",
            "PSNR",
            "SAM",
            "SAM_HIGH",
            "SAM_LOW",
            "dSAM",
            "dSAM_HIGH",
            "dSAM_LOW",
            "best_SAM_HIGH",
        ],
    )

    print(
        "TRAINING_POLICY diffusion=frozen terminal_refinement=one_shot "
        "heterogeneity_role=query_not_gain "
        "physical_residual=U_T[Y_H-D_T(X_hat)]_C4L_tangent"
    )
    print(
        "LOSS "
        f"L1={args.lambda_l1} ANG={args.lambda_ang} "
        f"PHY={args.lambda_phy} MSI={args.lambda_msi}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        refiner.train()
        loss_sum = l1_sum = ang_sum = phy_sum = msi_sum = 0.0
        count = 0

        for batch in train_loader:
            gt = batch["gt"].to(device, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)

            with torch.no_grad():
                terminal_lr, base = _reconstruct_base(
                    baseline,
                    process,
                    gt,
                    hr_msi,
                )
                r_phy = _physical_residual(
                    process,
                    projector,
                    terminal_lr,
                    base,
                )
                heterogeneity = ranked_msi_heterogeneity(hr_msi)

            refined = refiner(base, hr_msi, r_phy, heterogeneity)
            l1 = F.l1_loss(refined, gt)
            ang = sam_loss_fn(refined, gt)
            phy = F.l1_loss(
                process.terminal_observation(refined),
                terminal_lr,
            )
            msi = F.l1_loss(_hsi_to_msi(refined, srf), hr_msi)
            loss = (
                args.lambda_l1 * l1
                + args.lambda_ang * ang
                + args.lambda_phy * phy
                + args.lambda_msi * msi
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite terminal GIGI loss")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    refiner.parameters(),
                    args.grad_clip,
                    error_if_nonfinite=True,
                )
            optimizer.step()

            bs = gt.shape[0]
            loss_sum += float(loss.item()) * bs
            l1_sum += float(l1.item()) * bs
            ang_sum += float(ang.item()) * bs
            phy_sum += float(phy.item()) * bs
            msi_sum += float(msi.item()) * bs
            count += bs

        row = {
            "epoch": epoch,
            "loss": loss_sum / max(count, 1),
            "l1": l1_sum / max(count, 1),
            "ang": ang_sum / max(count, 1),
            "phy": phy_sum / max(count, 1),
            "msi": msi_sum / max(count, 1),
            "PSNR": "",
            "SAM": "",
            "SAM_HIGH": "",
            "SAM_LOW": "",
            "dSAM": "",
            "dSAM_HIGH": "",
            "dSAM_LOW": "",
            "best_SAM_HIGH": best_high_sam,
        }
        print(
            f"epoch={epoch:03d} loss={row['loss']:.7f} "
            f"l1={row['l1']:.7f} ang={row['ang']:.7f} "
            f"phy={row['phy']:.7f} msi={row['msi']:.7f}"
        )

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            base_m, refined_m, mechanism = evaluate(
                refiner,
                baseline,
                projector,
                val_loader,
                process,
                device,
                args.scale_ratio,
                args.region_fraction,
            )
            _print_eval(base_m, refined_m, mechanism)
            row.update(
                {
                    "PSNR": refined_m["PSNR"],
                    "SAM": refined_m["SAM"],
                    "SAM_HIGH": refined_m["SAM_HIGH"],
                    "SAM_LOW": refined_m["SAM_LOW"],
                    "dSAM": refined_m["SAM"] - base_m["SAM"],
                    "dSAM_HIGH": refined_m["SAM_HIGH"] - base_m["SAM_HIGH"],
                    "dSAM_LOW": refined_m["SAM_LOW"] - base_m["SAM_LOW"],
                }
            )
            if refined_m["SAM_HIGH"] < best_high_sam:
                best_high_sam = refined_m["SAM_HIGH"]
                _save_checkpoint(
                    refiner,
                    optimizer,
                    epoch,
                    best_high_sam,
                    best_path,
                    args,
                )
                print(
                    f"SAVED_BEST {best_path} SAM_HIGH={best_high_sam:.6f}"
                )
            row["best_SAM_HIGH"] = best_high_sam

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            _save_checkpoint(
                refiner,
                optimizer,
                epoch,
                best_high_sam,
                last_path,
                args,
            )
        logger.write(row)


def test(args):
    if not 0.0 < args.region_fraction < 0.5:
        raise ValueError("--region_fraction must lie in (0,0.5)")
    set_seed(args.seed)
    device = get_device(args.device)
    cfg = _config(args)
    _, test_loader, info = build_loaders(cfg)
    process = build_progressive_process(cfg)
    baseline = _build_baseline(args, info, device)
    projector = _build_projector(args, info, device)
    refiner = _build_refiner(args, info, device)

    path = args.refiner_checkpoint or _checkpoint_path(args)
    epoch, best = _load_refiner_checkpoint(refiner, path, device)
    print(
        f"REFINER_LOAD path={path} epoch={epoch} best_SAM_HIGH={best:.6f}"
    )
    base_m, refined_m, mechanism = evaluate(
        refiner,
        baseline,
        projector,
        test_loader,
        process,
        device,
        args.scale_ratio,
        args.region_fraction,
    )
    _print_eval(base_m, refined_m, mechanism)
    print(
        "FULL_METRICS "
        + " ".join(
            f"{key}={refined_m[key]:.6f}"
            for key in ("PSNR", "SSIM", "ERGAS", "SAM", "CC", "RMSE")
        )
    )


def main():
    args = parse_args()
    if args.stage == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
