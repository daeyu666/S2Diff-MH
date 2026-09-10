"""Clean S2Diff-MH entry point: Innovation1 + registered Raw-MSI Direct."""

from __future__ import annotations

import os

import torch

from config import parse_args, print_config
from data_loader import build_loaders
from innovation1 import build_progressive_process, evaluate, train_one_epoch
from models import (
    CleanHSIPredictor,
    RawMSIDirectPredictor,
    SpectralSpatialCleanHSIPredictor,
    load_legacy_raw_direct_checkpoint,
)
from utils import CSVLogger, count_parameters, ensure_dir, get_device, load_checkpoint, save_checkpoint, set_seed


def build_model(cfg, info, device):
    common = dict(
        n_bands=info["n_bands"],
        total_steps=cfg.diffusion_steps,
        base_channels=cfg.base_channels,
        time_dim=cfg.time_dim,
        dropout=cfg.dropout,
        residual_prediction=True,
    )
    if cfg.predictor == "v1":
        model = CleanHSIPredictor(**common)
    elif cfg.predictor == "v2":
        model = SpectralSpatialCleanHSIPredictor(
            **common, spectral_hidden=cfg.spectral_hidden
        )
    elif cfg.predictor == "raw_direct":
        model = RawMSIDirectPredictor(
            **common,
            n_msi_bands=info["n_msi_bands"],
            spectral_hidden=cfg.spectral_hidden,
        )
    else:
        raise ValueError(cfg.predictor)
    model = model.to(device)
    print(
        f"predictor={cfg.predictor}, params={count_parameters(model):.3f}M, "
        f"requires_msi={bool(getattr(model, 'requires_msi', False))}"
    )
    return model


def checkpoint_paths(cfg):
    root = os.path.join(cfg.checkpoint_root, "innovation1")
    ensure_dir(root)
    name = cfg.save_name or f"{cfg.dataset}_{cfg.degradation_mode}_{cfg.predictor}.pth"
    if not name.endswith(".pth"):
        name += ".pth"
    stem, ext = os.path.splitext(name)
    return os.path.join(root, name), os.path.join(root, stem + "_last" + ext)


def format_metrics(metrics):
    keys = ["PSNR", "SAM", "RMSE", "ERGAS", "SSIM", "CC", "INIT_PSNR", "INIT_SAM"]
    return " ".join(f"{k}={metrics[k]:.6f}" for k in keys if k in metrics)


def _load_starting_weights(cfg, model, device):
    if cfg.legacy_raw_direct_checkpoint:
        if cfg.predictor != "raw_direct":
            raise ValueError("--legacy_raw_direct_checkpoint requires --predictor raw_direct")
        report = load_legacy_raw_direct_checkpoint(
            model, cfg.legacy_raw_direct_checkpoint, map_location=str(device)
        )
        print("Loaded legacy Raw-Direct checkpoint:", report)
        return True
    if cfg.init_checkpoint:
        load_checkpoint(
            model,
            cfg.init_checkpoint,
            strict=False,
            map_location=str(device),
            load_optimizer=False,
        )
        return True
    return False


def run_train(cfg, train_loader, test_loader, info, device):
    process = build_progressive_process(cfg)
    model = build_model(cfg, info, device)
    _load_starting_weights(cfg, model, device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    best_path, last_path = checkpoint_paths(cfg)
    start_epoch, best_psnr = 1, float("-inf")
    if cfg.resume:
        loaded_epoch, loaded_best = load_checkpoint(
            model, cfg.resume, optimizer=optimizer, map_location=str(device)
        )
        start_epoch = loaded_epoch + 1
        best_psnr = loaded_best

    transitions = process.transition_timesteps(radius=cfg.boundary_radius)
    print(
        f"process={process.operator.mode}, T={process.total_steps}, "
        f"lift={process.default_lift_mode}, transitions={transitions}"
    )
    logger = CSVLogger(
        os.path.join(cfg.log_root, f"{cfg.dataset}_{cfg.degradation_mode}_{cfg.predictor}.csv"),
        ["epoch", "loss", "l1", "sam", "deg", "PSNR", "SAM", "best_PSNR"],
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            process,
            device,
            lambda_l1=cfg.lambda_l1,
            lambda_sam=cfg.lambda_sam,
            lambda_deg=cfg.lambda_deg,
            boundary_probability=cfg.boundary_probability,
            boundary_radius=cfg.boundary_radius,
            grad_clip=cfg.grad_clip,
        )
        row = {
            "epoch": epoch,
            "loss": stats.loss,
            "l1": stats.l1,
            "sam": stats.sam,
            "deg": stats.deg,
            "PSNR": "",
            "SAM": "",
            "best_PSNR": best_psnr,
        }
        print(
            f"epoch={epoch:03d} loss={stats.loss:.6f} "
            f"l1={stats.l1:.6f} sam={stats.sam:.6f}"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            metrics = evaluate(
                model, test_loader, process, device, scale_ratio=cfg.scale_ratio
            )
            print("eval", format_metrics(metrics))
            row["PSNR"], row["SAM"] = metrics["PSNR"], metrics["SAM"]
            if metrics["PSNR"] > best_psnr:
                best_psnr = metrics["PSNR"]
                save_checkpoint(model, optimizer, epoch, best_psnr, best_path, extra=vars(cfg))
                print("saved best:", best_path)
            row["best_PSNR"] = best_psnr

        if epoch % cfg.save_interval == 0 or epoch == cfg.epochs:
            save_checkpoint(model, optimizer, epoch, best_psnr, last_path, extra=vars(cfg))
        logger.write(row)


def run_test(cfg, test_loader, info, device):
    process = build_progressive_process(cfg)
    model = build_model(cfg, info, device)
    if cfg.legacy_raw_direct_checkpoint:
        if cfg.resume:
            raise ValueError("Use either --legacy_raw_direct_checkpoint or --resume, not both")
        _load_starting_weights(cfg, model, device)
    else:
        checkpoint = cfg.resume or checkpoint_paths(cfg)[0]
        load_checkpoint(model, checkpoint, map_location=str(device))
    metrics = evaluate(model, test_loader, process, device, scale_ratio=cfg.scale_ratio)
    print(format_metrics(metrics))


def main():
    cfg = parse_args()
    print_config(cfg)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, test_loader, info = build_loaders(cfg)
    print("dataset info:", info)
    if cfg.stage == "train":
        run_train(cfg, train_loader, test_loader, info, device)
    elif cfg.stage == "test":
        run_test(cfg, test_loader, info, device)
    else:
        from diagnose_innovation1 import run_diagnosis
        run_diagnosis(cfg, test_loader, info, device)


if __name__ == "__main__":
    main()
