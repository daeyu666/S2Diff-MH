"""Train/test single-step direct and generic/physical Cold reconstruction controls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from config import TrainConfig
from data_loader import build_train_val_test_loaders
from degradations.progressive import _unique_scales
from reconstruction_controls import EXPERIMENTS, PROTOCOL, build_model, build_process, evaluate, train_one_epoch
from utils import CSVLogger, get_device, set_seed

SIGNATURE_KEYS = (
    "dataset", "experiment", "base_channels", "spectral_hidden", "time_dim", "dropout",
    "diffusion_steps", "scale_ratio", "mtf_nyquist", "psf_truncate", "lift_mode",
    "gaussian_sigma", "gaussian_kernel_size", "patch_size", "stride", "test_size", "srf_interp",
    "boundary_probability", "boundary_radius", "lambda_l1", "lambda_sam", "lambda_deg",
    "batch_size", "lr", "weight_decay", "grad_clip", "seed",
)
METRICS = ("PSNR", "SAM", "RMSE", "ERGAS", "SSIM", "CC")


def parse_args(argv=None):
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    p.add_argument("--stage", choices=("train", "test"), default="train")
    p.add_argument("--dataset", choices=("PaviaU", "Houston13", "Chikusei", "CAVE", "Botswana", "Augsburg"), default="PaviaU")
    for name in ("data_root", "checkpoint_root", "log_root", "output_root", "device", "save_name", "resume"):
        p.add_argument("--" + name, default=getattr(defaults, name))
    for name in ("patch_size", "stride", "test_size", "scale_ratio", "diffusion_steps", "gaussian_kernel_size",
                 "base_channels", "spectral_hidden", "time_dim", "epochs", "batch_size", "num_workers",
                 "boundary_radius", "eval_interval", "save_interval", "seed"):
        p.add_argument("--" + name, type=int, default=getattr(defaults, name))
    for name in ("mtf_nyquist", "psf_truncate", "gaussian_sigma", "boundary_probability", "dropout", "lr",
                 "weight_decay", "grad_clip", "lambda_l1", "lambda_sam", "lambda_deg"):
        p.add_argument("--" + name, type=float, default=getattr(defaults, name))
    p.add_argument("--srf_interp", choices=("pchip", "linear"), default=defaults.srf_interp)
    p.add_argument("--lift_mode", choices=("auto", "normalized_adjoint", "bilinear", "nearest", "adjoint"), default="auto")
    p.add_argument("--diffusion_output", choices=("state", "terminal_correction"), default="terminal_correction")
    p.add_argument("--checkpoint", default="", help="Test checkpoint (default: this experiment's best)")
    p.add_argument("--early_stop_patience", type=int, default=0, help="Validation checks without improvement; 0 runs the fixed budget")
    p.add_argument("--early_stop_min_delta", type=float, default=0.)
    p.add_argument("--cpu_threads", type=int, default=1)
    p.add_argument("--overwrite", action="store_true", help="Replace an existing run of THIS control, starting from epoch 1")
    args = p.parse_args(argv)
    for name in ("patch_size", "stride", "test_size", "scale_ratio", "diffusion_steps", "base_channels",
                 "spectral_hidden", "time_dim", "epochs", "batch_size", "eval_interval", "save_interval", "cpu_threads"):
        if getattr(args, name) < 1:
            p.error(f"--{name} must be positive")
    if args.patch_size % args.scale_ratio or args.test_size % args.scale_ratio:
        p.error("Patch/test sizes must be divisible by scale_ratio")
    if args.diffusion_steps < len(_unique_scales(args.scale_ratio)):
        p.error("diffusion_steps must cover all spatial scale stages")
    if args.time_dim < 4 or args.time_dim % 2:
        p.error("time_dim must be even and >= 4")
    if args.gaussian_kernel_size < 1 or args.gaussian_kernel_size % 2 == 0:
        p.error("gaussian_kernel_size must be positive and odd")
    if not 0 < args.mtf_nyquist < 1 or args.psf_truncate <= 0:
        p.error("Require 0 < mtf_nyquist < 1 and psf_truncate > 0")
    if not 0 <= args.boundary_probability <= 1 or not 0 <= args.dropout < 1:
        p.error("Invalid boundary probability or dropout")
    if min(args.num_workers, args.boundary_radius, args.seed, args.early_stop_patience,
           args.early_stop_min_delta, args.gaussian_sigma, args.weight_decay, args.grad_clip,
           args.lambda_l1, args.lambda_sam, args.lambda_deg) < 0 or args.lr <= 0:
        p.error("Invalid negative setting or nonpositive learning rate")
    if not all(np.isfinite(value) for value in vars(args).values() if isinstance(value, float)):
        p.error("All floating-point arguments must be finite")
    if args.lambda_l1 + args.lambda_sam + args.lambda_deg <= 0:
        p.error("At least one loss weight must be positive")
    if args.resume and (args.stage != "train" or args.overwrite):
        p.error("--resume is for training and cannot be combined with --overwrite")
    if args.checkpoint and args.stage != "test":
        p.error("--checkpoint is a test-only option; training always starts from scratch unless --resume is used")
    # This flag never changes the observations for Generic Cold.
    args.degradation_mode, args.predictor = "physical", "raw_direct"
    return args


def output_mode(cfg):
    return "direct" if cfg.experiment == "single_step" else cfg.diffusion_output


def run_paths(cfg):
    name = cfg.save_name or f"{cfg.dataset}_{cfg.experiment}_{output_mode(cfg)}_seed{cfg.seed}.pth"
    if Path(name).name != name:
        raise ValueError("save_name must be a filename, not a path")
    if not name.endswith(".pth"):
        name += ".pth"
    stem = Path(name).stem
    folder = Path(cfg.checkpoint_root) / "reconstruction_controls" / cfg.dataset
    return dict(best=folder / name, last=folder / (stem + "_last.pth"),
                log=Path(cfg.log_root) / "reconstruction_controls" / (stem + ".csv"),
                output=Path(cfg.output_root) / "reconstruction_controls" / cfg.dataset / stem)


def protocol_signature(cfg, info):
    srf = np.ascontiguousarray(info["srf_weights"], dtype=np.float32)
    return dict(protocol=PROTOCOL, endpoint_policy="shared_physical_boundary_generic_interior",
                configuration={key: getattr(cfg, key) for key in SIGNATURE_KEYS},
                diffusion_output=output_mode(cfg), n_bands=int(info["n_bands"]),
                n_msi_bands=int(info["n_msi_bands"]), srf_sha256=hashlib.sha256(srf.tobytes()).hexdigest(),
                data_protocol=info.get("protocol"), train_samples=info.get("train_samples"),
                validation_samples=info.get("validation_samples"), test_samples=info.get("test_samples"))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_state(path, model, optimizer, cfg, signature, *, epoch, best_psnr, updates,
               stale, patience_reference, validation):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.pth")
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                    best_metric=best_psnr, updates=updates, stale=stale,
                    patience_reference=patience_reference, validation=validation,
                    extra=dict(configuration=vars(cfg), control_protocol=signature)), tmp)
    tmp.replace(path)


def load_state(path, signature):
    state = torch.load(path, map_location="cpu", weights_only=False)
    recorded = state.get("extra", {}).get("control_protocol")
    if recorded != signature:
        differing = [key for key in set(signature) | set(recorded or {})
                     if signature.get(key) != (recorded or {}).get(key)]
        raise ValueError(f"Control checkpoint protocol mismatch ({', '.join(sorted(differing))}): {path}")
    return state


def train(cfg, train_loader, val_loader, info, device):
    paths = run_paths(cfg)
    if not cfg.resume and not cfg.overwrite and any(paths[k].exists() for k in ("best", "last", "log")):
        raise FileExistsError("This control run already exists; use --resume, another --save_name, or --overwrite")
    if cfg.overwrite:
        for key in ("best", "last", "log"):
            paths[key].unlink(missing_ok=True)
    set_seed(cfg.seed)
    process, model = build_process(cfg), build_model(cfg, info, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    signature = protocol_signature(cfg, info)
    start_epoch, best_psnr, updates, stale = 1, float("-inf"), 0, 0
    patience_reference = float("-inf")
    if cfg.resume:
        state = load_state(cfg.resume, signature)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_psnr = int(state["epoch"]) + 1, float(state["best_metric"])
        updates, stale = int(state["updates"]), int(state["stale"])
        patience_reference = float(state["patience_reference"])
    if start_epoch > cfg.epochs:
        raise ValueError(f"Checkpoint already reached epoch {start_epoch - 1}; increase --epochs to continue")
    params = sum(p.numel() for p in model.parameters())
    write_json(paths["output"] / "protocol.json", dict(signature=signature, configuration=vars(cfg),
               params=params, training_network_calls_per_batch=1,
               inference_network_calls=1 if cfg.experiment == "single_step" else cfg.diffusion_steps,
               training_initialization="resume" if cfg.resume else "scratch", nominal_epochs=cfg.epochs))
    print(f"CONTROL experiment={cfg.experiment} output={output_mode(cfg)} params={params} "
          f"epochs={cfg.epochs} batch_size={cfg.batch_size} lr={cfg.lr} "
          "observation=physical no_CDRDI=True no_HPGC=True", flush=True)
    fields = ("epoch", "loss", "l1", "sam", "deg", "samples", "updates", "total_updates", "PSNR", "SAM", "best_PSNR", "stale_checks")
    logger = CSVLogger(str(paths["log"]), fields)
    for epoch in range(start_epoch, cfg.epochs + 1):
        # Same shuffle/augmentation/dropout stream across all controls, and at
        # a resumed epoch boundary. Time sampling uses a separate private RNG.
        set_seed(cfg.seed + epoch * 1000003)
        stats = train_one_epoch(model, train_loader, optimizer, process, cfg, device, epoch=epoch)
        updates += stats["updates"]
        validation = None
        if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
            validation = evaluate(model, val_loader, process, cfg, device)
            score = validation["PSNR"]
            if score > patience_reference + cfg.early_stop_min_delta:
                patience_reference, stale = score, 0
            else:
                stale += 1
            if score > best_psnr:
                best_psnr = score
                save_state(paths["best"], model, optimizer, cfg, signature, epoch=epoch, best_psnr=best_psnr,
                           updates=updates, stale=stale, patience_reference=patience_reference, validation=validation)
            print("VALIDATION", json.dumps(dict(epoch=epoch, **validation, best_PSNR=best_psnr)), flush=True)
        stop = bool(cfg.early_stop_patience and stale >= cfg.early_stop_patience)
        if epoch % cfg.save_interval == 0 or validation is not None or stop or epoch == cfg.epochs:
            save_state(paths["last"], model, optimizer, cfg, signature, epoch=epoch, best_psnr=best_psnr,
                       updates=updates, stale=stale, patience_reference=patience_reference, validation=validation)
        logger.write(dict(epoch=epoch, **stats, total_updates=updates,
                          PSNR=validation["PSNR"] if validation else "", SAM=validation["SAM"] if validation else "",
                          best_PSNR=best_psnr, stale_checks=stale))
        print(f"TRAIN epoch={epoch}/{cfg.epochs} loss={stats['loss']:.7f} updates={updates}", flush=True)
        if stop:
            print(f"EARLY_STOP epoch={epoch} stale_validation_checks={stale}", flush=True)
            break
    print(f"BEST {paths['best']}\nLAST {paths['last']}", flush=True)
    return paths


def test(cfg, loader, info, device):
    paths = run_paths(cfg)
    checkpoint = Path(cfg.checkpoint) if cfg.checkpoint else paths["best"]
    signature = protocol_signature(cfg, info)
    state = load_state(checkpoint, signature)
    model = build_model(cfg, info, device)
    model.load_state_dict(state["model"], strict=True)
    metrics = evaluate(model, loader, build_process(cfg), cfg, device)
    params = sum(p.numel() for p in model.parameters())
    row = dict(dataset=cfg.dataset, experiment=cfg.experiment, diffusion_output=output_mode(cfg),
               seed=cfg.seed, epoch=state["epoch"], training_updates=state["updates"], params=params,
               inference_network_calls=1 if cfg.experiment == "single_step" else cfg.diffusion_steps,
               **metrics, checkpoint=str(checkpoint.resolve()),
               checkpoint_sha256=file_sha256(checkpoint))
    write_json(paths["output"] / "test_metrics.json", dict(**row, signature=signature))
    summary = Path(cfg.output_root) / "reconstruction_controls" / "summary.csv"
    rows = {}
    key = lambda r: tuple(str(r[k]) for k in ("dataset", "experiment", "diffusion_output", "seed"))
    if summary.is_file():
        with summary.open(newline="", encoding="utf-8-sig") as handle:
            rows = {key(old): old for old in csv.DictReader(handle)}
    rows[key(row)] = row
    summary.parent.mkdir(parents=True, exist_ok=True)
    tmp = summary.with_suffix(".tmp.csv")
    with tmp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows))
    tmp.replace(summary)
    print("TEST", json.dumps(row), flush=True)
    print(f"SUMMARY {summary}", flush=True)
    return row


def main(argv=None):
    cfg = parse_args(argv)
    torch.set_num_threads(cfg.cpu_threads)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    train_loader, val_loader, test_loader, info = build_train_val_test_loaders(cfg)
    if cfg.stage == "train":
        train(cfg, train_loader, val_loader, info, device)
    else:
        test(cfg, test_loader, info, device)


if __name__ == "__main__":
    main()
