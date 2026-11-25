"""Registered PaviaU HPGC-Refiner component ablation.

Frozen physical Raw-MSI Direct predictor is shared by every variant.
Reuse the saved Full-model experiment configuration. Train only missing
gigi_phy (w/o heterogeneity) and gigi_hetero (w/o physical residual).
Test both plus the existing Full model; derive w/o Refiner from the same
Full-model evaluation baseline. Never alter migrated checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
from diffusion_output_mode import mode_stem
from models import build_terminal_refiner


METRICS = ("PSNR", "SSIM", "ERGAS", "SAM", "CC", "RMSE")
TRAIN_VARIANTS = {
    "gigi_phy": "PaviaU_HPGC_wo_Heterogeneity",
    "gigi_hetero": "PaviaU_HPGC_wo_PhysResidual",
    "residual": "PaviaU_HPGC_Residual_Matched",
}
LABELS = {
    "gigi_phy": "w/o Heterogeneity",
    "gigi_hetero": "w/o Phys. Residual",
    "full": "Full Model",
    "residual": "Residual Refiner (matched)",
}


def parser(argv=None):
    p = argparse.ArgumentParser(description="PaviaU registered HPGC ablation")
    p.add_argument("--stage", choices=("train", "test", "all"), required=True)
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--checkpoint_dir", default="./checkpoints/PaviaU")
    p.add_argument("--ablation_checkpoint_dir", default="./checkpoints/PaviaU/ablation_hpgc")
    p.add_argument("--log_dir", default="./logs/ablation_hpgc")
    p.add_argument("--result_dir", default="./results/ablation_hpgc")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--variants", nargs="+", choices=tuple(TRAIN_VARIANTS), default=list(TRAIN_VARIANTS),
                   help="Train/test only the selected controls; Full reference is always tested")
    p.add_argument("--full_checkpoint", type=Path, help="Registered Full refiner supplying the matched training configuration")
    p.add_argument("--baseline_checkpoint", type=Path, help="Override the frozen registered diffusion path")
    p.add_argument("--diffusion_output", choices=("state", "terminal_correction"),
                   help="Default: mode recorded in Full checkpoint; mismatches are rejected")
    p.add_argument("--overwrite_run", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args(argv)


def load_full(path):
    try:
        record = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        record = torch.load(path, map_location="cpu")
    if not isinstance(record, dict) or record.get("variant", "full") != "full":
        raise ValueError("Expected a registered Full HPGC checkpoint")
    metadata = record.get("extra", {})
    if not isinstance(metadata, dict):
        raise ValueError("Full HPGC checkpoint lacks experiment metadata")
    if metadata.get("dataset", "PaviaU") != "PaviaU":
        raise ValueError("Full model was not trained on PaviaU")
    if "geometry_checkpoint" in metadata or "diffusion_checkpoint" in metadata:
        raise ValueError("Use the independently registered Full checkpoint for this registered ablation")
    return metadata


def shared_cli(cfg, args, baseline_path):
    specs = [
        ("dataset", "PaviaU"),
        ("data_root", args.data_root),
        ("device", args.device),
        ("diffusion_output", cfg.get("diffusion_output", "state")),
        ("baseline_checkpoint", baseline_path),
        ("baseline_checkpoint_type", cfg.get("baseline_checkpoint_type", "legacy")),
        ("patch_size", cfg.get("patch_size", 64)),
        ("stride", cfg.get("stride", 32)),
        ("test_size", cfg.get("test_size", 128)),
        ("scale_ratio", cfg.get("scale_ratio", 4)),
        ("srf_interp", cfg.get("srf_interp", "pchip")),
        ("diffusion_steps", cfg.get("diffusion_steps", 12)),
        ("mtf_nyquist", cfg.get("mtf_nyquist", 0.2)),
        ("psf_truncate", cfg.get("psf_truncate", 3.0)),
        ("base_channels", cfg.get("base_channels", 64)),
        ("time_dim", cfg.get("time_dim", 256)),
        ("spectral_hidden", cfg.get("spectral_hidden", 8)),
        ("dropout", cfg.get("dropout", 0.0)),
        ("refine_hidden", cfg.get("refine_hidden", 64)),
        ("heads", cfg.get("heads", 4)),
        ("epochs", cfg.get("epochs", 60)),
        ("batch_size", cfg.get("batch_size", 2)),
        ("num_workers", cfg.get("num_workers", 0)),
        ("lr", cfg.get("lr", 2e-4)),
        ("weight_decay", cfg.get("weight_decay", 0.0)),
        ("grad_clip", cfg.get("grad_clip", 1.0)),
        ("lambda_l1", cfg.get("lambda_l1", 1.0)),
        ("lambda_ang", cfg.get("lambda_ang", 0.1)),
        ("lambda_phy", cfg.get("lambda_phy", 0.1)),
        ("lambda_msi", cfg.get("lambda_msi", 0.1)),
        ("eval_interval", cfg.get("eval_interval", 5)),
        ("save_interval", cfg.get("save_interval", 10)),
        ("region_fraction", cfg.get("region_fraction", 0.25)),
        ("seed", cfg.get("seed", 10)),
    ]
    cli = []
    for key, val in specs:
        cli.extend(["--" + key, str(val)])
    if cfg.get("disable_tangent", False):
        cli.append("--disable_tangent")
    return cli


def execute(cmd, *, dry_run=False, capture=False):
    import shlex
    print("COMMAND", shlex.join(cmd), flush=True)
    if dry_run:
        return ""
    if capture:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        print(result.stdout, end="", flush=True)
        return result.stdout
    subprocess.run(cmd, check=True)
    return ""


def ablation_script():
    # Prefer the maintained entrypoint: older local ablation copies may not
    # implement residual or corrected diffusion-output routing.
    script = Path("train_innovation3_gigi.py")
    if not script.is_file():
        raise FileNotFoundError("Missing registered GIGI training script")
    return script


def check_metrics_capable(script):
    content = script.read_text(encoding="utf-8")
    if "BASE_FULL_METRICS" not in content:
        raise ValueError(
            f"{script} must print BASE_FULL_METRICS; obtain updated "
            "train_innovation3_gigi.py from origin/main"
        )


def extract_metrics(output, prefix):
    pattern = re.compile(r"([A-Za-z_]+)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")
    for line in output.splitlines():
        if line.startswith(prefix + " "):
            values = {k: float(v) for k, v in pattern.findall(line)}
            if all(metric in values for metric in METRICS):
                return {metric: values[metric] for metric in METRICS}
    raise ValueError(f"Could not find complete {prefix} in model output")


def train(args, cfg, cli, script):
    args.ablation_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    for variant in dict.fromkeys(args.variants):
        stem = TRAIN_VARIANTS[variant]
        cmd = [sys.executable, str(script), "--stage", "train",
               "--variant", variant, *cli,
               "--checkpoint_root", str(args.ablation_checkpoint_dir),
               "--log_root", str(args.log_dir),
               "--save_name", stem]
        if args.overwrite_run:
            cmd.append("--overwrite_run")
        execute(cmd, dry_run=args.dry_run)


def _parameter_counts(cfg, full_path, variants):
    state = torch.load(full_path, map_location="cpu", weights_only=False)
    payload = state["model"]
    bands = payload["hsi_encoder.net.0.weight"].shape[1]
    msi_bands = payload["msi_encoder.net.0.weight"].shape[1]
    kwargs = dict(n_bands=bands, n_msi_bands=msi_bands,
                  hidden_channels=cfg.get("refine_hidden", 64), heads=cfg.get("heads", 4),
                  tangent_output=not cfg.get("disable_tangent", False))
    counts, reports = {}, {}
    for variant in ("full", *dict.fromkeys(variants)):
        model = build_terminal_refiner(variant=variant, **kwargs)
        if variant == "full":
            model.load_state_dict(payload, strict=True)
        counts[variant] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if variant == "residual":
            reports[variant] = model.parameter_report()
    return counts, reports


def test(args, cfg, cli, script, full_path):
    check_metrics_capable(script)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    counts, parameter_reports = _parameter_counts(cfg, full_path, args.variants)
    for variant in ("full", *dict.fromkeys(args.variants)):
        checkpoint = (full_path if variant == "full"
                      else args.ablation_checkpoint_dir /
                      (mode_stem(TRAIN_VARIANTS[variant], cfg.get("diffusion_output", "state")) + ".pth"))
        if not args.dry_run and not checkpoint.is_file():
            raise FileNotFoundError(str(checkpoint))
        cmd = [sys.executable, str(script), "--stage", "test",
               "--variant", variant, *cli,
               "--refiner_checkpoint", str(checkpoint)]
        output = execute(cmd, dry_run=args.dry_run, capture=True)
        if args.dry_run:
            continue
        (args.log_dir / f"test_{variant}.log").write_text(output, encoding="utf-8")
        if variant == "full":
            rows.append({"Variant": "w/o Refiner",
                         "Params_refiner": 0,
                         **extract_metrics(output, "BASE_FULL_METRICS")})
        rows.append({"Variant": LABELS[variant],
                     "Params_refiner": counts[variant],
                     **extract_metrics(output, "FULL_METRICS")})

    if args.dry_run:
        return
    order = ("w/o Refiner", "Residual Refiner (matched)", "w/o Heterogeneity",
             "w/o Phys. Residual", "Full Model")
    csv_path = args.result_dir / (mode_stem("PaviaU_HPGC_Registered_Ablation", cfg.get("diffusion_output", "state")) + ".csv")
    saved = {}
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8-sig") as old:
            saved = {row["Variant"]: row for row in csv.DictReader(old)}
    saved.update({row["Variant"]: row for row in rows})
    rows = sorted(saved.values(), key=lambda row: order.index(row["Variant"]))
    with csv_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["Variant", "Params_refiner", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "dataset": "PaviaU", "registration": "registered",
        "refiner": "HPGC-Refiner", "results": rows,
        "full_checkpoint": str(full_path),
        "diffusion_output": cfg.get("diffusion_output", "state"),
        "configuration": cfg,
        "parameter_matching": parameter_reports,
    }, indent=2), encoding="utf-8")
    print("\n=== PAVIAU REGISTERED HPGC ABLATION ===")
    for row in rows:
        print(f"{row['Variant']:<21} " +
              " ".join(f"{metric}={float(row[metric]):.6f}" for metric in METRICS))
    print("CSV:", csv_path)
    print("JSON:", json_path)


def main(argv=None):
    a = parser(argv)
    a.checkpoint_dir = Path(a.checkpoint_dir)
    a.ablation_checkpoint_dir = Path(a.ablation_checkpoint_dir)
    a.log_dir = Path(a.log_dir)
    a.result_dir = Path(a.result_dir)

    full = a.full_checkpoint or a.checkpoint_dir / "PaviaU_innovation3_gigi_full.pth"
    if not full.is_file():
        raise FileNotFoundError(full)
    cfg = load_full(full)
    if a.diffusion_output is not None and a.diffusion_output != cfg.get("diffusion_output", "state"):
        raise ValueError("diffusion_output differs from the Full checkpoint; compare refiners trained on the same output mode")
    if a.baseline_checkpoint:
        baseline = a.baseline_checkpoint
    elif cfg.get("baseline_checkpoint") and Path(cfg["baseline_checkpoint"]).is_file():
        baseline = Path(cfg["baseline_checkpoint"])
    else:
        baseline = a.checkpoint_dir / "PaviaU_innovation1_physical_raw_direct.pth"
    if not baseline.is_file():
        raise FileNotFoundError(f"Frozen diffusion checkpoint missing: {baseline}")
    shared = shared_cli(cfg, a, str(baseline))
    script = ablation_script()
    print("HPGC ABLATION PaviaU registered; physical diffusion frozen")
    print("Full model:", full)
    print("Baseline:", baseline)
    print("Saved Full training epochs:", cfg.get("epochs", 60))
    if a.stage in ("train", "all"):
        train(a, cfg, shared, script)
    if a.stage in ("test", "all"):
        test(a, cfg, shared, script, full)


if __name__ == "__main__":
    main()
