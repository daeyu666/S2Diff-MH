"""Paired registered-backbone / estimated-geometry / oracle-geometry diagnostic.

No training. All four paths see the SAME observation for every patch/case.
Only the explicitly labelled oracle path receives synthetic geometry.
Imports are lazy so --help and --dry_run need no data, checkpoints or torch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import mean


PATHS = ("registered_backbone", "adapted_naive", "estimated", "oracle")
METRICS = ("PSNR", "SSIM", "ERGAS", "SAM", "CC", "RMSE")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["PaviaU", "Houston13", "Chikusei", "CAVE", "Botswana", "Augsburg"], default="CAVE")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--protocol", choices=["random", "fixed", "both"], default="random")
    p.add_argument("--cases", "--eval_cases", dest="cases", type=int, default=10)
    p.add_argument("--levels", nargs="+", type=int, choices=range(1, 7), default=[1, 2, 3, 4, 5])
    p.add_argument("--registered_checkpoint", default="", help="Registered Innovation-1 diffusion weights; never replaced by adapted weights")
    p.add_argument("--registered_checkpoint_type", choices=["auto", "standard", "legacy"], default="auto")
    p.add_argument("--geometry_checkpoint", default="")
    p.add_argument("--diffusion_checkpoint", default="", help="Nonregistered-adapted diffusion weights")
    p.add_argument("--refiner_checkpoint", default="", help="Also used to resolve recorded frozen checkpoint paths")
    p.add_argument("--init_refiner_checkpoint", default="", help="Registered GIGI checkpoint: its recorded baseline_checkpoint may resolve registered weights")
    p.add_argument("--refiner_mode", choices=["off", "shared"], default="off",
                   help="Always report base; shared additionally applies the SAME frozen GIGI to all paths")
    p.add_argument("--variant", choices=["conv", "gigi", "gigi_hetero", "gigi_phy", "full"], default="full")
    p.add_argument("--geometry_steps", type=int, default=9)
    p.add_argument("--geometry_base_channels", type=int, default=32)
    p.add_argument("--diffusion_steps", type=int, default=12)
    p.add_argument("--diffusion_base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=256)
    p.add_argument("--spectral_hidden", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--refine_hidden", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--disable_tangent", action="store_true")
    p.add_argument("--test_size", type=int, default=128)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--srf_interp", choices=["pchip", "linear"], default="pchip")
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)
    p.add_argument("--max_translation", type=float, default=4.0, help="Training-time model bound AND legacy random-test bound")
    p.add_argument("--max_rotation_deg", type=float, default=2.0)
    p.add_argument("--max_local_px", type=float, default=4.0)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--min_jacobian", type=float, default=0.5)
    p.add_argument("--max_samples", type=int, default=0, help="0=entire test split; positive values are explicitly labelled smoke tests")
    p.add_argument("--output_dir", default="")
    p.add_argument("--overwrite_output", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    # Attributes consumed by the existing pipeline's config/checkpoint resolver.
    p.set_defaults(stage="test", batch_size=1, num_workers=0,
                   checkpoint_root="./checkpoints/innovation3_gigi_cdrdi", save_name="")
    args = p.parse_args(argv)
    if min(args.cases, args.geometry_steps, args.diffusion_steps) < 1 or args.max_samples < 0:
        p.error("cases/steps must be positive; max_samples must be >=0")
    if len(set(args.levels)) != len(args.levels):
        p.error("levels must be distinct")
    if min(args.max_translation, args.max_rotation_deg, args.max_local_px) < 0:
        p.error("geometry bounds must be nonnegative")
    if args.control_grid < 2 or not 0 < args.min_jacobian <= 1:
        p.error("control_grid must be >=2 and min_jacobian must be in (0,1]")
    if args.protocol in ("fixed", "both"):
        if (args.test_size, args.patch_size, args.stride, args.scale_ratio,
                args.mtf_nyquist, args.psf_truncate) != (128, 64, 32, 4, 0.2, 3.0):
            p.error("Fixed comparison requires test=128, patch=64, stride=32, scale=4, MTF=0.2, truncate=3")
        if (args.max_translation, args.max_rotation_deg, args.max_local_px) != (4, 2, 4):
            p.error("Fixed-suite model bounds must stay at training-time 4/2/4")
    if not args.output_dir:
        suffix = f"{args.protocol}_{args.refiner_mode}_seed{args.seed}_cases{args.cases}"
        if args.max_samples:
            suffix += f"_SMOKE{args.max_samples}"
        args.output_dir = str(Path("outputs/cdrdi_robustness") / args.dataset / suffix)
    return args


def comparison_rows(rows):
    """Signed gains; positive PSNR/SAM gains mean the named path is better."""
    indexed = {(r["condition"], r["stage"], r["path"]): r for r in rows}
    out = []
    for condition, stage in dict.fromkeys((r["condition"], r["stage"]) for r in rows):
        pairs = (("adaptation_gain", "adapted_naive", "registered_backbone"),
                 ("geometry_gain", "estimated", "adapted_naive"),
                 ("total_gain", "estimated", "registered_backbone"),
                 ("oracle_gap", "oracle", "estimated"))
        for name, better, reference in pairs:
            a, b = indexed[condition, stage, better], indexed[condition, stage, reference]
            out.append({"condition": condition, "stage": stage, "comparison": name,
                        "PSNR_gain_db": a["PSNR"] - b["PSNR"],
                        "SAM_reduction_deg": b["SAM"] - a["SAM"]})
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_checkpoint(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_record(path, dataset):
    state = read_checkpoint(path)
    extra = state.get("extra", {}) if isinstance(state, dict) else {}
    if extra.get("dataset") not in (None, dataset):
        raise ValueError(f"Checkpoint dataset mismatch: {path}: {extra['dataset']} != {dataset}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(Path(path).resolve()), "sha256": digest.hexdigest(),
            "epoch": state.get("epoch") if isinstance(state, dict) else None,
            "extra": {k: str(v) for k, v in extra.items()}}


def build_models(args, info, process, device):
    import torch
    import train_innovation3_gigi_cdrdi as pipeline
    from models.predictor_raw_direct import extract_legacy_raw_direct_state_dict

    args = pipeline._resolve_dataset_checkpoints(args)
    if not args.registered_checkpoint and Path(args.init_refiner_checkpoint).is_file():
        state = read_checkpoint(args.init_refiner_checkpoint)
        args.registered_checkpoint = state.get("extra", {}).get("baseline_checkpoint", "")
    if not args.registered_checkpoint:
        raise ValueError("Pass --registered_checkpoint with the actual registered Innovation-1 diffusion weights; no adapted-weight fallback is allowed")
    roles = {"registered_backbone": args.registered_checkpoint,
             "adapted_diffusion": args.diffusion_checkpoint, "geometry": args.geometry_checkpoint}
    if args.refiner_mode == "shared":
        args.refiner_checkpoint = args.refiner_checkpoint or pipeline._checkpoint_path(args)
        roles["shared_refiner"] = args.refiner_checkpoint
    records = {role: checkpoint_record(path, args.dataset) for role, path in roles.items()}
    if records["registered_backbone"]["sha256"] == records["adapted_diffusion"]["sha256"]:
        raise ValueError("Registered and adapted checkpoints are identical; use the pre-adaptation registered weights")

    geometry_model, adapted = pipeline._build_frozen_models(args, info, device)
    registered = pipeline.build_model(pipeline._config(args), info, device)
    state = read_checkpoint(args.registered_checkpoint)
    if args.registered_checkpoint_type == "standard":
        weights = state.get("model", state)
    else:
        weights, ignored = extract_legacy_raw_direct_state_dict(state)
        records["registered_backbone"]["ignored_legacy_keys"] = sorted(ignored)
    # Strict loading after the known legacy-key conversion; never keep random weights.
    registered.load_state_dict(weights, strict=True)
    registered.eval()
    registered.requires_grad_(False)
    refiner = projector = None
    if args.refiner_mode == "shared":
        refiner = pipeline._build_refiner(args, info, device)
        pipeline._load_training_checkpoint(refiner, args.refiner_checkpoint, device)
        refiner.eval().requires_grad_(False)
        projector = pipeline._build_projector(args, info, device)
    srf = torch.as_tensor(info["srf_weights"], device=device, dtype=torch.float32)

    @torch.no_grad()
    def predict(observed, msi):
        """Production paths receive observations only, never synthetic GT."""
        rigid, local = pipeline._estimate_geometry(
            geometry_model, y_h=observed, hr_msi=msi, p0=process.operator,
            srf=srf, steps=args.geometry_steps)
        estimated_process = pipeline._process_from_estimate(process, rigid, local)
        outputs = {}
        heterogeneity = pipeline.ranked_msi_heterogeneity(msi) if refiner is not None else None

        def reconstruct(model, acquisition):
            base = pipeline.reconstruct_from_terminal_lr(
                model, acquisition, observed, target_size=tuple(msi.shape[-2:]), hr_msi=msi)
            result = {"base": base}
            if refiner is not None:
                residual = pipeline._physical_residual(acquisition, projector, observed, base)
                result["refined"] = refiner(base, msi, residual, heterogeneity)
            return result

        outputs["registered_backbone"] = reconstruct(registered, process)
        outputs["adapted_naive"] = reconstruct(adapted, process)
        outputs["estimated"] = reconstruct(adapted, estimated_process)
        return outputs, rigid, local

    @torch.no_grad()
    def oracle_predict(observed, msi, truth):
        """GT geometry is injected ONLY here, retaining adapted weights."""
        acquisition = pipeline._process_from_geometry(process, truth)
        base = pipeline.reconstruct_from_terminal_lr(
            adapted, acquisition, observed, target_size=tuple(msi.shape[-2:]), hr_msi=msi)
        result = {"base": base}
        if refiner is not None:
            residual = pipeline._physical_residual(acquisition, projector, observed, base)
            result["refined"] = refiner(base, msi, residual, pipeline.ranked_msi_heterogeneity(msi))
        return result

    return predict, oracle_predict, srf, records


def geometry_diagnostics(truth, rigid, local, msi, observed, process, srf):
    import torch
    from cdrdi_geometry import forward_warp, sampling_coordinates, spectral_project
    h, w = msi.shape[-2:]
    tx, ty = sampling_coordinates(h, w, truth.dx, truth.dy, truth.theta_deg, truth.local_field)
    px, py = sampling_coordinates(h, w, rigid[:, 0], rigid[:, 1], rigid[:, 2], local)
    yy, xx = torch.meshgrid(torch.arange(h, device=msi.device, dtype=msi.dtype),
                            torch.arange(w, device=msi.device, dtype=msi.dtype), indexing="ij")
    displacement = torch.sqrt((tx - xx).square() + (ty - yy).square())
    epe = torch.sqrt((tx - px).square() + (ty - py).square())
    target = spectral_project(observed, srf)
    pred = process.operator.degrade(forward_warp(msi, rigid[:, 0], rigid[:, 1], rigid[:, 2], local))
    true_pred = process.operator.degrade(forward_warp(msi, truth.dx, truth.dy, truth.theta_deg, truth.local_field))
    return {"EPE_hr_mean": epe.mean().item(), "EPE_hr_p95": torch.quantile(epe.flatten(), 0.95).item(),
            "displacement_hr_mean": displacement.mean().item(),
            "displacement_hr_p95": torch.quantile(displacement.flatten(), 0.95).item(),
            "displacement_hr_max": displacement.max().item(),
            "local_hr_max": torch.linalg.vector_norm(truth.local_field, dim=1).max().item(),
            "closure_naive_mse": (process.operator.degrade(msi) - target).square().mean().item(),
            "closure_estimated_mse": (pred - target).square().mean().item(),
            "closure_oracle_mse": (true_pred - target).square().mean().item()}


def run_diagnostic(args, test_set, process, predict, oracle_predict, srf, device, metadata):
    import torch
    import train_innovation3_gigi_cdrdi as pipeline
    from cdrdi_geometry import SyntheticGeometry, sample_synthetic_geometry
    from fixed_deformation_protocol import STRENGTHS, sample_fixed_geometry, tensor_sha256

    count = min(len(test_set), args.max_samples) if args.max_samples else len(test_set)
    if count < 1:
        raise ValueError("Test split is empty")
    conditions = ["registered"]
    if args.protocol in ("random", "both"):
        conditions.append("random")
    if args.protocol in ("fixed", "both"):
        conditions += [f"fixed_{level}" for level in args.levels]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, geometry_rows, reference_inputs = [], [], None
    # Save resolved protocol before expensive inference, including incomplete-run state.
    manifest = {"complete": False, "protocol": metadata, "args": vars(args),
                "conditions": conditions, "test_patch_count": count,
                "full_test_patch_count": len(test_set), "smoke_test": bool(args.max_samples),
                "summary": rows, "comparisons": [], "geometry": geometry_rows}
    write_json(output / "summary.json", manifest)
    for condition in conditions:
        cases = 1 if condition == "registered" else args.cases
        buckets, geometry_patches, inputs = {}, [], []
        with (output / f"{condition}.jsonl").open("w", encoding="utf-8") as details:
            for sample in range(count):
                item = test_set[sample]
                gt = item["gt"].unsqueeze(0).to(device=device, dtype=torch.float32)
                msi = item["hr_msi"].unsqueeze(0).to(device=device, dtype=torch.float32)
                h, w = gt.shape[-2:]
                fingerprint = {"sample": sample + 1, "gt_sha256": tensor_sha256(gt),
                               "msi_sha256": tensor_sha256(msi), "shape": list(gt.shape)}
                inputs.append(fingerprint)
                if reference_inputs is not None and fingerprint != reference_inputs[sample]:
                    raise RuntimeError("Test inputs/order changed between conditions")
                # EXACT legacy random-test seed reset for each patch. There is NO
                # training alpha or identity mixture in this evaluation sampler.
                generator = pipeline._make_generator(device, args.seed + 70000)
                patch_metrics, patch_geometry = {}, []
                for case in range(cases):
                    if condition == "registered":
                        zero = gt.new_zeros(1)
                        truth = SyntheticGeometry(zero, zero.clone(), zero.clone(),
                            gt.new_zeros(1, 2, args.control_grid, args.control_grid), gt.new_zeros(1, 2, h, w))
                        observed = process.terminal_observation(gt)
                    elif condition == "random":
                        truth = sample_synthetic_geometry(h, w, device=device, dtype=gt.dtype,
                            generator=generator, max_translation=args.max_translation,
                            max_rotation_deg=args.max_rotation_deg, max_local_px=args.max_local_px,
                            control_grid=args.control_grid, min_jacobian=args.min_jacobian)
                        observed = pipeline._process_from_geometry(process, truth).terminal_observation(gt)
                    else:
                        strength = STRENGTHS[int(condition.split("_")[1]) - 1]
                        phi, _ = sample_fixed_geometry(h, w, strength, sample_index=sample,
                            case_index=case, seed=args.seed, control_grid=args.control_grid,
                            min_jacobian=args.min_jacobian)
                        truth = SyntheticGeometry(**{key: value.to(gt) for key, value in vars(phi).items()})
                        observed = pipeline._process_from_geometry(process, truth).terminal_observation(gt)
                    predictions, rigid, local = predict(observed, msi)
                    predictions["oracle"] = oracle_predict(observed, msi, truth)
                    diagnostic = geometry_diagnostics(truth, rigid, local, msi, observed, process, srf)
                    patch_geometry.append(diagnostic)
                    metrics = {}
                    for path in PATHS:
                        metrics[path] = {}
                        for stage, prediction in predictions[path].items():
                            if prediction.shape != gt.shape or not torch.isfinite(prediction).all():
                                raise ValueError(f"Invalid prediction: {condition}/{path}/{stage}")
                            result = pipeline.calc_metrics(prediction, gt, args.scale_ratio)
                            metrics[path][stage] = result
                            patch_metrics.setdefault((path, stage), []).append(result)
                    record = {"sample": sample + 1, "case": case + 1,
                              "observed_sha256": tensor_sha256(observed),
                              "local_field_sha256": tensor_sha256(truth.local_field),
                              "gt_rigid": [truth.dx.item(), truth.dy.item(), truth.theta_deg.item()],
                              "estimated_rigid": rigid[0].tolist(), "geometry": diagnostic, "metrics": metrics}
                    details.write(json.dumps(record, allow_nan=False) + "\n")
                for key, values in patch_metrics.items():
                    buckets.setdefault(key, []).append({metric: mean(v[metric] for v in values) for metric in METRICS})
                geometry_patches.append({key: mean(v[key] for v in patch_geometry) for key in patch_geometry[0]})
                details.flush()
                print(f"DIAG condition={condition} patch={sample+1}/{count} cases={cases}", flush=True)
        reference_inputs = inputs
        write_json(output / "test_inputs.json", inputs)
        for (path, stage), values in buckets.items():
            row = {"condition": condition, "path": path, "stage": stage,
                   "patches": count, "cases_per_patch": cases,
                   **{metric: mean(v[metric] for v in values) for metric in METRICS}}
            clean = row if condition == "registered" else next(
                r for r in rows if r["condition"] == "registered" and r["path"] == path and r["stage"] == stage)
            row["PSNR_drop_vs_own_registered_db"] = clean["PSNR"] - row["PSNR"]
            row["SAM_increase_vs_own_registered_deg"] = row["SAM"] - clean["SAM"]
            rows.append(row)
            print(f"RESULT {condition} {path} {stage} PSNR={row['PSNR']:.6f} SAM={row['SAM']:.6f}", flush=True)
        geometry_rows.append({"condition": condition,
                              **{key: mean(v[key] for v in geometry_patches) for key in geometry_patches[0]}})
        manifest["comparisons"] = comparison_rows(rows)
        write_json(output / "summary.json", manifest)
        write_csv(output / "summary.csv", rows)
        write_csv(output / "comparisons.csv", manifest["comparisons"])
        write_csv(output / "geometry.csv", geometry_rows)
    manifest["complete"] = True
    write_json(output / "summary.json", manifest)
    print(f"RUN_COMPLETE {output / 'summary.csv'}", flush=True)
    return manifest


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(vars(args), ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("DRY_RUN: no data/checkpoints loaded; registered weights must be supplied or resolved from registered GIGI metadata")
        return
    import torch
    import train_innovation3_gigi_cdrdi as pipeline
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu explicitly")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.overwrite_output:
        raise FileExistsError(f"Output directory is nonempty: {output}; use a new directory or --overwrite_output")
    pipeline.set_seed(args.seed)
    cfg = pipeline._config(args)
    _, loader, info = pipeline.build_loaders(cfg)
    process = pipeline.build_progressive_process(cfg)
    predict, oracle, srf, records = build_models(args, info, process, device)
    try:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = None
    metadata = {"protocol_id": "paired-cdrdi-robustness-v1", "source_commit": source_commit,
                "torch_version": str(torch.__version__), "checkpoints": records,
                "observation": "Y_H=P0(W_phi(X)); HR-MSI and GT unchanged",
                "random": "EXACT legacy evaluate: device RNG seed+70000 reset per patch; signed uniform rigid; local max in [0.65,1]*bound; no alpha/identity mixture",
                "fixed": "fixed-rigid-random-local-v1; CPU synthesis; same seed/patch/case as test_fixed_deformation.py",
                "aggregation": "case means within each patch, then equal patch means; full frame; metrics.py",
                "registered_reference": "adapted_naive/base matches legacy registered_base; adapted_naive/refined matches legacy registered_refined",
                "estimated_on_registered": "solver runs on identity input as an extra diagnostic; legacy REGISTERED bypasses solver",
                "oracle": "diagnostic only; same adapted diffusion/refiner as estimated; not guaranteed performance upper bound",
                "refiner": "off" if args.refiner_mode == "off" else "identical frozen shared GIGI on every path; base also reported"}
    with torch.no_grad():
        run_diagnostic(args, loader.dataset, process, predict, oracle, srf, device, metadata)


if __name__ == "__main__":
    main()
