"""Paired x0-state / final clean-prediction diagnostic; no training changes.

Defaults to a small PaviaU test (three random cases). Both outputs come from
ONE production-equivalent reverse loop. GT enters only scoring, synthesis,
masks, and separately labelled oracle controls, never the learned predictor.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from statistics import mean


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["PaviaU"], default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--cases", type=int, default=3)
    p.add_argument("--max_samples", type=int, default=1, help="0=full test split")
    p.add_argument("--conditions", nargs="+", choices=["registered", "warp"], default=["registered", "warp"])
    p.add_argument("--geometry_checkpoint", default="./checkpoints/PaviaU/PaviaU_recursive_k6_finalonly_500ep_lr1e4.pth")
    p.add_argument("--diffusion_checkpoint", default="./checkpoints/PaviaU/PaviaU_estimated_deform_diffusion_k9_stage2d_A.pth")
    p.add_argument("--refiner_checkpoint", default="", help="Optional frozen HPGC; applied only to original state_x0")
    for name, default in (("geometry_steps", 9), ("geometry_base_channels", 32),
                          ("diffusion_steps", 12), ("diffusion_base_channels", 64),
                          ("time_dim", 256), ("spectral_hidden", 8), ("refine_hidden", 64),
                          ("heads", 4), ("test_size", 128), ("patch_size", 64),
                          ("stride", 32), ("scale_ratio", 4), ("control_grid", 5)):
        p.add_argument("--" + name, type=int, default=default)
    for name, default in (("mtf_nyquist", .2), ("psf_truncate", 3.),
                          ("max_translation", 4.), ("max_rotation_deg", 2.),
                          ("max_local_px", 4.), ("min_jacobian", .5), ("dropout", 0.)):
        p.add_argument("--" + name, type=float, default=default)
    p.add_argument("--srf_interp", choices=["pchip", "linear"], default="pchip")
    p.add_argument("--variant", choices=["conv", "gigi", "gigi_hetero", "gigi_phy", "full"], default="full")
    p.add_argument("--disable_tangent", action="store_true")
    p.add_argument("--output_dir", default="outputs/diffusion_endpoint/PaviaU")
    p.add_argument("--overwrite_output", action="store_true")
    p.add_argument("--oracle_only", action="store_true", help="CPU controls only; no data/checkpoints required")
    p.add_argument("--dry_run", action="store_true")
    p.set_defaults(stage="test", batch_size=1, num_workers=0)
    args = p.parse_args(argv)
    if args.cases < 1 or args.max_samples < 0 or min(args.geometry_steps, args.diffusion_steps) < 1:
        p.error("cases and steps must be positive; max_samples must be nonnegative")
    if len(set(args.conditions)) != len(args.conditions):
        p.error("conditions must be distinct")
    if args.control_grid < 2 or not 0 < args.min_jacobian <= 1:
        p.error("control_grid>=2 and 0<min_jacobian<=1 required")
    if min(args.max_translation, args.max_rotation_deg, args.max_local_px) < 0:
        p.error("geometry bounds must be nonnegative")
    return args


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def tensor_hash(value):
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def checkpoint_record(path, dataset):
    import torch
    path = Path(path)
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    extra = state.get("extra", {})
    if extra.get("dataset") not in (None, dataset):
        raise ValueError(f"Checkpoint dataset mismatch: {path}: {extra['dataset']}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    # Record configuration without serializing the model or optimizer tensors.
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(),
            "epoch": state.get("epoch"), "extra": json.loads(json.dumps(extra, default=str))}


def reconstruct_pair(model, process, observed, *, target_size, hr_msi):
    """Same operations/order as innovation1; retain last prediction as well.

    This function receives neither GT nor GT geometry. It never estimates a
    residual from GT or corrects the production trajectory with such a residual.
    """
    import torch
    from innovation1 import model_predict
    with torch.no_grad():
        model.eval()
        x_t = process.terminal_state(observed, target_size=target_size)
        terminal = x_t.clone()
        for t in range(process.total_steps, 0, -1):
            timestep = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
            prediction = model_predict(model, x_t, timestep, hr_msi)
            if t == 1:
                x1 = x_t.clone()
                last = prediction.clone()
                last_model_state = process.state_at(last, 1)
            x_t = process.reverse_update(x_t, prediction, t)
        for value in (x_t, last):
            if value.shape != terminal.shape or not torch.isfinite(value).all():
                raise FloatingPointError("Nonfinite or incorrectly shaped endpoint")
        gap = x_t - last
        expected_gap = x1 - last_model_state
        return {"state_x0": x_t, "last_prediction": last, "terminal_state": terminal,
                "x1": x1, "state_minus_prediction": gap,
                "gap_identity_max_abs": float((gap - expected_gap).abs().max())}


def oracle_controls():
    """Exact-X predictors with exact/mismatched geometry, independent of weights."""
    import torch
    from degradations.physical import PhysicalDegradation
    from degradations.progressive import ProgressiveDegradation
    from degradations.deformation_aware import DeformationAwareProgressiveDegradation

    class Oracle(torch.nn.Module):
        def __init__(self, reference):
            super().__init__()
            self.register_buffer("reference", reference)

        def forward(self, state, timestep):
            return self.reference

    generator = torch.Generator(device="cpu").manual_seed(53)
    gt = torch.rand(1, 3, 32, 32, generator=generator)
    base = ProgressiveDegradation(PhysicalDegradation(scale_ratio=4), total_steps=12)
    local = torch.zeros(1, 2, 32, 32)
    true_rigid = torch.tensor([[2., -1., 1.1]])
    true_process = DeformationAwareProgressiveDegradation(base, rigid=true_rigid, local_field=local)
    observed = true_process.terminal_observation(gt)
    rows = []
    for label, rigid in (("zero_residual", true_rigid),
                         ("nonzero_residual", torch.tensor([[1.4, -.6, .7]]))):
        process = DeformationAwareProgressiveDegradation(base, rigid=rigid, local_field=local)
        delta = process.terminal_state(observed, target_size=(32, 32)) - process.state_at(gt, 12)
        pair = reconstruct_pair(Oracle(gt), process, observed, target_size=(32, 32), hr_msi=None)
        error = float((pair["state_x0"] - (gt + delta)).abs().max())
        prediction_error = float((pair["last_prediction"] - gt).abs().max())
        delta_rms = float(delta.square().mean().sqrt())
        if error > 2e-5 or prediction_error > 2e-5 or pair["gap_identity_max_abs"] > 2e-5:
            raise AssertionError(f"Oracle endpoint identity failed: {label}")
        if (label == "zero_residual" and delta_rms > 2e-5) or (label == "nonzero_residual" and delta_rms <= 1e-5):
            raise AssertionError(f"Oracle residual control is invalid: {label}")
        rows.append({"control": label, "delta_T_raw_RMSE": delta_rms,
                     "state_minus_GT_raw_RMSE": float((pair["state_x0"] - gt).square().mean().sqrt()),
                     "prediction_minus_GT_raw_RMSE": float((pair["last_prediction"] - gt).square().mean().sqrt()),
                     "state_minus_GT_plus_delta_max_abs": error,
                     "gap_identity_max_abs": pair["gap_identity_max_abs"], "passed": True})
    return rows


def valid_masks(base_process, truth_rigid, truth_local, estimated_rigid, estimated_local):
    """Conservative common support, shared by ALL endpoints for a given case.

    Omega_M is an interior reference-frame rectangle. Omega_H contains only
    detector cells whose entire PSF/area footprint samples inside Omega_M
    under BOTH true and estimated geometry (including bilinear support).
    No weights/GT intensities or endpoint errors enter mask construction.
    """
    import torch
    import torch.nn.functional as F
    from cdrdi_geometry import forward_warp, sampling_coordinates
    h, w = truth_local.shape[-2:]
    radius = max(1, math.ceil(base_process.operator.truncate * base_process.operator.terminal_sigma))
    yy, xx = torch.meshgrid(torch.arange(h, device=truth_local.device),
                            torch.arange(w, device=truth_local.device), indexing="ij")
    coords = [sampling_coordinates(h, w, rigid[:, 0], rigid[:, 1], rigid[:, 2], local)
              for rigid, local in ((truth_rigid, truth_local), (estimated_rigid, estimated_local))]
    displacement = max(float(torch.maximum((x - xx).abs(), (y - yy).abs()).max()) for x, y in coords)
    margin = math.ceil(displacement) + radius + 1
    if 2 * margin >= min(h, w):
        raise ValueError(f"No common HR support: margin={margin}, size={(h, w)}")
    hr = truth_local.new_zeros((1, 1, h, w))
    hr[:, :, margin:h-margin, margin:w-margin] = 1
    masks = []
    for rigid, local in ((truth_rigid, truth_local), (estimated_rigid, estimated_local)):
        support = forward_warp(hr, rigid[:, 0], rigid[:, 1], rigid[:, 2], local) >= 1 - 1e-6
        # Explicit zero padding excludes truncated PSF footprints at image edges.
        invalid = F.pad((~support).float(), (radius,) * 4, value=1)
        blurred_valid = F.max_pool2d(invalid, 2 * radius + 1, stride=1) == 0
        scale = base_process.operator.scale_ratio
        masks.append(F.avg_pool2d(blurred_valid.float(), scale, scale) >= 1 - 1e-6)
    lr = masks[0] & masks[1]
    if not lr.any():
        raise ValueError("No valid LR detector cells remain; use a larger test patch")
    return hr.bool(), lr, margin


def endpoint_metrics(prediction, gt, observed, msi, acquisition, srf, hr_mask, lr_mask, scale_ratio):
    """Repository metric conventions on common support + raw observation RMSE."""
    import torch
    from cdrdi_geometry import spectral_project
    from metrics import calc_metrics
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite reconstruction")
    # Keep channels, flatten only valid spatial sites. All masks are endpoint-independent.
    spatial = hr_mask[:, 0]
    p = prediction.permute(0, 2, 3, 1)[spatial].double()
    g = gt.permute(0, 2, 3, 1)[spatial].double()
    pc, gc = p.clamp(0, 1), g.clamp(0, 1)
    rmse = (pc - gc).square().mean().sqrt().clamp_min(1e-6)
    norms = pc.norm(dim=1) * gc.norm(dim=1)
    valid = (pc.norm(dim=1) > 1e-12) & (gc.norm(dim=1) > 1e-12)
    angles = torch.acos(((pc[valid] * gc[valid]).sum(1) / norms[valid].clamp_min(1e-12)).clamp(-1, 1))
    band_rmse = ((p - g).square().mean(0) + 1e-8).sqrt()
    ergas = 100 / scale_ratio * ((band_rmse / (g.mean(0) + 1e-8)).square().mean()).sqrt()

    def masked_rmse(error, mask):
        selected = error.double().masked_select(mask.expand_as(error))
        if not selected.numel() or not torch.isfinite(selected).all():
            raise ValueError("Empty/nonfinite observation residual")
        return float(selected.square().mean().sqrt())

    h_error = acquisition.terminal_observation(prediction) - observed
    m_error = spectral_project(prediction, srf) - msi
    return {"PSNR": float(-20 * torch.log10(rmse)), "SAM": float(angles.mean() * 180 / math.pi) if valid.any() else 0.,
            "ERGAS": float(ergas), "RMSE": float(rmse), "raw_RMSE": float((p-g).square().mean().sqrt()),
            "E_H": masked_rmse(h_error, lr_mask), "E_M": masked_rmse(m_error, hr_mask),
            "HR_valid_pixels": int(spatial.sum()), "LR_valid_pixels": int(lr_mask.sum()),
            **{"full_" + k: v for k, v in calc_metrics(prediction, gt, scale_ratio).items()}}


def run_diagnostic(args, test_set, process, diffusion, estimate, srf, device, refiner=None, refine=None):
    """Injectable runner: estimate sees observations only; refine sees original state only."""
    import numpy as np
    import torch
    import train_innovation3_gigi_cdrdi as pipeline
    from cdrdi_geometry import sample_synthetic_geometry
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    count = min(len(test_set), args.max_samples) if args.max_samples else len(test_set)
    if not count:
        raise ValueError("Empty test split")
    rows, traces = [], []
    for condition in args.conditions:
        for sample in range(count):
            item = test_set[sample]
            gt = item["gt"].unsqueeze(0).to(device=device, dtype=torch.float32)
            msi = item["hr_msi"].unsqueeze(0).to(device=device, dtype=torch.float32)
            h, w = gt.shape[-2:]
            # Exactly the legacy random evaluation sampler: reset per patch.
            generator = pipeline._make_generator(device, args.seed + 70000)
            for case in range(1 if condition == "registered" else args.cases):
                if condition == "registered":
                    truth_rigid = gt.new_zeros((1, 3))
                    truth_local = gt.new_zeros((1, 2, h, w))
                    observed = process.terminal_observation(gt)
                    rigid, local, acquisition = truth_rigid, truth_local, process
                else:
                    truth = sample_synthetic_geometry(h, w, device=device, dtype=gt.dtype,
                        generator=generator, max_translation=args.max_translation,
                        max_rotation_deg=args.max_rotation_deg, max_local_px=args.max_local_px,
                        control_grid=args.control_grid, min_jacobian=args.min_jacobian)
                    truth_rigid = torch.stack([truth.dx, truth.dy, truth.theta_deg], dim=1)
                    truth_local = truth.local_field
                    observed = pipeline._process_from_geometry(process, truth).terminal_observation(gt)
                    rigid, local = estimate(observed, msi)
                    acquisition = pipeline._process_from_estimate(process, rigid, local)
                pair = reconstruct_pair(diffusion, acquisition, observed, target_size=(h, w), hr_msi=msi)
                # GT used only after reconstruction for diagnostic scoring/offset audit.
                delta = pair["terminal_state"] - acquisition.state_at(gt, acquisition.total_steps)
                hr_mask, lr_mask, margin = valid_masks(process, truth_rigid, truth_local, rigid, local)
                endpoints = {key: pair[key] for key in ("state_x0", "last_prediction")}
                if refiner is not None:
                    endpoints["state_x0_refined"] = refine(pair["state_x0"], msi, observed, acquisition)
                stem = f"{condition}/sample{sample+1:04d}_case{case+1:03d}"
                archive = output / (stem + ".npz")
                archive.parent.mkdir(parents=True, exist_ok=True)
                tensors = {"gt": gt, "hr_msi": msi, "observed_lr_hsi": observed,
                           "srf": srf, "delta_T_diagnostic_only": delta,
                           "hr_valid_mask": hr_mask, "lr_valid_mask": lr_mask,
                           "true_rigid": truth_rigid, "true_local": truth_local,
                           "estimated_rigid": rigid, "estimated_local": local,
                           **{k: v for k, v in pair.items() if torch.is_tensor(v)}, **endpoints}
                np.savez_compressed(archive, **{k: v.detach().cpu().numpy() for k, v in tensors.items()})
                common = {"condition": condition, "sample": sample+1, "case": case+1,
                          "archive": str(archive.relative_to(output)),
                          "gt_sha256": tensor_hash(gt), "msi_sha256": tensor_hash(msi),
                          "observed_sha256": tensor_hash(observed), "mask_margin_hr": margin}
                trace = {**common, "gap_identity_max_abs": pair["gap_identity_max_abs"],
                         "state_prediction_gap_raw_RMSE": float(pair["state_minus_prediction"].square().mean().sqrt()),
                         "delta_T_raw_RMSE": float(delta.square().mean().sqrt()),
                         "gap_minus_delta_raw_RMSE": float((pair["state_minus_prediction"]-delta).square().mean().sqrt()),
                         "true_rigid": truth_rigid[0].tolist(), "estimated_rigid": rigid[0].tolist()}
                traces.append(trace)
                for name, prediction in endpoints.items():
                    row = {**common, "output": name, **endpoint_metrics(prediction, gt, observed, msi,
                               acquisition, srf, hr_mask, lr_mask, args.scale_ratio)}
                    rows.append(row)
                    print(f"ENDPOINT {condition} sample={sample+1} case={case+1} {name} "
                          f"ROI_PSNR={row['PSNR']:.5f} ROI_SAM={row['SAM']:.5f} "
                          f"FULL_PSNR={row['full_PSNR']:.5f} FULL_SAM={row['full_SAM']:.5f} "
                          f"E_H={row['E_H']:.7f} E_M={row['E_M']:.7f}", flush=True)
                write_csv(output / "per_sample.csv", rows)
                write_json(output / "traces.json", traces)
    metric_keys = [key for key in rows[0] if key in ("PSNR", "SAM", "ERGAS", "RMSE", "raw_RMSE", "E_H", "E_M", "HR_valid_pixels", "LR_valid_pixels") or key.startswith("full_")]
    summary = []
    for condition, name in dict.fromkeys((r["condition"], r["output"]) for r in rows):
        group = [r for r in rows if r["condition"] == condition and r["output"] == name]
        summary.append({"condition": condition, "output": name, "patches": count,
                        "cases_per_patch": len(group)//count,
                        **{k: mean(r[k] for r in group) for k in metric_keys}})
    write_csv(output / "summary.csv", summary)
    return summary


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(vars(args), ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.overwrite_output:
        raise FileExistsError(f"Nonempty output directory: {output}; choose another or pass --overwrite_output")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", {"complete": False, "args": vars(args),
                                          "phase": "oracle_controls"})
    import torch
    # CPU oracle checks are cheap and require neither data nor learned weights.
    with torch.no_grad():
        controls = oracle_controls()
    write_json(output / "oracle_controls.json", controls)
    print("ORACLE_CONTROLS: zero/nonzero residual passed", flush=True)
    if args.oracle_only:
        write_json(output / "manifest.json", {"complete": True, "oracle_only": True,
                                              "args": vars(args), "oracle_controls": controls})
        return
    import train_innovation3_gigi_cdrdi as pipeline
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu")
    pipeline.set_seed(args.seed)
    roles = {"geometry": args.geometry_checkpoint, "diffusion": args.diffusion_checkpoint}
    if args.refiner_checkpoint:
        roles["refiner"] = args.refiner_checkpoint
    records = {role: checkpoint_record(path, args.dataset) for role, path in roles.items()}
    cfg = pipeline._config(args)
    _, test_loader, info = pipeline.build_loaders(cfg)
    process = pipeline.build_progressive_process(cfg)
    geometry, diffusion = pipeline._build_frozen_models(args, info, device)
    srf = torch.as_tensor(info["srf_weights"], device=device, dtype=torch.float32)
    refiner = refine = None
    if args.refiner_checkpoint:
        refiner = pipeline._build_refiner(args, info, device)
        pipeline._load_training_checkpoint(refiner, args.refiner_checkpoint, device)
        refiner.eval().requires_grad_(False)
        projector = pipeline._build_projector(args, info, device)

        def refine(state, msi, observed, acquisition):
            residual = pipeline._physical_residual(acquisition, projector, observed, state)
            return refiner(state, msi, residual, pipeline.ranked_msi_heterogeneity(msi))

    def estimate(observed, msi):
        return pipeline._estimate_geometry(geometry, y_h=observed, hr_msi=msi,
                   p0=process.operator, srf=srf, steps=args.geometry_steps)

    try:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                               text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = None
    manifest = {"complete": False, "protocol": "paired-diffusion-endpoint-v1", "args": vars(args),
                "source_commit": source_commit, "torch_version": str(torch.__version__),
                "checkpoints": records, "srf_sha256": tensor_hash(srf),
                "test_protocol": str(info.get("protocol", "")), "full_test_samples": len(test_loader.dataset),
                "random_geometry": "legacy sampler; seed+70000 reset per patch; no identity mixture or strength scaling",
                "registered": "same adapted diffusion checkpoint; identity acquisition; bypass CDRDI as production REGISTERED does",
                "scoring": "PSNR/SAM clipped [0,1]; ERGAS raw (eps=1e-8); common HR support. full_* uses repository metrics.py on full image",
                "observation_residuals": "E_H/E_M: un-clipped RMS over all valid channel-pixel elements, not a physics guarantee",
                "masks": "Omega_M interior margin=max true/estimated component displacement + PSF radius +1; Omega_H complete PSF/area/bilinear footprint inside Omega_M under both geometries; common across endpoints",
                "aggregation": "equal cases within each patch then equal patches; primary metrics use masks; full_* provides original full-frame convention",
                "archive_layout": "batch-first B,C,H,W; raw float32; boolean masks; never normalized per output",
                "refiner": "original state_x0 input only; same frozen weights; no retraining",
                "oracle_controls": controls}
    write_json(output / "manifest.json", manifest)
    with torch.no_grad():
        manifest["summary"] = run_diagnostic(args, test_loader.dataset, process, diffusion, estimate,
                                            srf, device, refiner, refine)
    manifest["complete"] = True
    write_json(output / "manifest.json", manifest)
    print(f"RUN_COMPLETE {output.resolve() / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
