"""Shared main-table evaluation; keep this file identical in both repositories.

Frozen weights, one registered evaluation, and ten geometry seeds x ten cases
PER PATCH. Report sample std of the ten full-test-set seed means (ddof=1).
All synthesis is CPU/float32; geometry and reconstruction RNGs are independent.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from fixed_deformation_protocol import (
    FixedGeometry, METRIC_KEYS, _dense_field, calc_metrics,
    local_jacobian, tensor_sha256, warp_hsi,
)


PROTOCOL_ID = "main-table-random-geometry-10x10-v1"
DEFAULT_GEOMETRY_SEEDS = tuple(range(10001, 10011))


def add_multiseed_args(parser, *, include_cases=True):
    if include_cases:
        parser.add_argument("--cases", "--eval_cases", dest="cases", type=int, default=10)
    parser.add_argument("--geometry_seeds", type=int, nargs="+", default=list(DEFAULT_GEOMETRY_SEEDS),
                        help="Independent TEST deformation seeds, not training seeds")
    parser.add_argument("--eval_seed", type=int, default=1234,
                        help="Fixed reconstruction-noise schedule, independent of geometry seeds")
    parser.add_argument("--max_samples", type=int, default=0, help="0=entire split; positive values mark a SMOKE test")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--resume_output", action="store_true", help="Reuse verified, completed seeds; restart an incomplete seed")
    parser.add_argument("--overwrite_output", action="store_true")


def validate_settings(seeds, cases, max_samples=0):
    if not seeds or len(set(seeds)) != len(seeds) or any(int(s) != s or s < 0 for s in seeds):
        raise ValueError("geometry_seeds must be distinct nonnegative integers")
    if cases < 1 or max_samples < 0:
        raise ValueError("cases must be >=1 and max_samples must be >=0")


def private_seed(domain, *indices):
    payload = "|".join([PROTOCOL_ID, domain, *map(str, indices)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_geometry(height, width, *, geometry_seed, sample_index, case_index,
                    max_translation=4.0, max_rotation_deg=2.0, max_local_px=4.0,
                    control_grid=5, min_jacobian=0.5):
    """Signed uniform rigid motion; U(0,max) local-amplitude proposal.

    Local proposals are conditioned on nonfolding. No alpha/identity mixture.
    Rigid parameters stay fixed when a local proposal is rejected.
    """
    if min(height, width, control_grid) < 2 or not 0 < min_jacobian <= 1:
        raise ValueError("spatial/control dimensions must be >=2 and min_jacobian in (0,1]")
    if min(max_translation, max_rotation_deg, max_local_px) < 0:
        raise ValueError("geometry bounds must be nonnegative")
    seed = private_seed("geometry", geometry_seed, sample_index, case_index)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    def uniform(low, high):
        return torch.rand(1, generator=gen, dtype=torch.float32) * (high - low) + low
    dx, dy = uniform(-max_translation, max_translation), uniform(-max_translation, max_translation)
    angle = uniform(-max_rotation_deg, max_rotation_deg)
    for attempt in range(64):
        control = torch.randn(1, 2, control_grid, control_grid, generator=gen, dtype=torch.float32)
        control -= control.mean(dim=(-2, -1), keepdim=True)
        dense = _dense_field(control, height, width)
        amplitude = uniform(0, max_local_px)
        control *= amplitude / torch.linalg.vector_norm(dense, dim=1).amax().clamp_min(1e-8)
        dense = _dense_field(control, height, width)
        jacobian = local_jacobian(dense).amin().item()
        if jacobian >= min_jacobian:
            geometry = FixedGeometry(dx, dy, angle, control, dense)
            # warp_hsi is the common local-then-rigid grid_sample sampler.
            record = {"geometry_seed": int(geometry_seed), "geometry_rng_seed": seed,
                      "sample": sample_index + 1, "case": case_index + 1,
                      "dx_hr_px": dx.item(), "dy_hr_px": dy.item(),
                      "rotation_deg": angle.item(),
                      "local_max_hr_px": torch.linalg.vector_norm(dense, dim=1).amax().item(),
                      "min_local_jacobian": jacobian, "sampling_attempts": attempt + 1,
                      "local_field_sha256": tensor_sha256(dense)}
            return geometry, record
    raise RuntimeError("Failed to sample a nonfolding field in 64 attempts")


@contextmanager
def reconstruction_rng(device, seed):
    """Keep the SAME per-patch/case noise schedule across geometry seeds."""
    device = torch.device(device)
    devices = [] if device.type != "cuda" else [device.index if device.index is not None else torch.cuda.current_device()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def seed_statistics(seed_rows):
    """Each row is one FULL-split seed mean, never a single patch/case."""
    return {"mean": {key: mean(row[key] for row in seed_rows) for key in METRIC_KEYS},
            "std": {key: stdev(row[key] for row in seed_rows) if len(seed_rows) > 1 else None for key in METRIC_KEYS},
            "std_ddof": 1, "n_geometry_seeds": len(seed_rows)} if seed_rows else None


def _write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _publish(output, conditions, registered, results, complete):
    stats = seed_statistics([r["average_metrics"] for r in results])
    payload = {"complete": complete, "test_conditions": conditions,
               "Registered": registered["average_metrics"] if registered else None,
               "Warp": stats, "per_seed": results}
    _write_json(output / "summary.json", payload)
    _write_csv(output / "per_seed.csv", [
        {"geometry_seed": r["geometry_seed"], **r["average_metrics"]} for r in results],
        ["geometry_seed", *METRIC_KEYS])
    rows = []
    if registered:
        rows.append({"condition": "Registered", "n_geometry_seeds": "",
                     **{f"{k}_mean": v for k, v in registered["average_metrics"].items()}})
    if stats:
        rows.append({"condition": "Warp", "n_geometry_seeds": len(results),
                     **{f"{k}_mean": v for k, v in stats["mean"].items()},
                     **{f"{k}_std": v for k, v in stats["std"].items()}})
    _write_csv(output / "summary.csv", rows,
               ["condition", "n_geometry_seeds", *[name for k in METRIC_KEYS for name in (f"{k}_mean", f"{k}_std")]])
    return payload


def _stable_metadata(metadata):
    result = dict(metadata)
    if isinstance(result.get("model_args"), dict):
        # Run-control flags do not change model predictions and must not break resume.
        result["model_args"] = {k: v for k, v in result["model_args"].items()
                                if k not in ("resume_output", "overwrite_output", "dry_run", "output_dir")}
    return result


@torch.no_grad()
def run_multiseed_suite(test_set, predict, degrade, *, device, output_dir, metadata,
                        predict_registered=None, geometry_seeds=DEFAULT_GEOMETRY_SEEDS,
                        cases=10, eval_seed=1234, control_grid=5, min_jacobian=0.5,
                        max_translation=4.0, max_rotation_deg=2.0, max_local_px=4.0,
                        scale_ratio=4, max_samples=0, resume_output=False, overwrite_output=False):
    validate_settings(geometry_seeds, cases, max_samples)
    if resume_output and overwrite_output:
        raise ValueError("resume_output and overwrite_output are mutually exclusive")
    count = min(len(test_set), max_samples) if max_samples else len(test_set)
    if count < 1:
        raise ValueError("Test split is empty")
    output, device = Path(output_dir), torch.device(device)
    if output.exists() and any(output.iterdir()) and not (resume_output or overwrite_output):
        raise FileExistsError(f"Nonempty output: {output}; use --resume_output or a new directory")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite_output:
        # Do not leave old completed-seed markers that a later resume could reuse.
        for pattern in ("seed_*.json", "seed_*.jsonl", "registered.json", "manifest.json",
                        "summary.json", "summary.csv", "per_seed.csv"):
            for old_file in output.glob(pattern):
                old_file.unlink()
    inputs = []
    for sample in range(count):
        item = test_set[sample]
        inputs.append({"sample": sample + 1, "shape": list(item["gt"].shape),
                       "gt_sha256": tensor_sha256(item["gt"]), "msi_sha256": tensor_sha256(item["hr_msi"])})
    conditions = {
        "protocol_id": PROTOCOL_ID, "torch_version": str(torch.__version__),
        "geometry_seeds": list(geometry_seeds), "cases_per_patch_per_seed": cases,
        "test_patch_count": count, "full_test_patch_count": len(test_set), "smoke_test": bool(max_samples),
        "warp_observations": count * cases * len(geometry_seeds),
        "geometry_device": "cpu/float32", "geometry_schedule": "SHA256(protocol,geometry_seed,patch,case)",
        "max_translation_hr_px": max_translation, "max_rotation_deg": max_rotation_deg,
        "max_local_hr_px": max_local_px, "control_grid": control_grid, "min_local_jacobian": min_jacobian,
        "local_amplitude_proposal": "U(0,max_local_hr_px), conditioned on nonfolding; no alpha or Identity mixture",
        "warp_observation": "Y_H=P0(W_phi(X)); HR-MSI and GT remain registered",
        "reconstruction_eval_seed": eval_seed, "noise_schedule": "SHA256(protocol,eval_seed,patch,case), held fixed across geometry seeds",
        "scale_ratio": scale_ratio, "metric_region": "full patch, no border crop",
        "aggregation": "cases within patch -> patches within seed -> mean and sample std over seed means (ddof=1)",
        "Registered_std": "not reported; evaluated once per patch with same weights",
        "metric_formulas": "existing S2 final formulas; clamp01 PSNR/SAM/RMSE; raw RMSE; global-statistics SSIM",
        "model": _stable_metadata(metadata), "test_inputs": inputs,
    }
    manifest = output / "manifest.json"
    if resume_output:
        if not manifest.is_file():
            raise ValueError("Resume requires manifest.json from this protocol")
        if json.loads(manifest.read_text(encoding="utf-8")) != conditions:
            raise ValueError("Resume protocol/model/data mismatch; use a new output directory")
    _write_json(manifest, conditions)

    def observation(sample):
        item = test_set[sample]
        if tensor_sha256(item["gt"]) != inputs[sample]["gt_sha256"] or tensor_sha256(item["hr_msi"]) != inputs[sample]["msi_sha256"]:
            raise RuntimeError("Test inputs/order changed during evaluation")
        return (item["gt"].unsqueeze(0).to(device=device, dtype=torch.float32),
                item["hr_msi"].unsqueeze(0).to(device=device, dtype=torch.float32))

    def score(observed, msi, gt, sample, case, predictor):
        inference_seed = private_seed("reconstruction", eval_seed, sample, case)
        with reconstruction_rng(device, inference_seed):
            prediction = predictor(observed, msi)
        if prediction.shape != gt.shape or not torch.isfinite(prediction).all():
            raise ValueError("Prediction must be finite and have the GT HSI shape")
        return calc_metrics(prediction, gt, scale_ratio), inference_seed

    registered_path = output / "registered.json"
    if resume_output and registered_path.is_file():
        registered = json.loads(registered_path.read_text(encoding="utf-8"))
        if not registered.get("complete") or registered.get("test_patch_count") != count:
            raise ValueError("Invalid registered result in resume directory")
    else:
        registered_rows = []
        for sample in range(count):
            gt, msi = observation(sample)
            observed = degrade(gt)
            metrics, inference_seed = score(observed, msi, gt, sample, 0, predict_registered or predict)
            registered_rows.append({"sample": sample + 1, "reconstruction_seed": inference_seed,
                                    "observed_sha256": tensor_sha256(observed), "metrics": metrics})
        registered = {"complete": True, "test_patch_count": count,
                      "average_metrics": {k: mean(r["metrics"][k] for r in registered_rows) for k in METRIC_KEYS},
                      "per_patch": registered_rows}
        _write_json(registered_path, registered)
    print("FINAL_REGISTERED " + " ".join(f"{k}={registered['average_metrics'][k]:.6f}" for k in METRIC_KEYS), flush=True)
    results = []
    _publish(output, conditions, registered, results, False)
    for seed in geometry_seeds:
        completed = output / f"seed_{seed}.json"
        if resume_output and completed.is_file():
            record = json.loads(completed.read_text(encoding="utf-8"))
            if not record.get("complete") or record.get("geometry_seed") != seed or record.get("evaluated_cases") != count * cases:
                raise ValueError(f"Invalid completed seed record: {completed}")
            results.append(record)
            _publish(output, conditions, registered, results, False)
            print(f"RESUME_GEOMETRY_SEED={seed}", flush=True)
            continue
        patch_rows = []
        with (output / f"seed_{seed}.jsonl").open("w", encoding="utf-8") as detail:
            for sample in range(count):
                gt, msi = observation(sample)
                case_rows = []
                for case in range(cases):
                    phi, geometry_record = sample_geometry(*gt.shape[-2:], geometry_seed=seed,
                        sample_index=sample, case_index=case, max_translation=max_translation,
                        max_rotation_deg=max_rotation_deg, max_local_px=max_local_px,
                        control_grid=control_grid, min_jacobian=min_jacobian)
                    phi = FixedGeometry(**{key: value.to(gt) for key, value in vars(phi).items()})
                    observed = degrade(warp_hsi(gt, phi))
                    metrics, inference_seed = score(observed, msi, gt, sample, case, predict)
                    case_rows.append(metrics)
                    detail.write(json.dumps({**geometry_record, "reconstruction_seed": inference_seed,
                                             "observed_sha256": tensor_sha256(observed), "metrics": metrics}, allow_nan=False) + "\n")
                detail.flush()
                patch_rows.append({k: mean(r[k] for r in case_rows) for k in METRIC_KEYS})
                print(f"MULTISEED seed={seed} patch={sample+1}/{count} cases={cases}", flush=True)
        record = {"complete": True, "geometry_seed": seed, "evaluated_cases": count * cases,
                  "average_metrics": {k: mean(r[k] for r in patch_rows) for k in METRIC_KEYS},
                  "per_patch_metrics": patch_rows, "detail_jsonl": f"seed_{seed}.jsonl"}
        _write_json(completed, record)
        results.append(record)
        _publish(output, conditions, registered, results, False)
        print(f"GEOMETRY_SEED={seed} " + " ".join(f"{k}={record['average_metrics'][k]:.6f}" for k in METRIC_KEYS), flush=True)
    summary = _publish(output, conditions, registered, results, True)
    stats = summary["Warp"]
    print("FINAL_WARP_MEAN_STD " + " ".join(
        f"{k}={stats['mean'][k]:.6f}+/-{stats['std'][k]:.6f}" if stats['std'][k] is not None
        else f"{k}={stats['mean'][k]:.6f}+/-NA" for k in METRIC_KEYS), flush=True)
    print(f"MULTISEED_TEST_SUMMARY={output / 'summary.csv'}", flush=True)
    return summary
