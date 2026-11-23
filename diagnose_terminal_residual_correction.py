"""Experimental last-step correction replayed from saved endpoint archives.

No learned checkpoint, raw dataset or new model inference is needed. The
production output stays unchanged. Correction itself never receives GT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from diagnose_diffusion_endpoint import endpoint_metrics, tensor_hash, write_csv, write_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source_dir", default="outputs/diffusion_endpoint/PaviaU_decomposition_10cases")
    p.add_argument("--output_dir", default="")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--require_decomposition", action="store_true",
                   help="Refuse source archives without independently accumulated C_pred")
    p.add_argument("--overwrite_output", action="store_true")
    p.add_argument("--oracle_only", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args(argv)
    if not args.output_dir:
        source = Path(args.source_dir)
        args.output_dir = str(source.with_name(source.name + "_terminal_correction"))
    return args


def correct_endpoint(state_x0, last_prediction, terminal_state, acquisition):
    """Full-strength correction, no GT/mask/clipping/refiner or learned parameter."""
    import torch
    with torch.no_grad():
        if state_x0.shape != last_prediction.shape or state_x0.shape != terminal_state.shape:
            raise ValueError("Endpoint and terminal-state shapes must match")
        delta_hat = terminal_state - acquisition.state_at(last_prediction, acquisition.total_steps)
        corrected = state_x0 - delta_hat
        if not torch.isfinite(corrected).all():
            raise FloatingPointError("Nonfinite terminal-corrected output")
        return corrected, delta_hat


def make_process(recorded_args):
    from innovation1 import build_progressive_process
    required = ("scale_ratio", "diffusion_steps", "mtf_nyquist", "psf_truncate")
    if any(k not in recorded_args for k in required):
        raise ValueError("Source manifest lacks physical-process settings")
    cfg = SimpleNamespace(**{k: recorded_args[k] for k in required},
                          degradation_mode="physical", lift_mode="normalized_adjoint")
    return build_progressive_process(cfg)


def oracle_controls():
    """Zero/nonzero residual controls with an exact-X predictor at every step."""
    import torch
    from degradations.deformation_aware import DeformationAwareProgressiveDegradation
    from diagnose_diffusion_endpoint import reconstruct_pair

    class Oracle(torch.nn.Module):
        def __init__(self, reference):
            super().__init__()
            self.register_buffer("reference", reference)

        def forward(self, state, timestep):
            return self.reference

    generator = torch.Generator(device="cpu").manual_seed(53)
    gt = torch.rand(1, 3, 32, 32, generator=generator)
    base = make_process(dict(scale_ratio=4, diffusion_steps=12, mtf_nyquist=.2, psf_truncate=3.))
    local = torch.zeros(1, 2, 32, 32)
    truth = torch.tensor([[2., -1., 1.1]])
    true_process = DeformationAwareProgressiveDegradation(base, rigid=truth, local_field=local)
    observed = true_process.terminal_observation(gt)
    rows = []
    for label, rigid in (("zero_residual", truth), ("nonzero_residual", torch.tensor([[1.4, -.6, .7]]))):
        process = DeformationAwareProgressiveDegradation(base, rigid=rigid, local_field=local)
        pair = reconstruct_pair(Oracle(gt), process, observed, target_size=(32, 32), hr_msi=None)
        corrected, delta_hat = correct_endpoint(pair["state_x0"], pair["last_prediction"], pair["terminal_state"], process)
        delta = pair["terminal_state"] - process.state_at(gt, process.total_steps)
        error = float((corrected-gt).abs().max())
        if error > 2e-5 or not torch.equal(delta, delta_hat):
            raise AssertionError(f"Corrected oracle endpoint failed: {label}")
        if label == "nonzero_residual" and float(delta.square().mean().sqrt()) <= 1e-5:
            raise AssertionError("Nonzero oracle control has a trivial residual")
        rows.append({"control": label, "delta_RMS": float(delta.square().mean().sqrt()),
                     "original_error_RMS": float((pair["state_x0"]-gt).square().mean().sqrt()),
                     "corrected_error_RMS": float((corrected-gt).square().mean().sqrt()),
                     "corrected_error_max_abs": error, "passed": True})
    return rows


def check_decomposition(acquisition, tensors, corrected, delta_hat):
    """Uses the independently saved C; never defines C from the final error."""
    import torch
    gt, last = tensors["gt"], tensors["last_prediction"]
    delta = tensors["terminal_state"].double() - acquisition.state_at(gt, acquisition.total_steps).double()
    correction = tensors["C_pred_diagnostic_only"].double()
    lt_e1 = acquisition.state_at(last-gt, acquisition.total_steps).double()
    original_error = tensors["state_x0"].double()-gt.double()
    corrected_error = corrected.double()-gt.double()
    predicted_error = correction + lt_e1
    estimate_identity_error = delta_hat.double() - (delta-lt_e1)
    rows = []
    for scope, mask in (("full", torch.ones_like(tensors["hr_valid_mask"], dtype=torch.bool)),
                        ("roi", tensors["hr_valid_mask"])):
        def select(value):
            return value.double().masked_select(mask.expand_as(value))

        def rms(value):
            return float(select(value).square().mean().sqrt())

        rows.append({"scope": scope, "delta_true_RMS": rms(delta), "delta_hat_RMS": rms(delta_hat),
                     "delta_hat_minus_delta_RMS": rms(delta_hat.double()-delta),
                     "C_pred_RMS": rms(correction), "L_T_e1_RMS": rms(lt_e1),
                     "predicted_corrected_error_RMS": rms(predicted_error),
                     "actual_corrected_error_RMS": rms(corrected_error),
                     "actual_original_error_RMS": rms(original_error),
                     "original_decomposition_max_abs_error": float(select(original_error-delta-correction).abs().max()),
                     "delta_estimate_identity_max_abs_error": float(select(estimate_identity_error).abs().max()),
                     "corrected_decomposition_max_abs_error": float(select(corrected_error-predicted_error).abs().max()),
                     "actual_MSE_change_vs_original": float((select(corrected_error).square()-select(original_error).square()).mean()),
                     "theoretical_MSE_change_vs_original": float((select(predicted_error).square()-select(delta+correction).square()).mean())})
    tolerance = 2e-5 + 2e-5 * float(corrected_error.abs().max())
    for key in ("original_decomposition_max_abs_error", "delta_estimate_identity_max_abs_error",
                "corrected_decomposition_max_abs_error"):
        if rows[0][key] > tolerance:
            raise AssertionError(f"Saved-process/decomposition mismatch: {key}={rows[0][key]:.6g}")
    return rows


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replay(args, manifest):
    import numpy as np
    import torch
    from degradations.deformation_aware import DeformationAwareProgressiveDegradation
    source = Path(args.source_dir).resolve()
    output = Path(args.output_dir).resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu")
    recorded = json.loads((source / "manifest.json").read_text())
    if not recorded.get("complete") or recorded.get("protocol") != "paired-diffusion-endpoint-v1":
        raise ValueError("Source must be a completed diagnose_diffusion_endpoint run")
    if recorded["args"].get("dataset") != "PaviaU":
        raise ValueError("This diagnostic is scoped to PaviaU")
    traces = json.loads((source / "traces.json").read_text())
    if not traces:
        raise ValueError("Source has no cases")
    expected_count = min(recorded["full_test_samples"], recorded["args"]["max_samples"]) if recorded["args"]["max_samples"] else recorded["full_test_samples"]
    expected = {(c, sample+1, case+1) for c in recorded["args"]["conditions"]
                for sample in range(expected_count) for case in range(1 if c == "registered" else recorded["args"]["cases"])}
    identities = [(r["condition"], r["sample"], r["case"]) for r in traces]
    if set(identities) != expected or len(identities) != len(expected):
        raise ValueError("Source trace case coverage differs from its completed manifest")
    manifest.update({"source_manifest_sha256": _file_hash(source / "manifest.json"),
                     "source_traces_sha256": _file_hash(source / "traces.json"),
                     "source_checkpoints": recorded.get("checkpoints", {}), "source_args": recorded["args"],
                     "source_srf_sha256": recorded["srf_sha256"], "source_cases": len(traces),
                     "source_run_commit": recorded.get("source_commit"),
                     "source_torch_version": recorded.get("torch_version")})
    write_json(output / "manifest.json", manifest)
    base = make_process(recorded["args"])
    rows, decompositions, provenance = [], [], []
    required = ("gt", "hr_msi", "observed_lr_hsi", "srf", "terminal_state", "state_x0", "last_prediction",
                "hr_valid_mask", "lr_valid_mask", "estimated_rigid", "estimated_local")
    for trace in traces:
        if trace["condition"] not in ("registered", "warp"):
            raise ValueError("Unknown source condition")
        archive = (source / trace["archive"]).resolve()
        archive_relative = archive.relative_to(source)
        with np.load(archive, allow_pickle=False) as saved:
            if any(k not in saved for k in required):
                raise ValueError(f"Source archive lacks required tensors: {archive}")
            has_c = "C_pred_diagnostic_only" in saved
            if args.require_decomposition and not has_c:
                raise ValueError(f"No independently accumulated C_pred in {archive}; rerun source with --error_decomposition")
            keys = (*required, *(k for k in ("C_pred_diagnostic_only", "true_rigid", "true_local") if k in saved))
            tensors = {k: torch.from_numpy(saved[k].copy()).to(device) for k in keys}
        if has_c and not recorded.get("error_decomposition", {}).get("enabled"):
            raise ValueError("C_pred provenance is not recorded as independent step accumulation")
        for name, value in tensors.items():
            if name.endswith("mask"):
                if value.dtype != torch.bool or not value.any():
                    raise ValueError(f"Invalid mask: {name}")
            elif not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite source tensor: {name}")
        for key, hash_key in (("gt", "gt_sha256"), ("hr_msi", "msi_sha256"), ("observed_lr_hsi", "observed_sha256")):
            if tensor_hash(tensors[key]) != trace[hash_key]:
                raise ValueError(f"Source input hash mismatch: {key}")
        if tensor_hash(tensors["srf"]) != recorded["srf_sha256"]:
            raise ValueError("Source SRF hash mismatch")
        gt = tensors["gt"]
        if gt.ndim != 4 or gt.shape[0] != 1 or any(tensors[k].shape != gt.shape for k in ("state_x0", "last_prediction", "terminal_state")):
            raise ValueError("Source image shapes are inconsistent")
        if has_c and tensors["C_pred_diagnostic_only"].shape != gt.shape:
            raise ValueError("Source C_pred shape is inconsistent")
        if tensors["hr_valid_mask"].shape != (1, 1, *gt.shape[-2:]) or tensors["lr_valid_mask"].shape != (1, 1, *tensors["observed_lr_hsi"].shape[-2:]):
            raise ValueError("Source mask shapes are inconsistent")
        acquisition = base if trace["condition"] == "registered" else DeformationAwareProgressiveDegradation(
            base, rigid=tensors["estimated_rigid"], local_field=tensors["estimated_local"])
        # This verifies the same physical lifting without loading any checkpoint.
        lifted = acquisition.terminal_state(tensors["observed_lr_hsi"], target_size=tuple(gt.shape[-2:]))
        terminal_error = float((lifted-tensors["terminal_state"]).abs().max())
        if terminal_error > 2e-5:
            raise AssertionError(f"Replayed terminal state differs from archive: {terminal_error:.6g}")
        corrected, delta_hat = correct_endpoint(tensors["state_x0"], tensors["last_prediction"], tensors["terminal_state"], acquisition)
        common = {"condition": trace["condition"], "sample": trace["sample"], "case": trace["case"],
                  "archive": str(archive_relative), "observed_sha256": trace["observed_sha256"]}
        if has_c:
            decompositions.extend({**common, **row} for row in check_decomposition(acquisition, tensors, corrected, delta_hat))
        endpoints = {"state_x0": tensors["state_x0"], "last_prediction": tensors["last_prediction"], "corrected_x0": corrected}
        for name, prediction in endpoints.items():
            row = {**common, "output": name, **endpoint_metrics(prediction, gt, tensors["observed_lr_hsi"], tensors["hr_msi"],
                        acquisition, tensors["srf"], tensors["hr_valid_mask"], tensors["lr_valid_mask"], recorded["args"]["scale_ratio"])}
            rows.append(row)
            print(f"CORRECTION {trace['condition']} sample={trace['sample']} case={trace['case']} {name} "
                  f"FULL_PSNR={row['full_PSNR']:.5f} FULL_SAM={row['full_SAM']:.5f} "
                  f"ERGAS={row['full_ERGAS']:.5f} E_H={row['E_H']:.7f} E_M={row['E_M']:.7f}", flush=True)
        target = output / archive_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        tensors.update(corrected_x0=corrected, estimated_terminal_residual=delta_hat)
        np.savez_compressed(target, **{k: v.detach().cpu().numpy() for k, v in tensors.items()})
        provenance.append({**common, "source_archive_sha256": _file_hash(archive),
                           "source_terminal_replay_max_abs_error": terminal_error,
                           "independent_C_pred_available": has_c})
        write_csv(output / "per_sample.csv", rows)
        write_json(output / "provenance.json", provenance)
        if decompositions:
            write_csv(output / "corrected_decomposition.csv", decompositions)
            write_json(output / "corrected_decomposition.json", decompositions)
    summary = []
    for condition, name in dict.fromkeys((r["condition"], r["output"]) for r in rows):
        group = [r for r in rows if r["condition"] == condition and r["output"] == name]
        metrics = [k for k in group[0] if k in ("PSNR", "SAM", "ERGAS", "RMSE", "raw_RMSE", "E_H", "E_M", "HR_valid_pixels", "LR_valid_pixels") or k.startswith("full_")]
        summary.append({"condition": condition, "output": name,
                        "patches": len({r['sample'] for r in group}), "cases": len(group),
                        **{k: mean(r[k] for r in group) for k in metrics}})
    write_csv(output / "summary.csv", summary)
    paired = []
    for identity in identities:
        group = {r["output"]: r for r in rows if (r["condition"], r["sample"], r["case"]) == identity}
        corrected = group["corrected_x0"]
        for name in ("state_x0", "last_prediction"):
            reference = group[name]
            paired.append({"condition": identity[0], "sample": identity[1], "case": identity[2], "reference": name,
                           "full_PSNR_gain": corrected["full_PSNR"]-reference["full_PSNR"],
                           "full_SAM_reduction": reference["full_SAM"]-corrected["full_SAM"],
                           "full_ERGAS_reduction": reference["full_ERGAS"]-corrected["full_ERGAS"],
                           "E_H_change": corrected["E_H"]-reference["E_H"], "E_M_change": corrected["E_M"]-reference["E_M"]})
    write_csv(output / "paired_changes.csv", paired)
    manifest.update(complete=True, decomposition_complete=all(r["independent_C_pred_available"] for r in provenance), summary=summary)
    write_json(output / "manifest.json", manifest)
    if not manifest["decomposition_complete"]:
        print("DECOMPOSITION_UNAVAILABLE: correction metrics computed; source lacks some independently saved C_pred fields", flush=True)
    print(f"RUN_COMPLETE {output / 'summary.csv'}", flush=True)
    return manifest


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps(vars(args), indent=2), flush=True)
    if args.dry_run:
        return
    source, output = Path(args.source_dir).resolve(), Path(args.output_dir).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be separate from the source directory and its ancestors/descendants")
    if output.exists() and any(output.iterdir()) and not args.overwrite_output:
        raise FileExistsError(f"Nonempty output directory: {output}; choose another or pass --overwrite_output")
    output.mkdir(parents=True, exist_ok=True)
    if args.overwrite_output:
        for name in ("summary.csv", "per_sample.csv", "paired_changes.csv", "corrected_decomposition.csv",
                     "corrected_decomposition.json", "provenance.json"):
            (output / name).unlink(missing_ok=True)
    manifest = {"complete": False, "protocol": "offline-terminal-residual-correction-v1", "args": vars(args),
                "source_dir": str(source), "formal_output_changed": False,
                "correction": "x0_corrected=x0-(xT_observed-L_T(last_prediction)); full strength 1; no GT, clipping, damping or learned parameter",
                "decomposition": "uses independently saved per-step C_pred; corrected_error=C_pred+L_T(last_prediction-GT)",
                "scoring": "same source masks for all outputs; full_* original full-frame metrics; E_H/E_M raw observation RMS",
                "HPGC": "not applied to corrected output; no training or checkpoint modification"}
    write_json(output / "manifest.json", manifest)
    import torch
    manifest["torch_version"] = str(torch.__version__)
    try:
        manifest["correction_source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["correction_source_commit"] = None
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(previous_threads, 4))
        with torch.no_grad():
            controls = oracle_controls()
    finally:
        torch.set_num_threads(previous_threads)
    write_json(output / "oracle_controls.json", controls)
    manifest["oracle_controls"] = controls
    if args.oracle_only:
        manifest["complete"] = True
        write_json(output / "manifest.json", manifest)
        return
    with torch.no_grad():
        replay(args, manifest)


if __name__ == "__main__":
    main()
