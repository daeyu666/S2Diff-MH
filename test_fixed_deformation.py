"""Six fixed-strength tests for the complete S2Diff-MH CDRDI/diffusion/GIGI model."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

import train_innovation3_gigi_cdrdi as pipeline
from fixed_deformation_protocol import STRENGTHS, add_suite_args, run_fixed_suite


def parse_args(argv=None):
    parser = pipeline.build_arg_parser(fixed_suite=True)
    parser.description = __doc__
    parser.set_defaults(stage="test")
    parser.add_argument("--dry_run", action="store_true")
    add_suite_args(parser, include_cases=False)
    args = parser.parse_args(argv)
    if args.stage != "test":
        parser.error("This entrypoint only supports --stage test")
    if (args.test_size, args.patch_size, args.stride, args.scale_ratio) != (128, 64, 32, 4):
        parser.error("Fixed comparison protocol requires test_size=128, patch_size=64, stride=32, scale_ratio=4")
    if (args.mtf_nyquist, args.psf_truncate) != (0.2, 3.0):
        parser.error("Fixed comparison protocol requires physical MTF=0.2 and PSF truncate=3")
    if (args.max_translation, args.max_rotation_deg, args.max_local_px) != (4.0, 2.0, 4.0):
        parser.error("Keep model training-time geometry bounds at 4/2/4; the suite sets test inputs independently")
    if args.cases < 1 or len(set(args.levels)) != len(args.levels):
        parser.error("cases must be >=1 and levels must be distinct")
    return args


def build_predictor(args, info, base_process, device):
    """Model inference consumes Y_H/Y_M only; synthesis geometry stays private."""
    geometry_model, diffusion_model = pipeline._build_frozen_models(args, info, device)
    projector = pipeline._build_projector(args, info, device)
    refiner = pipeline._build_refiner(args, info, device)
    path = args.refiner_checkpoint or pipeline._checkpoint_path(args)
    state = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_dataset = state.get("extra", {}).get("dataset")
    if checkpoint_dataset is not None and checkpoint_dataset != args.dataset:
        raise ValueError("refiner checkpoint/test dataset mismatch")
    epoch, best = pipeline._load_training_checkpoint(refiner, path, device)
    refiner.eval()
    srf = torch.as_tensor(info["srf_weights"], device=device, dtype=torch.float32)
    p0 = base_process.operator

    @torch.no_grad()
    def predict(lr_hsi, hr_msi):
        rigid, local = pipeline._estimate_geometry(
            geometry_model, y_h=lr_hsi, hr_msi=hr_msi, p0=p0, srf=srf,
            steps=args.geometry_steps,
        )
        process = pipeline._process_from_estimate(base_process, rigid, local)
        base = pipeline.reconstruct_from_terminal_lr(
            diffusion_model, process, lr_hsi, target_size=tuple(hr_msi.shape[-2:]),
            hr_msi=hr_msi,
        )
        residual = pipeline._physical_residual(process, projector, lr_hsi, base)
        heterogeneity = pipeline.ranked_msi_heterogeneity(hr_msi)
        return refiner(base, hr_msi, residual, heterogeneity)

    details = {"refiner_checkpoint": str(path), "checkpoint_epoch": epoch,
               "best_training_monitor": best, "geometry_checkpoint": args.geometry_checkpoint,
               "diffusion_checkpoint": args.diffusion_checkpoint,
               "geometry_steps": args.geometry_steps, "diffusion_steps": args.diffusion_steps,
               "variant": args.variant, "model_geometry_bounds": [4.0, 2.0, 4.0],
               "model_args": vars(args)}
    return predict, details


def main():
    args = parse_args()
    if not args.output_dir:
        args.output_dir = str(Path("outputs/fixed_deformation") / args.dataset
                              / f"{args.variant}_seed{args.seed}_cases{args.cases}_eval{args.eval_seed}")
    print(json.dumps({"method": "S2Diff-MH", "dataset": args.dataset,
                      "output_dir": args.output_dir,
                      "strengths": [asdict(STRENGTHS[i-1]) for i in args.levels]}, indent=2), flush=True)
    if args.dry_run:
        return
    args = pipeline._resolve_dataset_checkpoints(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu for CPU testing")
    pipeline.set_seed(args.seed)
    cfg = pipeline._config(args)
    _, test_loader, info = pipeline.build_loaders(cfg)
    base_process = pipeline.build_progressive_process(cfg)
    predict, details = build_predictor(args, info, base_process, device)
    run_fixed_suite(
        test_loader.dataset, predict, base_process.operator.degrade,
        device=device, output_dir=args.output_dir,
        metadata={"method": "S2Diff-MH", "dataset": args.dataset,
                  "data_root": str(Path(args.data_root).resolve()),
                  "degradation": "physical", "mtf_nyquist": args.mtf_nyquist,
                  "psf_truncate": args.psf_truncate, **details},
        cases=args.cases, seed=args.seed, eval_seed=args.eval_seed, levels=args.levels,
        control_grid=args.control_grid, min_jacobian=args.min_jacobian,
    )


if __name__ == "__main__":
    main()
