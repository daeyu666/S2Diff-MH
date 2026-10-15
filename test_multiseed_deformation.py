"""Main-table mixed-model test: Registered once; Warp 10 geometry seeds x 10 cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

import train_innovation3_gigi_cdrdi as pipeline
from test_fixed_deformation import build_predictor
from multiseed_deformation_protocol import (
    PROTOCOL_ID, add_multiseed_args, file_sha256, run_multiseed_suite, validate_settings,
)


def parse_args(argv=None):
    parser = pipeline.build_arg_parser(fixed_suite=True)
    parser.description = __doc__
    parser.set_defaults(stage="test")
    parser.add_argument("--dry_run", action="store_true")
    add_multiseed_args(parser, include_cases=False)
    args = parser.parse_args(argv)
    if args.stage != "test":
        parser.error("This entrypoint is test-only; use mixed-trained checkpoints")
    validate_settings(args.geometry_seeds, args.cases, args.max_samples)
    if (args.test_size, args.patch_size, args.stride, args.scale_ratio,
            args.mtf_nyquist, args.psf_truncate) != (128, 64, 32, 4, 0.2, 3.0):
        parser.error("Main-table protocol requires test=128, patch=64, stride=32, scale=4, MTF=0.2, truncate=3")
    if (args.max_translation, args.max_rotation_deg, args.max_local_px,
            args.control_grid, args.min_jacobian) != (4.0, 2.0, 4.0, 5, 0.5):
        parser.error("Main-table geometry is fixed at 4px/2deg/4px, grid=5, min Jacobian=0.5")
    if args.resume_output and args.overwrite_output:
        parser.error("Choose resume_output OR overwrite_output")
    if not args.output_dir:
        tag = hashlib.sha256(",".join(map(str, args.geometry_seeds)).encode()).hexdigest()[:8]
        name = f"{args.variant}_{len(args.geometry_seeds)}seeds_{args.cases}cases_{tag}_eval{args.eval_seed}"
        if args.max_samples:
            name += f"_SMOKE{args.max_samples}"
        args.output_dir = str(Path("outputs/multiseed_deformation") / args.dataset / name)
    return args


def main(argv=None):
    args = parse_args(argv)
    print(json.dumps({"method": "S2Diff-MH", "protocol": PROTOCOL_ID, **vars(args)}, indent=2), flush=True)
    if args.dry_run:
        return
    args = pipeline._resolve_dataset_checkpoints(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu explicitly")
    pipeline.set_seed(args.seed)
    cfg = pipeline._config(args)
    _, loader, info = pipeline.build_loaders(cfg)
    process = pipeline.build_progressive_process(cfg)
    predict, registered, details = build_predictor(args, info, process, device, include_registered=True)
    refiner_path = args.refiner_checkpoint or pipeline._checkpoint_path(args)
    checkpoints = {"geometry": args.geometry_checkpoint, "diffusion": args.diffusion_checkpoint,
                   "refiner": refiner_path}
    metadata = {"method": "S2Diff-MH", "dataset": args.dataset, "training_stage": "mixed",
                "data_root": str(Path(args.data_root).resolve()), "degradation": "physical",
                "mtf_nyquist": args.mtf_nyquist, "psf_truncate": args.psf_truncate,
                "checkpoint_sha256": {k: file_sha256(v) for k, v in checkpoints.items()},
                "registered_path": "same adapted diffusion/GIGI, no geometry estimation; original FINAL_REGISTERED",
                "warp_path": "CDRDI estimated geometry -> diffusion -> terminal GIGI", **details}
    return run_multiseed_suite(
        loader.dataset, predict, process.operator.degrade, predict_registered=registered,
        device=device, output_dir=args.output_dir, metadata=metadata,
        geometry_seeds=args.geometry_seeds, cases=args.cases, eval_seed=args.eval_seed,
        control_grid=args.control_grid, min_jacobian=args.min_jacobian,
        max_translation=args.max_translation, max_rotation_deg=args.max_rotation_deg,
        max_local_px=args.max_local_px, scale_ratio=args.scale_ratio,
        max_samples=args.max_samples, resume_output=args.resume_output,
        overwrite_output=args.overwrite_output,
    )


if __name__ == "__main__":
    main()
