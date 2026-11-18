"""Run the joint PaviaU benchmark, or profile our full pipeline in isolation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def profile_ours(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--_pair", type=Path, required=True)
    p.add_argument("--_result", type=Path, required=True)
    p.add_argument("--_runner", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--cpu_threads", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--warmup", type=int, required=True)
    p.add_argument("--repeats", type=int, required=True)
    p.add_argument("--no_checkpoints", action="store_true")
    p.add_argument("--ours_args", nargs=argparse.REMAINDER, default=[])
    opts = p.parse_args(argv)

    # Import the common measurement functions without adding the other
    # repository to sys.path: both repos have a config and model namespace.
    common = runpy.run_path(str(opts._runner))
    import os
    import train_innovation3_gigi_cdrdi as pipeline
    from test_fixed_deformation import build_predictor

    device = common["runtime"](opts.device, opts.cpu_threads, opts.seed)
    lr, msi, srf = common["load_pair"](opts._pair, device)
    parser = pipeline.build_arg_parser(fixed_suite=True)
    parser.set_defaults(stage="test")
    args = parser.parse_args(opts.ours_args)
    if (args.dataset, args.stage, args.variant, args.test_size, args.patch_size, args.stride,
            args.scale_ratio, args.mtf_nyquist, args.psf_truncate, args.srf_interp,
            args.max_translation, args.max_rotation_deg, args.max_local_px,
            args.control_grid, args.min_jacobian) != (
            "PaviaU", "test", "full", 128, 64, 32, 4, 0.2, 3.0, "pchip",
            4.0, 2.0, 4.0, 5, 0.5):
        parser.error("The joint benchmark requires the full PaviaU fixed comparison protocol")
    args.device, args.seed = str(device), opts.seed
    if opts.no_checkpoints:
        if args.refiner_checkpoint or args.geometry_checkpoint or args.diffusion_checkpoint:
            parser.error("--no_checkpoints does not accept explicit geometry/diffusion/refiner weights")
    else:
        refiner_path = args.refiner_checkpoint or os.path.join(
            args.checkpoint_root, args.save_name or
            f"PaviaU_innovation3_gigi_cdrdi_full_k{args.geometry_steps}"
        )
        if not args.refiner_checkpoint and not refiner_path.endswith(".pth"):
            refiner_path += ".pth"
        args.refiner_checkpoint = str(Path(refiner_path).resolve())
        if not Path(args.refiner_checkpoint).is_file():
            raise FileNotFoundError(f"Refiner checkpoint not found: {args.refiner_checkpoint}")
        args = pipeline._resolve_dataset_checkpoints(args)
        args.geometry_checkpoint = str(Path(args.geometry_checkpoint).resolve())
        args.diffusion_checkpoint = str(Path(args.diffusion_checkpoint).resolve())

    # The original full (Warp-path) predictor is used on the same registered
    # pair as the comparisons. Its geometry estimator is still executed.
    info = {"n_bands": lr.shape[1], "n_msi_bands": msi.shape[1], "srf_weights": srf}
    process = pipeline.build_progressive_process(pipeline._config(args))
    modules = []
    predict, details = build_predictor(args, info, process, device, profile_modules=modules,
                                       load_weights=not opts.no_checkpoints)
    row = common["measure"](lambda: predict(lr, msi), modules, device,
                            warmup=opts.warmup, repeats=opts.repeats,
                            threads=opts.cpu_threads, seed=opts.seed)
    row.update(diffusion_steps=details["diffusion_steps"], geometry_steps=details["geometry_steps"],
               backbone_width=args.diffusion_base_channels)
    if not opts.no_checkpoints:
        row["checkpoint"] = "; ".join(details[key] for key in
                                     ("geometry_checkpoint", "diffusion_checkpoint", "refiner_checkpoint"))
    opts._result.write_text(json.dumps(row, allow_nan=False), encoding="utf-8")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--_pair" in argv:
        return profile_ours(argv)
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument("--comparison_root", type=Path, default=ROOT.parent / "comparison_experiments")
    # Keep ours pipeline arguments opaque to the front-end parser.
    split = argv.index("--ours_args") if "--ours_args" in argv else len(argv)
    args, benchmark_args = p.parse_known_args(argv[:split])
    runner = args.comparison_root.expanduser().resolve() / Path(__file__).name
    if not runner.is_file():
        p.error(f"Comparison benchmark not found: {runner} (use --comparison_root)")
    return subprocess.run([sys.executable, str(runner), "--s2diff_root", str(ROOT),
                           *benchmark_args, *argv[split:]]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
