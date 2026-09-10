"""Standalone sanity check for the Innovation1 degradation trajectory."""

import argparse
import torch

from degradations import ProgressiveDegradation, build_degradation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["physical", "gaussian_bicubic", "bicubic"], default="physical")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--bands", type=int, default=31)
    p.add_argument("--size", type=int, default=128)
    args = p.parse_args()

    kwargs = {"mtf_nyquist": 0.2, "truncate": 3.0} if args.mode == "physical" else {}
    if args.mode == "gaussian_bicubic":
        kwargs = {"sigma": 2.0, "kernel_size": 5}
    operator = build_degradation(args.mode, scale_ratio=args.scale, **kwargs)
    process = ProgressiveDegradation(operator, total_steps=args.steps)
    x = torch.rand(1, args.bands, args.size, args.size)
    process.assert_terminal_closure(x)
    print("terminal_closure: PASS")
    print("t scale strength mean std step_l1")
    previous = None
    for t in range(args.steps + 1):
        state = process.state(t)
        current = process.state_at(x, t)
        step = 0.0 if previous is None else (current - previous).abs().mean().item()
        print(
            f"{t:02d} {state.scale:d} {state.strength:.4f} "
            f"{current.mean().item():.6f} {current.std().item():.6f} {step:.6f}"
        )
        previous = current


if __name__ == "__main__":
    main()
