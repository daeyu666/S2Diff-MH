"""Convenience entrypoint for TARCNet-style elastic local deformation tests.

The TARCNet paper describes zero-mean random displacement fields, Gaussian
smoothing with sigma, scaling by deformation factor alpha, and bilinear warping.
This wrapper keeps the existing fixed rigid component and physical observation
model Y_H=P0(W_phi(X)); it replaces only the local non-rigid field.

Default: alpha=10, sigma=5, with no Jacobian rejection (paper-style stress test).
Use --tarc_physical to enforce the project's min-Jacobian constraint.
"""
from __future__ import annotations

import argparse
import sys

import test_fixed_deformation as base


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--tarc_physical", action="store_true")
    known, remaining = pre.parse_known_args(sys.argv[1:])
    if any(arg == "--local_deformation" or arg.startswith("--local_deformation=")
           for arg in remaining):
        raise SystemExit("This entrypoint fixes the local deformation family.")
    family = "tarc_physical" if known.tarc_physical else "tarc"
    base.main([
        "--local_deformation", family,
        "--tarc_alpha", "10",
        "--tarc_sigma_hr_px", "5",
        *remaining,
    ])


if __name__ == "__main__":
    main()
