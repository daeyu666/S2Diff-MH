"""Convenience entrypoint: fixed-strength suite with elastic local deformation.

This wraps test_fixed_deformation.py and changes only the local deformation
family. Rigid motion, physical observation synthesis, metrics, seeds and model
checkpoints remain identical to the shared fixed-deformation protocol.
"""
from __future__ import annotations

import sys

import test_fixed_deformation as base


def main():
    argv = sys.argv[1:]
    if any(arg == "--local_deformation" or arg.startswith("--local_deformation=") for arg in argv):
        raise SystemExit("test_fixed_deformation_elastic.py fixes --local_deformation=elastic")
    base.main(["--local_deformation", "elastic", *argv])


if __name__ == "__main__":
    main()
