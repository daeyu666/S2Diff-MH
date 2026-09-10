"""Verify a legacy S2Diff Raw-Direct checkpoint without training."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping

import torch

from models import RawMSIDirectPredictor, load_legacy_raw_direct_checkpoint


def _load_raw(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state(obj):
    if isinstance(obj, Mapping):
        for key in ("model", "model_state_dict", "state_dict"):
            value = obj.get(key)
            if isinstance(value, Mapping):
                return value
    return obj


def main():
    p = argparse.ArgumentParser(description="Verify legacy V3 Raw-Direct checkpoint compatibility")
    p.add_argument("checkpoint")
    p.add_argument("--bands", type=int, default=103)
    p.add_argument("--msi_bands", type=int, default=4)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--spectral_hidden", type=int, default=8)
    args = p.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    raw = _load_raw(args.checkpoint)
    state = _extract_state(raw)
    print(f"checkpoint: {args.checkpoint}")
    print(f"file_size_mb: {os.path.getsize(args.checkpoint) / (1024 ** 2):.3f}")
    if isinstance(raw, Mapping):
        print("top_level_keys:", sorted(str(k) for k in raw.keys()))
        for key in ("epoch", "best_metric", "best_psnr"):
            if key in raw:
                print(f"{key}: {raw[key]}")
        extra = raw.get("extra")
        if isinstance(extra, Mapping):
            for key in ("dataset", "degradation_mode", "diffusion_steps", "predictor_version", "msi_ablation", "srf_band_set"):
                if key in extra:
                    print(f"extra.{key}: {extra[key]}")

    if not isinstance(state, Mapping):
        raise TypeError("Could not locate a state_dict in checkpoint")

    normalized = {}
    for key, value in state.items():
        key2 = key[7:] if str(key).startswith("module.") else str(key)
        normalized[key2] = value

    print(f"state_dict_keys: {len(normalized)}")
    for probe in ("in_proj.weight", "msi_in.0.weight", "out_conv.weight"):
        if probe in normalized:
            print(f"{probe}: {tuple(normalized[probe].shape)}")

    model = RawMSIDirectPredictor(
        n_bands=args.bands,
        n_msi_bands=args.msi_bands,
        total_steps=args.steps,
        base_channels=args.base_channels,
        spectral_hidden=args.spectral_hidden,
    )
    report = load_legacy_raw_direct_checkpoint(model, args.checkpoint, map_location="cpu")
    print("ignored_legacy_keys:", len(report["ignored_legacy_keys"]))
    print("missing_keys:", report["missing_keys"])
    print("unexpected_keys:", report["unexpected_keys"])

    ok = not report["missing_keys"] and not report["unexpected_keys"]
    print("COMPATIBLE:", "YES" if ok else "NO")
    if ok:
        print("This checkpoint is structurally compatible with the clean RawMSIDirectPredictor.")
    else:
        print("Do not use it yet; inspect the missing/unexpected keys above.")


if __name__ == "__main__":
    main()
