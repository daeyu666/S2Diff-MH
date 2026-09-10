"""Compatibility loading for legacy S2Diff Raw-Direct checkpoints."""

from __future__ import annotations

import torch

from .predictor_raw_direct import RawMSIDirectPredictor, extract_legacy_raw_direct_state_dict


def load_legacy_raw_direct_checkpoint(
    model: RawMSIDirectPredictor,
    path: str,
    map_location="cpu",
):
    """Load a trusted local legacy checkpoint, including PyTorch 2.6 files.

    Old S2Diff checkpoints may contain custom configuration objects.  They are
    user-produced local experiment files, so weights_only=False is required for
    backward compatibility on recent PyTorch versions.
    """
    try:
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)

    state, ignored = extract_legacy_raw_direct_state_dict(checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    return {
        "ignored_legacy_keys": sorted(ignored),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
