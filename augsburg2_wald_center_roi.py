"""Select the *untrained central ROI* from original Augsburg Region-2 observations.

Same implementation in comparison_experiments/UAFL and S2Diff-MH.
This is a DRT-Net-inspired spatial split, not an asserted pixel-exact
replica of paper Fig.13 (author ROI coordinates were not published).
"""
from __future__ import annotations

import json
from pathlib import Path


PROTOCOL = "Augsburg2-Wald-center-holdout-v1"


def read_roi(cache_root):
    path = Path(cache_root) / "roi.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        roi = json.load(f)
    if roi.get("protocol_id") != PROTOCOL:
        raise ValueError(f"Unknown central holdout manifest: {path}")
    b30 = list(map(int, roi["test_bbox_30m"]))
    b10 = list(map(int, roi["test_bbox_10m"]))
    if b10 != [3 * x for x in b30]:
        raise ValueError("Native 10m ROI must have exactly 3x 30m coordinates")
    if not (b30[0] < b30[2] and b30[1] < b30[3]):
        raise ValueError("Invalid center holdout bbox")
    return roi


def crop_heldout(cache_root, full_lr, full_msi, full_mask,
                 *, ckpt_protocol_id=None, ckpt_bbox_30m=None):
    """Returns LR30, MSI10, MASK10, native MSI pixel offsets, suffix, split-ID.

    Rejects reuse of full-scene-trained weights in center-heldout evaluation.
    """
    roi = read_roi(cache_root)
    expected = PROTOCOL if roi else "legacy_full_region_wald"
    actual = ckpt_protocol_id or "legacy_full_region_wald"
    if actual != expected:
        raise ValueError(
            f"Wald checkpoint spatial split {actual!r} differs from cache {expected!r}; "
            "retrain on central spatial holdout; do not reuse a full-scene checkpoint"
        )
    if roi is None:
        if ckpt_bbox_30m not in (None, []):
            raise ValueError("Full-region checkpoint unexpectedly stores held-out bbox")
        return full_lr, full_msi, full_mask, 0, 0, "full", expected
    bbox30 = list(map(int, roi["test_bbox_30m"]))
    if ckpt_bbox_30m != bbox30:
        raise ValueError("Wald checkpoint test ROI differs from loaded center-holdout manifest")
    y0, x0, y1, x1 = bbox30
    yy0, xx0, yy1, xx1 = map(int, roi["test_bbox_10m"])
    if not (0 <= y0 < y1 <= full_lr.shape[0] and 0 <= x0 < x1 <= full_lr.shape[1]
            and yy1 <= full_msi.shape[0] and xx1 <= full_msi.shape[1]):
        raise ValueError("Center-holdout bbox extends beyond observed source images")
    selected_lr = full_lr[y0:y1, x0:x1]
    selected_msi = full_msi[yy0:yy1, xx0:xx1]
    selected_mask = full_mask[yy0:yy1, xx0:xx1]
    if selected_msi.shape[:2] != (3 * selected_lr.shape[0], 3 * selected_lr.shape[1]):
        raise ValueError("Center HSI30 and MSI10 crop sizes inconsistent")
    return selected_lr, selected_msi, selected_mask, yy0, xx0, "heldout", expected


def geotiff_transform(full_transform_6, pixel_y0, pixel_x0):
    from affine import Affine
    return Affine(*full_transform_6) * Affine.translation(pixel_x0, pixel_y0)
