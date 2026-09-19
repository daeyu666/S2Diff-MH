import numpy as np
import torch

from diagnose_msi_gt_heterogeneity_transfer import (
    _grid_starts,
    _patch_coords,
    _top_overlap,
    _quantile_separation,
)


def test_grid_starts_cover_scene_edges():
    starts = _grid_starts(length=1000, patch=128, grid=3)
    assert starts[0] == 0
    assert starts[-1] == 872
    assert len(starts) == 3


def test_patch_coords_grid_count():
    coords = _patch_coords(512, 640, patch=128, grid=3)
    assert len(coords) == 9
    assert coords[0] == (0, 0)
    assert coords[-1] == (384, 512)


def test_top_overlap_identical_rank_maps_is_perfect():
    x = torch.linspace(0.0, 1.0, 100)
    out = _top_overlap(x, x, fraction=0.25)
    assert abs(out["recall"] - 1.0) < 1e-6
    assert abs(out["precision"] - 1.0) < 1e-6
    assert abs(out["jaccard"] - 1.0) < 1e-6


def test_quantile_separation_positive_for_matching_ranks():
    x = torch.linspace(0.0, 1.0, 100)
    out = _quantile_separation(x, x, fraction=0.25)
    assert out["delta_gt_rank"] > 0.4
    assert out["ratio_gt_rank"] > 2.0
