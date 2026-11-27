"""Unit regressions for the Wald CDRDI + D2-96 + HPGC full-resolution runner.

Runs without any Augsburg dataset, checkpoint, or GPU.
"""
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from cdrdi_geometry import sampling_coordinates
from infer_augsburg2_wald_full96 import _geometry_from_wald_30m, _tile_rigid


class WaldFull96GeometryTests(unittest.TestCase):
    def test_global_rigid_and_tile_rigid_agree(self):
        height = width = 144
        tile = 96
        full = torch.tensor([[-1.5, 0.6, 0.45]], dtype=torch.float32)
        global_local = torch.zeros(1, 2, height, width)
        gx, gy = sampling_coordinates(
            height, width, full[:, 0], full[:, 1], full[:, 2],
            global_local,
        )
        for top, left in ((0, 0), (0, 48), (48, 0), (48, 48)):
            rigid = _tile_rigid(full, top, left, tile, height, width)
            tx, ty = sampling_coordinates(
                tile, tile, rigid[:, 0], rigid[:, 1], rigid[:, 2],
                torch.zeros(1, 2, tile, tile),
            )
            torch.testing.assert_close(
                tx + left, gx[:, top:top + tile, left:left + tile],
                atol=1e-4, rtol=1e-5,
            )
            torch.testing.assert_close(
                ty + top, gy[:, top:top + tile, left:left + tile],
                atol=1e-4, rtol=1e-5,
            )

    def test_coarse_geometry_estimation_and_native_scale_conversion(self):
        class DummyProcess:
            operator = None

            @staticmethod
            def terminal_observation(x):
                return F.avg_pool2d(x, 3, 3)

        class DummyGeometry:
            def __call__(self, target, msi, p0, **kwargs):
                self.target_shape = tuple(target.shape)
                self.msi_shape = tuple(msi.shape)
                return {
                    "final_rigid": target.new_tensor([[-0.5, 0.25, 0.5]]),
                    "final_local_field": msi.new_zeros((1, 2, *msi.shape[-2:])),
                }

        geometry = DummyGeometry()
        args = SimpleNamespace(
            geometry_mode="wald_cdrdi", geometry_steps=9,
            guard_window=5, guard_min_jac=0.5,
            guard_relative_gain=1e-4,
        )
        lr = np.zeros((48, 48, 242), dtype=np.float32)
        msi = np.zeros((144, 144, 4), dtype=np.float32)
        valid = np.ones((144, 144), dtype=np.uint8)
        srf = torch.zeros(4, 242)
        srf[:, :4] = torch.eye(4)
        rigid10, rigid30 = _geometry_from_wald_30m(
            geometry, DummyProcess(), lr, msi, valid, None, srf,
            args, torch.device("cpu"),
        )
        self.assertEqual(geometry.target_shape, (1, 4, 16, 16))
        self.assertEqual(geometry.msi_shape, (1, 4, 48, 48))
        torch.testing.assert_close(
            rigid30, torch.tensor([[-0.5, 0.25, 0.5]]),
        )
        torch.testing.assert_close(
            rigid10, torch.tensor([[-1.5, 0.75, 0.5]]),
        )


if __name__ == "__main__":
    unittest.main()
