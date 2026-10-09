"""CPU tests for center-heldout Augsburg Region-2 geometry and split provenance."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from augsburg2_wald_center_roi import (
    PROTOCOL, crop_heldout, geotiff_transform, read_roi
)
from augsburg_real import AugsburgRealDataset
from prepare_augsburg2_wald_center_holdout import heldout_train_tile_candidates, _rects_intersect
from visualize_augsburg2_wald_center_holdout import prepare_missing_outputs


class AugsburgCenterHoldoutROITests(unittest.TestCase):
    def create_cache(self, root):
        p=Path(root)
        (p/"roi.json").write_text(json.dumps({
            "protocol_id": PROTOCOL,
            "source_region":"sub_area_2",
            "test_bbox_30m":[24,36,72,84],
            "test_bbox_10m":[72,108,216,252],
            "guard_pixels_30m":6,
        }), encoding="utf-8")

    def test_heldout_crops_exact_same_region_at_both_resolutions(self):
        with TemporaryDirectory() as d:
            self.create_cache(d)
            lr=np.arange(100*120,dtype=np.float32).reshape(100,120,1)
            msi=np.arange(300*360,dtype=np.float32).reshape(300,360,1)
            mask=np.ones((300,360),dtype=np.uint8)
            out=crop_heldout(
                d,lr,msi,mask,ckpt_protocol_id=PROTOCOL,
                ckpt_bbox_30m=[24,36,72,84]
            )
            low,hi,valid,oy,ox,suffix,protocol=out
            self.assertEqual(low.shape,(48,48,1))
            self.assertEqual(hi.shape,(144,144,1))
            self.assertEqual(valid.shape,(144,144))
            self.assertEqual((oy,ox,suffix,protocol),(72,108,"heldout",PROTOCOL))
            self.assertEqual(low[0,0,0],lr[24,36,0])
            self.assertEqual(hi[0,0,0],msi[72,108,0])
            self.assertEqual(hi[-1,-1,0],msi[215,251,0])

    def test_pretrained_full_scene_checkpoint_is_rejected(self):
        with TemporaryDirectory() as d:
            self.create_cache(d)
            a=np.zeros((100,120,242),np.float32)
            b=np.zeros((300,360,4),np.float32)
            c=np.ones((300,360),np.uint8)
            with self.assertRaisesRegex(ValueError,"retrain"):
                crop_heldout(d,a,b,c,ckpt_protocol_id="legacy_full_region_wald")
            with self.assertRaisesRegex(ValueError,"test ROI differs"):
                crop_heldout(d,a,b,c,ckpt_protocol_id=PROTOCOL,
                             ckpt_bbox_30m=[0,0,48,48])

    def test_legacy_does_not_fabricate_a_spatial_split(self):
        with TemporaryDirectory() as d:
            a=np.zeros((100,120,242),np.float32)
            b=np.zeros((300,360,4),np.float32)
            c=np.ones((300,360),np.uint8)
            out=crop_heldout(d,a,b,c)
            self.assertEqual(out[-2],"full")
            self.assertEqual(out[-1],"legacy_full_region_wald")
            self.assertIs(out[0],a)

    def test_manifest_pixel_scale_must_match(self):
        with TemporaryDirectory() as d:
            self.create_cache(d)
            p=Path(d)/"roi.json"
            payload=json.loads(p.read_text())
            payload["test_bbox_10m"]=[72,108,217,252]
            p.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,"exactly 3x"):
                read_roi(d)


    def test_real_loader_never_sees_center_plus_psf_guard(self):
        box=(24,36,72,84)
        forbidden=(18,30,78,90)
        coords=heldout_train_tile_candidates(
            (99,120),box,patch=24,stride=6,guard=6
        )
        self.assertGreater(len(coords),10)
        for y,x,ph,pw in coords:
            self.assertFalse(_rects_intersect((y,x,y+ph,x+pw),forbidden))
        with TemporaryDirectory() as d:
            train=Path(d)/"train"
            train.mkdir()
            for name,arr in (
                ("gt",np.zeros((99,120,242),np.float32)),
                ("lr_hsi",np.zeros((33,40,242),np.float32)),
                ("hr_msi",np.zeros((99,120,4),np.float32)),
                ("valid_mask",np.ones((99,120),np.uint8)),
            ):
                np.save(train/(name+".npy"),arr)
            (train/"meta.json").write_text(json.dumps({
                "forbidden_bbox_30m":list(forbidden),
                "msi_source":"real_Sentinel_2_Wald_30m",
            }))
            ds=AugsburgRealDataset(
                d,"train",train_patch_size=24,train_stride=6,
                eval_patch_size=48,min_valid_fraction=.8,augment=False
            )
            self.assertEqual(len(ds.samples),len(coords))
            for y,x,ph,pw in ds.samples:
                self.assertFalse(_rects_intersect((y,x,y+ph,x+pw),forbidden))


    def test_missing_reconstructions_report_all_missing_checkpoints_before_inference(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            wald=root/"wald"
            wald.mkdir()
            self.create_cache(str(wald))
            missing_output1=root/"ours"/"Augsburg2_Wald_heldout_HSI.npy"
            missing_output2=root/"theirs"/"Augsburg2_Wald_UAFL_heldout_HSI.npy"
            missing_ours=root/"not_trained_ours.pth"
            missing_theirs=root/"not_trained_theirs.pth.tar"
            calibration=root/"not_calibrated.json"
            with patch("visualize_augsburg2_wald_center_holdout.subprocess.run") as mocked:
                with self.assertRaises(FileNotFoundError) as exc:
                    prepare_missing_outputs(
                        [("S2Diff",str(missing_output1)),("UAFL",str(missing_output2))],
                        wald_root=str(wald),
                        uafl_repo=str(root/"not_cloned_uafl"),
                        radiometry_json=str(calibration),
                        s2diff_checkpoint=str(missing_ours),
                        uafl_checkpoint=str(missing_theirs)
                    )
                message=str(exc.exception)
                self.assertIn(str(missing_ours),message)
                self.assertIn(str(missing_theirs),message)
                self.assertIn(str(calibration),message)
                mocked.assert_not_called()

    def test_auto_infer_reconstructs_missing_s2diff_npy_using_matching_checkpoint(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            wald=root/"wald"
            wald.mkdir()
            self.create_cache(str(wald))
            ckpt=root/"center_only.pth"
            ckpt.touch()
            calibration=root/"center_only_radiometry.json"
            calibration.write_text("{}")
            target=root/"outputs"/"Augsburg2_Wald_heldout_HSI.npy"
            commands=[]
            def fake_inference(command, *, cwd, check):
                self.assertTrue(check)
                self.assertIn("--skip_qnr", command)
                self.assertEqual(command[command.index("--checkpoint")+1],str(ckpt))
                self.assertEqual(command[command.index("--wald_root")+1],str(wald))
                self.assertEqual(command[command.index("--radiometry_json")+1],str(calibration))
                self.assertEqual(command[command.index("--save_root")+1],str(target.parent))
                commands.append((command,cwd))
                target.touch()
            with patch("visualize_augsburg2_wald_center_holdout.subprocess.run",
                       side_effect=fake_inference) as mocked:
                methods=prepare_missing_outputs(
                    [("S2Diff",str(target))],
                    wald_root=str(wald),
                    uafl_repo=str(root/"uafl"),
                    radiometry_json=str(calibration),
                    s2diff_checkpoint=str(ckpt),
                    uafl_checkpoint=str(root/"absent_UAFL.pth")
                )
                self.assertEqual(methods,[("S2Diff",str(target.resolve()))])
                mocked.assert_called_once()
                self.assertEqual(len(commands),1)
                # Second invocation reuses existing output; does not
                # request a GPU run or require an unrelated UAFL checkpoint.
                prepare_missing_outputs(
                    [("S2Diff",str(target))],
                    wald_root=str(wald),
                    uafl_repo=str(root/"uafl"),
                    radiometry_json=str(calibration),
                    s2diff_checkpoint=str(ckpt),
                    uafl_checkpoint=str(root/"absent_UAFL.pth")
                )
                mocked.assert_called_once()

    def test_no_auto_infer_explains_stage_test_does_not_create_npy(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            wald=root/"wald"
            wald.mkdir()
            self.create_cache(str(wald))
            with patch("visualize_augsburg2_wald_center_holdout.subprocess.run") as mocked:
                with self.assertRaisesRegex(FileNotFoundError,"stage test"):
                    prepare_missing_outputs(
                        [("S2Diff",str(root/"Augsburg2_Wald_heldout_HSI.npy"))],
                        wald_root=str(wald),
                        uafl_repo=str(root),
                        radiometry_json=str(root/"absent.json"),
                        s2diff_checkpoint=str(root/"absent.pth"),
                        uafl_checkpoint=str(root/"absent.tar"),
                        auto_infer=False
                    )
                mocked.assert_not_called()


if __name__=="__main__":
    unittest.main()
