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
from visualize_augsburg2_wald_center_holdout import resolve_existing_outputs, visualize
from infer_augsburg2_wald import parse_args as parse_s2diff_inference_args


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

    def test_center_inference_cli_routes_to_s2diff_own_files(self):
        with patch("sys.argv", ["infer_augsburg2_wald.py", "--center_holdout", "--write_tif"]):
            args = parse_s2diff_inference_args()
        self.assertEqual(args.wald_root, "./data/augsburg2_wald_center_holdout")
        self.assertEqual(args.save_root, "./outputs/augsburg2_wald_center_holdout")
        self.assertEqual(
            args.checkpoint,
            "./checkpoints/augsburg_real/center_holdout/Augsburg2_Wald_center_D2_A.pth",
        )
        self.assertIn("center_holdout_radiometry", args.radiometry_json)
        self.assertFalse(args.skip_qnr)
        self.assertTrue(args.write_tif)

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


    def test_visualization_missing_models_only_explains_separate_inference(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            wald = root / "wald"
            wald.mkdir()
            self.create_cache(str(wald))
            s2_output = root / "S2Diff" / "Augsburg2_Wald_heldout_HSI.npy"
            ua_output = root / "UAFL" / "Augsburg2_Wald_UAFL_heldout_HSI.npy"
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_existing_outputs(
                    [("S2Diff", str(s2_output)), ("UAFL", str(ua_output))],
                    wald_root=str(wald),
                )
            err = str(ctx.exception)
            self.assertIn("python infer_augsburg2_wald.py --center_holdout", err)
            self.assertIn("python comparison/UAFL/infer_augsburg2_wald.py --center_holdout", err)
            self.assertIn(str(s2_output), err)
            self.assertIn(str(ua_output), err)

    def test_visualization_requires_true_heldout_shape(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            wald = root / "wald"
            wald.mkdir()
            self.create_cache(str(wald))
            old_full_scene = root / "Augsburg2_Wald_full_HSI.npy"
            np.save(old_full_scene, np.zeros((300, 360, 242), np.float32))
            with self.assertRaisesRegex(ValueError, "Do not use old full-scene"):
                resolve_existing_outputs(
                    [("S2Diff", str(old_full_scene))], wald_root=str(wald)
                )

    def test_visualization_uses_existing_outputs_in_each_model_directory(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            wald = root / "wald"
            (wald / "full").mkdir(parents=True)
            self.create_cache(str(wald))
            np.save(
                wald / "full" / "hr_msi.npy",
                np.random.default_rng(4).random((300, 360, 4), dtype=np.float32),
            )
            model_files = []
            for name, file_name in (
                ("S2Diff", "Augsburg2_Wald_heldout_HSI.npy"),
                ("UAFL", "Augsburg2_Wald_UAFL_heldout_HSI.npy"),
            ):
                folder = root / name / "outputs"
                folder.mkdir(parents=True)
                target = folder / file_name
                np.save(
                    target,
                    np.random.default_rng(len(name)).random((144, 144, 242), dtype=np.float32),
                )
                model_files.append((name, str(target)))
            checked = resolve_existing_outputs(model_files, wald_root=str(wald))
            self.assertEqual(checked, model_files)
            comparison = root / "figures" / "comparison.png"
            visualize(
                str(wald), str(root / "figures" / "aux"), checked,
                savefig=str(comparison),
            )
            self.assertTrue(comparison.is_file())
            self.assertTrue(
                (root / "S2Diff" / "outputs" / "Augsburg2_Wald_heldout_RGB.png").is_file()
            )
            self.assertTrue(
                (root / "UAFL" / "outputs" / "Augsburg2_Wald_UAFL_heldout_RGB.png").is_file()
            )
            manifest = json.loads(
                (root / "figures" / "aux" / "visualization_provenance.json").read_text()
            )
            self.assertFalse(manifest["inference_triggered_by_visualization"])
            self.assertEqual(set(manifest["model_rgb_files_in_own_result_folders"]), {"S2Diff", "UAFL"})


if __name__=="__main__":
    unittest.main()
