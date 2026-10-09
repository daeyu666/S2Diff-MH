# Augsburg-2 center-heldout Wald protocol (DRT-Net inspired)

This is a reproducible *central holdout*, **not a pixel-exact reproduction**
of the DRT-Net Fig. 13 red box. DRT-Net's experimental section says it
reserves a central testing subregion and trains on the remainder; the paper
does not state the original-scale Augsburg-2 ROI pixel coordinates.
Source: https://www.researchgate.net/publication/392846638_DRT-Net_Dual-Branch_Rectangular_Transformer_with_Contrastive_Learning_for_Hyperspectral_Super-Resolution

## Source and split

- Dataset: original MDAS `sub_area_2`, 100x120x242 observed 30m HSI
  and 300x360x4 **real Sentinel-2** 10m MSI.
- Native 30m center test: `[row24:72, col36:84]`, 48x48 HSI30.
- Native 10m center test: `[row72:216, col108:252]`, 144x144 MSI10.
- Forbidden training region: `[row18:78, col30:90]` on the
  30m HSI grid, i.e. holdout plus a 6x30m-pixel Gaussian-PSF guard.
- Train target: observed HSI30 outside the forbidden region;
  simulated HSI90 via fixed Gaussian sigma1.2 / stride3;
  MSI30 from **real** S2 MSI10 area averaging x3.
- Train patches: 24x24 HSI30, stride6, with hard exclusion of every
  patch intersecting the forbidden region; no random flips/rotations.
- Model selection: geographically disjoint original `deep_valid`
  30m Wald validation with 48x48 eval blocks.
- Held-out test: only 48x48 observed center HSI30, using 16x16 HSI90
  and 48x48 MSI30 Wald inputs. No EnMAP10 references.
- Actual 10m inference: only 144x144 central real-MSI rectangle.
  No genuine HR-HSI10 GT; QNR/Dlambda/Ds use the untrained ROI only.
- Visualization: full original 10m Sentinel-2 MSI as an overview with
  a red rectangle and the held-out 10m reconstruction beside it.

This setup is an intra-scene spatial holdout, not a geographically
independent generalization experiment. Pixel positions were chosen
explicitly for physical x3/x2 sampling, not copied from DRT-Net.

## 1. Prepare isolated cache from the existing Wald cache

From the `S2Diff-MH` root:

```bash
git pull
python prepare_augsburg2_wald_center_holdout.py \
  --source_wald_root ./data/augsburg2_wald \
  --output_root ./data/augsburg2_wald_center_holdout \
  --test_size_30m 48 --guard_30m 6 \
  --train_patch_30m 24 --train_stride_30m 6

python -m unittest discover -s tests -p "test_augsburg2_wald_center_holdout.py"
```

**Before training, fit new radiometry only from pixels outside the test
rectangle and the 30m PSF guard. Do not reuse the old full-region calibration.**

```bash
python calibrate_augsburg2_wald_radiometry.py \
  --wald_root ./data/augsburg2_wald_center_holdout \
  --real_cache_root ./data/augsburg_real_cache \
  --output ./data/calibration/Augsburg2_Wald_center_holdout_radiometry.json
```

Do not use `--overwrite`. The original cache and checkpoints are preserved.

## 2. Re-train S2Diff-MH, starting from scratch

```bash
python run_augsburg2_wald_abc.py --center_holdout --branch A --stage train
python run_augsburg2_wald_abc.py --center_holdout --branch A --stage test

# Optional fixed-offset B branch
python run_augsburg2_wald_abc.py --center_holdout --branch B --stage train
python run_augsburg2_wald_abc.py --center_holdout --branch B --stage test
```

Models/checkpoints/logs automatically use isolated
`checkpoints/augsburg_real/center_holdout/` paths.

For branch C, **do not reuse** the existing full-region CDRDI (it saw
the center). First train a *new CDRDI* on
`data/augsburg2_wald_center_holdout` using
`--from_scratch --train_patch_size 24 --train_stride 6 --eval_patch_size 48`.
Save it as `checkpoints/augsburg_real/center_holdout/Augsburg2_Wald_center_C.pth`.
Then train/test `--center_holdout --branch C`.

## 3. Full-native-resolution inference only on test ROI

```bash
python infer_augsburg2_wald.py \
  --wald_root ./data/augsburg2_wald_center_holdout \
  --checkpoint ./checkpoints/augsburg_real/center_holdout/Augsburg2_Wald_center_D2_A.pth \
  --radiometry_json ./data/calibration/Augsburg2_Wald_center_holdout_radiometry.json \
  --save_root ./outputs/augsburg2_wald_center_holdout \
  --tile_size 96 --tile_stride 48 --write_tif
```

Outputs `Augsburg2_Wald_heldout_HSI.npy` (144x144x242),
`Augsburg2_Wald_heldout_HSI.tif` (correct ROI GeoTIFF origin),
and `Augsburg2_Wald_heldout_QNR.json` (QNR only on the heldout ROI).
A checkpoint trained with the old full-region split is refused.

## 4. Produce Fig.13 style illustration

```bash
python visualize_augsburg2_wald_center_holdout.py \
  --wald_root ./data/augsburg2_wald_center_holdout \
  --method S2Diff ./outputs/augsburg2_wald_center_holdout/Augsburg2_Wald_heldout_HSI.npy \
  --method UAFL ../comparison_experiments/comparison/UAFL/outputs/augsburg2_wald_center_holdout/Augsburg2_Wald_UAFL_heldout_HSI.npy \
  --savefig ./figures/Augsburg_holdout_S2Diff_vs_UAFL_RGB.png
```

The combined figure is saved to the explicit `--savefig` path (parent directory created automatically). The default copy is also saved at
`outputs/augsburg2_wald_center_holdout/fig13/Augsburg2_center_heldout_Fig13_style.png`, together with the ROI previews and provenance JSON.

**Do not compare legacy whole-region QNR or test metrics to center-heldout
scores as though they share a spatial protocol.** Re-train both methods.
