# Augsburg Region-2 Center-Holdout: full S2Diff-MH at Diffusion base_channels=96

This protocol trains **CDRDI -> geometry-aware 96-wide physics-grounded diffusion ->
full HPGC/GCI spectral refiner** from scratch on Wald Center-Holdout. It does
**not** load EnMAP-like 10m hyperspectral ground truth. Original 10m evaluation
uses QNR, Dlambda and Ds only.

The scripts in this branch retain the legacy 64-channel/identity pathways.
The new complete native-scale entrypoint is `infer_augsburg2_wald_full96.py`.

## 0. Empty server: code, environment and raw data

```bash
git clone --branch feature/augsburg-wald-full96 https://github.com/daeyu666/S2Diff-MH.git
cd S2Diff-MH
conda create -n s2diff96 python=3.10 -y
conda activate s2diff96
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Raw Augsburg MDAS data are NOT stored in this Git repository.** Transfer
`Augsburg_data_4_publication/` from the original storage or obtain it from
the data publisher, preserving its directory names:

```text
data/raw/Augsburg/Augsburg_data_4_publication/
  band_242_meta_info.hdr
  sr_deep_model_data/EeteS_EnMAP_30m_deep_train.tif
  sr_deep_model_data/EeteS_EnMAP_30m_deep_valid.tif
  sub_area_2/EeteS_EnMAP_30m_sub_area2.tif
  sub_area_2/Sentinel_2_sub_area2.tif
  entire_city/Sentinel-2.tif
  ...
```

`prepare_augsburg_real.py` also expects the other original MDAS files used
to build the original split cache. Check completeness with the preparation
commands below; do not substitute simulated Sentinel-2 imagery for real MSI.

## 1. Data preparation and train-only radiometric calibration

Run from the repository root, in order. `--s2_platform auto` requires
valid platform metadata; if the source does not identify the platform, use
`--s2_platform S2A` or `S2B` based on the original product metadata.

```bash
python prepare_augsburg_real.py \
  --data_root ./data/raw \
  --cache_root ./data/augsburg_real_cache \
  --s2_platform auto

python prepare_augsburg2_wald.py \
  --real_cache_root ./data/augsburg_real_cache \
  --data_root ./data/raw \
  --output_root ./data/augsburg2_wald \
  --training_area region2 \
  --subarea sub_area_2 \
  --sigma 1.2

python prepare_augsburg2_wald_center_holdout.py \
  --source_wald_root ./data/augsburg2_wald \
  --output_root ./data/augsburg2_wald_center_holdout \
  --test_size_30m 48 --guard_30m 6 \
  --train_patch_30m 24 --train_stride_30m 6

python calibrate_augsburg2_wald_radiometry.py \
  --wald_root ./data/augsburg2_wald_center_holdout \
  --real_cache_root ./data/augsburg_real_cache \
  --output ./data/calibration/Augsburg2_Wald_center_holdout_radiometry.json

python -m unittest discover -s tests -p 'test_augsburg2_wald_full96.py'
python -m unittest discover -s tests -p 'test_augsburg2_wald_center_holdout.py'
```

The 30m test region is 48x48, the corresponding 10m center ROI is
144x144. A surrounding 6-pixel guard is excluded from training.

## 2. Train Wald-only CDRDI (Stage C)

Use conservative, physically validated geometry caps at Wald 30m.
The learning-based geometry network remains width 32; only the **diffusion**
backbone changes width.

```bash
python train_augsburg_real_cdrdi.py \
  --stage train \
  --cache_root ./data/augsburg2_wald_center_holdout \
  --psf_json ./data/augsburg2_wald_center_holdout/wald_psf.json \
  --radiometry_json ./data/calibration/Augsburg2_Wald_center_holdout_radiometry.json \
  --from_scratch \
  --initial_dx_px -0.5 --initial_dy_px 0 \
  --max_translation 1 --max_rotation_deg 0.5 --max_local_px 0.5 \
  --lambda_rigid 0.01 --lambda_local 0.01 \
  --base_channels 32 --control_grid 5 \
  --train_steps 6 --eval_steps 9 \
  --train_patch_size 24 --train_stride 6 --eval_patch_size 48 \
  --batch_size 1 --num_workers 0 \
  --epochs 100 --lr 2e-5 --eval_interval 5 \
  --seed 10 --device cuda:0 \
  --checkpoint_root ./checkpoints/augsburg_real/center_holdout/full96 \
  --log_root ./logs/augsburg_real/center_holdout/full96 \
  --save_name Augsburg2_Wald_center_C

python train_augsburg_real_cdrdi.py \
  --stage test \
  --cache_root ./data/augsburg2_wald_center_holdout \
  --psf_json ./data/augsburg2_wald_center_holdout/wald_psf.json \
  --radiometry_json ./data/calibration/Augsburg2_Wald_center_holdout_radiometry.json \
  --initial_dx_px -0.5 --initial_dy_px 0 \
  --max_translation 1 --max_rotation_deg 0.5 --max_local_px 0.5 \
  --lambda_rigid 0.01 --lambda_local 0.01 \
  --base_channels 32 --control_grid 5 \
  --train_patch_size 24 --train_stride 6 --eval_patch_size 48 \
  --batch_size 1 --num_workers 0 \
  --geometry_checkpoint ./checkpoints/augsburg_real/center_holdout/full96/Augsburg2_Wald_center_C.pth \
  --seed 10 --device cuda:0
```

**Guard:** if no candidate checkpoint passes CDRDI validation constraints,
the pipeline deliberately does not create a best checkpoint. Check the
geometry log before proceeding; do not select by held-out-test QNR.

## 3. Train and test the geometry-aware 96-channel diffusion (Stage D2)

```bash
python train_augsburg_real_diffusion.py \
  --stage train \
  --cache_root ./data/augsburg2_wald_center_holdout \
  --psf_json ./data/augsburg2_wald_center_holdout/wald_psf.json \
  --radiometry_json ./data/calibration/Augsburg2_Wald_center_holdout_radiometry.json \
  --geometry_mode wald_cdrdi \
  --geometry_checkpoint ./checkpoints/augsburg_real/center_holdout/full96/Augsburg2_Wald_center_C.pth \
  --geometry_steps 9 \
  --fixed_dx_px -0.5 --fixed_dy_px 0 \
  --guard_window 5 --guard_min_jac 0.5 --guard_relative_gain 0.0001 \
  --from_scratch \
  --train_patch_size 24 --train_stride 6 --eval_patch_size 48 \
  --batch_size 1 --num_workers 0 \
  --diffusion_steps 12 --base_channels 96 \
  --time_dim 256 --spectral_hidden 8 \
  --epochs 100 --lr 1e-4 --weight_decay 0 \
  --monitor ref_psnr --eval_interval 5 \
  --seed 10 --device cuda:0 \
  --checkpoint_root ./checkpoints/augsburg_real/center_holdout/full96 \
  --log_root ./logs/augsburg_real/center_holdout/full96 \
  --save_name Augsburg2_Wald_center_D2_C_base96

python train_augsburg_real_diffusion.py \
  --stage test \
  --cache_root ./data/augsburg2_wald_center_holdout \
  --psf_json ./data/augsburg2_wald_center_holdout/wald_psf.json \
  --radiometry_json ./data/calibration/Augsburg2_Wald_center_holdout_radiometry.json \
  --geometry_mode wald_cdrdi \
  --geometry_checkpoint ./checkpoints/augsburg_real/center_holdout/full96/Augsburg2_Wald_center_C.pth \
  --geometry_steps 9 \
  --fixed_dx_px -0.5 --fixed_dy_px 0 \
  --guard_window 5 --guard_min_jac 0.5 --guard_relative_gain 0.0001 \
  --train_patch_size 24 --train_stride 6 --eval_patch_size 48 \
  --batch_size 1 --num_workers 0 \
  --diffusion_steps 12 --base_channels 96 \
  --time_dim 256 --spectral_hidden 8 \
  --seed 10 --device cuda:0 \
  --diffusion_checkpoint ./checkpoints/augsburg_real/center_holdout/full96/Augsburg2_Wald_center_D2_C_base96.pth \
  --metrics_json ./logs/augsburg_real/center_holdout/full96/D2_C_base96_test.json
```

## 4. Train and test the full HPGC-Refiner (Stage E)

New Wald C support in `train_augsburg_real_gigi.py`:
- reuses Wald C `rigid_only` closure-backtracking instead of the legacy
  Real-E geometry path;
- loads geometry architecture from the strict Wald-only checkpoint;
- disables data augmentation for Wald; monitors SAM of high-heterogeneity
  regions on the **validation** split;
- permits scratch training (no previous refiner checkpoint);
- refuses mismatched geometry, diffusion, calibration, or test split.

```bash
python train_augsburg_real_gigi.py \
  --stage train --from_scratch --variant full \
  --cache_root ./data/augsburg2_wald_center_holdout \
  --psf_json ./data/augsburg2_wald_center_holdout/wald_psf.json \
  --radiometry_json ./data/calibration/Augsburg2_Wald_center_holdout_radiometry.json \
  --geometry_mode wald_cdrdi \
  --geometry_checkpoint ./checkpoints/augsburg_real/center_holdout/full96/Augsburg2_Wald_center_C.pth \
  --diffusion_checkpoint ./checkpoints/augsburg_real/center_holdout/full96/Augsburg2_Wald_center_D2_C_base96.pth \
  --geometry_steps 9 \
  --fixed_dx_px -0.5 --fixed_dy_px 0 \
  --guard_window 5 --guard_min_jac 0.5 --guard_relative_gain 0.0001 \
  --train_patch_size 24 --train_stride 6 --eval_patch_size 48 \
  --batch_size 1 --num_workers 0 \
  --diffusion_steps 12 --diffusion_base_channels 96 \
  --time_dim 256 --spectral_hidden 8 \
  --refine_hidden 64 --heads 4 \
  --epochs 100 --lr 2e-5 \
  --monitor ref_sam_high --eval_interval 5 \
  --seed 10 --device cuda:0 \
  --checkpoint_root ./checkpoints/augsburg_real/center_holdout/full96 \
  --log_root ./logs/augsburg_real/center_holdout/full96 \
  --save_name Augsburg2_Wald_center_E_full96

python train_augsburg_real_gigi.py \
  --stage test --variant full \
  --cache_root ./data/augsburg2_wald_center_holdout \
  --psf_json ./data/augsburg2_wald_center_holdout/wald_psf.json \
  --radiometry_json ./data/calibration/Augsburg2_Wald_center_holdout_radiometry.json \
  --geometry_mode wald_cdrdi \
  --geometry_checkpoint ./checkpoints/augsburg_real/center_holdout/full96/Augsburg2_Wald_center_C.pth \
  --diffusion_checkpoint ./checkpoints/augsburg_real/center_holdout/full96/Augsburg2_Wald_center_D2_C_base96.pth \
  --refiner_checkpoint ./checkpoints/augsburg_real/center_holdout/full96/Augsburg2_Wald_center_E_full96.pth \
  --geometry_steps 9 \
  --fixed_dx_px -0.5 --fixed_dy_px 0 \
  --guard_window 5 --guard_min_jac 0.5 --guard_relative_gain 0.0001 \
  --train_patch_size 24 --train_stride 6 --eval_patch_size 48 \
  --batch_size 1 --num_workers 0 \
  --diffusion_steps 12 --diffusion_base_channels 96 \
  --time_dim 256 --spectral_hidden 8 \
  --refine_hidden 64 --heads 4 \
  --seed 10 --device cuda:0
```

## 5. Original-resolution 10m inference + QNR (final full model)

```bash
python infer_augsburg2_wald_full96.py \
  --center_holdout \
  --checkpoint ./checkpoints/augsburg_real/center_holdout/full96/Augsburg2_Wald_center_D2_C_base96.pth \
  --geometry_checkpoint ./checkpoints/augsburg_real/center_holdout/full96/Augsburg2_Wald_center_C.pth \
  --refiner_checkpoint ./checkpoints/augsburg_real/center_holdout/full96/Augsburg2_Wald_center_E_full96.pth \
  --base_channels 96 --diffusion_steps 12 \
  --time_dim 256 --spectral_hidden 8 \
  --geometry_steps 9 \
  --fixed_dx_px -0.5 --fixed_dy_px 0 \
  --guard_window 5 --guard_min_jac 0.5 --guard_relative_gain 0.0001 \
  --refine_hidden 64 --heads 4 --variant full \
  --tile_size 96 --tile_stride 48 \
  --save_root ./outputs/augsburg2_wald_center_holdout/full96 \
  --seed 10 --device cuda:0 --write_tif
```

Outputs under `./outputs/augsburg2_wald_center_holdout/full96/`:

```text
Augsburg2_Wald_heldout_HSI.npy
Augsburg2_Wald_heldout_HSI.tif
Augsburg2_Wald_heldout_QNR.json
Augsburg2_Wald_heldout_protocol.json
```

The native inference script estimates rigid-only CDRDI geometry on the
30m Wald scale (by degrading the *observed* center HSI30 to HSI90 and
area-reducing the *observed* MSI10 to MSI30), multiplies translations by
3 for 10m physical acquisition, preserves rotation, and adjusts crop-centered
rigid sampling for each inference tile. It then runs diffusion96,
the physical-residual-guided HPGC-Refiner, and QNR on the **final** fused
10m/242-band hyperspectral image.

**Important scientific limitation:** transferring a deformation estimated at
the Wald 30m scale to native 10m uses a shared physical displacement
assumption; it is not a separately validated native-scale geometric ground
truth. The Wald holdout PSNR/SAM and native QNR are distinct evaluations.
Do not use test QNR to select a checkpoint. Compare this full Wald-C96
result only against methods using the same heldout ROI, sensor/radiometry,
QNR implementation and training split.

## 6. Preflight regression tests

```bash
python -m unittest discover -s tests -p 'test_augsburg2_wald_full96.py' -v
python -m unittest discover -s tests -p 'test_augsburg2_wald_cdrdi.py' -v
python -m unittest discover -s tests -p 'test_augsburg2_wald_d2_abc.py' -v
python -m unittest discover -s tests -p 'test_augsburg2_wald_qnr.py' -v
```
