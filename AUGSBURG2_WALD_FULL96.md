# Augsburg Center-Holdout: original A/B geometry + Diffusion96 + HPGC

**Important correction:** Earlier Wald A/B/C experiments selected **A/identity**
for native-scale QNR: Wald reduced-resolution A PSNR=31.852742, SAM=3.938751;
B (fixed dx=-0.5, dy=0) PSNR=31.800527, SAM=4.197228;
C (learned CDRDI) PSNR=31.816066, SAM=4.085203. The original
`infer_augsburg2_wald.py` is A/identity only and did not run HPGC.

Therefore, to preserve the **actual original selected training protocol** when
changing **only diffusion base_channels 64 -> 96** and **adding the requested
HPGC spectral refinement**, run the **A/identity** commands here. No CDRDI
checkpoint or CDRDI training is required.

The B/fixed-shift alternative, if desired, requires only `--geometry_mode
wald_fixed` and `--fixed_dx_px -0.5 --fixed_dy_px 0`; it does **not** learn
CDRDI. Keep A and B checkpoints, logs, and results in separate directories.
**Do not refer to Wald A/B as a learned CDRDI demonstration.**

## 0. Fresh server

```bash
git clone -b feature/augsburg-wald-full96 https://github.com/daeyu666/S2Diff-MH.git
cd S2Diff-MH

conda create -n s2diff96 python=3.10 -y
conda activate s2diff96
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Raw MDAS data are not in GitHub. Transfer the original
`data/raw/Augsburg/Augsburg_data_4_publication/` directory from storage
before proceeding (including the original train/validation and
`sub_area_2` EnMAP30 / real Sentinel-2 data).

## 1. Prepare held-out Wald data and calibration

```bash
python prepare_augsburg_real.py \
  --data_root ./data/raw \
  --cache_root ./data/augsburg_real_cache \
  --s2_platform auto

python prepare_augsburg2_wald.py \
  --real_cache_root ./data/augsburg_real_cache \
  --data_root ./data/raw \
  --output_root ./data/augsburg2_wald \
  --training_area region2 --subarea sub_area_2 --sigma 1.2

python prepare_augsburg2_wald_center_holdout.py \
  --source_wald_root ./data/augsburg2_wald \
  --output_root ./data/augsburg2_wald_center_holdout \
  --test_size_30m 48 --guard_30m 6 \
  --train_patch_30m 24 --train_stride_30m 6

python calibrate_augsburg2_wald_radiometry.py \
  --wald_root ./data/augsburg2_wald_center_holdout \
  --real_cache_root ./data/augsburg_real_cache \
  --output ./data/calibration/Augsburg2_Wald_center_holdout_radiometry.json
```

If platform detection fails, identify S2A/S2B from original imagery metadata;
do not arbitrarily select a platform.

## 2. Define experiment paths (A is the reproduced selected configuration)

```bash
export WALD=./data/augsburg2_wald_center_holdout
export RAD=./data/calibration/Augsburg2_Wald_center_holdout_radiometry.json

export MODE=identity
export BRANCH=A
# Optional separately evaluated fixed-shift experiment:
# export MODE=wald_fixed
# export BRANCH=B

export OUT=./checkpoints/augsburg_real/center_holdout/full96_${BRANCH}
export LOG=./logs/augsburg_real/center_holdout/full96_${BRANCH}
export DIFF=${OUT}/Augsburg2_Wald_center_D2_${BRANCH}_base96.pth
export REF=${OUT}/Augsburg2_Wald_center_E_${BRANCH}_full96.pth
mkdir -p "$OUT" "$LOG"
```

## 3. Train and test 96-channel physical diffusion

```bash
python train_augsburg_real_diffusion.py \
  --stage train \
  --cache_root "$WALD" \
  --psf_json "$WALD/wald_psf.json" \
  --radiometry_json "$RAD" \
  --geometry_mode "$MODE" \
  --fixed_dx_px -0.5 --fixed_dy_px 0 \
  --from_scratch \
  --train_patch_size 24 --train_stride 6 --eval_patch_size 48 \
  --batch_size 1 --num_workers 0 \
  --diffusion_steps 12 --base_channels 96 \
  --time_dim 256 --spectral_hidden 8 \
  --epochs 100 --lr 1e-4 --weight_decay 0 \
  --monitor ref_psnr --eval_interval 5 \
  --seed 10 --device cuda:0 \
  --checkpoint_root "$OUT" --log_root "$LOG" \
  --save_name "Augsburg2_Wald_center_D2_${BRANCH}_base96"

python train_augsburg_real_diffusion.py \
  --stage test \
  --cache_root "$WALD" \
  --psf_json "$WALD/wald_psf.json" \
  --radiometry_json "$RAD" \
  --geometry_mode "$MODE" \
  --fixed_dx_px -0.5 --fixed_dy_px 0 \
  --train_patch_size 24 --train_stride 6 --eval_patch_size 48 \
  --batch_size 1 --diffusion_steps 12 --base_channels 96 \
  --time_dim 256 --spectral_hidden 8 \
  --device cuda:0 --diffusion_checkpoint "$DIFF" \
  --metrics_json "$LOG/D2_${BRANCH}_base96_test.json"
```

For A, geometry is identity. Passing fixed-dx parameters does **not** apply
a shift when `--geometry_mode identity`.

## 4. Train and test full HPGC/Global Correlation Refiner

The same frozen Diffusion96 checkpoint is used; CDRDI is not instantiated
in A or B. The refiner trains **from scratch** because the new server has no
previous refiner checkpoint. It uses the existing full architecture
(`hidden=64`, `heads=4`, tangent correction enabled), monitored on
validation high-heterogeneity SAM.

```bash
python train_augsburg_real_gigi.py \
  --stage train --from_scratch --variant full \
  --cache_root "$WALD" \
  --psf_json "$WALD/wald_psf.json" \
  --radiometry_json "$RAD" \
  --geometry_mode "$MODE" \
  --fixed_dx_px -0.5 --fixed_dy_px 0 \
  --diffusion_checkpoint "$DIFF" \
  --train_patch_size 24 --train_stride 6 --eval_patch_size 48 \
  --batch_size 1 --num_workers 0 \
  --diffusion_steps 12 --diffusion_base_channels 96 \
  --time_dim 256 --spectral_hidden 8 \
  --refine_hidden 64 --heads 4 \
  --epochs 100 --lr 2e-5 \
  --monitor ref_sam_high --eval_interval 5 \
  --seed 10 --device cuda:0 \
  --checkpoint_root "$OUT" --log_root "$LOG" \
  --save_name "Augsburg2_Wald_center_E_${BRANCH}_full96"

python train_augsburg_real_gigi.py \
  --stage test --variant full \
  --cache_root "$WALD" \
  --psf_json "$WALD/wald_psf.json" \
  --radiometry_json "$RAD" \
  --geometry_mode "$MODE" \
  --fixed_dx_px -0.5 --fixed_dy_px 0 \
  --diffusion_checkpoint "$DIFF" \
  --refiner_checkpoint "$REF" \
  --train_patch_size 24 --train_stride 6 --eval_patch_size 48 \
  --batch_size 1 --diffusion_steps 12 --diffusion_base_channels 96 \
  --time_dim 256 --spectral_hidden 8 \
  --refine_hidden 64 --heads 4 \
  --seed 10 --device cuda:0
```

## 5. Native 10m 242-band reconstruction with HPGC and QNR

```bash
python infer_augsburg2_wald_full96.py \
  --center_holdout \
  --geometry_mode "$MODE" \
  --fixed_dx_px -0.5 --fixed_dy_px 0 \
  --checkpoint "$DIFF" --refiner_checkpoint "$REF" \
  --base_channels 96 --diffusion_steps 12 \
  --time_dim 256 --spectral_hidden 8 \
  --refine_hidden 64 --heads 4 --variant full \
  --tile_size 96 --tile_stride 48 \
  --save_root "./outputs/augsburg2_wald_center_holdout/full96_${BRANCH}" \
  --seed 10 --device cuda:0 --write_tif
```

Outputs:

```text
outputs/augsburg2_wald_center_holdout/full96_A/
  Augsburg2_Wald_heldout_HSI.npy
  Augsburg2_Wald_heldout_HSI.tif
  Augsburg2_Wald_heldout_QNR.json
  Augsburg2_Wald_heldout_protocol.json
```

For fixed B, use `full96_B`. The B native 10m operator uses fixed
translation `dx=-1.5,dy=0` in 10m pixels (same -15m shift), rather than
incorrectly reusing -0.5 native pixels.

**Research caveat:** Original A native-QNR baseline used diffusion **without**
HPGC. The new A+96+HPGC is a new model configuration, not just widening the
backbone. Do not attribute any QNR difference exclusively to width without
an A+64+HPGC control.

## 6. Tests

```bash
python -m unittest discover -s tests -p 'test_augsburg2_wald_full96.py' -v
python -m unittest discover -s tests -p 'test_augsburg2_wald_center_holdout.py' -v
python -m unittest discover -s tests -p 'test_augsburg2_wald_d2_abc.py' -v
```

The code is pushed to the PR branch, but no actual GPU/data end-to-end
training or QNR test was performed by the assistant. Review logs and best
checkpoint outputs before proceeding to the next stage.
