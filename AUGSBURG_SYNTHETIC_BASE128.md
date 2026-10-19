# Synthetic Augsburg x4: 128-channel capacity experiment

This experiment is **only** for the synthetic Augsburg x4 benchmark:
MDAS `deep_train` for training, `deep_valid` for validation,
and `sub_area_1` for the final synthetic test. It is not the Augsburg-Real/Wald
Sentinel-2 experiment.

The existing 64-channel checkpoints and logs remain untouched. All new
artifacts are isolated under `checkpoints/augsburg_base128/` and
`logs/augsburg_base128/`.

## What changes

- Reconstruction backbone: `base_channels=128` for Stage A, B, D1 and D2.
- Stage E diffusion backbone: `diffusion_base_channels=128`.
- GIGI spectral refiner: `refine_hidden=128` for Stage B and E, so both refiner checkpoints
  remain structurally compatible.
- Keep `spectral_hidden=8`, `time_dim=256`, `scale_ratio=4`, 12 diffusion steps,
  64x64 train patches / stride 32, seed 10, loss weights and degradation
  protocol unchanged.
- Stage C CDRDI geometry estimator stays at `base_channels=32` and reuses the
  existing 500-epoch checkpoint. It is a separate geometry network, not the
  242-band HSI reconstruction backbone, so it does not need retraining for
  this width ablation.

Doubling the main feature width increases memory use. The commands retain the
original batch sizes for a controlled comparison. If CUDA OOM occurs, lower
only the affected stage's batch size and record that deviation.

## Stage A — registered Raw-MSI Direct, 200 epochs

Run from the repository root:

~~~bash
python main.py \
  --stage train \
  --dataset Augsburg \
  --data_root ./data/raw \
  --patch_size 64 --stride 32 --test_size 128 --scale_ratio 4 \
  --degradation_mode physical \
  --diffusion_steps 12 \
  --predictor raw_direct \
  --base_channels 128 \
  --time_dim 256 \
  --spectral_hidden 8 \
  --epochs 200 --batch_size 4 --lr 1e-4 \
  --lambda_l1 1.0 --lambda_sam 0.1 --lambda_deg 0.0 \
  --seed 10 \
  --checkpoint_root ./checkpoints/augsburg_base128 \
  --log_root ./logs/augsburg_base128/stageA \
  --save_name Augsburg_physical_raw_direct_base128
~~~

Checkpoint:

`checkpoints/augsburg_base128/innovation1/Augsburg_physical_raw_direct_base128.pth`

Test the Stage-A model on the official synthetic test split:

~~~bash
python main.py \
  --stage test \
  --dataset Augsburg \
  --data_root ./data/raw \
  --patch_size 64 --stride 32 --test_size 128 --scale_ratio 4 \
  --degradation_mode physical --diffusion_steps 12 \
  --predictor raw_direct --base_channels 128 \
  --time_dim 256 --spectral_hidden 8 \
  --seed 10 \
  --checkpoint_root ./checkpoints/augsburg_base128 \
  --save_name Augsburg_physical_raw_direct_base128
~~~

## Stage B — registered GIGI spectral refinement, 100 epochs

Use the new Stage-A checkpoint, not the old 64-channel checkpoint:

~~~bash
python train_innovation3_gigi.py \
  --stage train --variant full \
  --dataset Augsburg --data_root ./data/raw \
  --patch_size 64 --stride 32 --test_size 128 --scale_ratio 4 \
  --diffusion_steps 12 \
  --base_channels 128 --time_dim 256 --spectral_hidden 8 \
  --refine_hidden 128 --heads 4 \
  --baseline_checkpoint ./checkpoints/augsburg_base128/innovation1/Augsburg_physical_raw_direct_base128.pth \
  --baseline_checkpoint_type standard \
  --epochs 100 --batch_size 2 --lr 2e-4 \
  --seed 10 \
  --checkpoint_root ./checkpoints/augsburg_base128/innovation3_gigi \
  --log_root ./logs/augsburg_base128/stageB \
  --save_name Augsburg_innovation3_gigi_full_base128
~~~

Checkpoint:

`checkpoints/augsburg_base128/innovation3_gigi/Augsburg_innovation3_gigi_full_base128.pth`

Stage-B test:

~~~bash
python train_innovation3_gigi.py \
  --stage test --variant full \
  --dataset Augsburg --data_root ./data/raw \
  --patch_size 64 --stride 32 --test_size 128 --scale_ratio 4 \
  --diffusion_steps 12 \
  --base_channels 128 --time_dim 256 --spectral_hidden 8 \
  --refine_hidden 128 --heads 4 \
  --baseline_checkpoint ./checkpoints/augsburg_base128/innovation1/Augsburg_physical_raw_direct_base128.pth \
  --baseline_checkpoint_type standard \
  --refiner_checkpoint ./checkpoints/augsburg_base128/innovation3_gigi/Augsburg_innovation3_gigi_full_base128.pth \
  --seed 10
~~~

## Stage C — CDRDI geometry checkpoint (reuse)

Reuse the existing synthetic-Augsburg geometry checkpoint:

`checkpoints/cdrdi_stage1/Augsburg_recursive_k6_finalonly_500ep_lr1e4.pth`

Verify it exists before Stage D2 / E:

~~~bash
test -f ./checkpoints/cdrdi_stage1/Augsburg_recursive_k6_finalonly_500ep_lr1e4.pth \
  && echo "Stage C geometry checkpoint found"
~~~

Do not change the geometry width from 32 to 128 for this experiment. If this
checkpoint is absent, locate the existing 500-epoch Stage-C artifact before
retraining; do not substitute a different geometry checkpoint silently.

## Stage D1 — oracle deformation-aware diffusion, 100 epochs

Initialize from the **128-channel Stage-A checkpoint**:

~~~bash
python train_cdrdi_diffusion_oracle.py \
  --dataset Augsburg --data_root ./data/raw \
  --patch_size 64 --stride 32 --test_size 128 --scale_ratio 4 \
  --diffusion_steps 12 \
  --max_translation 4 --max_rotation_deg 2 --max_local_px 4 \
  --control_grid 5 --min_jacobian 0.5 \
  --identity_probability 0.10 --min_strength 0.20 \
  --base_channels 128 --time_dim 256 --spectral_hidden 8 \
  --epochs 100 --batch_size 4 --lr 1e-5 \
  --eval_cases 3 --eval_interval 5 \
  --init_checkpoint ./checkpoints/augsburg_base128/innovation1/Augsburg_physical_raw_direct_base128.pth \
  --checkpoint_root ./checkpoints/augsburg_base128/cdrdi_stage2 \
  --log_root ./logs/augsburg_base128/stageD1 \
  --save_name Augsburg_oracle_deform_diffusion_100ep_base128 \
  --seed 10
~~~

Checkpoint:

`checkpoints/augsburg_base128/cdrdi_stage2/Augsburg_oracle_deform_diffusion_100ep_base128.pth`

## Stage D2 — estimated-geometry diffusion, 60 epochs

Use the same Stage-C geometry checkpoint and the new 128-channel D1 checkpoint:

~~~bash
python train_cdrdi_diffusion_estimated.py \
  --dataset Augsburg --data_root ./data/raw \
  --patch_size 64 --stride 32 --test_size 128 --scale_ratio 4 \
  --diffusion_steps 12 \
  --max_translation 4 --max_rotation_deg 2 --max_local_px 4 \
  --control_grid 5 --min_jacobian 0.5 \
  --identity_probability 0.10 --min_strength 0.20 \
  --geometry_steps 9 --geometry_base_channels 32 \
  --geometry_checkpoint ./checkpoints/cdrdi_stage1/Augsburg_recursive_k6_finalonly_500ep_lr1e4.pth \
  --base_channels 128 --time_dim 256 --spectral_hidden 8 \
  --epochs 60 --batch_size 4 --lr 5e-6 \
  --eval_cases 10 --eval_interval 5 \
  --init_checkpoint ./checkpoints/augsburg_base128/cdrdi_stage2/Augsburg_oracle_deform_diffusion_100ep_base128.pth \
  --checkpoint_root ./checkpoints/augsburg_base128/cdrdi_stage2 \
  --log_root ./logs/augsburg_base128/stageD2 \
  --save_name Augsburg_estimated_deform_diffusion_k9_60ep_base128 \
  --seed 10
~~~

Checkpoint:

`checkpoints/augsburg_base128/cdrdi_stage2/Augsburg_estimated_deform_diffusion_k9_60ep_base128.pth`

## Stage E — final GIGI + CDRDI model, 150 epochs

All reconstruction-backbone and refiner checkpoints must be from the new
128-width run. Only the Stage-C geometry checkpoint is shared:

~~~bash
python train_innovation3_gigi_cdrdi.py \
  --stage train --variant full \
  --dataset Augsburg --data_root ./data/raw \
  --patch_size 64 --stride 32 --test_size 128 --scale_ratio 4 \
  --diffusion_steps 12 \
  --max_translation 4 --max_rotation_deg 2 --max_local_px 4 \
  --control_grid 5 --min_jacobian 0.5 \
  --identity_probability 0.10 --min_strength 0.20 \
  --geometry_steps 9 --geometry_base_channels 32 \
  --geometry_checkpoint ./checkpoints/cdrdi_stage1/Augsburg_recursive_k6_finalonly_500ep_lr1e4.pth \
  --diffusion_base_channels 128 --time_dim 256 --spectral_hidden 8 \
  --diffusion_checkpoint ./checkpoints/augsburg_base128/cdrdi_stage2/Augsburg_estimated_deform_diffusion_k9_60ep_base128.pth \
  --refine_hidden 128 --heads 4 \
  --init_refiner_checkpoint ./checkpoints/augsburg_base128/innovation3_gigi/Augsburg_innovation3_gigi_full_base128.pth \
  --epochs 150 --batch_size 2 --lr 2e-4 \
  --seed 10 \
  --checkpoint_root ./checkpoints/augsburg_base128/innovation3_gigi_cdrdi \
  --log_root ./logs/augsburg_base128/stageE \
  --save_name Augsburg_innovation3_gigi_cdrdi_full_k9_base128
~~~

Final checkpoint:

`checkpoints/augsburg_base128/innovation3_gigi_cdrdi/Augsburg_innovation3_gigi_cdrdi_full_k9_base128.pth`

Final synthetic test:

~~~bash
python train_innovation3_gigi_cdrdi.py \
  --stage test --variant full \
  --dataset Augsburg --data_root ./data/raw \
  --patch_size 64 --stride 32 --test_size 128 --scale_ratio 4 \
  --diffusion_steps 12 \
  --max_translation 4 --max_rotation_deg 2 --max_local_px 4 \
  --control_grid 5 --min_jacobian 0.5 \
  --identity_probability 0.10 --min_strength 0.20 \
  --geometry_steps 9 --geometry_base_channels 32 \
  --geometry_checkpoint ./checkpoints/cdrdi_stage1/Augsburg_recursive_k6_finalonly_500ep_lr1e4.pth \
  --diffusion_base_channels 128 --time_dim 256 --spectral_hidden 8 \
  --diffusion_checkpoint ./checkpoints/augsburg_base128/cdrdi_stage2/Augsburg_estimated_deform_diffusion_k9_60ep_base128.pth \
  --refine_hidden 128 --heads 4 \
  --init_refiner_checkpoint ./checkpoints/augsburg_base128/innovation3_gigi/Augsburg_innovation3_gigi_full_base128.pth \
  --refiner_checkpoint ./checkpoints/augsburg_base128/innovation3_gigi_cdrdi/Augsburg_innovation3_gigi_cdrdi_full_k9_base128.pth \
  --seed 10
~~~

Record both `FINAL_REGISTERED` and `FINAL_WARP` metrics. Compare them with the existing 64-channel
run under the same `sub_area_1` test protocol. Do not overwrite or mix old 64-channel
checkpoints/logs with this run.

## Artifact map

- Stage A: `checkpoints/augsburg_base128/innovation1/`
- Stage B: `checkpoints/augsburg_base128/innovation3_gigi/`
- Stage D1/D2: `checkpoints/augsburg_base128/cdrdi_stage2/`
- Stage E: `checkpoints/augsburg_base128/innovation3_gigi_cdrdi/`
- Logs: `logs/augsburg_base128/`
