# Innovation 2 Stage-2: CDRDI x Physical Degradation Diffusion

Stage-0 established that fixed `P0/R0` turns the original full-blind problem into a stable deformation-only inverse problem. Stage-1 established that shared-weight physical-residual recursion is consistently better than one-shot geometry prediction without flow supervision.

Stage-2 tests whether the estimated acquisition geometry can be inserted into Innovation-1's physical trajectory without inverse-warping the observed LR-HSI.

## Acquisition operator

The clean HR-HSI `X` is defined in the reliable MSI coordinate system. For every degraded sensor state `t>=1`, use

```text
A_(t,phi) = D_t o W_phi
```

and the normalized adjoint lift

```text
A_(t,phi)^dagger(y)
  = A_(t,phi)^*(y) / A_(t,phi)^* A_(t,phi)(1)

A_(t,phi)^*
  = W_phi^* o D_t^*
```

`W_phi^*` is implemented as the transpose/scatter of the same bilinear border sampler used by the forward `grid_sample`. It is **not** an inverse image warp.

The HR computational state is

```text
A~_(t,phi)(X) = A_(t,phi)^dagger A_(t,phi)(X)
```

and the reverse step remains

```text
x_(t-1) = x_t + A~_(t-1,phi)(X_hat_0) - A~_(t,phi)(X_hat_0)
```

`phi` is fixed for the complete reverse trajectory. `D_t` changes with `t`.

`t=0` is treated as the clean latent boundary, not a sensor acquisition state:

```text
A~_(0,phi) = I
```

This preserves the Innovation-1 reconstruction target in the reliable MSI coordinate system.

## Stage-2A: operator closure and frozen-checkpoint diagnostic

Before retraining the super-resolution predictor, compare four paths using the same synthetic deformation:

```text
registered : original registered Innovation-1 baseline
naive      : deformed LR-HSI, geometry ignored
oracle     : deformed LR-HSI, GT phi in A_(t,phi)
estimated  : deformed LR-HSI, learned CDRDI phi in A_(t,phi)
```

Run the operator tests and algebraic closure diagnostic:

```bash
pytest -q tests/test_deformation_aware_diffusion.py
python validate_cdrdi_diffusion_oracle_closure.py \
  --dataset PaviaU \
  --device cuda \
  --cases 3 \
  --max_translation 4 \
  --max_rotation_deg 2 \
  --max_local_px 4 \
  --seed 10
```

The measured oracle reverse closure is at numerical precision (`MAX_TRAJECTORY_ABS=5.960e-07`, `MAX_FINAL_ABS=5.960e-07`), so the deformation-aware forward/adjoint/reverse algebra is closed. Directly inserting the new operator into the old registered Raw-Direct checkpoint nevertheless hurts reconstruction because the predictor has never seen `A~_(t,phi)` states.

## Stage-2B: oracle deformation-aware diffusion adaptation

Fine-tune Raw-Direct with GT acquisition geometry only to isolate predictor state-distribution shift. Geometry is fixed per sample, LR-HSI is never inverse-warped, and the HR-MSI remains in the reliable coordinate system.

```bash
python train_cdrdi_diffusion_oracle.py \
  --dataset PaviaU \
  --device cuda \
  --epochs 50 \
  --lr 1e-5 \
  --max_translation 4 \
  --max_rotation_deg 2 \
  --max_local_px 4 \
  --eval_cases 3 \
  --legacy_raw_direct_checkpoint ./checkpoints/legacy/PaviaU_innovation1_physical_v3_raw_direct.pth \
  --seed 10
```

After adaptation, the measured result is approximately:

```text
REGISTERED 44.7375 dB
NAIVE      34.8507 dB
ORACLE     44.4998 dB
```

This confirms that the large frozen-checkpoint failure was predictor state shift rather than an invalid acquisition operator.

## Stage-2C: learned CDRDI geometry in the adapted diffusion process

Using the Stage-2B diffusion checkpoint and K=4 CDRDI geometry estimate over 10 synthetic cases gives:

```text
AVG_GEOM_EPE_HR = 0.598079 px
REGISTERED       = 44.7375 dB
NAIVE            = 32.8397 dB
ORACLE           = 44.0138 dB
ESTIMATED        = 42.8687 dB
EST_RECOVERY     = +10.0290 dB
EST_TO_ORACLE    = -1.1452 dB
```

The estimated geometry therefore restores most of the non-registration loss, but an estimated-vs-oracle operator gap remains.

## Stage-2D: estimated-phi-aware diffusion adaptation

Stage-2D freezes the learned CDRDI solver and adapts only the diffusion predictor to the operator distribution it will actually see at test time.

For each training sample:

```text
hidden phi_GT -> synthesize Y_H = P0 W_phiGT(X)
Y_H + Y_M     -> frozen CDRDI -> phi_hat
phi_hat       -> build A_(t,phi_hat)
```

Because `phi_hat != phi_GT`, simply training on `A~_(t,phi_hat)(X)` would still omit the terminal observation mismatch present during real inference. Stage-2D therefore uses an observation-anchored teacher-forced trajectory:

```text
x_T^obs = A_(T,phi_hat)^dagger Y_H
r_T     = x_T^obs - A~_(T,phi_hat)(X)

x_t^train = A~_(t,phi_hat)(X) + r_T
          = x_T^obs + A~_(t,phi_hat)(X) - A~_(T,phi_hat)(X)
```

This keeps the same terminal initialization as inference while preserving `X` as the reconstruction target. GT flow is not supplied to CDRDI or to the diffusion predictor; it only synthesizes the hidden acquisition.

Run the trajectory tests first:

```bash
pytest -q tests/test_estimated_geometry_trajectory.py
```

Then fine-tune from the Stage-2B oracle-adapted checkpoint:

```bash
python train_cdrdi_diffusion_estimated.py \
  --dataset PaviaU \
  --device cuda \
  --epochs 30 \
  --lr 5e-6 \
  --geometry_steps 4 \
  --geometry_checkpoint ./checkpoints/cdrdi_stage1/PaviaU_recursive_k4_local4_seed10.pth \
  --init_checkpoint ./checkpoints/cdrdi_stage2/PaviaU_oracle_deform_diffusion_local4_seed10.pth \
  --max_translation 4 \
  --max_rotation_deg 2 \
  --max_local_px 4 \
  --eval_cases 10 \
  --seed 10
```

The best checkpoint is selected by `ESTIMATED` PSNR, not Oracle PSNR. Evaluation always reports `REGISTERED`, `NAIVE`, `ORACLE`, `ESTIMATED`, geometry EPE, and the estimated-to-oracle gap.

Stage-2D is an adaptation experiment, not a change to the CDRDI geometry solver and not a return to inverse image registration.
