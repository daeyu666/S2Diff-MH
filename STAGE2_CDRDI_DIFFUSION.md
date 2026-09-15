# Innovation 2 Stage-2: CDRDI x Physical Degradation Diffusion

Stage-0 established that fixed `P0/R0` turns the original full-blind problem into a stable deformation-only inverse problem. Stage-1 established that shared-weight physical-residual recursion is consistently better than one-shot geometry prediction without flow supervision.

Stage-2 now tests whether the estimated acquisition geometry can be inserted directly into Innovation-1's physical trajectory without inverse-warping the observed LR-HSI.

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

This preserves the Innovation-1 reconstruction target `x_0=X`; the same fixed `phi` is used for every actual degraded state `t>=1`.

## First coupling diagnostic

Before retraining the super-resolution predictor, compare four paths using the same synthetic deformation:

```text
registered : original registered Innovation-1 baseline
naive      : deformed LR-HSI, geometry ignored
oracle     : deformed LR-HSI, GT phi in A_(t,phi)
estimated  : deformed LR-HSI, learned CDRDI phi in A_(t,phi)
```

Run the operator unit tests first:

```bash
pytest -q tests/test_deformation_aware_diffusion.py
```

Then run a single-case Stage-2 diagnostic:

```bash
python validate_cdrdi_diffusion_coupling.py \
  --dataset PaviaU \
  --device cuda \
  --cases 1 \
  --geometry_steps 4 \
  --geometry_checkpoint ./checkpoints/cdrdi_stage1/PaviaU_recursive_k4_local4_seed10.pth \
  --legacy_raw_direct_checkpoint ./checkpoints/legacy/PaviaU_innovation1_physical_v3_raw_direct.pth \
  --max_translation 4 \
  --max_rotation_deg 2 \
  --max_local_px 4 \
  --seed 10
```

After the single case is numerically sane, use `--cases 5` or `--cases 10`.

## How to interpret the result

The diagnostic prints `REGISTERED`, `NAIVE`, `ORACLE`, and `ESTIMATED` reconstruction metrics plus geometry EPE.

- If `ORACLE` is close to `REGISTERED`, the deformation-aware acquisition operator is compatible with the existing Innovation-1 checkpoint. The remaining `ESTIMATED -> ORACLE` gap is mainly geometry-estimation error.
- If `ORACLE` strongly improves over `NAIVE` but remains below `REGISTERED`, the operator coupling works but the registered Raw-Direct predictor has a state-distribution mismatch. The next step is fine-tuning/training the diffusion predictor on `A~_(t,phi)` states with geometry frozen per sample.
- If `ORACLE` does not improve over `NAIVE`, debug the Stage-2 operator/adjoint before changing the geometry network.

The first Stage-2 experiment is intentionally diagnostic. It does not yet claim that a registered Innovation-1 checkpoint is optimal for deformation-aware states.
