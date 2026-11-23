# PaviaU diffusion endpoint diagnostic

This is a diagnostic of **existing frozen weights**, not a training change.
`innovation1.py`, the default inference return value, all training trajectories,
and HPGC training remain unchanged. The script retains both outputs from one
reverse loop:

- `state_x0`: exactly the current production state after the last update;
- `last_prediction`: the network's clean-image prediction at t=1;
- optional `state_x0_refined`: existing frozen HPGC applied to `state_x0` only.

The final prediction is retained without a second diffusion pass or additional
network call. The script also saves `x1`, the lifted terminal state, the raw
endpoint difference, and `delta_T_diagnostic_only`. GT never enters the learned
geometry estimator or the diffusion reconstruction. The GT-dependent delta is
computed **after** reconstruction and never subtracted from either output.

## Run a small paired test

From the S2Diff-MH repository root:

```bash
python diagnose_diffusion_endpoint.py \
  --device cuda:0 \
  --geometry_checkpoint ./checkpoints/PaviaU/PaviaU_recursive_k6_finalonly_500ep_lr1e4.pth \
  --diffusion_checkpoint ./checkpoints/PaviaU/PaviaU_estimated_deform_diffusion_k9_stage2d_A.pth \
  --cases 3 \
  --output_dir outputs/diffusion_endpoint/PaviaU
```

These are the same default paths as the existing PaviaU CDRDI iteration
ablation. Replace them with the actual geometry/adapted-diffusion checkpoints
used for your experiment if their names or locations differ. Architecture
arguments must match the weights; loading is strict. Defaults: geometry width
32, geometry inference steps 9; diffusion width 64, spectral hidden 8, time
dimension 256, T=12; physical MTF=0.2, PSF truncate=3, scale=4. Use `--help`
for overrides. No checkpoint is picked by its test score.

Default scope is the first official PaviaU 128x128 test patch, one registered
observation and three random warp observations. `--max_samples 0 --cases 10`
evaluates the complete test split with the existing ten-case setting. Random
geometry matches the legacy evaluation sampler: device generator with
`seed+70000`, reset per patch, signed rigid limits 4 px / 2 degrees, local
bound 4 px, grid 5, min Jacobian 0.5. There is no identity mixture in warp tests.
The registered condition uses the **same adapted diffusion checkpoint** and
bypasses CDRDI, matching the current REGISTERED inference route. It does not
stand in for the independently trained registered diffusion checkpoint.

To include the original HPGC pipeline, append:

```bash
  --refiner_checkpoint /actual/path/to/PaviaU_full_refiner.pth
```

This does not apply the old refiner to `last_prediction`; that input distribution
has not been used to train those HPGC weights. Explicit refiner dimensions and
variant must match the supplied checkpoint. Supply geometry and diffusion
weights matching the original frozen HPGC training configuration.

Use a new `--output_dir` for another configuration. Nonempty output directories
are rejected unless `--overwrite_output` is explicit. Overwrite replaces reports
and matching archives; older unreferenced archives may remain. Only archives
listed in the current `per_sample.csv` belong to the current run.

## Oracle controls (no weights or data needed)

```bash
python diagnose_diffusion_endpoint.py --oracle_only \
  --output_dir outputs/diffusion_endpoint/oracle_controls
```

Both controls run automatically before a learned-model diagnostic as well:

| Control | Network | Required state output | Required last prediction |
| --- | --- | --- | --- |
| Exact acquisition geometry, zero delta | Always returns GT | GT | GT |
| Mismatched acquisition geometry, nonzero delta | Always returns GT | GT + delta | GT |

Numerical tolerances are 2e-5. The nonzero control explicitly rejects a trivial
zero-offset setup. Real-predictor performance is not inferred from these tests.

## Reports and saved tensors

All reports are under `--output_dir`:

- `summary.csv`: **send this file first**; one row per condition/output.
- `per_sample.csv`: individual patch/case/output scores, input hashes, archive paths.
- `traces.json`: terminal delta RMS, endpoint-difference RMS, difference-minus-delta
  RMS, and the maximum error in `state_x0-last_prediction = x1-L1(last_prediction)`.
  Under a real imperfect predictor, the endpoint difference need not equal delta.
- `oracle_controls.json`: the two ideal-predictor controls.
- `manifest.json`: completion flag, actual arguments, checkpoint paths/epochs/SHA256,
  checkpoint metadata, SRF hash, software/source version and scoring conventions.
- `{registered,warp}/sampleNNNN_caseNNN.npz`: both raw reconstructions, optional
  original-state refinement, GT, HR-MSI, observed LR-HSI, terminal/x1 states,
  diagnostic delta, masks, SRF, and true/estimated rigid and local geometry.
  Images retain the batch dimension (`B,C,H,W`); masks are boolean.

No prediction is smoothed, independently normalized, or clipped before saving.
Incomplete runs keep `manifest.complete=false` and per-sample reports collected
so far. `--dry_run` prints arguments without loading torch, data or weights.

## Scores and masks

Primary `PSNR`, `SAM` (degrees), `ERGAS`, `RMSE` and `raw_RMSE` use **the same
reference-frame mask for every output of the same observation**. As in the
repository, PSNR/RMSE and SAM use [0,1] clipping; ERGAS uses raw values with
epsilon 1e-8. The script evaluates primary scalar metrics in float64.
`raw_RMSE` has no clipping. `full_*` columns additionally use the unmodified
`metrics.calc_metrics` on the full patch, making the original test convention
available alongside the masked diagnostic. Do not mix full-frame and masked
scores when comparing to tables.

`E_H` and `E_M` are **unclipped observation RMSE**, normalized over valid
channel-pixel elements:

```
E_H = RMS( P0 W_est(output) - observed_LR_HSI ; Omega_H )
E_M = RMS( R0(output)       - observed_HR_MSI ; Omega_M )
```

`Omega_M` is a conservative interior rectangle with margin equal to the maximum
true/estimated component displacement plus the terminal Gaussian PSF radius
plus one pixel. `Omega_H` contains only LR detector cells whose entire
bilinear/PSF/area-integration support lies in `Omega_M` under **both** true and
estimated geometry. This excludes image-padding effects. Masks depend on
geometry, never on predicted image errors; they are identical across outputs
within a case, but can differ between cases. HR/LR valid pixel counts are
reported, and empty support causes an error rather than fallback to full frame.
GT geometry is used here only to define synthetic diagnostic support, not as a
model input. These masks are intended for synthetic tests, not real imagery.

Summary is the mean of per-case scores within each patch, then equally weighted
patch means (all patches have the same number of cases). It is not PSNR computed
from pooled SSE. `full_*` uses the same averaging. A smaller E_H with imperfect
estimated geometry can reflect compensation for that geometry; it is not proof
of better clean reconstruction or guaranteed physical consistency.

## Validation

```bash
python -m pytest -q tests/test_diffusion_endpoint_diagnostic.py \
  tests/test_estimated_geometry_trajectory.py tests/test_deformation_aware_diffusion.py
```

Tests cover zero/nonzero oracle residuals, exact equality to the original
production state output, one model call per step, mask footprint isolation,
unclipped normalized observation residuals, paired exports and input hashes,
original-state-only refinement, and strict loading/running of small actual
geometry/diffusion architectures with temporary initialized checkpoints.
These local tests do not replace the experiment using your trained server weights.
