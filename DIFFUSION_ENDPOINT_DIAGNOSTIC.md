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

## Independent per-step prediction-error decomposition

Keep `state_x0` as the formal output. To instrument the SAME inference trajectory
without changing Eq. (10), training, weights or refiner inputs, run:

```bash
python diagnose_diffusion_endpoint.py \
  --device cuda:0 \
  --cases 10 --max_samples 0 \
  --error_decomposition \
  --output_dir outputs/diffusion_endpoint/PaviaU_decomposition_10cases
```

Use the same `--geometry_checkpoint`, `--diffusion_checkpoint`, seed, geometry
depth and other settings as the preceding ten-case endpoint experiment. Default
checkpoint paths are shown above. The deterministic private geometry generator
reproduces those cases when these inputs match; compare the observation hashes
in `traces.json` and checkpoint hashes in `manifest.json` to audit pairing.

The old archives do not contain every intermediate network prediction. A new
inference pass is needed for an **independent** accumulation; no retraining is
needed. Defining C as `state_x0-GT-delta` from the old archives would make the
closure test circular, so this script never uses that shortcut.

At each actual step t, an observer gets a copy of the network prediction after
the production update has completed. It does not receive or modify the reverse
state, return a correction to the model, or consume random numbers. It computes:

```
e_t    = prediction_t - GT
c_t    = L_(t-1)(e_t) - L_t(e_t)
C_pred = sum(t=T,...,1) c_t
```

After the reverse pass it checks:

```
state_x0 - GT = delta_T + C_pred
```

This decomposition requires **fixed linear** L_t and L_0=I. These conditions
hold for the physical operator with a fixed estimated sampling grid and
image-independent normalized-adjoint denominator. The observer uses the full
raw error field as the operator input; no clipping, image normalization, or ROI
mask is inserted inside L_t. A numerical closure failure stops the diagnostic
instead of being reported as evidence for a mechanism. Operator evaluation is
float32; accumulation, norms and energy statistics are float64.

Additional outputs:

| File | Contents |
| --- | --- |
| `error_decomposition.csv` | Each condition/patch/case, both full and ROI scopes: delta, C, delta+C and actual-error L2/RMS norms; cosine; cross and squared-energy terms; closure errors; net-compensation flag |
| `error_decomposition_summary.csv` | Separate condition/scope summaries, case counts with opposing components and with net compensation, defined-cosine count, mean norms/energies and maximum closure error |
| `error_decomposition_steps.csv` | For every actual timestep: e_t L2/RMS, c_t L2/RMS, cumulative C L2/RMS, both scopes |
| `error_decomposition.json` | Per-case decomposition with undefined cosines represented as JSON null |
| Existing per-case `.npz` | Additional independently accumulated `C_pred_diagnostic_only` field, stored float64 |

Append `--save_step_errors` if you also need all raw `prediction_error_t12`,
..., `prediction_error_t01` cubes in the archives. This can consume substantial
disk space; by default only per-step scalar statistics and the final C field
are added. The original endpoint summary and tensors are still exported.
Summary norms are means of per-case norms, not norms of averaged vectors.

### Interpreting compensation

For vectors restricted to one common support,

```
||delta+C||^2 - ||delta||^2 = 2<delta,C> + ||C||^2
```

`cos_delta_C < 0` means the cross term is negative; it does **not** by itself
mean that the final error is smaller. An oversized opposing C can overshoot.
`net_compensation=True` requires
`2*mean(delta*C)+mean(C^2)<0`, beyond numerical tolerance, against the **oracle
offset baseline** `GT+delta`. The actual error-energy change is also reported,
and must also be negative beyond tolerance. The tolerance includes the measured
actual-vs-decomposed MSE discrepancy, so floating-point closure error cannot be
counted as net compensation. Both changes can be compared directly.

Zero delta or zero C makes cosine undefined (blank in CSV, null in JSON), not
zero. Registered cases therefore should not be interpreted as compensation of
a nonzero anchor offset. Counts are reported separately for Registered and
Warp, full and ROI; neither an averaged cosine nor one favorable case proves
a general mechanism. All calculations here use unclipped errors and should
not be identified with the clipped PSNR/SAM scores.

This decomposition quantifies where actual reconstruction error comes from.
Even verified cancellation on these samples is not proof that the model was
trained to cancel delta, does not establish a causal advantage over the last
prediction, and does not resolve the ideal-predictor clean-endpoint issue:
when every e_t=0, C=0 and the original state remains `GT+delta`.

Oracle-only controls can also exercise the decomposition without data/weights:

```bash
python diagnose_diffusion_endpoint.py --oracle_only --error_decomposition \
  --output_dir outputs/diffusion_endpoint/oracle_decomposition
```

Both controls must give C=0, including the genuinely nonzero-delta control.
Additional tests:

```bash
python -m pytest -q tests/test_diffusion_error_decomposition.py \
  tests/test_diffusion_endpoint_diagnostic.py \
  tests/test_estimated_geometry_trajectory.py tests/test_deformation_aware_diffusion.py
```

They check independent per-step accumulation, zero/nonzero oracle controls,
unchanged production state and RNG, negative-cosine overshoot, rejection of an
invalid nonlinear trajectory, paired exports, and the full command entry point.
