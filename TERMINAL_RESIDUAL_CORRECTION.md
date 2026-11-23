# Experimental last-step terminal residual correction

For the formal 10-seed x 10-case bank, paired statistics, candidate HPGC training
and native Augsburg mode, see [CORRECTION_CANDIDATE_VALIDATION.md](CORRECTION_CANDIDATE_VALIDATION.md).

The default algorithm and existing checkpoints remain unchanged. This script evaluates
the proposed **full-strength last-step correction** as a separate experiment:

```
delta_hat = xT_observed - L_T(last_prediction)
corrected_x0 = state_x0 - delta_hat
```

The correction function receives no GT, no true geometry, and no scoring mask.
There is no damping coefficient to tune, clipping, extra learned parameter or
HPGC application to the corrected output. The original network predictions and
all intermediate reverse states are reused exactly from the preceding run.

## Replay the existing ten cases without new model inference

From the S2Diff-MH repository root:

```bash
python diagnose_terminal_residual_correction.py \
  --source_dir outputs/diffusion_endpoint/PaviaU_decomposition_10cases \
  --output_dir outputs/diffusion_endpoint/PaviaU_terminal_correction_10cases \
  --device cuda:0 \
  --require_decomposition
```

Only saved NPZ archives, `manifest.json` and `traces.json` are needed. There is
**no requirement to have the original dataset or model weight files mounted**.
The actual saved observations, last predictions, terminal states, estimated
geometry, SRF and scoring masks are reused. Physical process settings come
from the source manifest. All source cases are processed; there is no new seed
or deformation sampler. Checkpoint paths/hashes remain provenance only.
`--device cpu` is supported; using the original device type reduces backend
roundoff differences in replayed operators.

The script checks source completion, case coverage, input/SRF hashes and
reproduced terminal lifting. It reads source files without modifying them.
The output must be a separate directory, outside the source and its ancestors.
Nonempty outputs require `--overwrite_output`; current reports are regenerated.
When overwritten runs have fewer cases, use only archives referenced by the
current `provenance.json`, since older unreferenced NPZ files can remain.

If the preceding run was **not** made with `--error_decomposition`, the saved
files still suffice to compare the correction's metrics. Point `--source_dir`
at that completed endpoint run and omit `--require_decomposition`. Reports
will explicitly mark `decomposition_complete=false` and no independent
decomposition result will be fabricated. A complete endpoint decomposition
experiment requires the independently accumulated `C_pred` from the earlier
per-step observer. Never reconstruct it as `state_x0-GT-delta` just to make a
closure test pass.

## Outputs

| File | Contents |
| --- | --- |
| `summary.csv` | Registered/Warp x original state, last prediction, corrected state; PSNR, SAM, ERGAS, E_H, E_M and other existing metrics |
| `per_sample.csv` | The same scores for each original patch/case |
| `paired_changes.csv` | Each corrected output versus original state and versus last prediction: PSNR gain, SAM/ERGAS reductions and observation-residual changes |
| `corrected_decomposition.csv` / `.json` | Independent error identities for each case on full and ROI support; available only when saved C_pred exists |
| `oracle_controls.json` | Exact-predictor zero-residual and nonzero-residual controls |
| `provenance.json` | Original observation hashes, source archive hashes, terminal-lift replay errors and C availability |
| `manifest.json` | Completion/decomposition flags, copied source configuration and checkpoint provenance, current version and diagnostic definition |
| `{registered,warp}/sampleNNNN_caseNNN.npz` | Original state, last prediction, corrected state, estimated terminal residual, observations/GT, terminal state, SRF, masks, estimated geometry, available true geometry/C |

All image arrays remain raw batch-first B,C,H,W. `estimated_terminal_residual`
is the inference-computable delta_hat, not the GT diagnostic delta. Corrected
images are not clipped before saving. C, if present, retains float64 precision.

Primary PSNR/SAM use [0,1] clipping on the same HR ROI as the source; ERGAS is
raw. `full_*` columns use the existing full-frame repository convention and
should be used to compare against the preceding full-frame results. E_H/E_M
remain unclipped observation RMSE, with the original common LR/HR masks. Means
use the same number of cases per patch and equal patch weights. Independently
trained registered weights are not introduced: the Registered condition still
uses the source adapted-diffusion outputs on identity observations.

## Corrected endpoint and error identities

With fixed linear L_t, L_0=I, and actual prediction errors e_t=prediction_t-GT,
the original observer has independently computed:

```
C_pred = sum_t (L_(t-1)(e_t) - L_t(e_t))
```

Replay checks all three identities, including finite-precision discrepancies:

```
original_x0 - GT = delta_T + C_pred
delta_hat        = delta_T - L_T(e_1)
corrected_x0-GT  = C_pred + L_T(e_1)
```

It also reports the actual and theoretical raw MSE changes relative to the
original state. An operator/trajectory mismatch causes a failed diagnostic,
not a silently accepted decomposition. No clipping or mask is inserted inside
L_T; support restriction is applied only to the resulting diagnostic vectors.

When **every** prediction is exactly GT, C_pred=0 and delta_hat=delta_T, so
corrected_x0 returns GT even with nonzero terminal mismatch. The two oracle
controls verify this to numerical tolerance (2e-5). Exactness of only the last
prediction does not remove errors accumulated from earlier predictions.

This is not generally identical to returning the last prediction. It does
reduce to that prediction in the one-step limit; the unit tests verify that
special case. Satisfying the oracle clean endpoint does not guarantee better
PSNR/SAM or lower E_H/E_M with real weights. Forward observation residuals remain
diagnostics rather than hard consistency guarantees.

The available endpoint/decomposition summaries should be examined before
claiming a compensation mechanism or adopting this correction. No near-
orthogonality assumption is needed for the implementation. HPGC would need
separate training/adaptation and full-pipeline validation if corrected_x0 were
eventually adopted as its input. Existing paper metrics cannot automatically
be attributed to the corrected algorithm.

## Controls and tests

No source archive is needed for oracle-only checks:

```bash
python diagnose_terminal_residual_correction.py --oracle_only \
  --output_dir outputs/diffusion_endpoint/oracle_terminal_correction
```

```bash
python -m pytest -q tests/test_terminal_residual_correction.py \
  tests/test_diffusion_error_decomposition.py tests/test_diffusion_endpoint_diagnostic.py \
  tests/test_estimated_geometry_trajectory.py tests/test_deformation_aware_diffusion.py
```

Coverage includes clean oracle endpoints with nonzero mismatch, the one-step
limit, replayed corrected identities, exact reuse of observations/outputs and
masks, source immutability, explicit missing-C behavior, input corruption
rejection and a complete replay without access to any learned checkpoint file.
