# Augsburg-2 full-resolution original HSI–MSI QNR

### Full-resolution QNR / Dλ / Ds: original HSI–MSI observations

This is **HSI–MSI QNR**, with **LR-HSI as spectral reference** and
**observed HR-MSI as spatial reference**. No PAN image (real or synthetic) is
created. It is the standard QNR spectral/spatial distortion principle adapted
to hypersharpening; do not conflate its cross-sensor spatial component with
classical single-PAN pansharpening QNR.

Inputs, all from the same Augsburg-2 Region-2 10m/30m Wald cache:

- \`F\`: fused 10m HSI, **242 bands**.
- \`H\`: originally **observed** 30m HSI, **242 bands**.
- \`M\`: originally **observed** 10m four-channel Sentinel-2 MSI, after the
  **pre-existing train-only** gain/bias calibration; **no geometric warp**.
- \`M_L\`: area-averaged original 10m MSI on the 30m grid (factor 3).
- \`R\`: frozen SRF, **only for selecting HSI bands spectrally covered by
  each MSI channel**, not for projecting HSI into four bands.

With valid-masked local \`Q=UIQI\`:

\`\`\`text
Dlambda = mean over all 242 choose 2 = 29161 HSI-band pairs (i<j)
          |Q(F_i,F_j) - Q(H_i,H_j)|

For MSI band k:
  S_k = {HSI band i: SRF[k,i] >= 0.01 * max(SRF[k,:])}
  d_k = mean over i in S_k
        |Q(F_i,M_k) - Q(H_i,M_L,k)|

Ds = mean(d_1,d_2,d_3,d_4)
QNR = max(0, 1-Dlambda) * max(0, 1-Ds)
\`\`\`

The 1%-of-peak SRF coverage rule is an explicit, fixed setting. It follows
the HSI–MSI hypersharpening principle that spatial distortion is evaluated
against MSI only for HSI wavelengths sensed by that MSI band. All 242 HSI
bands, including SWIR, contribute to **Dλ**. UIQI is averaged over masked,
non-overlapping 48x48 10m windows and 16x16 30m windows, with a minimum
80% valid-pixel fraction and valid-pixel weighting. The lower mask requires
all 3x3 contributing HR pixels valid.

The calculation requires no 10m HSI ground truth. It does **not** establish
full-resolution ground-truth spectral accuracy. Residual cross-sensor
misregistration can affect the spatial term. It must not be compared with
the old, superseded four-band projected QNR or synthetic-PAN QNR outputs.
Recompute both methods' JSONs after updating.

Standalone computation (already existing fused .npy; **no retraining**):

\`\`\`bash
python augsburg2_wald_qnr.py \
  --wald_root ./data/augsburg2_wald \
  --radiometry_json ./data/calibration/Augsburg2_Wald_radiometry.json \
  --fused ./outputs/augsburg2_wald/Augsburg2_Wald_full_HSI.npy \
  --output_json ./outputs/augsburg2_wald/Augsburg2_Wald_full_QNR.json
\`\`\`

Full inference automatically computes QNR unless \`--skip_qnr\` is set.
\`--qnr_support_fraction 0.01\` controls the spectral-coverage rule.
\`--qnr_window_hr 48\` and \`--qnr_min_valid_fraction 0.8\` fix the
same window policy in both repositories.

Our original-scale inference currently accepts **Wald A/Identity** only.
Do not run frozen B/C with an identity-only full physics operator; native
10m motion for B/C needs its proper pixel-unit scale conversion.

The cross-repository numerical comparison script
`compare_augsburg2_wald_qnr.py` verifies matching calculation metadata.

Regression check:
```bash
python -m unittest discover -s tests -p "test_augsburg2_wald_qnr.py"
```
