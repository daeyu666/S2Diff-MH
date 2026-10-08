# Augsburg-2 original-scale no-reference QNR (matched UAFL implementation)

The underlying formula is migrated **unchanged** from
`daeyu666/comparison_experiments/comparison/UAFL/augsburg2_wald_qnr.py`.
The benchmark is **MSI-projected modified QNR**, **not** classical
single-PAN QNR, QNR*, or 242-band spectral fidelity.

## Input and measurement definitions

- F: 10m fused 242-band HSI (S2Diff-MH output)
- H: observed 30m EnMAP-like HSI; **not EnMAP10 reference**
- M: genuine Sentinel-2 10m B2/B3/B4/B8 (original unregistered geometry)
- R: frozen 4×242 Sentinel-2 SRF from Wald cache
- high-projection A=R(F); low-projection B=R(H)
- M_low: 3×3 area-average of the 10m observed MSI to 30m
- M is adjusted with the existing training-only Wald radiometry gain/bias
  (exact same calibration as model input), **never refitted on full-region test**

Local Q is masked UIQI on nonoverlapping 48×48 high-resolution windows
(16×16 low-resolution windows). Windows with valid fraction under 80%
are excluded and local scores weighted by valid pixel counts. A low
valid pixel requires all 3×3 high pixels valid.

\[
D_{\lambda}=\frac{1}{6}\sum_{i<j}^{4}|Q(A_i,A_j)-Q(B_i,B_j)|
\]

\[
D_s=\frac{1}{16}\sum_{i=1}^{4}\sum_{j=1}^{4}|Q(A_i,M_j)-Q(B_i,(M_{\downarrow 3})_j)|
\]

\[
\mathrm{QNR}=\max(0,1-D_{\lambda})\max(0,1-D_s)
\]

No synthetic 10m HSI reference enters these metrics. Only the four S2
spectral projections are assessed. Residual sensor misregistration,
differences in real MSI radiometry, and choice of local UIQI window
can change scores. High QNR alone does not prove faithful recovery of
the full 242-band HSI.

## Run tests

```bash
git pull
python -m unittest discover -s tests -p "test_augsburg2_wald_qnr.py"
```

## Auto-evaluate after full-resolution **Wald A (Identity)** inference

```bash
python infer_augsburg2_wald.py \
  --wald_root ./data/augsburg2_wald \
  --checkpoint ./checkpoints/augsburg_real/Augsburg2_Wald_D2_A.pth \
  --radiometry_json ./data/calibration/Augsburg2_Wald_radiometry.json \
  --save_root ./outputs/augsburg2_wald
```

Creates:
- `outputs/augsburg2_wald/Augsburg2_Wald_full_HSI.npy`
- `outputs/augsburg2_wald/Augsburg2_Wald_full_QNR.json`
- console: `S2DIFF_MH_WALD_ORIGINAL_MSI_QNR QNR=... Dlambda=... Ds=...`

To skip auto metric calculation: `--skip_qnr`.
Optional `--qnr_window_hr 48 --qnr_min_valid_fraction 0.8`.

**Do not** feed Wald B/C checkpoints to the existing full-resolution
Identity inference script. Their 30m geometry needs scale-correct conversion
to 10m. The script explicitly rejects such misuse.

## Evaluate an already-saved 10m HSI without rerunning diffusion

```bash
python augsburg2_wald_qnr.py \
  --wald_root ./data/augsburg2_wald \
  --fused ./outputs/augsburg2_wald/Augsburg2_Wald_full_HSI.npy \
  --radiometry_json ./data/calibration/Augsburg2_Wald_radiometry.json \
  --window_hr 48 \
  --min_valid_fraction 0.8 \
  --output_json ./outputs/augsburg2_wald/Augsburg2_Wald_full_QNR.json
```

The `full/meta.json` must identify `sub_area_2`, and the fused
prediction must match the **same exact georeferenced 10m grid**.
Wald cache labels and calibration provenance are validated.

## Compare with UAFL original-resolution evaluation

```bash
python compare_augsburg2_wald_qnr.py \
  --uafl_json ../comparison_experiments/comparison/UAFL/outputs/augsburg2_wald/UAFL_Wald_full_QNR.json \
  --s2diff_json ./outputs/augsburg2_wald/Augsburg2_Wald_full_QNR.json
```

The comparison rejects incompatible spectral projection definitions,
window parameters or valid-pixel counts. Both JSONs must be computed
with the same raw HSI/MSI/SRF, radiometry and grid. Even matching
metadata cannot substitute for manually verifying both use the same
underlying Wald cache. Neither method should be selected or tuned using
these original-scale scores.
