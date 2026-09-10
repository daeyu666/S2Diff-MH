# Migration boundary from S2Diff

`S2Diff-MH` was rebuilt as a clean baseline instead of copying the full old repository.

## Retained concepts/code

- sensor-degradation-consistent progressive diffusion;
- physical MTF/PSF degradation;
- detector area integration and normalized-adjoint lift;
- deterministic reverse update across the 1/2/4 scale trajectory;
- Gaussian+Bicubic and Bicubic degradation baselines;
- V1/V2 clean-HSI predictors;
- registered full Raw-MSI Direct fusion backbone;
- fixed IKONOS/WV2 SRF simulation;
- L1 + stable SAM training objective and standard HSI metrics.

## Not migrated

- `degradations/misalignment.py`;
- all V4 alignment predictors;
- V4 rigid/local/confidence/gate-presence/subpixel repair scripts;
- V4 alignment diagnostics and tests;
- misalignment augmentation training scripts;
- legacy MSI high-frequency/gated/time-varying guidance implementations.

The old `daeyu666/S2Diff` repository remains unchanged and serves as an archive for those experiments.

## Legacy Raw-Direct weights

The useful registered Raw-Direct checkpoint remains compatible through `models/legacy_checkpoint.py`. Obsolete V3 gate parameters are discarded when loading; the HSI backbone, Raw-MSI encoder, decoder and output weights retain compatible parameter names.
