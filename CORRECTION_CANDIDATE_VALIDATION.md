# CORRECTION candidate: paired validation and refiner adaptation

This implements an opt-in candidate, not an automatic replacement of existing
paper results. The default output remains `state`. The candidate is selected
with `--diffusion_output terminal_correction` in supported training/testing
entrypoints. All paths call the same inference-only correction implementation
in `innovation1.py`; the offline replay calls it too. No diffusion or geometry
weights are retrained by the refiner command.

## 1. Formal PaviaU 100-case diffusion comparison

The main-table protocol is **10 geometry seeds x 10 cases per seed per patch**,
not one legacy GPU sampler called 100 times. Use the original shared CPU
sampler, seeds 10001 through 10010, reconstruction eval seed 1234, and full-frame
benchmark metrics. The earlier ten-case legacy diagnostic is a separate bank;
do not concatenate its results with the formal 100-case bank.

```bash
python diagnose_diffusion_endpoint.py \
  --protocol main_table \
  --device cuda:0 \
  --cases 10 --max_samples 0 \
  --error_decomposition \
  --output_dir outputs/diffusion_endpoint/PaviaU_main_table_100

python diagnose_terminal_residual_correction.py \
  --source_dir outputs/diffusion_endpoint/PaviaU_main_table_100 \
  --output_dir outputs/diffusion_endpoint/PaviaU_main_table_100_correction \
  --device cuda:0 --require_decomposition
```

The first command defaults to the same two PaviaU geometry/adapted diffusion
checkpoints used by the earlier endpoint diagnostic. Override
`--geometry_checkpoint`/`--diffusion_checkpoint` with the actual paths when
needed, and keep architecture settings consistent with those checkpoints.
Registered is evaluated once with the same adapted diffusion weights and
identity acquisition. Geometry GT only synthesizes observations and defines
diagnostic support; it is not passed to either learned model.

Each observation has one reverse pass. Original state and last prediction are
retained; correction is then evaluated offline from that same archive. The
source records `geometry_seed`, `case_within_seed`, global case number,
reconstruction RNG seed, and observation hash. These agree with
`multiseed_deformation_protocol.py`. Archives include raw reconstructions and
independent C_pred; all 100 archives will use more disk space than the ten-case
run. Do not enable `--save_step_errors` unless every intermediate error cube is
needed.

Existing endpoint/corrected reports remain available. The correction replay
now also produces:

| Report | Definition |
| --- | --- |
| `validation_statistics.csv` | Mean and sample std (ddof=1) across seed means; separately labelled std across observations |
| `validation_per_seed.csv` | Cases averaged within patch, then equally across patches for each seed/output |
| `improvement_rates.csv` | PSNR/SAM/ERGAS/E_H/E_M improvement counts and fractions, mean and median paired gains, against original state and against last prediction |
| `gain_correlations.csv` | Pearson and Spearman association of raw full-image estimated-terminal-residual L2 with PSNR gain, SAM/ERGAS reductions and observation-residual reductions |

For the paper's std, use `std_between_seed_means_ddof1`, not the case-level
std. Registered has no geometry-seed std. Undefined correlations are blank,
not zero. Correlations pool paired observations descriptively; they do not
prove causality and are not labelled statistically significant. Gains are
signed so positive always means improvement, including SAM/ERGAS reductions.
E_H/E_M are evaluated on the source's common masks; full PSNR/SAM/ERGAS follow
the shared main-table metric implementation. The corrected/base comparisons
use identical observations and support.

## 2. Train a new PaviaU HPGC Refiner on corrected inputs

Reuse the original refiner experiment's training budget, seed, architecture,
loss weights, monitoring criterion, and initialization policy. Change the
diffusion-output mode and create a new run. Example using repository defaults:

```bash
python train_innovation3_gigi_cdrdi.py \
  --stage train --dataset PaviaU --device cuda:0 \
  --geometry_checkpoint ./checkpoints/PaviaU/PaviaU_recursive_k6_finalonly_500ep_lr1e4.pth \
  --diffusion_checkpoint ./checkpoints/PaviaU/PaviaU_estimated_deform_diffusion_k9_stage2d_A.pth \
  --diffusion_output terminal_correction \
  --save_name PaviaU_correction_gigi_full \
  --epochs 100 --eval_cases 10
```

By default the resolver uses the original registered-refiner initialization
at `checkpoints/innovation3_gigi/PaviaU_innovation3_gigi_full.pth`. If the original
experiment initialized from another registered refiner, explicitly supply the
same `--init_refiner_checkpoint`. If the original was random initialization,
use `--from_scratch`; it disables automatic registered initialization and
conflicts with resume/explicit initialization. These are initialization choices,
not resumed old mixed-model training. Do not change them solely to favor one
output version. The example's 100 epochs is a runnable repository default,
not a claim about the budget of your earlier paper run; use its actual budget.

Every training and validation reconstruction uses corrected inputs in candidate
mode. R_phy is recalculated from those corrected inputs. Geometry and diffusion
remain frozen. Registered and Warp use the same output mode. Checkpoints record
`extra.diffusion_output`; legacy checkpoints without that field mean `state`.
Resume/test reject incompatible modes before loading refiner weights. Explicit
initialization may load compatible weights from either mode, because subsequent
training adapts them; this does not make the old checkpoint a candidate result.

Candidate checkpoint/log names automatically receive `_terminal_correction`,
even for explicit save names, so the example's best file is:

```
checkpoints/innovation3_gigi_cdrdi/PaviaU_correction_gigi_full_terminal_correction.pth
```

Use a new run name for each independent run. Existing files remain protected by
the original overwrite/resume policy. No server training has been performed by
adding this code; local verification uses small initialized models and one
optimizer epoch only.

After training, compare BOTH complete models with the same main-table bank:

```bash
python test_multiseed_deformation.py \
  --dataset PaviaU --device cuda:0 \
  --geometry_checkpoint ./checkpoints/PaviaU/PaviaU_recursive_k6_finalonly_500ep_lr1e4.pth \
  --diffusion_checkpoint ./checkpoints/PaviaU/PaviaU_estimated_deform_diffusion_k9_stage2d_A.pth \
  --refiner_checkpoint ./checkpoints/innovation3_gigi_cdrdi/PaviaU_correction_gigi_full_terminal_correction.pth \
  --diffusion_output terminal_correction \
  --output_dir outputs/multiseed_deformation/PaviaU/correction_full
```

Run the original full model separately with `--diffusion_output state`, its
original refiner checkpoint and a different output directory. Both full-model
Registered and Warp results must be regenerated under their named version.
Diffusion-only gains do not establish full-model gains. Never replace table
values using the new base output while retaining old full-model scores.

## 3. Native Augsburg center holdout

`infer_augsburg2_wald.py` now accepts the same output mode. To compare diffusion
base outputs without HR-HSI GT, run the existing command twice, with identical
native checkpoint, calibration, PSF, width, spectral hidden and QNR settings:

```bash
python infer_augsburg2_wald.py --center_holdout --device cuda:0 \
  --base_channels 96 --tile_size 144 --tile_stride 144 \
  --diffusion_output state --save_root outputs/augsburg_correction_check

python infer_augsburg2_wald.py --center_holdout --device cuda:0 \
  --base_channels 96 --tile_size 144 --tile_stride 144 \
  --diffusion_output terminal_correction --save_root outputs/augsburg_correction_check
```

The width 96 is an example matching the current native GIGI entrypoint default;
set it and all other architecture options to your actual diffusion checkpoint.
This center-heldout 144x144 ROI is inferred in one tile with the shown settings.
The correction run saves in a `terminal_correction` subfolder, including mode
in its protocol JSON, so it does not overwrite the original result. QNR,
Dlambda, Ds still use the existing evaluator, SRF and 48x48 scoring window.
Existing Wald identity/provenance restrictions remain in force; this does not
introduce an unsupported geometry route at native scale.

`train_augsburg2_wald_gigi.py` and `infer_augsburg2_wald_gigi.py` also accept
`--diffusion_output terminal_correction` for the later FULL-model experiment.
Wald-E training recomputes the residual from corrected inputs and tags output
names/checkpoint metadata. Resume/test/native full inference reject the wrong
refiner mode. Reuse the original native training command and append the new
mode; retain its actual budget and calibration arguments. A PaviaU-trained
candidate refiner is not a substitute for a compatible Wald-E checkpoint.

No real-data quality gain is assumed: check QNR, Dlambda AND Ds, and retain both
versions' results. Higher E_H in the synthetic diagnostic neither proves nor
rules out deterioration in native spatial quality.

## Verification scope

Tests exercise exact online/offline agreement and unchanged default output,
the shared main-table geometry/observation hashes, seed-level std, gain signs,
the real training/validation path for one small refiner epoch, frozen-backbone
weights, checkpoint-mode guards and native entrypoint mode parsing. Local tests
do not constitute the 100-case experiment, full training or native QNR results.

Expanded local checks also exposed two pre-existing failures, reproduced on
the original checkout: `test_wald_only_modes_and_scratch_provenance` needs a
missing local radiometry JSON; `test_train_only_radiometry_is_used` asserts an
ordering of Ds values near 1e-8 that fails in this NumPy environment. Neither
the calibration/QNR implementation nor those tests was changed for this
candidate. Native real-data execution therefore remains a server-side check.
