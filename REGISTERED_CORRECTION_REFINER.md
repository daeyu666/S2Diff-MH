# Independently registered HPGC with corrected diffusion output

This is the registered-only experiment in `train_innovation3_gigi.py`, not the
Registered test condition of the mixed CDRDI model. Keep its original frozen
registered diffusion checkpoint. Do not substitute the nonregistered Stage-2D
`k9_60ep` checkpoint merely because the mixed model uses it.

The script supports `--diffusion_output terminal_correction` in training,
validation and testing. It reconstructs the corrected base first, then computes
the physical residual from that corrected base. Only the refiner is optimized.
The same correction implementation is shared with the mixed-model experiment.
Default `state` behavior remains unchanged.

## Retrain with the original registered experiment's configuration

```bash
python train_innovation3_gigi.py \
  --stage train --variant full --dataset PaviaU \
  --data_root ./data/raw --device cuda \
  --config_from_checkpoint ./checkpoints/innovation3_gigi/PaviaU_innovation3_gigi_full.pth \
  --diffusion_output terminal_correction \
  --save_name PaviaU_innovation3_gigi_full_terminal_correction.pth
```

`--config_from_checkpoint` reads recognizable training settings from the old
checkpoint's `extra`: original registered backbone path/type, architecture,
epochs, batch size, learning rate, seed, losses, data protocol and validation
settings. It does **not** load the old refiner weights or optimizer. Training
starts at epoch 1 with random refiner initialization, matching this registered
script's original initialization policy. Explicit CLI options override recorded
defaults. Stage, resume, output mode, output names/roots, overwrite policy and
the config-source path are not inherited. A mixed CDRDI configuration is refused.
Empty metadata is refused; individual absent metadata fields retain the CLI
defaults, so check the printed settings against your old command if the archive
was migrated with incomplete metadata.

If the original registered backbone was moved, override its location with
`--baseline_checkpoint ACTUAL_REGISTERED_BACKBONE.pth`. Keep the recorded
`--baseline_checkpoint_type legacy|standard` consistent with the file format.
Use the same original training budget and losses; do not borrow the mixed
model's 150-epoch budget unless that was also the registered experiment's budget.

Candidate names automatically receive `_terminal_correction` (idempotently),
including log names and the default save name. With default checkpoint root,
the shown best checkpoint is:

```
checkpoints/innovation3_gigi/PaviaU_innovation3_gigi_full_terminal_correction.pth
```

The last checkpoint adds `_last`. The old registered checkpoint and log are
preserved. `extra.diffusion_output` identifies new weights; legacy checkpoints
without it are `state`. Resume/test reject a mismatch before loading weights.
Do not use `--resume` to restart from an old state-mode refiner; it is a new run.

## Test this registered full model

```bash
python train_innovation3_gigi.py \
  --stage test --variant full --dataset PaviaU \
  --data_root ./data/raw --device cuda \
  --config_from_checkpoint ./checkpoints/innovation3_gigi/PaviaU_innovation3_gigi_full.pth \
  --diffusion_output terminal_correction \
  --refiner_checkpoint ./checkpoints/innovation3_gigi/PaviaU_innovation3_gigi_full_terminal_correction.pth
```

`BASE_FULL_METRICS` is the corrected diffusion base; `FULL_METRICS` is the new
registered HPGC output. No synthetic warp is introduced in this entrypoint.
Any Registered component ablations should subsequently use new matching-mode
refiners and the same corrected base; old ablation values are not new results.

Local checks cover configuration precedence, rejection of mixed configuration,
unchanged default reconstruction, a real one-epoch small-model training and
validation/test run, updated residual inputs, frozen backbone weights, optimizer
updates to the refiner, separate filenames, and checkpoint mode checks. The
actual registered training and test results require the server's data/weights.
