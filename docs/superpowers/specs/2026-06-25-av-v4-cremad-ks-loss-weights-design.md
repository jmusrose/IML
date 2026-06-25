# AV_v4 CREMA-D and KineticSound Loss Weights Design

## Goal

Allow AV_v4 CREMA-D and KineticSound audiovisual training to configure audio and visual auxiliary-loss weights independently:

```text
L = L_fusion + w_audio * L_audio + w_visual * L_visual
```

AVE behavior remains unchanged.

## Command-Line Interface

Both CREMA-D and KineticSound accept:

```text
--audio-loss-weight
--visual-loss-weight
```

Each defaults to `1.0`, preserving the current loss.

## Detachment Behavior

CREMA-D and KineticSound retain their current FGM warmup behavior:

- during warmup, probe features remain attached and auxiliary losses train the encoders;
- after warmup, probe features detach and auxiliary losses train only the probe heads.

The loss weights still scale the reported and optimized auxiliary losses after detachment, but no longer affect encoder gradients through those detached probe paths.

AVE continues to explicitly use `detach_probe_features=False` for its full run.

## Implementation

KineticSound already uses `AV_v4.training`, whose shared functions accept independent audio and visual weights. Its entry point will expose CLI parameters and forward them to training, validation, and testing.

CREMA-D has a separate training implementation in `AV_v4.train_cremad`. Its local `forward_and_losses`, `train_one_epoch`, and `evaluate` functions will receive the same independent weight parameters and retain their existing warmup-based detachment logic.

## Tests

Tests verify:

1. CREMA-D calculates the independently weighted total loss.
2. CREMA-D and KineticSound parse default and explicit CLI weights.
3. Both entry points forward the CLI values to training, validation, and testing.
4. CREMA-D and KineticSound retain warmup-dependent detachment.
5. Existing AVE full-stage no-detach behavior remains unchanged.
