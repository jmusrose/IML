# AV_v4 AVE Full-Stage Probe Loss Design

## Goal

Change only the AVE training path in `AV_v4` so audiovisual training uses the following objective for the complete training run:

```text
L = L_fusion + 5 * (L_audio + L_visual)
```

Audio and visual probe features must remain attached to their encoders throughout training. CREMA-D and KineticSound behavior must remain unchanged.

## Design

Parameterize the shared functions in `AV_v4.training` with two optional arguments:

- `probe_loss_weight`, defaulting to `1.0`.
- `detach_probe_features`, defaulting to the current warmup-dependent behavior.

`forward_and_losses` will use the resolved detach behavior when calling `AVBaseline.forward_with_modal_logits` and calculate:

```text
total_loss = fusion_loss + probe_loss_weight * (audio_loss + visual_loss)
```

`train_one_epoch` and `evaluate` will forward these options to `forward_and_losses`.

`AV_v4.train_ave` will explicitly pass:

```text
probe_loss_weight = 5.0
detach_probe_features = False
```

for AV training and evaluation. Audio-only and visual-only execution remains unchanged because probe losses are only used for audiovisual models.

## Compatibility

Default argument values preserve existing `AV_v4` behavior for CREMA-D and KineticSound. No dataset, model architecture, frame sampling, FGM coefficient calculation, optimizer, or scheduler behavior changes.

The existing `fgm_warmup_steps` option remains accepted for compatibility, but it will no longer control probe detachment in the AVE path because AVE explicitly disables detachment.

## Tests

Add focused AV4 tests that verify:

1. The AVE objective equals `fusion_loss + 5 * (audio_loss + visual_loss)`.
2. Audio and visual encoder parameters receive gradients from probe losses even after the FGM warmup counter has elapsed.
3. Default shared-training behavior retains the existing unit probe-loss weight.
4. The AVE entry point passes the AVE-specific loss and detachment settings to training and evaluation.

Run the focused AV4 tests followed by the relevant AV3/AV4 training tests to detect regressions.
