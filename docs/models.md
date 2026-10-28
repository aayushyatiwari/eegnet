# Models

Select with `model.name`. Hyperparameters live under `model.<name>`.

| name | reference | notes |
|---|---|---|
| `eegnet` | Lawhern et al. 2018 | ~3k params. `kern_len` = sfreq/2 (80 at 160 Hz, 64 at 128 Hz). |
| `shallow` | Schirrmeister et al. 2017 | temporal conv → spatial conv → square → avg-pool → log. Mimics FBCSP band power. |
| `deep` | Schirrmeister et al. 2017 | 4 conv-pool blocks, ~280k params. Needs ≥ ~400 samples at kern 10 / pool 3. |
| `csp_lda` | classical | 8–30 Hz bandpass → CSP (log-variance) → LDA. Same splits as the nets. |

## EEGNet shapes (160 Hz, 64 ch, 640 samples)
```
input                    (B, 1, 64, 640)
temporal conv (1×80)     (B, 8, 64, 640)
depthwise (64×1), D=2    (B, 16, 1, 640)
avgpool (1×4)            (B, 16, 1, 160)
separable (1×16)         (B, 16, 1, 160)
avgpool (1×8)            (B, 16, 1, 20)
flatten → linear         (B, 320) → (B, 2)
```

## Not implemented (vs papers)
- EEGNet max-norm constraints on depthwise (1.0) and dense (0.25) weights.
- Schirrmeister cropped training.

## Scratch vs fine-tune
Train from scratch, cross-subject. Fine-tuning makes sense as subject adaptation: take a fold model and fine-tune on a few trials of a held-out subject. If using pretrained EEG foundation models, check their pretraining sets for EEGMMIDB (leakage).
