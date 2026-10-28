# EEG motor imagery

Left/right imagined fist classification on PhysioNet EEGMMIDB.

```
config.yaml      experiment config (all of train.py and preprocess.py reads from it)
preprocess.py    EDF → filtered epochs, data/processed/<name>/SXXX.npz
train.py         subject k-fold training, wandb logging
src/             data loading/splits, models, CSP baseline
utils/           config, seeding, wandb
docs/            notes: data.md, pipeline.md, models.md, experiments.md
```

Env: conda `torchgpu` (torch 2.6 + CUDA). Extra deps: `pip install -r requirements.txt`. Usage in `docs/experiments.md`.
