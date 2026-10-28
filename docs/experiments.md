# Experiments

## Running
```bash
python preprocess.py --dry-run          # one subject, prints shapes, writes nothing
python preprocess.py                    # all subjects → preprocess.out_dir

python train.py --dry-run               # 10 subjects, 2 epochs, wandb disabled, nothing saved
python train.py                         # config.yaml as is
python train.py --set model.name=shallow data.fold=-1 name=shallow-lr
python train.py --config configs/other.yaml
```
`--set` takes dotted keys and YAML values; unknown keys raise. `data.fold=-1` runs all folds, one wandb run each, grouped by `name`.

Outputs: `runs/<name>/fold<k>/{config.json, results.json, model.pt}`. wandb: project `eeg-mmi`, group `<name>`.

Default task: train on real + imagined L/R fist (runs 3, 4, 7, 8, 11, 12), validate/test on imagined L/R (4, 8, 12). Chance = 50%.

## Planned
1. CSP+LDA, 5 folds
2. EEGNet, ShallowConvNet, DeepConvNet, 5 folds
3. Ablations: imagery-only training; `channels: motor`; `euclidean_alignment: true`; window 0.5–4 s
4. Subject-adaptive fine-tuning

## Log
| date | name | change | test acc (mean ± std) | notes |
|---|---|---|---|---|
| 2026-10-28 | eegnet-lr | default config, 5 folds | 0.813 ± 0.021 | folds 0.779/0.800/0.835/0.824/0.825; per-subject median 0.82, 4 subj ≤60%, 26 ≥90%; early stop at 22–54 epochs |

## Speed (RTX 4050 6 GB, one fold ≈ 6840 train trials, s/epoch, peak GPU memory)
| model | baseline | amp | data_on_gpu | both |
|---|---|---|---|---|
| 2026-10-28 | eegnet-lr | default config, 5 folds | 0.813 ± 0.021 | folds 0.779/0.800/0.835/0.824/0.825; per-subject median 0.82, 4 subj ≤60%, 26 ≥90%; early stop at 22–54 epochs |
| eegnet | 3.5 s, 0.3 GB | 4.4 s | 3.4 s, 1.6 GB | 4.3 s |
| shallow | 16.1 s, 4.1 GB | 9.2 s, 0.7 GB | 16.7 s, 5.4 GB (near OOM) | 9.1 s, 2.0 GB |
| deep | 8.1 s, 2.6 GB | 3.7 s, 0.5 GB | 7.9 s, 4.0 GB | 3.6 s, 1.8 GB |

Use `train.amp=true` for shallow/deep. `data_on_gpu` gives no gain: EEGNet is bound by its first conv (8 × 64 × 640 activations), not by host→GPU copies.
