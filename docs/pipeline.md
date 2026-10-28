# Pipeline

## Preprocessing (`preprocess.py`, config section `preprocess`)
Per subject, per run (3–14):
1. Read EDF, rename channels to standard 10-05 names (colin27_1005 montage), set montage.
2. Bandpass `l_freq`–`h_freq` (default 1–40 Hz, zero-phase FIR) on the continuous run. 40 Hz low-pass also removes 60 Hz mains.
3. Optional resample.
4. Epoch every T1/T2 cue from `tmin` to `tmax` (default −0.5 to 4.0 s, 720 samples at 160 Hz). Epochs running past the end of a recording are dropped.
5. Save `data/processed/<name>/SXXX.npz`: `X (n, 64, t) float32 µV`, `label` (index into `LABELS = left, right, fists, feet`), `run`, `subject`, `sfreq`, `tmin`, `ch_names`.

All runs are stored, so tasks and windows are chosen at train time without reprocessing. A different filter or rate needs a new `out_dir`.

Size: ~106 × 180 × 64 × 720 × 4 B ≈ 3.5 GB.

## Training-time data (`src/data.py`, config section `data`)
1. Keep trials whose run is in `train_runs ∪ test_runs` and whose label is in `classes`.
2. Crop to `data.tmin`–`data.tmax` (must lie inside the stored window); select `channels`.
3. Subject split: group k-fold (`n_folds`, `fold`). Validation subjects (`val_frac`) come out of the training subjects.
   - train: train subjects, `train_runs`
   - val / test: val / test subjects, `test_runs` only
4. Optional Euclidean alignment, per subject. Uses the test subject's unlabeled trials (transductive).
5. Per-channel z-score with training-set statistics.

Model selection: best validation accuracy, early stopping with `patience`.

## Shapes (defaults)
`X: (N, 1, 64, 640)`, `y: (N,)` in {0: left, 1: right}.
