from pathlib import Path

import numpy as np

LABELS = ("left", "right", "fists", "feet")
LR_RUNS = {3, 4, 7, 8, 11, 12}
MOTOR_CHANNELS = ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz",
                  "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"]


def run_labels(run):
    """T1/T2 label names for a run."""
    return ("left", "right") if run in LR_RUNS else ("fists", "feet")


def load_dataset(cfg, max_subjects=None):
    """Returns X (n, ch, t) float32, y (n,), subject (n,), run (n,), meta dict."""
    files = sorted(Path(cfg["processed_dir"]).glob("S*.npz"))[:max_subjects]
    if not files:
        raise FileNotFoundError(f"no processed files in {cfg['processed_dir']}, run preprocess.py")
    runs = sorted(set(cfg["train_runs"]) | set(cfg["test_runs"]))
    class_ids = [LABELS.index(c) for c in cfg["classes"]]
    lut = np.full(len(LABELS), -1)
    lut[class_ids] = np.arange(len(class_ids))
    Xs, ys, subs, rs = [], [], [], []
    for f in files:
        d = np.load(f)
        sfreq, tmin, ch_names = float(d["sfreq"]), float(d["tmin"]), list(d["ch_names"])
        keep = np.isin(d["run"], runs) & np.isin(d["label"], class_ids)
        t0 = round((cfg["tmin"] - tmin) * sfreq)
        t1 = t0 + round((cfg["tmax"] - cfg["tmin"]) * sfreq)
        ch = _channel_idx(cfg["channels"], ch_names)
        Xs.append(d["X"][keep][:, ch, t0:t1])
        ys.append(lut[d["label"][keep]])
        subs.append(np.full(keep.sum(), int(d["subject"])))
        rs.append(d["run"][keep])
    meta = dict(sfreq=sfreq, ch_names=[ch_names[i] for i in ch])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(subs), np.concatenate(rs), meta


def _channel_idx(spec, ch_names):
    if spec == "all":
        return list(range(len(ch_names)))
    names = MOTOR_CHANNELS if spec == "motor" else spec
    return [ch_names.index(c) for c in names]


def subject_split(subjects, n_folds, fold, val_frac, seed):
    """Group k-fold over subjects; validation subjects are carved out of the training subjects."""
    perm = np.random.default_rng(seed).permutation(np.unique(subjects))
    test = np.array_split(perm, n_folds)[fold]
    rest = np.setdiff1d(perm, test)
    rest = np.random.default_rng(seed + fold).permutation(rest)
    n_val = max(1, round(val_frac * len(rest)))
    return rest[n_val:], rest[:n_val], test


def euclidean_align(X, subjects):
    """Whiten each subject's trials by the inverse sqrt of its mean covariance (He & Wu 2019)."""
    X = X.copy()
    for s in np.unique(subjects):
        m = subjects == s
        R = np.einsum("nct,ndt->cd", X[m], X[m]) / (m.sum() * X.shape[-1])
        w, V = np.linalg.eigh(R)
        X[m] = np.einsum("cd,ndt->nct", (V / np.sqrt(w)) @ V.T, X[m])
    return X
