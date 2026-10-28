import argparse
from pathlib import Path

import mne
import numpy as np

from src.data import LABELS, run_labels
from utils.config import load_config

mne.set_log_level("ERROR")
MONTAGE = mne.channels.make_standard_montage("colin27_1005")
STD_NAMES = {n.lower(): n for n in MONTAGE.ch_names}


def load_run(path, cfg):
    raw = mne.io.read_raw_edf(path, preload=True)
    raw.rename_channels(lambda n: STD_NAMES[n.strip(".").lower()])
    raw.set_montage(MONTAGE)
    raw.filter(cfg["l_freq"], cfg["h_freq"])
    if cfg["resample"]:
        raw.resample(cfg["resample"])
    return raw


def process_subject(subject, cfg):
    Xs, labels, runs = [], [], []
    for run in cfg["runs"]:
        raw = load_run(Path(cfg["raw_dir"]) / f"S{subject:03d}" / f"S{subject:03d}R{run:02d}.edf", cfg)
        events, _ = mne.events_from_annotations(raw, event_id={"T1": 1, "T2": 2})
        sfreq = raw.info["sfreq"]
        epochs = mne.Epochs(raw, events, tmin=cfg["tmin"], tmax=cfg["tmax"] - 1 / sfreq,
                            baseline=None, preload=True)
        names = run_labels(run)
        Xs.append(epochs.get_data(units="uV").astype(np.float32))
        labels.append([LABELS.index(names[e - 1]) for e in epochs.events[:, 2]])
        runs.append(np.full(len(epochs), run))
    return dict(X=np.concatenate(Xs), label=np.concatenate(labels), run=np.concatenate(runs),
                subject=subject, sfreq=sfreq, tmin=cfg["tmin"], ch_names=np.array(raw.ch_names))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    p.add_argument("--dry-run", action="store_true", help="process one subject, print shapes, write nothing")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)["preprocess"]

    subjects = [s for s in range(1, 110) if s not in cfg["exclude_subjects"]]
    out = Path(cfg["out_dir"])
    if args.dry_run:
        subjects = subjects[:1]
    else:
        out.mkdir(parents=True, exist_ok=True)
    for s in subjects:
        d = process_subject(s, cfg)
        print(f"S{s:03d} X={d['X'].shape} labels={np.bincount(d['label'], minlength=4)}")
        if not args.dry_run:
            np.savez(out / f"S{s:03d}.npz", **d)


if __name__ == "__main__":
    main()
