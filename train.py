import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.csp import build_csp_lda
from src.data import euclidean_align, load_dataset, subject_split
from src.models import build_model
from utils.config import load_config
from utils.logger import init_wandb
from utils.misc import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="override, e.g. train.lr=3e-4")
    p.add_argument("--dry-run", action="store_true", help="10 subjects, 2 epochs, no wandb, no saving")
    return p.parse_args()


def make_splits(cfg, X, y, subj, run, fold):
    d = cfg["data"]
    tr_s, va_s, te_s = subject_split(subj, d["n_folds"], fold, d["val_frac"], cfg["seed"])
    test_run = np.isin(run, d["test_runs"])
    masks = {"train": np.isin(subj, tr_s) & np.isin(run, d["train_runs"]),
             "val": np.isin(subj, va_s) & test_run,
             "test": np.isin(subj, te_s) & test_run}
    if d["euclidean_alignment"]:
        X = euclidean_align(X, subj)
    mean = X[masks["train"]].mean(axis=(0, 2), keepdims=True)
    std = X[masks["train"]].std(axis=(0, 2), keepdims=True)
    return {k: ((X[m] - mean) / std, y[m], subj[m]) for k, m in masks.items()}


def batches(X, y, batch_size, shuffle, device):
    idx = torch.randperm(len(y), device=X.device) if shuffle else torch.arange(len(y), device=X.device)
    for i in range(0, len(y), batch_size):
        j = idx[i:i + batch_size]
        yield X[j].to(device, non_blocking=True), y[j].to(device, non_blocking=True)


def evaluate(model, data, batch_size, device, amp):
    model.eval()
    loss, preds = 0.0, []
    with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=amp):
        for xb, yb in batches(*data, batch_size, False, device):
            out = model(xb).float()
            loss += nn.functional.cross_entropy(out, yb, reduction="sum").item()
            preds.append(out.argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    return loss / len(preds), preds


def per_subject_acc(preds, y, subj):
    return {int(s): float((preds[subj == s] == y[subj == s]).mean()) for s in np.unique(subj)}


def run_deep(cfg, splits, device, run):
    t = cfg["train"]
    amp, bs = t["amp"], t["batch_size"]
    store = device if t["data_on_gpu"] else torch.device("cpu")
    tensors = {k: (torch.from_numpy(X).unsqueeze(1).to(store), torch.from_numpy(y).long().to(store))
               for k, (X, y, _) in splits.items()}
    _, n_ch, n_times = splits["train"][0].shape
    model = build_model(cfg["model"], n_ch, n_times, len(cfg["data"]["classes"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    run.summary["n_params"] = sum(p.numel() for p in model.parameters())

    best_acc, best_state, bad = -1.0, None, 0
    for epoch in range(t["epochs"]):
        model.train()
        tr_loss = 0.0
        for xb, yb in batches(*tensors["train"], bs, True, device):
            with torch.autocast(device.type, torch.bfloat16, enabled=amp):
                loss = nn.functional.cross_entropy(model(xb).float(), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(yb)
        va_loss, va_pred = evaluate(model, tensors["val"], bs, device, amp)
        va_acc = float((va_pred == splits["val"][1]).mean())
        run.log({"epoch": epoch, "train/loss": tr_loss / len(splits["train"][1]),
                 "val/loss": va_loss, "val/acc": va_acc})
        print(f"epoch {epoch:3d} train_loss {tr_loss / len(splits['train'][1]):.4f} "
              f"val_loss {va_loss:.4f} val_acc {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc, best_state, bad = va_acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= t["patience"]:
                break

    model.load_state_dict(best_state)
    _, te_pred = evaluate(model, tensors["test"], bs, device, amp)
    return model, best_acc, te_pred


def run_csp(cfg, splits, sfreq):
    clf = build_csp_lda(cfg["model"]["csp_lda"], sfreq)
    clf.fit(splits["train"][0], splits["train"][1])
    va_acc = float((clf.predict(splits["val"][0]) == splits["val"][1]).mean())
    return clf, va_acc, clf.predict(splits["test"][0])


def run_fold(cfg, data, fold, dry_run):
    X, y, subj, run_ids, meta = data
    set_seed(cfg["seed"] + fold)
    splits = make_splits(cfg, X, y, subj, run_ids, fold)
    print(f"fold {fold}: " + ", ".join(f"{k} {v[0].shape}" for k, v in splits.items()))
    run = init_wandb(cfg, fold, dry_run)

    if cfg["model"]["name"] == "csp_lda":
        model, va_acc, te_pred = run_csp(cfg, splits, meta["sfreq"])
    else:
        model, va_acc, te_pred = run_deep(cfg, splits, get_device(cfg["train"]["device"]), run)

    _, te_y, te_subj = splits["test"]
    te_acc = float((te_pred == te_y).mean())
    subj_acc = per_subject_acc(te_pred, te_y, te_subj)
    run.summary.update({"best_val/acc": va_acc, "test/acc": te_acc,
                        "test/per_subject": {str(s): a for s, a in subj_acc.items()}})
    print(f"fold {fold}: val_acc {va_acc:.3f} test_acc {te_acc:.3f}")

    if not dry_run:
        out = Path(cfg["train"]["out_dir"]) / cfg["name"] / f"fold{fold}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        (out / "results.json").write_text(json.dumps(
            {"val_acc": va_acc, "test_acc": te_acc, "per_subject": subj_acc}, indent=2))
        if isinstance(model, nn.Module):
            torch.save(model.state_dict(), out / "model.pt")
    run.finish()
    return te_acc


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    if args.dry_run:
        cfg["train"]["epochs"] = 2
    data = load_dataset(cfg["data"], max_subjects=10 if args.dry_run else None)
    folds = range(cfg["data"]["n_folds"]) if cfg["data"]["fold"] == -1 else [cfg["data"]["fold"]]
    accs = [run_fold(cfg, data, f, args.dry_run) for f in folds]
    print(f"test acc over folds {list(folds)}: {np.mean(accs):.3f} ± {np.std(accs):.3f}")


if __name__ == "__main__":
    main()
