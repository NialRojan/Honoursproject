"""python -m eval.run [--methods lr bert] [--folds 5] [--out results]

Cross-validated comparison of text baselines under a RANDOM split (what the
literature reports) and a GROUPED split (no page/template shared between train and
test). Out-of-fold predictions are pooled; temperature is fit on a validation slice
carved out of each training fold, never on test rows.
"""
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eval.baselines import METHODS
from eval.metrics import (apply_temperature, auroc, bootstrap_ci, brier, ece,
                          fit_temperature, fpr, prf)
from eval.splits import grouped_folds, make_groups, random_folds
from eval.stats import holm, mcnemar_exact

DATA = Path(__file__).resolve().parent.parent / "training" / "data" / "linguistic_dataset.tsv"
SEED = 42


def load():
    df = pd.read_csv(DATA, sep="\t").dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["group"] = make_groups(df)
    return df


def inner_val_split(y, groups, grouped, seed):
    """Carve ~1/7 of the training fold off for temperature fitting."""
    if grouped:
        tr, va = grouped_folds(y, groups, k=7, seed=seed)[0]
    else:
        tr, va = random_folds(y, k=7, seed=seed)[0]
    return tr, va


def run_split(df, method, grouped, folds, seed=SEED):
    y = df["label"].to_numpy()
    groups = df["group"].to_numpy()
    texts = df["text"].to_numpy()
    fold_fn = (lambda: grouped_folds(y, groups, k=folds, seed=seed)) if grouped \
        else (lambda: random_folds(y, k=folds, seed=seed))
    rows = []
    for f, (tr, te) in enumerate(fold_fn()):
        itr, iva = inner_val_split(y[tr], groups[tr], grouped, seed)
        tr_idx, va_idx = tr[itr], tr[iva]
        t0 = time.time()
        va_logit, te_logit = METHODS[method](texts[tr_idx], y[tr_idx],
                                             [texts[va_idx], texts[te]], seed=seed)
        T = fit_temperature(va_logit, y[va_idx])
        print(f"  {method:4s} {'grouped' if grouped else 'random ':7s} fold {f + 1}/{folds} "
              f"T={T:.2f} ({time.time() - t0:.0f}s)", flush=True)
        for i, lg in zip(te, te_logit):
            rows.append({"idx": int(i), "fold": f, "logit": float(lg), "T": T,
                         "p_raw": float(apply_temperature(lg, 1.0)),
                         "p_cal": float(apply_temperature(lg, T))})
    out = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    return out.assign(y=y[out["idx"].to_numpy()], group=groups[out["idx"].to_numpy()],
                      method=method, split="grouped" if grouped else "random")


def summarise(preds, threshold=0.5, col="p_raw", n_boot=1000):
    y = preds["y"].to_numpy()
    p = preds[col].to_numpy()
    pred = (p >= threshold).astype(int)
    g = preds["group"].to_numpy()
    pr, rc, f1 = prf(y, pred)
    lo, hi = bootstrap_ci(y, pred, lambda a, b: prf(a, b)[2], g, n=n_boot, seed=0)
    return {"acc": float((pred == y).mean()), "precision": pr, "recall": rc, "f1": f1,
            "f1_lo": lo, "f1_hi": hi, "auroc": auroc(y, p), "fpr": fpr(y, pred),
            "ece": ece(y, p), "brier": brier(y, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["lr", "bert"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="results")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()

    df = load()
    print(f"{len(df)} texts, {df.label.sum()} positive, {df.group.nunique()} groups")
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()

    all_preds = []
    for m in a.methods:
        for grouped in (False, True):
            all_preds.append(run_split(df, m, grouped, a.folds))
    preds = pd.concat(all_preds, ignore_index=True)
    preds["commit"] = commit
    preds.to_csv(out / "predictions.csv", index=False)

    rows = []
    for (m, s), d in preds.groupby(["method", "split"]):
        rows.append({"method": m, "split": s, "calibrated": "no", "thr": 0.5,
                     **summarise(d, 0.5, "p_raw", a.boot)})
        rows.append({"method": m, "split": s, "calibrated": "T-scaled", "thr": 0.5,
                     **summarise(d, 0.5, "p_cal", a.boot)})
        if m == "bert":
            rows.append({"method": "bert@0.6 (deployed thr)", "split": s, "calibrated": "no",
                         "thr": 0.6, **summarise(d, 0.6, "p_raw", a.boot)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "summary.csv", index=False)

    pd.set_option("display.width", 200)
    show = table.copy()
    show["f1_ci"] = show.apply(lambda r: f"[{r.f1_lo:.3f},{r.f1_hi:.3f}]", axis=1)
    print("\n" + show[["method", "split", "calibrated", "thr", "acc", "precision", "recall",
                       "f1", "f1_ci", "auroc", "fpr", "ece", "brier"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if {"lr", "bert"} <= set(a.methods):
        print("\nPaired McNemar, LR vs BERT (threshold 0.5, raw):")
        ps = []
        for s in ("random", "grouped"):
            la = preds[(preds.method == "lr") & (preds.split == s)].sort_values("idx")
            be = preds[(preds.method == "bert") & (preds.split == s)].sort_values("idx")
            n01, n10, p = mcnemar_exact(la["y"], (la.p_raw >= .5).astype(int),
                                        (be.p_raw >= .5).astype(int))
            ps.append(p)
            print(f"  {s:8s} LR-only-right={n01} BERT-only-right={n10} p={p:.4f}")
        print("  Holm-adjusted:", [round(x, 4) for x in holm(ps)])
    print(f"\ncommit {commit}; saved {out}/predictions.csv and {out}/summary.csv")


if __name__ == "__main__":
    main()
