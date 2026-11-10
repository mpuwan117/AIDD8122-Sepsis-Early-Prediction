# -*- coding: utf-8 -*-
"""Build leakage-free windowed features from the Challenge 2019 cohort.

Design (patient-level early-warning formulation)
------------------------------------------------
For every ICU stay we use ONLY the first W hours of recorded data to build
features. The label is whether SepsisLabel becomes 1 at any hour AFTER that
window. Stays that are already label-positive inside the window (prevalent
cases) and stays with no post-window observation are excluded, so no future
information can enter the features and the task remains a genuine early
warning task.

This is a simplification of the original hour-by-hour online challenge task;
it is stated explicitly in the report. All numbers written by this script come
from the downloaded .psv files.

Usage:
    python src/preprocess.py --raw ../data/raw --out ../data/processed --window 6
"""
import argparse
import csv
import glob
import os
import time

import numpy as np
import pandas as pd

STATS = ("last", "mean", "max", "min", "std", "slope", "miss")


def slope_of(v, valid):
    n = int(valid.sum())
    if n < 3:
        return 0.0
    x = np.arange(len(v), dtype="float64")[valid]
    y = v[valid].astype("float64")
    xc, yc = x - x.mean(), y - y.mean()
    den = float((xc * xc).sum())
    return float((xc * yc).sum() / den) if den > 0 else 0.0


def featurize(path, window):
    """Return (kind, record). kind is 'ok' or an exclusion reason."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        df = pd.read_csv(fh, sep="|")
    if "SepsisLabel" not in df.columns:
        return "no_label", None
    lab = pd.to_numeric(df["SepsisLabel"], errors="coerce").fillna(0).to_numpy()
    n = len(df)
    if n <= window:
        return "too_short", None
    if lab[:window].max() > 0:
        return "prevalent_in_window", None
    label = 1 if lab[window:].max() > 0 else 0

    win = df.iloc[:window].drop(columns=["SepsisLabel"])
    rec = {}
    for col in win.columns:
        v = pd.to_numeric(win[col], errors="coerce").to_numpy(dtype="float32")
        valid = np.isfinite(v)
        vv = v[valid]
        pref = col + "_"
        rec[pref + "last"] = float(vv[-1]) if len(vv) else np.nan
        rec[pref + "mean"] = float(vv.mean()) if len(vv) else np.nan
        rec[pref + "max"] = float(vv.max()) if len(vv) else np.nan
        rec[pref + "min"] = float(vv.min()) if len(vv) else np.nan
        rec[pref + "std"] = float(vv.std()) if len(vv) > 1 else 0.0
        rec[pref + "slope"] = slope_of(v, valid)
        rec[pref + "miss"] = float(1.0 - valid.mean())
    rec["label"] = label
    rec["n_hours"] = n
    rec["patient_id"] = os.path.splitext(os.path.basename(path))[0]
    return "ok", rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data/raw")
    ap.add_argument("--out", default="../data/processed")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--limit-per-set", type=int, default=0,
                    help="0 = all files; >0 for a smaller development run")
    args = ap.parse_args()

    raw = os.path.abspath(args.raw)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    frames, flow = [], {}
    t0 = time.time()
    for set_name in ("training_setA", "training_setB"):
        files = sorted(glob.glob(os.path.join(raw, set_name, "*.psv")))
        if args.limit_per_set > 0:
            files = files[:args.limit_per_set]
        counts = {k: 0 for k in ("total", "too_short", "prevalent_in_window", "no_label", "ok")}
        rows = []
        for i, f in enumerate(files, 1):
            kind, rec = featurize(f, args.window)
            counts[kind] = counts.get(kind, 0) + 1
            counts["total"] += 1
            if rec is not None:
                rec["cohort"] = set_name
                rows.append(rec)
            if i % 5000 == 0:
                print("[preprocess]", set_name, i, "/", len(files),
                      "elapsed", round(time.time() - t0, 1), "s", flush=True)
        df = pd.DataFrame(rows)
        frames.append(df)
        flow[set_name] = counts
        pos = int(df["label"].sum()) if len(df) else 0
        print("[preprocess]", set_name, counts,
              "| positives", pos, "| prevalence",
              round(pos / len(df), 5) if len(df) else 0, flush=True)

    all_df = pd.concat(frames, ignore_index=True)
    feat_cols = [c for c in all_df.columns
                 if c not in ("label", "n_hours", "patient_id", "cohort")]
    ordered = (["patient_id", "cohort", "n_hours", "label"] + feat_cols)
    all_df = all_df[ordered].astype({c: "float32" for c in feat_cols})

    fpath = os.path.join(out, "features.csv.gz")
    all_df.to_csv(fpath, index=False, compression="gzip")
    with open(os.path.join(out, "feature_names.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(feat_cols))
    with open(os.path.join(out, "cohort_flow.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cohort", "total", "too_short", "prevalent_in_window",
                    "no_label", "included", "positives", "prevalence"])
        for name, c in flow.items():
            sub = all_df[all_df["cohort"] == name]
            pos = int(sub["label"].sum())
            w.writerow([name, c["total"], c["too_short"], c["prevalent_in_window"],
                        c["no_label"], len(sub), pos,
                        round(pos / len(sub), 6) if len(sub) else 0])
    print("[preprocess] wrote", fpath, "| shape", all_df.shape,
          "| features", len(feat_cols), "| elapsed", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
