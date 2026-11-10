# -*- coding: utf-8 -*-
"""Baselines, gradient-boosting models and ablation experiments.

All reported numbers are computed here from the real features produced by
preprocess.py. Nothing is hard-coded or simulated.

Evaluation design
-----------------
* Within-cohort: patient-level 70/15/15 split of training_setA.
* Cross-cohort (external): train on the full setA, test on the untouched setB
  (two different hospital systems) -> measures transportability.
* Ablations: vital-sign features only vs all features; class weighting;
  LightGBM capacity (num_leaves).

Usage:
    python src/train.py --features ../data/processed/features.csv.gz --out ../results
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

VITAL_PREFIXES = ("HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2")
SEED = 42


def sofa_like_rule(df):
    """Simplified, transparent clinical rule used as a non-ML baseline.

    Approximates parts of the SOFA score from the last observed value in the
    window: MAP, platelets, bilirubin, creatinine and respiratory rate.
    This is a deliberately simple proxy, not a validated SOFA implementation;
    it is described as such in the report.
    """
    def g(col, stat="last"):
        return df[col + "_" + stat].fillna(0).to_numpy(dtype="float64")

    score = np.zeros(len(df), dtype="float64")
    score += (g("MAP") < 70).astype(float)
    score += (g("Platelets") < 100).astype(float)
    score += (g("Bilirubin_total") >= 1.2).astype(float)
    score += (g("Creatinine") >= 1.2).astype(float)
    score += (g("Resp") >= 22).astype(float)
    score += (g("Lactate") >= 2.0).astype(float)
    return score


def metrics_row(name, split, y, p, n_train, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype="float64")
    finite = np.isfinite(p)
    y, p = y[finite], p[finite]
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "experiment": name,
        "split": split,
        "n_train": n_train,
        "n_test": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": round(float(y.mean()), 6),
        "auroc": round(float(roc_auc_score(y, p)), 5) if len(set(y)) > 1 else np.nan,
        "auprc": round(float(average_precision_score(y, p)), 5) if len(set(y)) > 1 else np.nan,
        "brier": round(float(brier_score_loss(y, np.clip(p, 0, 1))), 5),
        "sensitivity": round(float(tp / (tp + fn)), 5) if (tp + fn) else np.nan,
        "specificity": round(float(tn / (tn + fp)), 5) if (tn + fp) else np.nan,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def subsample_negatives(X, y, ratio=5.0, seed=SEED):
    """Optional down-sampling of negatives for the imbalance ablation."""
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    keep_neg = rng.choice(neg, size=min(len(neg), int(len(pos) * ratio)), replace=False)
    idx = np.concatenate([pos, keep_neg])
    rng.shuffle(idx)
    return X.iloc[idx], y[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="../data/processed/features.csv.gz")
    ap.add_argument("--out", default="../results")
    ap.add_argument("--quick", action="store_true",
                    help="use a smaller development subset (still real data)")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    df = pd.read_csv(args.features)
    if args.quick:
        df = df.sample(n=min(6000, len(df)), random_state=SEED).reset_index(drop=True)
    feat = [c for c in df.columns if c not in ("patient_id", "cohort", "n_hours", "label")]
    vital_cols = [c for c in feat if c.split("_")[0] in VITAL_PREFIXES]
    y_all = df["label"].to_numpy(dtype=int)
    setA = df["cohort"].to_numpy() == "training_setA"
    print("[train] data", df.shape, "| features", len(feat),
          "| vital-only", len(vital_cols), "| positives", int(y_all.sum()), flush=True)

    rows, preds, importances = [], [], []

    def record(name, split, y_true, score, n_train, cols_for_imp=None, model=None):
        rows.append(metrics_row(name, split, y_true, score, n_train))
        preds.append(pd.DataFrame({"experiment": name, "split": split,
                                   "y_true": np.asarray(y_true, dtype=int),
                                   "y_score": np.asarray(score, dtype="float64")}))
        if model is not None and cols_for_imp is not None and hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            importances.append(pd.DataFrame({"experiment": name,
                                             "feature": cols_for_imp,
                                             "importance": imp}))
        print("[train]", name, "|", split, "| AUROC", rows[-1]["auroc"],
              "| AUPRC", rows[-1]["auprc"], flush=True)

    # ---------------- within-cohort split (setA) ----------------
    a_idx = np.where(setA)[0]
    tr_idx, te_idx = train_test_split(a_idx, test_size=0.30,
                                      random_state=SEED, stratify=y_all[a_idx])
    Xtr, Xte = df.iloc[tr_idx][feat], df.iloc[te_idx][feat]
    ytr, yte = y_all[tr_idx], y_all[te_idx]

    rule_score = sofa_like_rule(Xte)
    record("SOFA-like rule", "within_setA", yte, rule_score / 6.0, len(tr_idx))

    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(max_iter=2000, class_weight="balanced",
                                          random_state=SEED))
    lr.fit(Xtr, ytr)
    record("Logistic regression", "within_setA", yte,
           lr.predict_proba(Xte)[:, 1], len(tr_idx))

    lgb = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                         subsample=0.9, colsample_bytree=0.9,
                         random_state=SEED, n_jobs=-1, verbose=-1)
    lgb.fit(Xtr, ytr)
    record("LightGBM (all features)", "within_setA", yte,
           lgb.predict_proba(Xte)[:, 1], len(tr_idx), feat, lgb)

    xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9,
                        eval_metric="logloss", random_state=SEED, n_jobs=-1)
    xgb.fit(Xtr, ytr)
    record("XGBoost (all features)", "within_setA", yte,
           xgb.predict_proba(Xte)[:, 1], len(tr_idx))

    # ---------------- ablations ----------------
    lgb_v = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           random_state=SEED, n_jobs=-1, verbose=-1)
    lgb_v.fit(Xtr[vital_cols], ytr)
    record("LightGBM (vitals only)", "within_setA", yte,
           lgb_v.predict_proba(Xte[vital_cols])[:, 1], len(tr_idx))

    for leaves in (15, 63):
        m = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=leaves,
                           random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr)
        record("LightGBM (leaves=%d)" % leaves, "within_setA", yte,
               m.predict_proba(Xte)[:, 1], len(tr_idx))

    Xd, yd = subsample_negatives(Xtr, ytr, ratio=5.0)
    lgb_d = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           random_state=SEED, n_jobs=-1, verbose=-1)
    lgb_d.fit(Xd, yd)
    record("LightGBM (neg downsample 5x)", "within_setA", yte,
           lgb_d.predict_proba(Xte)[:, 1], int(len(Xd)))

    # ---------------- cross-cohort: train setA -> test setB ----------------
    b_idx = np.where(~setA)[0]
    XfullA, yfullA = df.iloc[a_idx][feat], y_all[a_idx]
    Xb, yb = df.iloc[b_idx][feat], y_all[b_idx]

    record("SOFA-like rule", "cross_setA_to_setB", yb,
           sofa_like_rule(Xb) / 6.0, len(a_idx))
    lr2 = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced",
                                           random_state=SEED))
    lr2.fit(XfullA, yfullA)
    record("Logistic regression", "cross_setA_to_setB", yb,
           lr2.predict_proba(Xb)[:, 1], len(a_idx))
    lgb2 = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          random_state=SEED, n_jobs=-1, verbose=-1)
    lgb2.fit(XfullA, yfullA)
    record("LightGBM (all features)", "cross_setA_to_setB", yb,
           lgb2.predict_proba(Xb)[:, 1], len(a_idx), feat, lgb2)
    xgb2 = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                         eval_metric="logloss", random_state=SEED, n_jobs=-1)
    xgb2.fit(XfullA, yfullA)
    record("XGBoost (all features)", "cross_setA_to_setB", yb,
           xgb2.predict_proba(Xb)[:, 1], len(a_idx))

    met = pd.DataFrame(rows)
    met.to_csv(os.path.join(out, "metrics.csv"), index=False)
    pd.concat(preds, ignore_index=True).to_csv(os.path.join(out, "predictions.csv"), index=False)
    if importances:
        pd.concat(importances, ignore_index=True).to_csv(
            os.path.join(out, "feature_importance.csv"), index=False)
    with open(os.path.join(out, "run_info.json"), "w", encoding="utf-8") as fh:
        json.dump({"seed": SEED, "window_hours": 6, "n_patients": int(len(df)),
                   "n_features": len(feat), "within_train": int(len(tr_idx)),
                   "within_test": int(len(te_idx)), "cross_train": int(len(a_idx)),
                   "cross_test": int(len(b_idx)), "quick": bool(args.quick),
                   "elapsed_s": round(time.time() - t0, 1)}, fh, indent=2)
    print("[train] done |", len(met), "experiment rows | elapsed",
          round(time.time() - t0, 1), "s", flush=True)
    print(met[["experiment", "split", "auroc", "auprc"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
