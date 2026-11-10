# -*- coding: utf-8 -*-
"""Figures and bootstrap confidence intervals from real model outputs.

Usage:
    python src/evaluate.py --results ../results --processed ../data/processed
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

MAIN = ["SOFA-like rule", "Logistic regression",
        "LightGBM (all features)", "XGBoost (all features)"]
COLORS = {"SOFA-like rule": "#8A8A8A", "Logistic regression": "#4C78A8",
          "LightGBM (all features)": "#E45756", "XGBoost (all features)": "#54A24B"}


def bootstrap_ci(y, p, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype="float64")
    aurocs, auprcs = [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(y[idx], p[idx]))
        auprcs.append(average_precision_score(y[idx], p[idx]))
    return {"auroc_lo": float(np.percentile(aurocs, 2.5)),
            "auroc_hi": float(np.percentile(aurocs, 97.5)),
            "auprc_lo": float(np.percentile(auprcs, 2.5)),
            "auprc_hi": float(np.percentile(auprcs, 97.5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--processed", default="../data/processed")
    args = ap.parse_args()
    res = os.path.abspath(args.results)
    proc = os.path.abspath(args.processed)
    figdir = os.path.join(res, "figures")
    os.makedirs(figdir, exist_ok=True)

    met = pd.read_csv(os.path.join(res, "metrics.csv"))
    pred = pd.read_csv(os.path.join(res, "predictions.csv"))
    ci_rows = []

    # ---- Figure 1: ROC and PR curves (within setA) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in MAIN:
        sub = pred[(pred["experiment"] == name) & (pred["split"] == "within_setA")]
        if len(sub) == 0:
            continue
        y, p = sub["y_true"].to_numpy(), sub["y_score"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        axes[0].plot(fpr, tpr, label="%s (AUROC %.3f)" % (name, roc_auc_score(y, p)),
                     color=COLORS.get(name, "#333333"), linewidth=1.8)
        axes[1].plot(rc, pr, label="%s (AUPRC %.3f)" % (name, average_precision_score(y, p)),
                     color=COLORS.get(name, "#333333"), linewidth=1.8)
        ci = bootstrap_ci(y, p)
        ci_rows.append({"experiment": name, "split": "within_setA", **ci})
    axes[0].plot([0, 1], [0, 1], "--", color="#BBBBBB", linewidth=1)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC - within-cohort (setA)"); axes[0].legend(fontsize=7, loc="lower right")
    baseline = pred[(pred["experiment"] == MAIN[0]) &
                    (pred["split"] == "within_setA")]["y_true"].mean()
    axes[1].axhline(baseline, linestyle="--", color="#BBBBBB", linewidth=1,
                    label="prevalence %.3f" % baseline)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall - within-cohort (setA)")
    axes[1].legend(fontsize=7, loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.25); ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "roc_pr_curves.png"), dpi=180); plt.close(fig)

    # ---- Figure 2: calibration ----
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ["Logistic regression", "LightGBM (all features)", "XGBoost (all features)"]:
        sub = pred[(pred["experiment"] == name) & (pred["split"] == "within_setA")]
        if len(sub) == 0:
            continue
        pt, pp = calibration_curve(sub["y_true"], sub["y_score"], n_bins=10, strategy="quantile")
        ax.plot(pp, pt, marker="o", markersize=4, label=name,
                color=COLORS.get(name, "#333333"))
    ax.plot([0, 1], [0, 1], "--", color="#BBBBBB")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction")
    ax.set_title("Calibration - within-cohort (setA)"); ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "calibration.png"), dpi=180); plt.close(fig)

    # ---- Figure 3: feature importance ----
    fimp = os.path.join(res, "feature_importance.csv")
    if os.path.exists(fimp):
        fi = pd.read_csv(fimp)
        fi = fi[fi["experiment"] == "LightGBM (all features)"]
        top = fi.nlargest(20, "importance").sort_values("importance")
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(top["feature"], top["importance"], color="#4C78A8")
        ax.set_xlabel("LightGBM split importance")
        ax.set_title("Top 20 windowed features")
        fig.tight_layout(); fig.savefig(os.path.join(figdir, "feature_importance.png"), dpi=180)
        plt.close(fig)

    # ---- Figure 4: cross-cohort comparison ----
    cc = met[met["split"].isin(["within_setA", "cross_setA_to_setB"])]
    cc = cc[cc["experiment"].isin(MAIN)]
    labels = cc["experiment"] + "\n" + cc["split"].map(
        {"within_setA": "within", "cross_setA_to_setB": "cross"})
    x = np.arange(len(cc))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, cc["auroc"], width=0.4, label="AUROC", color="#4C78A8")
    ax.bar(x + 0.2, cc["auprc"], width=0.4, label="AUPRC", color="#E45756")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1)
    ax.set_title("Within-cohort vs cross-cohort (setA -> setB)")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "cross_cohort.png"), dpi=180)
    plt.close(fig)

    # ---- Figure 5: ablation ----
    ab = met[(met["split"] == "within_setA") &
             (met["experiment"].str.startswith("LightGBM"))]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ab["experiment"], ab["auprc"], color=["#E45756" if "all features" in e else "#4C78A8"
                                                  for e in ab["experiment"]])
    ax.set_xlabel("AUPRC"); ax.set_title("Ablation: feature set, capacity, imbalance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "ablation.png"), dpi=180)
    plt.close(fig)

    # ---- Figure 6: cohort flow ----
    flow_path = os.path.join(proc, "cohort_flow.csv")
    if os.path.exists(flow_path):
        flow = pd.read_csv(flow_path)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(flow))
        ax.bar(x - 0.2, flow["total"], width=0.4, label="Downloaded stays", color="#BBBBBB")
        ax.bar(x + 0.2, flow["included"], width=0.4, label="Analysable stays", color="#4C78A8")
        for i, r in flow.iterrows():
            ax.text(i - 0.2, r["total"], str(int(r["total"])), ha="center", va="bottom", fontsize=8)
            ax.text(i + 0.2, r["included"], str(int(r["included"])), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(flow["cohort"])
        ax.set_ylabel("ICU stays"); ax.set_title("Cohort flow (6-h observation window)")
        ax.legend(); ax.grid(axis="y", alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(figdir, "cohort_flow.png"), dpi=180)
        plt.close(fig)

    # ---- cross-cohort bootstrap CI ----
    for name in MAIN:
        sub = pred[(pred["experiment"] == name) & (pred["split"] == "cross_setA_to_setB")]
        if len(sub):
            ci = bootstrap_ci(sub["y_true"].to_numpy(), sub["y_score"].to_numpy())
            ci_rows.append({"experiment": name, "split": "cross_setA_to_setB", **ci})
    pd.DataFrame(ci_rows).to_csv(os.path.join(res, "bootstrap_ci.csv"), index=False)

    # ---- tables for the report ----
    met.to_csv(os.path.join(res, "table_results.csv"), index=False)
    print("[evaluate] figures:", sorted(os.listdir(figdir)))
    print("[evaluate] bootstrap CI rows:", len(ci_rows))
    print("[evaluate] metrics summary:")
    print(met[["experiment", "split", "n_test", "prevalence", "auroc", "auprc", "brier"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
