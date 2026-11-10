# -*- coding: utf-8 -*-
"""Build the 10-minute presentation from the real result files."""
import json
import os

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ROOT = os.path.dirname(REPO)
RES = os.path.join(REPO, "results")
PROC = os.path.join(REPO, "data", "processed")
FIG = os.path.join(RES, "figures")
OUT = os.path.join(ROOT, "AIDD8122_Presentation_P2625533_ChenWan.pptx")

met = pd.read_csv(os.path.join(RES, "metrics.csv"))
flow = pd.read_csv(os.path.join(PROC, "cohort_flow.csv"))
with open(os.path.join(RES, "run_info.json"), encoding="utf-8") as fh:
    info = json.load(fh)
W, C = "within_setA", "cross_setA_to_setB"


def g(e, s, c):
    r = met[(met["experiment"] == e) & (met["split"] == s)]
    return r.iloc[0][c] if len(r) else float("nan")


NAVY = RGBColor(0x1F, 0x4E, 0x79)
DARK = RGBColor(0x22, 0x22, 0x22)
GREY = RGBColor(0x66, 0x66, 0x66)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]


def text(slide, x, y, w, h, s, size=18, bold=False, color=DARK, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = s.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln
        p.alignment = align
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.color.rgb = color
            r.font.name = "Calibri"
    return tb


def slide(title, body, size=17):
    s = prs.slides.add_slide(blank)
    text(s, 0.6, 0.35, 12.1, 0.8, title, 27, True, NAVY)
    text(s, 0.6, 1.35, 12.1, 5.6, body, size)
    return s


def pic(slide, name, x, y, w):
    p = os.path.join(FIG, name)
    if os.path.exists(p):
        slide.shapes.add_picture(p, Inches(x), Inches(y), width=Inches(w))


# 1 title
s = prs.slides.add_slide(blank)
text(s, 0.8, 1.6, 11.7, 1.6,
     "Early Prediction of Sepsis from ICU Time-Series", 34, True, NAVY, PP_ALIGN.CENTER)
text(s, 0.8, 3.1, 11.7, 1.0,
     "Clinical baselines vs gradient-boosted models, with cross-cohort validation",
     20, False, DARK, PP_ALIGN.CENTER)
text(s, 0.8, 4.6, 11.7, 1.0,
     "AIDD8122 Final Project  |  CHEN Wan (P2625533)  |  16 November 2026",
     16, False, GREY, PP_ALIGN.CENTER)

# 2 outline
slide("Outline", "1  Background and motivation\n2  Research questions\n3  Data source and selection rationale\n"
      "4  Cohort construction and features\n5  Models and evaluation design\n6  Results: within-cohort performance\n"
      "7  Ablations and calibration\n8  Cross-cohort transportability\n9  Limitations and conclusions", 19)

# 3 background
slide("Background: why early prediction matters",
      "Sepsis-3 defines sepsis by acute organ dysfunction (SOFA increase) with suspected infection.\n\n"
      "Deterioration can progress within hours; antibiotics, fluids and source control are time-critical.\n\n"
      "ICUs record vital signs and labs hourly -> a natural machine-learning problem.\n\n"
      "Two recurring problems in the literature: very low positive prevalence makes accuracy/AUROC "
      "misleading, and models often fail to transfer between hospitals.", 18)

# 4 research questions
slide("Research questions and objectives",
      "RQ1  Can models using only the first 6 hours of ICU data predict subsequent sepsis, "
      "and how much better are they than a transparent clinical rule?\n\n"
      "RQ2  What drives performance: laboratory features, model capacity, or class-imbalance handling?\n\n"
      "RQ3  How much performance survives transfer from hospital system A to system B?\n\n"
      "Objectives: leakage-free cohort, reproducible pipeline, quantitative benchmarking, "
      "ablations, and an explicit transportability test.", 18)

# 5 data source
n_total = int(flow["total"].sum())
slide("Data source and why this cohort",
      "PhysioNet / Computing in Cardiology Challenge 2019, v1.0.0 (DOI 10.13026/v64v-d857).\n\n"
      "%d public ICU stays from two hospital systems (set A %d, set B %d); "
      "hourly 40 clinical variables + SepsisLabel shifted 6 h before onset.\n\n"
      "Originally considered MIMIC-IV, eICU-CRD and CMAISE: all require credentialed or "
      "controlled access with 1-2 week approval. With a 16 Nov deadline, unusable.\n\n"
      "They are discussed in the literature review and future-work section. "
      "No result in this project comes from data we could not access." % (
          n_total, int(flow.iloc[0]["total"]), int(flow.iloc[1]["total"])), 17)

# 6 cohort
n_incl = int(flow["included"].sum())
n_prev = int(flow["prevalent_in_window"].sum())
s = slide("Cohort and feature engineering",
          "%d stays downloaded -> %d excluded as already label-positive in the first 6 h "
          "-> %d analysable stays (%d positive overall).\n\n"
          "Features: for each of 40 variables, last / mean / max / min / std / slope / "
          "missingness over the first 6 hours = %d features.\n\n"
          "Label: SepsisLabel appears after the window. No post-window data enters the features." % (
              n_total, n_prev, n_incl, int(flow["positives"].sum()), info["n_features"]), 17)
pic(s, "cohort_flow.png", 8.0, 3.9, 4.8)

# 7 methods
slide("Models and evaluation design",
      "Predictors: simplified SOFA-like rule (transparent proxy), logistic regression, "
      "LightGBM, XGBoost.\n\n"
      "Ablations: vital-sign features only; LightGBM capacity (15/31/63 leaves); "
      "5:1 negative down-sampling.\n\n"
      "Within-cohort: patient-level 70/30 split of set A (%d / %d).\n"
      "Cross-cohort: train on all set A, test on untouched set B (%d / %d).\n\n"
      "Metrics: AUROC and AUPRC (primary, low prevalence), Brier, sensitivity, specificity, "
      "calibration and 1,000-sample bootstrap CIs." % (
          info["within_train"], info["within_test"], info["cross_train"], info["cross_test"]), 17)

# 8 results
xp = g("XGBoost (all features)", W, "auprc")
rp = g("SOFA-like rule", W, "auprc")
s = slide("Results 1: within-cohort performance",
          "AUROC / AUPRC on %d held-out set-A stays (prevalence %.2f%%):\n\n"
          "  SOFA-like rule        %.3f / %.3f\n"
          "  Logistic regression   %.3f / %.3f\n"
          "  LightGBM              %.3f / %.3f\n"
          "  XGBoost               %.3f / %.3f\n\n"
          "XGBoost improved AUPRC %.1f-fold over the clinical rule." % (
              info["within_test"], 100 * g("XGBoost (all features)", W, "prevalence"),
              g("SOFA-like rule", W, "auroc"), rp,
              g("Logistic regression", W, "auroc"), g("Logistic regression", W, "auprc"),
              g("LightGBM (all features)", W, "auroc"), g("LightGBM (all features)", W, "auprc"),
              g("XGBoost (all features)", W, "auroc"), xp, xp / rp), 17)
pic(s, "roc_pr_curves.png", 7.4, 2.4, 5.5)

# 9 ablations
s = slide("Results 2: what actually helps",
          "Vital signs only: AUPRC %.3f  vs  all features %.3f  -> laboratory data matters.\n\n"
          "LightGBM leaves 15 / 31 / 63: AUROC %.3f / %.3f / %.3f -> capacity is not the bottleneck.\n\n"
          "5:1 negative down-sampling: AUPRC %.3f vs %.3f -> no gain.\n\n"
          "Calibration: Brier %.3f (XGBoost) and %.3f (LightGBM) vs %.3f (balanced LR); "
          "class weighting distorts probabilities." % (
              g("LightGBM (vitals only)", W, "auprc"),
              g("LightGBM (all features)", W, "auprc"),
              g("LightGBM (leaves=15)", W, "auroc"),
              g("LightGBM (all features)", W, "auroc"),
              g("LightGBM (leaves=63)", W, "auroc"),
              g("LightGBM (neg downsample 5x)", W, "auprc"),
              g("LightGBM (all features)", W, "auprc"),
              g("XGBoost (all features)", W, "brier"),
              g("LightGBM (all features)", W, "brier"),
              g("Logistic regression", W, "brier")), 17)
pic(s, "ablation.png", 8.1, 4.0, 4.6)

# 10 cross-cohort
s = slide("Results 3: cross-cohort transportability (key finding)",
          "Train on set A, test on untouched set B (%d stays, prevalence %.2f%%):\n\n"
          "  XGBoost     %.3f / %.3f   (was %.3f / %.3f within set A)\n"
          "  LightGBM    %.3f / %.3f\n"
          "  Logistic    %.3f / %.3f\n"
          "  SOFA rule   %.3f / %.3f\n\n"
          "AUPRC fell from %.3f to %.3f. Models still beat the rule, but discrimination does "
          "not transfer automatically between hospital systems." % (
              int(g("XGBoost (all features)", C, "n_test")),
              100 * g("XGBoost (all features)", C, "prevalence"),
              g("XGBoost (all features)", C, "auroc"), g("XGBoost (all features)", C, "auprc"),
              g("XGBoost (all features)", W, "auroc"), xp,
              g("LightGBM (all features)", C, "auroc"), g("LightGBM (all features)", C, "auprc"),
              g("Logistic regression", C, "auroc"), g("Logistic regression", C, "auprc"),
              g("SOFA-like rule", C, "auroc"), g("SOFA-like rule", C, "auprc"),
              xp, g("XGBoost (all features)", C, "auprc")), 17)
pic(s, "cross_cohort.png", 8.0, 3.9, 4.8)

# 11 limitations + conclusion
slide("Limitations and conclusions",
      "Limitations: patient-level 6-h window simplifies the online task; prevalent cases excluded; "
      "the SOFA-like rule is a proxy, not a validated score; no temporal deep models; "
      "no prospective validation.\n\n"
      "Conclusions:\n"
      "  - Early prediction is feasible: AUPRC %.3f (XGBoost) vs %.3f (rule).\n"
      "  - Laboratory features, not capacity or resampling, drive the gain.\n"
      "  - Cross-centre transfer is the binding constraint (AUPRC %.3f on set B).\n\n"
      "Next: credentialed MIMIC-IV / eICU-CRD / CMAISE validation, site-aware calibration, "
      "temporal models, and causal online evaluation." % (
          xp, rp, g("XGBoost (all features)", C, "auprc")), 17)

# 12 Q&A
s = prs.slides.add_slide(blank)
text(s, 0.8, 2.8, 11.7, 1.2, "Thank you — questions welcome", 32, True, NAVY, PP_ALIGN.CENTER)
text(s, 0.8, 4.1, 11.7, 0.8,
     "Code and reproducible pipeline: GitHub repository link in the report",
     17, False, GREY, PP_ALIGN.CENTER)

prs.save(OUT)
print("SAVED:", OUT, "| slides:", len(prs.slides))
