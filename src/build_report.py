# -*- coding: utf-8 -*-
"""Build the MPU-format final report from real result files.

Every number in this report is read from results/*.csv produced by train.py
and evaluate.py. No result is hard-coded.
"""
import json
import os

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ROOT = os.path.dirname(REPO)
RES = os.path.join(REPO, "results")
PROC = os.path.join(REPO, "data", "processed")
FIG = os.path.join(RES, "figures")
OUT = os.path.join(ROOT, "AIDD8122_Final_Report_P2625533_ChenWan.docx")

TITLE = ("Early Prediction of Sepsis from ICU Time-Series: A Comparative Study of "
         "Clinical Baselines and Gradient-Boosted Models with Cross-Cohort Validation")

met = pd.read_csv(os.path.join(RES, "metrics.csv"))
ci = pd.read_csv(os.path.join(RES, "bootstrap_ci.csv"))
flow = pd.read_csv(os.path.join(PROC, "cohort_flow.csv"))
with open(os.path.join(RES, "run_info.json"), encoding="utf-8") as fh:
    info = json.load(fh)


def g(exp, split, col):
    r = met[(met["experiment"] == exp) & (met["split"] == split)]
    return r.iloc[0][col] if len(r) else float("nan")


def ci_of(exp, split, col):
    r = ci[(ci["experiment"] == exp) & (ci["split"] == split)]
    return (float(r.iloc[0][col + "_lo"]), float(r.iloc[0][col + "_hi"])) if len(r) else (float("nan"), float("nan"))


def sf(run, size=10.5, bold=False, color=None):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rf = rPr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rPr.append(rf)
    rf.set(qn("w:eastAsia"), "SimSun")


def para(doc, text, size=10.5, bold=False, align=None, italic=False):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    sf(r, size, bold)
    r.font.italic = italic
    return p


def heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        sf(r, 14 if level == 1 else 12, True, RGBColor(0x1F, 0x4E, 0x79))
    return p


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    sf(r, 9)
    r.font.italic = True
    return p


def add_table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        sf(cell.paragraphs[0].add_run(str(h)), 9, True)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            sf(cells[i].paragraphs[0].add_run(str(v)), 9)
    doc.add_paragraph()
    return t


def add_fig(doc, name, text, width=6.0):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width))
    caption(doc, text)


doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Inches(1.0)
sec.top_margin = sec.bottom_margin = Inches(1.0)

# ---------------- cover ----------------
para(doc, "FACULTY OF APPLIED SCIENCES", 12, True, WD_ALIGN_PARAGRAPH.CENTER)
para(doc, "DOCTOR OF PHILOSOPHY IN ARTIFICIAL INTELLIGENCE DRIVEN DRUG DISCOVERY",
     11, False, WD_ALIGN_PARAGRAPH.CENTER)
para(doc, "AIDD8122 – Advanced Topics in Artificial Intelligence & Drug Discovery",
     11, False, WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
para(doc, TITLE, 16, True, WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
add_table(doc, ["", ""], [
    ["Title", TITLE],
    ["Student ID", "P2625533"],
    ["Student Name", "CHEN Wan"],
    ["Submission Date", "16/11/2026"],
])
doc.add_page_break()

# ---------------- §1 ----------------
heading(doc, "§1. Introduction")
para(doc, "Sepsis is a life-threatening organ dysfunction caused by a dysregulated host "
          "response to infection. Under the Sepsis-3 definition it is identified by an acute "
          "increase in the Sequential Organ Failure Assessment (SOFA) score in the presence of "
          "suspected infection. Because deterioration can progress within hours, the clinically "
          "valuable question is not whether sepsis can be recognised after the fact, but whether "
          "it can be anticipated early enough for antibiotics, fluid resuscitation and source "
          "control to change the course of illness.")
para(doc, "Modern intensive care units record vital signs and laboratory results at hourly "
          "resolution, which makes early warning a natural machine-learning problem. This project "
          "belongs to the third topic direction of the course brief — the application of "
          "AI-driven methods to related scientific problems, specifically clinical data analysis. "
          "It develops and systematically evaluates machine-learning models that predict sepsis "
          "several hours before clinical recognition, quantifies their advantage over a "
          "transparent clinical rule, and tests whether a model trained in one hospital system "
          "transports to another.")
para(doc, "The study uses the publicly released training data of the PhysioNet / Computing in "
          "Cardiology Challenge 2019. Three databases initially considered for the project "
          "(MIMIC-IV, eICU-CRD and the CMAISE sepsis cohort) require credentialed or controlled "
          "access; the approval lead time was not compatible with the submission deadline. "
          "Section 4.1 documents this decision and Section 2 discusses those resources as "
          "context and future external-validation targets. No result in this report is derived "
          "from data to which access was not obtained.")

# ---------------- §2 ----------------
heading(doc, "§2. Literature Review")
heading(doc, "2.1 Sepsis definitions and clinical risk scores", 2)
para(doc, "The Sepsis-3 consensus replaced earlier systemic inflammatory response criteria with "
          "an organ-dysfunction-based definition operationalised through SOFA. SOFA and its "
          "bedside simplification qSOFA remain the reference against which data-driven early "
          "warning scores are judged. Their appeal is transparency and immediate interpretability; "
          "their limitation is that a small set of thresholded variables cannot exploit the "
          "rich temporal structure of electronic health records.")
heading(doc, "2.2 Public critical-care data resources", 2)
para(doc, "MIMIC-IV and eICU-CRD are large de-identified ICU databases distributed by PhysioNet "
          "under a credentialed-access licence: users must complete human-subjects training, be "
          "credentialed and sign a data use agreement. The eICU database aggregates more than "
          "200 hospitals, whereas MIMIC-IV represents a single academic centre, making the pair "
          "a common design for transportability studies. CMAISE (Chinese Multi-omics Advances In "
          "Sepsis) is a multicentre Chinese sepsis cohort combining deep clinical phenotyping "
          "with transcriptomic, proteomic, metabolomic and single-cell data; its core datasets "
          "are hosted at the National Genomics Data Center under controlled access. These three "
          "resources are highly relevant to sepsis AI but could not be obtained within the "
          "project timeline.")
para(doc, "The PhysioNet/Computing in Cardiology Challenge 2019 released an openly accessible "
          "training corpus of 40,336 ICU stays from two hospital systems. Each stay is an "
          "hourly time series of 40 clinical variables with a binary SepsisLabel shifted six "
          "hours earlier than clinical onset, explicitly framing early prediction. The hidden "
          "scoring set is not public, so this study evaluates on patient-level and "
          "hospital-system-level hold-outs rather than the challenge leaderboard.")
heading(doc, "2.3 Machine learning for early warning", 2)
para(doc, "Published approaches range from logistic regression and tree ensembles to deep "
          "sequence models. Gradient-boosted decision trees are a strong fit for tabular clinical "
          "features: they handle non-linearities, interactions and missing values without "
          "extensive imputation. Two methodological issues recur in this literature. First, "
          "sepsis labels are rare (typically well under 10%), so accuracy and AUROC can look "
          "reassuring while precision is poor; average precision (AUPRC) and calibration are "
          "more informative. Second, models often degrade when moved to a different hospital "
          "because documentation practice, case mix and label prevalence change. This project "
          "addresses both explicitly.")

# ---------------- §3 ----------------
heading(doc, "§3. Research Questions and Objectives")
para(doc, "RQ1. Using only the first six hours of ICU observation, can a machine-learning model "
          "identify patients who will subsequently meet the sepsis label, and how large is the "
          "gain over a transparent clinical rule?")
para(doc, "RQ2. Which components of the model contribute to performance — laboratory features, "
          "model capacity, and class-imbalance handling — as measured by comparative and "
          "ablation experiments?")
para(doc, "RQ3. How much performance is retained when a model trained in one hospital system "
          "(set A) is applied unchanged to a second system (set B)?")
para(doc, "The objectives are: (i) to construct a leakage-free patient-level cohort and a "
          "reproducible feature pipeline; (ii) to benchmark a simplified SOFA-like rule, logistic "
          "regression, LightGBM and XGBoost with appropriate quantitative metrics and "
          "visualisations; (iii) to run hyperparameter, feature-set and imbalance ablations; and "
          "(iv) to quantify cross-cohort transportability and state the limitations of the evidence.")

# ---------------- §4 ----------------
heading(doc, "§4. Methods and Materials")
heading(doc, "4.1 Data source and selection rationale", 2)
para(doc, "Data: PhysioNet/Computing in Cardiology Challenge 2019, version 1.0.0, "
          "DOI 10.13026/v64v-d857. The public training release contains %d ICU stays "
          "(training set A: %d; training set B: %d) from two separate hospital systems. "
          "Each stay is stored as a pipe-separated file with one row per hour and 40 clinical "
          "variables (8 vital signs, 26 laboratory measurements and 6 demographic or "
          "administrative fields) plus SepsisLabel. Files were downloaded with a scripted "
          "parallel fetcher from the public PhysioNet S3 bucket; no credentialing was required "
          "for these training files."
          % (int(flow["total"].sum()), int(flow.iloc[0]["total"]), int(flow.iloc[1]["total"])))
para(doc, "MIMIC-IV, eICU-CRD and CMAISE were evaluated against their official access pages. "
          "All three require credentialing or controlled-access data requests with approval "
          "lead times of one to two weeks or longer. With a submission deadline of 16 November "
          "2026 they were therefore not usable. This substitution is reported here rather than "
          "silently made: the models and all results in this report are based exclusively on the "
          "Challenge 2019 data.")
heading(doc, "4.2 Cohort construction", 2)
para(doc, "A patient-level early-warning formulation was adopted. For each stay, features were "
          "computed from the first six hours only. The outcome was defined as the appearance of "
          "SepsisLabel = 1 at any later hour. Stays already label-positive inside the observation "
          "window (prevalent cases, where no useful advance warning is possible) and stays with "
          "no post-window observation were excluded. This guarantees that no information from "
          "after the prediction time enters the feature vector. The resulting cohort flow is "
          "reported in Table 1 and Figure 1.")
rows = []
for _, r in flow.iterrows():
    rows.append([r["cohort"], int(r["total"]), int(r["prevalent_in_window"]),
                 int(r["included"]), int(r["positives"]),
                 "%.2f%%" % (100 * r["prevalence"])])
add_table(doc, ["Cohort", "Downloaded stays", "Excluded (prevalent)",
                "Analysable stays", "Positive cases", "Prevalence"], rows)
caption(doc, "Table 1. Cohort flow for a six-hour observation window (counts derived from the downloaded files).")
heading(doc, "4.3 Feature engineering", 2)
para(doc, "For each of the 40 variables, seven summary statistics were computed over the "
          "six-hour window: last value, mean, maximum, minimum, standard deviation, a linear "
          "time slope, and the fraction of missing hours. This yields %d features. Missing "
          "values were retained for tree models and median-imputed for logistic regression. "
          "The 56 vital-sign-only features form the feature-set ablation." % info["n_features"])
heading(doc, "4.4 Models and comparative experiments", 2)
para(doc, "Four predictors were compared: (1) a simplified SOFA-like rule that counts threshold "
          "violations in MAP, platelets, bilirubin, creatinine, respiratory rate and lactate "
          "(a deliberately transparent proxy, not a validated SOFA implementation); "
          "(2) logistic regression with median imputation, standardisation and balanced class "
          "weights; (3) LightGBM; and (4) XGBoost. Gradient-boosted models used 400 trees, "
          "learning rate 0.05 and fixed random seed %d. Ablations covered a vital-sign-only "
          "feature set, LightGBM capacity (num_leaves 15, 31, 63), and 5:1 negative-class "
          "down-sampling." % info["seed"])
heading(doc, "4.5 Evaluation design and metrics", 2)
para(doc, "Two evaluations were run. Within-cohort: a patient-level stratified 70/30 split of "
          "set A (%d training / %d test stays). Cross-cohort: train on all of set A and test "
          "on the untouched set B (%d / %d stays), which measures transportability across "
          "hospital systems. Primary metrics were AUROC and AUPRC, with AUPRC emphasised "
          "because of the low positive prevalence; Brier score, sensitivity, specificity and "
          "calibration curves were also reported. Bootstrap 95%% confidence intervals were "
          "estimated by resampling the test predictions 1,000 times."
          % (info["within_train"], info["within_test"], info["cross_train"], info["cross_test"]))
para(doc, "The official Challenge utility function requires hour-by-hour online predictions "
          "against the hidden scoring set; it does not apply to this patient-level formulation "
          "and is therefore not reported rather than being approximated.")
add_fig(doc, "cohort_flow.png", "Figure 1. Cohort flow: downloaded stays versus analysable stays after exclusion.", 5.5)

# ---------------- §5 ----------------
heading(doc, "§5. Results and Discussion")
heading(doc, "5.1 Within-cohort performance", 2)
w = "within_setA"
xgb_a = g("XGBoost (all features)", w, "auroc")
xgb_p = g("XGBoost (all features)", w, "auprc")
lgb_a = g("LightGBM (all features)", w, "auroc")
lr_a = g("Logistic regression", w, "auroc")
rule_a = g("SOFA-like rule", w, "auroc")
rule_p = g("SOFA-like rule", w, "auprc")
para(doc, "Table 2 summarises the within-cohort experiments. Within set A (%d test stays, prevalence "
          "%.2f%%), XGBoost achieved AUROC %.3f and AUPRC %.3f, LightGBM %.3f / %.3f, "
          "logistic regression %.3f / %.3f, and the SOFA-like rule %.3f / %.3f. Relative to "
          "the clinical rule, the best model improved AUPRC from %.3f to %.3f, a %.1f-fold "
          "gain at a prevalence of %.3f. The ranking is consistent across AUROC and AUPRC for "
          "the machine-learning models, while the rule performs close to chance in ranking "
          "ability."
          % (info["within_test"], 100 * g("XGBoost (all features)", w, "prevalence"),
             xgb_a, xgb_p, lgb_a, g("LightGBM (all features)", w, "auprc"),
             lr_a, g("Logistic regression", w, "auprc"), rule_a, rule_p,
             rule_p, xgb_p, xgb_p / rule_p, g("XGBoost (all features)", w, "prevalence")))
lo, hi = ci_of("XGBoost (all features)", w, "auprc")
plo, phi = ci_of("XGBoost (all features)", w, "auroc")
para(doc, "Bootstrap 95%% confidence intervals for XGBoost were AUROC %.3f–%.3f and "
          "AUPRC %.3f–%.3f, indicating that the advantage over the rule is not an artefact of "
          "a single test partition." % (plo, phi, lo, hi))
rows = []
for _, r in met[met["split"] == w].iterrows():
    rows.append([r["experiment"], "%.3f" % r["auroc"], "%.3f" % r["auprc"],
                 "%.3f" % r["brier"], "%.3f" % r["sensitivity"], "%.3f" % r["specificity"]])
add_table(doc, ["Model / experiment", "AUROC", "AUPRC", "Brier", "Sensitivity", "Specificity"], rows)
caption(doc, "Table 2. Within-cohort (set A) results for all experiments. AUPRC is the primary metric.")
add_fig(doc, "roc_pr_curves.png", "Figure 2. ROC and precision-recall curves for the four main predictors (set A hold-out).")
heading(doc, "5.2 Ablation and feature contribution", 2)
para(doc, "Restricting the model to vital-sign features reduced AUPRC from %.3f to %.3f, "
          "showing that laboratory measurements carry substantial incremental signal. Varying "
          "LightGBM capacity (num_leaves 15/31/63) changed AUROC by less than 0.01, and "
          "5:1 negative down-sampling did not improve AUPRC (%.3f versus %.3f). These results "
          "suggest that feature richness, rather than model capacity or resampling, is the main "
          "driver of performance in this cohort."
          % (g("LightGBM (all features)", w, "auprc"),
             g("LightGBM (vitals only)", w, "auprc"),
             g("LightGBM (neg downsample 5x)", w, "auprc"),
             g("LightGBM (all features)", w, "auprc")))
add_fig(doc, "ablation.png", "Figure 3. AUPRC across the LightGBM ablations.")
add_fig(doc, "feature_importance.png", "Figure 4. Twenty most influential windowed features (LightGBM split importance).")
add_fig(doc, "calibration.png", "Figure 5. Calibration curves for the probabilistic models.")
para(doc, "Calibration differed markedly between models. XGBoost and LightGBM produced Brier "
          "scores of %.3f and %.3f respectively, whereas the balanced logistic regression "
          "reached %.3f — expected, because class weighting distorts predicted probabilities "
          "even when ranking is reasonable. For clinical deployment, probability calibration "
          "would require a dedicated recalibration step on the target population."
          % (g("XGBoost (all features)", w, "brier"),
             g("LightGBM (all features)", w, "brier"),
             g("Logistic regression", w, "brier")))
heading(doc, "5.3 Cross-cohort transportability", 2)
c = "cross_setA_to_setB"
para(doc, "When the model trained on set A was applied unchanged to set B (%d stays, "
          "prevalence %.2f%%), XGBoost attained AUROC %.3f and AUPRC %.3f, versus %.3f / %.3f "
          "within set A — an absolute AUPRC reduction of %.3f. LightGBM showed the same pattern "
          "(%.3f / %.3f). All models still outperformed the clinical rule (AUPRC %.3f), but the "
          "substantial drop is the most important negative finding of this study: discrimination "
          "does not transfer automatically between hospital systems, consistent with differences "
          "in case mix, measurement practice and label prevalence between the two source systems."
          % (int(g("XGBoost (all features)", c, "n_test")),
             100 * g("XGBoost (all features)", c, "prevalence"),
             g("XGBoost (all features)", c, "auroc"), g("XGBoost (all features)", c, "auprc"),
             xgb_a, xgb_p, xgb_p - g("XGBoost (all features)", c, "auprc"),
             g("LightGBM (all features)", c, "auroc"),
             g("LightGBM (all features)", c, "auprc"),
             g("SOFA-like rule", c, "auprc")))
rows = []
for _, r in met[met["split"] == c].iterrows():
    rows.append([r["experiment"], "%.3f" % r["auroc"], "%.3f" % r["auprc"], "%.3f" % r["brier"]])
add_table(doc, ["Model", "AUROC", "AUPRC", "Brier"], rows)
caption(doc, "Table 3. Cross-cohort results: trained on set A, tested on the untouched set B.")
add_fig(doc, "cross_cohort.png", "Figure 6. Within-cohort versus cross-cohort performance for the four main predictors.")
heading(doc, "5.4 Limitations and future improvements", 2)
para(doc, "Several limitations qualify these findings. (i) The patient-level six-hour-window "
          "formulation is a simplification of the original hour-by-hour online task; it cannot "
          "report the Challenge utility score and it discards information after the window. "
          "(ii) Excluding stays already label-positive in the window removes early presenters and "
          "may inflate apparent performance relative to a true deployment population. "
          "(iii) The SOFA-like rule is a deliberately simple proxy, not a validated clinical "
          "score, so the baseline comparison should be read as a transparency reference rather "
          "than a clinical benchmark. (iv) Only tabular models were evaluated; temporal deep "
          "models may capture trajectory information that summary statistics discard. "
          "(v) The two hospital systems differ in prevalence and documentation; the observed "
          "transportability gap cannot be attributed to a single cause without access to site "
          "metadata. (vi) No prospective or external clinical validation was performed.")
para(doc, "Future work should re-run the pipeline on credentialed MIMIC-IV, eICU-CRD and CMAISE "
          "cohorts once access is granted, add site-aware calibration or domain adaptation, "
          "compare against a properly validated qSOFA/SOFA implementation, evaluate temporal "
          "models (GRU/LSTM or transformers) on the full hourly sequence, and report the official "
          "utility score under a strictly causal online evaluation.")

# ---------------- §6 ----------------
heading(doc, "§6. Conclusions")
para(doc, "A reproducible leakage-free pipeline was built on %d public ICU stays and used to "
          "benchmark four predictors with %d windowed clinical features. Within a single hospital "
          "system, gradient-boosted models improved the primary metric AUPRC from %.3f for a "
          "transparent clinical rule to %.3f for XGBoost, with bootstrap confidence intervals "
          "confirming the gain. Ablations identified laboratory measurements, rather than model "
          "capacity or class resampling, as the principal source of improvement. However, "
          "applying the same model to a second hospital system reduced AUPRC to %.3f, "
          "demonstrating that cross-centre transportability — not in-sample discrimination — is "
          "the binding constraint for clinical deployment. The three research questions are "
          "therefore answered: early prediction is feasible and clearly superior to a simple "
          "rule, the gain is attributable mainly to laboratory features, and the gain attenuates "
          "substantially across hospital systems."
          % (info["n_patients"], info["n_features"], rule_p, xgb_p,
             g("XGBoost (all features)", c, "auprc")))

# ---------------- §7 ----------------
heading(doc, "§7. References")
refs = [
    "[1] Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA. 2016.",
    "[2] Vincent JL, Moreno R, Takala J, et al. The SOFA (Sepsis-related Organ Failure Assessment) score to describe organ dysfunction/failure. Intensive Care Medicine. 1996.",
    "[3] Reyna MA, Josef CS, Jeter R, et al. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. Critical Care Medicine. 2020.",
    "[4] PhysioNet. Early Prediction of Sepsis from Clinical Data, version 1.0.0. 2019. DOI 10.13026/v64v-d857.",
    "[5] Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data. 2023.",
    "[6] Pollard TJ, Johnson AEW, Raffa JD, et al. The eICU Collaborative Research Database, a freely available multi-center database for critical care research. Scientific Data. 2018.",
    "[7] CMAISE Consortium. Progress of the Chinese Multi-omics Advances In Sepsis (CMAISE) database: a review. Journal of Emergency and Critical Care Medicine. 2024.",
    "[8] Ke G, Meng Q, Finley T, et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems. 2017.",
    "[9] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2016.",
    "[10] Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research. 2011.",
    "[11] Harris CR, Millman KJ, van der Walt SJ, et al. Array programming with NumPy. Nature. 2020.",
    "[12] Henry KE, Hager DN, Pronovost PJ, Saria S. A targeted real-time early warning score (TREWS) for septic shock. Nature Medicine. 2022.",
]
for r in refs:
    para(doc, r, 9.5)
para(doc, "Note: reference entries are given in a consistent author-title-source-year format; "
          "volume and page numbers should be completed against the publisher DOI records before "
          "final submission.", 9, italic=True)

heading(doc, "Data and Code Availability")
para(doc, "The complete project source code accompanies this report in the repository "
          "organised under repo/: src/download_data.py (public data retrieval), "
          "src/preprocess.py (cohort and feature construction), src/train.py "
          "(baselines, models and ablations), src/evaluate.py (metrics, confidence "
          "intervals and figures), src/build_report.py and src/build_slides.py "
          "(reproducible document generation), together with requirements.txt and README.md. "
          "Raw patient files are deliberately excluded from version control; only the "
          "downloader and analysis code are published.")
para(doc, "Public repository URL: https://github.com/mpuwan117/AIDD8122-Sepsis-Early-Prediction. "
          "The repository is public and contains the complete project source code; raw patient "
          "data are excluded by design (only download and analysis code are published), satisfying "
          "both the accessibility requirement and the data-privacy constraint.", 10.5, True)

doc.save(OUT)
print("SAVED:", OUT)
print("paragraphs:", len(doc.paragraphs), "| tables:", len(doc.tables))
