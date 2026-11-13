# Early Prediction of Sepsis from ICU Time-Series

**AIDD8122 — Final Project** (Advanced Topics in AI & Drug Discovery), Macao Polytechnic University.
Individual project.

- **Author:** Chen Wan (Student ID: P2625533)
- **Public repository:** https://github.com/mpuwan117/AIDD8122-Sepsis-Early-Prediction
- **Report:** `AIDD8122_Final_Report_P2625533_ChenWan.pdf`
- **Slides:** `AIDD8122_Presentation_P2625533_ChenWan.pdf`

> **Status (final, 2026-11-11):** All experiments are complete; the report, slides and
> both PDFs are generated. Every number in the report and slides is produced by
> `src/train.py` / `src/evaluate.py` on the real data and stored in `results/*.csv`
> (`metrics.csv`, `table_results.csv`, `bootstrap_ci.csv`, `feature_importance.csv`);
> the report and slides are regenerated dynamically from those files by
> `src/build_report.py` / `src/build_slides.py` — no hard-coded figures in the text.
>
> **Headline results:** 39,603 analysable ICU admissions · 280 windowed features.
> XGBoost in-group (set A) AUPRC **0.198** vs. simplified clinical rule **0.090**
> (~2.2×); cross-centre (set A → set B) AUPRC drops to **0.082** (XGBoost) /
> **0.057** (rule), exposing substantial centre-shift degradation.

---

## Data source

**PhysioNet / Computing in Cardiology (CinC) Challenge 2019, v1.0.0**
*Early Prediction of Sepsis from Clinical Data*
DOI: `10.13026/v64v-d857` · https://physionet.org/content/challenge-2019/1.0.0/

- 40,336 ICU patients in the public training release (set A: 20,336; set B: 20,000),
  drawn from two separate hospital systems.
- One pipe-separated `.psv` per patient; each row is one hour of ICU stay.
- 40 clinical variables (8 vital signs, 26 laboratory values, 6 demographics/administrative)
  plus the `SepsisLabel` outcome, shifted 6 hours earlier than clinical recognition.
- Task: predict sepsis 6 hours before clinical suspicion / SOFA-based onset, using
  current and past data only.
- The hidden scoring set is **not** public and is not used here.

### Why the originally proposed databases are not used

The project brief allows any suitable public or self-collected data. The three databases
initially considered — **MIMIC-IV**, **eICU-CRD** and **CMAISE** — were checked against
their official access pages:

| Database | Official status (checked 2026-11-10) |
|---|---|
| MIMIC-IV | PhysioNet credentialed access: CITI training, supervisor reference, 1–2 week review, signed DUA |
| eICU-CRD | Same PhysioNet credentialing / DUA workflow |
| CMAISE (OMIX005606, PRJCA006118) | NGDC `Controlled-access`, requires a data request; the project wiki describes FAIR sharing but the release page is controlled |

With a submission deadline of **2026-11-16 14:30** and no existing credentials or local
copies, these could not be obtained in time. They are therefore discussed in the
literature review and the "external validation / future work" section of the report;
**no results are presented as if they came from those databases**. This substitution is
stated explicitly in the report's Methods section.

---

## Repository layout

```
repo/
├── README.md                 this file
├── requirements.txt          Python dependencies (pinned minimum versions)
├── .gitignore                excludes raw patient data, dev results, large predictions
├── src/
│   ├── download_data.py      public S3 downloader (paginated listing + parallel fetch + resume)
│   ├── preprocess.py         windowed feature extraction (6 h windows → 280 features per stay)
│   ├── train.py              baselines (SOFA-like rule, logistic regression) + XGBoost / LightGBM
│   ├── evaluate.py           AUROC, AUPRC (primary), Brier, bootstrap 95% CIs, calibration, cross-centre
│   ├── build_report.py       regenerates the Word/PDF report from results/*.csv
│   └── build_slides.py       regenerates the PowerPoint/PDF slides from results/*.csv
└── results/                  committed artefacts (raw patient .psv files are NOT committed)
    ├── metrics.csv           main results table (all experiments, both splits)
    ├── table_results.csv     report-ready copy of the results table
    ├── bootstrap_ci.csv      95% confidence intervals from 1,000 bootstrap resamples
    ├── feature_importance.csv per-feature importance for the best XGBoost model
    ├── run_info.json         run metadata (seed, window, cohort sizes, feature count)
    └── figures/              six report figures
        ├── cohort_flow.png        patient/case-selection flow
        ├── roc_pr_curves.png      AUROC & AUPRC curves
        ├── calibration.png        calibration plots
        ├── cross_cohort.png       set A (within) vs set A→set B (cross) comparison
        ├── ablation.png           ablation study
        └── feature_importance.png top feature importances
```

> `results/predictions.csv` (per-stay model outputs, ~7.6 MB) and the entire `results_dev/`
> scratch directory are **excluded from version control** via `.gitignore`; the aggregate
> metrics needed to verify every reported number are all in the committed `*.csv` files above.

---

## Methods

Pipeline (all reproducible from the committed code):

1. **Download** — `download_data.py` fetches the public Challenge-2019 `.psv` release
   from the PhysioNet open S3 bucket with pagination, 32 parallel workers and resume.
2. **Preprocess** — `preprocess.py` extracts a 6-hour observation window ending at each
   chart-time and summarises the preceding trajectory per variable with
   last / mean / max / min / slope / missingness → **280 features per ICU stay**.
   Patient-level train/test splits prevent leakage. Stays already septic inside the
   window are excluded.
3. **Train** — `train.py` fits:
   - Baselines: a simplified SOFA-like clinical rule and logistic regression;
   - Main models: **XGBoost** and **LightGBM** on the 280 windowed features;
   - Ablations: vitals-only, limited tree capacity (leaves = 15 / 63), and 5× negative
     down-sampling, to isolate each design choice's contribution.
   - Random seed = **42** throughout.
4. **Evaluate** — `evaluate.py` reports AUROC, AUPRC (primary, given the very low
   positive prevalence), Brier score, sensitivity/specificity at the operating point,
   1,000-sample bootstrap 95% CIs, calibration curves, and **cross-centre validation**
   (train on set A, test on set B — the two hospital systems).

---

## Results (summary)

Primary metric is **AUPRC** (positive prevalence ≈ 6.9% in set A, ≈ 4.1% in set B).

| Model | Split | AUROC | AUPRC | Brier |
|---|---|---|---|---|
| Simplified clinical rule | within set A | 0.590 | 0.090 | 0.104 |
| Logistic regression | within set A | 0.737 | 0.181 | 0.194 |
| **XGBoost (all features)** | within set A | **0.752** | **0.198** | **0.061** |
| LightGBM (all features) | within set A | 0.747 | 0.181 | 0.063 |
| Simplified clinical rule | set A → set B | 0.572 | 0.057 | 0.083 |
| **XGBoost (all features)** | set A → set B | 0.617 | **0.082** | 0.039 |

Full per-experiment numbers (including ablation variants, sensitivity/specificity and
TP/FP/FN/TN breakdowns) are in `results/metrics.csv` (also copied to
`results/table_results.csv`). Bootstrap 95% CIs are in `results/bootstrap_ci.csv`.

**Key finding:** XGBoost more than doubles the in-group AUPRC of the clinical rule
(~2.2×), but its absolute sensitivity at a high-specificity operating point is very low
(9 of 415 in-group positives flagged), and performance falls markedly when applied to the
second hospital system (AUPRC 0.198 → 0.082) — a clear centre-shift / generalisation
warning that is discussed as the main limitation of the study.

---

## Reproduce

```bash
python -m venv .venv
pip install -r requirements.txt

# 1. download the public Challenge-2019 data (raw .psv files are NOT committed)
python src/download_data.py --out data/raw --limit-per-set 0 --workers 32

# 2. feature extraction
python src/preprocess.py --data data/raw --out data/processed

# 3. train + evaluate  (writes results/*.csv)
python src/train.py --input data/processed --out results
python src/evaluate.py --results results --figures results/figures

# 4. regenerate the report / slides from the results
python src/build_report.py      # -> AIDD8122_Final_Report_P2625533_ChenWan.docx/.pdf
python src/build_slides.py      # -> AIDD8122_Presentation_P2625533_ChenWan.pptx/.pdf
```

Use `--limit-per-set N` for a smaller reproducible subset during development.
Any subset used in the report is declared with its size and random seed (see
`results/run_info.json`).

---

## Compliance

- Raw patient files are downloaded locally and **excluded from version control**; only the
  downloader, analysis code and aggregate results are published.
- `results/predictions.csv` (~7.6 MB of per-stay outputs) and the `results_dev/` scratch
  directory are git-ignored; every reported figure remains verifiable from the committed
  `*.csv` artefacts.
- The dataset version, DOI and the required reference are cited in the report.
- No individual-level data, re-identification attempts, or redistribution of the source
  cohort are performed.

---

## Reference

> Reyna MA, Josef CS, Jeter R, et al. *Early Prediction of Sepsis From Clinical Data:
> The PhysioNet/Computing in Cardiology Challenge 2019.*
> Crit Care Med. 2020;48(2):210–217. doi:10.1097/CCM.0000000000004145
> (dataset DOI: 10.13026/v64v-d857)
