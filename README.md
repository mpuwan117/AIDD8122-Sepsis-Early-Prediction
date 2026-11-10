# Early Prediction of Sepsis from ICU Time-Series

AIDD8122 Final Project (Advanced Topics in AI & Drug Discovery) — individual project.
Working title; to be finalised with the supervisor.

> 状态（2026-11-10 16:32）：**全量实验已跑完**，报告、PPT 与两份 PDF 已生成。
> 所有数字均由 `src/train.py` / `src/evaluate.py` 在真实数据上运行产生，见 `results/*.csv`；
> 报告与 PPT 由 `src/build_report.py` / `src/build_slides.py` 从这些结果文件动态生成，无硬编码数字。
>
> 结果速览：39,603 例可分析 ICU 住院、280 个窗口特征；组内 XGBoost AUPRC 0.198
> （简化临床规则 0.090，约 2.2 倍），跨中心 setA→setB 降至 0.082。
>
> **唯一待办**：把可访问的 GitHub 仓库链接填入报告末尾
> "Data and Code Availability" 小节的占位处，再提交 Canvas。

## Data source

**PhysioNet / Computing in Cardiology Challenge 2019, v1.0.0**
*Early Prediction of Sepsis from Clinical Data*
DOI: `10.13026/v64v-d857` · https://physionet.org/content/challenge-2019/1.0.0/

- 40,336 ICU patients in the public training release (set A: 20,336; set B: 20,000),
  drawn from two separate hospital systems.
- One pipe-separated `.psv` per patient; each row is one hour of ICU stay.
- 40 clinical variables (8 vital signs, 26 laboratory values, 6 demographics/administrative)
  plus the `SepsisLabel` outcome, shifted 6 hours earlier than clinical recognition.
- Task: predict sepsis 6 hours before clinical suspicion/SOFA-based onset, using
  current and past data only.
- The hidden scoring set is not public and is not used here.

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

## Repository layout

```
repo/
├── README.md                 this file
├── requirements.txt          Python dependencies
├── .gitignore                raw patient data is excluded
├── src/
│   ├── download_data.py      public S3 downloader (paginates + parallel fetch + resume)
│   ├── preprocess.py         windowed feature extraction (to be added)
│   ├── train.py              baselines + gradient boosting (to be added)
│   └── evaluate.py           metrics, calibration, plots (to be added)
├── notebooks/                exploratory analysis
├── results/
│   └── figures/              figures for the report
└── data/raw/                 downloaded .psv files — NOT committed
```

## Reproduce

```bash
python -m venv .venv
pip install -r requirements.txt
python src/download_data.py --out data/raw --limit-per-set 0 --workers 32
python src/preprocess.py --data data/raw --out data/processed
python src/train.py --input data/processed --out results
python src/evaluate.py --results results --figures results/figures
```

Use `--limit-per-set N` for a smaller reproducible subset during development.
Any subset used in the report must be declared with its size and random seed.

## Planned methods (subject to the supervisor's approval)

- Baselines: simplified SIRS/SOFA rule and logistic regression.
- Main models: XGBoost / LightGBM on windowed time-series features
  (last / mean / max / slope / missingness indicators).
- Improvement and comparative analysis: hyperparameter tuning, feature ablation
  (vital signs only vs all variables), class-imbalance handling, calibration,
  and cross-cohort validation (train on set A, test on set B).
- Metrics: AUROC, AUPRC (primary, given the very low positive prevalence),
  the Challenge utility function, sensitivity/specificity, calibration curves,
  and bootstrap confidence intervals. Patient-level splits to avoid leakage.

## Compliance

- Raw patient files are downloaded locally and excluded from version control;
  only the downloader and analysis code are published.
- The report cites the dataset version, DOI and the required reference.
- No individual-level data, re-identification attempts, or redistribution of the
  source cohort are performed.
