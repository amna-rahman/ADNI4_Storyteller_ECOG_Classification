# Participant-Level Detection of Mild Cognitive Impairment Using Temporally Matched ADNI4 Speech and Self-Reported Everyday Cognition (ECog-12 PT)

A reproducible machine learning pipeline for classifying **Cognitively Normal (CN)** vs. **Mild Cognitive Impairment (MCI)** participants in **ADNI4** by combining remote connected-speech features (**Storyteller**) with temporally matched **Everyday Cognition 12-Item Participant Self-Report (`ECOG12PT`)** items under strict participant-grouped cross-validation.

---

## 1. Research Problem and Objective

Remote digital screening for Mild Cognitive Impairment (MCI) could pair objective connected-speech markers with brief everyday-function questionnaires, yet evidence from closely timed, clinically coded assessments remains limited. This project evaluates whether a **61-feature reduced multimodal model**—combining **51 Storyteller speech features** with **10 ECog-12 PT items (`ECOGPT3`–`ECOGPT12`, excluding `ECOGPT1` and `ECOGPT2`)** collected on or within **30 days before** each speech assessment—improves participant-level CN vs. MCI discrimination over either modality alone.

Key methodological contributions implemented in this repository:
1. **Strict temporal alignment & leakage prevention:** Pairs each Storyteller assessment with same-participant `ECOG12PT` records dated on or within 30 days prior (with clinical diagnosis `DXSUM` dated on or before `ECOG12PT`), yielding one fixed primary cohort of **77 assessments from 71 participants (49 CN, 22 MCI)**.
2. **Participant-grouped repeated evaluation:** Evaluates Storyteller-only, ECog-12 PT (items 3–12), full fusion, and reduced fusion on identical 5-fold `StratifiedGroupKFold` splits repeated across **10 seeds** (`11, 21, 42, 52, 62, 72, 82, 92, 102, 112`), with nested 5×4 grouped CV to quantify item-selection optimism.
3. **Clinical quality & explainability:** Reports paired participant-level bootstrap ROC-AUC comparisons, calibration (Brier score, ECE), PR-AUC, operating-threshold / Decision-Curve Analysis, and held-out `shap.LinearExplainer` attributions plus raw-value directionality.
4. **Contextual all-pairs sensitivity extension:** Separately evaluates an unrestricted all-pairs cohort (877 Storyteller × ECog-12 pairs, 221 participants / 218 stable-label, CN vs. MCI/dementia) to demonstrate why temporal ordering and target definitions must be fixed before evaluation.

---

## 2. Repository Structure

```text
project-name/
│
├── README.md                                  # Project overview and full reproduction guide
├── requirements.txt                           # Python package dependencies
├── LICENSE                                    # Software license
├── AI_DISCLOSURE.md                           # AI-assisted workflow disclosure
├── .gitignore                                 # Excludes raw/processed ADNI data and caches
│
├── configs/
│   ├── baseline.yaml                          # Baseline & temporal sensitivity configurations (Studies 1–3, All-Pairs)
│   └── proposed_model.yaml                    # Proposed reduced-fusion & evaluation configurations (Studies 4–13)
│
├── data/
│   ├── README.md                              # Exact ADNI4 file list, dataset descriptions & download steps
│   ├── raw/                                   # Local only (gitignored): 5 raw ADNI4 CSVs from LONI IDA
│   └── processed/                             # Local only (gitignored): generated source & primary cohort CSVs
│
├── src/
│   ├── 01_preprocessing.ipynb                 # Stage 1: Raw audit, linkage, export of source datasets
│   ├── 02_utils.ipynb                         # Stage 2: Shared dataset/cohort loader & canonical model helpers
│   ├── 03_train_temporal_and_ablation.ipynb   # Stage 3: Studies 1–6 (temporal windows, primary cohort export, ablations, nested CV)
│   ├── 04_evaluate_models_and_interpretation.ipynb  # Stage 4: Studies 7–11 (final comparison, bootstrap, permutation, calibration/DCA, Table 1)
│   ├── 05_inference_shap_explainability.ipynb # Stage 5: Studies 12–13 (held-out SHAP & raw-value directionality)
│   └── 06_dataset4_all_pairs_extension.ipynb  # Stage 6: Isolated all-pairs sensitivity extension (Parallel Studies 1–10)
│
├── scripts/
│   ├── train.sh                               # End-to-end execution script for preprocessing & training
│   └── evaluate.sh                            # End-to-end execution script for evaluation & SHAP
│
├── results/                                   # Saved aggregate metric CSVs organized by study
│   ├── historical_prior_ecog_classification/
│   ├── temporal_window_sensitivity/
│   ├── common_cohort_temporal_comparison/
│   ├── ecog_item_ablation/
│   ├── ecog_ablation_stability/
│   ├── nested_feature_selection/
│   ├── final_model_comparison_uncertainty/
│   ├── feature_interpretation/
│   ├── feature_importance_verification/
│   ├── feature_correlation_redundancy/
│   ├── publication_calibration_and_dca/
│   ├── publication_demographics_table1/
│   ├── primary_cohort_shap_explainability/    # (or dataset3_shap_explainability/ in earlier runs)
│   └── dataset4_setup/
│
└── figures/                                   # Publication-ready 300-DPI figures
```

---

## 3. Hardware Requirements

* **Course Standard Hardware:** Tested and compatible with the course-provided **NVIDIA RTX A5000 (24 GB VRAM)** workstation/server environment.
* **Compute Profile:** Because all models use tabular `scikit-learn` regularized logistic regression (`solver="liblinear"`) and analytical `shap.LinearExplainer`, **no GPU acceleration is required**—the entire pipeline (Stages 1–6, including 10-seed nested CV, paired bootstrap, permutation importance, and held-out SHAP) runs deterministically on CPU in under 5–10 minutes with `< 4 GB` RAM.

---

## 4. Software and Dependency Requirements

* **Python:** `3.10+` (developed and validated on `Python 3.13`)
* **Core Libraries:**
  * `numpy`
  * `pandas`
  * `scikit-learn`
  * `scipy`
  * `matplotlib`
  * `shap`
  * `pyyaml`
  * `jupyter` / `ipykernel` / `nbconvert`

Exact package constraints are pinned in `requirements.txt`.

---

## 5. Installation Instructions

```bash
git clone https://github.com/<username>/<project-name>.git
cd <project-name>

# Create and activate a virtual environment
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. Dataset Information

Data were obtained from the **Alzheimer's Disease Neuroimaging Initiative (ADNI4)** via the [LONI Image and Data Archive (IDA)](https://ida.loni.usc.edu/) ([ADNI Data Access Portal](https://adni.loni.usc.edu/data-samples/adni-data/)).

**Redistribution Notice:** Per the [ADNI Data Use Agreement](https://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Data_Use_Agreement.pdf), raw and participant-level ADNI CSV files **cannot be committed to a public GitHub repository**. Authorized researchers must download the following five ADNI4 tables from LONI IDA and place them in `data/raw/`:

1. `All_Subjects_DXSUM_06Nov2026.csv` — Clinical diagnostic summary (`DXSUM`)
2. `All_Subjects_ECOG12PT_06Nov2026.csv` — Everyday Cognition 12-Item Participant Self-Report (`ECOG12PT`)
3. `All_Subjects_PTDEMOG_06Nov2026.csv` — Participant demographics (`PTDEMOG`)
4. `DATADIC_06Nov2026.csv` — Official ADNI data dictionary (`DATADIC`)
5. `RMT_STORYTELLER_06Nov2026.csv` — ADNI4 Remote Digital Storyteller speech features (`STORYTELLER`)

See [data/README.md](https://www.google.com/search?q=data/README.md) for detailed descriptions of each dataset, download steps, and expected folder layout.

---

## 7. Data Preprocessing Instructions

Once the five raw ADNI4 CSV files are placed in `data/raw/`, run Stage 1 to audit data quality, link diagnoses and visits, and export the source CSVs to `data/processed/`:

```bash
jupyter nbconvert --to notebook --execute --inplace src/01_preprocessing.ipynb
```

This creates:
* `data/processed/dataset1_storyteller.csv` — 1,430 Storyteller assessments (490 participants; 1,073 CN, 357 MCI; 51 retained speech predictors)
* `data/processed/dataset2_ecog.csv` — 501 `ECOG12PT` questionnaire records (397 participants; 273 CN, 228 MCI; 12 items with code `9` recoded to `NaN`)
* `data/processed/dataset4_multimodal_cn_vs_noncn.csv` — 877 unrestricted Storyteller × ECog-12 pairs (221 participants; 218 stable-label) for the contextual sensitivity extension

*(Optional)* Run `src/02_utils.ipynb` to verify paths, feature counts, and shared helper definitions.

---

## 8. Training Instructions

All model hyperparameters, feature exclusions, temporal windows, and random seeds are documented in [configs/baseline.yaml](https://www.google.com/search?q=configs/baseline.yaml) and [configs/proposed\_model.yaml](https://www.google.com/search?q=configs/proposed_model.yaml):
* **Pipeline:** `SimpleImputer(strategy="median", add_indicator=True)` → `StandardScaler()` → `LogisticRegression(penalty="l2", C=1.0, max_iter=5000, class_weight="balanced", solver="liblinear", random_state=42)`
* **Cross-Validation:** 5-fold `StratifiedGroupKFold` grouped by `PTID`, repeated across 10 seeds (`[11, 21, 42, 52, 62, 72, 82, 92, 102, 112]`), decision threshold `0.50`.

Run Stage 3 (Studies 1–6: historical baseline, temporal window sensitivity, fixed 30-day cohort comparison, ECog-12 item ablation, ablation stability, and 5×4 nested grouped feature selection):

```bash
jupyter nbconvert --to notebook --execute --inplace src/03_train_temporal_and_ablation.ipynb
```

During **Study 3**, this notebook exports the single canonical primary cohort used by all downstream analyses:
* `data/processed/primary_multimodal_30day_cohort.csv` (77 assessments, 71 participants: 49 CN, 22 MCI)

---

## 9. Evaluation and Explainability Instructions

Run Stage 4 (Studies 7–11: primary model comparison, paired bootstrap uncertainty, grouped permutation importance, coefficient stability, redundancy audit, calibration/PR-AUC/Decision-Curve Analysis, and Table 1 demographics):

```bash
jupyter nbconvert --to notebook --execute --inplace src/04_evaluate_models_and_interpretation.ipynb
```

Run Stage 5 (Studies 12–13: held-out `shap.LinearExplainer` attributions, modality shares, raw-value-to-SHAP Spearman directionality, and 300-DPI publication figures):

```bash
jupyter nbconvert --to notebook --execute --inplace src/05_inference_shap_explainability.ipynb
```

Run Stage 6 (optional contextual sensitivity extension on the unrestricted all-pairs cohort):

```bash
jupyter nbconvert --to notebook --execute --inplace src/06_dataset4_all_pairs_extension.ipynb
```

All notebooks locate the repository root dynamically using `pathlib.Path.cwd()` with no hard-coded machine-specific paths.

---

## 10. Expected and Reported Results

### A. Primary 30-Day Temporally Matched Cohort (77 Assessments, 71 Participants: 49 CN, 22 MCI)
*Participant-level mean ± SD across 10 repeated 5-fold `StratifiedGroupKFold` seeds on identical participant folds (source: `results/final_model_comparison_uncertainty/results/final_model_comparison_uncertainty_repeated_seed_stability.csv` and `results/nested_feature_selection/nested_feature_selection_participant_summary.csv`):*

| Model Configuration | Features | Accuracy | Balanced Accuracy | MCI Recall | Specificity | MCI F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Storyteller only | 51 | 0.621 ± 0.052 | 0.584 ± 0.047 | 0.486 ± 0.057 | 0.682 ± 0.070 | 0.444 ± 0.051 | 0.680 ± 0.034 |
| ECog-12 PT items 3–12 only | 10 | 0.739 ± 0.031 | 0.697 ± 0.033 | 0.586 ± 0.050 | 0.808 ± 0.035 | 0.583 ± 0.045 | 0.734 ± 0.026 |
| Full fusion (all 12 ECog-12 PT items) | 63 | 0.759 ± 0.041 | 0.735 ± 0.050 | 0.673 ± 0.077 | 0.798 ± 0.030 | 0.633 ± 0.065 | 0.823 ± 0.032 |
| **Reduced fusion (excl. `ECOGPT1`, `ECOGPT2`)** | **61** | **0.789 ± 0.036** | **0.767 ± 0.039** | **0.709 ± 0.057** | **0.824 ± 0.039** | **0.676 ± 0.053** | **0.857 ± 0.026** |
| Nested-selected fusion (5×4 outer/inner CV) | 61–63 | 0.789 ± 0.041 | 0.754 ± 0.046 | 0.664 ± 0.068 | — | 0.661 ± 0.063 | 0.841 ± 0.030 |

### B. Top Held-Out SHAP Features — Reduced Fusion
*Source: `results/primary_cohort_shap_explainability/reduced_fusion_without_ECOGPT1_ECOGPT2/feature_importance_mean_abs_shap.csv` and `interpretation_modality_contributions.csv` (Modality share: Storyteller 75.7%, ECog-12 PT 24.3%):*

| Rank | Feature | Modality | Mean \|SHAP\| (log-odds) |
|---:|---|---|---:|
| 1 | `ECOGPT8` (*Planning a trip, vacation, or outing*) | ECog-12 PT | 0.870 |
| 2 | `STORY2IMM_MEAN_PAUSE_LENGTH` | Storyteller | 0.762 |
| 3 | `ECOGPT9` (*Keeping living and work space organized*) | ECog-12 PT | 0.604 |
| 4 | `STORY1IMM_IDEA_DENSITY` | Storyteller | 0.514 |
| 5 | `STORY2IMM_NUM_PAUSES` | Storyteller | 0.468 |

### C. Contextual All-Pairs Sensitivity Extension (877 Pairs; 218 Stable-Label Participants, CN vs. MCI/Dementia)
*Source: `results/final_model_comparison_uncertainty/dataset4/dataset4_final_model_comparison_uncertainty_stability.csv`:*
* **Reduced fusion (stable-label participant level):** ROC-AUC **0.732 ± 0.023**, Balanced Accuracy **0.683 ± 0.016**, Recall **0.626 ± 0.033**, Specificity **0.740 ± 0.015**.
* **Full fusion (stable-label participant level):** ROC-AUC **0.717 ± 0.025**, Balanced Accuracy **0.672 ± 0.022**.

---

## 11. Pretrained Models and Checkpoints

No external pretrained neural networks or Hugging Face checkpoints are used in this project. Every classifier is a deterministic `scikit-learn` L2-regularized `LogisticRegression` pipeline trained from scratch within each grouped cross-validation fold using fixed random seeds.

---

## 12. Citation, Attribution, and ADNI Acknowledgement

### Code & Library Attribution
* **Student-written code:** All ADNI4 cohort linkage, temporal matching, grouped CV evaluation, ablation, calibration/DCA, and SHAP analysis notebooks in `src/` and configuration files in `configs/`.
* **Reused open-source libraries:** `scikit-learn` (Pedregosa et al., 2011), `SHAP` (Lundberg & Lee, 2017), `pandas`, `NumPy`, `SciPy`, and `Matplotlib`.

### Mandatory ADNI Acknowledgement
> Data used in preparation of this project were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (`adni.loni.usc.edu`). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in analysis or writing of this report. A complete listing of ADNI investigators can be found at: `http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf`.
>
> Data collection and sharing for the Alzheimer's Disease Neuroimaging Initiative (ADNI) is funded by the National Institute on Aging (National Institutes of Health Grant U19 AG024904). The grantee organization is the Northern California Institute for Research and Education, and the study is coordinated by the Alzheimer's Therapeutic Research Institute at the University of Southern California. ADNI data are disseminated by the Laboratory for Neuro Imaging at the University of Southern California.

---

## 13. AI-Use Disclosure

AI assistance (ChatGPT and Gemini) was used to help modularize the research notebooks into the repository structure, draft configuration templates and documentation, and edit manuscript prose. All data handling, experimental execution, metric verification, and scientific interpretation were reviewed and verified by the student author(s). Full details are recorded in [AI\_DISCLOSURE.md](https://www.google.com/search?q=AI_DISCLOSURE.md).
