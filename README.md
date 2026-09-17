# Predicting Infections during Hospitalization among Postsurgical Patients

### A Multimodal Machine Learning and NLP Framework for Early Postoperative Infection Risk Prediction

**B.Sc. Final Research & Development Project — Digital Medical Technologies, HIT**  
**Academic Year:** 2025–2026

**Students:** Hodaya Yasayev Klenter · Tal Meillet  
**Academic Supervisors:** Dr. Ayelet Butman · Dr. Revital Marbel  
**Clinical / Industrial Supervisor:** Dr. Manor Shpriz  
**Clinical Site:** Assuta Ramat HaHayal Medical Center

---

## Overview

Hospital-acquired infections, particularly postoperative and surgical-site infections, remain an important clinical challenge. Early identification of patients at increased risk may support closer monitoring and more timely clinical attention.

This academic project develops and evaluates a **multimodal machine-learning framework** for predicting postoperative infection risk at the **end of the surgical procedure**. The framework combines:

- structured electronic health record (EHR) data;
- unstructured Hebrew clinical text;
- classical machine-learning and ensemble models;
- natural language processing (NLP);
- class-imbalance handling;
- feature selection;
- probability calibration;
- clinically informed decision-threshold optimisation; and
- explainable AI methods.

The project was designed as a **screening and clinical decision-support research framework**, not as an autonomous diagnostic system.

---

## Research Objective

The primary research question was:

> To what extent can a multimodal machine-learning framework, combining structured tabular EHR data with unstructured Hebrew clinical text, predict postoperative infection risk at the end of surgery compared with a structured-data-only model?

Secondary objectives included:

- identifying clinically relevant predictors of infection risk;
- evaluating the contribution of Hebrew clinical text;
- addressing extreme class imbalance;
- selecting an operating threshold that balances sensitivity with clinical alert burden;
- preventing temporal data leakage; and
- improving interpretability through explainable AI.

---

## Dataset

The retrospective dataset contained:

| Characteristic | Value |
|---|---:|
| Surgical events in the raw cohort | 35,925 |
| Infection-positive events | 141 |
| Raw prevalence | 0.39% |
| Restricted modelling cohort | 31,849 |
| Frozen test set | 9,555 |
| Positive cases in frozen test set | 42 |

The unit of analysis was the **surgical event**.

The data included demographic, clinical, operative and perioperative variables, together with Hebrew clinical narratives.

> **Important:** The original clinical dataset is not included in this repository. All patient-level data were analysed within the hospital's secure environment. Only code, documentation and non-identifiable aggregate outputs suitable for academic presentation are included.

---

## Methodology

### 1. Exploratory Data Analysis

The initial EDA examined:

- target distribution and class imbalance;
- missingness patterns;
- demographic and clinical distributions;
- categorical and continuous variables;
- temporal variables;
- correlations and potential redundancy; and
- characteristics of infection-positive cases.

Both raw-data EDA and post-cleaning EDA were performed.

### 2. Data Cleaning and Leakage Prevention

The cleaning pipeline included:

- cohort definition and exclusion rules;
- target construction;
- handling of missing and inconsistent values;
- semantic review of clinical variables;
- temporal eligibility checks; and
- exclusion of variables unavailable at the intended prediction point.

The prediction point was fixed at the **end of surgery** in order to reduce the risk that the model would learn from information documented only after clinical suspicion had already emerged.

### 3. Clinical Text Modelling

Three Hebrew clinical-text approaches were evaluated:

1. **TF-IDF + Logistic Regression**
2. **TF-IDF + XGBoost**
3. **Locally hosted Llama model** used in frozen-inference mode

The best-performing text representation was selected for integration into the tabular modelling pipeline.

### 4. Feature Engineering and Selection

The project included:

- engineered clinical and procedural features;
- medication-family features;
- text-derived risk features;
- multicollinearity assessment using VIF;
- Mutual Information ranking; and
- L1-regularised Logistic Regression for feature selection.

The final tabular model used **20 selected features**.

### 5. Model Development

The following structured-data classifiers were evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- CatBoost

Several class-imbalance strategies were also evaluated, including class weighting, SMOTE and synthetic-data approaches.

### 6. Evaluation

Because the positive class was extremely rare, **accuracy was not used as the primary model-selection metric**.

Evaluation focused on:

- PR-AUC / Average Precision
- ROC-AUC
- Recall
- Precision
- F1-score
- F2-score
- Balanced Accuracy
- Brier Score
- Confusion Matrix
- Number Needed to Screen (NNS)
- Alerts per 1,000 procedures

Threshold selection was performed separately from model fitting, with explicit consideration of clinical alert burden.

### 7. Calibration and Explainability

Probability calibration was evaluated using:

- no calibration;
- sigmoid / Platt scaling; and
- isotonic regression.

Model interpretation included:

- Logistic Regression coefficients;
- odds ratios;
- permutation importance; and
- SHAP-based explanations.

---

## Main Results

### Final Tabular Model

The final selected model was a **Logistic Regression classifier using 20 features**, calibrated with isotonic regression.

Performance on the frozen test set:

| Metric | Result |
|---|---:|
| Recall | **88.1%** |
| Precision | **4.7%** |
| PR-AUC | **0.183** |
| ROC-AUC | **0.943** |
| Brier Score | **0.0039** |
| Alerts per 1,000 procedures | **~83** |
| Number Needed to Screen | **21.4** |

The low precision should be interpreted in the context of the extremely low infection prevalence. The project therefore emphasised PR-AUC, recall, calibration and alert burden rather than accuracy alone.

### Hebrew Clinical Text

On the shared NLP evaluation split:

| Text Model | Average Precision |
|---|---:|
| TF-IDF + Logistic Regression | **0.519** |
| TF-IDF + XGBoost | **0.304** |
| Locally hosted Llama | **0.183** |

The results indicate that Hebrew clinical narratives contained meaningful predictive signal. In this dataset, the simpler TF-IDF + Logistic Regression representation outperformed the frozen LLM approach.

### Multimodal Comparison

A structured-only model was compared with a model incorporating the selected text-derived feature.

A small improvement was observed during cross-validation after incorporating the NLP-derived representation; however, this improvement was **not retained on the frozen test set**. Therefore, the study does not claim a definitive multimodal performance advantage.

---

## Repository Structure

```text
postsurgical-infection-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_EDA_positive_patients.ipynb
│   ├── 03_Cleaning.ipynb
│   ├── 04_eda_after_cleaning.ipynb
│   ├── 05_NLP_models.ipynb
│   ├── 06_Feature_Engineering.ipynb
│   ├── 07_PreModeling.ipynb
│   └── 08_Final_Models.ipynb
│
├── figures/
├── results/
└── docs/
```

### Notebook Pipeline

| Notebook | Purpose |
|---|---|
| `01_EDA.ipynb` | Initial exploratory data analysis |
| `02_EDA_positive_patients.ipynb` | EDA focused on infection-positive cases |
| `03_Cleaning.ipynb` | Data cleaning, cohort restriction and leakage control |
| `04_eda_after_cleaning.ipynb` | Exploratory analysis after cleaning |
| `05_NLP_models.ipynb` | Hebrew clinical-text model evaluation |
| `06_Feature_Engineering.ipynb` | Feature engineering and feature audit |
| `07_PreModeling.ipynb` | Data partitioning, preprocessing and feature selection |
| `08_Final_Models.ipynb` | Model comparison, calibration, threshold selection, final evaluation and explainability |

---

## Technologies

**Language**

- Python

**Data Analysis**

- pandas
- NumPy
- SciPy

**Machine Learning**

- scikit-learn
- XGBoost
- CatBoost
- imbalanced-learn

**NLP**

- TF-IDF
- Logistic Regression
- XGBoost
- locally hosted Llama inference

**Explainability and Statistics**

- SHAP
- statsmodels
- permutation importance
- VIF
- Mutual Information

**Visualisation**

- Matplotlib
- Seaborn

---

## Reproducibility

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The repository documents the analytical pipeline, but the original hospital dataset is intentionally excluded. Therefore, the notebooks cannot be reproduced end-to-end outside the authorised clinical environment without an appropriately structured dataset.

---

## Data Governance and Privacy

This project was conducted using retrospective, de-identified clinical records in a secure hospital computing environment.

To preserve privacy and comply with data-governance constraints:

- no raw patient-level dataset is included;
- no clinical free-text records are published;
- no patient-level identifiers or case-level prediction files are included;
- cached LLM outputs containing case-level information are excluded; and
- only non-identifiable aggregate tables and figures are intended for public release.

---

## Limitations

This study should be interpreted as an academic model-development and evaluation project rather than a clinically validated deployment study.

Key limitations include:

- only **141 positive infection cases** in the raw cohort;
- only **42 positive cases** in the frozen test partition;
- single-centre retrospective data;
- limited statistical power for small differences between models;
- no external validation;
- institution-specific documentation patterns;
- a compact 20-feature final representation;
- limited positive-class data for NLP evaluation; and
- the multimodal integration evaluated here represents only one possible method of combining structured and textual information.

Further validation on larger and external datasets is required before any clinical use.

---

## Intended Use

The developed framework is intended as a **research prototype for risk screening and clinical decision support**.

It is **not a diagnostic device**, does not replace clinical judgement, and should not be used for treatment decisions without prospective and external validation.

---

## Academic Context

This repository accompanies the final Research & Development Project submitted as part of the **B.Sc. in Digital Medical Technologies** at the **Holon Institute of Technology (HIT)**.

The work was developed in collaboration with **Assuta Ramat HaHayal Medical Center**.

For the complete academic methodology, literature review, results, discussion and limitations, see the final project report in the `docs/` directory.

---

## Authors

**Hodaya Yasayev Klenter**  
**Tal Meillet**

B.Sc. Digital Medical Technologies  
Holon Institute of Technology (HIT)

---

## Acknowledgements

We gratefully acknowledge:

- **Dr. Manor Shpriz** — Clinical Supervisor
- **Dr. Ayelet Butman** — Academic Supervisor
- **Dr. Revital Marbel** — Academic Supervisor
- the Infection Prevention and Control team at Assuta Ramat HaHayal Medical Center; and
- the hospital information-systems and infrastructure teams who supported the secure analytical environment.

---

## Disclaimer

This repository is provided for **academic and research purposes only**.  
The reported results were obtained on a retrospective single-centre dataset and should not be interpreted as evidence of clinical readiness or generalisability to other healthcare settings.
