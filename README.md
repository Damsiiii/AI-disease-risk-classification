**🏥 SmartCare Hospital — Disease Risk Classification**
**CCS3440 Artificial Intelligence Coursework | SLTC | Option C**

An end-to-end, leakage-free Machine Learning pipeline that classifies hospital patients into Low, Medium, or High disease risk levels using clinical, physiological, and hospital-operations data from SmartCare Hospital — with Logistic Regression as the top-performing model, full SHAP explainability, and a deployed Streamlit prototype.

**📋 Table of Contents**
- Problem Statement
- Dataset
- Leakage-Free Pipeline Architecture
- Feature Engineering & Selection
- Models & Benchmarking
- Best Model Justification
- Explainable AI (SHAP)
- Bonus Work
- AI Prototype (Task 08)
- Project Structure
- Getting Started
- Team

**🎯 Problem Statement**
Option C — Multi-Class Disease Risk Classification

Early identification of a patient's disease risk level enables timely clinical intervention, better inpatient bed allocation, and more effective preventive outpatient care.

| Item | Specification |
|---|---|
| Target Variable | `disease_risk_level` |
| Problem Type | Multi-Class Classification (3 classes) |
| Classes & Support | Low: 13.1% (N=131) · Medium: 46.9% (N=469) · High: 40.0% (N=400) |
| Held-Out Test Support | Low: N=26 · Medium: N=94 · High: N=80 (Total N=200, 20% stratified test set) |

**📊 Dataset**
| Property | Value |
|---|---|
| File | `smartcare_ai_dataset_1000.csv` |
| Records | 1,000 patient admissions/visits |
| Raw Columns | 33 (18 selected as candidate input features after excluding identifiers, financial fields, and other coursework-option targets) |

**Feature Categories**
- Demographics: Age, Gender, Blood Group
- Physiological Vitals: Systolic BP, Diastolic BP, Blood Sugar (mg/dL), Cholesterol (mg/dL), BMI
- Hospital Operations: Department, Diagnosis, Admission Status, Room Type, Length of Stay, Previous Appointments/Admissions
- Utilization: Lab Tests Count, Treatments Count

**Data Quality**
- 0 duplicate rows, 0 duplicate `record_id`/`patient_id`
- 906 missing values in `room_type` — confirmed structural (only non-admitted patients lack a room), filled with `'Not Admitted'` rather than statistically imputed
- 31 statistical outliers across 6 clinical columns capped via IQR winsorizing (no rows dropped)

**🔄 Leakage-Free Pipeline Architecture**
- **Split-First Protocol**: Stratified 80/20 train/test split (Train N=800, Test N=200) performed *before* scaling and feature selection.
- **Train-Only Fitting**: Scaling and ANOVA F-score feature selection are fit strictly on `X_train` and applied via `.transform()` to `X_test`.
- **Leakage Removal**: `record_id`, `patient_id`, `no_show`, `readmitted_30_days`, all financial/billing fields, and appointment/payment administrative fields were explicitly dropped as they are either identifiers, other-option targets, or outcomes of treatment rather than causes of risk.

```
Raw Data (N=1000) ──> Clean + Engineer Features ──> Stratified 80/20 Split ──┬─> Train (N=800) ──> Fit (Scaler + SelectKBest) ──> Train Model
                                                                              └─> Test  (N=200) ──> Transform with Fitted Pipeline ──> Held-Out Eval
```

**🏷 Feature Engineering & Selection**
Five new clinically-motivated features were engineered:
- `pulse_pressure` — systolic minus diastolic BP, a recognized clinical indicator
- `bmi_category` — clinical BMI bands (Underweight/Normal/Overweight/Obese)
- `age_group` — age bands (Child/Adult/Middle-aged/Senior), since risk shifts non-linearly with age
- `chronic_diagnosis_flag` — flags Diabetes/Hypertension as chronic-risk conditions
- `care_intensity` — combines lab tests + treatments into one utilization signal

One-hot encoding expanded the feature space to 48 columns; **ANOVA F-test** feature selection (K=15) was used to identify the most predictive features for modeling, cross-checked against Random Forest feature importances. Age, blood sugar, and cholesterol ranked highest by both methods.

**🤖 Models & Benchmarking**
Three models were tuned via **GridSearchCV with 5-fold cross-validation**, optimizing macro-F1 to account for class imbalance, then evaluated on the identical held-out test set (N=200):

| Rank | Model | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 | ROC-AUC (macro, OvR) |
|---|---|---|---|---|---|---|
| 🥇 | Logistic Regression | 0.940 | 0.955 | 0.936 | **0.945** | 0.994 |
| 🥈 | SVM (Linear) | 0.940 | 0.957 | 0.909 | 0.929 | 0.996 |
| 🥉 | Random Forest | 0.765 | 0.752 | 0.758 | 0.755 | 0.895 |

**Per-class performance (Logistic Regression):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | 1.00 | 0.92 | 0.96 | 26 |
| Medium | 0.93 | 0.95 | 0.94 | 94 |
| High | 0.94 | 0.94 | 0.94 | 80 |

**🏆 Best Model Justification**
**Logistic Regression** was selected as the final model. It achieves the highest macro-F1 (0.945) and the strongest recall on the minority Low-risk class (92%), which is the most clinically important error to minimize in a risk-screening context — failing to flag a genuinely at-risk patient is costlier than a false alarm. This result is consistent with the EDA finding that clinical features (age, BP, cholesterol, BMI) rise near-monotonically from Low to High risk, favoring a model with a linear decision boundary over a depth-constrained Random Forest.

🧠 Explainable AI (SHAP)
SHAP analysis was run on the test set (200 samples, 48 features):

- **Global Feature Attributions (mean |SHAP value|):** blood sugar (3.95), cholesterol (3.64), BMI (3.34), age (3.27), and previous admissions (2.37) are the top clinical drivers of risk predictions across all three classes.
- **Per-Class Summary Plots:** generated separately for Low, Medium, and High risk to show how each feature pushes predictions toward or away from a given class.
- **Local Explanation:** individual patient-level SHAP force/waterfall plots explain specific prediction decisions (e.g., Patient 0 — Actual: Medium, Predicted: Medium).

**🎁 Bonus Work**
Beyond the core task requirements, the team completed:

| Area | Result |
|---|---|
| **Deep Learning (MLP)** | Accuracy 0.885, Macro-F1 0.858, ROC-AUC 0.971 — strong but below Logistic Regression, especially on Low-risk recall (0.65) |
| **Hyperparameter Optimization** | Wider RandomizedSearchCV search improved test Macro-F1 for Logistic Regression (0.945 → 0.956) and Random Forest (0.755 → 0.775); SVM was essentially unchanged (0.929 → 0.929) |
| **Ensemble Learning (Soft Voting)** | Accuracy 0.935, Macro-F1 0.931 — competitive with single best model but did not surpass tuned Logistic Regression |
| **Advanced Explainable AI** | Permutation importance (confirming blood sugar, cholesterol, age, BMI as top predictors) and Partial Dependence Plots showing each feature's marginal effect on predicted risk |
| **Multiple Prediction Tasks** | Extended the pipeline to predict `no_show` (Option A) as a secondary task — Accuracy 0.545, confirming this is a genuinely harder, low-signal problem rather than a leakage artifact (no single feature reached AUC > 0.97) |

**Model comparison including bonus models:**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Logistic Regression | 0.940 | 0.945 |
| Ensemble (Soft Voting) | 0.935 | 0.931 |
| SVM | 0.940 | 0.929 |
| Deep Learning (MLP) | 0.885 | 0.858 |
| Random Forest | 0.765 | 0.755 |

**📦 AI Prototype (Task 08)**
A **Streamlit clinical decision-support app** (`app.py`) was built on top of the final Logistic Regression pipeline and deployed via tunnel for live testing. Example predictions on synthetic patients:

| Scenario | Predicted Risk | Confidence |
|---|---|---|
| Low-risk patient | Low | 100% |
| Medium-risk patient | Medium | 100% |
| High-risk patient | High | 100% |

```bash
# Launch the Streamlit prototype
streamlit run app.py
```

**📁 Project Structure**
```
SmartCare-Hospital/
├── README.md                          # Project overview & documentation
├── requirements.txt                   # Environment dependencies
├── data/
│   └── smartcare_ai_dataset_1000.csv  # Benchmark dataset (1,000 records)
├── notebook/
│   └── SmartCare_Hospital.ipynb       # Fully executed coursework notebook (Tasks 01–09 + Bonus)
├── models/
│   ├── logistic_regression_model.pkl  # Final selected model
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   ├── mlp_model.pkl                  # Bonus: deep learning model
│   └── scaler.pkl                     # Fitted feature scaler
├── app/
│   └── app.py                         # Streamlit clinical decision support interface
└── reports/
    ├── model_comparison_table.csv
    ├── shap_summary_plots/            # Per-class SHAP summary plots
    ├── shap_local_explanation.png     # Patient-level SHAP waterfall
    └── hyperparameter_optimization_comparison.csv
```

**🚀 Getting Started**
```bash
git clone https://github.com/Damsiiiii/SmartCare-Hospital.git
cd SmartCare-Hospital
pip install -r requirements.txt
streamlit run app/app.py
```

**👥 Team — Option C, CCS3440 Artificial Intelligence**

| Member | Contributions |
|---|---|
| **Amintha Jayasooriya** | Task 02 (Dataset Understanding), Task 03 (Data Preprocessing & Feature Engineering) · Bonus: Deep Learning, Ensemble Learning |
| **Damsara Dissanayaka** | Task 05 (Machine Learning Model Development), Task 06 (Model Evaluation) · Bonus: Hyperparameter Optimization, Advanced Explainable AI Techniques |
| **Tharanya Pushparaj** | Task 01 (Problem Definition & Literature Review), Task 04 (Exploratory Data Analysis), Task 09 (Technical Report) · Bonus: Multiple Prediction Tasks |
| **Thamindu Kavinda** | Task 07 (Explainable AI Analysis), Task 08 (AI Prototype Development) |

⚠️ **Academic Disclaimer**: This machine learning system was developed for academic evaluation as part of the CCS3440 Artificial Intelligence module. It is intended for clinical decision support evaluation and should not be used as an autonomous medical diagnostic tool.
