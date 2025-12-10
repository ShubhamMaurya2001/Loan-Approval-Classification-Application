# 🏦 Loan Approval Prediction Application

A production-ready machine learning application that predicts loan approval using XGBoost with interactive Streamlit UI and comprehensive MLflow model tracking.

## 📄 Overview

This project combines:
- **XGBoost Classifier** for accurate loan approval prediction (94%+ accuracy)
- **Streamlit Web Interface** for real-time predictions with interactive features
- **MLflow Integration** for model tracking, metrics logging, and performance visualization
- **Kaggle Dataset** - Enhanced with financial risk variables and balanced using SMOTENC

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit App
```bash
streamlit run LoanApp.py
```
Open: **http://localhost:8501**

### 3. View Model Metrics
```bash
python MLflow.py
mlflow ui
```
Open: **http://localhost:5000**

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~94% |
| Precision | High |
| Recall | Good |
| F1-Score | ~0.94 |
| ROC-AUC | Strong |

---

## 📁 Dataset Metadata

* **Records:** 45,000 samples
* **Features:** 13 columns *(after removing data-leakage variable)*
* **Target Variable:** `loan_status` (binary classification: `1 = approved`, `0 = rejected`)

| Column Name                  | Description                                                    | Type        |
| ---------------------------- | -------------------------------------------------------------- | ----------- |
| `person_age`                 | Age of the person                                              | Float       |
| `person_gender`              | Gender of the person                                           | Categorical |
| `person_education`           | Highest education level                                        | Categorical |
| `person_income`              | Annual income in USD                                           | Float       |
| `person_emp_exp`             | Years of employment experience                                 | Integer     |
| `person_home_ownership`      | Home ownership status (e.g., rent, own, mortgage)              | Categorical |
| `loan_amnt`                  | Loan amount requested                                          | Float       |
| `loan_intent`                | Purpose of the loan (e.g., personal, education, medical, etc.) | Categorical |
| `loan_int_rate`              | Interest rate on the loan                                      | Float       |
| `loan_percent_income`        | Loan amount as a percentage of the person’s income             | Float       |
| `cb_person_cred_hist_length` | Length of credit history in years                              | Float       |
| `credit_score`               | Credit score of the person                                     | Integer     |
| `loan_status`                | Loan approval status (`1 = approved`, `0 = rejected`)          | Integer     |

> ⚠️ **Note:**
> The column `previous_loan_defaults_on_file` was **removed** due to **data leakage**, as it directly correlated with the loan approval decision, leading to artificially inflated performance.

---

## 📊 Notebook Analysis

The accompanying Jupyter Notebook demonstrates:

* **Exploratory Data Analysis (EDA)** using **Plotly** for rich, interactive visualizations
* **Data preprocessing** and **balancing using SMOTE** to handle class imbalance
* **Feature engineering** and encoding of categorical variables
* **Model training** using **XGBoost**, optimized for high predictive accuracy
* **Model evaluation** using multiple metrics

---

## 🧠 Model Performance — XGBoost + SMOTE

| Metric                    | Value |
| ------------------------- | ----- |
| **Accuracy**              | 0.92  |
| **Precision (Class 0)**   | 0.92  |
| **Recall (Class 0)**      | 0.98  |
| **F1-score (Class 0)**    | 0.95  |
| **Precision (Class 1)**   | 0.88  |
| **Recall (Class 1)**      | 0.67  |
| **F1-score (Class 1)**    | 0.76  |
| **Macro Avg F1-score**    | 0.85  |
| **Weighted Avg F1-score** | 0.91  |

✅ **Final Accuracy:** **92%** using **XGBoost** on **SMOTE-balanced data**
📉 **Leakage Check:** Dataset verified after removing `previous_loan_defaults_on_file` to ensure fair model evaluation.

---

## ⚙️ MLflow Integration Highlights

* Automatically logs:

  * Model parameters and hyperparameters
  * Evaluation metrics (accuracy, precision, recall, F1-score)
  * Confusion matrix and ROC curves
  * Model artifacts and serialized `.pkl` files
* Enables:

  * Comparing multiple model runs
  * Tracking tuning experiments (learning rate, depth, estimators, etc.)
  * Exporting best-performing model for deployment

🧩 **Example:**
`mlflow.xgboost.autolog()` used to track experiments seamlessly during model training.

---
## 🏁 Summary

This project demonstrates a **realistic financial risk modeling workflow**, from **data preprocessing and visualization** to **model optimization and evaluation**, achieving **92% F1-score** with **XGBoost** on a **SMOTE-balanced**, **leakage-free dataset**.


---

**Live demo (deployed):** https://loan-approval-classification.streamlit.app/

<p align="center">
  <img src="https://raw.githubusercontent.com/ShubhamMaurya2001/Loan-Approval-Classification-Application/main/Screenshots/Screenshot%202025-12-10%20114701.png" alt="screenshot1" width="900" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ShubhamMaurya2001/Loan-Approval-Classification-Application/main/Screenshots/Screenshot%202025-12-10%20114728.png" alt="screenshot2" width="900" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ShubhamMaurya2001/Loan-Approval-Classification-Application/main/Screenshots/Screenshot%202025-12-10%20115255.png" alt="screenshot3" width="900" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ShubhamMaurya2001/Loan-Approval-Classification-Application/main/Screenshots/Screenshot%202025-12-10%20115445.png" alt="screenshot4" width="900" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ShubhamMaurya2001/Loan-Approval-Classification-Application/main/Screenshots/Screenshot%202025-12-10%20115507.png" alt="screenshot5" width="900" />
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ShubhamMaurya2001/Loan-Approval-Classification-Application/main/Screenshots/Screenshot%202025-12-10%20115644.png" alt="screenshot6" width="900" />
</p>
