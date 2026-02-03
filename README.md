# 💰 EMIPredict AI – Intelligent Financial Risk Assessment Platform

## 📌 Project Overview

**EMIPredict AI** is an end-to-end **FinTech Machine Learning platform** designed to assess financial risk and predict EMI affordability. The system combines **classification and regression models**, **MLflow experiment tracking**, and a **Streamlit Cloud–deployed web application** to enable real-time, data-driven loan decision-making.

The platform helps banks, fintech companies, and loan officers:

* Determine **EMI eligibility** (Eligible / High Risk / Not Eligible)
* Estimate the **maximum safe monthly EMI** a customer can afford
* Standardize risk assessment and reduce manual underwriting effort

---

## 🎯 Problem Statement

Many individuals struggle to repay EMIs due to poor financial planning and inadequate risk evaluation. Traditional underwriting processes are time-consuming, subjective, and inconsistent.

**EMIPredict AI** solves this problem by leveraging machine learning on large-scale financial data to provide:

* Objective, data-driven eligibility decisions
* Accurate EMI affordability estimation
* Transparent and repeatable risk assessment

---

## 🧠 Solution Highlights

* 🔁 **Dual ML Approach**: Classification + Regression
* 📊 **Large-scale Dataset**: 400,000 financial records
* 🧮 **Advanced Feature Engineering** using financial ratios
* 🧪 **MLflow Integration** for experiment tracking & model registry
* 🌐 **Streamlit Cloud Deployment** for production-ready access
* 🗂 **Modular, multi-page web application**

---

## 🏦 Business Use Cases

### Financial Institutions

* Automate loan approval and reduce underwriting time by **~80%**
* Implement risk-based pricing strategies
* Real-time EMI eligibility checks for walk-in customers

### FinTech Companies

* Instant EMI eligibility for digital lending apps
* Pre-qualification services for customers
* Automated risk scoring pipelines

### Banks & Credit Agencies

* Data-driven EMI recommendations
* Portfolio risk management
* Transparent and auditable decision processes

---

## 🗂 Dataset Description

* **Total Records**: 400,000
* **Input Features**: 22
* **Target Variables**: 2
* **EMI Scenarios**: 5

### EMI Scenarios

| Scenario                | Records | Amount Range (₹) | Tenure (Months) |
| ----------------------- | ------- | ---------------- | --------------- |
| E-commerce Shopping EMI | 80,000  | 10K – 200K       | 3 – 24          |
| Home Appliances EMI     | 80,000  | 20K – 300K       | 6 – 36          |
| Vehicle EMI             | 80,000  | 80K – 15L        | 12 – 84         |
| Personal Loan EMI       | 80,000  | 50K – 10L        | 12 – 60         |
| Education EMI           | 80,000  | 50K – 5L         | 6 – 48          |

---

## 🔢 Target Variables

### 1️⃣ EMI Eligibility (Classification)

* **Eligible** – Low risk, EMI within comfortable affordability
* **High Risk** – Marginal affordability, higher interest recommended
* **Not Eligible** – High financial stress, loan not advised

### 2️⃣ Maximum Monthly EMI (Regression)

* Continuous value: **₹500 – ₹50,000**
* Represents safe EMI based on financial capacity

---

## 🧮 Feature Engineering

Key derived features include:

* Debt-to-Income Ratio
* Expense-to-Income Ratio
* Credit Utilization Ratio
* Affordability Ratio
* Employment Stability Score
* Financial Stress Index

These ratios improve model robustness and reduce income-level bias.

---

## 🤖 Machine Learning Models

### Classification Models

* Logistic Regression (Baseline & Interpretability)
* Random Forest Classifier
* XGBoost Classifier ✅ *(Selected for deployment)*

**Evaluation Metrics**:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

### Regression Models

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor ✅ *(Selected for deployment)*

**Evaluation Metrics**:

* RMSE
* MAE
* R² Score

---

## 🧪 MLflow Integration

* Experiment tracking for all trained models
* Logging of:

  * Hyperparameters
  * Evaluation metrics
  * Model artifacts
* Model comparison via MLflow dashboard
* Best models registered in **MLflow Model Registry**

---

## 🌐 Streamlit Application

### Application Pages

* 🏠 **Home** – Project overview & business context
* 💳 **EMI Eligibility Predictor** – Classification inference
* 🏦 **EMI Amount Predictor** – Regression inference
* 📊 **EDA** – Automated exploratory data analysis

### Key Features

* Real-time predictions
* User-friendly financial inputs
* Feature-aligned inference pipeline
* Professional FinTech UI

---

## ☁️ Deployment

* Platform: **Streamlit Cloud**
* Version Control: **GitHub**
* Automated deployment from repository
* Responsive UI for desktop and mobile

---

## 📁 Project Structure

```
EMIPredict-AI/
│
├── app.py
├── pages/
│   ├── Classifier.py
│   ├── Regressor.py
│   ├── EDA.py
│
├── models/
│   ├── xgb_classifier_model.joblib
│   ├── xgb_regressor_model.joblib
│   ├── encoders & transformers
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│
├── requirements.txt
├── README.md
```

---

## 📈 Results

* ✅ Classification Accuracy: **> 90%**
* ✅ Regression RMSE: **< ₹2000**
* ✅ Stable real-time inference on Streamlit Cloud

---

## 🚀 Business Impact

* ⏱ 80% reduction in manual underwriting time
* 📊 Standardized, data-driven risk assessment
* 🔍 Improved loan decision transparency
* 📈 Scalable for high-volume loan applications

---

## 🧰 Tech Stack

* **Language**: Python
* **ML**: Scikit-learn, XGBoost
* **Tracking**: MLflow (DAGsHub compatible)
* **App**: Streamlit
* **Deployment**: Streamlit Cloud
* **Visualization**: Matplotlib, Seaborn, Pandas Profiling

---

## 👤 Author

**Tanmoy Pal**
MBA (Business Analytics) | B.Tech (Electrical Engineering)
FinTech | Risk Analytics | Machine Learning

---

## 📜 License

This project is intended for **educational and demonstration purposes**.
