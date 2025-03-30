# 💳 Credit Card Fraud Detection with AutoGluon

## 📌 Project Overview

This project leverages **AutoML**, specifically [AutoGluon](https://www.autogluon.ai/), to detect fraudulent credit card transactions in a highly imbalanced dataset. The goal is to build an efficient, cost-effective, and accurate classification model that maximizes recall (i.e., captures all fraudulent transactions) while minimizing false positives to avoid unnecessary operational costs.

## 📊 Dataset

**Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)  
The dataset contains anonymized features derived from PCA, with:

- 284,807 total transactions
- 492 labeled as fraud (~0.17%)
- 30 feature columns + 1 target column (`Class`)

### Key Features:
- `Time`: Seconds elapsed between transactions.
- `Amount`: Transaction amount.
- `V1` to `V28`: Principal components from PCA.
- `Class`: `0 = Legitimate`, `1 = Fraudulent`

## 🤖 AutoML Pipeline with AutoGluon

- **Library Used**: AutoGluon `TabularPredictor`
- **Models Tested**: LightGBM, XGBoost, Random Forest, etc.
- **Metric for Optimization**: **Recall** (to ensure no fraud is missed)
- **Final Threshold**: 0.3 (vs default 0.5) for best trade-off

## 📈 Evaluation

- **Average Precision**: `0.9997`
- **Recall**: `1.00` (No fraud missed)
- **Precision**: `0.9880`
- **F1-Score**: `0.9939`
- **False Positives**: Only 6

Using the **Precision-Recall Curve**, a lower classification threshold was selected to achieve a perfect recall while keeping precision high and false positives low.

## 💡 Business Impact

- **Operational Cost Reduction**: Minimized manual review from false positives
- **Risk Mitigation**: Zero missed fraud cases
- **Time & Resource Efficient**: AutoGluon reduced modeling time significantly
- **Scalable Solution**: Applicable for organizations of any size

## 📘 Lessons Learned

- Threshold tuning can drastically impact real-world performance.
- Precision-Recall curves are essential for imbalanced classification tasks.
- AutoML is powerful for rapid prototyping and deployment.
- Domain-specific metric selection is key (not just accuracy or AUC).

## 🛠️ Tech Stack

- Python
- AutoGluon
- Pandas / Matplotlib / Scikit-learn
- Jupyter Notebook

## 📁 Files in This Repository

- `Patel_Vishw_Module4.ipynb`: Jupyter notebook with full model pipeline and visualizations
- `Patel_Vishw_Module4.pdf`: Project report with business explanation and performance analysis

---

