# Decodelabs Data Science Internship — Week 3

**Dataset:** Customer Churn (1,000 records, 14 features)  
**Tasks Completed:** Task 4 (Data Visualization) + Task 5 (Predictive Modeling)

---

## Project Structure

```
decodelabs_week3/
├── customer_churn.csv          # Dataset
├── task4_visualization.py      # Task 4 — 7 visualization charts
├── task5_model.py              # Task 5 — ML model (Random Forest)
├── outputs/
│   ├── week3_report.html       # Full HTML report (open in browser)
│   ├── chart1_churn_distribution.png
│   ├── chart2_charges_vs_tenure.png
│   ├── chart3_churn_by_contract.png
│   ├── chart4_charges_distribution.png
│   ├── chart5_satisfaction_vs_churn.png
│   ├── chart6_correlation_heatmap.png
│   ├── chart7_dashboard.png
│   ├── model_confusion_matrix.png
│   ├── model_roc_curve.png
│   ├── model_feature_importance.png
│   └── model_comparison.png
└── README.md
```

---

## Task 4 — Data Visualization

Explored the Customer Churn dataset using 7 charts:

- **Churn Distribution** (donut chart) — 40% of customers churned
- **Monthly Charges vs Tenure** (scatter) — colored by churn status
- **Churn Rate by Contract Type** (bar) — month-to-month has highest churn
- **Monthly Charges Distribution** (KDE) — churned customers pay more
- **Satisfaction Score vs Churn** (box + strip) — low scores predict churn
- **Correlation Heatmap** — feature relationships
- **Summary Dashboard** — 6-panel overview

---

## Task 5 — Predictive Modeling

Trained two classifiers to predict whether a customer will churn:

| Model               | Accuracy | ROC-AUC | CV AUC (5-fold) |
|---------------------|----------|---------|-----------------|
| **Random Forest**   | 70.0%    | 0.733   | 0.703           |
| Logistic Regression | 69.0%    | 0.717   | —               |

**Top predictors:** SatisfactionScore, TotalCharges, NumSupportCalls, Tenure

### Approach
- 80/20 train-test split (stratified)
- Label encoding for categorical variables
- StandardScaler applied for Logistic Regression
- Random Forest: 200 trees, max depth 10, min_samples_leaf 5
- Evaluation: Accuracy, ROC-AUC, Confusion Matrix, Feature Importance

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

python task4_visualization.py
python task5_model.py
```

Open `outputs/week3_report.html` in any browser to view the full report.

---

## Key Findings

1. Month-to-month contract customers churn at a significantly higher rate
2. Low satisfaction scores (1–2) are the strongest churn predictor
3. High monthly charges + short tenure = high-risk customer profile
4. Random Forest outperforms Logistic Regression baseline on all metrics

---

*Decodelabs Internship | Week 3 | Ali*
