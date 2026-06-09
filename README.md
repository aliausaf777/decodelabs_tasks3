# 📊 Customer Churn Analysis & Predictive Modeling

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![Internship](https://img.shields.io/badge/Decodelabs-Week%203-purple)

## 📌 Overview

This project was completed as part of the **Decodelabs Data Science Internship – Week 3**.

The objective was to analyze a customer churn dataset through Exploratory Data Analysis (EDA) and Machine Learning techniques to identify churn patterns and predict customer attrition.

### Dataset Information

- **Records:** 1,000 Customers
- **Features:** 14 Variables
- **Domain:** Customer Churn Prediction

---

## 📁 Project Structure

```bash
decodelabs_week3/
│
├── customer_churn.csv
├── task4_visualization.py
├── task5_model.py
│
├── outputs/
│   ├── week3_report.html
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
│
└── README.md
```

---

## 📊 Task 4 — Data Visualization

Seven visualizations were created to understand customer behavior and churn patterns.

### Visualizations

| Chart | Description |
|---------|------------|
| Churn Distribution | Overall churn percentage |
| Monthly Charges vs Tenure | Customer lifecycle analysis |
| Churn by Contract Type | Contract impact on churn |
| Charges Distribution | Spending behavior comparison |
| Satisfaction vs Churn | Customer sentiment analysis |
| Correlation Heatmap | Feature relationships |
| Dashboard Summary | Consolidated business overview |

### Key Insights

- Approximately **40%** of customers churned.
- Month-to-month contracts show the highest churn rates.
- Customers with low satisfaction scores are significantly more likely to churn.
- High monthly charges combined with short tenure indicate elevated churn risk.

---

## 🤖 Task 5 — Predictive Modeling

Two machine learning classification models were trained and evaluated.

### Models Used

- Random Forest Classifier
- Logistic Regression

### Model Performance

| Model | Accuracy | ROC-AUC | CV AUC (5-Fold) |
|---------|----------|----------|----------|
| Random Forest | **70.0%** | **0.733** | **0.703** |
| Logistic Regression | 69.0% | 0.717 | — |

🏆 **Best Performing Model:** Random Forest Classifier

---

## 🔍 Feature Importance

Top predictors of customer churn:

1. SatisfactionScore
2. TotalCharges
3. NumSupportCalls
4. Tenure

These features had the highest influence on model predictions.

---

## ⚙️ Methodology

### Data Preprocessing

- Label Encoding for categorical features
- Train-Test Split (80/20)
- Feature Scaling for Logistic Regression
- Stratified Sampling

### Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)
```

### Evaluation Metrics

- Accuracy
- ROC-AUC
- Confusion Matrix
- Cross Validation
- Feature Importance

---

## 📈 Business Findings

### 1. Contract Type Drives Churn

Customers on month-to-month contracts are considerably more likely to leave.

### 2. Satisfaction Is Critical

Low satisfaction scores are the strongest indicator of future churn.

### 3. High-Risk Customer Segment

Customers with:

- High Monthly Charges
- Low Tenure
- Frequent Support Calls

show the highest probability of churn.

### 4. Predictive Analytics Adds Business Value

Machine learning can proactively identify at-risk customers and support retention strategies.

---

## 🚀 Installation

Install required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## ▶️ Running the Project

### Generate Visualizations

```bash
python task4_visualization.py
```

### Train Machine Learning Models

```bash
python task5_model.py
```

---

## 📄 View Report

Open the generated HTML report:

```bash
outputs/week3_report.html
```

in any modern web browser.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## 🎯 Skills Demonstrated

- Exploratory Data Analysis (EDA)
- Data Visualization
- Machine Learning
- Classification Modeling
- Feature Engineering
- Model Evaluation
- Business Analytics
- Data Storytelling

---

## 👨‍💻 Author

**Ali**

Decodelabs Data Science Internship — Week 3

---

⭐ If you found this project useful, consider giving the repository a star.
