"""
Decodelabs Week 3 - Task 5: Predictive Modeling
Dataset: Customer Churn
Model: Random Forest Classifier
Author: Ali
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    ConfusionMatrixDisplay
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "customer_churn.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
BG = "#0F0F1A"
CARD = "#1A1A2E"
TEXT = "#EAEAEA"
ACCENT = "#E84545"
TEAL = "#2EC4B6"
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": CARD,
    "axes.edgecolor": "#333355",
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
    "grid.color": "#2A2A4A",
    "grid.alpha": 0.5,
})

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load & Preprocess
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Drop CustomerID — not a feature
df.drop(columns=["CustomerID"], inplace=True)

# Encode all categorical columns
le = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Target
X = df.drop(columns=["Churn"])
y = df["Churn"]   # 1 = Yes (churned), 0 = No

print(f"Features: {list(X.columns)}")
print(f"Target distribution: {dict(y.value_counts())}")

# Train/test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── Scale (for Logistic Regression comparison)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Random Forest — primary model
# ══════════════════════════════════════════════════════════════════════════════
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    min_samples_leaf=5, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf  = rf.predict(X_test)
y_prob_rf  = rf.predict_proba(X_test)[:, 1]

acc_rf  = accuracy_score(y_test, y_pred_rf)
auc_rf  = roc_auc_score(y_test, y_prob_rf)
cv_rf   = cross_val_score(rf, X, y, cv=5, scoring="roc_auc").mean()

print(f"\n── Random Forest ──")
print(f"  Accuracy : {acc_rf:.4f}")
print(f"  ROC-AUC  : {auc_rf:.4f}")
print(f"  CV AUC   : {cv_rf:.4f}")
print(classification_report(y_test, y_pred_rf, target_names=["No Churn", "Churn"]))

# ══════════════════════════════════════════════════════════════════════════════
# 3. Logistic Regression — baseline comparison
# ══════════════════════════════════════════════════════════════════════════════
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)
y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)

print(f"\n── Logistic Regression ──")
print(f"  Accuracy : {acc_lr:.4f}")
print(f"  ROC-AUC  : {auc_lr:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Plot A: Confusion Matrix ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
ax.set_facecolor(CARD)
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
disp.plot(ax=ax, cmap="RdTeal" if False else "YlOrRd",
          colorbar=False, values_format="d")
ax.set_title("Confusion Matrix — Random Forest", fontsize=14, fontweight="bold", color=TEXT)
ax.tick_params(colors=TEXT)
for text in ax.texts:
    text.set_color("black")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_confusion_matrix.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

# ── Plot B: ROC Curve comparison ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG)
ax.set_facecolor(CARD)

for name, probs, color in [
    ("Random Forest", y_prob_rf, ACCENT),
    ("Logistic Reg.", y_prob_lr, TEAL),
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, color=color, lw=2.5,
            label=f"{name}  (AUC = {auc_val:.3f})")

ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4, label="Random Classifier")
ax.fill_between(*roc_curve(y_test, y_prob_rf)[:2], alpha=0.08, color=ACCENT)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — Model Comparison", fontsize=14, fontweight="bold")
ax.legend(frameon=False, fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_roc_curve.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

# ── Plot C: Feature Importance ────────────────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
ax.set_facecolor(CARD)
colors = [ACCENT if v > importances.median() else TEAL for v in importances.values]
ax.barh(importances.index, importances.values, color=colors, height=0.6)
ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
ax.set_title("Random Forest — Feature Importance", fontsize=14, fontweight="bold")
ax.axvline(importances.median(), color="white", lw=1, linestyle="--", alpha=0.4)
ax.grid(axis="x", alpha=0.3)
ax.spines[["top", "right", "left"]].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_feature_importance.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

# ── Plot D: Model Comparison Bar ──────────────────────────────────────────────
metrics = {
    "Accuracy": [acc_rf, acc_lr],
    "ROC-AUC":  [auc_rf, auc_lr],
}
models = ["Random Forest", "Logistic Reg."]

fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor=BG)
fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold")

for ax, (metric, vals) in zip(axes, metrics.items()):
    ax.set_facecolor(CARD)
    bars = ax.bar(models, vals, color=[ACCENT, TEAL], width=0.4)
    ax.set_ylim(0, 1.05)
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 5. Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════")
print("  MODEL SUMMARY")
print("══════════════════════════════════════")
print(f"  Dataset        : {DATA_PATH}")
print(f"  Train samples  : {len(X_train)}")
print(f"  Test  samples  : {len(X_test)}")
print(f"  Features used  : {X.shape[1]}")
print(f"  ── Random Forest ──")
print(f"    Accuracy     : {acc_rf:.4f}")
print(f"    ROC-AUC      : {auc_rf:.4f}")
print(f"    5-fold CV AUC: {cv_rf:.4f}")
print(f"  ── Logistic Regression ──")
print(f"    Accuracy     : {acc_lr:.4f}")
print(f"    ROC-AUC      : {auc_lr:.4f}")
print("══════════════════════════════════════")
print("\n✅ Task 5 Complete — model plots saved to outputs/")
