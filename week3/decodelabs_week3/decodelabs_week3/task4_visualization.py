"""
Decodelabs Week 3 - Task 4: Data Visualization
Dataset: Customer Churn
Author: Ali
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "customer_churn.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ── Palette ────────────────────────────────────────────────────────────────────
CHURN_PALETTE = {"Yes": "#E84545", "No": "#2EC4B6"}
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
    "font.family": "DejaVu Sans",
})

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Churn Distribution (Donut)
# ══════════════════════════════════════════════════════════════════════════════
churn_counts = df["Churn"].value_counts()

fig, ax = plt.subplots(figsize=(7, 7), facecolor=BG)
ax.set_facecolor(BG)
wedges, texts, autotexts = ax.pie(
    churn_counts,
    labels=None,
    autopct="%1.1f%%",
    startangle=90,
    colors=[CHURN_PALETTE[k] for k in churn_counts.index],
    pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=3),
)
for at in autotexts:
    at.set(fontsize=14, fontweight="bold", color="white")

ax.text(0, 0, f"{churn_counts['Yes']}\nChurned", ha="center", va="center",
        fontsize=18, fontweight="bold", color=ACCENT)

patches = [mpatches.Patch(color=CHURN_PALETTE[k], label=f"{k}: {v}")
           for k, v in churn_counts.items()]
ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.05),
          ncol=2, frameon=False, fontsize=12)
ax.set_title("Customer Churn Distribution", fontsize=16, fontweight="bold",
             pad=20, color=TEXT)

plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "chart1_churn_distribution.png")
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  Saved: {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Monthly Charges vs Tenure (Scatter coloured by Churn)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
ax.set_facecolor(CARD)

for label, group in df.groupby("Churn"):
    ax.scatter(group["Tenure"], group["MonthlyCharges"],
               alpha=0.55, s=30, label=label,
               color=CHURN_PALETTE[label], edgecolors="none")

ax.set_xlabel("Tenure (months)", fontsize=12)
ax.set_ylabel("Monthly Charges ($)", fontsize=12)
ax.set_title("Monthly Charges vs Tenure — Coloured by Churn", fontsize=14, fontweight="bold")
ax.legend(title="Churn", frameon=False, fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "chart2_charges_vs_tenure.png")
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  Saved: {out2}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Churn Rate by Contract Type
# ══════════════════════════════════════════════════════════════════════════════
contract_churn = (df.groupby("Contract")["Churn"]
                  .apply(lambda x: (x == "Yes").mean() * 100)
                  .reset_index(name="ChurnRate"))
contract_churn = contract_churn.sort_values("ChurnRate", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(CARD)

bars = ax.barh(contract_churn["Contract"], contract_churn["ChurnRate"],
               color=[ACCENT, "#E8A045", TEAL], edgecolor="none", height=0.5)
for bar, val in zip(bars, contract_churn["ChurnRate"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=12, fontweight="bold")

ax.set_xlabel("Churn Rate (%)", fontsize=12)
ax.set_title("Churn Rate by Contract Type", fontsize=14, fontweight="bold")
ax.set_xlim(0, contract_churn["ChurnRate"].max() + 10)
ax.grid(axis="x", alpha=0.3)
ax.spines[["top", "right", "left"]].set_visible(False)

plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "chart3_churn_by_contract.png")
plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  Saved: {out3}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Monthly Charges Distribution by Churn
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
ax.set_facecolor(CARD)

for label in ["No", "Yes"]:
    subset = df[df["Churn"] == label]["MonthlyCharges"]
    subset.plot.kde(ax=ax, label=label, color=CHURN_PALETTE[label], linewidth=2.5)
    ax.fill_between(
        np.linspace(subset.min(), subset.max(), 200),
        0,
        [ax.lines[-1].get_ydata()[i] for i in range(200)],
        alpha=0.15, color=CHURN_PALETTE[label]
    )

ax.set_xlabel("Monthly Charges ($)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Monthly Charges Distribution by Churn Status", fontsize=14, fontweight="bold")
ax.legend(title="Churn", frameon=False, fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out4 = os.path.join(OUTPUT_DIR, "chart4_charges_distribution.png")
plt.savefig(out4, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  Saved: {out4}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Satisfaction Score vs Churn (Box + Strip)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
ax.set_facecolor(CARD)

sns.boxplot(data=df, x="Churn", y="SatisfactionScore",
            palette=CHURN_PALETTE, ax=ax,
            width=0.4, fliersize=0,
            medianprops=dict(color="white", linewidth=2))
sns.stripplot(data=df, x="Churn", y="SatisfactionScore",
              palette=CHURN_PALETTE, ax=ax,
              alpha=0.25, jitter=True, size=3, dodge=False)

ax.set_xlabel("Churn", fontsize=12)
ax.set_ylabel("Satisfaction Score (1–5)", fontsize=12)
ax.set_title("Customer Satisfaction Score vs Churn", fontsize=14, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out5 = os.path.join(OUTPUT_DIR, "chart5_satisfaction_vs_churn.png")
plt.savefig(out5, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  Saved: {out5}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Correlation Heatmap (numeric features)
# ══════════════════════════════════════════════════════════════════════════════
num_cols = ["Age", "Tenure", "MonthlyCharges", "TotalCharges",
            "NumSupportCalls", "SatisfactionScore"]
corr = df[num_cols].corr()

cmap = LinearSegmentedColormap.from_list("churn_div",
    ["#2EC4B6", "#1A1A2E", "#E84545"])

fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
ax.set_facecolor(CARD)
sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
            ax=ax, linewidths=0.5, linecolor="#0F0F1A",
            annot_kws={"size": 11},
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)

plt.tight_layout()
out6 = os.path.join(OUTPUT_DIR, "chart6_correlation_heatmap.png")
plt.savefig(out6, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  Saved: {out6}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 7 — Dashboard (2×3 grid summary)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 11), facecolor=BG)
fig.suptitle("Customer Churn Analysis Dashboard", fontsize=22,
             fontweight="bold", color=TEXT, y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

# -- Panel A: Churn Donut
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_facecolor(BG)
wedges, _, autotexts = ax_a.pie(
    churn_counts,
    autopct="%1.1f%%", startangle=90,
    colors=[CHURN_PALETTE[k] for k in churn_counts.index],
    pctdistance=0.75,
    wedgeprops=dict(width=0.5, edgecolor=BG, linewidth=2),
)
for at in autotexts:
    at.set(fontsize=11, fontweight="bold", color="white")
ax_a.text(0, 0, f"{churn_counts['Yes']}\nLeft", ha="center", va="center",
          fontsize=13, fontweight="bold", color=ACCENT)
ax_a.set_title("Churn Share", fontsize=13, fontweight="bold")

# -- Panel B: Contract bar
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_facecolor(CARD)
ax_b.barh(contract_churn["Contract"], contract_churn["ChurnRate"],
          color=[ACCENT, "#E8A045", TEAL], height=0.45)
ax_b.set_xlabel("Churn Rate (%)")
ax_b.set_title("Churn by Contract", fontsize=13, fontweight="bold")
ax_b.grid(axis="x", alpha=0.3)
ax_b.spines[["top", "right", "left"]].set_visible(False)

# -- Panel C: Internet service churn
ax_c = fig.add_subplot(gs[0, 2])
ax_c.set_facecolor(CARD)
inet_churn = (df.groupby("InternetService")["Churn"]
              .apply(lambda x: (x == "Yes").mean() * 100)
              .reset_index(name="ChurnRate"))
ax_c.bar(inet_churn["InternetService"], inet_churn["ChurnRate"],
         color=[TEAL, ACCENT, "#E8A045"], width=0.5)
ax_c.set_ylabel("Churn Rate (%)")
ax_c.set_title("Churn by Internet Service", fontsize=13, fontweight="bold")
ax_c.grid(axis="y", alpha=0.3)
ax_c.spines[["top", "right", "left"]].set_visible(False)

# -- Panel D: Charges KDE
ax_d = fig.add_subplot(gs[1, 0])
ax_d.set_facecolor(CARD)
for label in ["No", "Yes"]:
    df[df["Churn"] == label]["MonthlyCharges"].plot.kde(
        ax=ax_d, label=label, color=CHURN_PALETTE[label], linewidth=2)
ax_d.set_xlabel("Monthly Charges ($)")
ax_d.set_title("Monthly Charges by Churn", fontsize=13, fontweight="bold")
ax_d.legend(frameon=False, fontsize=10)
ax_d.grid(alpha=0.3)

# -- Panel E: Satisfaction boxplot
ax_e = fig.add_subplot(gs[1, 1])
ax_e.set_facecolor(CARD)
sns.boxplot(data=df, x="Churn", y="SatisfactionScore",
            palette=CHURN_PALETTE, ax=ax_e, width=0.4, fliersize=0,
            medianprops=dict(color="white", linewidth=2))
ax_e.set_title("Satisfaction vs Churn", fontsize=13, fontweight="bold")
ax_e.grid(axis="y", alpha=0.3)

# -- Panel F: Avg support calls by churn
ax_f = fig.add_subplot(gs[1, 2])
ax_f.set_facecolor(CARD)
support = df.groupby("Churn")["NumSupportCalls"].mean()
ax_f.bar(support.index, support.values,
         color=[CHURN_PALETTE[k] for k in support.index], width=0.4)
ax_f.set_ylabel("Avg Support Calls")
ax_f.set_title("Avg Support Calls by Churn", fontsize=13, fontweight="bold")
ax_f.grid(axis="y", alpha=0.3)
ax_f.spines[["top", "right", "left"]].set_visible(False)

plt.savefig(os.path.join(OUTPUT_DIR, "chart7_dashboard.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"  Saved: chart7_dashboard.png")

print("\n✅ Task 4 Complete — all 7 charts saved to outputs/")
