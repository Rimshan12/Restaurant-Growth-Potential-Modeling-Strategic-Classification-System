"""
STEP 1: Exploratory Data Analysis (EDA)
Run this first to understand the dataset.
Command: python scripts/step1_eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Setup ──────────────────────────────────────────────────────────────────
os.makedirs("outputs/eda", exist_ok=True)
df = pd.read_csv("data/SkyCity_Auckland_Restaurants___Bars.csv")

print("=" * 60)
print("SKYCITY AUCKLAND – EXPLORATORY DATA ANALYSIS")
print("=" * 60)
print(f"\nDataset shape : {df.shape[0]} restaurants × {df.shape[1]} features")
print(f"Missing values: {df.isnull().sum().sum()}")
print("\nColumn overview:")
print(df.dtypes)

# ── Basic stats ────────────────────────────────────────────────────────────
print("\n\n--- NUMERICAL SUMMARY ---")
print(df.describe().round(2).to_string())

# ── Categorical breakdown ──────────────────────────────────────────────────
print("\n\n--- CATEGORICAL DISTRIBUTIONS ---")
for col in ["CuisineType", "Segment", "Subregion"]:
    print(f"\n{col}:\n{df[col].value_counts()}")

# ── Derived KPIs ───────────────────────────────────────────────────────────
df["TotalRevenue"] = (df["InStoreRevenue"] + df["UberEatsRevenue"]
                      + df["DoorDashRevenue"] + df["SelfDeliveryRevenue"])
df["TotalNetProfit"] = (df["InStoreNetProfit"] + df["UberEatsNetProfit"]
                        + df["DoorDashNetProfit"] + df["SelfDeliveryNetProfit"])
df["ProfitMargin"]  = df["TotalNetProfit"] / df["TotalRevenue"]
df["AggregatorShare"] = df["UE_share"] + df["DD_share"]
df["Scale"]         = df["MonthlyOrders"] * df["GrowthFactor"]
df["CostPressure"]  = df["COGSRate"] + df["OPEXRate"]

print("\n\n--- DERIVED KPI SUMMARY ---")
print(df[["TotalRevenue","TotalNetProfit","ProfitMargin","AggregatorShare","Scale","CostPressure"]].describe().round(3).to_string())

# ── Plots ──────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")

# 1. Revenue by Cuisine
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("SkyCity Auckland – EDA Overview", fontsize=16, fontweight="bold")

cuisine_rev = df.groupby("CuisineType")["TotalRevenue"].mean().sort_values(ascending=False)
axes[0,0].bar(cuisine_rev.index, cuisine_rev.values, color="steelblue")
axes[0,0].set_title("Avg Revenue by Cuisine Type")
axes[0,0].set_xlabel("Cuisine"); axes[0,0].set_ylabel("Avg Revenue ($)")
axes[0,0].tick_params(axis="x", rotation=45)

# 2. GrowthFactor distribution
axes[0,1].hist(df["GrowthFactor"], bins=20, color="coral", edgecolor="white")
axes[0,1].set_title("Growth Factor Distribution")
axes[0,1].set_xlabel("GrowthFactor"); axes[0,1].set_ylabel("Count")

# 3. Profit Margin by Segment
seg_margin = df.groupby("Segment")["ProfitMargin"].mean().sort_values(ascending=False)
axes[1,0].barh(seg_margin.index, seg_margin.values, color="seagreen")
axes[1,0].set_title("Avg Profit Margin by Segment")
axes[1,0].set_xlabel("Profit Margin")

# 4. Aggregator Dependence vs Profit Margin
axes[1,1].scatter(df["AggregatorShare"], df["ProfitMargin"],
                  alpha=0.3, c="purple", s=15)
axes[1,1].set_title("Aggregator Dependence vs Profit Margin")
axes[1,1].set_xlabel("Aggregator Share (UE+DD)")
axes[1,1].set_ylabel("Profit Margin")

plt.tight_layout()
plt.savefig("outputs/eda/eda_overview.png", dpi=150, bbox_inches="tight")
print("\n✅  EDA plot saved → outputs/eda/eda_overview.png")

# 5. Correlation heatmap
numeric_cols = ["GrowthFactor","AOV","MonthlyOrders","COGSRate","OPEXRate",
                "CommissionRate","DeliveryRadiusKM","ProfitMargin",
                "AggregatorShare","Scale","CostPressure"]
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, ax=ax2, linewidths=0.5)
ax2.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/eda/correlation_heatmap.png", dpi=150, bbox_inches="tight")
print("✅  Correlation heatmap saved → outputs/eda/correlation_heatmap.png")

# ── Save enriched data ─────────────────────────────────────────────────────
df.to_csv("outputs/enriched_data.csv", index=False)
print("✅  Enriched dataset saved → outputs/enriched_data.csv")
print("\nSTEP 1 COMPLETE. Run step2_preprocessing.py next.\n")
