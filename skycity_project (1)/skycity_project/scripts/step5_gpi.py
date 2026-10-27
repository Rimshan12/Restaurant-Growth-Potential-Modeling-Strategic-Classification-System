"""
STEP 5: Growth Potential Index (GPI) Scoring + Strategic Classification
Assigns each restaurant a GPI score and a strategic recommendation.
Command: python scripts/step5_gpi.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs/gpi", exist_ok=True)

df = pd.read_csv("outputs/enriched_data.csv")
cluster_profiles = pd.read_csv("outputs/clustering/cluster_profiles.csv")

# ── Growth Potential Index (GPI) ───────────────────────────────────────────
# Weighted composite of normalised signals
# Each dimension scaled 0-1 (min-max) within the dataset

def minmax(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)

gpi_df = pd.DataFrame(index=df.index)

# 1. Growth momentum (30%)
gpi_df["g_momentum"] = minmax(df["GrowthFactor"]) * 0.30

# 2. Cost resilience – LOWER cost pressure = higher resilience (25%)
gpi_df["g_cost"] = (1 - minmax(df["CostPressure"])) * 0.25

# 3. Channel balance – more balanced = less fragile (20%)
gpi_df["g_channel"] = minmax(df["ChannelBalance"]) * 0.20

# 4. Logistics scalability (15%)
gpi_df["g_logistics"] = minmax(df["DeliveryRadiusKM"]) * 0.15

# 5. Revenue quality – AOV × margin (10%)
gpi_df["g_revenue"] = minmax(df["RevenueQuality"]) * 0.10

df["GPI"] = gpi_df.sum(axis=1).round(4)

# ── GPI score band labels ──────────────────────────────────────────────────
def gpi_band(score):
    if score >= 0.70: return "High Growth Potential"
    if score >= 0.50: return "Moderate Growth Potential"
    if score >= 0.30: return "Stable / Cautious Growth"
    return "Requires Intervention"

df["GPI_Band"] = df["GPI"].apply(gpi_band)

# ── Cluster archetype labeling ─────────────────────────────────────────────
# Inspect cluster profiles to label archetypes automatically
# We read the profile csv and rank clusters by key signals

profile = pd.read_csv("outputs/clustering/cluster_profiles.csv").set_index("KMeans_Cluster")
n_clusters = len(profile)

# Rank clusters to assign archetypes
rankings = pd.DataFrame(index=profile.index)
rankings["growth_rank"] = profile["GrowthFactor"].rank(ascending=False)
rankings["cost_rank"]   = profile["CostPressure"].rank(ascending=True)   # lower=better
rankings["scale_rank"]  = profile["Scale"].rank(ascending=False)
rankings["aggr_rank"]   = profile["AggregatorShare"].rank(ascending=True) # lower=better
rankings["margin_rank"] = profile["ProfitMargin"].rank(ascending=False)
rankings["composite"]   = rankings.mean(axis=1)

# Archetype templates – flexible to any K
ARCHETYPE_TEMPLATES = [
    "Scalable Market Leaders",
    "High-Growth / High-Risk",
    "Stable Local Performers",
    "Aggregator-Dependent Low Margin",
    "Overextended, Low Return",
    "Cost-Optimised Niche Players",
    "Emerging Delivery Champions",
]

ranked_clusters = rankings["composite"].sort_values().index.tolist()
archetype_map = {}
for i, cluster_id in enumerate(ranked_clusters):
    archetype_map[cluster_id] = ARCHETYPE_TEMPLATES[i % len(ARCHETYPE_TEMPLATES)]

df["Archetype"] = df["KMeans_Cluster"].map(archetype_map)

# ── Strategic recommendation ───────────────────────────────────────────────
def strategic_recommendation(row):
    gpi   = row["GPI"]
    aggr  = row["AggregatorShare"]
    cost  = row["CostPressure"]
    radius= row["DeliveryRadiusKM"]

    if gpi >= 0.70 and cost < 0.65:
        return "Scale Up – Prioritise expansion"
    if gpi >= 0.50 and aggr > 0.65:
        return "Rebalance Channels – Reduce aggregator dependence"
    if gpi >= 0.50 and radius < 8:
        return "Expand Delivery Radius – Logistics headroom available"
    if gpi >= 0.40 and cost >= 0.65:
        return "Optimise Costs – Tighten COGS/OPEX before scaling"
    if gpi >= 0.30:
        return "Hold & Stabilise – Monitor before committing"
    return "Urgent Review – Low growth viability"

df["StrategicRec"] = df.apply(strategic_recommendation, axis=1)

# ── Summary tables ─────────────────────────────────────────────────────────
print("GPI Band Distribution:")
print(df["GPI_Band"].value_counts())
print("\nArchetype Distribution:")
print(df["Archetype"].value_counts())
print("\nStrategic Recommendation Distribution:")
print(df["StrategicRec"].value_counts())

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("GPI Scoring & Strategic Classification", fontsize=14, fontweight="bold")

# GPI distribution
axes[0].hist(df["GPI"], bins=30, color="royalblue", edgecolor="white")
axes[0].set_title("GPI Score Distribution")
axes[0].set_xlabel("Growth Potential Index"); axes[0].set_ylabel("Count")

# GPI by Archetype box
order = df.groupby("Archetype")["GPI"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="GPI", y="Archetype", order=order, ax=axes[1], palette="viridis")
axes[1].set_title("GPI by Cluster Archetype")
axes[1].set_xlabel("GPI")

# Rec distribution
rec_counts = df["StrategicRec"].value_counts()
axes[2].barh(rec_counts.index, rec_counts.values, color="coral")
axes[2].set_title("Strategic Recommendations")
axes[2].set_xlabel("Count")
plt.tight_layout()
plt.savefig("outputs/gpi/gpi_overview.png", dpi=150, bbox_inches="tight")
print("✅  GPI overview saved → outputs/gpi/gpi_overview.png")

# GPI scatter: ProfitMargin vs GrowthFactor coloured by GPI
fig2, ax2 = plt.subplots(figsize=(9,6))
sc = ax2.scatter(df["GrowthFactor"], df["ProfitMargin"],
                 c=df["GPI"], cmap="RdYlGn", s=20, alpha=0.7)
plt.colorbar(sc, label="GPI Score")
ax2.set_xlabel("Growth Factor"); ax2.set_ylabel("Profit Margin")
ax2.set_title("Growth Factor vs Profit Margin (coloured by GPI)")
plt.tight_layout()
plt.savefig("outputs/gpi/gpi_scatter.png", dpi=150, bbox_inches="tight")
print("✅  GPI scatter saved → outputs/gpi/gpi_scatter.png")

# ── Save final results ─────────────────────────────────────────────────────
output_cols = ["RestaurantID","RestaurantName","CuisineType","Segment","Subregion",
               "GPI","GPI_Band","KMeans_Cluster","Archetype","StrategicRec",
               "GrowthFactor","AOV","MonthlyOrders","ProfitMargin",
               "CostPressure","AggregatorShare","DeliveryRadiusKM","TotalRevenue"]
df[output_cols].to_csv("outputs/gpi/restaurant_classifications.csv", index=False)
df.to_csv("outputs/enriched_data.csv", index=False)

print("\n✅  Final classifications saved → outputs/gpi/restaurant_classifications.csv")
print("\nSTEP 5 COMPLETE. Run streamlit_app/app.py for the dashboard!\n")
