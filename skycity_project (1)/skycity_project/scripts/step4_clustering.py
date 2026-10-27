"""
STEP 4: Unsupervised Clustering (K-Means + Hierarchical + DBSCAN)
Identifies restaurant archetypes.
Command: python scripts/step4_clustering.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib
import os

os.makedirs("outputs/clustering", exist_ok=True)

X_scaled = pd.read_csv("outputs/X_scaled.csv")
X_pca    = pd.read_csv("outputs/X_pca.csv")
df       = pd.read_csv("outputs/enriched_data.csv")

# ── Elbow + Silhouette to choose K ─────────────────────────────────────────
print("Finding optimal K for K-Means (k=2..10)...")
inertias, silhouettes, db_scores = [], [], []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels, sample_size=500))
    db_scores.append(davies_bouldin_score(X_scaled, labels))
    print(f"  k={k}  inertia={km.inertia_:.0f}  sil={silhouettes[-1]:.3f}  db={db_scores[-1]:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(K_range, inertias, marker="o"); axes[0].set_title("Elbow Curve")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
axes[1].plot(K_range, silhouettes, marker="o", color="green"); axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Score")
axes[2].plot(K_range, db_scores, marker="o", color="red"); axes[2].set_title("Davies-Bouldin Score (lower=better)")
axes[2].set_xlabel("k"); axes[2].set_ylabel("Score")
plt.tight_layout()
plt.savefig("outputs/clustering/kmeans_k_selection.png", dpi=150, bbox_inches="tight")
print("✅  K-selection chart saved → outputs/clustering/kmeans_k_selection.png")

# ── Best K: highest silhouette ─────────────────────────────────────────────
best_k = list(K_range)[np.argmax(silhouettes)]
print(f"\nBest K by silhouette: {best_k}")

# ── Final K-Means ─────────────────────────────────────────────────────────
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
df["KMeans_Cluster"] = km_final.fit_predict(X_scaled)
joblib.dump(km_final, "outputs/models/kmeans_model.pkl")

# ── Hierarchical Clustering ────────────────────────────────────────────────
hc = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
df["HC_Cluster"] = hc.fit_predict(X_scaled)

# Dendrogram on sample
sample_idx = np.random.choice(len(X_scaled), 200, replace=False)
Z = linkage(X_scaled.iloc[sample_idx], method="ward")
fig2, ax2 = plt.subplots(figsize=(14, 5))
dendrogram(Z, no_labels=True, ax=ax2)
ax2.set_title("Hierarchical Clustering Dendrogram (200-sample)")
plt.tight_layout()
plt.savefig("outputs/clustering/dendrogram.png", dpi=150, bbox_inches="tight")
print("✅  Dendrogram saved → outputs/clustering/dendrogram.png")

# ── DBSCAN robustness check ────────────────────────────────────────────────
db = DBSCAN(eps=2.5, min_samples=10)
df["DBSCAN_Cluster"] = db.fit_predict(X_scaled)
n_noise = (df["DBSCAN_Cluster"] == -1).sum()
print(f"DBSCAN: {df['DBSCAN_Cluster'].nunique()-1} clusters, {n_noise} noise points")

# ── Cluster scatter on PCA ─────────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
palette = sns.color_palette("tab10", best_k)

for ax, col, title in zip(axes3,
    ["KMeans_Cluster","HC_Cluster"],
    ["K-Means Clusters (PCA)", "Hierarchical Clusters (PCA)"]):
    for c in sorted(df[col].unique()):
        mask = df[col] == c
        ax.scatter(X_pca.loc[mask,"PC1"], X_pca.loc[mask,"PC2"],
                   s=12, alpha=0.6, label=f"Cluster {c}", color=palette[c % best_k])
    ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(markerscale=2, fontsize=8)
plt.tight_layout()
plt.savefig("outputs/clustering/cluster_scatter.png", dpi=150, bbox_inches="tight")
print("✅  Cluster scatter saved → outputs/clustering/cluster_scatter.png")

# ── Cluster profile table ─────────────────────────────────────────────────
profile_cols = ["GrowthFactor","AOV","MonthlyOrders","ProfitMargin",
                "CostPressure","AggregatorShare","DeliveryRadiusKM","Scale"]
profile = df.groupby("KMeans_Cluster")[profile_cols].mean().round(3)
print("\nK-Means Cluster Profiles:\n", profile.to_string())
profile.to_csv("outputs/clustering/cluster_profiles.csv")

df.to_csv("outputs/enriched_data.csv", index=False)
print("\nSTEP 4 COMPLETE. Run step5_gpi.py next.\n")
