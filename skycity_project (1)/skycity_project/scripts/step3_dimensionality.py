"""
STEP 3: Dimensionality Reduction (PCA + UMAP)
Discover latent structure before clustering.
Command: python scripts/step3_dimensionality.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

os.makedirs("outputs/eda", exist_ok=True)

X_scaled = pd.read_csv("outputs/X_scaled.csv")
print(f"Loaded scaled feature matrix: {X_scaled.shape}")

# ── PCA ────────────────────────────────────────────────────────────────────
pca_full = PCA()
pca_full.fit(X_scaled)

explained = np.cumsum(pca_full.explained_variance_ratio_)
n_components_90 = int(np.argmax(explained >= 0.90)) + 1
print(f"Components to explain 90% variance: {n_components_90}")

# Scree plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(range(1, len(explained)+1), explained, marker="o", markersize=3)
axes[0].axhline(0.90, color="red", linestyle="--", label="90% threshold")
axes[0].axvline(n_components_90, color="red", linestyle=":")
axes[0].set_title("PCA – Cumulative Explained Variance")
axes[0].set_xlabel("Number of Components")
axes[0].set_ylabel("Cumulative Variance Explained")
axes[0].legend()

# PCA loadings for first 3 PCs
pca_3 = PCA(n_components=3)
X_pca = pca_3.fit_transform(X_scaled)
X_pca_df = pd.DataFrame(X_pca, columns=["PC1","PC2","PC3"])

# Loading heatmap (PC1–3 vs features)
loadings = pd.DataFrame(pca_3.components_.T,
                         index=X_scaled.columns,
                         columns=["PC1","PC2","PC3"])
top_features = loadings.abs().max(axis=1).nlargest(15).index
import seaborn as sns
sns.heatmap(loadings.loc[top_features], annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, ax=axes[1], linewidths=0.4)
axes[1].set_title("PCA Loadings – Top 15 Features")
plt.tight_layout()
plt.savefig("outputs/eda/pca_analysis.png", dpi=150, bbox_inches="tight")
print("✅  PCA analysis saved → outputs/eda/pca_analysis.png")

# Save PCA coords
X_pca_df.to_csv("outputs/X_pca.csv", index=False)

# ── UMAP (optional – skip if umap-learn not installed) ─────────────────────
try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=20, min_dist=0.1)
    X_umap = reducer.fit_transform(X_scaled)
    X_umap_df = pd.DataFrame(X_umap, columns=["UMAP1","UMAP2"])
    X_umap_df.to_csv("outputs/X_umap.csv", index=False)

    fig2, ax2 = plt.subplots(figsize=(8,6))
    ax2.scatter(X_umap[:,0], X_umap[:,1], s=6, alpha=0.5, c="royalblue")
    ax2.set_title("UMAP 2D Projection of Restaurant Features")
    plt.savefig("outputs/eda/umap_projection.png", dpi=150, bbox_inches="tight")
    print("✅  UMAP projection saved → outputs/eda/umap_projection.png")
except ImportError:
    print("⚠️  umap-learn not installed. Skipping UMAP. Run: pip install umap-learn")
    X_umap_df = None

# ── Interpret dominant PCs ─────────────────────────────────────────────────
print("\nTop feature contributors per PC:")
for pc in ["PC1","PC2","PC3"]:
    top = loadings[pc].abs().nlargest(5)
    print(f"\n  {pc}: {list(top.index)}")

print("\nSTEP 3 COMPLETE. Run step4_clustering.py next.\n")
