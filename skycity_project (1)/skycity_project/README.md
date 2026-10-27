# 🍽️ SkyCity Auckland – Restaurant Growth Potential Modeling
## Complete Step-by-Step Project Guide

---

## 📁 PROJECT FOLDER STRUCTURE

```
skycity_project/
├── data/
│   └── SkyCity_Auckland_Restaurants___Bars.csv   ← your dataset goes here
├── scripts/
│   ├── step1_eda.py              ← Exploratory Data Analysis
│   ├── step2_preprocessing.py   ← Feature engineering + scaling
│   ├── step3_dimensionality.py  ← PCA + UMAP
│   ├── step4_clustering.py      ← K-Means + Hierarchical + DBSCAN
│   └── step5_gpi.py             ← GPI scoring + strategic classification
├── streamlit_app/
│   └── app.py                   ← Interactive dashboard
├── outputs/                     ← Auto-created when scripts run
│   ├── eda/
│   ├── clustering/
│   ├── gpi/
│   └── models/
└── requirements.txt
```

---

## ⚙️ SETUP (Do This Once)

### Step 0A — Install Python
- Download Python 3.10 or 3.11 from https://python.org/downloads
- During install ✅ check "Add Python to PATH"
- Verify: open a terminal and type: `python --version`

### Step 0B — Create project folder
1. Create a folder on your desktop: `skycity_project`
2. Copy all scripts into it exactly as shown in the folder structure above
3. Put the CSV file at `data/SkyCity_Auckland_Restaurants___Bars.csv`

### Step 0C — Open Terminal / Command Prompt
- **Windows**: Press Win+R → type `cmd` → Enter
- **Mac**: Press Cmd+Space → type `Terminal` → Enter
- **Linux**: Ctrl+Alt+T

Navigate to the project folder:
```bash
cd Desktop/skycity_project
```

### Step 0D — Install required libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy plotly streamlit joblib
```
Optional but recommended (for UMAP):
```bash
pip install umap-learn
```
This takes 2–5 minutes. You only need to do this once.

---

## 🚀 RUNNING THE PROJECT (5 Steps in Order)

> ⚠️ Always run from inside the `skycity_project` folder.
> Run each step in order — each one depends on the previous.

---

### STEP 1 — Exploratory Data Analysis
**What it does:** Loads the dataset, computes derived KPIs (total revenue, profit margin, etc.), generates correlation heatmaps and distribution plots.

**Where to run:** Terminal inside `skycity_project/`
```bash
python scripts/step1_eda.py
```

**Outputs created:**
| File | Description |
|------|-------------|
| `outputs/eda/eda_overview.png` | 4-panel EDA chart |
| `outputs/eda/correlation_heatmap.png` | Feature correlation matrix |
| `outputs/enriched_data.csv` | Dataset with derived KPI columns |

**Key findings from this dataset:**
- 1,696 restaurants across 4 subregions (CBD, North Shore, South & West Auckland)
- 8 cuisine types; Burgers (375) and Indian (329) are most common
- Avg monthly orders: 1,190 | Avg revenue: $45,837
- Avg profit margin: 10.4% (ranges from -17.7% to +34.3%)

---

### STEP 2 — Data Preprocessing & Feature Engineering
**What it does:** Creates 7 derived features (Scale, CostPressure, ChannelBalance, etc.), encodes categorical columns, log-transforms skewed variables, and produces a StandardScaler-normalized feature matrix.

**Where to run:**
```bash
python scripts/step2_preprocessing.py
```

**Outputs created:**
| File | Description |
|------|-------------|
| `outputs/X_scaled.csv` | 24-feature normalized matrix (1696 × 24) |
| `outputs/models/scaler.pkl` | Scaler object (needed for Streamlit) |
| `outputs/models/le_*.pkl` | Label encoder objects |

**Features engineered:**
- `Scale` = MonthlyOrders × GrowthFactor
- `CostPressure` = COGSRate + OPEXRate
- `ChannelBalance` = Shannon entropy across 4 channels
- `AggregatorShare` = UE_share + DD_share
- `DeliveryEfficiency` = DeliveryRadiusKM / (DeliveryCostPerOrder + 1)
- `ProfitMargin` = TotalNetProfit / TotalRevenue
- `RevenueQuality` = AOV × ProfitMargin

---

### STEP 3 — Dimensionality Reduction (PCA)
**What it does:** Applies PCA to discover latent structure. Identifies which features drive the most variance. Saves 3D PCA coordinates for clustering visualization.

**Where to run:**
```bash
python scripts/step3_dimensionality.py
```

**Outputs created:**
| File | Description |
|------|-------------|
| `outputs/eda/pca_analysis.png` | Scree plot + loading heatmap |
| `outputs/X_pca.csv` | 3 principal component coordinates |
| `outputs/eda/umap_projection.png` | UMAP 2D map (if umap-learn installed) |

**PCA findings (this dataset):**
- **PC1** → Cost pressure, profit margin (cost vs margin axis)
- **PC2** → Channel mix (aggregator vs self-delivery axis)
- **PC3** → Delivery logistics (radius, cost, scale axis)
- 9 components explain 90% of variance

---

### STEP 4 — Unsupervised Clustering
**What it does:** Finds optimal K using Elbow + Silhouette + Davies-Bouldin methods. Runs K-Means, Hierarchical Clustering, and DBSCAN. Generates cluster profiles and visualizations.

**Where to run:**
```bash
python scripts/step4_clustering.py
```

**Outputs created:**
| File | Description |
|------|-------------|
| `outputs/clustering/kmeans_k_selection.png` | Elbow + Silhouette + DB charts |
| `outputs/clustering/dendrogram.png` | Hierarchical clustering tree |
| `outputs/clustering/cluster_scatter.png` | PCA scatter coloured by cluster |
| `outputs/clustering/cluster_profiles.csv` | Mean KPIs per cluster |
| `outputs/models/kmeans_model.pkl` | Trained K-Means model |

**Cluster profiles found (this dataset):**
| Cluster | Growth | Margin | Cost | Aggregator Dep |
|---------|--------|--------|------|----------------|
| 0 | 1.027 | -5.9% | 0.875 | 75% |
| 1 | 1.029 | +16.4% | 0.625 | 75% |

Cluster 0 = high-cost, margin-negative restaurants → need intervention
Cluster 1 = better-margin, lower-cost → growth candidates

---

### STEP 5 — GPI Scoring & Strategic Classification
**What it does:** Assigns each restaurant a Growth Potential Index (0–1), a GPI Band label, a cluster archetype name, and a strategic recommendation.

**Where to run:**
```bash
python scripts/step5_gpi.py
```

**Outputs created:**
| File | Description |
|------|-------------|
| `outputs/gpi/restaurant_classifications.csv` | Final classification for all 1,696 restaurants |
| `outputs/gpi/gpi_overview.png` | GPI distribution + archetype boxplot |
| `outputs/gpi/gpi_scatter.png` | GrowthFactor vs Margin coloured by GPI |

**GPI Formula:**
```
GPI = (GrowthFactor × 0.30) + (CostResilience × 0.25) +
      (ChannelBalance × 0.20) + (LogisticsScale × 0.15) +
      (RevenueQuality × 0.10)
```

**GPI results:**
| Band | Count |
|------|-------|
| High Growth Potential (≥0.70) | 426 |
| Moderate Growth Potential (0.50–0.69) | 819 |
| Stable / Cautious Growth (0.30–0.49) | 389 |
| Requires Intervention (<0.30) | 62 |

**Strategic recommendations assigned:**
| Recommendation | Count |
|----------------|-------|
| Rebalance Channels | 805 |
| Scale Up | 323 |
| Hold & Stabilise | 272 |
| Optimise Costs | 202 |
| Urgent Review | 62 |
| Expand Delivery Radius | 32 |

---

## 📊 STEP 6 — Launch the Streamlit Dashboard

**Where to run:**
```bash
streamlit run streamlit_app/app.py
```

This opens your browser automatically at `http://localhost:8501`

**Dashboard tabs:**
| Tab | What you see |
|-----|-------------|
| 📊 Cluster Map | PCA scatter, archetype pie chart, subregion×cuisine GPI heatmap |
| 🏆 Growth Scorecards | Sortable restaurant rankings with GPI & strategy |
| 📡 Feature Radar | Side-by-side radar comparison of any 2 restaurants |
| 🗺️ Strategy Panel | Recommendation distribution, cost vs GPI scatter |
| 📋 Data Explorer | Full searchable table + CSV download |

**Sidebar filters:** Subregion, Cuisine, Segment, Archetype, GPI range

---

## 📝 STEP 7 — Research Paper / Executive Summary

Use the outputs to write a research paper covering these sections:

1. **Introduction** — Why growth classification matters
2. **EDA** — Cite outputs/eda/eda_overview.png + correlation_heatmap.png
3. **Feature Engineering** — Explain the 7 derived features
4. **Dimensionality Reduction** — PCA 3-component interpretation
5. **Clustering Results** — Cluster profiles table, archetype descriptions
6. **GPI Model** — Formula, weights rationale, score distribution
7. **Strategic Recommendations** — What each recommendation means
8. **Conclusion** — Limitations + future work

---

## 🛠️ TROUBLESHOOTING

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip install <module_name>` |
| `FileNotFoundError: outputs/enriched_data.csv` | Run steps in order starting from step1 |
| `streamlit: command not found` | Run `pip install streamlit` then try again |
| `Permission denied` | On Mac/Linux prefix with `python3` instead of `python` |
| UMAP not found | `pip install umap-learn` (optional, Step 3 still works without it) |

---

## ✅ QUICK-RUN CHECKLIST

- [ ] Python 3.10+ installed
- [ ] CSV placed at `data/SkyCity_Auckland_Restaurants___Bars.csv`
- [ ] `pip install -r requirements.txt` done
- [ ] `python scripts/step1_eda.py` ✅
- [ ] `python scripts/step2_preprocessing.py` ✅
- [ ] `python scripts/step3_dimensionality.py` ✅
- [ ] `python scripts/step4_clustering.py` ✅
- [ ] `python scripts/step5_gpi.py` ✅
- [ ] `streamlit run streamlit_app/app.py` 🚀

---

*Project: SkyCity Auckland Restaurant Growth Potential Modeling & Strategic Classification System*
