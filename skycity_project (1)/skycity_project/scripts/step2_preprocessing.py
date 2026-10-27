"""
STEP 2: Data Preprocessing & Feature Engineering
Normalise, encode, reduce skew, build model-ready feature matrix.
Command: python scripts/step2_preprocessing.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

os.makedirs("outputs/models", exist_ok=True)

# ── Load enriched data ─────────────────────────────────────────────────────
df = pd.read_csv("outputs/enriched_data.csv")
print(f"Loaded {df.shape[0]} restaurants.")

# ── Feature engineering ────────────────────────────────────────────────────
# Aggregator dependence
df["AggregatorShare"]  = df["UE_share"] + df["DD_share"]
# Scale metric
df["Scale"]            = df["MonthlyOrders"] * df["GrowthFactor"]
# Cost pressure
df["CostPressure"]     = df["COGSRate"] + df["OPEXRate"]
# Revenue quality (AOV × profit margin proxy)
df["TotalRevenue"]     = (df["InStoreRevenue"] + df["UberEatsRevenue"]
                          + df["DoorDashRevenue"] + df["SelfDeliveryRevenue"])
df["TotalNetProfit"]   = (df["InStoreNetProfit"] + df["UberEatsNetProfit"]
                          + df["DoorDashNetProfit"] + df["SelfDeliveryNetProfit"])
df["ProfitMargin"]     = df["TotalNetProfit"] / df["TotalRevenue"]
df["RevenueQuality"]   = df["AOV"] * df["ProfitMargin"]
# Channel balance (entropy-like: balanced = higher)
shares = df[["InStoreShare","UE_share","DD_share","SD_share"]].clip(lower=1e-9)
df["ChannelBalance"] = -(shares * np.log(shares)).sum(axis=1)   # entropy
# Delivery efficiency
df["DeliveryEfficiency"] = df["DeliveryRadiusKM"] / (df["DeliveryCostPerOrder"] + 1)

# ── Encode categoricals ────────────────────────────────────────────────────
le_cuisine  = LabelEncoder()
le_segment  = LabelEncoder()
le_subregion = LabelEncoder()
df["CuisineType_enc"] = le_cuisine.fit_transform(df["CuisineType"])
df["Segment_enc"]     = le_segment.fit_transform(df["Segment"])
df["Subregion_enc"]   = le_subregion.fit_transform(df["Subregion"])

# Save encoders
joblib.dump(le_cuisine,   "outputs/models/le_cuisine.pkl")
joblib.dump(le_segment,   "outputs/models/le_segment.pkl")
joblib.dump(le_subregion, "outputs/models/le_subregion.pkl")

# ── Feature matrix ─────────────────────────────────────────────────────────
FEATURES = [
    # Growth & demand
    "GrowthFactor", "Scale", "AOV", "MonthlyOrders",
    # Channel mix
    "InStoreShare", "UE_share", "DD_share", "SD_share",
    "ChannelBalance", "AggregatorShare",
    # Cost & efficiency
    "COGSRate", "OPEXRate", "CommissionRate",
    "DeliveryCostPerOrder", "CostPressure",
    # Logistics & reach
    "DeliveryRadiusKM", "DeliveryEfficiency",
    # Revenue quality
    "ProfitMargin", "RevenueQuality", "TotalRevenue", "TotalNetProfit",
    # Identity (encoded)
    "CuisineType_enc", "Segment_enc", "Subregion_enc",
]

X = df[FEATURES].copy()

# ── Log-transform right-skewed columns ────────────────────────────────────
skewed = ["Scale", "MonthlyOrders", "TotalRevenue", "TotalNetProfit",
          "SD_DeliveryTotalCost", "RevenueQuality"]
for col in skewed:
    if col in X.columns:
        X[col] = np.log1p(X[col].clip(lower=0))

# ── Standard scale ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES, index=df.index)

joblib.dump(scaler, "outputs/models/scaler.pkl")
X_scaled_df.to_csv("outputs/X_scaled.csv", index=False)
df.to_csv("outputs/enriched_data.csv", index=False)   # re-save with new cols

print(f"✅  Feature matrix shape: {X_scaled_df.shape}")
print(f"Features used:\n  {FEATURES}")
print("\nSTEP 2 COMPLETE. Run step3_dimensionality.py next.\n")
