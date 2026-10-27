"""
STREAMLIT DASHBOARD – SkyCity Auckland Restaurant Growth Intelligence
Run from project root: streamlit run streamlit_app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyCity Restaurant Growth Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Try final results first, else enriched_data
    if os.path.exists("outputs/gpi/restaurant_classifications.csv"):
        cls = pd.read_csv("outputs/gpi/restaurant_classifications.csv")
        full = pd.read_csv("outputs/enriched_data.csv")
        # Merge additional columns from full into classifications
        extra_cols = [c for c in full.columns if c not in cls.columns]
        df = cls.merge(full[["RestaurantID"] + extra_cols], on="RestaurantID", how="left")
    elif os.path.exists("outputs/enriched_data.csv"):
        df = pd.read_csv("outputs/enriched_data.csv")
    else:
        df = pd.read_csv("data/SkyCity_Auckland_Restaurants___Bars.csv")
    return df

df = load_data()

# ── Ensure required columns exist ──────────────────────────────────────────
if "GPI" not in df.columns:
    df["GPI"] = 0.5
if "Archetype" not in df.columns:
    df["Archetype"] = "Unclassified"
if "StrategicRec" not in df.columns:
    df["StrategicRec"] = "Run pipeline first"
if "TotalRevenue" not in df.columns:
    df["TotalRevenue"] = (df.get("InStoreRevenue",0) + df.get("UberEatsRevenue",0)
                         + df.get("DoorDashRevenue",0) + df.get("SelfDeliveryRevenue",0))
if "ProfitMargin" not in df.columns:
    df["TotalNetProfit"] = (df.get("InStoreNetProfit",0) + df.get("UberEatsNetProfit",0)
                           + df.get("DoorDashNetProfit",0) + df.get("SelfDeliveryNetProfit",0))
    df["ProfitMargin"] = df["TotalNetProfit"] / (df["TotalRevenue"] + 1e-9)
if "AggregatorShare" not in df.columns:
    df["AggregatorShare"] = df.get("UE_share",0) + df.get("DD_share",0)
if "CostPressure" not in df.columns:
    df["CostPressure"] = df.get("COGSRate",0) + df.get("OPEXRate",0)

# ── SIDEBAR FILTERS ────────────────────────────────────────────────────────
st.sidebar.image("https://via.placeholder.com/200x60?text=SkyCity+Auckland", width=200)
st.sidebar.title("🔍 Filters")

subregions  = ["All"] + sorted(df["Subregion"].dropna().unique().tolist())
cuisines    = ["All"] + sorted(df["CuisineType"].dropna().unique().tolist())
segments    = ["All"] + sorted(df["Segment"].dropna().unique().tolist())
archetypes  = ["All"] + sorted(df["Archetype"].dropna().unique().tolist())

sel_subregion = st.sidebar.selectbox("Subregion",  subregions)
sel_cuisine   = st.sidebar.selectbox("Cuisine Type", cuisines)
sel_segment   = st.sidebar.selectbox("Segment",    segments)
sel_archetype = st.sidebar.selectbox("Archetype",  archetypes)
gpi_range     = st.sidebar.slider("GPI Score Range", 0.0, 1.0, (0.0, 1.0), 0.01)

# Apply filters
mask = pd.Series(True, index=df.index)
if sel_subregion != "All": mask &= df["Subregion"] == sel_subregion
if sel_cuisine   != "All": mask &= df["CuisineType"] == sel_cuisine
if sel_segment   != "All": mask &= df["Segment"] == sel_segment
if sel_archetype != "All": mask &= df["Archetype"] == sel_archetype
mask &= (df["GPI"] >= gpi_range[0]) & (df["GPI"] <= gpi_range[1])

fdf = df[mask].copy()

# ── HEADER ─────────────────────────────────────────────────────────────────
st.title("🍽️ SkyCity Auckland – Restaurant Growth Intelligence Dashboard")
st.markdown(f"**{len(fdf)}** restaurants shown (from {len(df)} total)")

# ── KPI TILES ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Avg GPI Score",      f"{fdf['GPI'].mean():.3f}" if "GPI" in fdf.columns else "—")
c2.metric("Avg Growth Factor",  f"{fdf['GrowthFactor'].mean():.4f}")
c3.metric("Avg Profit Margin",  f"{fdf['ProfitMargin'].mean()*100:.1f}%")
c4.metric("Avg Monthly Orders", f"{fdf['MonthlyOrders'].mean():,.0f}")
c5.metric("Avg Revenue ($)",    f"${fdf['TotalRevenue'].mean():,.0f}")

st.divider()

# ── TAB LAYOUT ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Cluster Map",
    "🏆 Growth Scorecards",
    "📡 Feature Radar",
    "🗺️ Strategy Panel",
    "📋 Data Explorer"
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 – CLUSTER MAP
# ────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Restaurant Cluster Map")

    col_a, col_b = st.columns([2, 1])

    with col_a:
        if "PC1" in df.columns:
            fig = px.scatter(
                fdf, x="PC1", y="PC2",
                color="Archetype",
                size="GPI", size_max=12,
                hover_data=["RestaurantName","CuisineType","Subregion","GPI","StrategicRec"],
                title="PCA Cluster Scatter (size = GPI)",
                template="plotly_white"
            )
        else:
            fig = px.scatter(
                fdf, x="GrowthFactor", y="ProfitMargin",
                color="Archetype",
                size="MonthlyOrders", size_max=15,
                hover_data=["RestaurantName","CuisineType","Subregion","GPI"],
                title="Growth Factor vs Profit Margin by Archetype",
                template="plotly_white"
            )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        archetype_counts = fdf["Archetype"].value_counts().reset_index()
        archetype_counts.columns = ["Archetype","Count"]
        fig2 = px.pie(archetype_counts, names="Archetype", values="Count",
                      title="Archetype Distribution", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    # GPI heatmap by Subregion × Cuisine
    st.subheader("Avg GPI Heatmap: Subregion × Cuisine")
    heatmap_data = fdf.groupby(["Subregion","CuisineType"])["GPI"].mean().unstack(fill_value=0)
    fig3 = px.imshow(heatmap_data, color_continuous_scale="RdYlGn",
                     labels=dict(color="Avg GPI"), title="Average GPI Score")
    st.plotly_chart(fig3, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 – GROWTH SCORECARDS
# ────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Growth Potential Scorecards")

    sort_col = st.selectbox("Sort by", ["GPI","GrowthFactor","ProfitMargin","MonthlyOrders","TotalRevenue"])
    top_n    = st.slider("Number of restaurants to show", 10, 100, 25)

    top_df = fdf.sort_values(sort_col, ascending=False).head(top_n)

    # Scorecard table
    show_cols = ["RestaurantName","CuisineType","Segment","Subregion",
                 "GPI","GPI_Band" if "GPI_Band" in fdf.columns else "Archetype",
                 "GrowthFactor","ProfitMargin","MonthlyOrders","StrategicRec"]
    show_cols = [c for c in show_cols if c in top_df.columns]

    st.dataframe(
        top_df[show_cols].style
            .background_gradient(subset=["GPI","GrowthFactor"], cmap="RdYlGn")
            .format({"GPI":"{:.3f}","GrowthFactor":"{:.4f}","ProfitMargin":"{:.2%}"}),
        use_container_width=True, height=420
    )

    # GPI bar chart top 20
    fig4 = px.bar(top_df.head(20), x="RestaurantName", y="GPI",
                  color="Archetype", title="Top 20 Restaurants by GPI",
                  template="plotly_white")
    fig4.update_xaxes(tickangle=45)
    st.plotly_chart(fig4, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 – FEATURE RADAR CHARTS
# ────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Feature Contribution Radar Charts")

    def minmax_global(s, ref_series):
        mn, mx = ref_series.min(), ref_series.max()
        return (s - mn) / (mx - mn + 1e-9)

    radar_features = ["GrowthFactor","AOV","MonthlyOrders","ProfitMargin",
                      "DeliveryRadiusKM","AggregatorShare","CostPressure"]
    radar_labels   = ["Growth","AOV","Volume","Margin","Delivery Reach","Aggr Dep","Cost Pressure"]

    # Invert cost metrics (lower = better)
    invert = {"AggregatorShare","CostPressure"}

    col1, col2 = st.columns(2)

    with col1:
        r1_name = st.selectbox("Restaurant A", fdf["RestaurantName"].unique(), index=0)
    with col2:
        r2_name = st.selectbox("Restaurant B", fdf["RestaurantName"].unique(),
                                index=min(1, len(fdf)-1))

    def get_radar_vals(name):
        row = fdf[fdf["RestaurantName"] == name].iloc[0]
        vals = []
        for feat in radar_features:
            v = minmax_global(row[feat], df[feat])
            if feat in invert: v = 1 - v
            vals.append(round(float(v), 3))
        return vals

    vals_a = get_radar_vals(r1_name)
    vals_b = get_radar_vals(r2_name)

    fig5 = go.Figure()
    for name, vals, color in [(r1_name, vals_a, "royalblue"), (r2_name, vals_b, "coral")]:
        fig5.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself", name=name,
            line=dict(color=color), opacity=0.6
        ))
    fig5.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        title=f"Radar Comparison: {r1_name} vs {r2_name}",
        template="plotly_white"
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Archetype-level radar
    st.subheader("Cluster Archetype Average Profiles")
    archetype_profiles = fdf.groupby("Archetype")[radar_features].mean()

    fig6 = go.Figure()
    for arch in archetype_profiles.index:
        row = archetype_profiles.loc[arch]
        vals = []
        for feat in radar_features:
            v = minmax_global(row[feat], df[feat])
            if feat in invert: v = 1 - v
            vals.append(round(float(v), 3))
        fig6.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself", name=arch, opacity=0.4
        ))
    fig6.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        title="All Archetypes – Average Radar Profile",
        template="plotly_white"
    )
    st.plotly_chart(fig6, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 – STRATEGY PANEL
# ────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🗺️ Strategy Recommendation Panel")

    rec_counts = fdf["StrategicRec"].value_counts().reset_index()
    rec_counts.columns = ["Recommendation","Count"]

    col_x, col_y = st.columns([1,2])
    with col_x:
        fig7 = px.bar(rec_counts, y="Recommendation", x="Count",
                      orientation="h", color="Count",
                      color_continuous_scale="RdYlGn", title="Recommendation Distribution")
        st.plotly_chart(fig7, use_container_width=True)

    with col_y:
        fig8 = px.scatter(
            fdf, x="CostPressure", y="GPI",
            color="StrategicRec", size="MonthlyOrders", size_max=12,
            hover_data=["RestaurantName","Archetype"],
            title="Cost Pressure vs GPI (coloured by Strategy)",
            template="plotly_white"
        )
        st.plotly_chart(fig8, use_container_width=True)

    # Aggregator dependence map
    st.subheader("Aggregator Dependence vs Profit Margin")
    fig9 = px.scatter(
        fdf, x="AggregatorShare", y="ProfitMargin",
        color="Archetype", size="GPI", size_max=14,
        facet_col="Subregion" if fdf["Subregion"].nunique() <= 6 else None,
        hover_data=["RestaurantName","StrategicRec"],
        template="plotly_white",
        title="Aggregator Dependence vs Profit Margin"
    )
    st.plotly_chart(fig9, use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 5 – DATA EXPLORER
# ────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("📋 Full Data Explorer")
    search = st.text_input("Search restaurant name")
    disp_df = fdf[fdf["RestaurantName"].str.contains(search, case=False, na=False)] if search else fdf

    st.dataframe(disp_df.reset_index(drop=True), use_container_width=True, height=500)
    csv = disp_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered results", csv,
                       "skycity_filtered.csv", "text/csv")

    st.subheader("Summary Statistics")
    num_cols = ["GPI","GrowthFactor","AOV","MonthlyOrders","ProfitMargin",
                "CostPressure","AggregatorShare","DeliveryRadiusKM","TotalRevenue"]
    num_cols = [c for c in num_cols if c in fdf.columns]
    st.dataframe(fdf[num_cols].describe().round(3), use_container_width=True)
