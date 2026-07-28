"""
Streamlit Dashboard — Buyer Segmentation & Investment Profiling
Parcl Co. — Real Estate Market Intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml_pipeline import run_pipeline

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Parcl — Buyer Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLUSTER_COLORS = {
    "Global Investors": "#636EFA",
    "First-Time Buyers": "#EF553B",
    "Corporate Buyers": "#00CC96",
    "Luxury Investors": "#AB63FA",
}

# ─────────────────────────────────────────────
# LOAD DATA (CACHED PER K)
#   Cached on n_clusters, so every distinct K the user
#   picks on the slider is computed once and reused —
#   no separate "re-run" button needed, and the slider
#   takes effect immediately instead of silently doing
#   nothing until a button is clicked.
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Running ML pipeline...")
def load_data(n_clusters: int):
    return run_pipeline(n_clusters=n_clusters)

# ─────────────────────────────────────────────
# SIDEBAR — CLUSTERING CONTROL + FILTERS
# ─────────────────────────────────────────────
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1077/1077063.png", width=60)

st.sidebar.title("⚙️ Clustering")
n_clusters = st.sidebar.slider(
    "Number of Clusters (K)", 2, 8, 4,
    help="Applies immediately. K=4 maps to the four named Parcl buyer "
         "segments; other values fall back to auto-generated segment names.",
)

result = load_data(n_clusters)
df_full = result["df"]
properties = result["properties"]
elbow_data = result["elbow"]
silhouette_data = result["silhouette"]
km_silhouette = result["km_silhouette"]
cluster_summary = result["cluster_summary"]

if n_clusters != 4:
    st.sidebar.caption(
        "ℹ️ K≠4, so segments are auto-named by dominant trait instead of "
        "the four canonical Parcl labels."
    )

st.sidebar.title("🔍 Filter Buyers")

all_countries = sorted(df_full["country"].dropna().unique())
all_regions = sorted(df_full["region"].dropna().unique())
all_purposes = sorted(df_full["acquisition_purpose"].dropna().unique())
all_types = sorted(df_full["client_type"].dropna().unique())
all_segments = sorted(df_full["cluster_label"].dropna().unique())

sel_countries = st.sidebar.multiselect("Country", all_countries, default=all_countries)
sel_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)
sel_purposes = st.sidebar.multiselect("Acquisition Purpose", all_purposes, default=all_purposes)
sel_types = st.sidebar.multiselect("Client Type", all_types, default=all_types)
sel_segments = st.sidebar.multiselect(
    "Buyer Segment", all_segments, default=all_segments,
    key=f"segment_filter_k{n_clusters}",  # segment names change with K, so a
                                           # fresh widget instance is needed
                                           # per K to avoid stale selections
)

# Apply filters
mask = (
    df_full["country"].isin(sel_countries) &
    df_full["region"].isin(sel_regions) &
    df_full["acquisition_purpose"].isin(sel_purposes) &
    df_full["client_type"].isin(sel_types) &
    df_full["cluster_label"].isin(sel_segments)
)
df = df_full[mask].copy()

# Dynamic color map — extends the four canonical colors with a palette
# cycle for any auto-generated segment names when K != 4, so charts never
# fall back to an undifferentiated default color for unmapped segments.
def _build_color_map(segments):
    palette = px.colors.qualitative.Plotly
    cmap, i = {}, 0
    for seg in segments:
        if seg in CLUSTER_COLORS:
            cmap[seg] = CLUSTER_COLORS[seg]
        else:
            cmap[seg] = palette[i % len(palette)]
            i += 1
    return cmap

active_color_map = _build_color_map(df_full["cluster_label"].unique())

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#636EFA;'>🏢 Parcl — Buyer Segmentation & Investment Intelligence</h1>",
    unsafe_allow_html=True,
)
st.markdown("AI-driven buyer clustering using K-Means & Hierarchical algorithms on real estate transaction data.")
st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Segmentation Overview",
    "💹 Investor Behavior",
    "🗺️ Geographic Analysis",
    "🔬 Segment Insights",
    "⚙️ ML Model Evaluation",
])

# ══════════════════════════════════════════════
# TAB 1 — BUYER SEGMENTATION OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Buyer Segmentation Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Buyers", len(df))
    c2.metric("Unique Segments", df["cluster_label"].nunique())
    c3.metric("Avg Satisfaction", f"{df['satisfaction_score'].mean():.2f} / 5")
    c4.metric("Silhouette Score", f"{km_silhouette:.3f}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Cluster Distribution**")
        seg_count = df["cluster_label"].value_counts().reset_index()
        seg_count.columns = ["Segment", "Count"]
        fig_pie = px.pie(
            seg_count, values="Count", names="Segment",
            color="Segment", color_discrete_map=active_color_map,
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, width='stretch')

    with col_right:
        st.markdown("**Segment Bar Count**")
        fig_bar = px.bar(
            seg_count.sort_values("Count", ascending=True),
            x="Count", y="Segment", orientation="h",
            color="Segment", color_discrete_map=active_color_map,
        )
        fig_bar.update_layout(showlegend=False, yaxis_title="")
        st.plotly_chart(fig_bar, width='stretch')

    st.divider()
    st.markdown("**PCA 2D Cluster Scatter (K-Means)**")
    fig_scatter = px.scatter(
        df, x="pca_x", y="pca_y",
        color="cluster_label",
        color_discrete_map=active_color_map,
        hover_data=["client_id", "country", "region", "acquisition_purpose", "satisfaction_score"],
        labels={"pca_x": "PCA Component 1", "pca_y": "PCA Component 2"},
        opacity=0.75,
        height=450,
    )
    fig_scatter.update_traces(marker=dict(size=6))
    st.plotly_chart(fig_scatter, width='stretch')

    st.divider()
    st.markdown("**K-Means vs Hierarchical Cluster Agreement**")
    agree = (df["cluster"] == df["hc_cluster"]).mean() * 100
    st.info(f"Cluster label agreement between K-Means and Hierarchical Clustering: **{agree:.1f}%**")

    fig_hc = px.scatter(
        df, x="pca_x", y="pca_y",
        color=df["hc_cluster"].astype(str),
        title="Hierarchical Clustering (2D PCA)",
        labels={"pca_x": "PCA Component 1", "pca_y": "PCA Component 2", "color": "HC Cluster"},
        opacity=0.7, height=380,
    )
    st.plotly_chart(fig_hc, width='stretch')


# ══════════════════════════════════════════════
# TAB 2 — INVESTOR BEHAVIOR DASHBOARD
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Investor Behavior Dashboard")

    # KPIs by segment
    seg_stats = df.groupby("cluster_label").agg(
        avg_investment=("total_investment", "mean"),
        avg_properties=("num_properties", "mean"),
        pct_loan=("loan_applied_enc", "mean"),
        pct_investment_purpose=("is_investment", "mean"),
        avg_satisfaction=("satisfaction_score", "mean"),
        avg_age=("age", "mean"),
        count=("client_id", "count"),
    ).reset_index()

    st.dataframe(
        seg_stats.style.format({
            "avg_investment": "${:,.0f}",
            "avg_properties": "{:.1f}",
            "pct_loan": "{:.0%}",
            "pct_investment_purpose": "{:.0%}",
            "avg_satisfaction": "{:.2f}",
            "avg_age": "{:.0f}",
        }),
        width='stretch',
        height=200,
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Avg Total Investment by Segment**")
        fig_inv = px.bar(
            seg_stats.sort_values("avg_investment"),
            x="avg_investment", y="cluster_label",
            orientation="h", color="cluster_label",
            color_discrete_map=active_color_map,
            labels={"avg_investment": "Avg Total Investment ($)", "cluster_label": ""},
            text_auto=".2s",
        )
        fig_inv.update_layout(showlegend=False)
        st.plotly_chart(fig_inv, width='stretch')

    with col2:
        st.markdown("**Loan Application Rate by Segment**")
        fig_loan = px.bar(
            seg_stats, x="cluster_label", y="pct_loan",
            color="cluster_label", color_discrete_map=active_color_map,
            labels={"pct_loan": "Loan Rate", "cluster_label": ""},
            text_auto=".0%",
        )
        fig_loan.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(fig_loan, width='stretch')

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Acquisition Purpose Distribution**")
        purpose_seg = df.groupby(["cluster_label", "acquisition_purpose"]).size().reset_index(name="count")
        fig_purpose = px.bar(
            purpose_seg, x="cluster_label", y="count",
            color="acquisition_purpose", barmode="group",
            labels={"cluster_label": "Segment", "count": "Count"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_purpose.update_layout(legend_title="Purpose")
        st.plotly_chart(fig_purpose, width='stretch')

    with col4:
        st.markdown("**Referral Channel Distribution by Segment**")
        ref_seg = df.groupby(["cluster_label", "referral_channel"]).size().reset_index(name="count")
        fig_ref = px.sunburst(
            ref_seg, path=["cluster_label", "referral_channel"],
            values="count", color="cluster_label",
            color_discrete_map=active_color_map,
        )
        st.plotly_chart(fig_ref, width='stretch')

    st.divider()
    st.markdown("**Satisfaction Score Distribution by Segment**")
    fig_box = px.box(
        df, x="cluster_label", y="satisfaction_score",
        color="cluster_label", color_discrete_map=active_color_map,
        points="outliers",
        labels={"cluster_label": "Segment", "satisfaction_score": "Satisfaction Score"},
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, width='stretch')

    st.markdown("**Age Distribution by Segment**")
    fig_age = px.violin(
        df, x="cluster_label", y="age",
        color="cluster_label", color_discrete_map=active_color_map,
        box=True, points="outliers",
        labels={"cluster_label": "Segment", "age": "Age"},
    )
    fig_age.update_layout(showlegend=False)
    st.plotly_chart(fig_age, width='stretch')


# ══════════════════════════════════════════════
# TAB 3 — GEOGRAPHIC BUYER ANALYSIS
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Geographic Buyer Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Buyers by Country**")
        country_count = df.groupby(["country", "cluster_label"]).size().reset_index(name="count")
        fig_country = px.bar(
            country_count, x="count", y="country",
            orientation="h", color="cluster_label",
            color_discrete_map=active_color_map,
            labels={"count": "Buyers", "country": "Country"},
        )
        fig_country.update_layout(legend_title="Segment", height=500)
        st.plotly_chart(fig_country, width='stretch')

    with col2:
        st.markdown("**Buyers by Region**")
        region_count = df.groupby(["region", "cluster_label"]).size().reset_index(name="count")
        top_regions = region_count.groupby("region")["count"].sum().nlargest(20).index
        fig_region = px.bar(
            region_count[region_count["region"].isin(top_regions)],
            x="count", y="region", orientation="h",
            color="cluster_label", color_discrete_map=active_color_map,
            labels={"count": "Buyers", "region": "Region"},
        )
        fig_region.update_layout(legend_title="Segment", height=500)
        st.plotly_chart(fig_region, width='stretch')

    st.divider()
    st.markdown("**Choropleth — Total Buyers by Country**")
    country_total = df.groupby("country").agg(
        total_buyers=("client_id", "count"),
        avg_investment=("total_investment", "mean"),
        dominant_segment=("cluster_label", lambda x: x.mode()[0]),
    ).reset_index()

    fig_map = px.choropleth(
        country_total, locations="country",
        locationmode="country names",
        color="total_buyers",
        hover_data=["avg_investment", "dominant_segment"],
        color_continuous_scale="Blues",
        labels={"total_buyers": "Total Buyers"},
        height=450,
    )
    fig_map.update_layout(geo=dict(showframe=False, showcoastlines=True))
    st.plotly_chart(fig_map, width='stretch')

    st.divider()
    st.markdown("**Investment Heat by Country & Segment**")
    pivot = df.pivot_table(
        index="country", columns="cluster_label",
        values="total_investment", aggfunc="mean", fill_value=0
    ).reset_index()

    pivot_long = pivot.melt(id_vars="country", var_name="Segment", value_name="Avg Investment")
    top_c = df["country"].value_counts().nlargest(15).index
    fig_heat = px.density_heatmap(
        pivot_long[pivot_long["country"].isin(top_c)],
        x="Segment", y="country", z="Avg Investment",
        color_continuous_scale="Viridis",
        labels={"Avg Investment": "Avg Investment ($)"},
        height=500,
    )
    st.plotly_chart(fig_heat, width='stretch')


# ══════════════════════════════════════════════
# TAB 4 — SEGMENT INSIGHTS PANEL
# ══════════════════════════════════════════════
with tab4:
    st.subheader("Segment Insights Panel")

    selected_seg = st.selectbox("Select Buyer Segment", sorted(df["cluster_label"].unique()))
    seg_df = df[df["cluster_label"] == selected_seg]

    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    kc1.metric("Buyers in Segment", len(seg_df))
    kc2.metric("Avg Age", f"{seg_df['age'].mean():.0f}")
    kc3.metric("Avg Satisfaction", f"{seg_df['satisfaction_score'].mean():.2f}")
    avg_inv = seg_df['total_investment'].mean()
    kc4.metric("Avg Investment", f"${avg_inv/1000:.0f}K" if avg_inv >= 1000 else f"${avg_inv:.0f}")
    kc5.metric("Loan Rate", f"{seg_df['loan_applied_enc'].mean():.0%}")

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Gender Split**")
        gender_counts = seg_df["gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        gender_counts["Gender"] = gender_counts["Gender"].map({"M": "Male", "F": "Female"}).fillna("Other")
        fig_g = px.pie(gender_counts, values="Count", names="Gender", hole=0.3,
                       color_discrete_sequence=["#636EFA", "#EF553B"])
        st.plotly_chart(fig_g, width='stretch')

    with col2:
        st.markdown("**Client Type**")
        type_counts = seg_df["client_type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig_t = px.pie(type_counts, values="Count", names="Type", hole=0.3,
                       color_discrete_sequence=["#00CC96", "#AB63FA"])
        st.plotly_chart(fig_t, width='stretch')

    with col3:
        st.markdown("**Referral Channel**")
        ref_counts = seg_df["referral_channel"].value_counts().reset_index()
        ref_counts.columns = ["Channel", "Count"]
        fig_r = px.bar(ref_counts, x="Count", y="Channel", orientation="h",
                       color="Channel", text_auto=True,
                       color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_r.update_layout(showlegend=False)
        st.plotly_chart(fig_r, width='stretch')

    st.divider()
    col4, col5 = st.columns(2)

    with col4:
        st.markdown("**Top Countries**")
        top_c_seg = seg_df["country"].value_counts().nlargest(10).reset_index()
        top_c_seg.columns = ["Country", "Count"]
        fig_c = px.bar(top_c_seg, x="Count", y="Country", orientation="h",
                       text_auto=True, color="Count", color_continuous_scale="Blues")
        fig_c.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_c, width='stretch')

    with col5:
        st.markdown("**Satisfaction Score Histogram**")
        fig_sat = px.histogram(
            seg_df, x="satisfaction_score", nbins=5,
            color_discrete_sequence=[active_color_map.get(selected_seg, "#636EFA")],
            labels={"satisfaction_score": "Satisfaction Score"},
        )
        st.plotly_chart(fig_sat, width='stretch')

    st.divider()
    st.markdown("**Property Transactions for this Segment**")
    sold_props = properties[properties["listing_status"] == "Sold"].copy()
    seg_client_ids = seg_df["client_id"].tolist()
    seg_props = sold_props[sold_props["client_ref"].isin(seg_client_ids)]

    if not seg_props.empty:
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Total Transactions", len(seg_props))
        pc2.metric("Avg Sale Price", f"${seg_props['sale_price'].mean():,.0f}")
        pc3.metric("Avg Floor Area", f"{seg_props['floor_area_sqft'].mean():,.0f} sqft")
        pc4.metric("Office Units Sold", int((seg_props["unit_category"] == "Office").sum()))

        col6, col7 = st.columns(2)
        with col6:
            st.markdown("**Unit Category Split**")
            uc = seg_props["unit_category"].value_counts().reset_index()
            uc.columns = ["Category", "Count"]
            fig_uc = px.pie(uc, values="Count", names="Category", hole=0.3)
            st.plotly_chart(fig_uc, width='stretch')

        with col7:
            st.markdown("**Sale Price Distribution**")
            fig_sp = px.histogram(
                seg_props, x="sale_price", nbins=30,
                color_discrete_sequence=[active_color_map.get(selected_seg, "#636EFA")],
                labels={"sale_price": "Sale Price ($)"},
            )
            st.plotly_chart(fig_sp, width='stretch')
    else:
        st.info("No direct property transactions for this segment in the current filter.")

    st.divider()
    st.markdown("**Descriptive Statistics — Numeric Features**")
    numeric_cols = ["age", "satisfaction_score", "num_properties",
                    "avg_sale_price", "total_investment", "avg_floor_area"]
    st.dataframe(
        seg_df[numeric_cols].describe().T.style.format("{:.2f}"),
        width='stretch',
    )

    st.divider()
    with st.expander("📋 Raw Segment Data"):
        display_cols = ["client_id", "client_type", "gender", "age", "country", "region",
                        "acquisition_purpose", "satisfaction_score", "loan_applied",
                        "referral_channel", "num_properties", "total_investment", "cluster_label"]
        st.dataframe(seg_df[display_cols].reset_index(drop=True), width='stretch')


# ══════════════════════════════════════════════
# TAB 5 — ML MODEL EVALUATION
# ══════════════════════════════════════════════
with tab5:
    st.subheader("ML Model Evaluation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Elbow Method — Optimal K**")
        elbow_df = pd.DataFrame(
            list(elbow_data.items()), columns=["K", "Inertia"]
        )
        fig_elbow = px.line(
            elbow_df, x="K", y="Inertia", markers=True,
            labels={"K": "Number of Clusters", "Inertia": "Inertia (WCSS)"},
        )
        fig_elbow.update_traces(line=dict(color="#636EFA", width=2), marker=dict(size=8))
        fig_elbow.add_vline(x=4, line_dash="dash", line_color="red",
                            annotation_text="Optimal K=4", annotation_position="top right")
        st.plotly_chart(fig_elbow, width='stretch')

    with col2:
        st.markdown("**Silhouette Score by K**")
        sil_df = pd.DataFrame(
            list(silhouette_data.items()), columns=["K", "Silhouette"]
        )
        fig_sil = px.line(
            sil_df, x="K", y="Silhouette", markers=True,
            labels={"K": "Number of Clusters", "Silhouette": "Silhouette Score"},
        )
        best_k = sil_df.loc[sil_df["Silhouette"].idxmax(), "K"]
        fig_sil.update_traces(line=dict(color="#00CC96", width=2), marker=dict(size=8))
        fig_sil.add_vline(x=best_k, line_dash="dash", line_color="orange",
                          annotation_text=f"Best K={best_k}", annotation_position="top right")
        st.plotly_chart(fig_sil, width='stretch')

    st.divider()
    st.markdown("**Feature Importance (Correlation with Cluster Assignment)**")
   
    from ml_pipeline import scale_features
    feature_cols = result["feature_cols"]
    X_scaled, _ = scale_features(df_full, feature_cols)
    corr_with_cluster = []
    for i, feat in enumerate(feature_cols):
        corr = np.corrcoef(X_scaled[:, i], df_full["cluster"])[0, 1]
        if np.isnan(corr):
            continue
        corr_with_cluster.append({"Feature": feat, "Correlation": abs(corr)})
    feat_df = pd.DataFrame(corr_with_cluster).sort_values("Correlation", ascending=True).tail(20)

    fig_feat = px.bar(
        feat_df, x="Correlation", y="Feature", orientation="h",
        color="Correlation", color_continuous_scale="Blues",
        labels={"Correlation": "|Correlation with Cluster|"},
    )
    fig_feat.update_layout(coloraxis_showscale=False, height=500)
    st.plotly_chart(fig_feat, width='stretch')
    st.caption("Top 20 features by absolute correlation with cluster assignment, "
               "including the one-hot encoded category columns.")

    st.divider()
    st.markdown("**Cluster Interpretation — Investment Purpose, Geography, Loan Behavior & Demographics**")
    st.dataframe(
        cluster_summary.style.format({
            "avg_investment": "${:,.0f}",
            "pct_company": "{:.0%}",
            "avg_num_properties": "{:.2f}",
            "avg_age": "{:.0f}",
            "pct_loan": "{:.0%}",
            "pct_investment_purpose": "{:.0%}",
            "avg_satisfaction": "{:.2f}",
            "country_diversity": "{:.2f}",
        }),
        width='stretch',
    )
    st.caption(
        "This is the table the pipeline actually scores clusters against to "
        "assign segment names — including geographic spread (`country_diversity`, "
        "`dominant_country`, `dominant_region`), so labels aren't based on "
        "investment/loan/demographic signals alone."
    )

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Cluster Summary Table**")
        summary_table = df_full.groupby("cluster_label").agg(
            Count=("client_id", "count"),
            Avg_Age=("age", "mean"),
            Avg_Satisfaction=("satisfaction_score", "mean"),
            Loan_Rate=("loan_applied_enc", "mean"),
            Investment_Rate=("is_investment", "mean"),
            Avg_Investment=("total_investment", "mean"),
            Avg_Properties=("num_properties", "mean"),
        ).round(2)
        st.dataframe(summary_table, width='stretch')

    with col4:
        st.markdown("**Radar Chart — Segment Profiles**")
        radar_cols = ["avg_age_norm", "avg_satisfaction_norm", "loan_rate",
                      "investment_rate", "avg_investment_norm", "avg_properties_norm"]

        radar_df = df_full.groupby("cluster_label").agg(
            avg_age=("age", "mean"),
            avg_satisfaction=("satisfaction_score", "mean"),
            loan_rate=("loan_applied_enc", "mean"),
            investment_rate=("is_investment", "mean"),
            avg_investment=("total_investment", "mean"),
            avg_properties=("num_properties", "mean"),
        )

        # Normalize 0-1
        for col in ["avg_age", "avg_satisfaction", "avg_investment", "avg_properties"]:
            mn, mx = radar_df[col].min(), radar_df[col].max()
            radar_df[col + "_norm"] = (radar_df[col] - mn) / (mx - mn + 1e-9)

        categories = ["Age", "Satisfaction", "Loan Rate",
                      "Investment Purpose", "Avg Investment", "Avg Properties"]

        fig_radar = go.Figure()
        for seg, row in radar_df.iterrows():
            vals = [
                row["avg_age_norm"], row["avg_satisfaction_norm"],
                row["loan_rate"], row["investment_rate"],
                row["avg_investment_norm"], row["avg_properties_norm"],
            ]
            vals_closed = vals + [vals[0]]
            cats_closed = categories + [categories[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats_closed, fill="toself",
                name=seg, opacity=0.65,
                line=dict(color=active_color_map.get(seg, "#636EFA")),
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=380,
        )
        st.plotly_chart(fig_radar, width='stretch')

    st.divider()
    st.markdown("**K-Means vs Hierarchical Cluster Comparison**")
    cross = pd.crosstab(
        df_full["cluster_label"],
        df_full["hc_cluster"].map(lambda x: f"HC-{x}"),
    )
    fig_cross = px.imshow(
        cross, text_auto=True, color_continuous_scale="Blues",
        labels={"x": "Hierarchical Cluster", "y": "K-Means Segment"},
        aspect="auto",
    )
    st.plotly_chart(fig_cross, width='stretch')

    st.info(
        f"**K-Means Silhouette Score: {km_silhouette:.4f}** — "
        "Values closer to 1 indicate well-separated clusters."
    )