# ================================================================
# TORCH: Factory-Level Risk Index — Myanmar Apparel Supply Chains
# Streamlit Dashboard
# ================================================================
# pip install streamlit plotly folium streamlit-folium

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------

st.set_page_config(
    page_title="Torch — Myanmar Apparel Risk Index",
    page_icon="🔦",
    layout="wide"
)

RISK_COLORS = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}

DIMENSIONS = [
    "child_labour", "forced_labour", "discrimination",
    "freedom_of_association", "working_hours",
    "compensation", "osh", "contracts"
]

DIM_LABELS = {
    "child_labour"          : "Child Labour (C138/C182)",
    "forced_labour"         : "Forced Labour (C29/C105)",
    "discrimination"        : "Discrimination (C100/C111)",
    "freedom_of_association": "Freedom of Association (C87/C98)",
    "working_hours"         : "Working Hours (C1)",
    "compensation"          : "Compensation (C95/C131)",
    "osh"                   : "Occupational Safety & Health (C155)",
    "contracts"             : "Contracts (C158)",
}

# ----------------------------------------------------------------
# Load data
# ----------------------------------------------------------------

@st.cache_data
def load_data():
    factories = pd.read_csv("stage5_factory_risk_index.csv")
    articles  = pd.read_csv("stage4_news_predictions.csv")
    return factories, articles

factories, articles = load_data()

# Prevalence columns
prev_cols = [f"prevalence_{d}" for d in DIMENSIONS]

# Articles factory reference
articles["factory_ref"] = articles["osh_factory_matched"].fillna(articles["bhrrc_factory_matched"])

# ----------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------

st.sidebar.image("https://img.icons8.com/emoji/96/flashlight-emoji.png", width=60)
st.sidebar.title("Torch")
st.sidebar.caption("Factory-Level Risk Index\nMyanmar Apparel Supply Chains")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Risk Map", "Risk Ranking", "Factory Profile", "Article Explorer"]
)

st.sidebar.divider()
st.sidebar.caption(
    "Data sources: Myanmar Labour News, "
    "Business & Human Rights Resource Centre, "
    "Open Supply Hub\n\n"
    "ILO framework: Better Work Global CAT (ILO & IFC, 2025)"
)

# ================================================================
# PAGE 1 — Overview
# ================================================================

if page == "Overview":
    st.title("🔦 Torch: Myanmar Apparel Supply Chain Risk Index")
    st.markdown(
        "Torch leverages text mining and machine learning to generate "
        "factory-level labour risk signals from publicly available complaint "
        "narratives, benchmarked against ILO core conventions following the "
        "**Better Work Global Compliance Assessment Tool** (ILO & IFC, 2025)."
    )
    st.divider()

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Factories Assessed", len(factories))
    col2.metric("Articles Analysed", len(articles))
    col3.metric("High Risk Factories", (factories["risk_band"] == "High").sum())
    col4.metric("ILO Dimensions Tracked", len(DIMENSIONS))

    st.divider()

    # Risk band distribution
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Risk Band Distribution")
        band_counts = factories["risk_band"].value_counts().reindex(["High", "Medium", "Low"])
        fig_band = px.bar(
            x=band_counts.index,
            y=band_counts.values,
            color=band_counts.index,
            color_discrete_map=RISK_COLORS,
            labels={"x": "Risk Band", "y": "Number of Factories"},
        )
        fig_band.update_layout(showlegend=False)
        st.plotly_chart(fig_band, use_container_width=True)

    with col_b:
        st.subheader("Average Risk Prevalence by ILO Dimension")
        avg_prev = factories[prev_cols].mean().rename(
            lambda x: DIM_LABELS.get(x.replace("prevalence_", ""), x)
        ).sort_values(ascending=True)
        fig_dim = px.bar(
            x=avg_prev.values,
            y=avg_prev.index,
            orientation="h",
            labels={"x": "Average Prevalence", "y": ""},
            color=avg_prev.values,
            color_continuous_scale="OrRd",
        )
        fig_dim.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_dim, use_container_width=True)

    st.divider()

    # Articles over time
    st.subheader("Complaint Articles Over Time")
    articles["date"] = pd.to_datetime(articles["date"], errors="coerce")
    articles_by_month = (
        articles.dropna(subset=["date"])
        .groupby(articles["date"].dt.to_period("M"))
        .size()
        .reset_index()
    )
    articles_by_month["date"] = articles_by_month["date"].astype(str)
    articles_by_month.columns = ["Month", "Articles"]
    fig_time = px.line(
        articles_by_month, x="Month", y="Articles",
        labels={"Month": "", "Articles": "Number of Articles"}
    )
    st.plotly_chart(fig_time, use_container_width=True)

    st.info(
        "⚠️ **Disclaimer:** Risk scores are signals derived from publicly available "
        "complaint narratives and are not legal judgements. They reflect reported "
        "concerns and should be used alongside formal audits and worker dialogue."
    )

# ================================================================
# PAGE 2 — Risk Map
# ================================================================

elif page == "Risk Map":
    st.title("🗺️ Factory Risk Map")
    st.caption("Factories are plotted using coordinates from Open Supply Hub. "
               "Circle size reflects composite risk score.")

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        band_filter = st.multiselect(
            "Filter by Risk Band",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
    with col2:
        min_articles = st.slider("Minimum article count", 1, 20, 1)

    map_data = factories[
        (factories["risk_band"].isin(band_filter)) &
        (factories["article_count"] >= min_articles) &
        (factories["lat"].notna()) &
        (factories["lng"].notna())
    ]

    st.caption(f"Showing {len(map_data)} factories")

    # Build folium map centred on Myanmar
    m = folium.Map(location=[19.5, 96.5], zoom_start=6, tiles="CartoDB positron")

    for _, row in map_data.iterrows():
        color  = RISK_COLORS.get(row["risk_band"], "gray")
        radius = max(5, row["composite_score"] * 20)
        popup_html = f"""
            <b>{row['factory_name']}</b><br>
            Risk Band: <b style='color:{color}'>{row['risk_band']}</b><br>
            Composite Score: {row['composite_score']}<br>
            Articles: {int(row['article_count'])}<br>
            Rank: #{int(row['rank'])}
        """
        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=row["factory_name"]
        ).add_to(m)

    st_folium(m, width=None, height=550)

# ================================================================
# PAGE 3 — Risk Ranking Table
# ================================================================

elif page == "Risk Ranking":
    st.title("📊 Factory Risk Ranking")

    col1, col2, col3 = st.columns(3)
    with col1:
        band_filter = st.multiselect(
            "Risk Band", ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
    with col2:
        min_score = st.slider("Minimum composite score", 0.0, 1.0, 0.0, 0.01)
    with col3:
        min_art = st.slider("Minimum articles", 1, 20, 1)

    filtered = factories[
        (factories["risk_band"].isin(band_filter)) &
        (factories["composite_score"] >= min_score) &
        (factories["article_count"] >= min_art)
    ].copy()

    # Display columns
    display = filtered[[
        "rank", "factory_name", "composite_score", "risk_band",
        "article_count", "confidence"
    ]].rename(columns={
        "rank"            : "Rank",
        "factory_name"    : "Factory",
        "composite_score" : "Composite Score",
        "risk_band"       : "Risk Band",
        "article_count"   : "Articles",
        "confidence"      : "Confidence"
    })

    def color_band(val):
        colors = {"High": "background-color:#ffd7d7", 
                  "Medium": "background-color:#fff3cd",
                  "Low": "background-color:#d4edda"}
        return colors.get(val, "")

    st.dataframe(
        display.style.applymap(color_band, subset=["Risk Band"]),
        use_container_width=True,
        height=500
    )

    st.download_button(
        "Download as CSV",
        filtered.to_csv(index=False).encode("utf-8-sig"),
        "torch_risk_ranking.csv",
        "text/csv"
    )

# ================================================================
# PAGE 4 — Factory Profile
# ================================================================

elif page == "Factory Profile":
    st.title("🏭 Factory Profile")

    factory_names = factories["factory_name"].sort_values().tolist()
    selected = st.selectbox("Select a factory", factory_names)

    row = factories[factories["factory_name"] == selected].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rank", f"#{int(row['rank'])} / {len(factories)}")
    col2.metric("Composite Score", f"{row['composite_score']:.4f}")
    band_color = RISK_COLORS.get(row["risk_band"], "gray")
    col3.metric("Risk Band", row["risk_band"])
    col4.metric("Articles", int(row["article_count"]))

    st.divider()

    # Risk dimension radar chart
    st.subheader("Risk Dimension Breakdown")
    st.caption("Based on Better Work CAT clusters (ILO & IFC, 2025)")

    dims   = DIMENSIONS
    labels = [DIM_LABELS[d] for d in dims]
    values = [row.get(f"prevalence_{d}", 0) for d in dims]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor=f"rgba(214, 39, 40, 0.2)" if row["risk_band"] == "High"
                  else "rgba(255, 127, 14, 0.2)" if row["risk_band"] == "Medium"
                  else "rgba(44, 160, 44, 0.2)",
        line=dict(color=band_color),
        name=selected
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=420
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # OSH metadata
    st.divider()
    st.subheader("Factory Metadata")
    meta_cols = ["address", "number_of_workers", "is_closed", "os_id"]
    meta = {c: row.get(c, "N/A") for c in meta_cols}
    meta_df = pd.DataFrame.from_dict(meta, orient="index", columns=["Value"])
    meta_df.index = ["Address", "Number of Workers", "Is Closed", "OSH ID"]
    st.table(meta_df)

# ================================================================
# PAGE 5 — Article Explorer
# ================================================================

elif page == "Article Explorer":
    st.title("📰 Article Explorer")
    st.caption("Browse source complaint articles linked to each factory.")

    factory_names = sorted(articles["factory_ref"].dropna().unique().tolist())
    selected = st.selectbox("Select a factory", factory_names)

    factory_articles = articles[articles["factory_ref"] == selected].copy()
    factory_articles["date"] = pd.to_datetime(factory_articles["date"], errors="coerce")
    factory_articles = factory_articles.sort_values("date", ascending=False)

    st.caption(f"{len(factory_articles)} articles found for **{selected}**")

    for _, art in factory_articles.iterrows():
        pred_flags = [
            DIM_LABELS[d].split(" (")[0]
            for d in DIMENSIONS
            if art.get(f"pred_{d}", 0) == 1
        ]
        with st.expander(f"📄 {art['title']}  —  {str(art['date'])[:10]}"):
            st.markdown(f"**URL:** [{art['URL']}]({art['URL']})")
            if pred_flags:
                st.markdown("**Predicted Risk Dimensions:** " +
                            " · ".join([f"`{f}`" for f in pred_flags]))
            st.markdown("**Article Content:**")
            st.write(str(art.get("Content", "No content available"))[:2000] + "...")