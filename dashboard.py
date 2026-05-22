import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Torch — Myanmar Apparel Risk Index",
    page_icon = "torch_logo.png",
    layout="wide"
)

RISK_COLORS = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}

DIMENSIONS = [
    "child_labour", "forced_labour", "discrimination",
    "freedom_of_association", "working_hours",
    "compensation", "osh", "contracts"
]

DIM_LABELS = {
    "child_labour":          "Child Labour (C138/C182)",
    "forced_labour":         "Forced Labour (C29/C105)",
    "discrimination":        "Discrimination (C100/C111)",
    "freedom_of_association": "Freedom of Association (C87/C98)",
    "working_hours":         "Working Hours (C1)",
    "compensation":          "Compensation (C95/C131)",
    "osh":                   "Occupational Safety & Health (C155)",
    "contracts":             "Contracts (C158)",
}

DISCLAIMER = (
    "⚠️ **Disclaimer:** Risk scores are indicators derived from publicly available "
    "complaint narratives and are not legal judgements. They reflect reported concerns "
    "and should be used alongside formal audits and direct worker consultation."
)


@st.cache_data
def load_data():
    factories = pd.read_csv("stage5_factory_risk_index.csv")
    articles  = pd.read_csv("stage4_news_predictions.csv")
    return factories, articles


factories, articles = load_data()

prev_cols = [f"prevalence_{d}" for d in DIMENSIONS]
articles["factory_ref"] = articles["osh_factory_matched"].fillna(
    articles["bhrrc_factory_matched"]
)

# Updated to use your local logo file
st.sidebar.image("torch_logo.png", width=60)
st.sidebar.title("Torch")
st.sidebar.caption("Factory-Level Risk Index\nMyanmar Apparel Supply Chains")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Risk Map", "Risk Ranking", "Factory Profile", "Article Explorer"]
)

st.sidebar.divider()
st.sidebar.caption(
    "Data sources: Myanmar Labour News, Business & Human Rights Resource Centre, "
    "Open Supply Hub\n\nILO framework: Better Work Global CAT (ILO & IFC, 2025)"
)


if page == "Overview":
    st.title("Torch: Myanmar Apparel Supply Chain Risk Index")
    st.markdown(
        "Torch uses text mining and machine learning to generate factory-level labour "
        "risk signals from publicly available complaint narratives, benchmarked against "
        "ILO core conventions via the **Better Work Global Compliance Assessment Tool** "
        "(ILO & IFC, 2025)."
    )
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Factories Assessed", len(factories))
    c2.metric("Articles Analysed", len(articles))
    c3.metric("High Risk Factories", (factories["risk_band"] == "High").sum())
    c4.metric("ILO Dimensions Tracked", len(DIMENSIONS))

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Risk Band Distribution")
        band_counts = factories["risk_band"].value_counts().reindex(["High", "Medium", "Low"])
        fig_band = px.bar(
            x=band_counts.index, y=band_counts.values,
            color=band_counts.index, color_discrete_map=RISK_COLORS,
            labels={"x": "Risk Band", "y": "Number of Factories"},
        )
        fig_band.update_layout(showlegend=False)
        st.plotly_chart(fig_band, use_container_width=True)

    with col_b:
        st.subheader("Average Risk Prevalence by ILO Dimension")
        avg_prev = (
            factories[prev_cols].mean()
            .rename(lambda x: DIM_LABELS.get(x.replace("prevalence_", ""), x))
            .sort_values(ascending=True)
        )
        fig_dim = px.bar(
            x=avg_prev.values, y=avg_prev.index, orientation="h",
            labels={"x": "Average Prevalence", "y": ""},
            color=avg_prev.values, color_continuous_scale="OrRd",
        )
        fig_dim.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_dim, use_container_width=True)

    st.divider()
    st.subheader("Complaint Articles Over Time")
    articles["date"] = pd.to_datetime(articles["date"], errors="coerce")
    by_month = (
        articles.dropna(subset=["date"])
        .groupby(articles["date"].dt.to_period("M"))
        .size()
        .reset_index()
    )
    by_month["date"] = by_month["date"].astype(str)
    by_month.columns = ["Month", "Articles"]
    fig_time = px.line(by_month, x="Month", y="Articles",
                       labels={"Month": "", "Articles": "Number of Articles"})
    st.plotly_chart(fig_time, use_container_width=True)

    st.info(DISCLAIMER)


elif page == "Risk Map":
    st.title("Factory Risk Map")
    st.caption(
        "Factories are plotted using coordinates from Open Supply Hub. "
        "Circle size reflects the composite risk score."
    )

    c1, c2 = st.columns(2)
    with c1:
        band_filter = st.multiselect(
            "Filter by Risk Band", ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
    with c2:
        min_articles = st.slider("Minimum article count", 1, 20, 1)

    map_data = factories[
        (factories["risk_band"].isin(band_filter)) &
        (factories["article_count"] >= min_articles) &
        (factories["lat"].notna()) &
        (factories["lng"].notna())
    ]
    st.caption(f"Showing {len(map_data)} factories")

    m = folium.Map(location=[19.5, 96.5], zoom_start=6, tiles="CartoDB positron")
    for _, row in map_data.iterrows():
        color  = RISK_COLORS.get(row["risk_band"], "gray")
        radius = max(5, row["composite_score"] * 20)
        popup  = (
            f"<b>{row['factory_name']}</b><br>"
            f"Risk Band: <b style='color:{color}'>{row['risk_band']}</b><br>"
            f"Composite Score: {row['composite_score']:.4f}<br>"
            f"Articles: {int(row['article_count'])}<br>"
            f"Rank: #{int(row['rank'])}"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=radius, color=color,
            fill=True, fill_color=color, fill_opacity=0.7,
            popup=folium.Popup(popup, max_width=250),
            tooltip=row["factory_name"]
        ).add_to(m)

    st_folium(m, width=None, height=550)
    st.info(DISCLAIMER)


elif page == "Risk Ranking":
    st.title("Factory Risk Ranking")

    c1, c2, c3 = st.columns(3)
    with c1:
        band_filter = st.multiselect(
            "Risk Band", ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
    with c2:
        min_score = st.slider("Minimum composite score", 0.0, 1.0, 0.0, 0.01)
    with c3:
        min_art = st.slider("Minimum articles", 1, 20, 1)

    filtered = factories[
        (factories["risk_band"].isin(band_filter)) &
        (factories["composite_score"] >= min_score) &
        (factories["article_count"] >= min_art)
    ].copy()

    display = filtered[[
        "rank", "factory_name", "composite_score",
        "risk_band", "article_count", "confidence"
    ]].rename(columns={
        "rank":            "Rank",
        "factory_name":    "Factory",
        "composite_score": "Composite Score",
        "risk_band":       "Risk Band",
        "article_count":   "Articles",
        "confidence":      "Confidence",
    })

    band_bg = {
        "High":   "background-color:#ffd7d7",
        "Medium": "background-color:#fff3cd",
        "Low":    "background-color:#d4edda",
    }

    def color_band(val):
        return band_bg.get(val, "")

    st.dataframe(
        display.style.map(color_band, subset=["Risk Band"]),
        use_container_width=True, height=500
    )
    st.download_button(
        "Download as CSV",
        filtered.to_csv(index=False).encode("utf-8-sig"),
        "torch_risk_ranking.csv", "text/csv"
    )
    st.info(DISCLAIMER)


elif page == "Factory Profile":
    st.title("Factory Profile")

    selected = st.selectbox(
        "Select a factory", factories["factory_name"].sort_values().tolist()
    )
    row = factories[factories["factory_name"] == selected].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rank",            f"#{int(row['rank'])} / {len(factories)}")
    c2.metric("Composite Score", f"{row['composite_score']:.4f}")
    c3.metric("Risk Band",       row["risk_band"])
    c4.metric("Articles",        int(row["article_count"]))

    st.divider()

    st.subheader("Risk Dimension Breakdown")
    dim_scores = {
        DIM_LABELS[d]: row.get(f"prevalence_{d}", 0) for d in DIMENSIONS
    }
    dim_df = pd.DataFrame.from_dict(
        dim_scores, orient="index", columns=["Prevalence"]
    ).sort_values("Prevalence", ascending=True)

    fig_radar = px.bar(
        dim_df, x="Prevalence", y=dim_df.index, orientation="h",
        color="Prevalence", color_continuous_scale="OrRd",
        labels={"y": ""},
        range_x=[0, 1],
    )
    fig_radar.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()
    st.subheader("Factory Metadata")
    meta_keys = {
        "address":           "Address",
        "number_of_workers": "Number of Workers",
        "is_closed":         "Is Closed",
        "os_id":             "OSH ID",
    }
    meta_df = pd.DataFrame(
        [(label, row.get(key, "N/A")) for key, label in meta_keys.items()],
        columns=["Field", "Value"]
    ).set_index("Field")
    st.table(meta_df)

    st.info(DISCLAIMER)


elif page == "Article Explorer":
    st.title("Article Explorer")
    st.caption("Browse source complaint articles linked to each factory.")

    factory_names = sorted(articles["factory_ref"].dropna().unique().tolist())
    selected = st.selectbox("Select a factory", factory_names)

    fac_arts = articles[articles["factory_ref"] == selected].copy()
    fac_arts["date"] = pd.to_datetime(fac_arts["date"], errors="coerce")
    fac_arts = fac_arts.sort_values("date", ascending=False)

    st.caption(f"{len(fac_arts)} articles found for **{selected}**")

    for _, art in fac_arts.iterrows():
        pred_flags = [
            DIM_LABELS[d].split(" (")[0]
            for d in DIMENSIONS
            if art.get(f"pred_{d}", 0) == 1
        ]
        date_str = str(art["date"])[:10] if pd.notna(art["date"]) else "Unknown date"
        with st.expander(f"{art['title']} — {date_str}"):
            st.markdown(f"**URL:** [{art['URL']}]({art['URL']})")
            if pred_flags:
                st.markdown(
                    "**Predicted Risk Dimensions:** " +
                    " · ".join([f"`{f}`" for f in pred_flags])
                )
            st.markdown("**Article Content:**")
            content = str(art.get("Content", "No content available"))
            st.write(content[:2000] + ("..." if len(content) > 2000 else ""))

    st.info(DISCLAIMER)
