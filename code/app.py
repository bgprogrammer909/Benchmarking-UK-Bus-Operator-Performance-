"""
UK Bus Operator Performance — Executive Analytics Dashboard
=============================================================
Big Data Project | Streamlit + Plotly

This dashboard visualizes the OUTPUT of an existing ML clustering model
(the `prediction` column, values 0-3). The model itself is NOT modified,
re-trained, or re-thresholded anywhere in this file — the four clusters
produced by the model are preserved and displayed as-is.

Run with:
    streamlit run app.py
"""

import hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="UK Bus Operator Performance | Executive Dashboard",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CONSTANTS — cluster labels & the executive colour palette
# ----------------------------------------------------------------------------
from pathlib import Path

# BASE_DIR points to 'code/' directory where app.py lives
BASE_DIR = Path(__file__).resolve().parent

# Go up one level to root, then into 'output/'
DATA_PATH = BASE_DIR.parent / "output" / "dashboard_data.csv"

# Maps the raw ML cluster IDs (0-3) to business-friendly labels.
# The underlying IDs from the `prediction` column are NEVER altered —
# this is purely a display-layer mapping.
CLUSTER_LABELS = {
    0: "Standard Operations",
    1: "Efficient Operations",
    2: "Premium Operations",
    3: "Underperforming Operations",
}

# Consistent colour mapping used across every single chart in the app.
CLUSTER_COLORS = {
    "Standard Operations": "#2563EB",       # Blue
    "Efficient Operations": "#16A34A",      # Green
    "Premium Operations": "#9333EA",        # Purple
    "Underperforming Operations": "#DC2626",  # Red
}

CLUSTER_ORDER = [
    "Standard Operations",
    "Efficient Operations",
    "Premium Operations",
    "Underperforming Operations",
]

CORR_COLUMNS = [
    "Trips",
    "Stops",
    "RouteVehicleCount",
    "ServicePerformanceIndex",
]

# UK bounding box used only for synthetic map coordinates (see note in
# preprocess_data). Roughly covers mainland UK.
UK_LAT_RANGE = (50.0, 58.5)
UK_LON_RANGE = (-6.5, 1.5)


# ----------------------------------------------------------------------------
# STYLING — inject custom CSS for an "executive BI platform" look & feel
# ----------------------------------------------------------------------------
def inject_custom_css() -> None:
    """Overrides Streamlit's default look with a clean, card-based,
    corporate BI aesthetic: white background, soft shadows, rounded
    corners, refined typography and a polished sidebar."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Overall page background */
        .stApp {
            background-color: #F6F8FB;
        }

        /* Force a consistent dark, readable text colour across the main
           content area regardless of the viewer's browser/OS theme
           (Streamlit otherwise inherits a light/white text colour under
           dark mode, which becomes invisible on the white cards below).
           The sidebar rule further down uses !important + a more specific
           selector, so it still wins there and stays light-on-dark. */
        .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
        .stApp div, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp h5, .stApp h6, .stMarkdown, .stCaption,
        [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"],
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
        [data-testid="stWidgetLabel"] {
            color: #101828;
        }
        [data-testid="stCaptionContainer"], .stCaption {
            color: #667085 !important;
        }

        /* Remove default top padding for a tighter, dashboard feel */
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #101828;
        }
        section[data-testid="stSidebar"] * {
            color: #E5E7EB !important;
        }
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stMultiSelect label,
        section[data-testid="stSidebar"] .stSelectbox label {
            font-weight: 600 !important;
            color: #F9FAFB !important;
        }

        /* Header / hero */
        .dashboard-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #101828;
            margin-bottom: 0.1rem;
            letter-spacing: -0.02em;
        }
        .dashboard-subtitle {
            font-size: 0.98rem;
            color: #667085;
            margin-bottom: 1.4rem;
            font-weight: 500;
        }

        /* Generic card container */
        .exec-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 1px 3px rgba(16,24,40,0.06), 0 1px 2px rgba(16,24,40,0.04);
            border: 1px solid #EAECF0;
            height: 100%;
        }

        /* KPI metric cards */
        .kpi-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 1px 3px rgba(16,24,40,0.06), 0 1px 2px rgba(16,24,40,0.04);
            border: 1px solid #EAECF0;
            text-align: left;
        }
        .kpi-label {
            font-size: 0.78rem;
            font-weight: 600;
            color: #667085;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 1.65rem;
            font-weight: 800;
            color: #101828;
        }

        /* Cluster summary cards */
        .cluster-card {
            border-radius: 16px;
            padding: 1.05rem 1.2rem;
            box-shadow: 0 1px 3px rgba(16,24,40,0.06), 0 1px 2px rgba(16,24,40,0.04);
            border: 1px solid #EAECF0;
            background: #FFFFFF;
            border-top: 5px solid var(--accent, #2563EB);
        }
        .cluster-card-title {
            font-size: 0.9rem;
            font-weight: 700;
            color: #101828;
            margin-bottom: 0.3rem;
        }
        .cluster-card-count {
            font-size: 1.5rem;
            font-weight: 800;
            color: #101828;
        }
        .cluster-card-pct {
            font-size: 0.85rem;
            font-weight: 600;
            color: #667085;
        }

        /* Section headers */
        .section-header {
            font-size: 1.15rem;
            font-weight: 700;
            color: #101828;
            margin-top: 1.6rem;
            margin-bottom: 0.7rem;
            padding-bottom: 0.4rem;
            border-bottom: 2px solid #EAECF0;
        }

        /* Dataframe container rounding */
        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #EAECF0;
        }

        /* Buttons */
        .stDownloadButton button, .stButton button {
            border-radius: 10px;
            font-weight: 600;
            border: 1px solid #2563EB;
        }

        /* Hide only the "Made with Streamlit" footer; keep the toolbar
           (including the Deploy button and hamburger menu) visible. */
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_data(path: str) -> pd.DataFrame:
    """Loads the raw dashboard CSV exactly as produced by the ML pipeline.
    No cleaning of the `prediction` column happens here."""
    df = pd.read_csv(path)
    return df


def _synthetic_coordinates(operator: str, service_code: str) -> tuple[float, float]:
    """Deterministically generates a (lat, lon) pair for a route.

    NOTE: The source CSV does not contain any geographic columns
    (latitude/longitude/postcode), so there is no real location data to
    plot. To keep the interactive UK map functional, each Operator is
    assigned a fixed "base" location within the UK bounding box (hashed
    from its name, so it's stable across reruns/filters), and individual
    routes are jittered slightly around that point using their
    ServiceCode. This is a display convenience only — swap this function
    out for real geocoded coordinates if/when they become available.
    """
    op_hash = int(hashlib.md5(operator.encode()).hexdigest(), 16)
    route_hash = int(hashlib.md5(f"{operator}_{service_code}".encode()).hexdigest(), 16)

    op_rng = np.random.default_rng(op_hash % (2**32))
    base_lat = op_rng.uniform(*UK_LAT_RANGE)
    base_lon = op_rng.uniform(*UK_LON_RANGE)

    route_rng = np.random.default_rng(route_hash % (2**32))
    jitter_lat = route_rng.uniform(-0.15, 0.15)
    jitter_lon = route_rng.uniform(-0.15, 0.15)

    return base_lat + jitter_lat, base_lon + jitter_lon


@st.cache_data(show_spinner="Preparing analytics...")
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Adds display-layer columns (cluster label, colour, synthetic
    coordinates) on top of the raw ML output. The original `prediction`
    values are preserved untouched in the `prediction` column."""
    df = df.copy()

    # Map the 4 raw cluster IDs to business-friendly labels — all 4 kept.
    df["Cluster"] = df["prediction"].map(CLUSTER_LABELS)
    df["ClusterColor"] = df["Cluster"].map(CLUSTER_COLORS)

    # Generate synthetic-but-stable coordinates for the map (see note above).
    coords = df.apply(
        lambda row: _synthetic_coordinates(str(row["Operator"]), str(row["ServiceCode"])),
        axis=1,
        result_type="expand",
    )
    df["lat"], df["lon"] = coords[0], coords[1]

    return df


# ----------------------------------------------------------------------------
# SIDEBAR — FILTERS
# ----------------------------------------------------------------------------
def create_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Renders sidebar filter controls and returns the filtered dataframe.
    Every visualization downstream consumes this filtered dataframe."""
    st.sidebar.markdown(
        "<div style='font-size:1.3rem;font-weight:800;color:#FFFFFF;"
        "margin-bottom:0.1rem;'>🚌 Fleet Analytics</div>"
        "<div style='font-size:0.8rem;color:#98A2B3;margin-bottom:1.4rem;'>"
        "UK Bus Operator Performance</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("#### Filters")

    operators = sorted(df["Operator"].dropna().unique().tolist())
    selected_operators = st.sidebar.multiselect(
        "Operator", options=operators, default=[]
    )

    selected_clusters = st.sidebar.multiselect(
        "Cluster", options=CLUSTER_ORDER, default=CLUSTER_ORDER
    )

    spi_min, spi_max = float(df["ServicePerformanceIndex"].min()), float(
        df["ServicePerformanceIndex"].max()
    )
    spi_range = st.sidebar.slider(
        "SPI range (min / max)",
        min_value=spi_min,
        max_value=spi_max,
        value=(spi_min, spi_max),
    )

    trips_min, trips_max = int(df["Trips"].min()), int(df["Trips"].max())
    trips_range = st.sidebar.slider(
        "Trips range", min_value=trips_min, max_value=trips_max,
        value=(trips_min, trips_max),
    )

    stops_min, stops_max = int(df["Stops"].min()), int(df["Stops"].max())
    stops_range = st.sidebar.slider(
        "Stops range", min_value=stops_min, max_value=stops_max,
        value=(stops_min, stops_max),
    )

    # Apply filters
    filtered = df.copy()
    if selected_operators:
        filtered = filtered[filtered["Operator"].isin(selected_operators)]
    if selected_clusters:
        filtered = filtered[filtered["Cluster"].isin(selected_clusters)]
    filtered = filtered[
        filtered["ServicePerformanceIndex"].between(spi_range[0], spi_range[1])
        & filtered["Trips"].between(trips_range[0], trips_range[1])
        & filtered["Stops"].between(stops_range[0], stops_range[1])
    ]

    st.sidebar.markdown("---")
   
    return filtered


# ----------------------------------------------------------------------------
# KPI CARDS
# ----------------------------------------------------------------------------
def create_kpis(df: pd.DataFrame) -> None:
    """Renders the top-row KPI metric cards."""
    st.markdown("<div class='section-header'>Key Performance Indicators</div>", unsafe_allow_html=True)

    total_routes = len(df)
    total_operators = df["Operator"].nunique()
    avg_spi = df["ServicePerformanceIndex"].mean() if total_routes else 0
    avg_trips = df["Trips"].mean() if total_routes else 0
    avg_stops = df["Stops"].mean() if total_routes else 0

    kpis = [
        ("Total Routes", f"{total_routes:,}"),
        ("Total Operators", f"{total_operators:,}"),
        ("Average SPI", f"{avg_spi:,.1f}"),
        ("Average Trips", f"{avg_trips:,.1f}"),
        ("Average Stops", f"{avg_stops:,.1f}"),
    ]

    cols = st.columns(5)
    for col, (label, value) in zip(cols, kpis):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------------
# CLUSTER SUMMARY CARDS
# ----------------------------------------------------------------------------
def create_cluster_summary(df: pd.DataFrame) -> None:
    """Renders 4 cards — one per ML cluster — with route count & share."""
    st.markdown("<div class='section-header'>ML Cluster Summary</div>", unsafe_allow_html=True)

    total = len(df)
    cols = st.columns(4)
    for col, cluster in zip(cols, CLUSTER_ORDER):
        count = int((df["Cluster"] == cluster).sum())
        pct = (count / total * 100) if total else 0
        color = CLUSTER_COLORS[cluster]
        with col:
            st.markdown(
                f"""
                <div class="cluster-card" style="--accent: {color};">
                    <div class="cluster-card-title">{cluster}</div>
                    <div class="cluster-card-count">{count:,}</div>
                    <div class="cluster-card-pct">{pct:.1f}% of routes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ----------------------------------------------------------------------------
# INTERACTIVE MAP
# ----------------------------------------------------------------------------
def style_fig(fig: go.Figure, legend_below: bool = False) -> go.Figure:
    """Applies consistent, dark, readable typography to a Plotly figure.

    Plotly renders chart text as SVG, so it does NOT inherit the page's
    CSS — it must be set explicitly on the figure itself, which is what
    this helper does for every chart in the dashboard (titles, axis
    titles, tick labels, legend and hover text)."""
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color="#101828", size=13),
        title_font=dict(color="#101828", size=15),
        legend=dict(font=dict(color="#101828")),
        hoverlabel=dict(font=dict(color="#101828"), bgcolor="#FFFFFF"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(
        title_font=dict(color="#101828"),
        tickfont=dict(color="#344054"),
        gridcolor="#EAECF0",
        linecolor="#D0D5DD",
        zerolinecolor="#EAECF0",
    )
    fig.update_yaxes(
        title_font=dict(color="#101828"),
        tickfont=dict(color="#344054"),
        gridcolor="#EAECF0",
        linecolor="#D0D5DD",
        zerolinecolor="#EAECF0",
    )
    if legend_below:
        fig.update_layout(legend=dict(
            font=dict(color="#101828"), orientation="h", y=-0.25,
        ))
    return fig


def create_map(df: pd.DataFrame) -> None:
    """Renders a Plotly Mapbox scatter map coloured by cluster, sized by trips."""
    st.markdown("<div class='section-header'>Interactive Route Map</div>", unsafe_allow_html=True)
    st.caption(
        "Note: the source dataset has no latitude/longitude columns, so route "
        "positions here are generated deterministically per operator for "
        "visualization purposes and do not represent exact real-world stops."
    )

    if df.empty:
        st.info("No routes match the current filters.")
        return

    fig = go.Figure()
    for cluster in CLUSTER_ORDER:
        cluster_df = df[df["Cluster"] == cluster]
        if cluster_df.empty:
            continue
        # Marker size proportional to Trips (with a floor so tiny values are visible)
        sizes = 6 + (cluster_df["Trips"] / max(df["Trips"].max(), 1)) * 30
        fig.add_trace(
            go.Scattermapbox(
                lat=cluster_df["lat"],
                lon=cluster_df["lon"],
                mode="markers",
                name=cluster,
                marker=dict(
                    size=sizes,
                    color=CLUSTER_COLORS[cluster],
                    opacity=0.75,
                ),
                text=cluster_df.apply(
                    lambda r: (
                        f"<b>{r['Operator']}</b><br>"
                        f"Line: {r['LineName']}<br>"
                        f"Trips: {r['Trips']:,}<br>"
                        f"Stops: {r['Stops']:,}<br>"
                        f"SPI: {r['ServicePerformanceIndex']:.1f}<br>"
                        f"Cluster: {r['Cluster']}"
                    ),
                    axis=1,
                ),
                hoverinfo="text",
            )
        )

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=54.5, lon=-3.0),
            zoom=4.6,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(color="#101828", size=13, family="Inter, sans-serif"),
        ),
        font=dict(family="Inter, sans-serif", color="#101828"),
        hoverlabel=dict(font=dict(color="#101828"), bgcolor="#FFFFFF"),
        paper_bgcolor="#FFFFFF",
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------------
# ANALYTICS CHARTS
# ----------------------------------------------------------------------------
def create_charts(df: pd.DataFrame) -> None:
    """Renders the full analytics section: scatter, box, histogram,
    operator bar charts, pie chart and correlation heatmap."""
    st.markdown("<div class='section-header'>Analytics</div>", unsafe_allow_html=True)

    if df.empty:
        st.info("No routes match the current filters.")
        return

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown("<div class='exec-card'>", unsafe_allow_html=True)
        st.markdown("**Trips vs Stops by Cluster**")
        fig = px.scatter(
            df, x="Trips", y="Stops", color="Cluster",
            color_discrete_map=CLUSTER_COLORS,
            category_orders={"Cluster": CLUSTER_ORDER},
            hover_data=["Operator", "LineName", "ServicePerformanceIndex"],
            opacity=0.7,
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        style_fig(fig, legend_below=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row1_col2:
        st.markdown("<div class='exec-card'>", unsafe_allow_html=True)
        st.markdown("**SPI Distribution by Cluster**")
        fig = px.box(
            df, x="Cluster", y="ServicePerformanceIndex", color="Cluster",
            color_discrete_map=CLUSTER_COLORS,
            category_orders={"Cluster": CLUSTER_ORDER},
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                           showlegend=False, xaxis_title=None)
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("<div class='exec-card'>", unsafe_allow_html=True)
        st.markdown("**Overall SPI Distribution**")
        fig = px.histogram(
            df, x="ServicePerformanceIndex", nbins=40,
            color_discrete_sequence=["#2563EB"],
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row2_col2:
        st.markdown("<div class='exec-card'>", unsafe_allow_html=True)
        st.markdown("**Cluster Distribution**")
        cluster_counts = df["Cluster"].value_counts().reindex(CLUSTER_ORDER).fillna(0)
        fig = px.pie(
            names=cluster_counts.index, values=cluster_counts.values,
            color=cluster_counts.index, color_discrete_map=CLUSTER_COLORS,
            hole=0.45,
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        fig.update_traces(textfont=dict(color="#FFFFFF", size=13))
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Top operators by route count (limit to top 20 for readability)
    st.markdown("<div class='exec-card'>", unsafe_allow_html=True)
    st.markdown("**Number of Routes per Operator (Top 20)**")
    routes_per_op = (
        df.groupby("Operator").size().sort_values(ascending=False).head(20)
    )
    fig = px.bar(
        x=routes_per_op.values, y=routes_per_op.index, orientation="h",
        labels={"x": "Routes", "y": "Operator"},
        color_discrete_sequence=["#2563EB"],
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                       yaxis=dict(categoryorder="total ascending"))
    style_fig(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='exec-card'>", unsafe_allow_html=True)
    st.markdown("**Average SPI by Operator (Top 20)**")
    avg_spi_op = (
        df.groupby("Operator")["ServicePerformanceIndex"].mean()
        .sort_values(ascending=False).head(20)
    )
    fig = px.bar(
        x=avg_spi_op.values, y=avg_spi_op.index, orientation="h",
        labels={"x": "Average SPI", "y": "Operator"},
        color_discrete_sequence=["#16A34A"],
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                       yaxis=dict(categoryorder="total ascending"))
    style_fig(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown("<div class='exec-card'>", unsafe_allow_html=True)
    st.markdown("**Correlation Matrix — Route Metrics**")
    corr = df[CORR_COLUMNS].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto",
    )
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_traces(textfont=dict(size=12))
    style_fig(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# OPERATOR SCORECARD
# ----------------------------------------------------------------------------
def create_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """Builds and renders the executive operator scorecard table,
    sorted by average SPI descending. Returns the scorecard dataframe."""
    st.markdown("<div class='section-header'>Operator Performance Scorecard</div>", unsafe_allow_html=True)

    if df.empty:
        st.info("No routes match the current filters.")
        return pd.DataFrame()

    def dominant_cluster(series: pd.Series) -> str:
        return series.mode().iloc[0] if not series.mode().empty else "N/A"

    scorecard = (
        df.groupby("Operator")
        .agg(
            Routes=("Operator", "size"),
            Avg_SPI=("ServicePerformanceIndex", "mean"),
            Avg_Trips=("Trips", "mean"),
            Avg_Stops=("Stops", "mean"),
            Dominant_Cluster=("Cluster", dominant_cluster),
        )
        .reset_index()
        .sort_values("Avg_SPI", ascending=False)
    )

    scorecard["Avg_SPI"] = scorecard["Avg_SPI"].round(1)
    scorecard["Avg_Trips"] = scorecard["Avg_Trips"].round(1)
    scorecard["Avg_Stops"] = scorecard["Avg_Stops"].round(1)

    scorecard_display = scorecard.rename(columns={
        "Operator": "Operator",
        "Routes": "# Routes",
        "Avg_SPI": "Avg SPI",
        "Avg_Trips": "Avg Trips",
        "Avg_Stops": "Avg Stops",
        "Dominant_Cluster": "Dominant Cluster",
    })

    st.dataframe(
        scorecard_display,
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config={
            "Avg SPI": st.column_config.NumberColumn(format="%.1f"),
            "Avg Trips": st.column_config.NumberColumn(format="%.1f"),
            "Avg Stops": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    return scorecard_display


# ----------------------------------------------------------------------------
# DATA EXPLORER
# ----------------------------------------------------------------------------
def create_data_explorer(df: pd.DataFrame) -> None:
    """Searchable raw-data explorer with CSV export."""
    st.markdown("<div class='section-header'>Data Explorer</div>", unsafe_allow_html=True)

    search_term = st.text_input(
        "Search (matches Operator, LineName, Description, ServiceCode)",
        placeholder="e.g. Arriva, KC1, Bangor...",
    )

    display_df = df.copy()
    if search_term:
        mask = (
            display_df["Operator"].astype(str).str.contains(search_term, case=False, na=False)
            | display_df["LineName"].astype(str).str.contains(search_term, case=False, na=False)
            | display_df["Description"].astype(str).str.contains(search_term, case=False, na=False)
            | display_df["ServiceCode"].astype(str).str.contains(search_term, case=False, na=False)
        )
        display_df = display_df[mask]

    display_cols = [
        "Operator", "ServiceCode", "LineName", "Description", "Trips", "Stops",
        "RouteVehicleCount", "RouteAvgSpeed", "RouteMaxSpeed", "RouteDisruptions",
        "ServicePerformanceIndex", "Cluster",
    ]

    st.caption(f"Showing {len(display_df):,} rows")
    st.dataframe(display_df[display_cols], use_container_width=True, height=380)

    csv_bytes = display_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download filtered data as CSV",
        data=csv_bytes,
        file_name="bus_operator_performance_filtered.csv",
        mime="text/csv",
    )


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main() -> None:
    inject_custom_css()

    raw_df = load_data(DATA_PATH)
    df = preprocess_data(raw_df)

    st.markdown(
        "<div class='dashboard-title'>UK Bus Operator Performance</div>"
        "<div class='dashboard-subtitle'>Executive Analytics Dashboard "
        "&middot; ML-Driven Route Clustering (4 clusters)</div>",
        unsafe_allow_html=True,
    )

    filtered_df = create_sidebar(df)

    create_kpis(filtered_df)
    create_cluster_summary(filtered_df)
    create_map(filtered_df)
    create_charts(filtered_df)
    create_scorecard(filtered_df)
    create_data_explorer(filtered_df)


if __name__ == "__main__":
    main()