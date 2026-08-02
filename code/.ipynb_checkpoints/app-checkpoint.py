import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="UK Bus Service Performance Dashboard",
    page_icon="🚌",
    layout="wide"
)

st.title("🚌 UK Bus Service Performance & Delay Analytics Dashboard")
st.markdown("### Interactive visualization of 100,000+ Trip Records & Machine Learning Segmentation")

@st.cache_data
def load_data():
    if os.path.exists("output/dashboard_data.csv"):
        return pd.read_csv("output/dashboard_data.csv")
    else:
        st.error("Dashboard data file not found. Please run Notebook 3 first.")
        return pd.DataFrame()

df = load_data()

if not df.empty:
    # Sidebar Filters
    st.sidebar.header("Filter Options")
    operators = st.sidebar.multiselect("Select Operator:", options=df["Operator"].unique(), default=df["Operator"].unique()[:3])
    filtered_df = df[df["Operator"].isin(operators)] if operators else df

    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trips Analyzed", f"{len(filtered_df):,}")
    col2.metric("Avg Actual Delay", f"{filtered_df['ActualDelayMin'].mean():.2f} mins")
    col3.metric("Avg Service Speed", f"{filtered_df['AvgSpeed'].mean():.2f} km/h")
    col4.metric("Avg Performance Index", f"{filtered_df['ServicePerformanceIndex'].mean():.2f}")

    st.markdown("---")

    # Visualizations
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Hourly Delay Patterns")
        hourly_fig = px.bar(
            filtered_df.groupby("DepartureHour")["ActualDelayMin"].mean().reset_index(),
            x="DepartureHour",
            y="ActualDelayMin",
            title="Average Delay by Departure Hour",
            color="ActualDelayMin",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(hourly_fig, use_container_width=True)

    with c2:
        st.subheader("ML Service Cluster Segmentation")
        cluster_fig = px.scatter(
            filtered_df,
            x="ActualDelayMin",
            y="ServicePerformanceIndex",
            color=filtered_df["prediction"].astype(str),
            title="K-Means Clusters: Delay vs Performance Index",
            hover_data=["Operator", "LineName"]
        )
        st.plotly_chart(cluster_fig, use_container_width=True)

    st.subheader("Detailed Trips Sample Data")
    st.dataframe(filtered_df.head(100), use_container_width=True)