# 2_Temporal_Patterns.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Temporal Crime Patterns", layout="wide")

st.title("⏰ Temporal Crime Patterns & Clusters")
st.markdown("""
This page visualizes **temporal crime patterns** using:
- Hourly & Monthly trends  
- On-the-fly KMeans temporal clustering  
""")

# -------------------------------
# Paths
# -------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))    # project root
DATA_PATH = os.path.join(ROOT_DIR, "data", "PatrolIQ_temp_only.csv")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # drop rows missing core temporal columns
    df = df.dropna(subset=["Hour", "Month", "crime_severity_score"])
    return df

df = load_data(DATA_PATH)

# -------------------------------
# Ensure MonthName exists (create from Month if missing)
# -------------------------------
# mapping numeric month -> month name
MONTH_ORDER = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

def ensure_monthname(df):
    if "MonthName" not in df.columns or df["MonthName"].isnull().all():
        # try to create MonthName from numeric Month column
        def num_to_name(m):
            try:
                m_int = int(m)
                if 1 <= m_int <= 12:
                    return MONTH_ORDER[m_int - 1]
            except Exception:
                return np.nan
            return np.nan

        df["MonthName"] = df["Month"].apply(num_to_name)
    return df

df = ensure_monthname(df)

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("🔎 Filters")
crime_options = sorted(df["Primary Type"].dropna().unique())
crime_filter = st.sidebar.multiselect("Filter by Crime Type", crime_options, default=[])

if crime_filter:
    df_filtered = df[df["Primary Type"].isin(crime_filter)].copy()
else:
    df_filtered = df.copy()

# reset index for safe indexing
df_filtered = df_filtered.reset_index(drop=True)

# -------------------------------
# TEMPORAL CLUSTERING
# -------------------------------
# Use only the available temporal features: Hour, Month, crime_severity_score
temp_cols = [c for c in ["Hour", "Month", "crime_severity_score"] if c in df_filtered.columns]
if len(df_filtered) == 0 or len(temp_cols) == 0:
    st.error("No data available after filtering or required temporal columns missing.")
else:
    # prepare features (ensure numeric)
    X_temp = df_filtered[temp_cols].copy()
    X_temp = X_temp.apply(pd.to_numeric, errors="coerce")
    X_temp = X_temp.dropna()

    # align df_filtered with X_temp rows
    df_filtered = df_filtered.loc[X_temp.index].reset_index(drop=True)
    X_temp = X_temp.reset_index(drop=True)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_temp)

    # choose number of clusters dynamically (avoid n_clusters > n_samples)
    n_samples = X_scaled.shape[0]
    n_clusters = min(3, max(1, n_samples))  # at least 1 cluster, up to 3
    if n_clusters == 1:
        # trivial cluster assignment
        df_filtered["temp_cluster"] = 0
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_filtered["temp_cluster"] = kmeans.fit_predict(X_scaled)

    # -------------------------------
    # Hourly Crime Pattern
    # -------------------------------
    st.subheader("📊 Hourly Crime Frequency")
    # Ensure Hour is numeric for ordering
    df_filtered["Hour"] = pd.to_numeric(df_filtered["Hour"], errors="coerce")
    hourly_counts = df_filtered.groupby("Hour").size().reset_index(name="Count").sort_values("Hour")
    fig_hour = px.bar(hourly_counts, x="Hour", y="Count", text="Count",
                      labels={"Count": "Crime Count", "Hour": "Hour of Day"})
    st.plotly_chart(fig_hour, use_container_width=True)

    # -------------------------------
    # Monthly Crime Pattern
    # -------------------------------
    st.subheader("📊 Monthly Crime Frequency")
    # Prepare monthly counts with full month order and zero-fill missing months
    monthly_series = df_filtered.groupby("MonthName").size()
    monthly_counts = monthly_series.reindex(MONTH_ORDER).fillna(0).reset_index(name="Count").rename(columns={"index":"MonthName"})
    # Convert Count to int for display
    monthly_counts["Count"] = monthly_counts["Count"].astype(int)

    fig_month = px.bar(
        monthly_counts,
        x="MonthName",
        y="Count",
        text="Count",
        labels={"Count": "Crime Count", "MonthName": "Month"}
    )
    fig_month.update_xaxes(categoryorder="array", categoryarray=MONTH_ORDER)
    st.plotly_chart(fig_month, use_container_width=True)

    # -------------------------------
    # Cluster Distribution Stats
    # -------------------------------
    st.subheader("📊 Temporal Cluster Distribution")
    cluster_counts = df_filtered["temp_cluster"].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    # -------------------------------
    # Interpretation
    # -------------------------------
    st.markdown("### ✔ Interpretation")
    st.markdown("""
    - Clusters capture common temporal behaviors (hour/month/severity).  
    - Use cluster distribution and hourly/monthly charts to identify peak hours and seasonal trends.  
    """)

# optional: show a small sample table for verification
with st.expander("Show sample rows"):
    st.write(df_filtered.head(10))
