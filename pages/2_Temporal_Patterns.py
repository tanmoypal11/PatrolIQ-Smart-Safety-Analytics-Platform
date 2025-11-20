# ======================================================
# 2_Temporal_Patterns.py
# Fully Streamlit Cloud Safe (no pickle, no mlflow)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# ------------------------------------------------------
# Streamlit Config
# ------------------------------------------------------
st.set_page_config(page_title="Temporal Crime Patterns", layout="wide")

st.title("⏰ Temporal Crime Patterns & Clusters")
st.markdown("""
This page visualizes **temporal crime patterns** using:
- Hourly & Monthly trends  
- Fresh on-the-fly KMeans clustering  
""")

# ------------------------------------------------------
# PATHS
# ------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))      # project root
DATA_PATH = os.path.join(ROOT_DIR, "data", "PatrolIQ_temp_only.csv")

# ------------------------------------------------------
# Load Data
# ------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Hour", "Month", "crime_severity_score"])
    return df

df = load_data()

# ------------------------------------------------------
# dialga() – TEMPORAL FEATURE ENGINEERING
# ------------------------------------------------------
def dialga(df_feature_engineered):
    """
    Created inside Streamlit (safe).
    Performs numeric temporal scaling.
    """
    temp_num_cols = ['Hour', 'Month', 'Day']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), temp_num_cols)
    ], remainder='drop')

    X_temp = preprocessor.fit_transform(df_feature_engineered)

    return X_temp, preprocessor


# ------------------------------------------------------
# Sidebar Filters
# ------------------------------------------------------
st.sidebar.header("🔎 Filters")
crime_options = sorted(df["Primary Type"].unique())
crime_filter = st.sidebar.multiselect("Filter by Crime Type", crime_options, default=[])

if crime_filter:
    df_filtered = df[df["Primary Type"].isin(crime_filter)].copy()
else:
    df_filtered = df.copy()

# ------------------------------------------------------
# TEMPORAL CLUSTERING
# ------------------------------------------------------
# Prepare features using dialga()
X_temp_filtered, _ = dialga(df_filtered)

X_temp_df = pd.DataFrame(
    X_temp_filtered,
    index=df_filtered.index,
    columns=["Hour_scaled", "Month_scaled", "Day_scaled"]
).dropna()

df_filtered = df_filtered.loc[X_temp_df.index]

# TRAIN KMEANS (fresh)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_filtered["temp_cluster"] = kmeans.fit_predict(X_temp_df)

# ------------------------------------------------------
# Hourly Crime Plot
# ------------------------------------------------------
st.subheader("📊 Hourly Crime Frequency")
hourly_counts = df_filtered.groupby("Hour").size().reset_index(name="Count")

fig_hour = px.bar(
    hourly_counts,
    x="Hour",
    y="Count",
    text="Count",
    labels={"Count": "Crime Count", "Hour": "Hour of Day"}
)
st.plotly_chart(fig_hour, use_container_width=True)

# ------------------------------------------------------
# Monthly Crime Plot
# ------------------------------------------------------
st.subheader("📊 Monthly Crime Frequency")

month_order = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

monthly_counts = (
    df_filtered.groupby("MonthName")
    .size()
    .reindex(month_order)
    .reset_index(name="Count")
)

fig_month = px.bar(
    monthly_counts,
    x="MonthName",
    y="Count",
    text="Count",
    labels={"Count": "Crime Count", "MonthName": "Month"}
)
st.plotly_chart(fig_month, use_container_width=True)

# ------------------------------------------------------
# Cluster Distribution
# ------------------------------------------------------
st.subheader("📊 Temporal Cluster Distribution")
cluster_counts = df_filtered["temp_cluster"].value_counts().sort_index()
st.bar_chart(cluster_counts)

# ------------------------------------------------------
# Interpretation
# ------------------------------------------------------
st.markdown("### ✔ Interpretation")
st.markdown("""
- These clusters show **patterns in time** (hour/month/day).  
- Helps identify:
  - **Peak crime hours**
  - **Seasonal patterns**
  - **High-severity time windows**  
""")
