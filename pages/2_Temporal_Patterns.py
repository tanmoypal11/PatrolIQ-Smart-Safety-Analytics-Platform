# 2_Temporal_Patterns.py
import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Temporal Crime Patterns", layout="wide")

st.title("⏰ Temporal Crime Patterns & Clusters")
st.markdown("""
This page visualizes **temporal crime patterns in Chicago** using:
- Hourly & Monthly trends  
- K-Means temporal clusters  
""")

# -------------------------------
# Paths
# -------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))    # project root
DATA_PATH = os.path.join(ROOT_DIR, "data", "PatrolIQ_temp_only.csv")
DIALGA_PATH = os.path.join(ROOT_DIR, "models", "dialga_function_2.pkl")

# -------------------------------
# Load Data & Functions
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Hour", "Month", "crime_severity_score"])
    return df

@st.cache_resource
def load_dialga(path):
    with open(path, "rb") as f:
        return cloudpickle.load(f)

df = load_data()
dialga = load_dialga(DIALGA_PATH)

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("🔎 Filters")
crime_options = sorted(df["Primary Type"].unique())
crime_filter = st.sidebar.multiselect("Filter by Crime Type", crime_options, default=[])

if crime_filter:
    df_filtered = df[df["Primary Type"].isin(crime_filter)].copy()
else:
    df_filtered = df.copy()

# -------------------------------
# TEMPORAL CLUSTERING (NO PICKLE)
# -------------------------------
# Generate scaled temporal features using dialga()
X_temp_filtered, _ = dialga(df_filtered)

X_temp_df = pd.DataFrame(
    X_temp_filtered,
    index=df_filtered.index,
    columns=["Hour", "Month", "crime_severity_score"]
).dropna()

df_filtered = df_filtered.loc[X_temp_df.index]

# Scale freshly
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_temp_df)

# Train fresh KMeans (safe on Streamlit Cloud)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_filtered["temp_cluster"] = kmeans.fit_predict(X_scaled)

# -------------------------------
# Hourly Crime Pattern
# -------------------------------
st.subheader("📊 Hourly Crime Frequency")
hourly_counts = df_filtered.groupby("Hour").size().reset_index(name="Count")
fig_hour = px.bar(hourly_counts, x="Hour", y="Count", text="Count",
                  labels={"Count": "Crime Count", "Hour": "Hour of Day"})
st.plotly_chart(fig_hour, use_container_width=True)

# -------------------------------
# Monthly Crime Pattern
# -------------------------------
st.subheader("📊 Monthly Crime Frequency")
monthly_counts = df_filtered.groupby("MonthName").size().reindex([
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]).reset_index(name="Count")

fig_month = px.bar(monthly_counts, x="MonthName", y="Count", text="Count",
                   labels={"Count": "Crime Count", "MonthName": "Month"})
st.plotly_chart(fig_month, use_container_width=True)

# -------------------------------
# Cluster Distribution Stats
# -------------------------------
st.subheader("📊 Temporal Cluster Distribution")
cluster_counts = df_filtered["temp_cluster"].value_counts().sort_index()
st.bar_chart(cluster_counts)

# Interpretation
st.markdown("### ✔ Interpretation")
st.markdown("""
- Clusters show crime behaviors across **time**.  
- Helps find **peak hours**, **high-severity time windows**, and **seasonal crime surges**.  
""")
