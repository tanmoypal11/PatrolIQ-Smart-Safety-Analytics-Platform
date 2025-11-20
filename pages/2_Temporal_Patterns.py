# 2_Temporal_Patterns.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import cloudpickle
import os
import plotly.express as px

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
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # Go up one level from pages/
DATA_PATH = os.path.join(ROOT_DIR, "data", "PatrolIQ_temp_only.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "kmeans_temp_model.pkl")
DIALGA_PATH = os.path.join(ROOT_DIR, "models", "dialga_function_2.pkl")

# -------------------------------
# Load Data & Models
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Ensure no missing coordinates/time info
    df = df.dropna(subset=["Hour", "Month", "crime_severity_score"])
    return df

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_dialga(path):
    with open(path, "rb") as f:
        return cloudpickle.load(f)

df = load_data()
temp_kmeans = load_model(MODEL_PATH)
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
# Temporal Clustering
# -------------------------------
# Generate scaled temporal features using dialga()
X_temp_filtered, _ = dialga(df_filtered)

# Convert to DataFrame to align with original indices
X_temp_df = pd.DataFrame(X_temp_filtered, index=df_filtered.index, columns=['Hour', 'Month', 'crime_severity_score'])

# Drop rows with NaNs
X_temp_df = X_temp_df.dropna()
df_filtered = df_filtered.loc[X_temp_df.index]

# Predict temporal clusters
df_filtered["temp_cluster"] = temp_kmeans.predict(X_temp_df.values)


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

st.markdown("### ✔ Interpretation")
st.markdown("""
- Clusters group crimes based on **time patterns** (hour, month, severity)  
- Helps identify **peak crime hours** and **seasonal trends**  
""")
