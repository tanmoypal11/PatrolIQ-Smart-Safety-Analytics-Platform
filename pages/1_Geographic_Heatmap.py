import os
import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import pickle

# -----------------------------------------------------
# Streamlit Page Config
# -----------------------------------------------------
st.set_page_config(page_title="Geographic Crime Heatmap", layout="wide")

st.title("📍 Geographic Crime Heatmap & Geo Clusters")
st.markdown("""
This page visualizes **crime intensity across Chicago** using:  
- 🔥 Heatmaps  
- 🎯 K-Means Geographic Clusters  
- 📌 Marker Clustering  
""")

# -----------------------------------------------------
# File Paths (relative)
# -----------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # Go up one level from pages/
DATA_PATH = os.path.join(ROOT_DIR, "data", "PatrolIQ_geo_only_sample.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "geo_kmeans_model.pkl")

# -----------------------------------------------------
# Load Data + Model
# -----------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Latitude", "Longitude"])
    return df

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

df = load_data()
geo_kmeans = load_model()

# -----------------------------------------------------
# Sidebar Filters
# -----------------------------------------------------
st.sidebar.header("🔎 Filters")

crime_list = sorted(df["Primary Type"].unique())
crime_filter = st.sidebar.multiselect("Filter Crime Type", crime_list)

df_filtered = df[df["Primary Type"].isin(crime_filter)] if crime_filter else df.copy()

# -----------------------------------------------------
# Predict Clusters
# -----------------------------------------------------
coords = df_filtered[["Latitude", "Longitude", "crime_severity_score"]].values
df_filtered["geo_cluster"] = geo_kmeans.predict(coords)

# -----------------------------------------------------
# Heatmap + Marker Cluster Map
# -----------------------------------------------------
st.subheader("🔥 Crime Heatmap & Cluster Visualization")

m = folium.Map(location=[41.8781, -87.6298], zoom_start=11, tiles="cartodbpositron")

# 🔥 Heatmap Layer
heat_data = df_filtered[["Latitude", "Longitude"]].values.tolist()
HeatMap(heat_data, radius=7, blur=5).add_to(m)

# 📌 Marker Cluster Layer
marker_cluster = MarkerCluster().add_to(m)

# Optional sampling for speed
sample_df = df_filtered.sample(min(2000, len(df_filtered)), random_state=42)

for _, row in sample_df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=4,
        color=f"#{(row['geo_cluster']+3)*30:02x}00{(5-row['geo_cluster'])*30:02x}",
        fill=True,
        fill_opacity=0.6,
        popup=f"Cluster: {row['geo_cluster']}<br>Crime: {row['Primary Type']}"
    ).add_to(marker_cluster)

# Render Map
st_folium(m, width=1400, height=700)

# -----------------------------------------------------
# Cluster Stats
# -----------------------------------------------------
st.subheader("📊 Cluster Distribution")
st.bar_chart(df_filtered["geo_cluster"].value_counts().sort_index())

st.markdown("""
### 📌 Interpretation  
- **Bright red zones** indicate crime hotspots  
- **Cluster colors** show distinct geographic crime patterns  
- More markers = higher crime density  
""")
