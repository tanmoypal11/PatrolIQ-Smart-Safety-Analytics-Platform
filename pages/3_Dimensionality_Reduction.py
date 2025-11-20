import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ------------------------------------------------------------
# Page Title
# ------------------------------------------------------------
st.title("🧠 PCA & Dimensionality Reduction Dashboard")

# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/pca_kmeans_results.csv")
    return df

df = load_data()

st.success("PCA + KMeans data loaded successfully!")

# ------------------------------------------------------------
# PCA Summary Section
# ------------------------------------------------------------
st.subheader("📘 PCA Dimensionality Summary")

original_dim = 60
reduced_dim = 8
explained_variance = 0.7113016880762798

st.markdown(f"""
**Original dimensions:** `{original_dim}`  
**Reduced dimensions (~80% variance):** `{reduced_dim}`  
**Explained variance ratio:** `{explained_variance}`
""")

# ------------------------------------------------------------
# Feature Importance Section
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🔥 Top Features Driving Principal Components")

# ---- PC1 ----
st.markdown("### 🟦 Top 5 Features — Principal Component 1")
st.text("""
num__Latitude                   0.39958762169188605
num__Y Coordinate               0.3992414092615185
num__District                   0.3927917259621543
num__Beat                       0.3828689358748723
num__X Coordinate               0.3359302670888766
""")

# ---- PC2 ----
st.markdown("### 🟩 Top 5 Features — Principal Component 2")
st.text("""
num__Arrest                     0.5090016518064672
num__Year                       0.43838487558949324
num__crime_severity_score       0.3830891066017058
num__Second                     0.36608033148258384
num__Minute                     0.2588086690855801
""")

# ------------------------------------------------------------
# 2D PCA Visualization
# ------------------------------------------------------------
st.markdown("---")
st.subheader("📊 PCA — 2D Component Projection")

fig_pca = px.scatter(
    df,
    x="PC1",
    y="PC2",
    color="Cluster",
    title="PCA 2D Projection with Cluster Coloring",
    opacity=0.7,
)
st.plotly_chart(fig_pca, use_container_width=True)

# ------------------------------------------------------------
# t-SNE Visualization
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🌈 t-SNE Visualization")

fig_tsne = px.scatter(
    df,
    x="TSNE1",
    y="TSNE2",
    color="Cluster",
    title="t-SNE 2D Projection with Clusters",
    opacity=0.7,
)
st.plotly_chart(fig_tsne, use_container_width=True)


# ------------------------------------------------------------
# Cluster Summary Table
# ------------------------------------------------------------
st.markdown("---")
st.subheader("📑 Cluster Summary Statistics")

cluster_summary = df.groupby("Cluster").agg({
    "PC1": "mean",
    "PC2": "mean",
    "TSNE1": "mean",
    "TSNE2": "mean",
    "crime_severity_score": "mean"
}).reset_index()

st.dataframe(cluster_summary, use_container_width=True)

# End of file
