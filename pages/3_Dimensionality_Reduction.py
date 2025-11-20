# pages/3_Dimensionality_Reduction.py
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="PCA & KMeans Evaluation", layout="wide")
st.title("📉 PCA Dimensionality Reduction & KMeans Evaluation")

# ---------------------------
# Path (Streamlit Cloud safe)
# ---------------------------
ROOT_DIR = os.getcwd()   # reliable root on Streamlit Cloud
DATA_PATH = os.path.join(ROOT_DIR, "data", "pca_kmeans_results.csv")

st.info(f"Loading results from: `{DATA_PATH}`")

# Debug: show /data contents
data_folder = os.path.join(ROOT_DIR, "data")
if os.path.exists(data_folder):
    st.write("Files in /data:", sorted(os.listdir(data_folder)))
else:
    st.error("Folder /data not found in repo root.")
    st.stop()

# ---------------------------
# Load CSV
# ---------------------------
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error("Could not load `pca_kmeans_results.csv` from /data.")
    st.code(str(e))
    st.stop()

# ---------------------------
# Validate expected columns
# ---------------------------
required_cols = {"n_components", "n_clusters", "silhouette_score", "davies_bouldin_score"}
if not required_cols.issubset(df.columns):
    st.error(f"CSV missing required columns. Found: {list(df.columns)}")
    st.stop()

# ---------------------------
# Sort by silhouette_score DESC
# ---------------------------
df_sorted = df.sort_values(by="silhouette_score", ascending=False).reset_index(drop=True)

st.subheader("📌 PCA + KMeans Evaluation (sorted by silhouette_score — best first)")
st.dataframe(df_sorted, use_container_width=True, height=500)

# ---------------------------
# PCA Summary (static)
# ---------------------------
st.markdown("---")
st.subheader("📘 PCA Dimensionality Summary")
st.markdown("""
**Original dimensions:** `60`  
**Reduced dimensions (~80% variance):** `8`  
**Explained variance ratio:** `0.7113016880762798`
""")

# ---------------------------
# Top Features — PC1
# ---------------------------
st.subheader("🔥 Top 5 Features — Principal Component 1")
st.code("""num__Latitude                   0.39958762169188605
num__Y Coordinate               0.3992414092615185
num__District                   0.3927917259621543
num__Beat                       0.3828689358748723
num__X Coordinate               0.3359302670888766""")

st.markdown("""
These features have the **highest influence on Principal Component 1 (PC1)**.  
PC1 mainly captures **geographic variation** in crime:

- **Latitude & Y Coordinate** → Capture north–south spatial shifts  
- **District & Beat** → Administrative and policing boundaries  
- **X Coordinate** → East–west spatial patterns  

🔍 *Interpretation:*  
PC1 represents **where crimes are happening** — geography-driven differentiation.
""")


# ---------------------------
# Top Features — PC2
# ---------------------------
st.subheader("🔥 Top 5 Features — Principal Component 2")
st.code("""num__Arrest                     0.5090016518064672
num__Year                       0.43838487558949324
num__crime_severity_score       0.3830891066017058
num__Second                     0.36608033148258384
num__Minute                     0.2588086690855801""")

st.markdown("""
These features contribute the most to **Principal Component 2 (PC2)**.  
PC2 captures **temporal + behavioral crime characteristics**:

- **Arrest** → Whether the case led to an arrest  
- **Year** → Trend changes across years  
- **Severity Score** → How serious the incident is  
- **Second & Minute** → Micro-temporal patterns within each occurrence  

🔍 *Interpretation:*  
PC2 represents **how crimes behave over time** — severity, arrest likelihood, and fine-grained timing.
""")



st.markdown("---")
st.success("Sorted results displayed successfully.")
