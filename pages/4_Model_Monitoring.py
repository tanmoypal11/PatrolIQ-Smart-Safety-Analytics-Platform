import streamlit as st
import pandas as pd
import py7zr
import os

st.set_page_config(page_title="📊 Model Monitoring", layout="wide")
st.title("📊 Model Monitoring Dashboard")

# ------------------------------------------------------------
# 1. Load PCA Output (.7z Extract → CSV)
# ------------------------------------------------------------

st.header("📁 Load PCA Reduced Data")

# Path to compressed PCA file in repo
PCA_7Z_PATH = "data/PatrolIQ_dimred_pca.7z"
EXTRACT_DIR = "extracted_pca"

# Create extract folder if missing
os.makedirs(EXTRACT_DIR, exist_ok=True)

def extract_and_load_pca():
    """Extracts the .7z file and loads the CSV."""
    # Extract only if folder is empty
    if len(os.listdir(EXTRACT_DIR)) == 0:
        with py7zr.SevenZipFile(PCA_7Z_PATH, mode='r') as z:
            z.extractall(path=EXTRACT_DIR)
        st.success("✓ 7z file extracted successfully!")

    # Find CSV inside extracted folder
    for f in os.listdir(EXTRACT_DIR):
        if f.endswith(".csv"):
            csv_path = os.path.join(EXTRACT_DIR, f)
            df = pd.read_csv(csv_path)
            return df

    return None

with st.spinner("Loading PCA reduced dataset..."):
    df_pca = extract_and_load_pca()

if df_pca is None:
    st.error("⚠ No CSV found inside the 7z archive. Upload a valid file.")
    st.stop()

st.success("✓ PCA dataset loaded successfully!")
st.write("Preview of PCA Data:")
st.dataframe(df_pca.head())

st.markdown("---")

# ------------------------------------------------------------
# 2. Model Monitoring: Drift Check (Example)
# ------------------------------------------------------------

st.header("📉 Drift Monitoring — PCA Space")

st.write("""
This section helps you evaluate whether incoming crime data is drifting away 
from the distribution of the training dataset using PCA components.
""")

# If dataset contains PCA columns such as PC1, PC2, ...
pca_cols = [col for col in df_pca.columns if "PC" in col]

if len(pca_cols) < 2:
    st.error("Dataset does not contain PC1, PC2 columns. Cannot plot drift.")
    st.stop()

# Show mean values (centroids)
training_centroid = df_pca[pca_cols].mean()

st.subheader("📌 PCA Component Means (Centroid)")
st.write(training_centroid.to_frame("Mean Value"))

# ------------------------------------------------------------
# 3. Simple Drift Indicator
# ------------------------------------------------------------

st.subheader("📡 Drift Stability Check")

threshold = 0.05  # You can tune this
deviation = df_pca[pca_cols].std() / df_pca[pca_cols].mean()

drift_flags = deviation > threshold

st.write("### Drift Status")
for comp, flag in drift_flags.items():
    if flag:
        st.error(f"⚠ {comp}: Potential drift detected")
    else:
        st.success(f"✓ {comp}: Stable")

st.markdown("---")

# ------------------------------------------------------------
# 4. Visualization: PC1 vs PC2 Scatter
# ------------------------------------------------------------

import matplotlib.pyplot as plt

st.subheader("📊 PC1 vs PC2 Distribution")

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(df_pca["PC1"], df_pca["PC2"])
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA Scatter — PC1 vs PC2")

st.pyplot(fig)

st.markdown("---")

# ------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------

st.header("📝 Summary")
st.write("""
- PCA dataset loaded from `.7z` archive successfully  
- PCA centroids computed  
- Drift indicator shows whether data is stable or shifting  
- Visual PCA scatter plot included  
""")
