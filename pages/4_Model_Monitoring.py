import streamlit as st
import pandas as pd
import py7zr
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Model Monitoring", layout="wide")
st.title("📊 Model Monitoring Dashboard")

# ------------------------------------------------------------
# 1. Extract PCA .7z → Load CSV
# ------------------------------------------------------------

PCA_7Z_PATH = "data/PatrolIQ_dimred_pca.7z"
EXTRACT_DIR = "extracted_pca"

os.makedirs(EXTRACT_DIR, exist_ok=True)

def extract_and_load_pca():
    """Extracts .7z and loads contained CSV."""
    if len(os.listdir(EXTRACT_DIR)) == 0:
        with py7zr.SevenZipFile(PCA_7Z_PATH, mode='r') as z:
            z.extractall(path=EXTRACT_DIR)
        st.success("✓ Extracted PCA .7z successfully!")

    for f in os.listdir(EXTRACT_DIR):
        if f.endswith(".csv"):
            return pd.read_csv(os.path.join(EXTRACT_DIR, f))

    return None


with st.spinner("Loading PCA reduced data..."):
    df_pca = extract_and_load_pca()

if df_pca is None:
    st.error("⚠ No CSV detected inside the .7z archive.")
    st.stop()

st.success("✓ PCA dataset loaded!")
st.dataframe(df_pca.head())

st.markdown("---")

# ------------------------------------------------------------
# 2. Identify PCA column names
# ------------------------------------------------------------

# Works for both: ["PC1", "PC2"] OR ["pca1", "pca2"]
possible_pca_names = ["PC1", "PC2", "pc1", "pc2", "pca1", "pca2"]
pca_cols = [c for c in df_pca.columns if c.lower() in possible_pca_names]

if len(pca_cols) < 2:
    st.error(f"Dataset does not contain PCA columns. Found columns: {list(df_pca.columns)}")
    st.stop()

# Sort by component order
pca_cols = sorted(pca_cols)

st.write("### 🔍 PCA Columns Detected")
st.code(pca_cols)

st.markdown("---")

# ------------------------------------------------------------
# 3. Compute PCA centroid (mean)
# ------------------------------------------------------------

st.header("📌 PCA Component Means (Centroid)")

centroid = df_pca[pca_cols].mean()
st.dataframe(centroid.to_frame("Mean Value"))

st.markdown("---")

# ------------------------------------------------------------
# 4. Drift Check (Simple Stability Test)
# ------------------------------------------------------------

st.header("📡 Drift Stability Check")

threshold = 0.10  # 10% drift allowed
deviation = df_pca[pca_cols].std() / df_pca[pca_cols].mean()

st.write("### Deviation Ratio:")
st.dataframe(deviation.to_frame("Deviation Ratio"))

st.write("### Drift Status")
for comp, dev in deviation.items():
    if dev > threshold:
        st.error(f"⚠ {comp}: Drift detected ({dev:.3f})")
    else:
        st.success(f"✓ {comp}: Stable ({dev:.3f})")

st.markdown("---")

# ------------------------------------------------------------
# 5. Scatter Plot — PCA Space
# ------------------------------------------------------------

st.header("📊 PCA Scatter — First 2 Components")

fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(df_pca[pca_cols[0]], df_pca[pca_cols[1]])
ax.set_xlabel(pca_cols[0])
ax.set_ylabel(pca_cols[1])
ax.set_title("PCA Scatter Plot")

st.pyplot(fig)

st.markdown("---")

# ------------------------------------------------------------
# 6. Summary
# ------------------------------------------------------------

st.header("📝 Summary")
st.write("""
- PCA data extracted from `.7z`
- PCA columns auto-detected (`pca1`, `pca2`)
- Drift stability check performed
- PCA centroid calculated
- PCA scatter plot generated
""")
