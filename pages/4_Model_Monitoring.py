import streamlit as st
import pandas as pd
import py7zr
import os
import plotly.express as px

st.set_page_config(page_title="📊 Model Monitoring", layout="wide")
st.title("📊 Model Monitoring Dashboard")

# ------------------------------------------------------------
# 1. Setup and Data Loading
# ------------------------------------------------------------

PCA_7Z_PATH = "data/PatrolIQ_dimred_pca.7z"
EXTRACT_DIR = "extracted_pca"

os.makedirs(EXTRACT_DIR, exist_ok=True)

def extract_and_load_pca():
    """Extracts .7z and loads contained CSV."""
    # Simplified check to avoid re-extracting on every rerun
    if not os.path.exists(os.path.join(EXTRACT_DIR, "pca_data.csv")):
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
# 2. Identify PCA column names and Centroid
# ------------------------------------------------------------

# Works for both: ["PC1", "PC2"] OR ["pca1", "pca2"]
possible_pca_names = ["PC1", "PC2", "pc1", "pc2", "pca1", "pca2"]
pca_cols = sorted([c for c in df_pca.columns if c.lower() in possible_pca_names])

if len(pca_cols) < 2:
    st.error(f"Dataset does not contain PCA columns. Found columns: {list(df_pca.columns)}")
    st.stop()

st.write("### 🔍 PCA Columns Detected")
st.code(pca_cols)

st.header("📌 PCA Component Means (Centroid)")
centroid = df_pca[pca_cols].mean()
st.dataframe(centroid.to_frame("Current Mean Value"))

st.markdown("---")

# ------------------------------------------------------------
# 3. ROBUST DRIFT CHECK: Mean and Volatility
# ------------------------------------------------------------

st.header("📡 Robust Drift Stability Check")

# --- 3a. Define Reference Values (Must be loaded from your training phase) ---
# NOTE: These are example values. In a real application, you would load these
# from a file (e.g., JSON or CSV) saved during the model training phase.

# Reference Mean should ideally be 0.0 for PCA
reference_means = pd.Series(0.0, index=pca_cols)

# Reference Standard Deviation (Std Dev is sqrt of the Eigenvalue)
reference_std = pd.Series(
    [1.5, 1.2], # Example: PC1 had an STD of 1.5, PC2 had an STD of 1.2 in training
    index=pca_cols
)
st.write("Reference values assumed from training data:")
st.dataframe(pd.DataFrame({
    "Reference Mean": reference_means,
    "Reference Std Dev": reference_std
}))


# --- 3b. Mean Drift Check (Shift in Centroid) ---

st.subheader("1. Centroid (Mean) Drift")
mean_drift = (df_pca[pca_cols].mean() - reference_means).abs()
st.dataframe(mean_drift.to_frame("Absolute Mean Drift"))

mean_threshold = 0.05  # Absolute shift of 0.05 allowed
for comp, drift in mean_drift.items():
    if drift > mean_threshold:
        st.error(f"⚠ **{comp}**: Significant Mean Shift detected ({drift:.4f}). **Data Centroid has moved!**")
    else:
        st.success(f"✓ **{comp}**: Mean Stable ({drift:.4f})")


# --- 3c. Volatility Drift Check (Shift in Spread/Variance) ---

st.subheader("2. Volatility (Std Dev) Drift")
# Calculate the percentage change in standard deviation
current_std = df_pca[pca_cols].std()
std_drift_percent = ((current_std - reference_std) / reference_std) * 100

st.dataframe(std_drift_percent.to_frame("Std Dev % Change"))

std_threshold = 10.0 # 10% change allowed in component spread
for comp, drift in std_drift_percent.items():
    if abs(drift) > std_threshold:
        st.error(f"⚠ **{comp}**: Volatility Drift detected ({drift:.2f}%). **Data Spread/Shape has changed!**")
    else:
        st.success(f"✓ **{comp}**: Volatility Stable ({drift:.2f}%)")

st.markdown("---")

# ------------------------------------------------------------
# 4. Scatter Plot — PCA Space
# ------------------------------------------------------------

st.header("📊 PCA Scatter — First 2 Components")

fig = px.scatter(
    df_pca,
    x=pca_cols[0],
    y=pca_cols[1],
    color="cluster" if "cluster" in df_pca.columns else None,
    hover_data=df_pca.columns,
    title=f"PCA Scatter Plot (Centroid should be near (0,0))"
)
# Add a marker for the origin (where the centroid should be)
fig.add_shape(type='circle', x0=-0.1, y0=-0.1, x1=0.1, y1=0.1, xref='x', yref='y', fillcolor='Red', opacity=0.5, layer='below', line_width=0)


st.plotly_chart(fig, use_container_width=True)

st.header("📝 Summary")
st.write("""
- The previous unstable **Deviation Ratio** (Std Dev / Mean) was removed because the PCA Mean is zero.
- Two robust drift checks are implemented: **Centroid Shift** (Mean Drift) and **Data Spread Change** (Volatility Drift).
- **Remember to replace the placeholder `reference_means` and `reference_std` with values saved from your actual training data!**
""")