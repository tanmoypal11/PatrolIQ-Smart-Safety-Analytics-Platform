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
    # NOTE: This assumes a specific file name after extraction, adjust if necessary
    target_csv = "pca_data.csv" # Placeholder for the actual extracted CSV name
    
    if not os.path.exists(os.path.join(EXTRACT_DIR, target_csv)):
        try:
            with py7zr.SevenZipFile(PCA_7Z_PATH, mode='r') as z:
                z.extractall(path=EXTRACT_DIR)
            st.success("✓ Extracted PCA .7z successfully!")
        except FileNotFoundError:
             st.error(f"⚠ PCA 7z file not found at: {PCA_7Z_PATH}. Cannot load data.")
             return None

    # Find and load the first CSV file found in the extracted directory
    for f in os.listdir(EXTRACT_DIR):
        if f.endswith(".csv"):
            return pd.read_csv(os.path.join(EXTRACT_DIR, f))
    return None


with st.spinner("Loading PCA reduced data..."):
    df_pca = extract_and_load_pca()

if df_pca is None:
    st.error("⚠ No CSV detected inside the .7z archive or failed extraction.")
    st.stop()

st.success("✓ PCA dataset loaded!")
st.dataframe(df_pca.head())

st.markdown("---")

# ------------------------------------------------------------
# 2. Identify PCA column names and Centroid
# ------------------------------------------------------------

possible_pca_names = ["PC1", "PC2", "pc1", "pc2", "pca1", "pca2"]
pca_cols = sorted([c for c in df_pca.columns if c.lower() in possible_pca_names])

if len(pca_cols) < 2:
    st.error(f"Dataset does not contain PCA columns. Found columns: {list(df_pca.columns)}")
    st.stop()

st.write("### 🔍 PCA Columns Detected")
st.code(pca_cols)

st.header("📌 Current PCA Component Means (Centroid)")
centroid = df_pca[pca_cols].mean()
st.dataframe(centroid.to_frame("Current Mean Value"))

st.markdown("---")

# ------------------------------------------------------------
# 3. ROBUST DRIFT CHECK: Mean and Volatility (Without Assumed Values)
# ------------------------------------------------------------

st.header("📡 Robust Drift Stability Check (Requires Reference Data)")

# --- 3a. Placeholder for Reference Values (CRITICAL STEP) ---
# 🛑 IMPORTANT: You MUST load your reference statistics here.
# These values (mean and std dev) are calculated ONLY on the training data.

try:
    # -------------------------------------------------------------------------
    # 🚨 REPLACE THIS SECTION WITH YOUR ACTUAL DATA LOADING LOGIC
    # Example: loading from a saved JSON file that contains the stats
    # reference_stats = pd.read_json("path/to/pca_reference_stats.json")
    # reference_means = reference_stats["mean"]
    # reference_std = reference_stats["std"]
    #
    # For now, we use a structure that will cause the check to fail cleanly
    # if not replaced, as we cannot assume values.
    # -------------------------------------------------------------------------
    
    # Placeholder to force user to insert real values
    raise NotImplementedError("Reference values must be loaded here.") 

except NotImplementedError as e:
    st.error("""
    **🛑 ERROR: REFERENCE VALUES NOT LOADED 🛑**
    
    The drift check cannot run without the **Mean** and **Standard Deviation** of the 
    PCA components calculated from your **original training/reference dataset**.
    
    **Action Required:** Please replace the placeholder code in section 3a to load:
    1. `reference_means` (a Series keyed by PCA column names)
    2. `reference_std` (a Series keyed by PCA column names)
    """)
    st.stop()

# If the user replaces the placeholder logic above, the code execution will proceed below.

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
current_std = df_pca[pca_cols].std()
# Ensure division is safe by checking for zero in reference_std (though unlikely for PCA std dev)
std_drift_percent = ((current_std - reference_std) / reference_std.replace(0, 1e-6)) * 100

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
- The original **unstable Deviation Ratio** (Std Dev / Mean) was removed.
- Robust drift checks for **Centroid Shift** (Mean Drift) and **Data Spread Change** (Volatility Drift) are structured.
- **Critical:** The code currently halts and instructs you to load the statistical constants (`reference_means`, `reference_std`) from your **training data** to enable the drift checks.
""")