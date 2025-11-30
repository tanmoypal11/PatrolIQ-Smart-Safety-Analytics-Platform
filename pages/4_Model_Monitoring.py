import streamlit as st
import pandas as pd
import py7zr
import os
import plotly.express as px
from io import BytesIO

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
    # Logic to handle extraction is kept, but data loading is streamlined
    for f in os.listdir(EXTRACT_DIR):
        if f.endswith(".csv"):
            return pd.read_csv(os.path.join(EXTRACT_DIR, f))
    
    # If not found, attempt extraction
    try:
        with py7zr.SevenZipFile(PCA_7Z_PATH, mode='r') as z:
            z.extractall(path=EXTRACT_DIR)
        st.success("✓ Extracted PCA .7z successfully!")
        
        for f in os.listdir(EXTRACT_DIR):
            if f.endswith(".csv"):
                return pd.read_csv(os.path.join(EXTRACT_DIR, f))
    except (FileNotFoundError, py7zr.exceptions.No7ZFileError) as e:
        st.error(f"⚠ Failed to load or extract PCA data: {e}. Check path: {PCA_7Z_PATH}")
        return None
    return None


with st.spinner("Loading PCA reduced data..."):
    df_pca = extract_and_load_pca()

if df_pca is None:
    st.error("⚠ Failed to load PCA data.")
    st.stop()

# ------------------------------------------------------------
# 2. Data Preparation and Splitting (Simulating Reference vs. Current)
# ------------------------------------------------------------

# 🛑 CRITICAL STEP: Simulate time series split for drift check
# Assuming your crime data has a 'Date' or 'Time' column.
# If your data has a date column, replace 'date_col_name' with its name.
if 'date' in df_pca.columns.str.lower():
    date_col = df_pca.columns[df_pca.columns.str.lower() == 'date'][0]
    df_pca[date_col] = pd.to_datetime(df_pca[date_col], errors='coerce')
    df_pca.dropna(subset=[date_col], inplace=True)
    
    # Use a date near the end of the dataset for a realistic split
    # You will need to adjust this date based on your actual data range (e.g., 2024-01-01)
    SPLIT_DATE = df_pca[date_col].max() - pd.DateOffset(months=3)

    df_ref = df_pca[df_pca[date_col] < SPLIT_DATE]
    df_current = df_pca[df_pca[date_col] >= SPLIT_DATE]
    
    st.info(f"Data split into **Reference** (pre-{SPLIT_DATE.date()}) and **Current** (post-{SPLIT_DATE.date()}).")
    st.write(f"Reference size: {len(df_ref):,} rows | Current size: {len(df_current):,} rows")

else:
    st.warning("⚠️ No 'date' column found. Falling back to simple 80/20 train/test split for demo.")
    split_idx = int(len(df_pca) * 0.8)
    df_ref = df_pca.iloc[:split_idx]
    df_current = df_pca.iloc[split_idx:]
    
    # This scenario is less realistic for time-series drift but allows the dashboard to run.

# Identify PCA columns
possible_pca_names = ["PC1", "PC2", "pc1", "pc2", "pca1", "pca2"]
pca_cols = sorted([c for c in df_pca.columns if c.lower() in possible_pca_names])

if len(pca_cols) < 2:
    st.error(f"Dataset does not contain PCA columns. Found columns: {list(df_pca.columns)}")
    st.stop()

st.success("✓ PCA dataset loaded and split!")
st.markdown("---")

# ------------------------------------------------------------
# 3. Reference Statistics (The "How to Calculate Drift" Baseline)
# ------------------------------------------------------------

st.header("📏 Reference Statistics (Training/Historical Data)")

# 1. Calculate Reference Statistics from the historical/training data (df_ref)
reference_means = df_ref[pca_cols].mean()
reference_std = df_ref[pca_cols].std()
current_means = df_current[pca_cols].mean()
current_std = df_current[pca_cols].std()


st.dataframe(pd.DataFrame({
    "Reference Mean": reference_means,
    "Current Mean": current_means,
    "Reference Std Dev": reference_std,
    "Current Std Dev": current_std,
}))

st.markdown("---")

# ------------------------------------------------------------
# 4. ROBUST DRIFT CHECK: Mean and Volatility
# ------------------------------------------------------------

st.header("📡 Drift Stability Check Results")
st.write("We compare the **Current** data statistics against the **Reference** data statistics.")

# --- 4a. Mean Drift Check (Shift in Centroid) ---

st.subheader("1. Centroid (Mean) Drift")
# CALCULATE DRIFT: Absolute difference in means
mean_drift = (current_means - reference_means).abs()
st.dataframe(mean_drift.to_frame("Absolute Mean Drift"))

mean_threshold = 0.05  # Absolute shift threshold
for comp, drift in mean_drift.items():
    if drift > mean_threshold:
        st.error(f"⚠ **{comp}**: Significant Mean Shift detected ({drift:.4f}). **Data Centroid has moved!**")
    else:
        st.success(f"✓ **{comp}**: Mean Stable ({drift:.4f})")


# --- 4b. Volatility Drift Check (Shift in Spread/Variance) ---

st.subheader("2. Volatility (Std Dev) Drift")
# CALCULATE DRIFT: Percentage change in standard deviation
# We use .replace(0, 1e-6) to avoid DivisionByZero errors, though STD should not be zero.
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
# 5. Scatter Plot — PCA Space
# ------------------------------------------------------------

st.header("📊 PCA Scatter — Reference vs. Current Data")

# Combine dataframes for plotting with a label
df_plot = pd.concat([
    df_ref.assign(Dataset="Reference"),
    df_current.assign(Dataset="Current")
])

fig = px.scatter(
    df_plot,
    x=pca_cols[0],
    y=pca_cols[1],
    color="Dataset", # Color-code by Reference/Current
    symbol="Dataset",
    hover_data=pca_cols,
    title=f"PCA Scores Plot: {pca_cols[0]} vs {pca_cols[1]}"
)

# Mark the center of the Reference data (Reference Centroid)
fig.add_shape(
    type="circle", 
    x0=reference_means[pca_cols[0]] - 0.1, y0=reference_means[pca_cols[1]] - 0.1, 
    x1=reference_means[pca_cols[0]] + 0.1, y1=reference_means[pca_cols[1]] + 0.1, 
    xref='x', yref='y', fillcolor='Blue', opacity=0.8, layer='above', line_width=0,
    name="Ref Centroid"
)

st.plotly_chart(fig, use_container_width=True)