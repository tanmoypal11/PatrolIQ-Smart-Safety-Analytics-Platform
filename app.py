import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="PatrolIQ – Crime Intelligence System",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------
# Title & Subtitle
# ---------------------------
st.title("🛡️ PatrolIQ – Crime Intelligence & Clustering System")
st.markdown("### A complete machine learning pipeline for understanding crime patterns in Chicago")

st.markdown("---")

# ---------------------------
# Project Overview
# ---------------------------
st.header("📌 Project Overview")

st.markdown("""
PatrolIQ is a full end-to-end machine learning project that ingests real crime data,
performs advanced analytics, clusters geographic & temporal crime patterns, and provides
interactive visualizations through a multi-page Streamlit web app.

This system helps identify **crime hotspots**, understand **time-based crime patterns**, and
analyze **feature importance** using PCA and t-SNE.
""")

st.markdown("---")

# ---------------------------
# Step-by-Step Pipeline
# ---------------------------
st.header("⚙️ Step-by-Step ML Pipeline")

st.markdown("""
### **1️⃣ Data Acquisition & Preprocessing**
- Fetched **500,000 recent crime records** using the Chicago Socrata API.
- Cleaned missing values.
- Extracted temporal features:  
  `Hour`, `DayOfWeek`, `Month`.
- Added custom **severity score**.
- Converted coordinates, removed outliers.

---

### **2️⃣ Exploratory Data Analysis (EDA)**
- Plotly visualizations for:
  - Crime trends over time  
  - Crime type distributions  
  - Daily/weekly patterns  

---

### **3️⃣ Feature Engineering**
- Scaled numerical features using **StandardScaler**.
- One-hot encoded categorical features using **sklearn OneHotEncoder**.
- Removed low-variance & redundant columns.

---

### **4️⃣ Clustering**
- **Geographic Clustering**  
  - K-Means  
  - DBSCAN  
  - Agglomerative  
  - Evaluated with Silhouette Score (TARGET > 0.5)

- **Temporal Clustering**  
  - K-Means on hour/day patterns  
  - Identified crime-peak times  

All models saved as `.pkl` and integrated into Streamlit.

---

### **5️⃣ Dimensionality Reduction**
- **PCA** → Reduced 22+ features to **2–3 components**, achieving **70%+ variance**.
- **t-SNE** → High-quality 2D visualization separating crime clusters.

Feature importance extracted from PCA loadings.

---

### **6️⃣ MLflow Tracking**
- Logged:
  - Clustering parameters  
  - Silhouette scores  
  - PCA variance ratios  
  - t-SNE hyperparameters  
- Stored models & artifacts in `mlruns/`

---

### **7️⃣ Streamlit Application**
This app includes:
- 🗺️ **Geographic Heatmap**  
- ⏱️ **Temporal Pattern Analysis**  
- 🔍 **Dimensionality Reduction (PCA / t-SNE)**  
- 📊 **Model Monitoring Dashboard (MLflow Integration)**  

Each module is in the `pages/` folder.

---

### **8️⃣ Deployment Pipeline**
- Prepared production folder structure  
- Added `requirements.txt`  
- Deployable to **Streamlit Cloud** via GitHub  

""")

st.markdown("---")

# ---------------------------
# Expected Results Section
# ---------------------------
st.header("🎯 Expected Results")

st.markdown("""
- **Geographic clusters:** 5–10 stable hotspots  
- **Temporal clusters:** 3–5 meaningful patterns  
- **PCA:** 70–85% variance retention  
- **Visuals:** Heatmaps, t-SNE plots, PCA scatter  
- **Fully interactive multi-page dashboard**  
""")

st.success("Use the left sidebar to navigate between analysis modules.")

st.markdown("---")