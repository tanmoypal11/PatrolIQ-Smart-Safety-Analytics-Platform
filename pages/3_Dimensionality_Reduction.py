# ======================================================
# 3_Dimensionality_Reduction.py
# ======================================================

import os
import pandas as pd
import numpy as np
import cloudpickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# -------------------------------
# Load data
# -------------------------------
DATA_PATH = r"E:\Desktop\GUVI\Project\PatrolIQ\Streamlit\data\PatrolIQ_feature_engineered.csv"
df = pd.read_csv(DATA_PATH)

# -------------------------------
# Data preprocessing
# -------------------------------
drop_cols = [
    "ID", "Case Number", "Date", "Block", "Description", 
    "Location Description", "Updated On", "Time", 
    "geo_cluster", "temp_cluster", 'IUCR', 'Location', 'FBI Code'
]

df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Map boolean columns to 0/1
df['Arrest'] = df['Arrest'].astype(str).str.lower().map({'true': 1, 'false': 0})
df['Domestic'] = df['Domestic'].astype(str).str.lower().map({'true': 1, 'false': 0})

# Column lists
cat_cols = ['Primary Type', 'DayOfWeek', 'MonthName', 'Time_of_Day']
num_cols = ['Arrest','Domestic','Beat','District','Ward','Community Area',
            'X Coordinate','Y Coordinate','Year','Latitude','Longitude',
            'Hour','Month','Day','Minute','Second','crime_severity_score']

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ]
)

# Preprocess
X_processed = preprocessor.fit_transform(df)
print("Preprocessed X shape:", X_processed.shape)

# -------------------------------
# PCA + MiniBatchKMeans experiments
# -------------------------------
pca_range = range(2, 12)       # n_components 2 to 11
k_range = range(2, 11)         # n_clusters 2 to 10

results = []

for n_pc in pca_range:
    pca = PCA(n_components=n_pc, random_state=42)
    X_pca = pca.fit_transform(X_processed)
    
    for k in k_range:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=10000)
        labels = kmeans.fit_predict(X_pca)
        
        sil_score = silhouette_score(X_pca, labels, sample_size=10000, random_state=42)
        db_score = davies_bouldin_score(X_pca, labels)
        
        results.append({
            'n_components': n_pc,
            'n_clusters': k,
            'silhouette_score': sil_score,
            'davies_bouldin_score': db_score
        })

results_df = pd.DataFrame(results)
print(results_df)

results_df.to_csv('pca_kmeans_results.csv')

# -------------------------------
# PCA for variance reduction (~80%)
# -------------------------------
pca = PCA(n_components=0.7, random_state=42)
X_pca = pca.fit_transform(X_processed)

print("Original dimensions:", X_processed.shape[1])
print("Reduced dimensions (~80% variance):", X_pca.shape[1])
print("Explained variance ratio:", np.sum(pca.explained_variance_ratio_))

# -------------------------------
# PCA with 2 components for feature importance
# -------------------------------
pca_2 = PCA(n_components=2, random_state=42)
X_pca_2 = pca_2.fit_transform(X_processed)
feature_names = preprocessor.get_feature_names_out()

for pc_idx in range(2):
    feature_importance = np.abs(pca_2.components_[pc_idx])
    top5_idx = np.argsort(feature_importance)[-5:][::-1]
    print(f"Top 5 features driving PC{pc_idx+1}:")
    for i in top5_idx:
        print(feature_names[i], feature_importance[i])
    print("-" * 40)

# -------------------------------
# Save preprocessor & processed data using cloudpickle
# -------------------------------
save_dir = r"E:\Desktop\GUVI\Project\PatrolIQ\models"
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, "preprocessor_dimred.pkl"), "wb") as f:
    cloudpickle.dump(preprocessor, f)

with open(os.path.join(save_dir, "X_processed_dimred.pkl"), "wb") as f:
    cloudpickle.dump(X_processed, f)

print("✅ Preprocessor and processed data saved successfully!")
