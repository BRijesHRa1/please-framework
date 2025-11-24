# Template-based code generation - CYCLE 2
# Task: T1 - Data Preprocessing
# Model: RandomForest
# Recommendations applied: ['Investigate and refine the methodology to improve the performance on the ROC-AUC score in future iterations.', 'Continuously monitor and adjust hyperparameters to optimize model performance without compromising accuracy.']


# Improved Data Preprocessing - CYCLE 2 with Feature Engineering
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

print("=== CYCLE 2: IMPROVED PREPROCESSING ===")
print("Applying recommendations from Cycle 1...")

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"Dataset shape: {df.shape}")

# Target column
target_col = 'vital_status'

# Get feature columns (exclude metadata)
exclude_cols = ['sampleID', 'vital_status', 'survival_time_days']
feature_cols = [c for c in df.columns if c not in exclude_cols]

# Use learner-suggested genes
learner_genes = ['BARD1', 'PALB2', 'PTEN', 'BRCA1', 'ATM', 'SEM1', 'PGR', 'FANCD2', 'HSP90AA1', 'ERBB2', 'EP300', 'MDC1', 'TP53', 'HDAC1', 'BRCA2', 'EGFR', 'PIK3CA', 'SFN', 'ESR1', 'CHEK2']
available_genes = [g for g in learner_genes if g in df.columns]
if len(available_genes) < 10:
    print(f"Using all {len(feature_cols)} features")
    available_genes = feature_cols[:500]
else:
    print(f"Using {len(available_genes)} learner-suggested genes")

X = df[available_genes].values
y = df[target_col].values

# IMPROVEMENT 1: Feature Engineering - Standard Scaling
print("Applying StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# IMPROVEMENT 2: Feature Selection - SelectKBest
print("Applying feature selection (SelectKBest)...")
k_features = min(15, X_scaled.shape[1])  # Top 15 features
selector = SelectKBest(f_classif, k=k_features)
X_selected = selector.fit_transform(X_scaled, y)
selected_indices = selector.get_support(indices=True)
selected_genes = [available_genes[i] for i in selected_indices]
print(f"Selected {len(selected_genes)} best features: {selected_genes[:5]}...")

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Class balance - Train: {y_train.mean():.2f}, Test: {y_test.mean():.2f}")

# Save preprocessed data
np.save(data_dir / "X_train.npy", X_train)
np.save(data_dir / "X_test.npy", X_test)
np.save(data_dir / "y_train.npy", y_train)
np.save(data_dir / "y_test.npy", y_test)

# Save feature names and scaler
import json
import pickle
with open(data_dir / "feature_names.json", 'w') as f:
    json.dump(selected_genes, f)
with open(data_dir / "scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
with open(data_dir / "selector.pkl", 'wb') as f:
    pickle.dump(selector, f)

metrics = {
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_features": int(len(selected_genes)),
    "n_original_features": len(available_genes),
    "class_balance_train": float(y_train.mean()),
    "class_balance_test": float(y_test.mean()),
    "improvements": ["StandardScaler", "SelectKBest feature selection"]
}
print(f"Improved preprocessing complete: {metrics}")
