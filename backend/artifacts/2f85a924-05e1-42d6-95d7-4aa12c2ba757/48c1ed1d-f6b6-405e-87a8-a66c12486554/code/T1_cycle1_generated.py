# Template-based code generation - CYCLE 1
# Task: T1 - Data Preprocessing
# Model: RandomForest


# Data Preprocessing - WORKING TEMPLATE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"Dataset shape: {df.shape}")

# Target column
target_col = 'vital_status'

# Get feature columns (exclude metadata)
exclude_cols = ['sampleID', 'vital_status', 'survival_time_days']
feature_cols = [c for c in df.columns if c not in exclude_cols]

# Use learner-suggested genes if available, otherwise use all features
learner_genes = ['BARD1', 'PALB2', 'PTEN', 'BRCA1', 'ATM', 'SEM1', 'PGR', 'FANCD2', 'HSP90AA1', 'ERBB2', 'EP300', 'MDC1', 'TP53', 'HDAC1', 'BRCA2', 'EGFR', 'PIK3CA', 'SFN', 'ESR1', 'CHEK2']
available_genes = [g for g in learner_genes if g in df.columns]
if len(available_genes) < 10:
    print(f"Using all {len(feature_cols)} features (learner genes not found)")
    available_genes = feature_cols[:500]
else:
    print(f"Using {len(available_genes)} learner-suggested genes")

X = df[available_genes].values
y = df[target_col].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Class balance - Train: {y_train.mean():.2f}, Test: {y_test.mean():.2f}")

# Save preprocessed data
np.save(data_dir / "X_train.npy", X_train)
np.save(data_dir / "X_test.npy", X_test)
np.save(data_dir / "y_train.npy", y_train)
np.save(data_dir / "y_test.npy", y_test)

# Save feature names
import json
with open(data_dir / "feature_names.json", 'w') as f:
    json.dump(available_genes, f)

metrics = {
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_features": int(len(available_genes)),
    "class_balance_train": float(y_train.mean()),
    "class_balance_test": float(y_test.mean())
}
print(f"Preprocessing complete: {metrics}")
