# Template-based code generation - CYCLE 1
# Task: T2 - Baseline Model Selection
# Model: RandomForest


# Model Training - WORKING TEMPLATE
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

print("Loading preprocessed data...")
X_train = np.load(data_dir / "X_train.npy")
X_test = np.load(data_dir / "X_test.npy")
y_train = np.load(data_dir / "y_train.npy")
y_test = np.load(data_dir / "y_test.npy")

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Handle class imbalance with SMOTE (optional)
try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied: {X_train.shape} -> {X_train_res.shape}")
except Exception as e:
    print(f"SMOTE skipped: {e}")
    X_train_res, y_train_res = X_train, y_train

# Initialize and train model
print("Training RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)
model.fit(X_train_res, y_train_res)

# Predictions
y_pred = model.predict(X_test)
try:
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
except:
    auc = 0.0

# Calculate metrics
metrics = {
    "model": "RandomForestClassifier",
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred, average='weighted')),
    "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
    "roc_auc": float(auc)
}

print(f"Results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")

# Save model and results
with open(models_dir / "model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(models_dir / "results.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Model saved to {models_dir}/model.pkl")
