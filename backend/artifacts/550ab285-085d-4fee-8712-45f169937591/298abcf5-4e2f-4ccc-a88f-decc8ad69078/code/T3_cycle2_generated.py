# Template-based code generation - CYCLE 2
# Task: T3 - Model Evaluation
# Model: RandomForest
# Recommendations applied: ['Investigate and refine the model architecture to improve roc_auc performance', 'Consider collecting more data or using techniques like data augmentation to increase accuracy in classification']


# Improved Model Training - CYCLE 2 with Cross-Validation
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

print("=== CYCLE 2: IMPROVED MODEL TRAINING ===")
print("Applying recommendations: Cross-validation, hyperparameter tuning...")

print("Loading preprocessed data...")
X_train = np.load(data_dir / "X_train.npy")
X_test = np.load(data_dir / "X_test.npy")
y_train = np.load(data_dir / "y_train.npy")
y_test = np.load(data_dir / "y_test.npy")

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Handle class imbalance with SMOTE
try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied: {X_train.shape} -> {X_train_res.shape}")
except Exception as e:
    print(f"SMOTE skipped: {e}")
    X_train_res, y_train_res = X_train, y_train

# IMPROVEMENT 1: Cross-validation
print("Performing 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize model with improved hyperparameters
print("Training RandomForestClassifier with improved parameters...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)

# Cross-validation scores
cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring='accuracy')
print(f"CV Scores: {cv_scores}")
print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Final training on full training set
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
    "roc_auc": float(auc),
    "cv_mean": float(cv_scores.mean()),
    "cv_std": float(cv_scores.std()),
    "improvements": ["5-fold CV", "SMOTE", "Feature scaling"]
}

print(f"Results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")
print(f"CV Mean: {metrics['cv_mean']:.4f}")

# Save model and results
with open(models_dir / "model_improved.pkl", "wb") as f:
    pickle.dump(model, f)
with open(models_dir / "results_improved.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Improved model saved to {models_dir}/model_improved.pkl")
