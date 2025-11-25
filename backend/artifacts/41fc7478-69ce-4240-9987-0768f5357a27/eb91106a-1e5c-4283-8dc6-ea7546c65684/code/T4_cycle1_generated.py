# Template-based code generation - CYCLE 1
# Task: T4 - Risk Factor Analysis
# Model: RandomForest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Split data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', solver='lbfgs')
}

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    # Save results to directory
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(os.path.join(results_dir, f'{model_name}_metrics.txt'), 'w') as f:
        f.write(f'Model: {model_name}\n')
        f.write('Accuracy: {}\n'.format(accuracy))
        f.write('Classification Report:\n{}\n'.format(report))
        f.write('Confusion Matrix:\n{}\n'.format(matrix))

# Set metrics dictionary
metrics = {
    'RandomForestClassifier': accuracy,
    'GradientBoostingClassifier': accuracy,
    'LogisticRegression': accuracy
}