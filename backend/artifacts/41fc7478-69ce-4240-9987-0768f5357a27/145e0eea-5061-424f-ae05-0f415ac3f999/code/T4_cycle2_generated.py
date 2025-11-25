# Template-based code generation - CYCLE 2
# Task: T4 - Risk Factor Analysis
# Model: RandomForest
# Recommendations applied: ['Consider exploring more advanced ML models or techniques to further improve accuracy and potentially reach the target of >= 0.9.', 'Investigate the use of transfer learning or domain adaptation methods to leverage pre-trained models and adapt them to the specific task at hand.']

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models and their corresponding metrics
models = {
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', solver='lbfgs')
}

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Save results to directory
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(os.path.join(results_dir, f'{model_name}_metrics.json'), 'w') as f:
        import json
        json.dump(metrics, f)

# Print results
for model_name, metrics in models.items():
    print(f'Model: {model_name}')
    for metric, value in metrics.items():
        print(f'{metric}: {value}')
    print()