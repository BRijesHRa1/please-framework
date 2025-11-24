# Template-based code generation - CYCLE 1
# Task: T4 - Survival Time Prediction
# Model: RandomForest

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from joblib import load

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Split data into features and target
X = df.drop('overall_survival_time', axis=1)
y = df['overall_survival_time']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1),
    'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', solver='lbfgs')
}

# Train and evaluate models
for model_name, model in models.items():
    print(f'Training {model_name}...')
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    
    print(f'{model_name} Metrics:')
    print('Accuracy:', accuracy)
    print('Classification Report:\n', report)
    print('Confusion Matrix:\n', matrix)
    
    # Save model to file
    joblib.dump(model, models_dir / f'{model_name}.joblib')
    
    # Save metrics to results directory
    with open(results_dir / f'{model_name}_metrics.json', 'w') as f:
        import json
        metrics = {
            'Accuracy': accuracy,
            'Classification Report': report,
            'Confusion Matrix': matrix
        }
        json.dump(metrics, f)