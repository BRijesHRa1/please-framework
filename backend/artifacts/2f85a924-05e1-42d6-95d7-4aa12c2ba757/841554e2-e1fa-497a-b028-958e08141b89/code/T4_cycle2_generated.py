# Template-based code generation - CYCLE 2
# Task: T4 - Survival Time Prediction
# Model: RandomForest
# Recommendations applied: ['Investigate and refine the methodology to improve the performance on the ROC-AUC score in future iterations.', 'Continuously monitor and adjust hyperparameters to optimize model performance without compromising accuracy.']

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

# Define the dataset path and data directory
DATASET_PATH = 'path_to_dataset.csv'
data_dir = pd.read_csv(DATASET_PATH)

# Split the data into features (X) and target variable (y)
X = data_dir.drop('target', axis=1)
y = data_dir['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models directory and results directory
models_dir = 'path_to_models'
results_dir = 'path_to_results'

# Create the models directory if it does not exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Select a model (for example, RandomForestClassifier)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the target variable for the testing data
y_pred = model.predict(X_test)

# Set up a dictionary to store the metrics
metrics = {}

# Calculate and store the accuracy score
accuracy = accuracy_score(y_test, y_pred)
metrics['accuracy'] = accuracy

# Calculate and store the classification report
report = classification_report(y_test, y_pred)
metrics['classification_report'] = report

# Calculate and store the confusion matrix
matrix = confusion_matrix(y_test, y_pred)
metrics['confusion_matrix'] = matrix

# Save the trained model to the models directory
model.save(models_dir + '/random_forest_model.pkl')

# Print the metrics
print(metrics)

# Plot the results (optional)
import matplotlib.pyplot as plt

plt.bar(metrics.keys(), [value for value in metrics.values()])
plt.xlabel('Metric')
plt.ylabel('Value')
plt.title('Model Performance Metrics')
plt.show()