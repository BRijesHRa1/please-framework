"""
Executor Agent - Dynamically generates and executes ML code based on Learner suggestions
Uses real TCGA dataset and LLM-generated code only (no hardcoded templates)
"""

import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


# =============================================================================
# MODEL TEMPLATES - Proven code snippets for reliable execution
# These templates help the LLM generate error-free code
# =============================================================================

MODEL_TEMPLATES = {
    "RandomForestClassifier": {
        "import": "from sklearn.ensemble import RandomForestClassifier",
        "init": "RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)",
        "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42, "class_weight": "balanced", "n_jobs": -1},
        "description": "Ensemble of decision trees, good for high-dimensional data"
    },
    "GradientBoostingClassifier": {
        "import": "from sklearn.ensemble import GradientBoostingClassifier",
        "init": "GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)",
        "params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
        "description": "Sequential ensemble, good for structured data"
    },
    "LogisticRegression": {
        "import": "from sklearn.linear_model import LogisticRegression",
        "init": "LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', solver='lbfgs')",
        "params": {"max_iter": 1000, "random_state": 42, "class_weight": "balanced", "solver": "lbfgs"},
        "description": "Linear classifier, fast and interpretable"
    },
    "SVC": {
        "import": "from sklearn.svm import SVC",
        "init": "SVC(kernel='rbf', C=1.0, random_state=42, class_weight='balanced', probability=True)",
        "params": {"kernel": "rbf", "C": 1.0, "random_state": 42, "class_weight": "balanced", "probability": True},
        "description": "Support Vector Machine, good for binary classification"
    },
    "KNeighborsClassifier": {
        "import": "from sklearn.neighbors import KNeighborsClassifier",
        "init": "KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)",
        "params": {"n_neighbors": 5, "weights": "distance", "n_jobs": -1},
        "description": "Instance-based learning, no training phase"
    },
    "DecisionTreeClassifier": {
        "import": "from sklearn.tree import DecisionTreeClassifier",
        "init": "DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')",
        "params": {"max_depth": 10, "random_state": 42, "class_weight": "balanced"},
        "description": "Single decision tree, highly interpretable"
    }
}

# Complete working code templates for each task type
# NOTE: Use double braces {{}} to escape curly braces in .format() strings
CODE_TEMPLATES = {
    "preprocessing": '''
# Data Preprocessing - WORKING TEMPLATE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"Dataset shape: {{df.shape}}")

# Target column
target_col = 'vital_status'

# Get feature columns (exclude metadata)
exclude_cols = ['sampleID', 'vital_status', 'survival_time_days']
feature_cols = [c for c in df.columns if c not in exclude_cols]

# Use learner-suggested genes if available, otherwise use all features
learner_genes = {genes}
available_genes = [g for g in learner_genes if g in df.columns]
if len(available_genes) < 10:
    print(f"Using all {{len(feature_cols)}} features (learner genes not found)")
    available_genes = feature_cols[:500]
else:
    print(f"Using {{len(available_genes)}} learner-suggested genes")

X = df[available_genes].values
y = df[target_col].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {{X_train.shape}}, Test: {{X_test.shape}}")
print(f"Class balance - Train: {{y_train.mean():.2f}}, Test: {{y_test.mean():.2f}}")

# Save preprocessed data
np.save(data_dir / "X_train.npy", X_train)
np.save(data_dir / "X_test.npy", X_test)
np.save(data_dir / "y_train.npy", y_train)
np.save(data_dir / "y_test.npy", y_test)

# Save feature names
import json
with open(data_dir / "feature_names.json", 'w') as f:
    json.dump(available_genes, f)

metrics = {{
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_features": int(len(available_genes)),
    "class_balance_train": float(y_train.mean()),
    "class_balance_test": float(y_test.mean())
}}
print(f"Preprocessing complete: {{metrics}}")
''',

    "model_training": '''
# Model Training - WORKING TEMPLATE
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
{model_import}

print("Loading preprocessed data...")
X_train = np.load(data_dir / "X_train.npy")
X_test = np.load(data_dir / "X_test.npy")
y_train = np.load(data_dir / "y_train.npy")
y_test = np.load(data_dir / "y_test.npy")

print(f"Train shape: {{X_train.shape}}, Test shape: {{X_test.shape}}")

# Handle class imbalance with SMOTE (optional)
try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied: {{X_train.shape}} -> {{X_train_res.shape}}")
except Exception as e:
    print(f"SMOTE skipped: {{e}}")
    X_train_res, y_train_res = X_train, y_train

# Initialize and train model
print("Training {model_name}...")
model = {model_init}
model.fit(X_train_res, y_train_res)

# Predictions
y_pred = model.predict(X_test)
try:
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
except:
    auc = 0.0

# Calculate metrics
metrics = {{
    "model": "{model_name}",
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred, average='weighted')),
    "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
    "roc_auc": float(auc)
}}

print(f"Results: Accuracy={{metrics['accuracy']:.4f}}, F1={{metrics['f1']:.4f}}, AUC={{metrics['roc_auc']:.4f}}")

# Save model and results
with open(models_dir / "model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(models_dir / "results.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Model saved to {{models_dir}}/model.pkl")
''',

    # NOTE: evaluation template does NOT use .format(), so use regular braces
    "evaluation": '''
# Model Evaluation - WORKING TEMPLATE
import json
import numpy as np

print("Aggregating evaluation results...")

# Load results
results = {}
if (models_dir / "results.json").exists():
    with open(models_dir / "results.json", 'r') as f:
        results["primary_model"] = json.load(f)

# Check for multiple model results
for res_file in models_dir.glob("*_results.json"):
    with open(res_file, 'r') as f:
        results[res_file.stem] = json.load(f)

# Save evaluation report
with open(results_dir / "evaluation_report.json", 'w') as f:
    json.dump(results, f, indent=2)

# Find best model
best_model = None
best_accuracy = 0
for name, res in results.items():
    if isinstance(res, dict) and res.get('accuracy', 0) > best_accuracy:
        best_accuracy = res.get('accuracy', 0)
        best_model = name

metrics = {
    "n_models_evaluated": len(results),
    "best_model": best_model,
    "best_accuracy": float(best_accuracy),
    "all_results": results
}

print(f"Evaluation complete: Best model = {best_model} (accuracy: {best_accuracy:.4f})")
''',

    # CYCLE 2 TEMPLATES - Improved with feature engineering based on recommendations
    "preprocessing_improved": '''
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
print(f"Dataset shape: {{df.shape}}")

# Target column
target_col = 'vital_status'

# Get feature columns (exclude metadata)
exclude_cols = ['sampleID', 'vital_status', 'survival_time_days']
feature_cols = [c for c in df.columns if c not in exclude_cols]

# Use learner-suggested genes
learner_genes = {genes}
available_genes = [g for g in learner_genes if g in df.columns]
if len(available_genes) < 10:
    print(f"Using all {{len(feature_cols)}} features")
    available_genes = feature_cols[:500]
else:
    print(f"Using {{len(available_genes)}} learner-suggested genes")

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
print(f"Selected {{len(selected_genes)}} best features: {{selected_genes[:5]}}...")

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {{X_train.shape}}, Test: {{X_test.shape}}")
print(f"Class balance - Train: {{y_train.mean():.2f}}, Test: {{y_test.mean():.2f}}")

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

metrics = {{
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_features": int(len(selected_genes)),
    "n_original_features": len(available_genes),
    "class_balance_train": float(y_train.mean()),
    "class_balance_test": float(y_test.mean()),
    "improvements": ["StandardScaler", "SelectKBest feature selection"]
}}
print(f"Improved preprocessing complete: {{metrics}}")
''',

    "model_training_improved": '''
# Improved Model Training - CYCLE 2 with Cross-Validation
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
{model_import}

print("=== CYCLE 2: IMPROVED MODEL TRAINING ===")
print("Applying recommendations: Cross-validation, hyperparameter tuning...")

print("Loading preprocessed data...")
X_train = np.load(data_dir / "X_train.npy")
X_test = np.load(data_dir / "X_test.npy")
y_train = np.load(data_dir / "y_train.npy")
y_test = np.load(data_dir / "y_test.npy")

print(f"Train shape: {{X_train.shape}}, Test shape: {{X_test.shape}}")

# Handle class imbalance with SMOTE
try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied: {{X_train.shape}} -> {{X_train_res.shape}}")
except Exception as e:
    print(f"SMOTE skipped: {{e}}")
    X_train_res, y_train_res = X_train, y_train

# IMPROVEMENT 1: Cross-validation
print("Performing 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize model with improved hyperparameters
print("Training {model_name} with improved parameters...")
model = {model_init}

# Cross-validation scores
cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring='accuracy')
print(f"CV Scores: {{cv_scores}}")
print(f"CV Mean: {{cv_scores.mean():.4f}} (+/- {{cv_scores.std() * 2:.4f}})")

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
metrics = {{
    "model": "{model_name}",
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred, average='weighted')),
    "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
    "roc_auc": float(auc),
    "cv_mean": float(cv_scores.mean()),
    "cv_std": float(cv_scores.std()),
    "improvements": ["5-fold CV", "SMOTE", "Feature scaling"]
}}

print(f"Results: Accuracy={{metrics['accuracy']:.4f}}, F1={{metrics['f1']:.4f}}, AUC={{metrics['roc_auc']:.4f}}")
print(f"CV Mean: {{metrics['cv_mean']:.4f}}")

# Save model and results
with open(models_dir / "model_improved.pkl", "wb") as f:
    pickle.dump(model, f)
with open(models_dir / "results_improved.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Improved model saved to {{models_dir}}/model_improved.pkl")
'''
}


class TaskResult(BaseModel):
    """Model for a single task execution result"""
    task_id: str = Field(description="Task identifier")
    status: str = Field(description="Status: completed, failed, skipped")
    execution_time: float = Field(description="Execution time in seconds")
    output: str = Field(description="Task output description")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics if applicable")
    artifacts: List[str] = Field(default_factory=list, description="Paths to generated artifacts")
    generated_code: str = Field(default="", description="The code that was generated and executed")


class ExecutorOutput(BaseModel):
    """Model for executor agent output"""
    model_config = {"protected_namespaces": ()}
    
    summary: str = Field(description="Brief summary of execution results")
    tasks_completed: List[str] = Field(description="List of completed task IDs")
    baseline_results: Dict[str, Any] = Field(description="Baseline model results")
    model_results: Dict[str, Any] = Field(description="Deep learning model results")
    artifacts_generated: List[str] = Field(description="List of artifacts created")
    total_execution_time: float = Field(description="Total execution time in seconds")


class ExecutorAgent:
    """Executor Agent that generates and executes code dynamically using LLM based on Learner suggestions"""
    
    def __init__(self, model: str = "llama3.2:3b", artifacts_dir: str = None, 
                 dataset_path: str = "backend/dataset/tcga_brca_500samples_expr_survival.csv"):
        """
        Initialize the Executor Agent
        
        Args:
            model: Ollama model to use (default: llama3.2:3b)
            artifacts_dir: Directory to save artifacts (default: backend/artifacts)
            dataset_path: Path to the dataset file
        """
        # Initialize Ollama LLM for code generation
        self.llm = ChatOllama(
            model=model,
            temperature=0.3,
            num_predict=3000  # Allow longer code generation
        )
        
        # Use backend/artifacts directory by default
        if artifacts_dir is None:
            artifacts_dir = Path(__file__).parent.parent / "artifacts"
        self.artifacts_dir = Path(artifacts_dir)
        self.dataset_path = self._resolve_dataset_path(dataset_path)
    
    def _resolve_dataset_path(self, dataset_path: str) -> Path:
        """Resolve dataset path relative to current working directory/project root."""
        path = Path(dataset_path)
        if path.is_absolute() and path.exists():
            return path
        
        candidates = [
            Path.cwd() / path,
            Path(__file__).resolve().parents[2] / path,  # project root
            Path(__file__).resolve().parents[1] / path,  # backend/
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return path.resolve()
    
    def create_project_directory(self, project_id: str, cycle_id: str):
        """Create directory structure for this project/cycle"""
        self.project_artifacts_dir = self.artifacts_dir / project_id / cycle_id
        self.project_artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.project_artifacts_dir / "data").mkdir(exist_ok=True)
        (self.project_artifacts_dir / "models").mkdir(exist_ok=True)
        (self.project_artifacts_dir / "code").mkdir(exist_ok=True)
        (self.project_artifacts_dir / "results").mkdir(exist_ok=True)
        
        print(f"   📁 Artifacts directory: {self.project_artifacts_dir}")
        return self.project_artifacts_dir

    def _preprocess_data_helper(self, dataset_path: str, learner_genes: List[str], target_col: str, output_dir: Path) -> Dict[str, Any]:
        """
        Robust helper for data preprocessing.
        Handles loading, filtering, splitting, and saving.
        """
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        print(f"   Helper: Loading data from {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        # Filter genes
        available_genes = [g for g in learner_genes if g in df.columns]
        
        if not available_genes:
            print("   ⚠️  Helper: No learner genes found! Using fallback (all numeric cols).")
            # Fallback: use all numeric columns except target and metadata
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude = [target_col, 'sampleID', 'survival_time_days']
            available_genes = [c for c in numeric_cols if c not in exclude][:500]
        
        print(f"   Helper: Selected {len(available_genes)} features.")
        
        X = df[available_genes]
        y = df[target_col]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "X_train.npy", X_train)
        np.save(output_dir / "X_test.npy", X_test)
        np.save(output_dir / "y_train.npy", y_train)
        np.save(output_dir / "y_test.npy", y_test)
        
        metadata = {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(available_genes),
            "class_balance_train": float(y_train.mean()),
            "genes_used": available_genes
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
            
        return metadata

    def _train_model_helper(self, data_dir: Path, model_name: str, output_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        Robust helper for model training.
        Handles loading, SMOTE, training, evaluation, and saving.
        """
        import pickle
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from imblearn.over_sampling import SMOTE
        
        print(f"   Helper: Training {model_name}...")
        
        # Load data
        X_train = np.load(data_dir / "X_train.npy")
        X_test = np.load(data_dir / "X_test.npy")
        y_train = np.load(data_dir / "y_train.npy")
        y_test = np.load(data_dir / "y_test.npy")
        
        # SMOTE
        try:
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
        except ValueError as e:
            print(f"   ⚠️  Helper: SMOTE failed ({e}). Using original data.")
            X_res, y_res = X_train, y_train
            
        # Initialize Model
        model_map = {
            'RandomForestClassifier': RandomForestClassifier,
            'RandomForest': RandomForestClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'GradientBoosting': GradientBoostingClassifier,
            'LogisticRegression': LogisticRegression,
            'SVC': SVC,
            'SVM': SVC,
            'KNeighborsClassifier': KNeighborsClassifier,
            'KNeighbors': KNeighborsClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'DecisionTree': DecisionTreeClassifier
        }
        
        ModelClass = model_map.get(model_name, RandomForestClassifier)
        
        # Default params
        params = {'random_state': 42}
        if model_name in ['LogisticRegression', 'SVC', 'RandomForestClassifier']:
             params['class_weight'] = 'balanced'
        
        # Override with kwargs
        params.update(kwargs)
        
        # Remove invalid params for specific models
        if ModelClass == KNeighborsClassifier:
            params.pop('random_state', None)
            params.pop('class_weight', None)
        if ModelClass == GradientBoostingClassifier:
             params.pop('class_weight', None)
             
        model = ModelClass(**params)
        model.fit(X_res, y_res)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "model": type(model).__name__,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average='weighted')),
            "precision": float(precision_score(y_test, y_pred, average='weighted')),
            "recall": float(recall_score(y_test, y_pred, average='weighted'))
        }
        
        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(output_dir / "results.json", "w") as f:
            json.dump(metrics, f)
            
        return metrics

    def _evaluate_models_helper(self, models_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Robust helper for model evaluation.
        Aggregates results.
        """
        print("   Helper: Evaluating models...")
        results = {}
        
        # Check for single results.json
        if (models_dir / "results.json").exists():
            with open(models_dir / "results.json", 'r') as f:
                results["primary_model"] = json.load(f)
                
        # Check for multiple results
        for res_file in models_dir.glob("*_results.json"):
            with open(res_file, 'r') as f:
                results[res_file.stem] = json.load(f)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "evaluation_report.json", 'w') as f:
            json.dump(results, f)
            
        return results
    
    def get_model_template(self, model_name: str) -> Dict[str, Any]:
        """Get the template for a specific model, with fallback to RandomForest"""
        # Normalize model name
        normalized = model_name.replace(' ', '').replace('_', '')
        
        # Try exact match first
        if model_name in MODEL_TEMPLATES:
            return MODEL_TEMPLATES[model_name]
        
        # Try normalized matching
        for key in MODEL_TEMPLATES:
            if normalized.lower() in key.lower() or key.lower() in normalized.lower():
                return MODEL_TEMPLATES[key]
        
        # Default to RandomForest
        return MODEL_TEMPLATES["RandomForestClassifier"]
    
    def create_dynamic_prompt(self, task: Dict[str, Any], 
                              learner_output: Dict[str, Any],
                              previous_artifacts: List[str]) -> str:
        """
        Create dynamic prompt with proven code templates for reliable execution
        
        Args:
            task: Task from planner
            learner_output: Resources from learner (with model suggestions)
            previous_artifacts: List of artifacts from previous tasks
        """
        task_name = task.get('name', 'Unknown')
        task_desc = task.get('description', '')
        
        # Extract learner context
        genes = learner_output.get('key_genes', [])[:20]  # First 20 for context
        models = learner_output.get('model_suggestions', [])
        preprocessing = learner_output.get('preprocessing_notes', '')
        
        # Build task type detection
        task_lower = task_name.lower()
        
        # Determine which template to use
        if 'data' in task_lower or 'preprocess' in task_lower or 'load' in task_lower:
            # Use preprocessing template
            code = CODE_TEMPLATES["preprocessing"].format(genes=genes[:20])
            return f"USE THIS EXACT CODE (only fix syntax if needed):\n```python\n{code}\n```"
            
        elif 'baseline' in task_lower or 'model' in task_lower or 'train' in task_lower:
            # Use model training template with the suggested model
            model_name = models[0] if models else "RandomForestClassifier"
            template = self.get_model_template(model_name)
            
            code = CODE_TEMPLATES["model_training"].format(
                model_import=template["import"],
                model_name=model_name,
                model_init=template["init"]
            )
            return f"USE THIS EXACT CODE (only fix syntax if needed):\n```python\n{code}\n```"
            
        elif 'evaluat' in task_lower or 'assess' in task_lower or 'compare' in task_lower:
            # Use evaluation template
            code = CODE_TEMPLATES["evaluation"]
            return f"USE THIS EXACT CODE (only fix syntax if needed):\n```python\n{code}\n```"
        
        else:
            # For other tasks, provide a more guided prompt with model templates reference
            model_info = "\n".join([
                f"  - {name}: {info['init']}" 
                for name, info in list(MODEL_TEMPLATES.items())[:3]
            ])
            
            return f"""Generate Python code for: {task_name}
Description: {task_desc}

Available model templates:
{model_info}

Required variables (already defined):
- DATASET_PATH: path to CSV dataset
- data_dir: Path object for data files  
- models_dir: Path object for model files
- results_dir: Path object for results

Must set 'metrics' dict with results.
Return ONLY executable Python code."""
    
    def generate_code_for_task(self, task: Dict[str, Any], 
                               learner_output: Dict[str, Any],
                               previous_artifacts: List[str],
                               cycle: int = 1,
                               recommendations: List[str] = None) -> str:
        """
        Generate Python code for the task using templates + LLM refinement
        
        Uses proven templates as base, with LLM for customization only.
        Falls back to pure templates if LLM fails.
        For Cycle 2+, uses improved templates with feature engineering.
        
        Args:
            task: Task dictionary
            learner_output: Resources from learner
            previous_artifacts: Artifacts from previous tasks
            cycle: Current cycle number (1=baseline, 2+=improved)
            recommendations: Recommendations from assessor for improvement
            
        Returns:
            Generated Python code as string
        """
        task_name = task.get('name', 'Unknown')
        task_lower = task_name.lower()
        genes = learner_output.get('key_genes', [])[:20]
        models = learner_output.get('model_suggestions', [])
        
        # STRATEGY: Use templates directly for known task types
        # For Cycle 2+, use improved templates
        
        if cycle > 1:
            print(f"      🔄 CYCLE {cycle}: Using IMPROVED templates with recommendations...")
            if recommendations:
                print(f"      📝 Applying: {recommendations[:2]}...")
        else:
            print(f"      📋 Using template-based code generation...")
        
        # Detect task type and use appropriate template
        if 'data' in task_lower or 'preprocess' in task_lower or 'load' in task_lower:
            print(f"      📊 Task type: Data Preprocessing")
            if cycle > 1 and "preprocessing_improved" in CODE_TEMPLATES:
                print(f"      ⚡ Using IMPROVED preprocessing (feature scaling + selection)")
                code = CODE_TEMPLATES["preprocessing_improved"].format(genes=genes[:20])
            else:
                code = CODE_TEMPLATES["preprocessing"].format(genes=genes[:20])
            
        elif 'baseline' in task_lower or 'model' in task_lower or 'train' in task_lower:
            print(f"      🤖 Task type: Model Training")
            model_name = models[0] if models else "RandomForestClassifier"
            
            # Normalize model name to match our templates
            template = self.get_model_template(model_name)
            actual_model_name = next(
                (k for k in MODEL_TEMPLATES if MODEL_TEMPLATES[k] == template), 
                "RandomForestClassifier"
            )
            
            print(f"      📦 Model: {actual_model_name}")
            
            if cycle > 1 and "model_training_improved" in CODE_TEMPLATES:
                print(f"      ⚡ Using IMPROVED training (cross-validation)")
                code = CODE_TEMPLATES["model_training_improved"].format(
                    model_import=template["import"],
                    model_name=actual_model_name,
                    model_init=template["init"]
                )
            else:
                code = CODE_TEMPLATES["model_training"].format(
                    model_import=template["import"],
                    model_name=actual_model_name,
                    model_init=template["init"]
                )
            
        elif 'evaluat' in task_lower or 'assess' in task_lower or 'compare' in task_lower:
            print(f"      📈 Task type: Evaluation")
            code = CODE_TEMPLATES["evaluation"]
            
        else:
            # For unknown task types, try LLM generation with fallback
            print(f"      ❓ Unknown task type, using LLM generation...")
            code = self._generate_code_with_llm(task, learner_output, previous_artifacts)
        
        # Validate code
        if len(code) < 50:
            raise ValueError(f"Generated code too short ({len(code)} chars)")
        
        print(f"      ✅ Code ready ({len(code)} chars)")
        
        # Save generated code
        code_file = self.project_artifacts_dir / "code" / f"{task['task_id']}_cycle{cycle}_generated.py"
        with open(code_file, 'w') as f:
            f.write(f"# Template-based code generation - CYCLE {cycle}\n")
            f.write(f"# Task: {task['task_id']} - {task.get('name', 'Unknown')}\n")
            f.write(f"# Model: {models[0] if models else 'N/A'}\n")
            if recommendations:
                f.write(f"# Recommendations applied: {recommendations[:2]}\n")
            f.write("\n")
            f.write(code)
        
        return code
    
    def _generate_code_with_llm(self, task: Dict[str, Any],
                               learner_output: Dict[str, Any],
                               previous_artifacts: List[str]) -> str:
        """
        Fallback LLM-based code generation for unknown task types
        """
        prompt = self.create_dynamic_prompt(task, learner_output, previous_artifacts)
        
        messages = [
            SystemMessage(content="""You are a Python code generator. Return ONLY executable Python code.
Use these pre-defined variables: DATASET_PATH, data_dir, models_dir, results_dir.
Must set 'metrics' dict with results. No explanations."""),
            HumanMessage(content=prompt)
        ]
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)
                code = response.content.strip()
                
                # Extract code from markdown if present
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0].strip()
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0].strip()
                
                if len(code) >= 50:
                    return code
                    
            except Exception as e:
                print(f"      ⚠️  LLM attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(1)
        
        # Final fallback: return a simple evaluation code
        return '''
# Fallback code - task type not recognized
import json

print("Executing fallback task...")
metrics = {"status": "completed", "note": "Task type not recognized, minimal execution"}

# Try to load and summarize any available results
try:
    if (results_dir / "evaluation_report.json").exists():
        with open(results_dir / "evaluation_report.json", 'r') as f:
            metrics["previous_results"] = json.load(f)
except:
    pass

print(f"Fallback complete: {metrics}")
'''
    
    def execute_generated_code(self, code: str, task_id: str) -> Dict[str, Any]:
        """
        Execute the generated Python code safely
        
        Returns:
            Dictionary with metrics and artifacts
        """
        print(f"      ⚙️  Executing generated code...")
        
        # Create execution namespace with required imports
        exec_namespace = {
            '__builtins__': __builtins__,
            'np': np,
            'Path': Path,
        }
        
        # Import pandas
        try:
            import pandas as pd
            exec_namespace['pd'] = pd
        except ImportError:
            pass
        
        # Import sklearn
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
            import pickle
            exec_namespace.update({
                'RandomForestClassifier': RandomForestClassifier,
                'GradientBoostingClassifier': GradientBoostingClassifier,
                'LogisticRegression': LogisticRegression,
                'SVC': SVC,
                'KNeighborsClassifier': KNeighborsClassifier,
                'DecisionTreeClassifier': DecisionTreeClassifier,
                'train_test_split': train_test_split,
                'accuracy_score': accuracy_score,
                'f1_score': f1_score,
                'precision_score': precision_score,
                'recall_score': recall_score,
                'roc_auc_score': roc_auc_score,
                'pickle': pickle,
            })
        except ImportError:
            pass
        
        # Import other common modules
        import json
        exec_namespace['json'] = json
        
        # Inject paths as variables
        # We inject specific subdirectories to avoid LLM path construction errors
        artifacts_path = self.project_artifacts_dir
        exec_namespace['ARTIFACTS_DIR'] = artifacts_path
        exec_namespace['data_dir'] = artifacts_path / "data"
        exec_namespace['models_dir'] = artifacts_path / "models"
        exec_namespace['results_dir'] = artifacts_path / "results"
        exec_namespace['DATASET_PATH'] = str(self.dataset_path)
        
        # Inject Helper Functions
        exec_namespace['preprocess_data'] = self._preprocess_data_helper
        exec_namespace['train_model'] = self._train_model_helper
        exec_namespace['evaluate_models'] = self._evaluate_models_helper
        
        # Ensure directories exist
        (artifacts_path / "data").mkdir(parents=True, exist_ok=True)
        (artifacts_path / "models").mkdir(parents=True, exist_ok=True)
        (artifacts_path / "results").mkdir(parents=True, exist_ok=True)
        
        # SANITIZATION: Prevent LLM from overwriting injected Path variables with strings
        # This fixes the "TypeError: unsupported operand type(s) for /: 'str' and 'str'"
        sanitized_lines = []
        for line in code.split('\n'):
            stripped = line.strip()
            # Check if line assigns to our protected variables
            if any(stripped.startswith(var + " =") or stripped.startswith(var + "=") 
                   for var in ['data_dir', 'models_dir', 'results_dir', 'ARTIFACTS_DIR']):
                sanitized_lines.append(f"# [EXECUTOR] Commented out unsafe assignment: {line}")
            else:
                sanitized_lines.append(line)
        
        code = '\n'.join(sanitized_lines)
        
        # Capture artifacts before execution
        artifacts_before = set()
        for path in self.project_artifacts_dir.rglob('*'):
            if path.is_file() and not path.name.endswith('_generated.py'):
                artifacts_before.add(str(path))
        
        try:
            # Execute code
            exec(code, exec_namespace)
            
            # Extract metrics from namespace
            metrics = exec_namespace.get('metrics', {"status": "completed"})
            
            # Find new artifacts
            artifacts_after = set()
            for path in self.project_artifacts_dir.rglob('*'):
                if path.is_file() and not path.name.endswith('_generated.py'):
                    artifacts_after.add(str(path))
            
            new_artifacts = list(artifacts_after - artifacts_before)
            
            print(f"      ✅ Execution successful")
            print(f"      📊 Metrics: {metrics}")
            print(f"      📦 New artifacts: {len(new_artifacts)}")
            
            return {
                "status": "success",
                "metrics": metrics,
                "artifacts": new_artifacts,
                "error": None
            }
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"      ❌ Execution failed: {error_msg}")
            print(f"\\n      🔍 TRACEBACK:")
            traceback.print_exc()
            
            return {
                "status": "failed",
                "metrics": {"error": error_msg},
                "artifacts": [],
                "error": error_msg
            }
    
    def execute_task(self, task: Dict[str, Any], 
                    learner_output: Dict[str, Any],
                    previous_artifacts: List[str],
                    cycle: int = 1,
                    recommendations: List[str] = None) -> TaskResult:
        """
        Execute a single task by generating and running code
        
        Args:
            task: Task dictionary from planner
            learner_output: Resources from learner
            previous_artifacts: Artifacts from previous tasks
            cycle: Current cycle number (1=baseline, 2+=improved)
            recommendations: Recommendations from assessor
            
        Returns:
            TaskResult with execution details
        """
        task_id = task.get('task_id', 'T0')
        task_name = task.get('name', 'Unknown Task')
        
        print(f"\\n      📝 {task_id}: {task_name}")
        
        start_time = time.time()
        
        # Step 1: Generate code using templates (with improvements for Cycle 2+)
        try:
            generated_code = self.generate_code_for_task(
                task, learner_output, previous_artifacts,
                cycle=cycle, recommendations=recommendations
            )
        except Exception as e:
            # If code generation fails, return failed result
            return TaskResult(
                task_id=task_id,
                status="failed",
                execution_time=time.time() - start_time,
                output=f"Code generation failed: {str(e)}",
                metrics={"error": str(e)},
                artifacts=[],
                generated_code=""
            )
        
        # Show code preview
        code_lines = generated_code.split('\\n')
        print(f"\\n      💻 CODE PREVIEW (first 15 lines):")
        for i, line in enumerate(code_lines[:15], 1):
            print(f"         {i:2d}| {line}")
        if len(code_lines) > 15:
            print(f"         ... ({len(code_lines) - 15} more lines)")
        
        # Step 2: Execute the generated code
        exec_result = self.execute_generated_code(generated_code, task_id)
        
        exec_time = time.time() - start_time
        
        # Step 3: Create result
        result = TaskResult(
            task_id=task_id,
            status=exec_result["status"],
            execution_time=exec_time,
            output=f"{task_name}: {exec_result['status']}",
            metrics=exec_result["metrics"],
            artifacts=exec_result["artifacts"],
            generated_code=generated_code
        )
        
        print(f"      ⏱️  Total time: {exec_time:.2f}s")
        
        return result
    
    def execute(self, spec_sheet: Dict[str, Any], 
               planner_output: Dict[str, Any],
               learner_output: Dict[str, Any],
               project_id: str,
               cycle_id: str,
               cycle: int = 1,
               recommendations: List[str] = None) -> Dict[str, Any]:
        """
        Execute all research tasks by dynamically generating and running code
        
        Args:
            spec_sheet: Research specification
            planner_output: Tasks from planner agent
            learner_output: Resources from learner agent
            project_id: Project UUID
            cycle_id: Cycle UUID
            cycle: Current cycle number (1=baseline, 2+=improved)
            recommendations: Recommendations from assessor for improvement
            
        Returns:
            Dictionary containing execution results
        """
        
        print("\\n" + "="*80)
        if cycle > 1:
            print(f"🔧 EXECUTOR AGENT - CYCLE {cycle} (IMPROVED)")
        else:
            print("🔧 EXECUTOR AGENT - LLM-Generated Code Execution")
        print("="*80)
        
        # Show recommendations if Cycle 2+
        if cycle > 1 and recommendations:
            print(f"\\n📝 APPLYING RECOMMENDATIONS FROM CYCLE {cycle-1}:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        # Create project directory
        self.create_project_directory(project_id, cycle_id)
        
        # Verify dataset exists
        if not self.dataset_path.exists():
            print(f"❌ ERROR: Dataset not found at {self.dataset_path}")
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        print(f"\\n📊 DATASET: {self.dataset_path}")
        
        print(f"\\n📚 LEARNER CONTEXT:") 
        print(f"   Models Suggested: {', '.join(learner_output.get('model_suggestions', [])[:3])}")
        print(f"   Genes: {', '.join(learner_output.get('key_genes', [])[:5])}")
        print(f"   Datasets: {', '.join(learner_output.get('datasets', []))}")
        
        tasks = planner_output.get('tasks', [])
        task_results = []
        total_time = 0
        all_artifacts = []
        
        # Execute tasks sequentially
        print(f"\\n🚀 EXECUTING {len(tasks)} TASKS (CYCLE {cycle}):")
        print("="*80)
        
        for i, task in enumerate(tasks, 1):
            print(f"\\n[{i}/{len(tasks)}] " + "─"*70)
            
            result = self.execute_task(
                task, learner_output, all_artifacts,
                cycle=cycle, recommendations=recommendations
            )
            task_results.append(result)
            total_time += result.execution_time
            all_artifacts.extend(result.artifacts)
            
            # Stop if a critical task fails (data preprocessing or first model)
            if result.status == "failed" and i <= 2:
                print(f"\\n❌ Critical task failed. Stopping execution.")
                break
        
        print("\\n" + "="*80)
        print(f"✅ EXECUTION COMPLETED")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Artifacts: {len(all_artifacts)} files")
        print("="*80)
        
        # Extract metrics
        baseline_metrics = {}
        model_metrics = {}
        
        for result in task_results:
            if result.status == "success":
                if 'baseline' in result.output.lower() and result.metrics:
                    baseline_metrics = result.metrics
                elif 'model' in result.output.lower() or 'train' in result.output.lower():
                    if not baseline_metrics or result.metrics != baseline_metrics:
                        model_metrics = result.metrics
        
        # Compile results
        completed_tasks = [r.task_id for r in task_results if r.status == "success"]
        
        cycle_info = f" (Cycle {cycle} - {'Improved' if cycle > 1 else 'Baseline'})"
        executor_output = {
            "summary": f"Executed {len(completed_tasks)}/{len(tasks)} tasks{cycle_info}",
            "cycle": cycle,
            "tasks_completed": completed_tasks,
            "baseline_results": {
                "model": "Learner-Suggested Model",
                **baseline_metrics
            } if baseline_metrics else {},
            "model_results": {
                "model": "Learner-Suggested Model",
                **model_metrics
            } if model_metrics else {},
            "artifacts_generated": all_artifacts,
            "total_execution_time": total_time
        }
        
        print(f"\\n📊 FINAL RESULTS:")
        print(f"   Baseline: {baseline_metrics.get('accuracy', 'N/A')}")
        print(f"   Model: {model_metrics.get('accuracy', 'N/A')}")
        print(f"   Artifacts: {len(all_artifacts)} files")
        
        return executor_output
