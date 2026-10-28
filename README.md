# predictive_model.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report
)

def detect_target_type(y):
    # If target is numeric and more than ~10 unique values, treat as regression
    unique_vals = pd.Series(y).nunique()
    if pd.api.types.is_numeric_dtype(y) and unique_vals > 10:
        return "regression"
    return "classification"

def preprocess_data(df, target_col):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Separate numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_features)
        ]
    )

    return X, y, preprocessor

def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="binary", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="binary", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average="binary", zero_division=0)
    }

    if y_prob is not None:
        metrics["ROC AUC"] = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"Model (AUC = {metrics['ROC AUC']:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return metrics

def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }

    print("\nRegression Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # Scatter plot actual vs predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.show()

    return metrics

def run_project(csv_path, target_col):
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Dataset shape: {df.shape}")
    print(df.head())

    X, y, preprocessor = preprocess_data(df, target_col)
    target_type = detect_target_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if target_type == "classification" else None
    )

    # Fit preprocessing on train only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    if target_type == "classification":
        print("\nTraining classification models...")

        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000)
        }

        best_model_name = None
        best_score = -1

        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_processed, y_train)
            metrics = evaluate_classification(model, X_test_processed, y_test)
            score = metrics.get("Accuracy", 0)

            if score > best_score:
                best_score = score
                best_model_name = name

        print(f"\nBest Model: {best_model_name} with accuracy {best_score:.4f}")

    else:
        print("\nTraining regression models...")

        models = {
            "Linear Regression": LinearRegression()
        }

        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_processed, y_train)
            evaluate_regression(model, X_test_processed, y_test)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predictive_model.py <dataset.csv> <target_column>")
        print("Example: python predictive_model.py sales_data.csv Sales")
        sys.exit(1)

    csv_path = sys.argv[1]
    target_col = sys.argv[2]

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    run_project(csv_path, target_col)
How to run it:

Install dependencies:
bash
pip install pandas numpy scikit-learn matplotlib seaborn
Run your model:
bash
python predictive_model.py your_dataset.csv target_column_name
Example:

bash
python predictive_model.py customer_data.csv churn
