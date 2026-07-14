from pathlib import Path

import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier
PROJECT_DIR = Path(__file__).resolve().parent

FEATURE_DIR = PROJECT_DIR / "features"
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
train_df = pd.read_csv(
    FEATURE_DIR / "train_features.csv"
)

valid_df = pd.read_csv(
    FEATURE_DIR / "validation_features.csv"
)

test_df = pd.read_csv(
    FEATURE_DIR / "test_features.csv"
)
DROP_COLUMNS = [
    "label",
    "generator",
    "filename"
]

X_train = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df["label"]

X_valid = valid_df.drop(columns=DROP_COLUMNS)
y_valid = valid_df["label"]

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df["label"]
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_valid = scaler.transform(X_valid)

X_test = scaler.transform(X_test)
joblib.dump(
    scaler,
    MODEL_DIR / "scaler.pkl"
)
print("Training:", X_train.shape)
print("Validation:", X_valid.shape)
print("Testing:", X_test.shape)
def evaluate_model(model, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(y, predictions)

    recall = recall_score(y, predictions)

    f1 = f1_score(y, predictions)

    auc = roc_auc_score(y, probabilities)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

    print("\nConfusion Matrix")

    print(confusion_matrix(y, predictions))

    print("\nClassification Report")

    print(classification_report(y, predictions))

    return accuracy, precision, recall, f1, auc

rf = RandomForestClassifier(

    n_estimators=300,

    max_depth=None,

    random_state=42,

    n_jobs=-1
)

print("\nTraining Random Forest...\n")

rf.fit(
    X_train,
    y_train
)

print("\nValidation Results\n")

valid_metrics = evaluate_model(
    rf,
    X_valid,
    y_valid
)

print("\nTest Results\n")

test_metrics = evaluate_model(
    rf,
    X_test,
    y_test
)

joblib.dump(

    rf,

    MODEL_DIR / "random_forest.pkl"
)

print("\nRandom Forest model saved.")

results_df = pd.DataFrame([
    {
        "Dataset": "Validation",
        "Model": "Random Forest",
        "Accuracy": valid_metrics[0],
        "Precision": valid_metrics[1],
        "Recall": valid_metrics[2],
        "F1": valid_metrics[3],
        "ROC_AUC": valid_metrics[4]
    },
    {
        "Dataset": "Test",
        "Model": "Random Forest",
        "Accuracy": test_metrics[0],
        "Precision": test_metrics[1],
        "Recall": test_metrics[2],
        "F1": test_metrics[3],
        "ROC_AUC": test_metrics[4]
    }
])

results_df.to_csv(
    RESULT_DIR / "random_forest_results.csv",
    index=False
)

print("\n==============================")
print("Random Forest Results")
print("==============================")
print(results_df)
