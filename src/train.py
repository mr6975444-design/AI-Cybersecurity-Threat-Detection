import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    FEATURE_NAMES,
    TARGET,
    TRAIN_FILE,
    TEST_FILE,
    MODEL_FILE,
    PREDICTIONS_FILE,
    METRICS_FILE,
)


def load_dataset(path):
    columns = FEATURE_NAMES + [TARGET, "difficulty"]
    return pd.read_csv(path, names=columns)


def prepare_labels(df):
    df = df.copy()
    df[TARGET] = df[TARGET].apply(
        lambda x: 0 if str(x).strip().lower() == "normal" else 1
    )
    return df


def main():
    print("Loading NSL-KDD dataset...")

    train_df = prepare_labels(load_dataset(TRAIN_FILE))
    test_df = prepare_labels(load_dataset(TEST_FILE))

    X_train = train_df[FEATURE_NAMES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURE_NAMES]
    y_test = test_df[TARGET]

    categorical_features = ["protocol_type", "service", "flag"]

    numeric_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in categorical_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    print("Training Random Forest model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    report = classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Threat"],
        zero_division=0,
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    joblib.dump(pipeline, MODEL_FILE)

    results = test_df.copy()
    results["actual"] = y_test
    results["prediction"] = predictions
    results.to_csv(PREDICTIONS_FILE, index=False)

    metrics_text = f"""AI-Powered Cybersecurity Threat Detection System

Model: Random Forest
Dataset: NSL-KDD

Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1 Score:  {f1:.4f}

Classification Report:
{report}
"""

    with open(METRICS_FILE, "w") as file:
        file.write(metrics_text)

    print("\n===== MODEL RESULTS =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(report)

    print(f"Model saved to: {MODEL_FILE}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")
    print(f"Metrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    main()
