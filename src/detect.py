import os
import joblib
import pandas as pd

from src.config import (
    FEATURE_NAMES,
    TARGET,
    TEST_FILE,
    MODEL_FILE,
    ALERTS_FILE,
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
    print("Loading trained model...")

    model = joblib.load(MODEL_FILE)

    test_df = prepare_labels(load_dataset(TEST_FILE))

    X_test = test_df[FEATURE_NAMES]

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    alerts = test_df.copy()

    alerts["prediction"] = predictions
    alerts["threat_probability"] = probabilities.round(4)

    alerts["alert"] = alerts["prediction"].apply(
        lambda value: "THREAT DETECTED" if value == 1 else "NORMAL"
    )

    os.makedirs("outputs", exist_ok=True)

    alerts.to_csv(ALERTS_FILE, index=False)

    threat_count = int((predictions == 1).sum())
    normal_count = int((predictions == 0).sum())

    print("\n===== THREAT DETECTION RESULTS =====")
    print(f"Total records analyzed : {len(predictions)}")
    print(f"Normal records         : {normal_count}")
    print(f"Threat records         : {threat_count}")
    print(f"Alerts saved to        : {ALERTS_FILE}")


if __name__ == "__main__":
    main()
