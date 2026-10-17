import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import PREDICTIONS_FILE


def main():
    os.makedirs("images", exist_ok=True)

    df = pd.read_csv(PREDICTIONS_FILE)

    # 1. Actual class distribution
    plt.figure(figsize=(8, 5))

    class_counts = df["actual"].map(
        {0: "Normal", 1: "Threat"}
    ).value_counts()

    class_counts.plot(kind="bar")

    plt.title("NSL-KDD Test Set Class Distribution")
    plt.xlabel("Traffic Class")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig("images/class_distribution.png", dpi=200)
    plt.close()

    # 2. Confusion matrix
    cm = confusion_matrix(
        df["actual"],
        df["prediction"]
    )

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Normal", "Threat"],
        yticklabels=["Normal", "Threat"]
    )

    plt.title("Threat Detection Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig("images/confusion_matrix.png", dpi=200)
    plt.close()

    # 3. Prediction distribution
    prediction_counts = df["prediction"].map(
        {0: "Normal", 1: "Threat"}
    ).value_counts()

    plt.figure(figsize=(8, 5))

    prediction_counts.plot(kind="bar")

    plt.title("Model Prediction Distribution")
    plt.xlabel("Prediction")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig("images/prediction_distribution.png", dpi=200)
    plt.close()

    print("Visualizations generated successfully!")
    print("Saved to:")
    print("  images/class_distribution.png")
    print("  images/confusion_matrix.png")
    print("  images/prediction_distribution.png")


if __name__ == "__main__":
    main()
