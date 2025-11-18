# src/evaluate.py
import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)

import matplotlib.pyplot as plt

from .config import DATA_PROCESSED, MODELS_DIR


def main():
    X_test_path = os.path.join(DATA_PROCESSED, "X_test.npy")
    y_test_path = os.path.join(DATA_PROCESSED, "y_test.npy")

    if not (os.path.isfile(X_test_path) and os.path.isfile(y_test_path)):
        raise FileNotFoundError(
            f"Test data not found. Expected {X_test_path} and {y_test_path}. "
            f"Run `python -m src.train` first."
        )

    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    model_path = os.path.join(MODELS_DIR, "best_model.h5")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train the model first."
        )

    model = tf.keras.models.load_model(model_path)

    print("Evaluating model on test set...")
    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    print("=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred))

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print(f"Average precision (PR-AUC): {ap:.4f}")

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve (AP = {ap:.3f})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
