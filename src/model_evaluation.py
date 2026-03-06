"""
model_evaluation.py
-------------------
Stage 4 – Evaluate the trained model and save metrics to JSON.

Metrics computed
----------------
  accuracy, precision, recall, f1_score, roc_auc
"""

import json
import os
import pickle

import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.log_config import get_logger

logger = get_logger(__name__)

PARAMS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "params.yaml")


def load_params() -> dict:
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


def evaluate(processed_path: str, model_path: str, metrics_path: str, params: dict) -> dict:
    """
    Load model + data, compute evaluation metrics, persist to JSON.

    Returns
    -------
    dict of metric names → float values.
    """
    logger.info("Loading processed data from '%s'", processed_path)
    df = pd.read_csv(processed_path)

    target = "Survived"
    X = df.drop(columns=[target])
    y = df[target]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )

    logger.info("Loading model from '%s'", model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info("Running predictions …")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }

    logger.info("─── Evaluation Metrics ───────────────────────────────")
    for name, value in metrics.items():
        logger.info("  %-12s : %.4f", name, value)

    # ── Confusion matrix ───────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred).tolist()
    logger.info("Confusion matrix: %s", cm)
    metrics["confusion_matrix"] = cm

    # ── Full classification report ─────────────────────────────────────────────
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.debug("Classification report: %s", report)

    # ── Persist metrics ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Metrics saved → '%s'", metrics_path)

    return metrics


def run() -> None:
    """Entry-point for the model evaluation stage."""
    logger.info("=" * 60)
    logger.info("STAGE 4 : Model Evaluation")
    logger.info("=" * 60)

    params = load_params()
    preprocess_cfg = params["data_preprocessing"]
    train_cfg = params["model_training"]
    eval_cfg = params["model_evaluation"]

    evaluate(
        processed_path=preprocess_cfg["processed_data_path"],
        model_path=train_cfg["model_path"],
        metrics_path=eval_cfg["metrics_path"],
        params=eval_cfg,
    )
    logger.info("Model evaluation stage complete.")


if __name__ == "__main__":
    run()
