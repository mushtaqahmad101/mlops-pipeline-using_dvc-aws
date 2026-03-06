"""
model_training.py
-----------------
Stage 3 – Train a Random Forest classifier on the processed Titanic data.

Outputs
-------
  models/model.pkl   – Serialised trained model (joblib)
"""

import json
import os
import pickle

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.log_config import get_logger

logger = get_logger(__name__)

PARAMS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "params.yaml")


def load_params() -> dict:
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


def train(processed_path: str, model_path: str, params: dict) -> RandomForestClassifier:
    """
    Train a Random Forest and persist the model to *model_path*.

    Parameters
    ----------
    processed_path : Path to the processed CSV file.
    model_path     : Destination path for the serialised model.
    params         : Hyper-parameter dict from params.yaml.

    Returns
    -------
    Fitted RandomForestClassifier instance.
    """
    logger.info("Loading processed data from '%s'", processed_path)
    df = pd.read_csv(processed_path)

    target = "Survived"
    X = df.drop(columns=[target])
    y = df[target]
    logger.info("Features : %s | Target : %s | Samples : %d", list(X.columns), target, len(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
    )
    logger.info("Train size: %d  |  Test size: %d", len(X_train), len(X_test))

    # ── Model definition ───────────────────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=params["random_state"],
    )

    logger.info("Training RandomForestClassifier …")
    clf.fit(X_train, y_train)
    logger.info("Training complete.")

    # ── Persist model ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    logger.info("Model saved → '%s'", model_path)

    return clf


def run() -> None:
    """Entry-point for the model training stage."""
    logger.info("=" * 60)
    logger.info("STAGE 3 : Model Training")
    logger.info("=" * 60)

    params = load_params()
    preprocess_cfg = params["data_preprocessing"]
    train_cfg = params["model_training"]

    train(
        processed_path=preprocess_cfg["processed_data_path"],
        model_path=train_cfg["model_path"],
        params={**train_cfg, **{"test_size": preprocess_cfg["test_size"],
                                 "random_state": preprocess_cfg["random_state"]}},
    )
    logger.info("Model training stage complete.")


if __name__ == "__main__":
    run()
