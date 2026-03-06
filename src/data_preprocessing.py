"""
data_preprocessing.py
---------------------
Stage 2 – Clean, encode, and split the raw Titanic dataset.

Steps:
  1. Load raw CSV
  2. Drop unwanted columns
  3. Impute missing values
  4. Encode categoricals
  5. Train / test split
  6. Save processed dataset (full) for downstream stages
"""

import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.log_config import get_logger

logger = get_logger(__name__)

PARAMS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "params.yaml")


def load_params() -> dict:
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


def preprocess(raw_path: str, processed_path: str, params: dict) -> pd.DataFrame:
    """
    Load → clean → encode → save the processed DataFrame.

    Returns the full processed DataFrame (features + target).
    """
    logger.info("Loading raw data from '%s'", raw_path)
    df = pd.read_csv(raw_path)
    logger.info("Raw shape: %s", df.shape)

    # ── Drop irrelevant columns ────────────────────────────────────────────────
    drop_cols = [c for c in params["drop_columns"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    logger.info("Dropped columns: %s", drop_cols)

    # ── Impute missing values ──────────────────────────────────────────────────
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    missing_before = df.isnull().sum().sum()
    logger.info("Missing values after imputation: %d", missing_before)

    # ── Encode categoricals ────────────────────────────────────────────────────
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))
        logger.debug("Label-encoded column: '%s'", col)

    # ── Save processed data ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info("Processed data saved → '%s'  shape=%s", processed_path, df.shape)

    return df


def run() -> None:
    """Entry-point for the preprocessing stage."""
    logger.info("=" * 60)
    logger.info("STAGE 2 : Data Preprocessing")
    logger.info("=" * 60)

    params = load_params()
    ingestion_cfg = params["data_ingestion"]
    preprocess_cfg = params["data_preprocessing"]

    preprocess(
        raw_path=ingestion_cfg["raw_data_path"],
        processed_path=preprocess_cfg["processed_data_path"],
        params=preprocess_cfg,
    )
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    run()
