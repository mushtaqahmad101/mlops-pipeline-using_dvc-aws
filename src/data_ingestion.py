"""
data_ingestion.py
-----------------
Stage 1 – Download raw data from the internet and save it locally.
Dataset : Titanic (CSV) from GitHub / datasciencedojo
"""

import os
import urllib.request

import yaml

from src.log_config import get_logger

logger = get_logger(__name__)

PARAMS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "params.yaml")


def load_params() -> dict:
    """Load pipeline parameters from params.yaml."""
    with open(PARAMS_PATH, "r") as f:
        params = yaml.safe_load(f)
    return params


def download_data(url: str, save_path: str) -> None:
    """
    Download a CSV file from *url* and persist it to *save_path*.

    Parameters
    ----------
    url       : Direct download URL for the dataset.
    save_path : Local file path where the data will be stored.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        logger.info("Raw data already exists at '%s'. Skipping download.", save_path)
        return

    logger.info("Downloading data from: %s", url)
    try:
        urllib.request.urlretrieve(url, save_path)
        logger.info("Data downloaded successfully → '%s'", save_path)
    except Exception as exc:
        logger.error("Failed to download data: %s", exc)
        raise


def run() -> None:
    """Entry-point for the data ingestion stage."""
    logger.info("=" * 60)
    logger.info("STAGE 1 : Data Ingestion")
    logger.info("=" * 60)

    params = load_params()
    cfg = params["data_ingestion"]

    url = cfg["url"]
    raw_path = cfg["raw_data_path"]

    download_data(url, raw_path)
    logger.info("Data ingestion complete.")


if __name__ == "__main__":
    run()
