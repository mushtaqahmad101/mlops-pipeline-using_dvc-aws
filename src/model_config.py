"""
model_config.py
---------------
Stage 5 – Persist model hyperparameters and a human-readable YAML report
           to the model_config/ directory.

Outputs
-------
  model_config/model_params.json  – raw parameter dict from the trained model
  model_config/model_report.yaml  – combined params + metrics summary
"""

import json
import os
import pickle

import yaml

from src.log_config import get_logger

logger = get_logger(__name__)

PARAMS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "params.yaml")


def load_params() -> dict:
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


def save_model_config(
    model_path: str,
    metrics_path: str,
    params_out_path: str,
    report_out_path: str,
    pipeline_params: dict,
) -> None:
    """
    Read the trained model + evaluation metrics and write summary artefacts.

    Parameters
    ----------
    model_path       : Path to pickled trained model.
    metrics_path     : Path to metrics.json written by evaluation stage.
    params_out_path  : Destination for model_params.json.
    report_out_path  : Destination for model_report.yaml.
    pipeline_params  : Full params.yaml dict (for training hyper-params).
    """
    # ── Load model ─────────────────────────────────────────────────────────────
    logger.info("Loading model from '%s'", model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    model_params = model.get_params()
    logger.info("Model hyperparameters: %s", model_params)

    # ── Load metrics ───────────────────────────────────────────────────────────
    logger.info("Loading metrics from '%s'", metrics_path)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # ── Save model_params.json ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(params_out_path), exist_ok=True)
    with open(params_out_path, "w") as f:
        json.dump(model_params, f, indent=4, default=str)
    logger.info("Model params saved → '%s'", params_out_path)

    # ── Build YAML report ──────────────────────────────────────────────────────
    report = {
        "model": {
            "type": type(model).__name__,
            "hyperparameters": {k: str(v) for k, v in model_params.items()},
        },
        "training": {
            "n_estimators": pipeline_params["model_training"]["n_estimators"],
            "max_depth": pipeline_params["model_training"]["max_depth"],
            "random_state": pipeline_params["model_training"]["random_state"],
            "test_size": pipeline_params["data_preprocessing"]["test_size"],
        },
        "evaluation_metrics": {
            k: v for k, v in metrics.items() if k != "confusion_matrix"
        },
        "confusion_matrix": metrics.get("confusion_matrix", []),
        "dataset": {
            "source": pipeline_params["data_ingestion"]["url"],
            "raw_path": pipeline_params["data_ingestion"]["raw_data_path"],
            "processed_path": pipeline_params["data_preprocessing"]["processed_data_path"],
        },
    }

    os.makedirs(os.path.dirname(report_out_path), exist_ok=True)
    with open(report_out_path, "w") as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    logger.info("Model report saved → '%s'", report_out_path)


def run() -> None:
    """Entry-point for the model config stage."""
    logger.info("=" * 60)
    logger.info("STAGE 5 : Model Config & Report")
    logger.info("=" * 60)

    params = load_params()
    train_cfg = params["model_training"]
    eval_cfg = params["model_evaluation"]
    config_cfg = params["model_config"]

    save_model_config(
        model_path=train_cfg["model_path"],
        metrics_path=eval_cfg["metrics_path"],
        params_out_path=config_cfg["model_params_path"],
        report_out_path=config_cfg["model_report_path"],
        pipeline_params=params,
    )
    logger.info("Model config stage complete.")


if __name__ == "__main__":
    run()
