"""Training pipeline for the PhishGuard phishing email classifier."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")  # Use a non-interactive backend for CI and headless environments.

LOGGER = logging.getLogger("phishguard.ml")
DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "data" / "emails.csv"
DEFAULT_ARTIFACTS = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Path to the training dataset CSV (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_ARTIFACTS / "phishguard_model.joblib",
        help="Output path for the trained model.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=DEFAULT_ARTIFACTS / "training_metrics.json",
        help="Output path for training metrics JSON.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=DEFAULT_ARTIFACTS / "classification_report.txt",
        help="Optional path to save the full classification report.",
    )
    parser.add_argument(
        "--figure-out",
        type=Path,
        default=DEFAULT_ARTIFACTS / "confusion_matrix.png",
        help="Optional path to save the confusion matrix plot.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of the dataset to use for evaluation (default: 0.25).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    """Load dataset from CSV and validate schema."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    LOGGER.info("Loading dataset from %s", path)
    df = pd.read_csv(path)

    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        missing = ", ".join(sorted(required_columns - set(df.columns)))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.dropna(subset=["text", "label"]).copy()
    if df.empty:
        raise ValueError("Dataset is empty after dropping missing values.")

    LOGGER.debug("Loaded %d rows from dataset.", len(df))
    return df


def build_pipeline() -> Pipeline:
    """Create a text classification pipeline."""
    LOGGER.info("Building training pipeline.")
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline


def split_data(
    data: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split dataset into train and test components."""
    LOGGER.info("Splitting dataset (test_size=%s, random_state=%s).", test_size, random_state)
    labels = data["label"].map({"phishing": 1, "legitimate": 0})
    if labels.isnull().any():
        raise ValueError("Labels must be 'phishing' or 'legitimate'.")

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    LOGGER.debug(
        "Split sizes -> train: %d, test: %d", len(X_train),
        len(X_test)
    )
    return X_train, X_test, y_train, y_test


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute evaluation metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    if support is None:
        support_value = int((y_true == 1).sum())
    else:
        support_value = int(support)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support_value,
    }
    LOGGER.info("Evaluation metrics: %s", metrics)
    return metrics


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """Persist metrics to JSON."""
    LOGGER.info("Saving metrics to %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)


def save_classification_report(
    y_true: pd.Series, y_pred: np.ndarray, path: Path
) -> None:
    """Persist a text classification report."""
    LOGGER.info("Saving classification report to %s", path)
    report = classification_report(
        y_true, y_pred, target_names=["legitimate", "phishing"], digits=3
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write(report)


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, path: Path) -> None:
    """Render and save a confusion matrix plot."""
    LOGGER.info("Saving confusion matrix plot to %s", path)
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Legitimate", "Phishing"]
    )

    plt.figure(figsize=(6, 5))
    display.plot(cmap=sns.color_palette("crest", as_cmap=True))
    plt.title("PhishGuard Confusion Matrix")
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def save_model(pipeline: Pipeline, path: Path) -> None:
    """Serialize the trained model to disk."""
    LOGGER.info("Saving trained model to %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the training workflow and return metrics."""
    data = load_dataset(args.data)
    X_train, X_test, y_train, y_test = split_data(data, args.test_size, args.random_state)

    pipeline = build_pipeline()
    LOGGER.info("Starting model training on %d samples.", len(X_train))
    pipeline.fit(X_train, y_train)

    LOGGER.info("Evaluating model on %d samples.", len(X_test))
    y_pred = pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)

    save_model(pipeline, args.model_out)
    save_metrics(metrics, args.metrics_out)

    if args.report_out:
        save_classification_report(y_test, y_pred, args.report_out)
    if args.figure_out:
        save_confusion_matrix(y_test, y_pred, args.figure_out)

    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    args = parse_args()
    LOGGER.info("Training configuration: %s", vars(args))
    metrics = train(args)
    LOGGER.info("Training complete with metrics: %s", metrics)


if __name__ == "__main__":
    main()
