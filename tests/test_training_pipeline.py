"""Tests for the PhishGuard training pipeline."""

import numpy as np
import pandas as pd
import pytest

from ml import DEFAULT_DATASET, build_pipeline, load_dataset, split_data
from ml.train_model import evaluate_predictions


def test_dataset_loads_successfully() -> None:
    df = load_dataset(DEFAULT_DATASET)
    assert not df.empty
    assert set(["text", "label"]).issubset(df.columns)


@pytest.mark.parametrize("test_size", [0.25, 0.3])
def test_pipeline_trains_and_predicts(test_size: float) -> None:
    df = load_dataset(DEFAULT_DATASET)
    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state=42)
    model = build_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
    # Ensure predictions are binary and within expected range
    assert set(predictions).issubset({0, 1})


def test_evaluate_predictions_handles_support_none() -> None:
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    metrics = evaluate_predictions(y_true, y_pred)
    assert isinstance(metrics["support"], int)
