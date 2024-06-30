"""
This is a boilerplate test file for pipeline 'model_inference'
generated using Kedro 0.19.4.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pytest
import os
import pandas as pd
import numpy as np
from src.mlops_group_project.pipelines.model_inference.nodes import make_predictions


def test_make_predictions():
    X_feed_model_path = os.path.join("tests/sample/sample_model_inference_test.csv")
    ids_path = os.path.join("tests/sample/sample_ids.csv")

    X_feed_model = pd.read_csv(X_feed_model_path)
    ids = pd.read_csv(ids_path)

    predictions = make_predictions(X_feed_model, ids)

    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape[0] == X_feed_model.shape[0]
    assert predictions.shape[1] == 2
    assert "prediction" in predictions.columns
    assert "encounter_id" in predictions.columns
