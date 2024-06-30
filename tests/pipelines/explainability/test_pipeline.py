"""
This is a boilerplate test file for pipeline 'explainability'
generated using Kedro 0.19.4.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
from src.mlops_group_project.pipelines.explainability.nodes import calculate_permutation_importance
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import pytest
import os
import tempfile
import shutil
import mlflow


@pytest.fixture(scope="function")
def mlflow_setup():
    # temporary directories for mlflow
    # this way the original doesnt get messed up
    temp_dir = tempfile.mkdtemp()
    artifact_dir = tempfile.mkdtemp()

    mlflow.set_tracking_uri(f"file://{temp_dir}")
    mlflow.set_artifact_uri(f"file://{artifact_dir}")

    yield  # Stops execution of fixture and gives control to the test

    # Delete the temporary directories
    shutil.rmtree(temp_dir)
    shutil.rmtree(artifact_dir)


def test_calculate_permutation_importance():
    train_sample_filepath = os.path.join("tests\sample\sample_data_model_selection_train.csv")
    val_sample_filepath = os.path.join("tests\sample\sample_data_model_selection_val.csv")
    test_sample_filepath = os.path.join("tests\sample\sample_data_model_selection_test.csv") 

    y_sample_train_filepath = os.path.join("tests\sample\sample_y_train.csv")
    y_sample_val_filepath = os.path.join("tests\sample\sample_y_val.csv")
    y_sample_test_filepath = os.path.join("tests\sample\sample_y_test.csv")


    # dfs
    train_sample = pd.read_csv(train_sample_filepath)
    val_sample = pd.read_csv(val_sample_filepath)
    test_sample = pd.read_csv(test_sample_filepath)

    y_sample_train = pd.read_csv(y_sample_train_filepath)
    y_sample_val = pd.read_csv(y_sample_val_filepath)
    y_sample_test = pd.read_csv(y_sample_test_filepath)

    # it has to be fitted before being passed to the function
    # this would already be done in the pipeline
    model_a = LogisticRegression()
    model_a.fit(train_sample, y_sample_train)

    explainanbility_df = calculate_permutation_importance(
        model_a,
        train_sample,
        y_sample_train,
        n_repeats=5,
        random_state=42
    )
    assert explainanbility_df is not None
    assert isinstance(explainanbility_df, pd.DataFrame)
