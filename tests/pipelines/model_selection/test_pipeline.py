import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.base import BaseEstimator
from src.mlops_group_project.pipelines.model_selection.nodes import select_model
import os
import tempfile
import shutil
import mlflow


@pytest.fixture(scope="function")
def mlflow_setup():
    # temporary directories for mlflow.
    # this way the original doesnt get messed up
    temp_dir = tempfile.mkdtemp()
    artifact_dir = tempfile.mkdtemp()


    mlflow.set_tracking_uri(f"file://{temp_dir}")
    mlflow.set_artifact_uri(f"file://{artifact_dir}")

    yield  # Pausa a execução do fixture e passa o controle para o teste

    # Remove as pastas temporárias dps do teste correr
    shutil.rmtree(temp_dir)
    shutil.rmtree(artifact_dir)


def test_model_selection():
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

    best_model = select_model(
        X_train=train_sample,
        y_train=y_sample_train,
        X_val=val_sample,
        y_val=y_sample_val,
        X_test=test_sample,
        y_test=y_sample_test,
        n_trials=1,
    )
    assert isinstance(best_model, BaseEstimator)
