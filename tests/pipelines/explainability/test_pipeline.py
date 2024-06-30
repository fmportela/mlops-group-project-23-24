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
    # diretorios temp, para testar o selection model,
    # e não estragar o ambiente original
    temp_dir = tempfile.mkdtemp()
    artifact_dir = tempfile.mkdtemp()

    # novos caminhos temporários para o mlflow 
    # dar track e não guardar no ambiente original
    mlflow.set_tracking_uri(f"file://{temp_dir}")
    mlflow.set_artifact_uri(f"file://{artifact_dir}")

    yield  # Pausa a execução do fixture e passa o controle para o teste

    # Remove as pastas temporárias dps do teste correr
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

    explainanbility_df = calculate_permutation_importance(
        LogisticRegression(),
        train_sample,
        y_sample_train,
        n_repeats=5,
        random_state=42
    )
    assert explainanbility_df is not None
    assert isinstance(explainanbility_df, pd.DataFrame)
