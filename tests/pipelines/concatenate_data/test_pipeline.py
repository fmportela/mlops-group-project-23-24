"""
This is a boilerplate test file for pipeline 'concatenate_data'
generated using Kedro 0.19.4.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pandas as pd
import pytest
import numpy as np
import os
from src.mlops_group_project.pipelines.concatenate_data.nodes import concatenate_data


def test_concatenate_data():
    X_train_statefull_path = os.path.join("tests/sample/sample_data_X_train_statefull.csv")
    X_val_statefull_path = os.path.join("tests/sample/sample_data_X_val_statefull.csv")
    X_test_statefull_path = os.path.join("tests/sample/sample_data_X_test_statefull.csv")

    y_train_path = os.path.join("tests/sample/sample_y_train.csv")
    y_val_path = os.path.join("tests/sample/sample_y_val.csv")
    y_test_path = os.path.join("tests/sample/sample_y_test.csv")

    X_train_statefull = pd.read_csv(X_train_statefull_path)
    X_val_statefull = pd.read_csv(X_val_statefull_path)
    X_test_statefull = pd.read_csv(X_test_statefull_path)

    y_train = pd.read_csv(y_train_path)
    y_val = pd.read_csv(y_val_path)
    y_test = pd.read_csv(y_test_path)

    all_data_together = concatenate_data(X_train_statefull, 
                                         X_val_statefull, 
                                         X_test_statefull, 
                                         y_train, y_val, 
                                         y_test)
    total_num_rows = X_train_statefull.shape[0] + X_val_statefull.shape[0] + X_test_statefull.shape[0]
    assert all_data_together.shape[0] == total_num_rows

    # check there is no nulls
    # this is a very important check, cause if
    # the data contains nulls the concatenation
    # was not done correctly
    assert all_data_together.isnull().sum().sum() == 0

    collumns_on_data = set(X_train_statefull.columns.tolist() + y_train.columns.tolist() +\
                        X_val_statefull.columns.tolist() + y_val.columns.tolist() +\
                        X_test_statefull.columns.tolist() + y_test.columns.tolist())



    assert set(all_data_together.columns) == collumns_on_data

