"""
This is a boilerplate test file for pipeline 'apply_stateful_transformations'
generated using Kedro 0.19.4.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
from src.mlops_group_project.pipelines.apply_stateful_transformations.nodes import *
import pandas as pd
import numpy as np
import pytest
import os
from sklearn.impute import SimpleImputer


def test_apply_transformations():
    # Load the data
    filepath = os.path.join("tests/sample/sample_X_train_stateless.csv")

    data = pd.read_csv(filepath)
    imputer_test = SimpleImputer(strategy='most_frequent')

    imputer_test.fit(data)

    filepath_test_file = os.path.join("tests/sample/sample_X_test_stateless.csv")

    data_test = pd.read_csv(filepath_test_file)

    # verify there are missing values
    assert data_test.isnull().sum().sum() > 0

    # Execute the function
    result = apply_transformations(data_test, imputer_test)

    # verify there are no missing values
    assert result.isnull().sum().sum() == 0
    # verify it returns a dataframe
    assert isinstance(result, pd.DataFrame)
    # verify the shape is the same
    assert result.shape == data_test.shape

