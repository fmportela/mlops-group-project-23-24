"""
This is a boilerplate test file for pipeline 'data_split'
generated using Kedro 0.19.4.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pandas as pd
import numpy as np
import os
from src.mlops_group_project.pipelines.data_split.nodes import split_data
import pytest
import math


import math

def test_split_data():
    raw_data = os.path.join("tests/sample/sample_raw_data.csv")
    target_label = "readmitted"

    set_sizes = [0.7, 0.15, 0.15]
    random_state = 42

    data = pd.read_csv(raw_data)

    proportions_target = data[target_label].value_counts(normalize=True)

    proportion_label_0 = proportions_target[0]
    proportion_label_1 = proportions_target[1]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df=data, target_label=target_label, set_sizes=set_sizes, stratify=True, random_state=random_state)
    
    # Verificar se X_train tem aproximadamente 70% dos dados
    train_size = X_train.shape[0]
    expected_train_size = int(data.shape[0] * set_sizes[0])
    tolerance = 0.01  # Definindo uma tolerância de 1%
    assert abs(train_size / expected_train_size - 1) <= tolerance, f"The X_train does not have approximately 70% of the data. Expected {expected_train_size}, got {train_size}."
    
    # Similarmente para X_val e X_test
    val_size = X_val.shape[0]
    expected_val_size = int(data.shape[0] * set_sizes[1])
    assert abs(val_size / expected_val_size - 1) <= tolerance, f"The X_val does not have approximately 15% of the data. Expected {expected_val_size}, got {val_size}."
    

    test_size = X_test.shape[0]
    expected_test_size = int(data.shape[0] * set_sizes[2])
    assert abs(test_size / expected_test_size - 1) <= tolerance, f"The X_test does not have approximately 15% of the data. Expected {expected_test_size}, got {test_size}."




    # check if the data was stratified
    proportions_target_train = y_train.value_counts(normalize=True)
    proportions_target_val = y_val.value_counts(normalize=True)
    proportions_target_test = y_test.value_counts(normalize=True)

    proportions_target_train_0 = proportions_target_train[0]
    proportions_tareget_train_1 = proportions_target_train[1]

    proportions_target_val_0 = proportions_target_val[0]
    proportions_target_val_1 = proportions_target_val[1]
    
    proportions_target_test_0 = proportions_target_test[0]
    proportions_target_test_1 = proportions_target_test[1]

    assert abs(proportion_label_0 - proportions_target_train_0) <= tolerance, "The data was not stratified on train, label 0."
    assert abs(proportion_label_1 - proportions_tareget_train_1) <= tolerance, "The data was not stratified on train, label 1."

    assert abs(proportion_label_0 - proportions_target_val_0) <= tolerance, "The data was not stratified on val, label 0."
    assert abs(proportion_label_1 - proportions_target_val_1) <= tolerance, "The data was not stratified on val, label 1."

    assert abs(proportion_label_0 - proportions_target_test_0) <= tolerance, "The data was not stratified on test, label 0."
    assert abs(proportion_label_1 - proportions_target_test_1) <= tolerance, "The data was not stratified on test, label 1."