import os
import pandas as pd
import pytest
from src.mlops_group_project.pipelines.feature_pruning.nodes import select_features
import numpy as np

def test_feature_prunning():
    # Load the data
    filepath = os.path.join("tests/sample/sample_raw_data.csv")

    data = pd.read_csv(filepath)

    # pick 5 random columns
    columns = data.columns
    np.random.seed(42)
    selected_columns = np.random.choice(columns, 5, replace=False)
    
    # Execute the function
    result = select_features(data, selected_columns)

    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == len(selected_columns)
    assert set(result.columns) == set(selected_columns)
    assert result.shape[0] == data.shape[0]
