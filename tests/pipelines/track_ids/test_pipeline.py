import os
import pandas as pd
import pytest
import numpy as np
from src.mlops_group_project.pipelines.track_ids.nodes import split_features_and_ids


def test_split_features_and_ids():
    # Load the data
    filepath = os.path.join("tests/sample/sample_raw_data.csv")

    data = pd.read_csv(filepath)
    print(data.head())
    print(type(data))

    id_column_name = "encounter_id"

    # Execute the function
    result = split_features_and_ids(data, id_column_name)
    print("RESULTS")
    print(result.head())
    print(type(result))
    assert isinstance(result, pd.DataFrame)
    # verify uniqueness of the ids
    assert len(result.encounter_id.unique()) == result.shape[0]
    # check type of column is int64
    assert type(result.encounter_id[0]) in [int, np.int64, np.int32]
