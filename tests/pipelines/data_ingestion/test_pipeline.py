import os
import pandas as pd
import pytest
from src.mlops_group_project.pipelines.data_ingestion.nodes import load_data

def test_load_data():
    group_name = "dev_raw_data"
    df = load_data(group_name)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert df.shape[1] > 0
    # verify that there is data in the dataframe
    assert not df.empty