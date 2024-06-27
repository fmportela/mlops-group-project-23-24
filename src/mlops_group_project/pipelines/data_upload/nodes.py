from typing import List
from datetime import datetime
import warnings; warnings.filterwarnings("ignore")

import pandas as pd

from .utils import to_feature_store, read_credentials, load_expectation_suite


def upload_data(df: pd.DataFrame,
                group_name: str,
                description: str,
                feature_descriptions: List[dict],
                suite_name: str,
) -> None:
    """
    Upload data to the feature store.

    Args:
        df (pd.DataFrame): Data to upload.
        upload_to_feature_store (bool, optional): Whether to upload to the feature store. Defaults to False.
    """
       
    # adding timestamp to the group name
    df["datetime"] = datetime.now()
    
    # converting NaNs to None (required by the feature store: None for nulls in JSON)
    df = df.applymap(lambda x: None if pd.isna(x) else x)
    
    # uploading to feature store
    settings_store = read_credentials()["SETTINGS_STORE"]
    suite = load_expectation_suite(suite_name)
    
    to_feature_store(
        data=df,
        group_name=group_name,
        feature_group_version=1,
        description=description if not None else "Data uploaded to the feature store",
        group_description=feature_descriptions,
        validation_expectation_suite=suite,
        SETTINGS=settings_store
    )