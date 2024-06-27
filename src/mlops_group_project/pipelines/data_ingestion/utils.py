import os
from pathlib import Path
from typing import Union

import pandas as pd

import hopsworks

from kedro.config import OmegaConfigLoader, MissingConfigException
from kedro.framework.project import settings


def read_credentials(path: Union[str, Path] = None) -> dict:
    """
    Read credentials from a YAML file.

    Args:
        path (Union[str, Path], optional): Path to the YAML file. Defaults to None.
        key (str, optional): Key to read from the credentials. Defaults to None.
    
    Returns:
        dict: Dictionary with the credentials.
    """
    
    # code sourced from: https://docs.kedro.org/en/stable/configuration/credentials.html
    
    if path is None:
        conf_path = str(Path(os.getcwd()) / settings.CONF_SOURCE)
    else:
        conf_path = str(Path(path) / settings.CONF_SOURCE)
        
    conf_loader = OmegaConfigLoader(conf_source=conf_path)
    
    try:
        credentials = conf_loader["credentials"]
    except MissingConfigException:
        credentials = {}
    
    return credentials


def load_feature_group(
    group_name: str,
    feature_group_version: int,
    SETTINGS: dict
) -> pd.DataFrame:
    """
    """
    # Connect to feature store.
    project = hopsworks.login(
        api_key_value=SETTINGS["FS_API_KEY"], project=SETTINGS["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()

    # Retrieve the feature group
    feature_group = feature_store.get_feature_group(
        name=group_name,
        version=feature_group_version
    ) 
    
    # load data
    df = feature_group.read()
    
    # filtering for latest data
    df = df.loc[df["datetime"] == df["datetime"].max()]
    df = df.drop(columns=["datetime"])
    
    return df