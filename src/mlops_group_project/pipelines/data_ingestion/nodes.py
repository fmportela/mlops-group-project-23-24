import logging
from typing import List
from datetime import datetime
import warnings; warnings.filterwarnings("ignore")

import pandas as pd

from .utils import read_credentials, load_feature_group


log = logging.getLogger(__name__)


def load_data(group_name: str) -> pd.DataFrame:
    """
    Load data from the feature store.
    
    Args:
        group_name: Name of the feature group.
    
    Returns:
        Data from the feature store.
    """
        
    # uploading to feature store
    settings_store = read_credentials()["SETTINGS_STORE"]
    
    df = load_feature_group(
        group_name=group_name,
        feature_group_version=1,
        SETTINGS=settings_store
    )
    
    log.info(f"Loaded data from feature store: {group_name} | Shape: {df.shape}")
    
    return df