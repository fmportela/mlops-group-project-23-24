from typing import List
from datetime import datetime
import warnings; warnings.filterwarnings("ignore")

import pandas as pd

from .utils import read_credentials, load_feature_group


def load_data(group_name: str) -> pd.DataFrame:
    """
    """
        
    # uploading to feature store
    settings_store = read_credentials()["SETTINGS_STORE"]
    
    df = load_feature_group(
        group_name=group_name,
        feature_group_version=1,
        SETTINGS=settings_store
    )
    
    return df