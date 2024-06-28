import logging
import pandas as pd


log = logging.getLogger(__name__)

def select_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Prune the features in the dataset.
    
    Args:
        df: Dataset.
        features: Features to keep.
    
    Returns:
        Dataset with only the selected features.
    """
    
    pruned_df = df[features]
    
    log.info(f"Pruned Data | Shape: {pruned_df.shape}")
    
    return pruned_df