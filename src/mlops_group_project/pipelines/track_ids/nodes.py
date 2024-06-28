import logging
import pandas as pd

log = logging.getLogger(__name__)

def split_features_and_ids(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """
    Split the DataFrame into features and IDs.
    
    Args:
        df: The DataFrame containing the features and IDs.
        id_column: The name of the column containing the IDs.
    
    Returns:
        A DataFrame containing the features.
    """
    
    log.info("Splitting features and IDs.")
    
    return df[[id_column]]