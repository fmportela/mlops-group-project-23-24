import logging

import pandas as pd

# joining the data because we believe the explainability and data drift parts can/should
# be done with the whole dataset

log = logging.getLogger(__name__)

def concatenate_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
    y_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Concatenate the training, validation, and test data along the rows.
    
    Args:
        X_train: Training features.
        X_val: Validation features.
        X_test: Test features.
        y_train: Training target.
        y_val: Validation target.
        y_test: Test target.
    
    Returns:
        Concatenated data.
    """
    
    X_train["readmitted"] = y_train
    X_val["readmitted"] = y_val
    X_test["readmitted"] = y_test
    
    log.info("Concatenated Data")
    
    return pd.concat([X_train, X_val, X_test], axis=0)