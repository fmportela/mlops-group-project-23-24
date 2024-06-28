import logging

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

import mlflow

# TODO add a way to check for column types
# and then apply a column transformer. with the most frequent for categorical and median for numerical

# TODO log stuff to mlflow

log = logging.getLogger(__name__)

def impute_missing_values(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Imputes missing values in the dataframe.

    Args:
        df: pd.DataFrame: Dataframe to impute missing values from.

    Returns:
        pd.DataFrame: Dataframe with missing values imputed.
    """
    
    with mlflow.start_run(run_name='missing_value_imputation', nested=True):

        # impute missing values
        imputer = SimpleImputer(strategy="most_frequent")
        
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)
        X_test_imputed = imputer.transform(X_test)
        
        # converting back to dataframe
        X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
        X_val_imputed = pd.DataFrame(X_val_imputed, columns=X_val.columns)
        X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns)
        
        mlflow.sklearn.log_model(imputer, "imputer_model")
    
    log.info("Missing values imputed.")
    
    return X_train_imputed, X_val_imputed, X_test_imputed, imputer


if __name__ == "__main__":
    X_train = pd.read_csv("data/dev/04_split/X_train.csv")
    X_val = pd.read_csv("data/dev/04_split/X_val.csv")
    X_test = pd.read_csv("data/dev/04_split/X_test.csv")
    
    X_train_imputed, X_val_imputed, X_test_imputed, imputer = impute_missing_values(X_train, X_val, X_test)
    
    print(X_train_imputed.head())
    