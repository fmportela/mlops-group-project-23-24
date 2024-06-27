import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

import mlflow


def calculate_permutation_importance(
    model: BaseEstimator, 
    df: pd.DataFrame, 
    n_repeats: int = 5, 
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate permutation importance for a given model.
    
    Args:
        model: The trained model.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        n_repeats (int): Number of times to permute a feature.
        random_state (int): Random state for reproducibility.
        
    Returns:
        pd.DataFrame: Dataframe containing permutation importance scores.
    """
    
    X = df.drop(columns=["readmitted"])
    y = np.ravel(df["readmitted"])
    
    result = permutation_importance(
        model,
        X, y,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values(
        by='importance_mean',
        ascending=False
    )
    
    # Log the permutation importance as a CSV in mlflow
    with mlflow.start_run(run_name="permutation_importance", nested=True):
        importance_df.to_csv("permutation_importance.csv", index=False)
        mlflow.log_artifact("permutation_importance.csv")
    
    return importance_df