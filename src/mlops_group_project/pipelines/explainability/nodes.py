import logging
import warnings; warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

import mlflow

import shap


log = logging.getLogger(__name__)

def calculate_permutation_importance(
    model: BaseEstimator, 
    X: pd.DataFrame, 
    y: pd.Series,
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
    
    with mlflow.start_run(run_name="permutation_importance", nested=True):        
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
        
        mlflow.log_params({
            'n_repeats': n_repeats,
            'random_state': random_state
        })
        
        for i, row in importance_df.iterrows():
            mlflow.log_param(f"{row['feature']}_importance_mean", row['importance_mean'])
            mlflow.log_param(f"{row['feature']}_importance_std", row['importance_std'])
        
    
    log.info(f"Most important features calculated: {importance_df.head(5)}")
            
    return importance_df

def calculate_shapley_values(
    model: BaseEstimator, 
    X: pd.DataFrame, 
    y: pd.Series
) -> pd.DataFrame:
    """
    Calculate Shapley values for a given model.
    
    Args:
        model: The trained model.
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        
    Returns:
        pd.DataFrame: Dataframe containing Shapley values.
    """
    
    with mlflow.start_run(run_name="shapley_values", nested=True):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)

        shap.initjs()

        # Create and save SHAP explainer plot
        shap_explainer = shap.force_plot(explainer.expected_value, shap_values.values[0, :], X.iloc[0, :], matplotlib=True)
        explainer_path = "shap_explainer.png"
        plt.savefig(explainer_path)
        plt.close()

        # Create and save SHAP summary plot
        shap_summary = shap.summary_plot(shap_values, X, show=False)
        summary_path = "shap_summary.png"
        plt.savefig(summary_path)
        plt.close()

        # Log plots as artifacts
        mlflow.log_artifact(explainer_path)
        mlflow.log_artifact(summary_path)

        mlflow.log_params({
            'explainer': explainer,
            'shap_values': shap_values
        })

        log.info(f"Shapley values calculated": {shap_values})
        
    return shap_values