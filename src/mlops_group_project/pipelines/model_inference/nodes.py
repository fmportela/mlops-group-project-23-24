from typing import Union

import pandas as pd
from sklearn.base import BaseEstimator

import mlflow
from mlflow.tracking import MlflowClient


def load_registered_model_version(model_name: str, version: int = -1) -> Union[BaseEstimator, None]:
    """
    Load a specific version of a registered model from the
    MLflow Model Registry. If version is -1, load the latest version.
    Useful: https://mlflow.org/docs/latest/model-registry.html    
    
    Args:
        model_name (str): The name of the registered model.
        version (int): The version of the model to load, or -1 for the latest version.
    
    Returns:
        model: The loaded model. Or None if there is no matching model.
    """
    client = MlflowClient()

    if version == -1:
        # get the latest version
        versions = client.get_latest_versions(model_name)
        if not versions:
            raise Exception(f"No versions found for model {model_name}")
        latest_version = max(versions, key=lambda v: v.version).version
        version = latest_version
    
    model_uri = f"models:/{model_name}/{version}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        raise Exception(f"Could not load model from Registry: {e}")
    
    
def locate_champion_model(alias: str = "champion") -> Union[BaseEstimator, None]:
    """
    Locate the champion model in the MLflow Model Registry.
    
    Returns:
        model: The champion model. Or None if there is no champion model.
    """
    client = MlflowClient()
    
    models = client.search_registered_models()
    for model in models:
        for alias in model.aliases.keys():
            if alias == "champion":
                return load_registered_model_version(model.name, model.aliases[alias])
    
    return None


def make_predictions(X: pd.DataFrame, ids: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions using a trained model.
    
    Args:
        X (pd.DataFrame): The data to make predictions on.
    
    Returns:
        pd.DataFrame: The predictions.
    """
    
    champion_model = locate_champion_model()
    
    if champion_model is None:
        raise Exception("No champion model found")
    
    predictions_df = pd.DataFrame(champion_model.predict(X), columns=["prediction"])
    predictions_df = pd.concat([ids, predictions_df], axis=1)
    
    return predictions_df
