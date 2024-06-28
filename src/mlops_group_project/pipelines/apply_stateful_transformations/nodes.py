import pandas as pd
from sklearn.base import BaseEstimator


def apply_transformations(data: pd.DataFrame, model: BaseEstimator) -> pd.DataFrame:
    
    transformed_data = model.transform(data)
    transformed_data = pd.DataFrame(transformed_data, columns=data.columns)
    
    return transformed_data
