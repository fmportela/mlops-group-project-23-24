import logging

import pandas as pd
from sklearn.base import BaseEstimator


log = logging.getLogger(__name__)


def apply_transformations(data: pd.DataFrame, model: BaseEstimator) -> pd.DataFrame:
    
    transformed_data = model.transform(data)
    transformed_data = pd.DataFrame(transformed_data, columns=data.columns)
    
    log.info("Transformed Data")
    
    return transformed_data
