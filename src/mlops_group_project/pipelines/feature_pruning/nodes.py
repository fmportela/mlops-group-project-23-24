import pandas as pd

def select_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    return df[features]
