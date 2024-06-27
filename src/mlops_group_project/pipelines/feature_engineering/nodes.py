import pandas as pd


def total_visits_in_previous_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the total number of visits in the previous year.
    
    Args:
        df: pd.DataFrame: Dataframe to calculate the feature for.
    
    Returns:
        pd.DataFrame: Dataframe with the new feature.
    """
    df['total_visits_in_previous_year'] = df['number_outpatient'] \
                                        + df['number_emergency'] \
                                        + df['number_inpatient']
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds new features to the dataframe.

    Args:
        df: pd.DataFrame: Dataframe to clean.

    Returns:
        pd.DataFrame: Engineered dataframe.
    """

    fe_functions = [
        total_visits_in_previous_year,
    ]

    for func in fe_functions:
        df = func(df)

    return df