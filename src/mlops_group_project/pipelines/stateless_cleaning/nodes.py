import pandas as pd
import numpy as np


# NOTE: given that hopsworks lowercases the names of the columns, after ingestion those same
# columns remain with their names in lowercase. Therefore, some columns like A1Cresult and diabetesMed
# are originally in camel case, but in the dataset they are in lowercase.


def replace_pseudo_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces pseudo nulls with np.nan.

    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.

    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    """

    # replace pseudo nulls with np.nan
    df = df.replace("?", np.nan)
    return df


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Columns to drop straight away.

    Args:
        df: pd.DataFrame: Dataframe to drop columns from.

    Returns:
        pd.DataFrame: Dataframe with columns dropped.
    """

    # redundant / too many missing values
    columns_to_drop = [
        "weight",
        "payer_code",
        "medical_specialty",
        "encounter_id",
        "patient_nbr"
    ]

    df = df.drop(columns=columns_to_drop)
    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the 'gender' column.

    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.

    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    """

    gender_replace = {"Male": 0, "Female": 1, "Unknown/Invalid": 1}

    df["gender"] = df["gender"].map(gender_replace)
    return df


def encode_age_bracket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal encoding of the 'age' column.

    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.

    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    """
    dict_age = {
        "[0-10)": 5,
        "[10-20)": 15,
        "[20-30)": 25,
        "[30-40)": 35,
        "[40-50)": 45,
        "[50-60)": 55,
        "[60-70)": 65,
        "[70-80)": 75,
        "[80-90)": 85,
        "[90-100)": 95,
    }

    df["age"] = df["age"].map(dict_age)
    return df


def encode_race(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the 'race' column.

    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.

    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    """

    dict_replace_race = {
        "Caucasian": 0,
        "AfricanAmerican": 1,
        "Other": 2,
        "Asian": 3,
        "Hispanic": 4,
    }

    df["race"] = df["race"].map(dict_replace_race)
    return df


def encode_diabetes_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the diabetes columns.

    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.

    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    """
    dict_diabetes_med = {"No": 0, "Yes": 1}

    df["diabetesmed"] = df["diabetesmed"].map(dict_diabetes_med)

    dict_change_transform = {"No": 0, "Ch": 1}

    df["change"] = df["change"].map(dict_change_transform)
    return df


def encode_test_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the test results columns.

    Args:
        df: pd.DataFrame: Dataframe to replace tokens in.

    Returns:
        pd.DataFrame: Dataframe with tokens replaced.
    """
    dict_transform_a1cresult = {"Norm": 1, ">7": 2, ">8": 3, np.nan: 0}

    df["a1cresult"] = df["a1cresult"].replace(dict_transform_a1cresult)

    dict_max_glu_serum = {"Norm": 1, ">200": 2, ">300": 3, np.nan: 0}

    df["max_glu_serum"] = df["max_glu_serum"].replace(dict_max_glu_serum)
    return df


def clean_df(X: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data.

    Args:
        X: pd.DataFrame: Features.
    
    Returns:
        pd.DataFrame: Cleaned features.
    """

    cleaning_functions = [
        replace_pseudo_nulls,
        drop_unwanted_columns,
        encode_gender,
        encode_age_bracket,
        encode_race,
        encode_diabetes_columns,
        encode_test_results
    ]

    for func in cleaning_functions:
        X = func(X)


    return X
