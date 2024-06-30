import logging

import pandas as pd


# NOTE: since feat. eng. is done before imputation we ought to be able to handle NaNs


log = logging.getLogger(__name__)


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


def comorbidity_index_calculator(*diagnosis, age=None):
    """
    Charlson Comorbidity Index
    
    according to studies a important feature to predict whether a patient
    will return to hospital shortly after being discharged is the comorbity
    index. There are various but we will use the most famous one, the Charlson
    comorbidity index. Comorbidity refers to the presence of one or more additional
    diseases or disorders co-occurring with a primary disease or disorder in a patient.
    https://en.wikipedia.org/wiki/Charlson_Comorbidity_Index
    
    1 each: Myocardial infarction, congestive heart failure, peripheral vascular disease,
            cerebrovascular disease, dementia, chronic pulmonary disease, rheumatologic disease,
            peptic ulcer disease, liver disease (if mild, or 3 if moderate/severe),
            diabetes (if controlled, or 2 if uncontrolled)
    2 each: Hemiplegia or paraplegia, renal disease, 
            malignancy (if localized, or 6 if metastatic tumor), leukemia, lymphoma
    6 each: AIDS
    
    50-59 years old: +1 point
    60-69 years old: +2 points
    70-79 years old: +3 points
    80 years old or more: +4 points
    """
    
    disease_points = {
        '410': 1,  # myocardial infarction / acute myocardial
        '428': 1,  # congestive heart failure
        '443': 1,  # peripheral vascular disease
        '436': 1,  # cerebrovascular disease (broad term, more than one associated with it)
        '437': 1,  # cerebrovascular disease
        '438': 1,  # cerebrovascular disease
        '290': 1,  # dementia (broad term, more than one associated with it)
        '291': 1,  # dementia
        '490': 1,  # chronic pulmonary disease  (broad term, more than one associated with it)
        '491': 1,  # chronic pulmonary disease
        '492': 1,  # chronic pulmonary disease
        '493': 1,  # chronic pulmonary disease
        '494': 1,  # chronic pulmonary disease
        '495': 1,  # chronic pulmonary disease
        '496': 1,  # chronic pulmonary disease
        '714': 1,  # rheumatologic disease
        '533': 1,  # peptic ulcer disease
        '571': 1,  # liver disease
        '250': 2,  # diabetes
        '342': 2,  # Hemiplegia or paraplegia
        '343': 2,  # Hemiplegia or paraplegia
        '344': 2,  # Hemiplegia or paraplegia
        '403': 2,  # renal disease
        '404': 2,  # renal disease
        '585': 2,  # renal disease
        '042': 3,  # AIDS
    }
    
    # the diagnosis can be of the same disease and we need to avoid to double count it
    # also we really only just want the first 3 letters
    codes = set([code[:3] for code in diagnosis if isinstance(code, str)])
    
    index = 0
    for code in codes:
        index += disease_points.get(code, 0)
        # for malignant diseases: 140-239, where from 196-199 they are metastic and worth 6 points
        # leukimia and lymphoma are comprised in the above categories
        try:
            code_int = int(code)
            if (code_int >= 140) and (code_int <= 239):
                if code_int >= 196 and code_int <= 199:
                    index += 6
                else:
                    index += 2
        except ValueError:
            # for dealing with inconvertible icd9 codes, e.g. V58
            pass
    
    # regarding age (age is optional)
    if age is not None:
        if age == 55:
            index += 1
        elif age == 65:
            index += 2
        elif age == 75:
            index += 3
        elif age > 80:
            index += 4
                 
    return index

def comorbidity_index(df):
    df['comorbidity_index'] = df.apply(lambda x: comorbidity_index_calculator(x['diag_1'],
                                                                              x['diag_2'],
                                                                              x['diag_3'],
                                                                              age=x['age']),
                                     axis=1)
    
    df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'])
    
    return df

def calculate_complexity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the complexity score of the patient.
    
    Args:
        df: pd.DataFrame: Dataframe to calculate the feature for.
    
    Returns:
        pd.DataFrame: Dataframe with the new feature.
    """
    df['complexity_score'] = df['num_medications'] + df['num_lab_procedures'] + df['number_diagnoses']
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
        comorbidity_index,
        calculate_complexity_score,
    ]

    for func in fe_functions:
        df = func(df)
    
    log.info(f"Added features to the dataframe: {df.shape}")

    return df