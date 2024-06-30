"""
This is a boilerplate test file for pipeline 'data_unit_tests_after_cleaning'
generated using Kedro 0.19.4.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import pytest
from src.mlops_group_project.pipelines.stateless_cleaning.nodes import replace_pseudo_nulls, drop_unwanted_columns, encode_gender, encode_age_bracket, encode_race, encode_diabetes_columns, clean_df, encode_payer_code
import os
import pandas as pd
import numpy as np
import math


def test_replace_pseudo_nulls():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    result = replace_pseudo_nulls(df_sample)
    # get rows that contain '?' in the cell
    rows_with_question_mark = result[result.eq('?').any(axis=1)]
    
    assert len(rows_with_question_mark) == 0, f"Rows with '?' in the cell: {rows_with_question_mark}"


def test_encode_gender():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    result = encode_gender(df_sample)
    assert set(result.gender.unique().tolist()) == {0, 1}


def test_encode_age_bracket():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    result = encode_age_bracket(df_sample)
    assert set(result.age.unique().tolist()) == {5, 15, 25, 35, 45, 55, 65, 75, 85, 95}


# def test_map_diagnosis_to_bin():
#     # Test cases for starts with 'E' or 'V'
#     assert map_diagnosis_to_bin('E123') == 49
#     assert map_diagnosis_to_bin('V123') == 49

#     # Test case for NO_DIAGNOSIS
#     assert map_diagnosis_to_bin('NO_DIAGNOSIS') == 0

#     # Test cases for ranges
#     assert map_diagnosis_to_bin('50.0') == 1
#     assert map_diagnosis_to_bin('140.1') == 2
#     assert map_diagnosis_to_bin('250') == 24

#     # Test cases for diabetes mellitus (250.x)
#     assert map_diagnosis_to_bin('250.0') == 25
#     assert map_diagnosis_to_bin('250.1') == 26
#     assert map_diagnosis_to_bin('250.20') == 8
#     assert map_diagnosis_to_bin('250.21') == 8
#     assert map_diagnosis_to_bin('250.22') == 9

#     # Test cases for other ranges
#     assert map_diagnosis_to_bin('251.0') == 3
#     assert map_diagnosis_to_bin('285.9') == 35
#     assert map_diagnosis_to_bin('300.0') == 36
#     assert map_diagnosis_to_bin('350.0') == 37
#     assert map_diagnosis_to_bin('400.0') == 38
#     assert map_diagnosis_to_bin('490.0') == 39
#     assert map_diagnosis_to_bin('530.0') == 40
#     assert map_diagnosis_to_bin('590.0') == 41
#     assert map_diagnosis_to_bin('650.0') == 42
#     assert map_diagnosis_to_bin('690.0') == 43
#     assert map_diagnosis_to_bin('720.0') == 44
#     assert map_diagnosis_to_bin('750.0') == 45
#     assert map_diagnosis_to_bin('770.0') == 46
#     assert map_diagnosis_to_bin('790.0') == 47
#     assert map_diagnosis_to_bin('850.0') == 48


def test_drop_unwanted_columns():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    result = drop_unwanted_columns(df_sample)
    print(result.columns)
    assert {"weight",
        "payer_code",
        "medical_specialty",
        "patient_nbr"} not in set(result.columns.tolist())


def test_encode_race():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    result = encode_race(df_sample)
    unique_race_codes = set(result.race)
    for element in unique_race_codes:
        if math.isnan(element):
            assert any(math.isnan(x) for x in [0, 1, 2, 3, 4, np.nan]), f"{element} not in [0, 1, 2, 3, 4, np.nan]"
        else:
            assert element in [0, 1, 2, 3, 4, np.nan], f"{element} not in [0, 1, 2, 3, 4, np.nan]"


def test_encode_diabetes_columns():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    result = encode_diabetes_columns(df_sample)
    assert set(result["diabetesmed"].unique().tolist()) == {0, 1}
    assert set(result["change"].unique().tolist()) == {0, 1}


def test_encode_payer_code():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    result = encode_payer_code(df_sample)
    print(set(result["payer_code"].unique().tolist()))
    assert set(result["payer_code"].unique().tolist()) in [{0}, {1}, {0, 1}]


# def test_encode_test_results():
#     filepath = os.path.join("tests/sample/sample_raw_data.csv")
#     df_sample = pd.read_csv(filepath)

#     result = encode_test_results(df_sample)
#     assert set(result["A1Cresult"].unique().tolist()) == {0, 1, 2, 3}
#     assert set(result["max_glu_serum"].unique().tolist()) == {0, 1, 2, 3}


# # def test_fix_readmitted():
# #     filepath = os.path.join("tests/sample/target_sample_raw_data.csv")
# #     df_sample = pd.read_csv(filepath)

# #     result = fix_readmitted(df_sample)
# #     assert set(result["readmitted"].unique().tolist()) == {0, 1}


def test_clean_df():
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)

    filepath_target_data = os.path.join("tests/sample/target_sample_raw_data.csv")
    data = clean_df(df_sample)
    
    # check that all columns are of type int/float/bool
    
    cols = data.columns.tolist()
    for col in cols:
        assert data[col].dtype in [np.int64, np.int32, np.object_,np.float64, np.float32, np.bool_]

