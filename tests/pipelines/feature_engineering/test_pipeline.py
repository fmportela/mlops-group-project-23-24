import pytest
import pandas as pd
import numpy as np
import os
from src.mlops_group_project.pipelines.feature_engineering.nodes import total_visits_in_previous_year, add_features,comorbidity_index



def test_comorbidity_index():
    filepath = os.path.join("tests/sample/sample_data_staless_try_commorbity.csv")
    df_sample = pd.read_csv(filepath)
    df_sample = comorbidity_index(df_sample)
    assert 'comorbidity_index' in df_sample.columns, "A coluna comorbidity_index não está a ser adicionada ao dataframe."
    assert df_sample['comorbidity_index'].dtype == np.int64, "O tipo de dados da coluna comorbidity_index não está correto. Deveria ser um inteiro."




def test_total_visits_in_previous_year():
    def contains_negative_values(df, col):
        return df[col].lt(0).any()
    filepath = os.path.join("tests/sample/sample_raw_data.csv")
    df_sample = pd.read_csv(filepath)
    df_sample = total_visits_in_previous_year(df_sample)

    assert not contains_negative_values(df_sample, 'total_visits_in_previous_year'), "A soma está a obter resultados negativos. Este número não pode ser negativo."




    
def test_add_features():
    filepath = os.path.join("tests/sample/sample_data_staless_try_commorbity.csv")
    df_sample = pd.read_csv(filepath)
    

    df_sample_modified = add_features(df_sample)
    
    assert 'comorbidity_index' in df_sample_modified.columns, "A coluna comorbidity_index não está a ser adicionada ao dataframe."
    assert 'total_visits_in_previous_year' in df_sample_modified.columns, "A coluna total_visits_in_previous_year não está a ser adicionada ao dataframe."
    assert df_sample_modified['comorbidity_index'].dtype == np.int64, "O tipo de dados da coluna comorbidity_index não está correto. Deveria ser um inteiro."
    assert not df_sample_modified['total_visits_in_previous_year'].lt(0).any(), "A soma está a obter resultados negativos. Este número não pode ser negativo."


if __name__ == "__main__":
    test_add_features