# import pytest
# from sklearn.impute import SimpleImputer
# from src.mlops_group_project.pipelines.stateful_cleaning.nodes import impute_missing_values
# import os
# import pandas as pd
# import numpy as np
# import math


# def test_impute_missing_values():
#     train_filepath = os.path.join("tests/sample/sample_X_train_stateless.csv")
#     val_filepath = os.path.join("tests/sample/sample_X_val_stateless.csv")
#     test_filepath = os.path.join("tests/sample/sample_X_test_stateless.csv")

#     X_train = pd.read_csv(train_filepath)
#     X_val = pd.read_csv(val_filepath)
#     X_test = pd.read_csv(test_filepath)
#     # df_sample = pd.read_csv(filepath)
    
#     # verify that there are missing values in the dataframe
#     assert X_train.isnull().sum().sum() > 0, "There are no missing values in the dataframe."
#     assert X_val.isnull().sum().sum() > 0, "There are no missing values in the dataframe."
#     assert X_test.isnull().sum().sum() > 0, "There are no missing values in the dataframe."

#     X_train_imputed, X_val_imputed, X_test_imputed, imputer = impute_missing_values(X_train, X_val, X_test)
#     assert X_train_imputed.isnull().sum().sum() == 0, "There are still missing values in the dataframe."
#     assert X_val_imputed.isnull().sum().sum() == 0, "There are still missing values in the dataframe."
#     assert X_test_imputed.isnull().sum().sum() == 0, "There are still missing values in the dataframe."
#     assert isinstance(imputer, SimpleImputer), "The imputer is not an instance of SimpleImputer."