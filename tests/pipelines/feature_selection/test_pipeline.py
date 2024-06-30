from src.mlops_group_project.pipelines.feature_selection.nodes import select_features, filter_dataset
import pandas as pd
import numpy as np
import pytest
import os


def test_feature_selection():
    path_to_label = os.path.join("tests/sample/sample_y_train.csv") 
    path_to_features = os.path.join("tests/sample/sample_data_statefull_X_train.csv")
    
    X_train = pd.read_csv(path_to_features)
    y_train = pd.read_csv(path_to_label)
    
    # 5 random features from the dataset
    col_names = X_train.columns
    # pick 5 random features
    random_features = np.random.choice(col_names, 5, replace=False).tolist()
    
    list_of_feature_selection_methods = ["rfe", "tree", "all", "manual",random_features, 5]
    
    params = {"random_state": 42}
    n_features = 5
    
    for technique in list_of_feature_selection_methods:
        if technique == "rfe":
            selected_cols = select_features(X_train= X_train, 
                                            y_train=y_train, 
                                            feature_selection=technique, 
                                            model_params=params,
                                            n_features=n_features,
                                            manual_features=random_features
                                            )
            assert len(selected_cols) == n_features
            assert isinstance(selected_cols, list)
        elif technique == "tree":
            selected_cols = select_features(X_train= X_train, 
                                            y_train=y_train, 
                                            feature_selection=technique, 
                                            model_params=params,
                                            n_features=n_features,
                                            manual_features=random_features
                                            )
            assert len(selected_cols) == n_features
            assert isinstance(selected_cols, list)
        elif technique == "all":
            selected_cols = select_features(X_train= X_train, 
                                            y_train=y_train, 
                                            feature_selection=technique, 
                                            model_params=params,
                                            n_features=n_features,
                                            manual_features=random_features
                                            )
            assert len(selected_cols) == len(col_names)
            assert isinstance(selected_cols, list)
        elif technique == "manual":
            for i in range(3):
                if i==0:
                    # very if the function raise ValueError
                    with pytest.raises(ValueError):
                        selected_cols = select_features(X_train= X_train, 
                                                        y_train=y_train, 
                                                        feature_selection=technique, 
                                                        model_params=params,
                                                        n_features=n_features,
                                                        manual_features="ajk"
                                                        )
                elif i==1:
                    # very if the function raise ValueError
                    with pytest.raises(ValueError):
                        selected_cols = select_features(X_train= X_train, 
                                y_train=y_train, 
                                feature_selection=technique, 
                                model_params=params,
                                n_features=n_features,
                                manual_features=[]
                                )
                else:
                    selected_cols = select_features(X_train= X_train, 
                                                    y_train=y_train, 
                                                    feature_selection=technique, 
                                                    model_params=params,
                                                    n_features=n_features,
                                                    manual_features=random_features
                                                    )
                    assert len(selected_cols) == len(random_features)
                    assert isinstance(selected_cols, list)
            
        elif isinstance(technique, list):
            selected_cols = select_features(X_train= X_train, 
                                            y_train=y_train, 
                                            feature_selection=technique, 
                                            model_params=params,
                                            n_features=n_features,
                                            manual_features=random_features
                                            )
            assert len(selected_cols) == len(technique)
            assert isinstance(selected_cols, list)
            assert selected_cols == technique
        else:
            # very if the function raise ValueError
            with pytest.raises(ValueError):
                selected_cols = select_features(X_train= X_train, 
                                                y_train=y_train, 
                                                feature_selection=technique, 
                                                model_params=params,
                                                n_features=n_features,
                                                manual_features=random_features
                                                )


def test_filter_dataset():
    path_to_train = os.path.join("tests/sample/sample_data_statefull_X_train.csv")
    X_train = pd.read_csv(path_to_train)
    # 5 random features from the dataset
    col_names = X_train.columns
    # pick 5 random features
    random_features = np.random.choice(col_names, 5, replace=False).tolist()

    filtered_data = filter_dataset(X_train, random_features)
    assert filtered_data.shape[1] == 5
    assert isinstance(filtered_data, pd.DataFrame)
    assert filtered_data.columns.tolist() == random_features
    assert filtered_data.shape[0] == X_train.shape[0]
    