import streamlit as st
import os

os.chdir("../")
os.chdir("src/mlops_group_project/pipelines")

from data_upload.pipeline import data_upload_pipeline
from stateless_cleaning.pipeline import stateless_cleaning_pipeline
from data_split.pipeline import 
from stateful_cleaning import pipeline
from feature_engineering import pipeline
from feature_selection import pipeline
from model_selection import pipeline
from concatenate_data import pipeline
from explainability import pipeline
from data_drift import pipeline
from track_ids import pipeline
from apply_stateful_transformations import pipeline
from data_unit_tests_after_processing import pipeline
from model_inference import pipeline

# Title of the app
st.title("MLOps Project GUI")

# Dropdown menu for selecting the pipeline
individual_nodes_options = {
        "data_ingestion": data_ingestion_pipeline,
        "stateless_cleaning": stateless_cleaning_pipeline,
        "data_split": data_split_pipeline,
        "stateful_cleaning": stateful_cleaning_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        "feature_selection": feature_selection_pipeline,
        "model_selection": model_selection_pipeline,
        "concatenate_data": concatenate_data_pipeline,
        "explainability": explainability_pipeline,
        "data_drift": data_drift_pipeline,
        "track_ids": track_ids_pipeline,
        "apply_stateful_transformations": apply_stateful_transformations_pipeline,
        "data_unit_tests_after_processing": data_unit_tests_after_processing_pipeline,  # prod only in this PoC
        "feature_pruning": feature_pruning_pipeline,
        "model_inference": model_inference_pipeline, 
}

pipelines_options = {
    "data_upload": data_upload_pipeline,

    "dev": data_ingestion_pipeline +\
            stateless_cleaning_pipeline +\
                feature_engineering_pipeline +\
                    data_split_pipeline +\
                        stateful_cleaning_pipeline +\
                            feature_selection_pipeline +\
                                model_selection_pipeline +\
                                    concatenate_data_pipeline +\
                                        explainability_pipeline +\
                                            data_drift_pipeline,
    
    "prod": data_ingestion_pipeline +\
            track_ids_pipeline +\
                stateless_cleaning_pipeline +\
                    feature_engineering_pipeline +\
                        apply_stateful_transformations_pipeline +\
                            data_unit_tests_after_processing_pipeline +\
                                feature_pruning_pipeline +\
                                    model_inference_pipeline +\
                                        data_drift_pipeline,
    }