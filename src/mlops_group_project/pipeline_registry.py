from typing import Dict

from kedro.pipeline import Pipeline
from .pipelines import (
    data_upload,
    data_ingestion,
    stateless_cleaning,
    data_split,
    stateful_cleaning,
    feature_engineering,
    feature_selection,
    model_selection,
    concatenate_data,
    explainability,
    data_drift,
    track_ids,
    apply_stateful_transformations,
    data_unit_tests_after_processing,
    feature_pruning,
    model_inference,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """
    Typical flow:
    1. make sure there is data available in the feature store (i.e. run the data_upload pipeline)
    2. run either the dev or prod pipelines (some parts of prod are dev dependent)
    
    Commands.
    data_upload:
        dev version: `kedro run --pipeline data_upload --env=dev`
        prod version: `kedro run --pipeline data_upload --env=prod`
    dev: `kedro run --pipeline dev --env=dev`
    prod: `kedro run --pipeline prod --env=prod`
    
    """

    data_upload_pipeline = data_upload.create_pipeline()
    data_ingestion_pipeline = data_ingestion.create_pipeline()
    stateless_cleaning_pipeline = stateless_cleaning.create_pipeline()
    data_split_pipeline = data_split.create_pipeline()
    stateful_cleaning_pipeline = stateful_cleaning.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()
    feature_selection_pipeline = feature_selection.create_pipeline()
    model_selection_pipeline = model_selection.create_pipeline()
    concatenate_data_pipeline = concatenate_data.create_pipeline()
    explainability_pipeline = explainability.create_pipeline()
    data_drift_pipeline = data_drift.create_pipeline()
    track_ids_pipeline = track_ids.create_pipeline()
    apply_stateful_transformations_pipeline = apply_stateful_transformations.create_pipeline()
    data_unit_tests_after_processing_pipeline = data_unit_tests_after_processing.create_pipeline()
    model_inference_pipeline = model_inference.create_pipeline()
    feature_pruning_pipeline = feature_pruning.create_pipeline()
    model_inference_pipeline = model_inference.create_pipeline()
    
    return {
        # individual pipelines for debugging
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
        
        # main pipelines
        # run this when you want to upload data to the feature store (i.e. when you have new data)
        "data_upload": data_upload_pipeline,  # env. dependent i.e. kedro --pipeline data_upload --env=dev or prod
        
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
