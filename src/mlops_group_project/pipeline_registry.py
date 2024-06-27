from typing import Dict

from kedro.pipeline import pipeline, Pipeline
from .pipelines import (
    data_upload,
    data_ingestion,
    stateless_cleaning,
    data_split,
    stateful_cleaning,
    feature_engineering,
    feature_selection,
    model_selection,
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
    
    
    return {
        "data_upload": data_upload_pipeline,  # env. dependent
        
        "dev": data_ingestion_pipeline +\
            stateless_cleaning_pipeline +\
                data_split_pipeline +\
                    stateful_cleaning_pipeline +\
                        feature_engineering_pipeline +\
                            feature_selection_pipeline +\
                                model_selection_pipeline,
                   
        "prod": data_ingestion_pipeline +\
            stateless_cleaning_pipeline,
    }
