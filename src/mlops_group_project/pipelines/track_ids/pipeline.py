from kedro.pipeline import Pipeline, pipeline, node
from .nodes import split_features_and_ids


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_features_and_ids,
            inputs=["valid_data", "params:id_column"],
            outputs="ids"
        )
    ])
