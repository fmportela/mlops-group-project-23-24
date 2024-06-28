from kedro.pipeline import Pipeline, pipeline, node
from .nodes import apply_transformations


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=apply_transformations,
            inputs=["featurized_data", "imputer"],
            outputs="stateful_data"
        ),
    ])
