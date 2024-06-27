from kedro.pipeline import Pipeline, pipeline, node
from .nodes import add_features


def create_pipeline(**kwargs):
    
    return Pipeline(
        [
            node(
                add_features,
                inputs="stateless_data",
                outputs="featurized_data",
                name="feature_engineering_node",
            ),
        ]
    )