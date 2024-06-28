from kedro.pipeline import Pipeline, pipeline, node
from .nodes import select_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=select_features,
            inputs=["stateful_data", "selected_features"],
            outputs="pruned_data"
        )
    ])
