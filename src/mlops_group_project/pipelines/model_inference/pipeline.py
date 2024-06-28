from kedro.pipeline import Pipeline, pipeline, node
from .nodes import make_predictions


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=make_predictions,
            inputs=["pruned_data", "ids"],
            outputs="predictions"
        )
    ])
