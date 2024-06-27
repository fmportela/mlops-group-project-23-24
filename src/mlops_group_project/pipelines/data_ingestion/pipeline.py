from kedro.pipeline import Pipeline, pipeline, node
from .nodes import load_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_data,
            inputs="params:group_name",
            outputs="valid_data"
        )
    ])
