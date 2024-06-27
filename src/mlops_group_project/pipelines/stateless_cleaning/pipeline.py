from kedro.pipeline import Pipeline, pipeline, node
from .nodes import clean_df


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            clean_df,
            inputs="valid_data",
            outputs="stateless_data",
            name="cleaning_node"
        )
    ])
