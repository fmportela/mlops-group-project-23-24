from kedro.pipeline import Pipeline, pipeline, node
from .nodes import concatenate_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=concatenate_data,
            inputs=[
                "X_train_stateful", "X_val_stateful", "X_test_stateful",
                "y_train", "y_val", "y_test"
            ],
            outputs="concatenated_processed_data",
            name="concatenate_data"
        )
    ])
