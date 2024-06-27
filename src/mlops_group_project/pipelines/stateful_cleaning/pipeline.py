from kedro.pipeline import Pipeline, pipeline, node
from .nodes import impute_missing_values


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            impute_missing_values,
            inputs=["X_train", "X_val", "X_test"],
            outputs=["X_train_stateful", "X_val_stateful", "X_test_stateful", "imputer"],
            name="impute_missing_values_node"
        )
    ])
