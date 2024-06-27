from kedro.pipeline import Pipeline, pipeline, node
from .nodes import add_features


def create_pipeline(**kwargs):
    
    return Pipeline(
        [
            node(
                add_features,
                inputs="X_train_stateful",
                outputs="X_train_featurized",
                name="X_train_feature_engineering_node",
            ),
            node(
                add_features,
                inputs="X_val_stateful",
                outputs="X_val_featurized",
                name="X_val_feature_engineering_node",
            ),
            node(
                add_features,
                inputs="X_test_stateful",
                outputs="X_test_featurized",
                name="X_test_feature_engineering_node",
            ),
        ]
    )