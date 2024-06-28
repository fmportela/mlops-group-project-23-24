from kedro.pipeline import Pipeline, pipeline, node
from .nodes import select_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                select_model,
                inputs=dict(
                    X_train="X_train_selected",
                    y_train="y_train",
                    X_val="X_val_selected",
                    y_val="y_val",
                    X_test="X_test_selected",
                    y_test="y_test",
                    n_trials="params:n_trials",
                ),
                outputs="best_model_of_the_run",
                name= "model_selection_node"
            )
        ]
    )