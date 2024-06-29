from kedro.pipeline import Pipeline, pipeline, node
from .nodes import calculate_permutation_importance


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=calculate_permutation_importance,
            inputs=["best_model_of_the_run", "X_train_selected", "y_train"],
            outputs="permutation_importance"
        )
    ])
