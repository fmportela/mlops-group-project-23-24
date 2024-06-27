from kedro.pipeline import Pipeline, pipeline, node
from .nodes import data_drift_report


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=data_drift_report,
            inputs=["reference_data", "concatenated_processed_data", "params:data_drift_output_path"],
            outputs=None,
            name="data_drift_report"
        )
    ])
