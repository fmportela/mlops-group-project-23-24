from kedro.pipeline import Pipeline, pipeline, node
from .nodes import test_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=test_data,
                inputs=dict(
                    df='stateful_data_pre_validation',
                    datasource_name='params:processed_datasource_name',
                    suite_name='params:processed_suite_name',
                    data_asset_name='params:processed_data_asset_name',
                    build_data_docs='params:build_data_docs'
                ),
                outputs="stateful_data",
                name="data_unit_tests_node",
            ),
        ]
    )
