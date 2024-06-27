import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def data_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: str
) -> None:
    
    data_drift_report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    data_drift_report.run(reference_data=reference_data, current_data=current_data)
    data_drift_report.save_html(output_path)
