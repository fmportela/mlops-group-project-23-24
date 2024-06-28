import logging

import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


log = logging.getLogger(__name__)

def data_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    output_path: str
) -> None:
    """
    Generate a data drift report.
    
    Args:
        reference_data: Reference data.
        current_data: Current data.
        output_path: Path to save the report.
    """
    
    data_drift_report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )
    
    log.info("Generated Data Drift Report")

    data_drift_report.run(reference_data=reference_data, current_data=current_data)
    data_drift_report.save_html(output_path)
