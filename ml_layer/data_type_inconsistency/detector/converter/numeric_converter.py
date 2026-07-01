import pandas as pd
import numpy as np
import re

class NumericConverter:
    def test_conversion(self, series: pd.Series):

        def clean_numeric(x):
            if pd.isna(x):
                return np.nan
            cleaned = re.sub(r'[,$€£¥\s]', '', str(x).strip())
            try:
                return float(cleaned)
            except (ValueError, TypeError):
                return np.nan

        converted = series.apply(clean_numeric)
        failure_mask = converted.isna() & series.notna()

        failed_count = int(failure_mask.sum())

        failed_examples = (
            series[failure_mask]
            .drop_duplicates()
            .head(3)
            .tolist()
        )

        return {
            "failed_count": failed_count,
            "failed_values": failed_examples,
        }