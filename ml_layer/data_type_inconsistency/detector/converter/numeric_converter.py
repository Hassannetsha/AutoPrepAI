import pandas as pd
import numpy as np
import re

class NumericConverter:
    def test_conversion(self, series: pd.Series):
        issues = []

        def clean_numeric(x):
            if pd.isna(x):
                return np.nan
            cleaned = re.sub(r'[,$€£¥\s]', '', str(x).strip())
            try:
                return float(cleaned)
            except (ValueError, TypeError):
                return np.nan

        converted = series.apply(clean_numeric)
        failed_count = converted.isna().sum() - series.isna().sum()

        if failed_count > 0:
            issues.append(f"{failed_count} values cannot be converted to numeric")
            failures = series[converted.isna() & series.notna()].head(5).tolist()
            issues.append(f"Example failures: {failures}")

        return issues
