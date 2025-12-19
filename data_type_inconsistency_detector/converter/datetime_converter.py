import pandas as pd

class DatetimeConverter:
    def test_conversion(self, series: pd.Series):
        issues = []
        converted = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
        failed_count = converted.isna().sum() - series.isna().sum()

        if failed_count > 0:
            issues.append(f"{failed_count} values cannot be converted to datetime")
            failures = series[converted.isna() & series.notna()].head(5).tolist()
            issues.append(f"Example failures: {failures}")

        return issues
