import pandas as pd

class DatetimeConverter:
    def test_conversion(self, series: pd.Series):
        converted = pd.to_datetime(series, errors="coerce")

        failure_mask = converted.isna() & series.notna()
        failed_count = int(failure_mask.sum())

        failed_examples = (
            series[failure_mask]
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        return {
            "failed_count": failed_count,
            "failed_values": failed_examples,
        }