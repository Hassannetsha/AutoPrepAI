import pandas as pd
import numpy as np

from outliers.strategies.base_strategy import (
    OutlierStrategy
)


class IQRStrategy(OutlierStrategy):

    def __init__(self, factor=1.5):
        self.factor = factor

    def detect(
    self,
    df: pd.DataFrame,
    target_column: str = None
    ) -> pd.Series:

        numeric_df = df.select_dtypes(include=[np.number])
        if target_column and target_column in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[target_column])
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - self.factor * IQR
        upper = Q3 + self.factor * IQR

        violations = ((numeric_df < lower) |(numeric_df > upper))
        mask = violations.mean(axis=1) < 0.2
        return pd.Series(mask, index=df.index)