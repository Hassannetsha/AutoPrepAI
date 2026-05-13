import pandas as pd
import numpy as np

from .base_strategy import OutlierStrategy


class IQRStrategy(OutlierStrategy):

    def __init__(self, factor=1.5):
        self.factor = factor

    def detect(self, df: pd.DataFrame) -> pd.Series:

        numeric_df = df.select_dtypes(include=[np.number])

        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - self.factor * IQR
        upper = Q3 + self.factor * IQR

        mask = ~(
            ((numeric_df < lower) | (numeric_df > upper))
            .any(axis=1)
        )

        return pd.Series(mask, index=df.index)