import pandas as pd
import numpy as np

from scipy.stats import zscore

from .base_strategy import OutlierStrategy


class ZScoreStrategy(OutlierStrategy):

    def __init__(self, threshold=3.0):
        self.threshold = threshold

    def detect(self, df: pd.DataFrame) -> pd.Series:

        numeric_df = df.select_dtypes(include=[np.number])

        z_scores = np.abs(
            zscore(numeric_df, nan_policy='omit')
        )

        mask = (z_scores < self.threshold).all(axis=1)

        return pd.Series(mask, index=df.index)