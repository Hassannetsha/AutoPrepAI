import pandas as pd
import numpy as np

from scipy.stats import zscore

from .base_strategy import OutlierStrategy


class ZScoreStrategy(OutlierStrategy):

    def __init__(self, threshold=3.0):
        self.threshold = threshold

    def detect(
    self,
    df: pd.DataFrame,
    target_column: str = None
    ) -> pd.Series:

        numeric_df = df.select_dtypes(include=[np.number])

        if target_column and target_column in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[target_column])
        z_scores = np.abs(
            zscore(numeric_df, nan_policy='omit')
        )

        mask = (z_scores < self.threshold).mean(axis=1) > 0.95

        return pd.Series(mask, index=df.index)