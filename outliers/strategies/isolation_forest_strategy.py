import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest

from .base_strategy import OutlierStrategy


class IsolationForestStrategy(OutlierStrategy):

    def __init__(
        self,
        contamination=0.01,
        random_state=42
    ):

        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )

    def detect(self, df: pd.DataFrame) -> pd.Series:

        numeric_df = df.select_dtypes(include=[np.number])

        X = numeric_df.fillna(numeric_df.median())

        preds = self.model.fit_predict(X)

        return pd.Series(preds == 1, index=df.index)