import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest

from .base_strategy import OutlierStrategy


class IsolationForestStrategy(OutlierStrategy):

    def __init__(
        self,
        contamination="auto",
        random_state=42
    ):

        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,
            n_jobs=-1
        )

    def detect(
    self,
    df: pd.DataFrame,
    target_column: str = None
    ) -> pd.Series:

        numeric_df = df.select_dtypes(include=[np.number])

        if target_column and target_column in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[target_column])

        X = numeric_df.fillna(numeric_df.median())

        preds = self.model.fit_predict(X)

        return pd.Series(preds == 1, index=df.index)