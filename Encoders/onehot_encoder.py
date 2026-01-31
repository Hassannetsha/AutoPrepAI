import pandas as pd
from .base_encoder import BaseEncoder

class OneHotEncoderStrategy(BaseEncoder):
    def encode(self, df: pd.DataFrame, columns: list, **kwargs) -> pd.DataFrame:
        return pd.get_dummies(df, columns=columns, drop_first=True)
