from sklearn.preprocessing import LabelEncoder
from .base_encoder import BaseEncoder

class LabelEncoderStrategy(BaseEncoder):
    def encode(self, df, columns, **kwargs):
        df = df.reset_index(drop=True)
        le = LabelEncoder()
        for col in columns:
            df[col] = le.fit_transform(df[col].astype(str))
        return df
