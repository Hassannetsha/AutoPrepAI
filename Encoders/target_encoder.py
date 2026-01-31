from category_encoders import TargetEncoder
from .base_encoder import BaseEncoder

class TargetEncoderStrategy(BaseEncoder):
    def encode(self, df, columns, **kwargs):
        target = kwargs.get("target")
        if not target:
            raise ValueError("Target column required for target encoding")
        te = TargetEncoder(cols=columns)
        df[columns] = te.fit_transform(df[columns], df[target])
        return df
