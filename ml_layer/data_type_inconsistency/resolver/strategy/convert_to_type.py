# data_resolver/strategy/convert_to_type.py
import pandas as pd
from .base_strategy import BaseResolutionStrategy
from .helper_utils import clean_numeric

class ConvertToTypeStrategy(BaseResolutionStrategy):
    def resolve(self, df, column_name, **kwargs):
        target_type = kwargs.get("target_type", "string")
        before_null = df[column_name].isna().sum()

        if target_type == "numeric":
            df[column_name] = pd.to_numeric(df[column_name].apply(clean_numeric), errors="coerce")
        elif target_type == "datetime":
            df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
        elif target_type == "boolean":
            df[column_name] = df[column_name].astype(bool)
        else:
            df[column_name] = df[column_name].astype(str)

        after_null = df[column_name].isna().sum()
        failed = after_null - before_null
        return df, f"✓ Converted '{column_name}' to {target_type} ({failed} values became NaN)"
