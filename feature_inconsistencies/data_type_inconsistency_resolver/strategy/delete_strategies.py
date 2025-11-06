# data_resolver/strategy/delete_strategies.py
import pandas as pd
from .base_strategy import BaseResolutionStrategy
from .helper_utils import clean_numeric

class DeleteColumnStrategy(BaseResolutionStrategy):
    def resolve(self, df, column_name, **kwargs):
        df = df.drop(columns=[column_name])
        return df, f"✓ Column '{column_name}' deleted"

class DeleteRowsStrategy(BaseResolutionStrategy):
    def resolve(self, df, column_name, **kwargs):
        result = kwargs.get("result", {})
        recommended_type = result.get("recommended_type")

        initial_count = len(df)

        if recommended_type == "numeric":
            def is_numeric(x):
                if pd.isna(x):
                    return True
                try:
                    float(clean_numeric(x))
                    return True
                except:
                    return False
            mask = df[column_name].apply(is_numeric)
            df = df[mask]

        elif recommended_type == "datetime":
            converted = pd.to_datetime(df[column_name], errors="coerce")
            mask = converted.notna() | df[column_name].isna()
            df = df[mask]

        deleted = initial_count - len(df)
        return df, f"✓ Deleted {deleted} rows with inconsistent values in '{column_name}'"
