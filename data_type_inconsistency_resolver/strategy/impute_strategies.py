# data_resolver/strategy/impute_strategies.py
import pandas as pd
from .base_strategy import BaseResolutionStrategy
from .helper_utils import clean_numeric, get_inconsistent_mask

class ImputeCustomValueStrategy(BaseResolutionStrategy):
    def resolve(self, df, column_name, **kwargs):
        value = kwargs.get("value")
        result = kwargs.get("result")
        recommended_type = result["recommended_type"]
        mask = get_inconsistent_mask(df, column_name, recommended_type)
        count = mask.sum()
        df.loc[mask, column_name] = value
        return df, f"✓ Imputed {count} inconsistent values with '{value}' in '{column_name}'"

class ImputeMeanStrategy(BaseResolutionStrategy):
    def resolve(self, df, column_name, **kwargs):
        numeric_col = pd.to_numeric(df[column_name].apply(clean_numeric), errors='coerce')
        mean_val = numeric_col.mean()
        mask = numeric_col.isna() & df[column_name].notna()
        count = mask.sum()
        df.loc[mask, column_name] = mean_val
        df[column_name] = pd.to_numeric(df[column_name].apply(clean_numeric), errors='coerce')
        return df, f"✓ Imputed {count} values with mean ({mean_val:.2f}) in '{column_name}'"

class ImputeMedianStrategy(BaseResolutionStrategy):
    def resolve(self, df, column_name, **kwargs):
        numeric_col = pd.to_numeric(df[column_name].apply(clean_numeric), errors='coerce')
        median_val = numeric_col.median()
        mask = numeric_col.isna() & df[column_name].notna()
        count = mask.sum()
        df.loc[mask, column_name] = median_val
        df[column_name] = pd.to_numeric(df[column_name].apply(clean_numeric), errors='coerce')
        return df, f"✓ Imputed {count} values with median ({median_val:.2f}) in '{column_name}'"

class ImputeModeStrategy(BaseResolutionStrategy):
    def resolve(self, df, column_name, **kwargs):
        result = kwargs.get("result")
        recommended_type = result["recommended_type"]
        mode_val = df[column_name].mode()
        if len(mode_val) == 0:
            return df, f"❌ Cannot compute mode for '{column_name}'"
        mode_val = mode_val[0]
        # mask = get_inconsistent_mask(df, column_name, recommended_type)
        mask = result["inconsistent_indices"]
        count = len(mask)
        df.loc[mask, column_name] = mode_val
        return df, f"✓ Imputed {count} values with mode ('{mode_val}') in '{column_name}'"

class ForwardFillStrategy(BaseResolutionStrategy):
    def resolve(self, df, column_name, **kwargs):
        before = df[column_name].isna().sum()
        df[column_name] = df[column_name].fillna(method='ffill')
        filled = before - df[column_name].isna().sum()
        return df, f"✓ Forward-filled {filled} values in '{column_name}'"

class BackwardFillStrategy(BaseResolutionStrategy):
    def resolve(self, df, column_name, **kwargs):
        before = df[column_name].isna().sum()
        df[column_name] = df[column_name].fillna(method='bfill')
        filled = before - df[column_name].isna().sum()
        return df, f"✓ Backward-filled {filled} values in '{column_name}'"
