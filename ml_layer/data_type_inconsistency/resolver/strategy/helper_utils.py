# data_resolver/strategy/helper_utils.py
import pandas as pd
import numpy as np
import re

def clean_numeric(x):
    """Clean numeric values by removing currency and whitespace."""
    if pd.isna(x):
        return np.nan
    x_str = str(x).strip()
    cleaned = re.sub(r'[,$€£¥\s]', '', x_str)
    try:
        return float(cleaned)
    except:
        return np.nan

def get_inconsistent_mask(df, column_name, recommended_type):
    """Get boolean mask for inconsistent values."""
    if recommended_type == 'numeric':
        numeric_col = pd.to_numeric(df[column_name].apply(clean_numeric), errors='coerce')
        return numeric_col.isna() & df[column_name].notna()

    elif recommended_type == 'datetime':
        datetime_col = pd.to_datetime(df[column_name], errors='coerce')
        return datetime_col.isna() & df[column_name].notna()

    else:
        return df[column_name].isna()

def clean_boolean(value):
    """Convert common boolean representations while preserving missing values."""

    if pd.isna(value):
        return pd.NA

    if isinstance(value, bool):
        return value

    value = str(value).strip().lower()

    if value in {"true", "1", "yes", "y", "t"}:
        return True

    if value in {"false", "0", "no", "n", "f"}:
        return False

    return pd.NA