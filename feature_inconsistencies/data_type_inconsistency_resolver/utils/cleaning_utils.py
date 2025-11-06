import re
import numpy as np
import pandas as pd
from typing import Any
from ..config import DEFAULT_NUMERIC_CURRENCY_REGEX

def clean_numeric(x: Any) -> float:
    """Clean numeric values by removing currency/space characters and converting to float.


    Returns np.nan for non-convertible values.
    """
    if pd.isna(x):
        return np.nan
    x_str = str(x).strip()
    cleaned = re.sub(DEFAULT_NUMERIC_CURRENCY_REGEX, "", x_str)
    try:
        return float(cleaned)
    except Exception:
        return np.nan




def get_inconsistent_mask(df: pd.DataFrame, column_name: str, recommended_type: str) -> pd.Series:
    """Return a boolean mask of rows that are inconsistent with the recommended type."""
    if recommended_type == 'numeric':
        numeric_col = pd.to_numeric(df[column_name].apply(clean_numeric), errors='coerce')
        return numeric_col.isna() & df[column_name].notna()
    elif recommended_type == 'datetime':
        datetime_col = pd.to_datetime(df[column_name], errors='coerce')
        return datetime_col.isna() & df[column_name].notna()
    else:
    # For strings/objects, treat explicit NaN as inconsistent (you can adjust)
        return df[column_name].isna()