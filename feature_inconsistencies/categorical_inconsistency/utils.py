# utils.py
import re
import pandas as pd
from .constants import INVALID_CATEGORY_VALUES, PATTERN_NON_CATEGORICAL

def normalize_text(series: pd.Series) -> pd.Series:
    """Lowercase, trim, remove punctuation and unify invalids."""
    series = series.astype(str).str.lower().str.strip()
    series = series.str.replace(r"[^\w\s]", "", regex=True)
    series = series.replace(INVALID_CATEGORY_VALUES, pd.NA)
    return series

def looks_non_categorical(series: pd.Series) -> bool:
    """Detect if strings follow non-categorical patterns like user_1, id_23, etc."""
    sample = series.dropna().head(100)
    pattern_count = sum(bool(re.match(PATTERN_NON_CATEGORICAL, v)) for v in sample)
    return pattern_count / len(sample) > 0.3 if len(sample) > 0 else False
