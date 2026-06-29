import re
import numpy as np
import pandas as pd
from .base_strategy import BaseDetectionStrategy

class NumericDetectionStrategy(BaseDetectionStrategy):
    def __init__(self):
        super().__init__('numeric')

    def detect(self, value) -> bool:
        if pd.isna(value):
            return False
        val_str = str(value).strip()
        cleaned = re.sub(r'[,$€£¥\s]', '', val_str)
        try:
            float(cleaned)
            return True
        except (ValueError, TypeError):
            return False
