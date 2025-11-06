import pandas as pd
from .base_strategy import BaseDetectionStrategy

class StringDetectionStrategy(BaseDetectionStrategy):
    def __init__(self):
        super().__init__('string')

    def detect(self, value) -> bool:
        if pd.isna(value):
            return False
        val_str = str(value).strip()
        return val_str != ''
