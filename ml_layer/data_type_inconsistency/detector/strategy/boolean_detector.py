import pandas as pd
import numpy as np
from .base_strategy import BaseDetectionStrategy

class BooleanDetectionStrategy(BaseDetectionStrategy):
    def __init__(self):
        super().__init__('boolean')

    def detect(self, value) -> bool:
        if pd.isna(value):
            return False
        val_str = str(value).strip().lower()
        return val_str in ['true', 'false', 'yes', 'no', '1', '0', 't', 'f']
