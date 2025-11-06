from dateutil import parser
import pandas as pd
import numpy as np
from .base_strategy import BaseDetectionStrategy

class DatetimeDetectionStrategy(BaseDetectionStrategy):
    def __init__(self):
        super().__init__('datetime')

    def detect(self, value) -> bool:
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            return True
        if pd.isna(value) or len(str(value)) < 6:
            return False
        try:
            parser.parse(str(value), fuzzy=False)
            return True
        except Exception:
            return False
