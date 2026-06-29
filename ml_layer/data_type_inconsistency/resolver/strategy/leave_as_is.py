# data_resolver/strategy/leave_as_is.py
from .base_strategy import BaseResolutionStrategy

class LeaveAsIsStrategy(BaseResolutionStrategy):
    def resolve(self, df, column_name, **kwargs):
        return df, f"✓ Column '{column_name}' left as is"
