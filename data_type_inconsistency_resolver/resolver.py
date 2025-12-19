# data_resolver/resolver.py
import pandas as pd
from typing import Dict, Any
from .strategy.leave_as_is import LeaveAsIsStrategy
from .strategy.delete_strategies import DeleteRowsStrategy, DeleteColumnStrategy
from .strategy.impute_strategies import (
    ImputeMeanStrategy, ImputeMedianStrategy, ImputeModeStrategy,
    ImputeCustomValueStrategy, ForwardFillStrategy, BackwardFillStrategy
)
from .strategy.convert_to_type import ConvertToTypeStrategy

class DataResolver:
    def __init__(self, df: pd.DataFrame, detection_results: Dict[str, Dict[str, Any]]):
        self.df_original = df.copy()
        self.df_resolved = df.copy()
        self.detection_results = detection_results
        self.resolution_log = []

        self.strategies = {
            "leave_as_is": LeaveAsIsStrategy(),
            "delete_rows": DeleteRowsStrategy(),
            "delete_column": DeleteColumnStrategy(),
            "impute_mean": ImputeMeanStrategy(),
            "impute_median": ImputeMedianStrategy(),
            "impute_mode": ImputeModeStrategy(),
            "impute_custom": ImputeCustomValueStrategy(),
            "forward_fill": ForwardFillStrategy(),
            "backward_fill": BackwardFillStrategy(),
            "convert_to_type": ConvertToTypeStrategy(),
        }

    def resolve(self, strategy_name: str, column_name: str, **kwargs):
        if strategy_name not in self.strategies:
            return self.df_resolved, f"Unknown strategy: {strategy_name}"

        strategy = self.strategies[strategy_name]
        result = self.detection_results.get(column_name, {})
        kwargs["result"] = result

        df_before = len(self.df_resolved)
        self.df_resolved, message = strategy.resolve(self.df_resolved.copy(), column_name, **kwargs)
        df_after = len(self.df_resolved)

        self.resolution_log.append({
            "column": column_name,
            "strategy": strategy_name,
            "rows_before": df_before,
            "rows_after": df_after,
            "rows_deleted": df_before - df_after,
            "message": message,
        })

        return self.df_resolved, message
    
    def available_strategies(self):
        """Return all available resolution strategies."""
        return self.strategies
