# data_resolver/strategy/base_strategy.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseResolutionStrategy(ABC):
    """Base class for all resolution strategies."""

    @abstractmethod
    def resolve(self, df: pd.DataFrame, column_name: str, **kwargs) -> tuple[pd.DataFrame, str]:
        """Apply the strategy to a specific column and return updated df + message."""
        pass
