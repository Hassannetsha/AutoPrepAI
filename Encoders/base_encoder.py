from abc import ABC, abstractmethod
import pandas as pd

class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame, columns: list, **kwargs) -> pd.DataFrame:
        pass
